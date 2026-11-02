### Core modules ###
import os
from time import perf_counter
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue
)


### Type hints ###
from typing import Any
from langchain_core.documents import Document


### Internal modules ###
from .base import ServiceBase
from .vector_database import VectorDatabase
from ...utils.log_tool import set_color


# Stored memory_type -> (tier, human-readable label) used when citing sources.
SOURCE_TIERS: dict[str, tuple[str, str]] = {
    "session":          ("session", "Attached file / session"),
    "user":             ("user", "Your saved memory"),
    "global_memory":    ("global", "Global knowledge base")
}


class RAGBase(ServiceBase):
    def __init__(
        self,
        vector_database: VectorDatabase,
        **kwargs: Any
    ) -> None:
        """
        Context retriever service.

        Retrieval tuning is read from the environment (see `.env.example`):
            RAG_TOPK            (int):  contexts kept when reranking is disabled.
            RAG_RERANK_ENABLED  (bool): two-stage retrieval (dense pool -> cross-encoder).
            RAG_CANDIDATE_POOL  (int):  dense candidates fetched before reranking.
            RAG_RERANK_TOPK     (int):  contexts kept after reranking.
            RAG_ATTACHMENT_CANDIDATE_POOL (int, optional): candidates per pass when
                a question with an attached file searches both the file and the
                knowledge base. Defaults to RAG_CANDIDATE_POOL.

        Args:
            vector_database (VectorDatabase): vector database.

        Raises:
            KeyError: if a required environment key is missing.
        """
        super().__init__(**kwargs)

        # Set config.
        self.topk               = int(os.environ["RAG_TOPK"])
        self.vector_database    = vector_database

        # Reranking config.
        self.rerank_enabled         = os.environ["RAG_RERANK_ENABLED"].strip().lower() == "true"
        self.candidate_pool         = int(os.environ["RAG_CANDIDATE_POOL"])
        self.rerank_topk            = int(os.environ["RAG_RERANK_TOPK"])

        # Two reranked passes cost twice the cross-encoder work, so they may use a smaller pool.
        self.attachment_candidate_pool  = int(os.getenv("RAG_ATTACHMENT_CANDIDATE_POOL", str(self.candidate_pool)))


    def set_vector_database(
        self,
        vector_database: VectorDatabase
    ) -> None:
        """
        Set the vector database externally on demand.

        Args:
            vector_database (VectorDatabase): an external vector database.
        """
        self.vector_database = vector_database


    @staticmethod
    def eq(
        field: str,
        value: str
    ) -> FieldCondition:
        return FieldCondition(
            key=f"metadata.{field}",
            match=MatchValue(value=value)
        )


    @staticmethod
    def _title(
        doc: Any
    ) -> str:
        meta: dict[str, Any] = getattr(doc, "metadata", {})

        return meta.get("title", "")


    @staticmethod
    def build_memory_filter(
        memory_type: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        service_name: str | None = None
    ) -> Filter | None:
        """
        Build a Qdrant payload filter scoping retrieval to a single memory tier.

        Args:
            memory_type (str, optional): memory tier (session, user, global_memory).
            user_id (str, optional): owner of the memory.
            session_id (str, optional): chat session of the memory.
            service_name (str, optional): service a global memory belongs to.

        Returns:
            memory_filter (Filter | None): filter, or None if no field is set.
        """
        must: list[FieldCondition | None] = []

        if memory_type:
            must.append(RAGBase.eq(field="memory_type", value=memory_type))

        if user_id:
            must.append(RAGBase.eq(field="user_id", value=user_id))

        if session_id:
            must.append(RAGBase.eq(field="session_id", value=session_id))

        if service_name:
            must.append(RAGBase.eq(field="service_name", value=service_name))

        if not must:
            return None

        memory_filter: Filter | None = Filter(must=must) # pyright: ignore[reportArgumentType]

        return memory_filter


    @staticmethod
    def build_union_filter(
        user_id: str | None = None,
        session_id: str | None = None,
        global_service_names: list[str] | None = None,
        include_session: bool = True,
        include_user: bool = False,
        include_global: bool = True,
        document_ids: list[str] | None = None
    ) -> Filter | None:
        """
        Build a Qdrant filter spanning the active memory tiers as a union.

        Args:
            user_id (str, optional): owner of the session and user memory.
            session_id (str, optional): chat session of the session memory.
            global_service_names (list[str], optional): services whose global
                memory is searched.
            include_session (bool, optional): search session memory. Defaults to True.
            include_user (bool, optional): search user memory. Defaults to False.
            include_global (bool, optional): search global memory. Defaults to True.
            document_ids (list[str], optional): narrow the session tier to these
                attached documents.

        Returns:
            memory_filter (Filter | None): filter, or None if no tier is active.
        """
        should: list[Filter | None] = []

        if include_session and user_id and session_id:
            session_must: list[FieldCondition | None] = [
                RAGBase.eq(field="memory_type", value="session"),
                RAGBase.eq(field="user_id", value=user_id),
                RAGBase.eq(field="session_id", value=session_id)
            ]
            if document_ids:
                session_must.append(
                    FieldCondition(
                        key="metadata.document_id",
                        match=MatchAny(any=list(document_ids))
                    )
                )
            should.append(Filter(must=session_must)) # pyright: ignore[reportArgumentType]

        if include_user and user_id:
            should.append(Filter(must=[
                RAGBase.eq(field="memory_type", value="user"),
                RAGBase.eq(field="user_id", value=user_id)
            ]))

        if include_global and global_service_names:
            should.append(Filter(must=[
                RAGBase.eq(field="memory_type", value="global_memory"),
                FieldCondition(
                    key="metadata.service_name",
                    match=MatchAny(any=list(global_service_names))
                )
            ]))

        if not should:
            return None

        memory_filter: Filter | None = Filter(should=should) # pyright: ignore[reportArgumentType]

        return memory_filter


    @staticmethod
    def _scope_rank(
        doc: Any
    ) -> int:
        """
        Priority rank of a chunk's memory tier.

        Args:
            doc (Document): retrieved chunk.

        Returns:
            rank (int): session (0) > user (1) > global (2) > unknown (3).
        """
        meta: dict[str, Any] = getattr(doc, "metadata", {})

        if "memory_type" in meta:
            match meta["memory_type"]:
                case "session":
                    return 0

                case "user":
                    return 1

                case "global_memory":
                    return 2

                case _:
                    return 3

        return 3


    @staticmethod
    def _source_info(
        doc: Document,
        score: float
    ) -> dict[str, Any]:
        """
        Describe where a chunk came from, for citing it in the answer.

        Args:
            doc     (Document): retrieved chunk.
            score   (float):    relevance score of the chunk.

        Returns:
            source (dict[str, Any]): tier ("session" | "user" | "global" | "unknown"),
                label, title, service_name and score of the chunk.
        """
        meta: dict[str, Any] = doc.metadata or {}
        tier, label = SOURCE_TIERS.get(
            meta.get("memory_type", ""),
            ("unknown", "Unknown source")
        )

        return {
            "tier":         tier,
            "label":        label,
            "title":        str(meta.get("title") or meta.get("file_name") or ""),
            "service_name": str(meta.get("service_name") or ""),
            "score":        float(score)
        }


    @staticmethod
    def _doc_key(
        doc: Document
    ) -> tuple[str | None, str]:
        """
        Identity of a chunk, used to de-duplicate two retrieval passes.

        Args:
            doc (Document): retrieved chunk.

        Returns:
            key (tuple[str | None, str]): (document_id, page_content).
        """
        return (doc.metadata or {}).get("document_id"), doc.page_content


    @staticmethod
    def _select(
        pairs: list[tuple[Document, float]],
        keep_n: int
    ) -> list[tuple[Document, float]]:
        """
        Rank chunks by relevance score and keep the best ones.

        Ties on score are broken by memory tier (session > user > global), so a
        weakly related session chunk no longer beats a highly relevant global one.

        Args:
            pairs   (list[tuple[Document, float]]): (chunk, score) candidates.
            keep_n  (int):                          number of chunks to keep.

        Returns:
            selected (list[tuple[Document, float]]): top keep_n (chunk, score) pairs.
        """
        ordered: list[tuple[Document, float]] = sorted(
            pairs,
            key=lambda pair: (-pair[1], RAGBase._scope_rank(doc=pair[0]))
        )

        return ordered[:keep_n]


    @staticmethod
    def _log_timing(
        search_ms: float,
        rerank_ms: float,
        num_candidates: int
    ) -> None:
        """
        Log the time of one retrieval pass, to see where retrieval latency goes.

        Args:
            search_ms (float): query embedding and vector search time.
            rerank_ms (float): cross-encoder time (0 when not reranked).
            num_candidates (int): chunks fetched by the search.
        """
        print(
            set_color(
                status="info",
                information=f"[rag] Retrieval pass: search {search_ms:.0f} ms, rerank {rerank_ms:.0f} ms, {num_candidates} candidates"
            )
        )

        return None


    def _retrieve_pairs(
        self,
        user_prompt: str,
        user_id: str | None = None,
        session_id: str | None = None,
        global_service_names: list[str] | None = None,
        include_session: bool = True,
        include_user: bool = False,
        include_global: bool = True,
        document_ids: list[str] | None = None,
        candidate_pool: int | None = None
    ) -> list[tuple[Document, float]]:
        """
        Search the active memory tiers and select the most relevant chunks.

        Args:
            Same as `__call__()`, plus:
            candidate_pool (int | None, optional): candidates fetched before
                reranking. Defaults to None (RAG_CANDIDATE_POOL).

        Returns:
            selected (list[tuple[Document, float]]): selected (chunk, score) pairs,
                most relevant first.
        """
        selected: list[tuple[Document, float]] = []

        # Scope retrieval to the union of active memory tiers.
        memory_filter: Filter | None = self.build_union_filter(
            user_id=user_id,
            session_id=session_id,
            global_service_names=global_service_names,
            include_session=include_session,
            include_user=include_user,
            include_global=include_global,
            document_ids=document_ids
        )

        # Never search the collection unfiltered: that would leak other
        # users', sessions' and services' points.
        if memory_filter is None:
            return selected

        # Fetch a larger candidate pool when reranking.
        fetch_k: int = (candidate_pool or self.candidate_pool) if self.rerank_enabled else self.topk

        search_start: float = perf_counter()
        retrieved_contents: list[tuple[Document, float]] = (
            self.vector_database.similarity_search_with_relevance_scores(
                query=user_prompt,
                k=fetch_k,
                filter=memory_filter
            )
        )
        search_ms: float = (perf_counter() - search_start) * 1000

        if self.rerank_enabled and retrieved_contents:
            candidate_docs: list[Document] = [doc for doc, _ in retrieved_contents]

            rerank_start: float = perf_counter()
            rerank_scores: list[float] = self.vector_database.rerank(
                user_prompt,
                [doc.page_content for doc in candidate_docs]
            )

            RAGBase._log_timing(
                search_ms=search_ms,
                rerank_ms=(perf_counter() - rerank_start) * 1000,
                num_candidates=len(candidate_docs)
            )

            if rerank_scores:
                selected = RAGBase._select(
                    pairs=list(zip(candidate_docs, rerank_scores)),
                    keep_n=self.rerank_topk
                )

                return selected

        else:
            RAGBase._log_timing(
                search_ms=search_ms,
                rerank_ms=0.0,
                num_candidates=len(retrieved_contents)
            )

        # Reranking disabled or unavailable: use the dense similarity order.
        selected = RAGBase._select(
            pairs=list(retrieved_contents),
            keep_n=self.topk
        )

        return selected


    @staticmethod
    def _format_context(
        selected: list[tuple[Document, float]]
    ) -> tuple[str, list[float], list[dict[str, Any]]]:
        """
        Join selected chunks into one context string, each prefixed by its title.

        Args:
            selected (list[tuple[Document, float]]): selected (chunk, score) pairs.

        Returns:
            context                 (str):                  retrieved context for the LLM.
            retrieved_context_score (list[float]):          score of each chunk.
            sources                 (list[dict[str, Any]]): origin of each chunk (see `_source_info()`).
        """
        context: str = "".join(
            f"{RAGBase._title(doc=doc)}: {doc.page_content.replace("\n", " ")}. "
            for doc, _ in selected
        )
        retrieved_context_score: list[float] = [score for _, score in selected]
        sources: list[dict[str, Any]] = [
            RAGBase._source_info(doc=doc, score=score)
            for doc, score in selected
        ]

        return (
            context,
            retrieved_context_score,
            sources
        )


    def __call__(
        self,
        user_prompt: str,
        user_id: str | None = None,
        session_id: str | None = None,
        global_service_names: list[str] | None = None,
        include_session: bool = True,
        include_user: bool = False,
        include_global: bool = True,
        document_ids: list[str] | None = None
    ) -> tuple[str, list[float], list[dict[str, Any]]]:
        """
        Retrieve context for a given user prompt.

        Args:
            user_prompt (str): a question from the user.
            user_id (str, optional): owner of the session and user memory.
            session_id (str, optional): chat session of the session memory.
            global_service_names (list[str], optional): services whose global
                memory is searched.
            include_session (bool, optional): search session memory. Defaults to True.
            include_user (bool, optional): search user memory. Defaults to False.
            include_global (bool, optional): search global memory. Defaults to True.
            document_ids (list[str], optional): narrow the session tier to these
                attached documents.

        Returns:
            context (str): retrieved context from the vector database.
            retrieved_context_score (list[float]): score of each retrieved chunk,
                most relevant first.
            sources (list[dict[str, Any]]): origin of each chunk (see `_source_info()`).
        """
        selected: list[tuple[Document, float]] = self._retrieve_pairs(
            user_prompt=user_prompt,
            user_id=user_id,
            session_id=session_id,
            global_service_names=global_service_names,
            include_session=include_session,
            include_user=include_user,
            include_global=include_global,
            document_ids=document_ids
        )

        return RAGBase._format_context(selected=selected)


    def retrieve_with_attachment(
        self,
        user_prompt: str,
        document_ids: list[str],
        user_id: str | None = None,
        session_id: str | None = None,
        global_service_names: list[str] | None = None,
        include_user: bool = False,
        file_directed: bool = False,
        kb_service_name: str | None = None
    ) -> tuple[str, list[float], list[dict[str, Any]]]:
        """
        Retrieve context for a question asked with an attached file.

        The file and the rest of memory are retrieved in two separate passes, so
        the file's chunks do not compete with the knowledge base for the top-k.
        What is kept depends on intent:
            - file directed, no KB route: the file only.
            - file directed, KB route: the file first, then the KB chunks.
            - not file directed, KB route: the KB only (the file is stale).
            - not file directed, no KB route: the file only.
        If the file has no indexed chunks, the rest of memory is used.

        Args:
            user_prompt (str): a question from the user.
            document_ids (list[str]): attached documents.
            user_id (str, optional): owner of the session and user memory.
            session_id (str, optional): chat session of the session memory.
            global_service_names (list[str], optional): memory-capable services.
            include_user (bool, optional): search user memory. Defaults to False.
            file_directed (bool, optional): the question points at the file.
                Defaults to False.
            kb_service_name (str, optional): memory-capable service the query
                analyser routed to, if any. Narrows the global tier to it.

        Returns:
            context (str): retrieved context from the vector database.
            retrieved_context_score (list[float]): score of each retrieved chunk.
            sources (list[dict[str, Any]]): origin of each chunk (see `_source_info()`).
        """
        file_pairs: list[tuple[Document, float]] = self._retrieve_pairs(
            user_prompt=user_prompt,
            user_id=user_id,
            session_id=session_id,
            include_session=True,
            include_user=False,
            include_global=False,
            document_ids=document_ids,
            candidate_pool=self.attachment_candidate_pool
        )
        routed_to_kb: bool = kb_service_name is not None

        if file_pairs and not routed_to_kb:
            return RAGBase._format_context(selected=file_pairs)

        memory_pairs: list[tuple[Document, float]] = self._retrieve_pairs(
            user_prompt=user_prompt,
            user_id=user_id,
            session_id=session_id,
            global_service_names=[kb_service_name] if kb_service_name else global_service_names,
            include_session=False,
            include_user=include_user,
            include_global=True,
            candidate_pool=self.attachment_candidate_pool
        )

        if not file_pairs or not file_directed:
            return RAGBase._format_context(selected=memory_pairs)

        file_keys: set[tuple[str | None, str]] = {
            RAGBase._doc_key(doc=doc) for doc, _ in file_pairs
        }
        merged: list[tuple[Document, float]] = file_pairs + [
            (doc, score) for doc, score in memory_pairs
            if RAGBase._doc_key(doc=doc) not in file_keys
        ]

        return RAGBase._format_context(selected=merged)
