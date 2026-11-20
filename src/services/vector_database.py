### Core modules ###
import os
from pathlib import Path
from glob import glob
from threading import Lock
from datetime import datetime
from zoneinfo import ZoneInfo
from qdrant_client.models import (
    Modifier,
    SparseVectorParams
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.document_loaders import PyPDFLoader
from qdrant_client import QdrantClient, models
from langchain_core.documents import Document


### Type hints ###
from typing import Any


### Internal modules ###
from ...utils.log_tool import set_color
from .base import ServiceBase


# Metadata flag on chunks stored with a "<title>: " prefix.
TITLE_PREFIXED_KEY: str = "title_prefixed"


class VectorDatabase(ServiceBase):
    # Name of the BM25 sparse vector in a hybrid collection.
    SPARSE_VECTOR_NAME: str = "sparse"

    # FastEmbed model that encodes text as BM25 sparse vectors.
    BM25_MODEL_NAME: str = "Qdrant/bm25"


    def __init__(
        self,
        device: str = "cuda",
        **kwargs
    ):
        """
        Vector database service.

        Storage is backed by Qdrant (running as its own container, reached via the
        ``QDRANT_URL`` environment variable); there is no local on-disk database.

        Model and deduplication settings are read from the environment (see `.env.example`):
            RAG_EMBEDDING_MODEL     (str):      document analyser/process model.
            RAG_UPDATE_THRESHOLD    (float):    contents with similarity >= this threshold are skipped.
            RAG_RERANKER_MODEL      (str):      cross-encoder model used for reranking.

        Args:
            device (str, optional): use cuda or cpu for LLM. Defaults to "cuda".

        Raises:
            KeyError: if a required environment key is missing.
        """
        super().__init__(**kwargs)

        # Cross-encoder reranker
        self.device = device
        self.reranker_model = os.environ["RAG_RERANKER_MODEL"]
        self._reranker = None

        # Similarity threshold for deduplicating content before adding to Qdrant.
        self.vector_database_update_threshold = float(os.environ["RAG_UPDATE_THRESHOLD"])

        # Keep model weights on a persistent volume instead of the container
        # layer; HF_HOME makes transformers and tokenizers use the same place.
        self._model_cache_dir: str | None = os.getenv("EMBEDDING_MODEL_DIR")

        if self._model_cache_dir:
            os.environ.setdefault("HF_HOME", self._model_cache_dir)
            os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", self._model_cache_dir)

            print(
                set_color(
                    status="info",
                    information=f"[vector_database] Model cache directory: {self._model_cache_dir}"
                )
            )

        # Build the document splitter, the embedding model and the vector store.
        # Chunk size and overlap are counted in tokens of the embedding model.
        self.document_splitter: RecursiveCharacterTextSplitter = VectorDatabase._build_document_splitter(
            model_name=VectorDatabase._resolve_embedding_model_name(),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "256")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "32"))
        )

        self.database_update_embedding: HuggingFaceEmbeddings = VectorDatabase._build_embeddings(
            model_name=VectorDatabase._resolve_embedding_model_name(),
            device=device,
            cache_folder=self._model_cache_dir
        )

        self.database: QdrantVectorStore = VectorDatabase._connect_vector_store(
            embedding=self.database_update_embedding
        )

        # Set a time stamp to highlight the most recently updated information.
        self.time_stamper:              str = datetime.now(tz=ZoneInfo(key="Australia/Sydney")).strftime(format="%B, %Y")


    @staticmethod
    def _resolve_embedding_model_name() -> str:
        """
        Resolve `RAG_EMBEDDING_MODEL` to a Hugging Face model name.

        Known short aliases map to their full Hugging Face repo id; any other
        value is treated as a direct Hugging Face model name.

        Returns:
            model_name (str): Hugging Face name of the embedding model.
        """
        EMBEDDING_MODEL_DICT: dict[str, str] = {"gte-small": "thenlper/gte-small"}
        document_analyser_model: str = os.environ["RAG_EMBEDDING_MODEL"]

        return EMBEDDING_MODEL_DICT.get(document_analyser_model, document_analyser_model)


    @staticmethod
    def _build_document_splitter(
        model_name: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> RecursiveCharacterTextSplitter:
        """
        Build a hybrid token-aware splitter for documents.

        Two limits apply together. The separators split at paragraph, sentence,
        clause and then word boundaries, so chunks keep whole language units. The
        size limit is counted in tokens of the embedding model, so no chunk
        exceeds its context window (gte-small has 512 tokens; 256 leaves room for
        a query prefix in asymmetric search).

        When the tokenizer cannot be loaded (offline, or first boot before the
        download), a character splitter is used instead so the service still
        starts, with limits of about 5 characters per token.

        Args:
            model_name (str): Hugging Face name of the embedding model.
            chunk_size (int): maximum tokens per chunk.
            chunk_overlap (int): tokens shared by neighbouring chunks.

        Returns:
            document_splitter (RecursiveCharacterTextSplitter): splitter for documents.
        """
        # Coarse structural breaks come first, so finer splits are only used
        # when a passage exceeds the token budget.
        SEMANTIC_SEPARATORS: list[str] = [
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            "; ",
            ", ",
            " ",
            ""
        ]

        document_splitter: RecursiveCharacterTextSplitter

        try:
            # Deferred import: transformers is heavy and only needed here.
            from transformers import AutoTokenizer

            document_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer=AutoTokenizer.from_pretrained(model_name),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True,
                strip_whitespace=True,
                separators=SEMANTIC_SEPARATORS
            )

            print(
                set_color(
                    status="info",
                    information=f"[vector_database] Hybrid token-aware chunking: {chunk_size} / {chunk_overlap} tokens ({model_name} tokenizer)"
                )
            )

        except Exception as exc:
            print(
                set_color(
                    status="warning",
                    information=f"[vector_database] Token-aware splitter init failed ({exc}); using character fallback."
                )
            )

            document_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size * 5,
                chunk_overlap=chunk_overlap * 5,
                add_start_index=True,
                strip_whitespace=True,
                separators=SEMANTIC_SEPARATORS
            )

        return document_splitter


    @staticmethod
    def _build_embeddings(
        model_name: str,
        device: str,
        cache_folder: str | None = None
    ) -> HuggingFaceEmbeddings:
        """
        Build the embedding model for documents and queries.

        Args:
            model_name (str): Hugging Face name of the embedding model.
            device (str): use cuda or cpu.
            cache_folder (str | None, optional): where model weights are downloaded.
                Defaults to None (the Hugging Face default cache).

        Returns:
            embeddings (HuggingFaceEmbeddings): embedding model.
        """
        embed_kwargs: dict[str, Any] = {
            "model_name":       model_name,
            "multi_process":    False,  # TODO
            "model_kwargs":     {"device": device},
            "encode_kwargs":    {"normalize_embeddings": True}
        }

        if cache_folder:
            embed_kwargs["cache_folder"] = cache_folder

        return HuggingFaceEmbeddings(**embed_kwargs)


    @staticmethod
    def sparse_vectors_config() -> dict[str, SparseVectorParams]:
        """
        Sparse vector config of a hybrid collection.

        The IDF modifier makes Qdrant weight rare terms above common ones at
        query time, which turns the stored term frequencies into BM25 scores.

        Returns:
            sparse_vectors_config (dict[str, SparseVectorParams]): config by vector name.
        """
        return {
            VectorDatabase.SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)
        }


    @staticmethod
    def _hybrid_store_kwargs(
        client: QdrantClient,
        collection_name: str,
        use_hybrid_search: bool
    ) -> dict[str, Any]:
        """
        Vector store arguments for hybrid (dense + BM25) search.

        Search fuses dense and BM25 results with Reciprocal Rank Fusion inside
        Qdrant, and new documents get their BM25 vector when they are added.
        Qdrant cannot add a sparse vector to an existing collection with an
        unnamed dense vector, so an older collection stays dense-only until it is
        copied with `scripts/migrations/migrate_to_hybrid_collection.py`.

        Args:
            client (QdrantClient): Qdrant client.
            collection_name (str): collection to search.
            use_hybrid_search (bool): RAG_HYBRID_ENABLED is set.

        Returns:
            store_kwargs (dict[str, Any]): extra `QdrantVectorStore` arguments;
                empty for dense-only search.
        """
        store_kwargs: dict[str, Any] = {}

        if not use_hybrid_search:
            return store_kwargs

        sparse_vectors: dict[str, Any] = (
            client.get_collection(collection_name=collection_name).config.params.sparse_vectors or {}
        )

        if VectorDatabase.SPARSE_VECTOR_NAME not in sparse_vectors:
            print(
                set_color(
                    status="warning",
                    information=(
                        f"[vector_database] Collection '{collection_name}' has no BM25 vector; "
                        f"using dense search. Run scripts/migrations/migrate_to_hybrid_collection.py to enable hybrid search."
                    )
                )
            )

            return store_kwargs

        # Deferred import: FastEmbed is only needed for hybrid search.
        from langchain_qdrant import (
            FastEmbedSparse,
            RetrievalMode
        )

        store_kwargs = {
            "retrieval_mode":       RetrievalMode.HYBRID,
            "sparse_embedding":     FastEmbedSparse(model_name=VectorDatabase.BM25_MODEL_NAME),
            "sparse_vector_name":   VectorDatabase.SPARSE_VECTOR_NAME
        }

        print(
            set_color(
                status="success",
                information=f"[vector_database] Hybrid search (dense + BM25) enabled on '{collection_name}'."
            )
        )

        return store_kwargs


    @staticmethod
    def _connect_vector_store(
        embedding: HuggingFaceEmbeddings
    ) -> QdrantVectorStore:
        """
        Connect to the Qdrant collection, creating it on first start.

        The URL points to the qdrant service defined in docker-compose.yml.

        Args:
            embedding (HuggingFaceEmbeddings): embedding model of the collection.

        Returns:
            database (QdrantVectorStore): vector store over the collection.
        """
        qdrant_url: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
        collection_name: str = os.getenv("QDRANT_COLLECTION", "cosmic_collection")
        use_hybrid_search: bool = os.getenv("RAG_HYBRID_ENABLED", "false").strip().lower() == "true"

        print(
            set_color(
                status="info",
                information=f"[vector_database] Connecting to Qdrant at {qdrant_url}..."
            )
        )

        client: QdrantClient = QdrantClient(url=qdrant_url)

        # The vector size comes from the embedding model, so probe it once.
        if not client.collection_exists(collection_name):
            embedding_dim: int = len(embedding.embed_query("dimension probe"))

            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_dim,
                    distance=models.Distance.COSINE
                ),
                sparse_vectors_config=(
                    VectorDatabase.sparse_vectors_config()
                    if use_hybrid_search
                    else None
                )
            )

            print(
                set_color(
                    status="info",
                    information=f"[vector_database] Created Qdrant collection: {collection_name}"
                )
            )

        database: QdrantVectorStore = QdrantVectorStore(
            client=client,
            **VectorDatabase._hybrid_store_kwargs(
                client=client,
                collection_name=collection_name,
                use_hybrid_search=use_hybrid_search
            ),
            collection_name=collection_name,
            embedding=LastQueryCachedEmbeddings(embeddings=embedding)
        )

        print(
            set_color(
                status="success",
                information=f"[vector_database] Connected to Qdrant collection: {collection_name}"
            )
        )

        # Set search strategy.
        database.distance_strategy = DistanceStrategy.COSINE

        return database


    @staticmethod
    def _title_enrich(
        chunks: list[Document]
    ) -> list[Document]:
        """
        Prepend the document title to each chunk before it is embedded.

        Embedding a chunk as "<title>: <body>" anchors the dense vector to its
        source document, which helps when the query names the document. The
        stored text carries the prefix too (the vector store embeds what it
        stores), so `TITLE_PREFIXED_KEY` marks it for `_strip_title()`.

        Args:
            chunks (list[Document]): chunks of one or more documents.

        Returns:
            enriched (list[Document]): chunks with the title prepended; chunks
                without a title are returned unchanged.
        """
        enriched: list[Document] = []

        for chunk in chunks:
            title: str = str(chunk.metadata.get("title", "")).strip()

            if title and not chunk.page_content.startswith(title):
                enriched.append(
                    Document(
                        page_content=f"{title}: {chunk.page_content}",
                        metadata={**chunk.metadata, TITLE_PREFIXED_KEY: True}
                    )
                )

            else:
                enriched.append(chunk)

        return enriched


    @staticmethod
    def _strip_title(
        doc: Document
    ) -> Document:
        """
        Remove the title prefix added by `_title_enrich()`, so the LLM context
        does not repeat the title in every chunk.

        Args:
            doc (Document): retrieved chunk.

        Returns:
            doc (Document): chunk with its plain body text.
        """
        prefix: str = f"{doc.metadata.get('title', '')}: "

        if doc.metadata.get(TITLE_PREFIXED_KEY) and doc.page_content.startswith(prefix):
            doc.page_content = doc.page_content[len(prefix):]

        return doc


    def similarity_search_with_relevance_scores(
        self,
        *args: Any,
        **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """
        Retriever from the vector database.

        Returns:
            retrieved (list[tuple[Document, float]]): (chunk, relevance score)
                pairs, with title prefixes removed from the chunk text.
        """
        retrieved: list[tuple[Document, float]] = self.database.similarity_search_with_relevance_scores(
            *args,
            **kwargs
        )

        return [
            (VectorDatabase._strip_title(doc=doc), score)
            for (doc, score) in retrieved
        ]


    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """ Score query text pairs with a cross encoder reranker """
        if not texts:
            return []

        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                reranker_kwargs: dict[str, Any] = {"device": self.device}

                if self._model_cache_dir:
                    reranker_kwargs["cache_folder"] = self._model_cache_dir

                self._reranker = CrossEncoder(self.reranker_model, **reranker_kwargs)
            except Exception as e:
                print(set_color("warning", f"Reranker load failed ({self.reranker_model}): {e}"))
                return []

        try:
            scores = self._reranker.predict([(query, t) for t in texts])
            return [float(s) for s in scores]
        except Exception as e:
            print(set_color("warning", f"Reranker scoring failed: {e}"))
            return []


    def quit(self):
        """
        Release document analyser model.
        """
        if self.database_update_embedding:
            del self.database_update_embedding


    def add_documents(
        self,
        document_paths
    ):
        """
        Add context from a document or multiple documents to the vector database.

        Args:
            document_paths (string or list): a document path or multiple such paths.
        """
        print("Adding documents to vector database...")

        # Set as a list for loop.
        if not isinstance(document_paths, list):
            document_paths = [document_paths]

        # Update per document.
        for document_path in document_paths:
            if not Path(document_path).resolve(strict=True).exists(follow_symlinks=True):
                continue

            print(f"{document_path=}")
            self.update_database_from_document(document_path)


    def add_document_directory(
        self,
        document_dir: str
    ):
        """
        Add all .pdf in a folder to the vector database.

        Args:
            document_dir (str): a directory of .pdf to be added to the vector database.
        """
        self.document_dir: Path = Path(document_dir).resolve(strict=True)

        if self.document_dir.exists(follow_symlinks=True):
            # Find all pdf in a folder.
            document_paths = glob(f"{str(object=self.document_dir)}/*.pdf")

            # Add these documents.
            self.add_documents(document_paths)


    def update_database_from_document(
        self,
        document_path: str,
        extra_metadata: dict | None = None,
    ) -> int:
        """
        Add a document to the vector database.

        Args:
            document_path (str): a document path.
        """
        extra_metadata = extra_metadata or {}

        # Check if the document exists.
        self.document_path: Path = Path(document_path).resolve(strict=False)

        if self.document_path.exists(follow_symlinks=True):
            # Prefer the clean original title from metadata; fall back to the
            # on-disk stem (which still carries the file_id prefix).
            document_title = extra_metadata.get("title") or self.document_path.stem

            # Read pages of a document.
            loader = PyPDFLoader(self.document_path)
            pages = loader.load_and_split() # split by page number

            for i in range(len(pages)):
                pages[i].page_content = pages[i].page_content.replace("\t", " ")

            # Split each page into tokens.
            document_processed = []

            for doc in pages:
                # Attach per-document metadata (propagates to all chunks).
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata.update(extra_metadata)
                doc.metadata["title"] = document_title

                # Scope-level dedup is already handled by the watcher's content_hash
                if not extra_metadata:
                    hits = self.similarity_search_with_relevance_scores(
                        doc.page_content,
                        k=1
                    )

                    if hits:
                        content_retrieved, similarity_score = hits[0]

                        # Skip if already in the database or has a high similiarity.
                        if similarity_score >= self.vector_database_update_threshold \
                            or content_retrieved.page_content.find(doc.page_content) > -1 \
                            or doc.page_content.find(content_retrieved.page_content) > -1:
                            continue

                # Ready to add to the vector database.
                chunks = self.document_splitter.split_documents([doc])
                for chunk in chunks:
                    if chunk.metadata is None:
                        chunk.metadata = {}
                    for key, value in extra_metadata.items():
                        chunk.metadata.setdefault(key, value)
                    chunk.metadata.setdefault("title", document_title)
                document_processed += chunks

            # Obtain new knowledge from the splitted tokens.
            if len(document_processed) > 0:  # for invalid pdf such as a scanned .pdf
                # Embed each chunk with its document title (see `_title_enrich()`).
                # Qdrant handles persistence automatically; no explicit save is needed.
                self.database.add_documents(VectorDatabase._title_enrich(chunks=document_processed))

                print(
                    set_color(
                        status="info",
                        information=f"Add '{str(object=self.document_path)}'."
                    )
                )
            else:
                print(
                    set_color(
                        status="warning",
                        information=f"Contents of '{str(object=self.document_path)}' exist."
                    )
                )

            return len(document_processed)
        else:
            print(
                set_color(
                    status="warning",
                    information=f"Document {str(object=document_path)} not exists."
                )
            )
            return 0


    def get_session_document_ids(
        self,
        user_id: str,
        session_id: str,
    ) -> list[tuple[str, str]]:
        """
        List the attached files indexed under a user's chat session.

        Args:
            user_id (str): owner of the session.
            session_id (str): chat session.

        Returns:
            documents (list[tuple[str, str]]): one (document_id, file_name) pair
                per distinct document, in first-seen order.
        """
        if not (user_id and session_id):
            return []

        session_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.memory_type",
                    match=models.MatchValue(value="session"),
                ),
                models.FieldCondition(
                    key="metadata.user_id",
                    match=models.MatchValue(value=user_id),
                ),
                models.FieldCondition(
                    key="metadata.session_id",
                    match=models.MatchValue(value=session_id),
                ),
            ]
        )

        documents: dict[str, str] = {}
        next_offset = None
        while True:
            points, next_offset = self.database.client.scroll(
                collection_name=self.database.collection_name,
                scroll_filter=session_filter,
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=next_offset,
            )
            for point in points:
                payload = point.payload or {}
                meta = payload.get("metadata", payload)
                document_id = meta.get("document_id")
                if document_id and document_id not in documents:
                    documents[document_id] = str(meta.get("file_name") or meta.get("title") or "")
            if next_offset is None:
                break

        return list(documents.items())


    def delete_by_session_id(self, session_id: str) -> int:
        """ 
        Remove all vector points belonging to a chat session.
        Scoped strictly to `metadata.session_id == session_id`, so other
        sessions' vectors are left untouched. Unlike `delete_by_document_id`,
        this does not swallow errors - callers are expected to handle/log
        failures themselves.

        Returns:
            int: number of points deleted.
        """
        session_filter = models.Filter(
            must = [models.FieldCondition(key = "metadata.session_id",
                                          match=models.MatchValue(value=session_id))]
        )

        point_count = self.database.client.count(
            collection_name=self.database.collection_name,
            count_filter=session_filter,
            exact=True,
        ).count

        if point_count == 0:
            return 0

        self.database.client.delete(
            collection_name=self.database.collection_name,
            points_selector=models.FilterSelector(filter=session_filter),
        )

        return point_count


    def delete_by_document_id(self, document_id: str) -> None:
        """ Remove all vector points belonging to a document to avoid stale chunks """
        try:
            from qdrant_client import models

            self.database.client.delete(
                collection_name=self.database.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.document_id",
                                match=models.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
            )
        except Exception as e:
            print(
                set_color(
                    status="warning",
                    information=f"Failed to delete vectors for document '{document_id}': {e}"
                )
            )


    def update_database_from_text(
        self,
        text: str,
        extra_metadata: dict | None = None,
    ):
        """
        Add a sentence to the vector database.

        Args:
            text (str): a text sentence.

        Returns:
            status (int): skip (-1) or not (0).
        """
        if text != '':
            # Skip for high-similar text. On an empty collection the search
            hits = self.similarity_search_with_relevance_scores(text, k=1)

            if hits:
                content_retrieved = hits[0][0].page_content

                # If the same as existing contents, skip the text
                if content_retrieved.find(text) > -1:
                    print(set_color(
                        "warning",
                        f"Similar contents found: '{content_retrieved}' for '{text}'."
                    ))

                    return -1

            # Update the text with timestamp.
            text = f"{text} by the date {self.time_stamper}"

            # Add text to database with per-tier payload
            if extra_metadata:
                self.database.add_texts([text], metadatas=[extra_metadata])
            else:
                self.database.add_texts([text])

            # Qdrant handles persistence automatically; no explicit save is needed.

            # Print the progress.
            print(
                set_color(
                    status='info',
                    information=f"Update database with '{text}'."
                )
            )

            return 0


class LastQueryCachedEmbeddings(Embeddings):
    def __init__(
        self,
        embeddings: Embeddings
    ) -> None:
        """
        Embeddings that reuse the vector of the most recent query.

        One question is searched in up to two passes (attached file, then the
        knowledge base), each embedding the same text; the second pass is served
        from here. Documents are always embedded fresh.

        Args:
            embeddings (Embeddings): embedding model to wrap.
        """
        self._embeddings:   Embeddings                      = embeddings
        self._last_query:   tuple[str, list[float]] | None  = None
        self._lock:         Lock                            = Lock()


    def embed_documents(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        """
        Embed documents with the wrapped model.

        Args:
            texts (list[str]): texts to embed.

        Returns:
            embeddings (list[list[float]]): one embedding per text.
        """
        return self._embeddings.embed_documents(texts)


    def embed_query(
        self,
        text: str
    ) -> list[float]:
        """
        Embed a query, reusing the vector when it repeats the previous query.

        Args:
            text (str): query to embed.

        Returns:
            embedding (list[float]): embedding of the query.
        """
        with self._lock:
            if self._last_query and self._last_query[0] == text:
                return self._last_query[1]

        embedding: list[float] = self._embeddings.embed_query(text)

        with self._lock:
            self._last_query = (text, embedding)

        return embedding
