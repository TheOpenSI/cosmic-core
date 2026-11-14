### Core modules ###
import os
from pathlib import Path
from glob import glob
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.document_loaders import PyPDFLoader
from qdrant_client import QdrantClient, models


### Type hints ###


### Internal modules ###
from ...utils.log_tool import set_color
from .base import ServiceBase


class VectorDatabase(ServiceBase):
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

        # For document analysis and knowledge database generation/update.
        # Known short aliases map to their full Hugging Face repo id; any other
        # value is treated as a direct Hugging Face model name.
        EMBEDDING_MODEL_DICT = {'gte-small': "thenlper/gte-small"}

        # Set page separators.
        MARKDOWN_SEPARATORS = ["\n\n", "\n", ""]

        # Set splitter to split a document into pages.
        self.document_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            add_start_index=True,
            strip_whitespace=True,
            separators=MARKDOWN_SEPARATORS,
        )

        # Build a document analyser.
        document_analyser_model = os.environ["RAG_EMBEDDING_MODEL"]
        EMBEDDING_MODEL_NAME = EMBEDDING_MODEL_DICT.get(document_analyser_model, document_analyser_model)

        self.database_update_embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            multi_process=False,  # TODO
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Initialize Qdrant client connecting to the Docker service.
        # Collection name defaults to "cosmic_collection".
        # The URL points to the qdrant service defined in docker-compose.yml.
        qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        collection_name = "cosmic_collection"

        print(set_color("info", f"Connecting to Qdrant at {qdrant_url}..."))

        client = QdrantClient(url=qdrant_url)

        # Check collection exists if not create one
        if not client.collection_exists(collection_name):
            embedding_dim = len(
                self.database_update_embedding.embed_query("dimension probe")
            )
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_dim,
                    distance=models.Distance.COSINE,
                ),
            )
            print(set_color("info", f"Created Qdrant collection: {collection_name}"))

        # Attach to the existing collection
        self.database = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=self.database_update_embedding,
        )

        print(set_color("success", f"Connected to Qdrant collection: {collection_name}"))

        # Set search strategy.
        self.database.distance_strategy = DistanceStrategy.COSINE

        # Set a time stamp to highlight the most recently updated information.
        self.time_stamper:              str = datetime.now(tz=ZoneInfo(key="Australia/Sydney")).strftime(format="%B, %Y")


    def similarity_search_with_relevance_scores(
        self,
        *args,
        **kwargs
    ):
        """
        Retriever from the vector database.

        Returns:
            context (str): retrieved information.
        """
        return self.database.similarity_search_with_relevance_scores(*args, **kwargs)


    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """ Score query text pairs with a cross encoder reranker """
        if not texts:
            return []

        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(self.reranker_model, device=self.device)
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
                # Qdrant handles persistence automatically; no explicit save is needed.
                self.database.add_documents(document_processed)

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