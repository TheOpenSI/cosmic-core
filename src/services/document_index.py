### Core modules ###
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from zoneinfo import ZoneInfo

### Type hints ###

### Internal modules ###
from .base import ServiceBase


def compute_content_hash(file_path: str) -> str:
    """ Compute the SHA-256 hash of a file's bytes to distinguish a true duplicate """
    sha = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha.update(block)
    except OSError:
        return ""
    return sha.hexdigest()


class DocumentMetadata:
    """Metadata for indexed documents"""
    def __init__(
        self,
        document_id: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        file_name: str = "",
        memory_type: str = "session",
        chunk_count: int = 0,
        upload_date: Optional[str] = None,
        service_id: Optional[int] = None,
        service_name: Optional[str] = None,
        content_hash: Optional[str] = None,
        file_size: int = 0,
        file_ext: str = "",
        content_type: Optional[str] = None,
        updated_date: Optional[str] = None,
    ):
        self.document_id = document_id
        self.user_id = user_id
        self.session_id = session_id
        self.file_name = file_name
        self.memory_type = memory_type
        self.chunk_count = chunk_count
        self.upload_date = upload_date or datetime.now(tz=ZoneInfo("Australia/Sydney")).isoformat()
        self.service_id = service_id
        self.service_name = service_name
        # Enrichment fields shared by every memory tier.
        self.content_hash = content_hash
        self.file_size = file_size
        self.file_ext = file_ext or (os.path.splitext(file_name)[1].lower() if file_name else "")
        self.content_type = content_type
        self.updated_date = updated_date or self.upload_date

    @property
    def title(self) -> str:
        return os.path.splitext(self.file_name)[0] if self.file_name else ""

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "document_id": self.document_id,
            "file_name": self.file_name,
            "memory_type": self.memory_type,
            "chunk_count": self.chunk_count,
            "upload_date": self.upload_date,
            "updated_date": self.updated_date,
            "content_hash": self.content_hash,
            "file_size": self.file_size,
            "file_ext": self.file_ext,
            "content_type": self.content_type,
        }
        if self.memory_type == "global_memory":
            data["service_id"] = self.service_id
            data["service_name"] = self.service_name
        else:
            data["user_id"] = self.user_id
            data["session_id"] = self.session_id
        return data

    def to_vector_payload(self) -> Dict[str, Any]:
        # Metadata dict attached to every chunk
        payload: Dict[str, Any] = {
            "document_id": self.document_id,
            "file_name": self.file_name,
            "title": self.title,
            "memory_type": self.memory_type,
            "content_hash": self.content_hash,
            "file_size": self.file_size,
            "file_ext": self.file_ext,
            "content_type": self.content_type,
            "upload_date": self.upload_date,
            "updated_date": self.updated_date,
        }
        if self.memory_type == "global_memory":
            payload["service_id"] = self.service_id
            payload["service_name"] = self.service_name
        else:
            payload["user_id"] = self.user_id
            payload["session_id"] = self.session_id
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        allowed = {
            "document_id", "user_id", "session_id", "file_name", "memory_type",
            "chunk_count", "upload_date", "service_id", "service_name",
            "content_hash", "file_size", "file_ext", "content_type", "updated_date",
        }
        return cls(**{k: v for k, v in data.items() if k in allowed})


class DocumentIndex(ServiceBase):
    """In-memory document metadata index with optional persistence"""

    def __init__(
        self,
        persist_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize document index.

        Args:
            persist_path (str, optional): Path to persist index as JSON. Defaults to None.
        """
        super().__init__(**kwargs)

        self.documents: Dict[str, DocumentMetadata] = {}
        self.persist_path: Optional[Path] = None

        if persist_path:
            self.persist_path = Path(persist_path).resolve()
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def add_document(
        self,
        document_id: str,
        user_id: Optional[str] = None,
        file_name: str = "",
        memory_type: str = "session",
        session_id: Optional[str] = None,
        chunk_count: int = 0,
        service_id: Optional[int] = None,
        service_name: Optional[str] = None,
        content_hash: Optional[str] = None,
        file_size: int = 0,
        file_ext: str = "",
        content_type: Optional[str] = None,
        upload_date: Optional[str] = None,
        updated_date: Optional[str] = None,
    ) -> DocumentMetadata:
        # Add document metadata to index
        metadata = DocumentMetadata(
            document_id=document_id,
            user_id=user_id,
            session_id=session_id,
            file_name=file_name,
            memory_type=memory_type,
            chunk_count=chunk_count,
            upload_date=upload_date,
            service_id=service_id,
            service_name=service_name,
            content_hash=content_hash,
            file_size=file_size,
            file_ext=file_ext,
            content_type=content_type,
            updated_date=updated_date,
        )

        self.documents[document_id] = metadata
        self._save_to_disk()

        return metadata

    def get_document_info(self, document_id: str) -> Optional[DocumentMetadata]:
        """Get metadata for a specific document"""
        return self.documents.get(document_id)

    def get_documents_by_filter(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        memory_type: Optional[str] = None
    ) -> List[DocumentMetadata]:
        """
        Retrieve documents matching filter criteria.

        Args:
            user_id (str, optional): Filter by user
            session_id (str, optional): Filter by session
            memory_type (str, optional): Filter by memory type

        Returns:
            List[DocumentMetadata]: Matching documents
        """
        results = []

        for doc in self.documents.values():
            # Apply filters
            if user_id and doc.user_id != user_id:
                continue
            if session_id and doc.session_id != session_id:
                continue
            if memory_type and doc.memory_type != memory_type:
                continue

            results.append(doc)

        # Sort by upload date (newest first)
        results.sort(key=lambda x: x.upload_date, reverse=True)

        return results

    def get_documents_by_user(self, user_id: str) -> List[DocumentMetadata]:
        """Get all documents for a user"""
        return self.get_documents_by_filter(user_id=user_id)

    def get_documents_by_session(self, session_id: str) -> List[DocumentMetadata]:
        """Get all documents for a session"""
        return self.get_documents_by_filter(session_id=session_id)

    def get_documents_by_service(self, service_id: int) -> List[DocumentMetadata]:
        """Get all documents for a specific service"""
        return [d for d in self.documents.values() if d.service_id == service_id]

    def get_documents_by_service_name(self, service_name: str) -> List[DocumentMetadata]:
        """Get all documents for a specific service by name"""
        return [d for d in self.documents.values() if d.service_name == service_name]

    def has_file_in_service(self, file_name: str, service_name: str) -> bool:
        """Check if a file with the same name already exists under a given service.

        Different services are allowed to have files with the same name, but
        within a single service each filename must be unique.

        Args:
            file_name (str): Original file name (without the file_id prefix).
            service_name (str): Service folder name.

        Returns:
            True if a document with the same file_name + service_name already
            exists in the index.
        """
        for doc in self.documents.values():
            if doc.file_name == file_name and doc.service_name == service_name:
                return True
        return False

    def get_file_in_service(
        self, file_name: str, service_name: str
    ) -> Optional[DocumentMetadata]:
        """ Return the indexed document with this file_name under this service.
        Distinguish a true duplicate with same content_hash to reject) from
        an update and different content_hash to re-index when a same-named file is
        added again. Returns None if no match exists.
        """
        for doc in self.documents.values():
            if doc.file_name == file_name and doc.service_name == service_name:
                return doc
        return None

    def remove_document(self, document_id: str) -> bool:
        """
        Remove document from index.

        Args:
            document_id (str): Document to remove

        Returns:
            bool: True if document was removed, False if not found
        """
        if document_id in self.documents:
            del self.documents[document_id]
            self._save_to_disk()
            return True
        return False

    def get_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about indexed documents.

        Args:
            user_id (str, optional): Get stats for specific user only

        Returns:
            Dict with statistics
        """
        docs = (
            self.get_documents_by_user(user_id)
            if user_id
            else list(self.documents.values())
        )

        total_chunks = sum(doc.chunk_count for doc in docs)
        docs_by_type = {}

        for doc in docs:
            memory_type = doc.memory_type
            if memory_type not in docs_by_type:
                docs_by_type[memory_type] = {"count": 0, "chunks": 0}
            docs_by_type[memory_type]["count"] += 1
            docs_by_type[memory_type]["chunks"] += doc.chunk_count

        return {
            "total_documents": len(docs),
            "total_chunks": total_chunks,
            "documents_by_type": docs_by_type
        }

    def _save_to_disk(self) -> None:
        """Persist index to disk"""
        if not self.persist_path:
            return

        data = {
            doc_id: doc.to_dict()
            for doc_id, doc in self.documents.items()
        }

        with self.persist_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_from_disk(self) -> None:
        """Load index from disk"""
        if not self.persist_path or not self.persist_path.exists():
            return

        try:
            with self.persist_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                self.documents = {
                    doc_id: DocumentMetadata.from_dict(metadata)
                    for doc_id, metadata in data.items()
                }
        except Exception as e:
            print(f"Error loading document index from {self.persist_path}: {e}")
            self.documents = {}

    def sync_with_filesystem(self, base_path: str = "/app/data/memories/global/") -> None:
        """Sync document_index with actual filesystem state.

        Removes entries for files that no longer exist and logs orphaned files.

        Args:
            base_path (str): Base path to scan for files. Defaults to global memory path.
        """
        import os

        if not os.path.exists(base_path):
            return

        # Collect all files in the filesystem
        filesystem_files = set()
        for root, dirs, files in os.walk(base_path):
            for file in files:
                filepath = os.path.join(root, file)
                filesystem_files.add(filepath)

        # Remove index entries for missing files
        documents_to_remove = []
        for doc_id, metadata in self.documents.items():
            if metadata.memory_type == "global_memory":
                # Reconstruct expected file path
                service_name = metadata.service_name or ""
                expected_path = os.path.join(
                    base_path, service_name, f"{doc_id}_{metadata.file_name}"
                )
                if not os.path.exists(expected_path):
                    documents_to_remove.append(doc_id)

        for doc_id in documents_to_remove:
            self.remove_document(doc_id)

        self._save_to_disk()