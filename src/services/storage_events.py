### Core modules ###
import json
import logging
import mimetypes
import os
import time
from uuid import uuid7
from datetime import (
    datetime,
    timezone
)
from zoneinfo import ZoneInfo
from watchdog.events import (
    FileSystemEventHandler,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent
)
# PollingObserver is used instead of the native observer
from watchdog.observers.polling import PollingObserver


### Type hints ###
from typing import Optional


### Internal modules ###
from .document_index import (
    DocumentIndex,
    DocumentMetadata,
    compute_content_hash
)


logger = logging.getLogger(__name__)

# Only these extensions can be embedded into the vector database (PyPDFLoader).
EMBEDDABLE_EXTENSIONS = {".pdf"}


class StorageEventHandler(FileSystemEventHandler):
    def __init__(
        self,
        document_index: DocumentIndex,
        watched_path: str,
        get_rag_required_services=None,
        get_service_id_by_name=None,
        vector_database=None,
    ):
        self.document_index = document_index
        self.watched_path = watched_path
        self.get_rag_required_services = get_rag_required_services
        self.get_service_id_by_name = get_service_id_by_name
        self.vector_database = vector_database
        self.event_log_path = os.path.join(
            os.path.dirname(watched_path), "storage_events.log"
        )

    @staticmethod
    def _is_settled(file_path: str) -> bool:
        """
        Compares the size across a short interval; an in-progress copy will keep
        growing, so wait for the next event.
        """
        try:
            size1 = os.path.getsize(file_path)
            time.sleep(0.5)
            size2 = os.path.getsize(file_path)
        except OSError:
            return False
        return size1 == size2 and size1 > 0

    def on_created(self, event: FileCreatedEvent) -> None:
        if event.is_directory:
            return

        try:
            # If the file is still being copied, skip and let the next poll re-fire
            if not self._is_settled(event.src_path):
                return
            self.process_file(event.src_path)
        except Exception as e:
            logger.error(f"Error processing file creation event: {e}", exc_info=True)

    def on_modified(self, event: FileModifiedEvent) -> None:
        if event.is_directory:
            return

        try:
            if not self._is_settled(event.src_path):
                return
            self.process_file(event.src_path)
        except Exception as e:
            logger.error(f"Error processing file modification event: {e}", exc_info=True)

    def process_file(self, file_path: str) -> None:
        """Process a single file for indexing.
        If it has a valid file identifier and is already indexed, compare hashes.
        Skip if unchanged. If changed, reindex.
        """
        rel_path = os.path.relpath(file_path, self.watched_path)
        parts = rel_path.split(os.sep)

        if len(parts) < 2:
            logger.warning(f"File not in service folder: {file_path}")
            return

        service_name = parts[0]
        filename = parts[-1]

        content_hash = compute_content_hash(file_path)

        # Check if file already has file_id format
        file_id, original_filename = self._extract_file_id(filename)

        # If it already has a file_id, check if it's already indexed
        if file_id and file_id in self.document_index.documents:
            existing = self.document_index.documents[file_id]
            if existing.content_hash == content_hash:
                # Unchanged content, nothing to do.
                return
            # Content changed re-index under the same document_id
            self._reindex(file_path, existing, service_name, content_hash)
            return

        # Determine the original filename for dedup check
        dedup_filename = original_filename if file_id else filename

        # Per-service filename handling
        existing = self.document_index.get_file_in_service(dedup_filename, service_name)
        if existing is not None:
            if existing.content_hash == content_hash:
                # Duplicate with same name, same content then drop from disk.
                logger.warning(
                    f"Rejected duplicate: file '{dedup_filename}' already exists "
                    f"under service '{service_name}'. Removing the duplicate from disk."
                )
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted duplicate file from disk: {file_path}")
                except OSError as e:
                    logger.error(f"Failed to delete duplicate file {file_path}: {e}")
                return

            # Same name, different content then update the existing document.
            # Overwrite the existing on-disk file (which carries the file_id
            # prefix) with the new content, then re-index under the same id.
            target_path = os.path.join(
                os.path.dirname(file_path),
                f"{existing.document_id}_{existing.file_name}",
            )
            try:
                os.replace(file_path, target_path)
            except OSError as e:
                logger.error(f"Failed to overwrite file for update {target_path}: {e}")
                return
            self._reindex(target_path, existing, service_name, content_hash)
            return

        renamed = False

        if not file_id:
            file_id = uuid7().hex[:8]
            original_filename = filename
            renamed = True

            # Rename file to add file_id prefix
            new_filename = f"{file_id}_{original_filename}"
            new_path = os.path.join(os.path.dirname(file_path), new_filename)

            try:
                os.rename(file_path, new_path)
                file_path = new_path
                logger.info(f"Renamed file: {filename} -> {new_filename}")
            except OSError as e:
                logger.error(f"Failed to rename file {file_path}: {e}")
                return

        # Resolve the service and confirm it has memory_capability
        service_id = self._resolve_memory_service(service_name)

        if service_id is None:
            logger.warning(
                f"Service '{service_name}' has no memory_capability / not in registry. "
                f"Skipping indexing."
            )
            self._log_event(
                "created", file_path, file_id, original_filename, service_name,
                None, os.path.getsize(file_path), "skipped", renamed,
            )
            return

        # Embed into the vector database and index
        chunk_count = self._embed_document(
            file_path, file_id, original_filename, service_id, service_name, content_hash
        )

        self.document_index.add_document(
            document_id=file_id,
            file_name=original_filename,
            memory_type="global_memory",
            chunk_count=chunk_count,
            service_id=service_id,
            service_name=service_name,
            content_hash=content_hash,
            file_size=os.path.getsize(file_path),
            file_ext=os.path.splitext(original_filename)[1].lower(),
            content_type=mimetypes.guess_type(original_filename)[0],
        )

        self._log_event(
            "created", file_path, file_id, original_filename, service_name,
            service_id, os.path.getsize(file_path), "indexed", renamed,
        )

        logger.info(
            f"Indexed file: {original_filename} (id: {file_id}, "
            f"service: {service_name}, chunks: {chunk_count})"
        )

    def _resolve_memory_service(self, service_name: str) -> Optional[int]:
        # Return the service_id if the service has memory_capability, else None
        rag_services = self.get_rag_required_services()
        service_id = self.get_service_id_by_name(service_name)

        if service_id is None or service_id not in rag_services:
            return None
        return service_id

    def _embed_document(
        self,
        file_path: str,
        file_id: str,
        original_filename: str,
        service_id: int,
        service_name: str,
        content_hash: str,
    ) -> int:
        # Embed a global memory document into the vector DB, returning chunks
        ext = os.path.splitext(original_filename)[1].lower()
        if self.vector_database is None or ext not in EMBEDDABLE_EXTENSIONS:
            return 0

        payload = DocumentMetadata(
            document_id=file_id,
            file_name=original_filename,
            memory_type="global_memory",
            service_id=service_id,
            service_name=service_name,
            content_hash=content_hash,
        ).to_vector_payload()

        try:
            return self.vector_database.update_database_from_document(
                file_path, extra_metadata=payload
            )
        except Exception as e:
            logger.error(f"Failed to embed '{file_path}': {e}", exc_info=True)
            return 0

    def _reindex(
        self,
        file_path: str,
        existing: DocumentMetadata,
        service_name: str,
        content_hash: str,
    ) -> None:
        """Re-embed a changed document under its existing document_id.

        Purges the document's stale vectors, re-embeds the new content, and
        refreshes the index row (hash, chunk_count, updated_date, size).
        """
        service_id = self._resolve_memory_service(service_name)
        if service_id is None:
            return

        if self.vector_database is not None:
            self.vector_database.delete_by_document_id(existing.document_id)

        chunk_count = self._embed_document(
            file_path, existing.document_id, existing.file_name,
            service_id, service_name, content_hash,
        )

        self.document_index.add_document(
            document_id=existing.document_id,
            file_name=existing.file_name,
            memory_type="global_memory",
            chunk_count=chunk_count,
            service_id=service_id,
            service_name=service_name,
            content_hash=content_hash,
            file_size=os.path.getsize(file_path),
            file_ext=os.path.splitext(existing.file_name)[1].lower(),
            content_type=mimetypes.guess_type(existing.file_name)[0],
            upload_date=existing.upload_date,
            updated_date=datetime.now(tz=ZoneInfo("Australia/Sydney")).isoformat(),
        )

        self._log_event(
            "modified", file_path, existing.document_id, existing.file_name,
            service_name, service_id, os.path.getsize(file_path), "reindexed", False,
        )
        logger.info(
            f"Re-indexed changed file: {existing.file_name} "
            f"(id: {existing.document_id}, chunks: {chunk_count})"
        )

    def on_deleted(self, event: FileDeletedEvent) -> None:
        if event.is_directory:
            return

        try:
            file_path = event.src_path
            filename = os.path.basename(file_path)

            file_id, original_filename = self._extract_file_id(filename)

            if not file_id:
                logger.warning(f"File does not have valid file_id format: {filename}")
                return

            # Purge the document's vectors from the shared collection so no
            # orphan chunks linger after the file is gone.
            if self.vector_database is not None:
                self.vector_database.delete_by_document_id(file_id)

            # Remove from index
            if self.document_index.remove_document(file_id):
                self._log_event(
                    "deleted",
                    file_path,
                    file_id,
                    original_filename,
                    None,
                    None,
                    0,
                    "removed",
                    False,
                )
                logger.info(f"Removed document from index: {file_id}")
            else:
                logger.warning(f"Document not found in index: {file_id}")

        except Exception as e:
            logger.error(f"Error processing file deletion event: {e}", exc_info=True)

    def _extract_file_id(self, filename: str) -> tuple[Optional[str], Optional[str]]:
        """Extract file_id and original_filename from filename.

        Expected format: {file_id}_{original_filename}
        file_id is 8-character hex string

        Returns:
            Tuple of (file_id, original_filename) or (None, None) if format invalid
        """
        if "_" not in filename:
            return None, None

        parts = filename.split("_", 1)
        potential_id = parts[0]

        # Check if it's 8-char hex string
        if len(potential_id) == 8 and all(c in "0123456789abcdef" for c in potential_id):
            return potential_id, parts[1] if len(parts) > 1 else filename

        return None, None

    def _log_event(
        self,
        event_type: str,
        file_path: str,
        file_id: str,
        original_filename: str,
        service_name: Optional[str],
        service_id: Optional[int],
        file_size: int,
        status: str,
        renamed: bool,
    ) -> None:
        """Log storage event to file."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "path": file_path.replace(os.sep, "/"),
            "document_id": file_id,
            "original_filename": original_filename,
            "service_name": service_name,
            "service_id": service_id,
            "file_size": file_size,
            "status": status,
            "renamed": renamed,
        }

        try:
            os.makedirs(os.path.dirname(self.event_log_path), exist_ok=True)
            with open(self.event_log_path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")


class StorageEventWatcher:
    def __init__(
        self,
        watched_path: str,
        document_index: DocumentIndex,
        get_rag_required_services=None,
        get_service_id_by_name=None,
        vector_database=None,
        get_raw_services=None,
    ):
        self.watched_path = watched_path
        self.document_index = document_index
        # Raw service records for folder provisioning OpenSICoSMIC.get_services(raw=True).
        self.get_raw_services = get_raw_services or (lambda: [])
        self.observer = PollingObserver()
        self.event_handler = StorageEventHandler(
            document_index,
            watched_path,
            get_rag_required_services,
            get_service_id_by_name,
            vector_database=vector_database,
        )

    def start(self) -> None:
        """Start monitoring for file system changes."""
        if not os.path.exists(self.watched_path):
            os.makedirs(self.watched_path, exist_ok=True)
            try:
                os.chmod(self.watched_path, 0o777)
            except Exception:
                pass

        # Pre-create folders for ACTIVE services with memory_capability=True
        try:
            services = self.get_raw_services()
            memory_services = [
                s for s in services
                if s.get("status") is True and s.get("memory_capability") is True
            ]

            if not memory_services:
                logger.warning(
                    "No active service with memory_capability=True — no global "
                    "memory folders created."
                )

            for service in memory_services:
                service_name = service.get("name")
                if service_name:
                    folder_path = os.path.join(self.watched_path, service_name)
                    os.makedirs(folder_path, exist_ok=True)
                    try:
                        os.chmod(folder_path, 0o777)
                    except Exception:
                        pass
                    logger.info(f"Ensured global memory folder exists with write permission: {folder_path}")
        except Exception as e:
            logger.error(f"Failed to create service-specific global folders: {e}")

        self.observer.schedule(self.event_handler, self.watched_path, recursive=True)
        self.observer.start()
        logger.info(f"StorageEventWatcher started, monitoring: {self.watched_path}")

    def stop(self) -> None:
        """Stop monitoring."""
        self.observer.stop()
        self.observer.join()
        logger.info("StorageEventWatcher stopped")

    def sync_with_filesystem(self) -> None:
        """Sync document_index with actual filesystem state on startup.

        Two-phase sync:
        1. **Prune** – remove index entries whose files no longer exist on disk.
        2. **Discover** – walk the watched directory tree and call
           ``process_file`` for every file that is not yet in the index.
           ``process_file`` internally handles deduplication (by file_id and
           by filename-within-service), so it is safe to call unconditionally.
        """
        logger.info("Starting filesystem sync...")

        # Phase 1: prune stale entries
        self.document_index.sync_with_filesystem(self.watched_path)

        # Phase 2: discover and index files that exist on disk but not in the index
        if os.path.exists(self.watched_path):
            for root, _dirs, files in os.walk(self.watched_path):
                for file in files:
                    # Skip non-document files (e.g. event logs, index JSON)
                    if file == "storage_events.log":
                        continue
                    if file.endswith(".json"):
                        continue

                    file_path = os.path.join(root, file)
                    if not os.path.isfile(file_path):
                        continue

                    try:
                        self.event_handler.process_file(file_path)
                    except Exception as e:
                        logger.error(
                            f"Error syncing file {file_path} on startup: {e}",
                            exc_info=True,
                        )

        logger.info("Filesystem sync completed")