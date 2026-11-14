### Core modules ###
import os
import mimetypes
import shutil
from uuid import uuid7
from fastapi import (
    APIRouter,
    UploadFile,
    status,
    HTTPException
)
from pathlib import Path
from pydantic import BaseModel
from pydantic.types import UUID7


### Type hints ###
from typing import Any


### Internal modules ###
from ...src.services.document_index import (
    DocumentIndex,
    DocumentMetadata,
    compute_content_hash
)
from ...src.services.vector_database import VectorDatabase
from ...types.tags import APITag
from ...utils.log_tool import set_color


# Embeddable file types
EMBEDDABLE_EXTENSIONS:  set[str] = {".pdf"}

# Global and User document index instances. These are wired up at startup.
_global_document_index: DocumentIndex | None = None
_user_document_index:   DocumentIndex | None = None

# Vector database used to embed uploaded files.
_vector_database:       VectorDatabase | None = None


class SessionDeleteRequest(BaseModel):
    chat_id: UUID7
    user_id: UUID7


def set_document_indexes(
    global_index: DocumentIndex,
    user_index: DocumentIndex
) -> None:
    """Initialize document indexes (called once at startup)."""
    global _global_document_index, _user_document_index

    _global_document_index = global_index
    _user_document_index = user_index


def set_vector_database(vector_database: VectorDatabase) -> None:
    """Initialize the vector database (called once at startup)."""
    global _vector_database

    _vector_database = vector_database


router: APIRouter = APIRouter(
    prefix="/api/v1/memory",
    tags=[APITag.memory]
)


@router.post(
    path="/upload",
    status_code=status.HTTP_201_CREATED
)
async def upload_file(
    file:               UploadFile,
    memory_type:        str = "session",
    chat_session_id:    str = "default",
    user_id:            str = "default"
) -> dict[str, str | int]:
    if memory_type == "global_memory":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Global memory uploads are not supported at the API level."
        )

    # ---- Duplicate filename check ----
    # Reject the upload if the same filename already exists in the index
    # for this user / session / memory_type combination.
    if _user_document_index:
        existing_docs = _user_document_index.get_documents_by_filter(
            user_id=user_id,
            session_id=chat_session_id if memory_type == "session" else None,
            memory_type=memory_type,
        )
        for doc in existing_docs:
            if doc.file_name == file.filename:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"A file with the name '{file.filename}' already exists."
                )

    file_id: str = uuid7().hex[:8]

    if memory_type == "user":
        save_dir = Path(f"/app/data/memories/users/{user_id}")
    else:  # memory_type == "session"
        save_dir = Path(f"/app/data/memories/users/{user_id}/sessions/{chat_session_id}")

    save_dir.mkdir(parents=True, exist_ok=True)

    # combine path + file name -> example: Path("/app/data/chat_session_memory/123/manile.pdf")
    file_path: Path = save_dir / f"{file_id}_{file.filename}"

    # Create a folder and write that content in opened file
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Compute content hash for change-detection
    content_hash:   str = compute_content_hash(str(file_path))
    file_ext:       str = os.path.splitext(str(file.filename))[1].lower()

    # Embed into the vector database with per-tier metadata
    # so retrieval can be scoped to this user / session
    chunk_count: int = 0
    if (
        _vector_database is not None
        and
        file_ext in EMBEDDABLE_EXTENSIONS
    ):
        payload: dict[str, Any] = DocumentMetadata(
            document_id=file_id,
            user_id=user_id,
            session_id=(
                chat_session_id
                if   (memory_type == "session")
                else (None)
            ),
            file_name=str(file.filename),
            memory_type=memory_type,
            content_hash=content_hash
        ).to_vector_payload()

        try:
            chunk_count = _vector_database.update_database_from_document(
                document_path=str(file_path),
                extra_metadata=payload
            )

        except Exception:
            chunk_count = 0

    # Add to document index if available
    if _user_document_index:
        _user_document_index.add_document(
            document_id=file_id,
            user_id=user_id,
            file_name=str(file.filename),
            memory_type=memory_type,
            session_id=(
                chat_session_id
                if   (memory_type == "session")
                else (None)
            ),
            chunk_count=chunk_count,
            content_hash=content_hash,
            file_size=len(content),
            file_ext=file_ext,
            content_type=(
                str(file.content_type)
                or
                mimetypes.guess_type(str(file.filename))[0]
            )
        )

    return {
        "file_id":      file_id,
        "file_name":    str(file.filename),
        "file_path":    str(file_path),
        "content_type": str(file.content_type),
        "chunk_count":  chunk_count
    }


@router.post(
    path="/session/delete",
    status_code=status.HTTP_200_OK,
)
async def delete_session_data(request: SessionDeleteRequest) -> dict[str, bool | str | int | list[str]]:
    """
    Receiver for chat-session deletion. Called when the user clicks the
    "delete chat" action, so CoSMIC can drop the on-disk files, the
    document-index metadata, and the vectors it holds for that session_id.

    Args:
        request (SessionDeleteRequest): chat_id and user_id of the deleted session.

    Returns:
        dict: A dictionary with:
            - Success bool
            - Session id
            - The number of files deleted,
            - The number of metadata removed,
            - The number of vectors deleted,
            - A list of failed paths, and
            - Whether vector cleanup failed.
    """
    chat_session_id:    str = str(request.chat_id)
    user_id:            str = str(request.user_id)

    docs: list[DocumentMetadata] = (
        _user_document_index.get_documents_by_session(chat_session_id)
        if   (_user_document_index)
        else ([])
    )

    deleted_files:  int         = 0
    failed_paths:   list[str]   = []
    session_dir:    Path        = Path(f"/app/data/memories/users/{user_id}/sessions/{chat_session_id}")

    if session_dir.exists():
        try:
            file_count: int = sum(
                    1
                    for p in session_dir.rglob("*")
                    if p.is_file()
                )

            shutil.rmtree(session_dir)

            deleted_files += file_count
            print(
                set_color(
                    status = "info",
                    information = (
                        f"Deleted session directory {session_dir} ({file_count} files) "
                        f"for session_id = {chat_session_id}"
                    )
                )
            )

        except OSError as exc:
            failed_paths.append(str(session_dir))
            print(
                set_color(
                    status = "error",
                    information = (
                        f"Failed to delete session directory {session_dir} "
                        f"for session_id = {chat_session_id}: {exc}"
                    )
                )
            )

    removed_docs: int = 0

    for doc in docs:
        if _user_document_index.remove_document(doc.document_id):
            removed_docs += 1

    vectors_deleted_count: int  = 0
    vector_cleanup_failed: bool = False

    if _vector_database is not None:
        try:
            vectors_deleted_count = _vector_database.delete_by_session_id(chat_session_id)
            print(
                set_color(
                    status = "info",
                    information = (
                        f"Deleted {vectors_deleted_count} vector(s) from Qdrant "
                        f"for session_id = {chat_session_id}"
                    )
                )
            )

        except Exception as exc:
            vector_cleanup_failed = True
            print(
                set_color(
                    status = "error",
                    information = (f"Failed to delete vectors for session_id = {chat_session_id}: {exc}")
                )
            )

    success: bool = (
        not failed_paths
        and
        not vector_cleanup_failed
    )
    print(
        set_color(
            status = "info",
            information = (
                f"Session cleanup for session_id = {chat_session_id}: "
                f"files_deleted = {deleted_files}, metadata_removed = {removed_docs}, "
                f"vectors_deleted_count = {vectors_deleted_count}, failed_paths = {failed_paths}, "
                f"vector_cleanup_failed = {vector_cleanup_failed}"
            )
        )
    )

    return {
        "success":                      success,
        "session_id":                   chat_session_id,
        "files_deleted_count":          deleted_files,
        "metadata_removed_count":       removed_docs,
        "vectors_deleted_count_count":  vectors_deleted_count,
        "failed_paths":                 failed_paths,
        "vector_cleanup_failed":        vector_cleanup_failed
    }