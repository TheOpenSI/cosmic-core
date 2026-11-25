### Core modules ###
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue
)


### Type hint modules ###


### Internal modules ###


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEMORIES_ROOT = PROJECT_ROOT.joinpath("data", "memories")
COLLECTION_NAME = "cosmic_collection"


def _merge_move(src: Path, dst: Path) -> None:
    """Move src to dst, merging directories when dst already exists."""
    if not src.exists():
        return

    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        return

    for item in src.iterdir():
        target = dst.joinpath(item.name)

        if item.is_dir() and target.is_dir():
            for child in item.iterdir():
                child.rename(target.joinpath(child.name))
            item.rmdir()
        elif not target.exists():
            item.rename(target)

    src.rmdir()


def _rewrite_index_types(index_file: Path) -> int:
    if not index_file.is_file():
        return 0

    data = json.loads(index_file.read_text(encoding="utf-8"))
    changed = 0

    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, dict) and value.get("memory_type") == "global_memory":
                value["memory_type"] = "system"
                changed += 1

    if changed:
        index_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    return changed


def migrate_filesystem() -> None:
    system_root = MEMORIES_ROOT.joinpath("system")
    user_root = MEMORIES_ROOT.joinpath("user")
    session_root = MEMORIES_ROOT.joinpath("session")

    _merge_move(MEMORIES_ROOT.joinpath("global"), system_root)
    _merge_move(MEMORIES_ROOT.joinpath("users"), user_root)

    old_system_index = system_root.joinpath("global_document_index.json")
    new_system_index = system_root.joinpath("system_document_index.json")

    if old_system_index.is_file() and not new_system_index.is_file():
        old_system_index.rename(new_system_index)

    if user_root.is_dir():
        for user_dir in [p for p in user_root.iterdir() if p.is_dir()]:
            sessions_dir = user_dir.joinpath("sessions")

            if not sessions_dir.is_dir():
                continue

            for session_dir in [p for p in sessions_dir.iterdir() if p.is_dir()]:
                _merge_move(session_dir, session_root.joinpath(session_dir.name))

            try:
                sessions_dir.rmdir()
            except OSError:
                pass

    changed = _rewrite_index_types(new_system_index)
    changed += _rewrite_index_types(user_root.joinpath("user_document_index.json"))

    print(f"Migrated filesystem indexes: {changed} entries")


def migrate_qdrant() -> None:
    for env_file in (
        PROJECT_ROOT.joinpath("apis", "cores", "cosmic_cfg.env"),
        PROJECT_ROOT.joinpath("src", "cosmic_core.env"),
    ):
        if env_file.is_file():
            load_dotenv(dotenv_path=env_file, override=False)

    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")

    try:
        client = QdrantClient(url=qdrant_url)
    except Exception as exc:
        print(f"Skipped Qdrant migration (client unavailable): {exc}")
        return

    try:
        if not client.collection_exists(COLLECTION_NAME):
            print(f"Skipped Qdrant migration (collection {COLLECTION_NAME} does not exist)")
            return

        legacy_filter = Filter(must=[
            FieldCondition(
                key="metadata.memory_type",
                match=MatchValue(value="global_memory")
            )
        ])
        migrated = 0

        while True:
            points, _ = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=legacy_filter,
                limit=100,
                with_payload=True,
                with_vectors=False
            )

            if not points:
                break

            updated_ids = []

            for point in points:
                payload = dict(point.payload or {})
                metadata = dict(payload.get("metadata") or {})

                if metadata.get("memory_type") != "global_memory":
                    continue

                metadata["memory_type"] = "system"
                updated_ids.append(point.id)
                client.set_payload(
                    collection_name=COLLECTION_NAME,
                    payload={"metadata": metadata},
                    points=[point.id]
                )

            if not updated_ids:
                break

            migrated += len(updated_ids)

        print(f"Migrated Qdrant points: {migrated}")

    except Exception as exc:
        print(f"Skipped Qdrant migration: {exc}")

    finally:
        client.close()


if __name__ == "__main__":
    migrate_filesystem()
    migrate_qdrant()