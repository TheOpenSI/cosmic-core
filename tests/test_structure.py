"""
Structure smoke test for the COSMIC-236 refactor.

Run from the repo root: `python tests/test_structure.py`.

It imports the app, checks the routed endpoints, the model/tokenise/prompt
registries, the prompt assets and the relocated path constants. The lifespan is
not started, so Qdrant, Ollama and `cosmic-db` are not required.
"""
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main as entrypoint  # noqa: E402  (registers the package for a top-level import)
from cosmic_core.prompts import system_prompts, user_prompts  # noqa: E402
from cosmic_core.src import models, tokenise  # noqa: E402
from cosmic_core.src.services.chess.chess import STOCKFISH_BINARY_PATH  # noqa: E402
from cosmic_core.utils.env import PROJECT_ROOT as ENV_PROJECT_ROOT  # noqa: E402


EXPECTED_ROUTES: set[tuple[str, str]] = {
    ("/api/v1/cosmic", "POST"),
    ("/api/v1/cosmic", "PATCH"),
    ("/api/v1/memory/session/delete", "POST"),
    ("/api/v1/memory/upload", "POST"),
    ("/api/v1/models", "GET"),
    ("/api/v1/models/pull", "POST"),
    ("/api/v1/models/pull/{job_id}", "GET"),
    ("/api/v1/models/{model_name}", "DELETE"),
}

SYSTEM_PROMPTS: tuple[str, ...] = (
    "Mistral7bv01",
    "Mistral7bInstructv01",
    "Gemma7b",
    "Gemma7bIt",
    "GPT",
    "Ollama",
    "MistralFinetuned",
    "FENNextMoveAnalyse",
    "FENNextMoveAnalyseMistralFinetuned",
)
USER_PROMPTS: tuple[str, ...] = (
    "GeneralUserPrompt",
    "FENNextMovePredict",
    "FENNextMoveAnalyse",
    "FENNextMoveAnalyseMistralFinetuned",
    "CoTGeneration",
)
MODELS: tuple[str, ...] = (
    "Mistral7bv01",
    "Mistral7bInstructv01",
    "MistralFinetuned",
    "Gemma7b",
    "Gemma7bIt",
    "GPT",
    "Ollama",
)
TOKENISERS: tuple[str, ...] = (
    "TokenizerBase",
    "Mistral7bv01",
    "Mistral7bInstructv01",
    "MistralFinetuned",
    "Gemma7b",
    "Gemma7bIt",
    "GPT",
    "Ollama",
)
LEGACY_PATHS: tuple[str, ...] = (
    "backend",
    "modules",
    "pipelines",
    "types",
    "third_party",
    "default",
    "docker/dockerfiles",
    "bins/third_party",
    "apis/cores/dependencies.py",
    "src/services/qa.py",
    "src/services/memory",
    "src/services/llms",
    "src/services/rag.py",
    "src/services/vector_database.py",
    "src/services/document_index.py",
    "src/services/storage_events.py",
    "src/services/attachment_context.py",
    "src/services/general_qa_answering/fallback_service.py",
    "src/services/general_qa_answering/system_information_service.py",
    "src/models/llms.py",
    "src/models/tokenizer.py",
    "src/query_analyser/user_prompt.py",
    "prompts/system_prompts/slms",
    "prompts/user_prompts/slms",
    "demo.py",
    "test.ipynb",
    "file_list.csv",
    ".env.example",
)
ABSOLUTE_IMPORT_PREFIXES: tuple[str, ...] = (
    "from src.",
    "from apis.",
    "from prompts.",
    "from utils.",
    "import src.",
    "import apis.",
    "import prompts.",
    "import utils.",
)


def check_routes() -> None:
    routes = {
        (route.path, method)
        for route in entrypoint.app.routes
        for method in (getattr(route, "methods", None) or ())
    }

    missing = EXPECTED_ROUTES - routes
    assert not missing, f"missing API routes: {sorted(missing)}"


def check_prompts() -> None:
    for name in SYSTEM_PROMPTS:
        assert hasattr(system_prompts, name), f"system prompt registry misses {name}"

    for name in USER_PROMPTS:
        assert hasattr(user_prompts, name), f"user prompt registry misses {name}"

    prompter = system_prompts.GPT()

    for persona in ("academic_governance", "document_qa", "system_information"):
        assert prompter._load_service_prompt(persona).strip(), f"missing prompt asset {persona}"


def check_registries() -> None:
    for name in MODELS:
        assert hasattr(models, name), f"model registry misses {name}"

    for name in TOKENISERS:
        assert hasattr(tokenise, name), f"tokenise registry misses {name}"


def check_paths() -> None:
    assert ENV_PROJECT_ROOT == PROJECT_ROOT
    assert STOCKFISH_BINARY_PATH == PROJECT_ROOT.joinpath(
        "bins",
        "Stockfish",
        "src",
        "stockfish"
    )

    for key in ("OLLAMA_SERVICE_NAME", "QDRANT_URL", "RAG_EMBEDDING_MODEL", "RAG_TOPK"):
        assert os.environ.get(key), f"missing env {key}"

    for legacy in LEGACY_PATHS:
        assert not PROJECT_ROOT.joinpath(legacy).exists(), f"legacy path still exists: {legacy}"


def check_relative_imports() -> None:
    skip_parts = {
        "Fen-To-Board-Image",
        "Stockfish",
        ".venv",
        "__pycache__",
    }

    for path in PROJECT_ROOT.rglob("*.py"):
        if skip_parts & set(path.parts):
            continue

        relative = path.relative_to(PROJECT_ROOT).as_posix()

        if not relative.startswith(("apis/", "src/", "prompts/", "utils/")):
            continue

        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()

            if stripped.startswith(ABSOLUTE_IMPORT_PREFIXES):
                raise AssertionError(f"absolute internal import in {relative}: {stripped}")


def run() -> None:
    check_routes()
    check_prompts()
    check_registries()
    check_paths()
    check_relative_imports()
    print("OK: routes, registries, prompts, paths and relative imports")


if __name__ == "__main__":
    run()