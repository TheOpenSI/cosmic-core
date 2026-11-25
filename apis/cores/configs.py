### Core modules ###
from fastapi import Request
from os import environ


### Type hint modules ###


### Internal modules ###
from ...src.opensi_cosmic import OpenSICoSMIC
from ...utils.env import load_env


# Load the runtime/example env files before anything reads os.environ.
load_env()


DEFAULT_CORS_ALLOW_ORIGINS: tuple[str, ...] = (
    "http://localhost:8080",   # FE binded Docker port (prod)
    "http://localhost:5173",   # FE binded Docker port (dev)
    "http://localhost:11434",  # Ollama binded Docker port
)


def get_opensi_cosmic(request: Request) -> OpenSICoSMIC:
    return request.app.state.opensi_cosmic


def get_cors_allow_origins() -> list[str]:
    """Docker-local origins plus any `CORS_ALLOW_ORIGIN` overrides."""
    origins: list[str] = list(DEFAULT_CORS_ALLOW_ORIGINS)

    if environ.get("CORS_ALLOW_ORIGIN"):
        origins.extend(str(object=environ.get("CORS_ALLOW_ORIGIN")).split(";"))

    return origins