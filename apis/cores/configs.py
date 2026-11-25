### Core modules ###
from os import environ


### Type hints ###


### Internal modules ###
from ...utils.env import load_env


# Load the runtime/example env files before anything reads os.environ.
load_env()


DEFAULT_CORS_ALLOW_ORIGINS: tuple[str, ...] = (
    "http://localhost:8080",   # FE binded Docker port (prod)
    "http://localhost:5173",   # FE binded Docker port (dev)
    "http://localhost:11434",  # Ollama binded Docker port
)


def get_cors_allow_origins() -> list[str]:
    """Docker-local origins plus any `CORS_ALLOW_ORIGIN` overrides."""
    origins: list[str] = list(DEFAULT_CORS_ALLOW_ORIGINS)

    if environ.get("CORS_ALLOW_ORIGIN"):
        origins.extend(str(object=environ.get("CORS_ALLOW_ORIGIN")).split(";"))

    return origins