### Core modules ###
from pathlib import Path
from dotenv import load_dotenv


# Repository root: utils/env.py -> utils/ -> root.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


# Runtime files are created by `bins/setup.*` and are gitignored; the examples
# keep a fresh clone runnable. Secrets belong in the runtime config file.
ENV_FILES: tuple[Path, ...] = (
    PROJECT_ROOT / "apis" / "cores" / "cosmic_cfg.env",
    PROJECT_ROOT / "src" / "cosmic_rag.env",
)
EXAMPLE_ENV_FILES: tuple[Path, ...] = (
    PROJECT_ROOT / "examples" / "cosmic_cfg.example.env",
    PROJECT_ROOT / "examples" / "cosmic_rag.example.env",
)


def load_env() -> None:
    """
    Load configuration into `os.environ` without overwriting anything already set.

    Precedence: OS environment > runtime env files > examples.
    """
    for env_file in (*ENV_FILES, *EXAMPLE_ENV_FILES):
        if env_file.is_file():
            load_dotenv(
                dotenv_path=env_file,
                override=False
            )