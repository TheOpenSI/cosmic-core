# Set Python environment.
FROM python:3.14-trixie AS base


FROM base AS setup
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY ./ ./

# The Stockfish binary is fetched by `bins/setup.sh|ps1` before the image build;
# make sure the executable bit survives a Windows -> Linux copy.
RUN if [ -f bins/Stockfish/src/stockfish ]; then \
        chmod +x bins/Stockfish/src/stockfish; \
    fi

RUN uv sync --frozen --no-cache


EXPOSE 3000/tcp

# TODO:
# In production environment:
# - Remove `--reload` flag (or add '--no-reload' flag if FastAPI CLI usable)
# - Change `--host` flag value to hosting server IP address

# NOTE:
# There's a known issue about ANSI escape codes output to terminal when running
# with `fastapi dev` under some conditions (https://github.com/fastapi/fastapi/discussions/13866).
# Therefore, until the CLI maintainer patched this, we'll be using `uvicorn` CLI directly.
CMD [ ".venv/bin/uvicorn", "--app-dir", "/", "--host", "0.0.0.0", "--port", "3000", "--reload", "main:app" ]