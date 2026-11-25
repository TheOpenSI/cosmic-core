#!/usr/bin/env bash

set -euo pipefail


# Navigate to project root (one level up from 'bins/')
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

STOCKFISH_VERSION="sf_19"
STOCKFISH_ASSET="stockfish-linux-x86-64-universal.tar.gz"
STOCKFISH_BINARY="stockfish-linux-x86-64-universal"
STOCKFISH_URL="https://github.com/official-stockfish/Stockfish/releases/download/${STOCKFISH_VERSION}/${STOCKFISH_ASSET}"
STOCKFISH_DIR="bins/Stockfish"
STOCKFISH_BIN="${STOCKFISH_DIR}/src/stockfish"

echo "==> Initialising git submodules..."
git submodule update --init --recursive

echo "==> Checking and preparing environment/configuration files..."

safe_copy() {
    local src="$1"
    local dest="$2"

    if [ -f "$dest" ]; then
        echo "Skipped: [$dest] already exists (left untouched)."
    else
        mkdir -p "$(dirname "$dest")"
        cp "$src" "$dest"
        echo "Copied: [$src] -> [$dest]"
    fi
}

safe_copy "examples/cosmic_cfg.example.env" "apis/cores/cosmic_cfg.env"
safe_copy "examples/cosmic_rag.example.env" "src/cosmic_rag.env"

echo "==> Checking the Stockfish binary..."
if [ -x "$STOCKFISH_BIN" ]; then
    echo "Skipped: [$STOCKFISH_BIN] already exists (left untouched)."
else
    if [ ! -d "$STOCKFISH_DIR" ]; then
        echo "ERROR: [$STOCKFISH_DIR] is missing; initialise the submodules first."
        exit 1
    fi

    mkdir -p "${STOCKFISH_DIR}/src"
    TMP_DIR="$(mktemp -d)"
    trap 'rm -rf "$TMP_DIR"' EXIT

    echo "Downloading: [$STOCKFISH_URL]"
    curl --fail --location --silent --show-error "$STOCKFISH_URL" --output "$TMP_DIR/stockfish.tar.gz"
    tar --extract --gzip --file "$TMP_DIR/stockfish.tar.gz" --directory "$TMP_DIR"
    # ponytail: prebuilt linux-x86-64 universal binary; build from bins/Stockfish/src when a custom ARCH is needed.
    cp "$TMP_DIR/stockfish/${STOCKFISH_BINARY}" "$STOCKFISH_BIN"
    chmod +x "$STOCKFISH_BIN"
    echo "Installed: [$STOCKFISH_BIN]"
fi

echo "==> Setup complete!"
echo "IMPORTANT:"
echo " 1. Review [apis/cores/cosmic_cfg.env] and [src/cosmic_rag.env]."
echo " 2. Put secrets (OPENAI_API_KEY, hf_token, hf_token_finetune) in [apis/cores/cosmic_cfg.env]."