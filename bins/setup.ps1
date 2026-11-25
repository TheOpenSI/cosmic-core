$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $ProjectRoot

try {
    $StockfishVersion = "sf_19"
    $StockfishAsset = "stockfish-linux-x86-64-universal.tar.gz"
    $StockfishBinary = "stockfish-linux-x86-64-universal"
    $StockfishUrl = "https://github.com/official-stockfish/Stockfish/releases/download/$StockfishVersion/$StockfishAsset"
    $StockfishDir = "bins/Stockfish"
    $StockfishBin = Join-Path $StockfishDir "src/stockfish"

    Write-Output "==> Initialising git submodules..."
    git submodule update --init --recursive

    Write-Output "==> Checking and preparing environment/configuration files..."

    function Copy-IfMissing {
        param([string]$Source, [string]$Destination)

        if (Test-Path $Destination) {
            Write-Output "Skipped: [$Destination] already exists (left untouched)."
        } else {
            New-Item -ItemType Directory -Force (Split-Path $Destination) | Out-Null
            Copy-Item -Path $Source -Destination $Destination
            Write-Output "Copied: [$Source] -> [$Destination]"
        }
    }

    Copy-IfMissing "examples/cosmic_cfg.example.env" "apis/cores/cosmic_cfg.env"
    Copy-IfMissing "examples/cosmic_rag.example.env" "src/cosmic_rag.env"

    Write-Output "==> Checking the Stockfish binary..."
    if (Test-Path $StockfishBin) {
        Write-Output "Skipped: [$StockfishBin] already exists (left untouched)."
    } else {
        if (-not (Test-Path $StockfishDir)) {
            throw "[$StockfishDir] is missing; initialise the submodules first."
        }

        New-Item -ItemType Directory -Force (Join-Path $StockfishDir "src") | Out-Null
        $TempDir = New-Item -ItemType Directory -Force (Join-Path ([System.IO.Path]::GetTempPath()) ([System.Guid]::NewGuid().ToString()))

        try {
            Write-Output "Downloading: [$StockfishUrl]"
            Invoke-WebRequest -Uri $StockfishUrl -OutFile (Join-Path $TempDir "stockfish.tar.gz")
            tar --extract --gzip --file (Join-Path $TempDir "stockfish.tar.gz") --directory $TempDir
            # ponytail: prebuilt linux-x86-64 universal binary; build from bins/Stockfish/src when a custom ARCH is needed.
            Copy-Item -Path (Join-Path $TempDir "stockfish/$StockfishBinary") -Destination $StockfishBin
            Write-Output "Installed: [$StockfishBin]"
        } finally {
            Remove-Item -Recurse -Force $TempDir
        }
    }

    Write-Output "==> Setup complete!"
    Write-Output "IMPORTANT:"
    Write-Output " 1. Review [apis/cores/cosmic_cfg.env] and [src/cosmic_rag.env]."
    Write-Output " 2. Put secrets (OPENAI_API_KEY, hf_token, hf_token_finetune) in [apis/cores/cosmic_cfg.env]."
} finally {
    Pop-Location
}