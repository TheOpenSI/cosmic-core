<h1 align="center">OpenSI-CoSMIC<br />-<br />Cognitive System of Machine Intelligent Computing</h1>

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/ACIS-2024-oliver.svg)](https://arxiv.org/abs/2408.04910)
[![Python](https://img.shields.io/badge/Python-3.14-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Media](https://img.shields.io/badge/Media-2024-purple.svg)](https://www.canberra.edu.au/about-uc/media/newsroom/2024/november/ucs-opensi-researchers-develop-framework-to-integrate-and-interpret-ai-tools)
**This is the official implementation of OpenSI flagship product - CoSMIC.**

---
# Directory Hierarchy

```md
cosmic-core/
├── apis/                         # FastAPI application layer
│   ├── cores/                    # App setup, dependencies, configs and API tags
│   │   ├── apis.py               # FastAPI app + lifespan
│   │   ├── configs.py            # Env loading, CORS and the shared FastAPI dependency
│   │   ├── tags.py               # API tags
│   │   └── cosmic_cfg.env        # Generated config (gitignored)
│   ├── routers/
│   │   ├── apis/                 # Versioned API endpoints (memory, models)
│   │   └── regulars/cosmic.py    # Chat/query endpoint
│   └── types/schemas/chatboxes.py
├── bins/                         # Clone/setup helpers and third-party tools
│   ├── Fen-To-Board-Image/       # Git submodule (not referenced by runtime code yet)
│   ├── Stockfish/                # Git submodule; setup fetches src/stockfish
│   ├── setup.sh                  # Linux/macOS bootstrap
│   ├── setup.ps1                 # Windows bootstrap
│   └── migrate_memory_tiers.py   # One-off system/user/session migration
├── data/                         # Datasets and the on-disk memory store
│   └── memories/{system,user,session}/  # Memory tiers, priority system > user > session
├── docker/                       # Containerisation resources per service
│   ├── fastapi/
│   ├── ollama/
│   └── qdrant/
├── docs/                         # Repository conventions and documentation
├── examples/                     # Example env files and the file-list template
├── prompts/                      # Prompting logic, grouped by layer
│   ├── system_prompts/
│   │   ├── models/               # Per-company/model-family system prompts (base.py + families)
│   │   ├── services/             # Per-service system prompts and persona assets
│   │   └── query_analyser/       # Query-analyser system prompt slot
│   └── user_prompts/
│       ├── models/base.py        # Shared user prompt base
│       ├── services/             # Per-service user prompts
│       └── query_analyser/       # Query-analyser user prompts
├── src/                          # The "heart" of CoSMIC
│   ├── models/                   # LLM providers (base.py + model families)
│   │   ├── mistral/              # Mistral AI
│   │   ├── google/               # Google (Gemma, future Gemini/Nano Banana)
│   │   ├── openai/               # OpenAI (ChatGPT/GPT)
│   │   └── ollama/               # Ollama runtime provider
│   ├── tokenise/                 # Tokenising methods (base.py + company families)
│   ├── rag/                      # Retrieval-augmented generation and vector storage
│   │   ├── base.py
│   │   ├── vector_database.py
│   │   ├── document_index.py
│   │   ├── storage_events.py
│   │   └── attachment_context.py
│   ├── query_analyser/           # SLM-based query routing
│   ├── qa.py                     # QA orchestrator (beside opensi_cosmic.py)
│   ├── services/                 # Specialised AI services, one directory each
│   │   ├── base.py
│   │   ├── chess/
│   │   ├── code_generation/
│   │   ├── general_qa_answering/ # Placeholder for the future modular split
│   │   ├── fallback/
│   │   ├── system_information/
│   │   └── academic_governance/
│   ├── cosmic_core.env           # Generated core/RAG config (gitignored)
│   ├── maps.py                   # SLM name -> instance mappings
│   └── opensi_cosmic.py          # System orchestrator
├── tests/                        # Runnable test harnesses
├── utils/                        # Helper functions and shared utility scripts
├── __init__.py                   # Package root (enables relative imports)
├── .dockerignore
├── .gitattributes
├── .gitignore
├── .gitmodules
├── .python-version
├── CONTRIBUTING
├── LICENSE
├── README.md
├── compose.yaml
├── main.py                       # API entry point (relative imports + native fallback)
├── pyproject.toml
└── uv.lock
```

---
# Quick Start

Before setting up, decide which one is the correct purpose when you get to this modular repository:

> [!NOTE]
> The rest of this guide covers **Purpose 1**. For **Purpose 2**, refer to the
> setup instructions in [cosmic-docker](https://github.com/TheOpenSI/cosmic-docker) repository.

1. **Module-only**: you are working on this part of the project in isolation (e.g., only **CoSMIC**).
2. **Full-stack**: you need an end-to-end test run across all services (**Frontend &rarr; Backend &rarr; CoSMIC**).

Next, ensure you have the appropriate tools installed depending on your chosen execution method. This guide supports:

- **Native setup** (running **CoSMIC** directly on your machine)
- **Docker setup** (running **CoSMIC** in isolated containers)


| **Tool** | **Docker Setup**                       | **Native Setup**                               |
| -------- | -------------------------------------- | ---------------------------------------------- |
| Docker   | $\textcolor{green}{\text{Mandatory}}$  | $\textcolor{red}{\text{Not required}}$         |
| Python   | $\textcolor{red}{\text{Not required}}$ | $\textcolor{green}{\text{Mandatory (v3.14+)}}$ |
| uv       | $\textcolor{red}{\text{Not required}}$ | $\textcolor{green}{\text{Mandatory (latest)}}$ |
| Ollama   | $\textcolor{red}{\text{Not required}}$ | $\textcolor{green}{\text{Mandatory}}$          |


Then, start by cloning the repository using your preferred method:

```bash
# Linux/macOS
git clone https://github.com/TheOpenSI/cosmic-core.git    # Using HTTPS (recommended for most users)
git clone git@github.com:TheOpenSI/cosmic-core.git        # Using SSH (recommended if you have SSH keys configured)
```
```ps1
# Windows
git clone https://github.com/TheOpenSI/cosmic-core.git    # Using HTTPS (recommended for most users)
git clone git@github.com:TheOpenSI/cosmic-core.git        # Using SSH (recommended if you have SSH keys configured)
```

Once cloned, navigate to the project root directory:

```bash
# Linux/macOS
cd cosmic-core/
```
```ps1
# Windows
Set-Location cosmic-core\
```

---
# Understanding Configuration Setup

## Configuration

Run the setup script once after cloning (see [Setup & Execution](#setup--execution)). It creates two configuration files from the templates in `examples/`:

- `apis/cores/cosmic_cfg.env` — Ollama host, Qdrant URL, backend API URLs and CORS overrides.
- `src/cosmic_core.env` — embedding, retrieval and reranking settings.

Both are gitignored, so editing them never leaks secrets. A fresh clone still runs from the `examples/*.example.env` templates until those files exist.

Secrets (`OPENAI_API_KEY`, `hf_token`, `hf_token_finetune`) and local overrides belong in the gitignored `apis/cores/cosmic_cfg.env` (or the shell/container environment). Nothing is read from a root `.env`.

Precedence is: shell/container environment > `apis/cores/cosmic_cfg.env` + `src/cosmic_core.env` > `examples/*.example.env`.

Existing deployments upgrading from the previous layout should run `bins/migrate_memory_tiers.py` once with Qdrant available; it renames the memory directories and indexes and rewrites the legacy `global_memory` Qdrant payloads to `system`.


---
# Setup & Execution

> [!IMPORTANT]
> Before following either setup you have chosen from above instruction, ensure
> that our **backend & database** are available before **CoSMIC** since all API
> endpoints called within the codebase are coming from [cosmic-db](https://github.com/TheOpenSI/cosmic-db)
> repository. Refer to the [setup guide](https://github.com/TheOpenSI/cosmic-db/blob/main/README.md)
> from linked repository for more details.

> [!TIP]
> Docker provides an isolated environment where all services run in containers.
> This is recommended if you want to avoid installing Ollama or other dependencies
> directly on your machine.

## Docker Setup

> [!IMPORTANT]
> Running on a non-Linux machine or machine **without** an NVIDIA GPU? Open `compose.yaml` and comment out any traces of these lines before starting:
> ```yaml
> deploy:
>   resources:
>     reservations:
>       devices:
>         - driver: nvidia
>           count: all
>           capabilities: [gpu]
> ```

> [!IMPORTANT]
> Run `./bins/setup.sh` (Linux/macOS) or `.\bins\setup.ps1` (Windows) once
> before building. It initialises the git submodules, creates the runtime env
> files and downloads the latest Stockfish prebuilt binary to `bins/Stockfish/src/stockfish`.

Before you begin, ensure you have **Docker** and **Docker Compose** installed:

1. [**Docker**](https://docs.docker.com/get-docker/)
2. [**Docker Compose**](https://docs.docker.com/compose/install/)

### **1. Starting Docker Services**

> [!NOTE]
> It's possible to run **Docker** in rootless mode on Linux. However, the way to set
> it up is different on each Linux distro. Please refer to [this guide](https://docs.docker.com/engine/security/rootless/)
> from **Docker** for further details.

From project root directory, start all services using the Docker Compose file:

```bash
# Linux/MacOS
docker compose up --build -d # Refer to NOTE if running on rootless mode
```
```ps1
# Windows
docker compose up --build -d # Docker run through lightweight Linux VM on Windows so it's rootless by default
```

### **2. Monitoring First-Run Model Download**

> [!NOTE]
> On the first run, Ollama will automatically pull the default model pre-defined
> in our configuration, which is fetched from **Configurations API** endpoint in
> [cosmic-db](https://github.com/TheOpenSI/cosmic-db) repo. This may take a while depending on your connection speed.

```bash
# Linux/MacOS
docker compose logs -f ollama # Refer to above note if running on rootless mode
```
```ps1
# Windows
docker compose logs -f ollama # Rootless by default since Docker runs through a lightweight Linux VM
```

> [!IMPORTANT]
> **Please wait until you see `Model <SLM> is available on the server.` before sending requests to CoSMIC.**

### **3. Verifying Docker Services**

Once containers are running, verify everything is healthy:

1. **FastAPI**: [localhost:3000/docs](http://localhost:3000/docs)
2. **Ollama**: [localhost:11435](http://localhost:11435)

---

## Native Setup

> [!IMPORTANT]
> Run `./bins/setup.sh` (Linux/macOS) or `.\bins\setup.ps1` (Windows) once
> after cloning. It initialises the git submodules, creates the runtime env
> files and downloads the latest Stockfish prebuilt binary to `bins/Stockfish/src/stockfish`.

Before you begin, ensure that you have `Python (v3.14+)`, `Ollama` and `uv` running on your system:

```bash
# Linux/MacOS
python --version
ollama --version
uv --version
```
```ps1
# Windows
py --version
ollama --version
uv --version
```

### **1. Installing Dependencies**

From the project root directory, install all Python dependencies:

```bash
# Linux/macOS
uv sync --frozen --no-cache
```
```ps1
# Windows
uv sync --frozen --no-cache
```

### **2. Pull a Small Language Model (SLM) via Ollama**

> [!NOTE]
> On the first run, Ollama will automatically pull the default model pre-defined
> in our configuration, which is fetched from **Configurations API** endpoint in
> [cosmic-db](https://github.com/TheOpenSI/cosmic-db) repo. This may take a while depending on your connection speed.

If using a local model, pull it before starting the server:

```bash
# Linux/MacOS
ollama serve
ollama pull 'qwen2.5:7b'
```
```ps1
# Windows
ollama serve
ollama pull 'qwen2.5:7b'
```

### **3. Starting the Application**

After dependencies are installed, start the FastAPI development server:

```bash
# Linux/macOS
uv run fastapi dev
```
```ps1
# Windows
uv run fastapi dev
```

### **4. Verifying Native Setup**

1. **FastAPI**: [localhost:8000/docs](http://localhost:8000/docs)
2. **Ollama**:
- On `Windows/MacOS`, search for and open the **Ollama** application from your applications menu.
- On `Linux`, search for **ollama** or type `ollama` in your terminal to start the application.
---

# Framework

**CoSMIC** routes every incoming query through an [SLM-based Query Analyser](src/query_analyser/query_analyser.py) that selects the most relevant service. Its configuration is fetched dynamically from the **Configurations API** endpoint in [cosmic-db](https://github.com/TheOpenSI/cosmic-db) repository. This allows our product to stay in sync with the latest settings without requiring a restart or redeploy.

Currently, **CoSMIC** provides 5 core services, each discoverable via the **Services API** endpoint within the same repository:


| Service                 | Description                                                                     |
| ----------------------- | ------------------------------------------------------------------------------- |
| **Chess**               | Next-move prediction, position analysis, puzzle QA, and FEN generation          |
| **Code Generation**     | Python code generation via [PyCapsule](https://github.com/TheOpenSI/PyCapsule)  |
| **Memory (RAG)**        | Retrieval-augmented generation over a local vector database (e.g., Qdrant)      |
| **General QA**          | Open-ended question answering and reasoning                                     |
| **Academic Governance** | Document-grounded QA over academic governance materials                         |



---
# Conventions

This repository follows the shared [EditorConfig conventions](docs/conventions/EDITORCONFIG.md). See the [live documentation](https://github.com/TheOpenSI/cosmic-docs/blob/main/docs/06-development/editorconfig.md) for the rules and editor setup.

Python files use three import groups in this order, with empty groups kept for consistency:

```python
### Core modules ###


### Type hint modules ###


### Internal modules ###
```

---

# Reference

If this repository is useful for your work, please cite the paper below:

```bibtex
@misc{
    title         = {Unleashing Artificial Cognition: Integrating Multiple AI Systems},
    author        = {Muntasir Adnan and Buddhi Gamage and Zhiwei Xu and Damith Herath and Carlos C. N. Kuhn},
    howpublished  = {Australasian Conference on Information Systems},
    year          = {2024}
}
```

---

# Contact

For technical support, please contact:
1. [Carlos Kuhn](mailto:carlos.kuhn@canberra.edu.au)
2. [Muntasir Adnan](mailto:adnan.adnan@canberra.edu.au)

For project support, please contact:
1. [Carlos C. N. Kuhn](mailto:carlos.noschangkuhn@canberra.edu.au)

---

# Contributing

We welcome any contributions from the community no matter if you are researchers, developers, and enthusiasts:

- **Report issues**: found a bug or have a feature request? Open an issue on our [GitHub page](https://github.com/TheOpenSI/cosmic-core/issues)
- **Submit pull requests**: please follow our [contribution guidelines](./CONTRIBUTING)
- **Make a donation**: support the project [here](https://payments.canberra.edu.au/Misc/tran?tran-type=OPENSI)

---

# License

This code is distributed under [the MIT License](./LICENSE).

If models from:

1. **Hugging Face (Mistral 7B, Gemma 7B, etc.)** are used, please also follow [Hugging Face's licence terms](https://huggingface.co/docs/hub/repositories-licenses).
2. **OpenAI API models (GPT-3.5-Turbo, GPT-4o, etc.)** are used, please also follow [OpenAI's licence terms](https://github.com/openai/openai-openapi/blob/master/LICENSE).
3. **Meta (Llama 2, Llama 3, and Llama 4)** are used, please also follow licence terms specifically to each model (due to how **Meta** structuring their policy page). For quick redirection, please click on one of the following link for each model: [Llama 2](https://www.llama.com/llama2/use-policy/), [Llama 3](https://www.llama.com/llama3/use-policy/), and [Llama 4](https://www.llama.com/llama4/use-policy/).

---

# Funding

This project is funded under the agreement with:

1. **ACT Government for Future Jobs Fund with Open Source Institute** (OpenSI-R01553).
2. **NetApp Technology Alliance Agreement** (OpenSI-R01657).