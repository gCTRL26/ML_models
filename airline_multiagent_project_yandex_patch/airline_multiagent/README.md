# Airline Multi-Agent Assistant (LangGraph + LangChain)

A presentation-ready multi-agent airline assistant built from the patterns used in your notebooks:

- hierarchical supervisor-worker architecture
- ReAct/TAO execution loops
- short-term memory via `MemorySaver`
- long-term passenger profile memory
- layered guardrails
- Pydantic-validated tool arguments
- human approval via `interrupt()` for transactional actions
- selective critic for answer QA
- lightweight evaluation harness

## Architecture

```text
START
  -> hydrate_profile
  -> input_guard
  -> orchestrator
     -> plan with structured output
     -> delegate to specialist workers
     -> synthesize grounded answer
  -> critic (conditional)
  -> respond
END
```

## Where to run the project

Yes: `airline_multiagent/demo.py` is the executable entry point.

Now it works as an **interactive CLI chat**:

```bash
python -m airline_multiagent.demo
```

Then you can type sequential user requests just like in a notebook:

```text
You: Find me flights from Moscow to London on 2026-04-16
Assistant: ...

You: Book the cheapest option for Ivan Petrov
Assistant: ...

Pending approval:
{...}

You: /approve
Assistant: ...
```

Commands:

- `/approve` — approve pending transactional action
- `/reject some_reason` — reject pending action
- `/exit` — quit the CLI

## Specialist workers

- `flight_agent` — flight search and comparison
- `policy_agent` — baggage / refunds / rebooking / pets / cancellation rules
- `booking_agent` — booking lookup and transactional actions
- `profile_agent` — persistent passenger profile

## Project structure

```text
airline_multiagent/
  .env.example
  requirements.txt
  airline_multiagent/
    config.py
    llm_factory.py
    schemas.py
    guards.py
    tools.py
    workers.py
    orchestrator.py
    graph.py
    evals.py
    demo.py
    memory_store.py
    data/mock_data.py
```

## Setup

### 1) Create and activate venv

```bash
cd airline_multiagent
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Create your local `.env`

Copy `.env.example` to `.env` and fill it in.

Example:

```env
MODEL_PROVIDER=yandex
MODEL_NAME=your-model-name
MODEL_API_KEY=your-yandex-api-key
MODEL_BASE_URL=https://your-openai-compatible-endpoint
YANDEX_CLOUD_FOLDER=your-folder-id
MODEL_TEMPERATURE=0
APP_NAME=airline-multi-agent
PROFILE_DIR=./data
DEFAULT_THREAD_ID=demo-thread
```

### Are quotes needed in `.env`?

Usually **no**.

Use this:

```env
MODEL_API_KEY=abc123
```

Not this unless you really want quotes as part of parsing style:

```env
MODEL_API_KEY="abc123"
```

For normal API keys, URLs, IDs, and model names, keep values **without quotes**.

### 4) Run

```bash
python -m airline_multiagent.demo
```

## How config reads `.env`

`config.py` calls `load_dotenv()` once at import time and then builds the settings object:

```python
from dotenv import load_dotenv
load_dotenv()
```

Then values are available through:

```python
from airline_multiagent.config import SETTINGS

print(SETTINGS.model_name)
print(SETTINGS.model_api_key)
print(SETTINGS.model_base_url)
print(SETTINGS.yandex_cloud_folder)
```

## How the model is created

Use the factory from `llm_factory.py`:

```python
from airline_multiagent.llm_factory import build_chat_llm

llm = build_chat_llm()
```

This factory automatically reads:

- `MODEL_NAME`
- `MODEL_API_KEY`
- `MODEL_BASE_URL`
- `YANDEX_CLOUD_FOLDER`

and passes them into `ChatOpenAI(...)`.

## Why this design is good for interviews

1. **Supervisor-worker decomposition**
   The orchestrator plans and delegates, instead of one monolithic agent doing everything.

2. **Defense in depth**
   Input guard, tool-output sanitization, typed tool schemas, and human approval for side effects.

3. **Clear tool boundaries**
   Each worker gets only the tools it actually needs.

4. **Selective quality control**
   The critic only runs on risky or multi-domain tasks.

5. **Mixed memory strategy**
   `MemorySaver` for thread memory plus persistent passenger profile.

6. **Evaluation-first mindset**
   The project includes deterministic checks and reusable eval cases.

## What is intentionally simplified

- Mock airline DB in memory
- Simple keyword/HyDE-inspired policy retrieval instead of embeddings
- No auth / RBAC / tracing backend yet
- No async parallel worker execution yet

These are easy to extend later if you want a more production-like v2.
