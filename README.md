# Recursive Tool-Calling Agent (LLM + SLM) with KB Logging

End-to-end reference implementation:
- **LLM (OpenAI Responses API)**: decomposition / verification / synthesis, with native tool-calling
- **SLM (HF local: Qwen/Qwen3-4B-Instruct-2507)**: solves subtasks, calls tools via strict JSON actions
- **Tools**: SerpApi (HTTP), Jina Reader (HTTP), safe math calc, echo
- **Recursive decomposition** when unsolved
- **Solvability metrics** + **KB(JSONL)** logging for root and all subtasks

## Install

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

> Notes:
> - You need a working `torch` install compatible with your platform/GPU.
> - For GPU, install the correct CUDA wheel per PyTorch instructions.

## Environment Variables

```bash
export OPENAI_API_KEY="..."
export SERPAPI_API_KEY="..."
export JINA_API_KEY="..."   # optional
```

## Run

```bash
python -m agent_system.main
```

Outputs:
- terminal prints a JSON result (status, final_answer, root_max_derived_depth, etc.)
- logs written to `kb.jsonl` in the project root
