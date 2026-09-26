# ReTool Companion Code

> **Code source:** The code in this directory is adapted and organized from the [05-retool](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/05-retool) implementation in [agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab), which is maintained by the author of this chapter. The current version is adapted for Chapter 8 of Happy-LLM and PyTRIO 0.2.6.

This directory corresponds to Section 8.4 of the main text. The code implements multi-turn code-interpreter rollouts on math problems, outcome rewards, the observation mask, PPO updates, and unified evaluation.

> Safety note: `sandbox.py` uses a separate subprocess, timeouts, and resource limits to contain accidental resource consumption, but it does not provide trustworthy security isolation. Model-generated code may still access local files, the network, and inherited environment variables. When handling untrusted code, use a disposable container, a low-privilege virtual machine, or a dedicated sandbox service, and remove all credentials.

## Setup

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r docs/chapter8/requirements.txt
trio login
```

If you want to log experiments online, also run `swanlab login`.

## Preparing the Data

```bash
python docs/chapter8/retool/prepare_data.py
```

## Minimal Trial Run

First confirm the executor's permissions in an isolated environment, then run:

```bash
python docs/chapter8/retool/train.py \
    --max-steps 1 \
    --questions-per-batch 1 \
    --group-size 4 \
    --max-code-calls 2 \
    --sandbox-workers 2 \
    --swanlab-mode disabled
```

## Files

| File | Purpose |
| --- | --- |
| `protocol.py` | Defines the `code_interpreter` tool and the message concatenation rules |
| `sandbox.py` | Executes Python code while limiting time, processes, and output size |
| `rollout.py` | Runs the multi-turn "generate - run code - observe - continue generating" state machine |
| `reward.py` | Extracts the last `\boxed{}` and checks mathematical equivalence |
| `train.py` | Builds the observation mask and PyTRIO Datum and performs PPO updates |
| `eval.py` | Evaluates text-only and ReTool modes in a unified way |
| `analysis.py` | Aggregates metrics across different checkpoints |
