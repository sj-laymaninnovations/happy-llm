# OPD Companion Code

> **Code source:** The code in this directory is adapted and organized from the [02-opd](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/02-opd) implementation in [agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab), which is maintained by the author of this chapter. The current version is adapted for Chapter 8 of Happy-LLM and PyTRIO 0.2.6.

This directory corresponds to Section 8.2 of the main text and implements complete On-Policy Distillation, using DeepMath-103K prompts as the example. The Student generates answers, the Teacher computes per-token logprobs on the same Student trajectory, and the training signal is then built from the reverse KL.

## Files

| File | Purpose | Role in the main text |
| --- | --- | --- |
| `01-demo-sync.py` | Runs Student rollout, Teacher scoring, and Student updates sequentially | Explained section by section |
| `02-demo-async.py` | Runs the rollouts and Teacher logprob requests within a batch concurrently | Full code provided; only the interface differences are explained |

## Setup

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r docs/chapter8/requirements.txt
trio login
```

If you want to log experiments online, also run `swanlab login`.

## Running the Synchronous Version

```bash
python docs/chapter8/opd/01-demo-sync.py \
    --steps 1 \
    --batch-size 1 \
    --group-size 1 \
    --max-tokens 512 \
    --sample-size 20 \
    --num-shards 1 \
    --swanlab-mode disabled
```

The script downloads the DeepMath-103K parquet files from ModelScope. `--num-shards 1` is suitable for a low-cost trial run, while `--num-shards 10` uses all shards.

## Running the Asynchronous Version

```bash
python docs/chapter8/opd/02-demo-async.py \
    --steps 1 \
    --batch-size 4 \
    --group-size 2 \
    --max-tokens 512 \
    --sample-size 20 \
    --num-shards 1 \
    --swanlab-mode disabled
```

The asynchronous version runs Student rollouts concurrently within a batch and submits Teacher logprob requests concurrently within the same prompt. Token alignment, the reverse KL, and the `importance_sampling` Datum remain unchanged.

## Suggested Reading Order for the Synchronous Code

1. `parse_args()` and `load_deepmath()`: prepare the training arguments and the prompt-only data.
2. `build_prompt()`: renders a question into the Student's input.
3. `completion_teacher_logprobs()`: has the Teacher score the Student's completion.
4. `build_opd_datum()`: writes the per-token reverse KL into the advantage.
5. `main()`: refreshes the Student sampler, updates the Student, logs metrics, and saves the weights.

The Teacher defaults to `Qwen/Qwen3.6-27B`. Which models are actually available depends on what the PyTRIO service returns; you can also specify a different Teacher with `--teacher-base-model` or `--teacher-model-path`. The token ids of the Teacher and Student must be compatible.
