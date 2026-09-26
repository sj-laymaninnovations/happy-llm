# GRPO Companion Code

> **Code source:** The code in this directory is adapted and organized from the [01-grpo](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/01-grpo) implementation in [agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab), which is maintained by the author of this chapter. The current version is adapted for Chapter 8 of Happy-LLM and PyTRIO 0.2.6.

This directory corresponds to Section 8.1 of the main text and implements a complete PyTRIO GRPO training pipeline, using GSM8K as the example. The synchronous and asynchronous versions use the same prompt, rule-based reward, group-relative advantage, Datum alignment, and loss.

## Files

| File | Purpose | Role in the main text |
| --- | --- | --- |
| `01-demo-sync.py` | Runs rollouts and training one prompt at a time | Explained section by section |
| `02-demo-async.py` | Uses `asyncio.gather()` to run the rollouts within a batch concurrently | Full code provided; only the interface differences are explained |

## Setup

From the root of the Happy-LLM repository, create a Python 3.13 environment and install the shared dependencies for Chapter 8:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r docs/chapter8/requirements.txt
trio login
```

If you want to log experiments online, also run `swanlab login`.

## Running the Synchronous Version

```bash
python docs/chapter8/grpo/01-demo-sync.py \
    --steps 1 \
    --batch-size 1 \
    --group-size 4 \
    --max-tokens 512 \
    --loss-fn importance_sampling \
    --swanlab-mode disabled
```

GSM8K is downloaded the first time training starts. For real experiments, increase `steps`, `batch-size`, and `group-size`, and keep the rest of the configuration fixed before comparing `importance_sampling` with `ppo`.

## Running the Asynchronous Version

```bash
python docs/chapter8/grpo/02-demo-async.py \
    --steps 1 \
    --batch-size 4 \
    --group-size 4 \
    --max-tokens 512 \
    --loss-fn importance_sampling \
    --swanlab-mode disabled
```

The asynchronous version processes the rollouts for different prompts in the same batch concurrently. The reward, advantage, Datum, and loss are identical to those in the synchronous version.

## Suggested Reading Order for the Synchronous Code

1. `parse_args()` and `RolloutSample`: define the training configuration and the data for a single rollout.
2. `grade_answer()`: extracts `\boxed{}` and computes the rule-based reward.
3. `run_rollout_group()`: samples a group of answers for the same question and computes the relative advantages.
4. `build_grpo_datum()`: performs the autoregressive right shift and the prompt mask.
5. `main()`: selects the built-in loss and ties together sampler refresh, rollout, policy update, logging, and weight saving.

In the corresponding functions, the asynchronous version uses `sample_async()`, `forward_backward_async()`, `optim_step_async()`, and `asyncio.gather()`.
