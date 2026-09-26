# Search-R1 Companion Code

> **Code source:** The code in this directory is adapted and organized from the [03-search-r1](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/03-search-r1) implementation in [agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab), which is maintained by the author of this chapter. The current version is adapted for Chapter 8 of Happy-LLM and PyTRIO 0.2.6.

This directory corresponds to Section 8.3 of the main text. The code keeps clear boundaries between data, the tool protocol, the search environment, multi-turn rollout, reward, training, and evaluation, making it easy to see how Agentic RL adds environment interaction on top of an ordinary GRPO training loop.

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
python docs/chapter8/search-r1/prepare_data.py
```

This command prepares the NQ and HotpotQA training, development, and test data. The Wikipedia backend is free and requires no API key, so it is a good choice for validating the pipeline first.

## Minimal Trial Run

```bash
python docs/chapter8/search-r1/train.py \
    --max-steps 1 \
    --questions-per-batch 1 \
    --group-size 4 \
    --max-search-calls 2 \
    --search-backend wikipedia \
    --search-concurrency 3 \
    --swanlab-mode disabled
```

Online search results change over time. When comparing the Base Model with a checkpoint, you must keep the question set, search backend, maximum number of searches, and sampling parameters the same.

## Files

| File | Purpose |
| --- | --- |
| `protocol.py` | Defines the tool protocol and parses search actions and final answers |
| `search.py` | Wraps the DeepSeek Search, Wikipedia, and Zhihu search backends |
| `rollout.py` | Runs the multi-turn "generate - search - observe - continue generating" state machine |
| `reward.py` | Computes the format reward and the exact-match answer reward |
| `train.py` | Builds the observation mask and PyTRIO Datum and updates the policy |
| `eval.py` | Evaluates the Base Model or a checkpoint in the same environment |
| `analyse.py` | Aggregates evaluation results |
