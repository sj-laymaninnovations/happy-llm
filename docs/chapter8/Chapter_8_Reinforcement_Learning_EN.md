# Chapter 8: Reinforcement Learning for LLMs

In Chapter 4, we started from Reinforcement Learning from Human Feedback (RLHF) and introduced reward models, Proximal Policy Optimization (PPO), and the basic workflow of instruction alignment. Chapter 6 then laid out the engineering foundations of large language model (LLM) training, and Chapter 7 gave the model the ability to plan, remember, and call tools.

Once an Agent actually operates in environments such as search or code execution, supervised fine-tuning alone can hardly cover every possible interaction trajectory. The model needs to try actions on its own, judge from the final outcome whether a trajectory worked, and then fold the successful experience back into its policy. In this way, reinforcement learning for LLMs has gradually moved from "making answers better match preferences" toward "teaching the model to act within an environment."

This chapter is organized around four closely connected topics. We first introduce Group Relative Policy Optimization (GRPO), which is widely used in reinforcement learning for LLMs today, to build a complete picture of rollouts, rewards, advantages, and policy updates. Next, we introduce On-Policy Distillation (OPD) and see how a Student continuously receives token-by-token guidance from a Teacher on its own state distribution. Finally, using Search-R1 and ReTool as examples, we extend the same training framework to a search engine and a code interpreter, completing two runnable Agentic RL practice projects.

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/8-images/8-1-chapter-roadmap.png" alt="Structure of Chapter 8" width="90%">
    <p>Figure 8.1 Structure of Chapter 8</p>
</div>

All examples in this chapter use PyTRIO 0.2.6. PyTRIO connects local data processing, environment interaction, and training control with remote model sampling, forward and backward passes, and weight management. Although the four examples tackle different tasks, their core pipeline can always be summarized as follows:

1. Use the current policy to generate one trajectory or a group of trajectories;
2. Compute the training signal from the outcome or from the Teacher's feedback;
3. Align the prompt, response, old-policy logprobs, and advantages into a `Datum`;
4. Call `forward_backward()` and `optim_step()` to update the policy;
5. Refresh the sampling client so that the next round of rollouts uses the new weights.

The accompanying code is in `docs/chapter8`. Install the shared dependencies from the repository root:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r docs/chapter8/requirements.txt
trio login
```

The models supported by the PyTRIO service are updated continually. Before running a real experiment, you can check the current list with the following code:

```python
import pytrio as trio

service_client = trio.ServiceClient()
print(service_client.get_supported_models())
```

> **Further reading:** If you would like to explore more Agentic RL algorithms and runnable practice projects, see [agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab), another open-source project maintained by the author. Beyond what this chapter covers, that repository also contains conceptual breakdowns, training code, and experiment logs for algorithms and environments such as OPSD, DAPO, GSPO, and ALFWorld.

> Note: The author chose PyTrio rather than Verl for this chapter's examples in order to lower the engineering barrier and experimental cost of Agentic RL. With the minimal trial-run configurations given in the text, the author aims to keep the baseline experiment budget for all four examples under 200 RMB, so that readers can run the full loop of rollout, reward, advantage, and policy update themselves. Actual costs will vary with the model, sampling length, number of training steps, and evaluation scale.

## 8.1 GRPO

> **Code source:** This section is based on `01-demo-sync.py` and `02-demo-async.py` in the author's open-source repository [agentic-rl-lab/01-grpo](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/01-grpo). The main text walks through the synchronous version, `01-demo-sync.py`, piece by piece; the asynchronous version is kept as a complete companion implementation.

GRPO was first systematically proposed in DeepSeekMath. It keeps PPO's idea of on-policy updates while estimating relative advantages from a group of sampled outputs for the same question, which removes the need to train a separate Value Model. For math, code, and logic tasks with verifiable answers, this approach can use outcome rewards directly, and it has become one of the most common foundational algorithms in reinforcement learning for LLMs.

### 8.1.1 From Language Models to Reinforcement Learning Policies

An autoregressive language model takes a prompt $x$ and generates a token sequence $y=(y_1,y_2,\ldots,y_T)$ one token at a time. From a probabilistic modeling perspective, the probability of the complete answer can be written as:

$$\pi_\theta(y\mid x) = \prod_{t=1}^{T} \pi_\theta(y_t\mid x,y_{\lt t}),$$

where $\pi_\theta$ denotes the language model policy with parameters $\theta$. Placing the language model in a reinforcement learning framework gives the correspondence shown in Table 8.1.

| Reinforcement learning concept | Meaning in a large language model |
| --- | --- |
| State $s_t$ | The prompt plus the tokens generated so far |
| Action $a_t$ | The next token |
| Policy $\pi_\theta(a_t\mid s_t)$ | The model's probability distribution over the next token |
| Trajectory $\tau$ | A complete answer, or a multi-turn interaction that includes tool calls |
| Environment | A dataset grader, search engine, code interpreter, etc. |
| Reward $R(\tau)$ | Feedback such as answer correctness, formatting, or tool execution results |

A single model generation is usually called a rollout. The goal of training is to raise the probability of actions that appear in high-reward trajectories and lower the probability of actions that appear in low-reward trajectories. The most intuitive form of the policy gradient is:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta} \left[ A(\tau) \nabla_\theta \log \pi_\theta(\tau) \right],$$

where $A(\tau)$ is the advantage function. A positive advantage means the trajectory is relatively worth encouraging; a negative advantage means the current policy should reduce similar behavior.

The action space of an LLM is the entire vocabulary, and a single answer may contain hundreds to thousands of tokens. During training we also need to control how much the new policy differs from the old one, so that a single update does not destroy abilities the model already has. As a result, the genuinely hard part usually comes down to two questions: how to obtain stable advantages, and how to use those advantages to update the policy safely.

### 8.1.2 From PPO to GRPO

PPO typically uses a Value Model to estimate the state value $V_\phi(s_t)$, and then combines it with returns to compute the advantage. In the LLM setting, the Value Model is often about the same size as the policy model, which brings extra GPU memory, training, and synchronization costs. Errors in value estimation also propagate further into the policy update.

GRPO switches to relative comparison within a group. For a question $x$, it first samples $G$ answers from the old policy $\pi_{\theta_{\mathrm{old}}}$:

$$\{y_1,y_2,\ldots,y_G\} \sim \pi_{\theta_{\mathrm{old}}}(\cdot\mid x).$$

The grader assigns rewards $r_1,r_2,\ldots,r_G$ to each of them. Classic GRPO normalizes the rewards using the within-group mean and standard deviation:

$$A_i = \frac{r_i-\mathrm{mean}(r_1,\ldots,r_G)}{\mathrm{std}(r_1,\ldots,r_G)+\varepsilon}.$$

In this way, high-scoring answers to the same question receive positive advantages and low-scoring answers receive negative advantages. The difficulty of the question itself is canceled out by the within-group baseline, so the policy can focus on learning "under the same conditions, which ways of generating are better."

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/8-images/8-2-grpo.png" alt="GRPO group sampling and policy update" width="90%">
    <p>Figure 8.2 GRPO group sampling and policy update</p>
</div>

For example, suppose we sample 4 answers to the same math problem and the rule-based grader gives rewards $[1,0,0,1]$. The group mean is $0.5$, and with mean-centering only, the 4 advantages are:

$$[0.5,-0.5,-0.5,0.5].$$

The two correct answers are encouraged and the two wrong answers are suppressed. To show the within-group baseline directly, the code in this chapter uses by default:

$$A_i=r_i-\bar r.$$

A full experiment can also add standard-deviation normalization. The two formulations express the same idea of within-group relative comparison, but they produce different gradient scales, so when comparing experiments you should keep the advantage computation fixed.

There is also an easily overlooked degenerate case. When every answer in a group is correct, or every answer is wrong, each $A_i$ equals 0, and that group produces no useful policy gradient. If the training logs show a persistently high fraction of degenerate groups, you usually need to adjust the problem difficulty, sampling temperature, base model capability, or group size.

### 8.1.3 Rewards and Verifiable Reinforcement Learning

GRPO is only responsible for turning within-group rewards into relative advantages; the quality of the rewards still determines what the model ultimately learns. Math and code tasks can usually use deterministic rule-based graders, and this kind of training is also called Reinforcement Learning with Verifiable Rewards (RLVR).

This chapter uses GSM8K as an example and requires the model to put its final numerical answer inside `\boxed{}`. The minimal grading logic is as follows:

```python
def extract_boxed(text: str) -> str | None:
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if not matches:
        return None
    return matches[-1].strip()


def normalize_answer(text: str) -> str:
    return text.replace(",", "").strip().rstrip(".")


def grade_answer(response: str, ground_truth: str) -> float:
    answer = extract_boxed(response)
    if answer is None:
        return 0.0
    return 1.0 if normalize_answer(answer) == normalize_answer(ground_truth) else 0.0
```

Rule-based rewards are simple, cheap, and reproducible. However, once a rule has a loophole, the model may learn to exploit it to get high scores. When designing a reward, check at least the following:

- **Is answer extraction robust?** The same correct answer may appear as an integer, a decimal, a fraction, or a number with commas;
- **Does the format reward overpower correctness?** Formatting should only be an auxiliary constraint; a well-formatted wrong answer must not receive most of the reward;
- **Are the reference answers reliable?** Wrong labels give the model training signals that point in the opposite direction;
- **Does the grader leak information?** The model should not be able to read the reference answer directly through the prompt or tool outputs;
- **Do the rewards discriminate?** A large number of all-0 or all-1 groups will deprive GRPO of its relative comparison signal.

Before formal training, you can save rollouts from the base model and manually inspect, offline, a batch of "response–extracted answer–reward" triples. This step often uncovers problems earlier than tuning the learning rate directly.

### 8.1.4 Policy Ratio and Update Loss

Rollouts are generated by the old policy, while parameter updates happen on the current policy. For the $t$-th token of the $i$-th answer, define the probability ratio between the new and old policies:

$$\rho_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t}\mid x,y_{i,\lt t})}{\pi_{\theta_{\mathrm{old}}}(y_{i,t}\mid x,y_{i,\lt t})} = \exp\left(\log\pi_\theta-\log\pi_{\theta_{\mathrm{old}}}\right).$$

The most direct importance sampling objective can be written as:

$$J_{\mathrm{IS}}(\theta) = \mathbb{E}_{i,t} \left[\rho_{i,t}(\theta)A_i\right].$$

If the probability ratio drifts too far from 1, a handful of tokens can dominate the gradient. PPO uses a clipped objective to limit the size of a single update:

$$J_{\mathrm{PPO}}(\theta) = \mathbb{E}_{i,t} \left[\min\left(\rho_{i,t}A_i, \mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i\right)\right].$$

The classic GRPO objective in DeepSeekMath adopts PPO-style clipping and adds a KL constraint against a reference policy. The synchronous version in this chapter, `01-demo-sync.py`, uses PyTRIO's built-in `importance_sampling` by default, which makes it easy to observe the data relationships among `old_logprobs`, `advantages`, and the current policy; changing the command-line argument to `--loss-fn ppo` switches to the built-in PPO loss.

GRPO and PPO should therefore be understood at different levels:

- GRPO describes how to sample the same question in groups and derive advantages from the within-group rewards;
- importance sampling or PPO describes how to turn those advantages into a policy gradient.

### 8.1.5 Implementing Synchronous GRPO from Scratch

Following the execution order of the synchronous script, this section breaks down a complete GRPO training program step by step. The companion directory provides two files that can each be run independently:

| File | Execution mode | Purpose |
| --- | --- | --- |
| `01-demo-sync.py` | Synchronous | The main teaching thread, explained piece by piece in the text |
| `02-demo-async.py` | Asynchronous | Processes rollouts for different prompts in the same batch concurrently |

Both files use the same data, reward, advantage, Datum, and loss. Below we only walk through the synchronous version, so that readers can first understand one complete update by following a linear control flow.

**(1) Define the training configuration and rollout data**

The script keeps only the two policy-update losses provided directly by PyTRIO, `importance_sampling` and `ppo`:

```python
LOSS_FNS = ("importance_sampling", "ppo")
```

`GRPOConfig` centralizes the model, sampling, optimizer, and SwanLab parameters. `RolloutSample` stores everything about a single completion that later training steps will need:

```python
@dataclass
class RolloutSample:
    """One sampled output, plus the old-policy logprobs needed to build importance_sampling."""

    tokens: list[int]
    logprobs: list[float]
    text: str
    reward: float
    advantage: float
```

The `logprobs` here must come from the Student at rollout time. Logprobs recomputed after a policy update belong to the new policy and cannot stand in for the old-policy probabilities. `parse_args()` converts command-line arguments into a `GRPOConfig` and uses `choices=LOSS_FNS` to restrict the choice to these two built-in losses.

**(2) Load GSM8K and build the prompt**

The reference answer in GSM8K comes after `####`. The model's output, on the other hand, is required to put the final number inside `\boxed{}`. The code applies the same light normalization to both sides:

```python
def extract_boxed(text: str) -> str | None:
    """Take the last \\boxed{...} as the model's final answer."""
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if not matches:
        return None
    return matches[-1].strip()


def normalize_answer(text: str) -> str:
    """Apply only light normalization to GSM8K answers so that 1,000 and 1000 are not judged different."""
    return text.replace(",", "").strip().rstrip(".")


def grade_answer(response: str, ground_truth: str) -> float:
    """Return 1 if the boxed answer exactly matches the reference answer, otherwise 0."""
    answer = extract_boxed(response)
    if answer is None:
        return 0.0
    return 1.0 if normalize_answer(answer) == normalize_answer(ground_truth) else 0.0


def extract_gsm8k_answer(answer_text: str) -> str:
    """The final GSM8K answer comes after `####`."""
    match = re.search(r"####\s*(.+)", answer_text)
    if match is None:
        raise ValueError(f"No GSM8K final answer found: {answer_text!r}")
    return normalize_answer(match.group(1))
```

`build_prompt()` passes a few-shot example together with the current question through the model's own chat template, and then encodes the result into tokens:

```python
def build_prompt(tokenizer: Any, question: str) -> list[int]:
    messages = [
        *FEWSHOT_PREFIX,
        {"role": "user", "content": question + QUESTION_SUFFIX},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not prompt_tokens:
        raise ValueError("Prompt tokens are empty")
    return prompt_tokens
```

The training set is loaded with `load_dataset("openai/gsm8k", "main", split="train")`. During short trial runs, `pick_batch()` allows the data to wrap around; with `--all-data` turned on, each training example is used at most once.

**(3) Sample a group of answers for the same question**

`run_rollout_group()` is the heart of the GRPO rollout. A single `sample()` call takes one prompt and, via `num_samples=group_size`, returns a group of completions for that same question:

```python
def run_rollout_group(
    sampling_client: Any,
    tokenizer: Any,
    prompt_tokens: list[int],
    ground_truth: str,
    sampling_params: trio.SamplingParams,
    group_size: int,
) -> list[RolloutSample]:
    result = sampling_client.sample(
        prompt=trio.ModelInput.from_ints(prompt_tokens),
        num_samples=group_size,
        sampling_params=sampling_params,
        return_text=True,
    ).result()

    rewards: list[float] = []
    raw_samples: list[tuple[list[int], list[float], str]] = []

    for sequence in result.sequences:
        text = sequence.text
        if text is None:
            text = tokenizer.decode(sequence.tokens, skip_special_tokens=True)

        tokens = list(sequence.tokens)
        logprobs = [float(value) for value in sequence.logprobs]
        if len(tokens) != len(logprobs):
            raise ValueError(
                f"Generated token/logprob length mismatch: "
                f"{len(tokens)} != {len(logprobs)}"
            )

        reward = grade_answer(text, ground_truth)
        rewards.append(reward)
        raw_samples.append((tokens, logprobs, text))

    mean_reward = sum(rewards) / len(rewards)
    return [
        RolloutSample(
            tokens=tokens,
            logprobs=logprobs,
            text=text,
            reward=reward,
            advantage=reward - mean_reward,
        )
        for (tokens, logprobs, text), reward
        in zip(raw_samples, rewards, strict=True)
    ]
```

This code does four things in order:

1. Saves the completion tokens;
2. Saves the old policy's logprob for each completion token at sampling time;
3. Uses the rule-based grader to obtain a reward of 0 or 1;
4. Computes `advantage = reward - mean_reward`.

When all answers in a group are correct or all are wrong, every advantage is 0. The main loop records such a group as degenerate and skips it, to avoid submitting Datums that carry no training signal.

**(4) Build a right-shifted, aligned PyTRIO Datum**

An autoregressive model uses the token at the current position to predict the next token. Let the prompt length be $m$ and the completion length be $T$; the four arrays built by the script all have length $m+T-1$:

```python
def build_grpo_datum(
    prompt_tokens: list[int],
    sample: RolloutSample,
) -> trio.Datum:
    if not sample.tokens:
        raise ValueError("Cannot train on an empty completion")

    observation_len = len(prompt_tokens) - 1
    input_tokens = prompt_tokens + sample.tokens[:-1]
    target_tokens = [0] * observation_len + sample.tokens
    padded_logprobs = [0.0] * observation_len + sample.logprobs
    padded_advantages = (
        [0.0] * observation_len
        + [sample.advantage] * len(sample.tokens)
    )

    if not (
        len(input_tokens)
        == len(target_tokens)
        == len(padded_logprobs)
        == len(padded_advantages)
    ):
        raise ValueError("GRPO datum fields must have the same token length")

    return trio.Datum(
        model_input=trio.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_tokens, dtype=np.int64),
            "logprobs": np.asarray(padded_logprobs, dtype=np.float32),
            "advantages": np.asarray(padded_advantages, dtype=np.float32),
        },
    )
```

The prompt region only provides context, so `target_tokens`, `logprobs`, and `advantages` are filled with 0 as placeholders there. The completion region holds the real target tokens, the old-policy logprobs, and the within-group advantage shared by the entire answer.

Take 3 prompt tokens and 2 completion tokens as an example:

```text
prompt       = [x1, x2, x3]
completion   = [y1, y2]

model_input  = [x1, x2, x3, y1]
target       = [ 0,  0, y1, y2]
old_logprob  = [ 0,  0, l1, l2]
advantage    = [ 0,  0,  A,  A]
```

Prediction of `y1` starts at the position of the last prompt token, `x3`, which is why the prompt mask has length `len(prompt_tokens) - 1`.

**(5) Choose importance sampling or PPO**

`importance_sampling` and `ppo` reuse the same batch of GRPO Datums; you only need to switch `loss_fn`. Both losses are implemented natively in PyTRIO:

```python
fwd_bwd_future = training_client.forward_backward(
    datums,
    loss_fn=config.loss_fn,
)
```

With `importance_sampling`, training directly multiplies the new-to-old policy probability ratio by the advantage. With `ppo`, PyTRIO applies PPO clipping to the same data. GRPO's rollout, reward, and within-group advantage construction remain unchanged.

**(6) Put together one round of synchronous training**

`main()` first creates the training client, the sampling parameters, and the optimizer:

```python
service_client = trio.ServiceClient()
training_client = service_client.create_lora_training_client(
    base_model=config.base_model,
    rank=config.lora_rank,
)
tokenizer = training_client.get_tokenizer()

sampling_params = trio.SamplingParams(
    max_tokens=config.max_tokens,
    temperature=config.temperature,
    top_p=config.top_p,
    stop=get_stop_sequences(tokenizer),
)
adam_params = trio.AdamParams(
    learning_rate=config.learning_rate,
    beta1=config.beta1,
    beta2=config.beta2,
)
```

At every step, it first creates a new sampler from the current training weights and then performs rollouts question by question:

```python
for step in range(effective_steps):
    batch_rows = pick_batch(
        train_data,
        step,
        config.batch_size,
        config.all_data,
    )
    sampling_client = (
        training_client.save_weights_and_get_sampling_client()
    )

    datums: list[trio.Datum] = []
    prompt_mean_rewards: list[float] = []
    rollout_lengths: list[int] = []
    n_degenerate = 0

    for row in tqdm(
        batch_rows,
        desc=f"GRPO step {step}",
        unit="prompt",
    ):
        prompt_tokens = build_prompt(tokenizer, row["question"])
        ground_truth = extract_gsm8k_answer(row["answer"])
        rollout_samples = run_rollout_group(
            sampling_client=sampling_client,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            ground_truth=ground_truth,
            sampling_params=sampling_params,
            group_size=config.group_size,
        )

        rewards = [sample.reward for sample in rollout_samples]
        prompt_mean_rewards.append(sum(rewards) / len(rewards))
        rollout_lengths.extend(
            len(sample.tokens) for sample in rollout_samples
        )

        if all(
            sample.advantage == 0.0
            for sample in rollout_samples
        ):
            n_degenerate += 1
            continue

        for sample in rollout_samples:
            datums.append(
                build_grpo_datum(prompt_tokens, sample)
            )
```

Where the sampler is refreshed reflects the on-policy constraint: after the update at step $k$ finishes, step $k+1$ regenerates trajectories with the new weights. After submitting the forward-backward and optimizer tasks, the synchronous version explicitly waits for their results:

```python
optim_future = training_client.optim_step(adam_params)
fwd_bwd_result = fwd_bwd_future.result()
optim_future.result()
loss_metrics = dict(fwd_bwd_result.metrics)
```

Finally, the script computes the batch mean reward, the fraction of degenerate groups, the average generation length, the number of effective training tokens, and `loss_mean`, and it saves LoRA weights that can be used for sampling:

```python
final_weights = training_client.save_weights_for_sampler(
    name=run_name
).result()
print(
    f"Saved weights name: {run_name}, "
    f"path: {final_weights.path}"
)
```

### 8.1.6 Running the Synchronous and Asynchronous Versions

First log in to TRIO, then run a one-step trial of the synchronous version from the repository root:

```bash
trio login

python docs/chapter8/grpo/01-demo-sync.py \
    --steps 1 \
    --batch-size 1 \
    --group-size 4 \
    --max-tokens 512 \
    --loss-fn importance_sampling \
    --swanlab-mode disabled
```

The asynchronous version uses exactly the same training parameters:

```bash
python docs/chapter8/grpo/02-demo-async.py \
    --steps 1 \
    --batch-size 4 \
    --group-size 4 \
    --max-tokens 512 \
    --loss-fn importance_sampling \
    --swanlab-mode disabled
```

The asynchronous version does not change the algorithm's data; what changes is how it executes:

| Training stage | Synchronous version | Asynchronous version |
| --- | --- | --- |
| Rollout for a single question | `sample(...).result()` | `await sample_async(...)` |
| Multiple questions in a batch | Processed one after another | Processed concurrently with `asyncio.gather()` |
| Refreshing the sampler | Returns synchronously | `await save_weights_and_get_sampling_client_async()` |
| Forward-backward | `forward_backward()` followed by `.result()` | `forward_backward_async()`, then `await` the returned future |
| Optimizer | `optim_step()` followed by `.result()` | `optim_step_async()`, then `await` the returned future |
| Program entry point | `main(config)` | `asyncio.run(main(config))` |

The main text uses the synchronous version to build intuition for the algorithm. The asynchronous version suits real experiments where a batch contains multiple independent prompts, since it reduces serial waiting time by issuing remote requests concurrently.

During training, focus first on the metrics the code actually logs:

| Metric | Meaning | Warning signs |
| --- | --- | --- |
| `reward` | Mean over prompts in the batch of each prompt's average accuracy | If it stays flat for a long time, check the reward, data difficulty, and learning rate |
| `frac_degenerate` | Fraction of groups that are all correct or all wrong | When too high, there are not enough useful relative advantages |
| `rollout/avg_gen_len` | Average number of tokens per completion | When close to `max_tokens`, check for truncation |
| `train_tokens` | Number of completion tokens with a nonzero advantage | When 0, the current step has no training signal |
| `loss_mean` | The main loss of the current policy update | When it fluctuates wildly, check the learning rate and the probability ratio |

With this, we have the first training thread of this chapter: the current policy samples in groups, a rule-based environment gives outcome rewards, within-group comparison produces advantages, and PyTRIO performs the policy update. The OPD method that comes next keeps the same on-policy data flow, but replaces the training signal with the Teacher's feedback on every token.

## 8.2 On-Policy Distillation

> **Code source:** This section is based on `01-demo-sync.py` and `02-demo-async.py` from the author's open-source repository [agentic-rl-lab/02-opd/general-opd](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/02-opd/general-opd). The main text walks through the synchronous version, `01-demo-sync.py`, piece by piece; the asynchronous version is kept as a complete companion implementation.

Capability distillation for large language models (LLMs) usually involves a stronger Teacher and a smaller Student. Traditional knowledge distillation uses data collected in advance and has the Student fit the Teacher's probability distribution on these fixed samples. Because the Student keeps changing during training, a gap can open up between the fixed data and the states the Student actually visits.

On-Policy Distillation (OPD) has the Student generate answers with its current policy, and then has the Teacher score, token by token, the trajectory the Student has already taken. This way, every round of supervision lies on the Student's current state distribution. The Teacher does not need to generate its own set of answers, nor does it take part in backpropagation; it only provides dense probability feedback on the Student's actions.

OPD has become an important direction in LLM training in recent years. It can be used for transferring reasoning ability, recovering capabilities and continual learning, and it also fits naturally into the data flow of reinforcement learning (RL): the Student rollout corresponds to on-policy trajectories, the Teacher logprobs correspond to a per-token training signal, and after the Student is updated it generates the next batch of trajectories.

### 8.2.1 From Offline Distillation to On-Policy Distillation

Let the Teacher be $\pi_T$ and the Student be $\pi_\theta$. Offline distillation first fixes a dataset $\mathcal D$ and then minimizes the distributional difference on the states in it:

$$\mathcal L_{\mathrm{offline}} = \mathbb E_{s\sim\mathcal D} \left[D\bigl(\pi_T(\cdot\mid s),\pi_\theta(\cdot\mid s)\bigr)\right].$$

This approach is simple to implement, and how well it works depends on whether the dataset covers the states the Student encounters at inference time. Suppose the fixed data consists mostly of high-quality Teacher trajectories, while the Student, during real generation, makes a wrong token early on. The states it then enters may never have appeared in the distillation data, so the Teacher's guidance cannot cover this out-of-distribution trajectory.

OPD switches state sampling to the current Student:

$$y\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid x),$$

and then has the Teacher compute the conditional probability of that same Student completion:

$$\log\pi_T(y_t\mid x,y_{\lt t}).$$

One training iteration can be divided into three steps:

1. **Student rollout**: the Student generates answers with its current weights;
2. **Teacher feedback**: the Teacher computes logprobs on exactly the same prompt and completion;
3. **Student update**: a per-token signal is built from the probability difference between the two, and the Student is updated.

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/8-images/8-3-opd.png" alt="OPD obtains Teacher feedback on the Student's own trajectories" width="90%">
    <p>Figure 8.3 OPD obtains Teacher feedback on the Student's own trajectories</p>
</div>

In Figure 8.3, the Teacher performs only forward computation. The Student's sampled actions also do not take part in backpropagation; gradients flow from the constructed training objective back to the current Student parameters. This stop-gradient boundary lets OPD directly reuse existing rollout and policy-optimization infrastructure.

### 8.2.2 Reverse KL and the Per-Token Training Signal

OPD can use different distributional divergences. This chapter uses reverse KL:

$$D_{\mathrm{KL}}\left(\pi_\theta\,\|\,\pi_T\right) = \mathbb E_{y\sim\pi_\theta} \left[\log\pi_\theta(y\mid x)-\log\pi_T(y\mid x)\right].$$

For the $t$-th token the Student actually sampled, we obtain a Monte Carlo estimate:

$$d_t = \log\pi_{\theta_{\mathrm{old}}}(y_t\mid x,y_{\lt t})-\log\pi_T(y_t\mid x,y_{\lt t}).$$

When $d_t>0$, the Student prefers this token more than the Teacher does; when $d_t<0$, the Teacher assigns it higher probability. To minimize reverse KL, this chapter writes the training advantage as:

$$A_t=-\beta d_t,$$

where $\beta$ controls the distillation strength. In this way, every completion token receives its own training signal, and the signal density is far higher than a single 0/1 reward given only at the end of the answer.

Reverse KL is strongly mode-seeking. The Student will first concentrate on regions to which the Teacher already assigns high probability, which makes it well suited to compressing the Teacher's high-confidence behavior into a small model. At the same time, the capability gap between Teacher and Student, their reasoning styles and tokenizer compatibility all affect distillation quality. A larger Teacher is only a candidate; the Teacher also needs to provide valuable, learnable new signal on the Student's current trajectories.

Table 8.2 compares the data sources of GRPO and this chapter's OPD example.

| Item | GRPO | OPD |
| --- | --- | --- |
| Trajectory source | Current policy | Current Student |
| Feedback source | Environment or rule-based grader | Teacher probability distribution |
| Signal granularity | Usually a whole-trajectory reward | Per completion token |
| Needs reference answers? | Usually needs verifiable results | Prompt-only data is enough |
| Needs an extra model? | Can drop the Value Model | Needs Teacher forward computation |
| Main risks | Reward hacking, degenerate groups | Incompatible Teacher, unsuitable capability gap |

### 8.2.3 The Teacher Must Evaluate the Same Student Trajectory

The key data constraint of OPD is that the Teacher and Student must evaluate the same target tokens under the same prefix. If the Teacher generates its own answer and the Student then learns from that answer, you fall back to the common synthetic-data distillation pipeline.

In PyTRIO, you can first concatenate the prompt and the Student completion, and then call the Teacher's `compute_logprobs()`:

```python
def completion_teacher_logprobs(
    teacher_client,
    prompt_ids: list[int],
    completion_ids: list[int],
):
    all_ids = prompt_ids + completion_ids
    all_logprobs = teacher_client.compute_logprobs(
        trio.ModelInput.from_ints(all_ids)
    ).result()

    start = len(prompt_ids)
    end = start + len(completion_ids)
    return [float(x) for x in all_logprobs[start:end]]
```

Different SDKs may define logprob positions with an off-by-one shift, so you cannot guess the slice from array length alone. The safest approach is to construct a very short known sequence, confirm whether each position in the returned values means "the logprob of the current token" or "the logprob of the next token", and then run a boundary test between the prompt and the completion. The code in this chapter has already been aligned to the return convention of PyTRIO 0.2.6.

The Teacher and Student also need to share a compatible tokenizer. If the same token ids represent different strings in the two models, the probability difference has no correct meaning even when the arrays have the same length. Before real training, you should verify:

```python
text = "A short tokenizer alignment test."
student_ids = student_tokenizer.encode(text, add_special_tokens=False)
teacher_ids = teacher_tokenizer.encode(text, add_special_tokens=False)

assert student_ids == teacher_ids
```

For models from the same family but with different chat templates, the Student tokenizer should render the full prompt once, and the Teacher should then evaluate the corresponding token sequence directly. Do not apply two separate chat templates, otherwise the context each side sees will already differ.

### 8.2.4 Implementing Synchronous OPD from Scratch

The OPD companion directory likewise provides two standalone scripts, one synchronous and one asynchronous:

| File | Execution mode | Purpose |
| --- | --- | --- |
| `01-demo-sync.py` | Synchronous | The teaching mainline explained piece by piece in the text |
| `02-demo-async.py` | Asynchronous | Handles Student rollout and Teacher logprob requests concurrently |

Following the code order of `01-demo-sync.py`, we now implement one complete round of OPD, from DeepMath prompts to the Student update.

**(1) Prepare arguments and the DeepMath data**

By default, the script downloads the data to `datasets/DeepMath-103K` under the current code directory:

```python
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "datasets" / "DeepMath-103K"
DEEPMATH_SHARDS = 10
```

The arguments of `parse_args()` fall into four groups:

- Data arguments: `dataset_repo`, `dataset_dir`, `num_shards`, `sample_size`;
- Model arguments: the Student base model, LoRA rank, and the Teacher base model or Teacher weight path;
- Training arguments: steps, batch, group, sampling length, KL coefficient and optimizer parameters;
- Logging arguments: SwanLab project, experiment name and logging mode.

DeepMath-103K on ModelScope contains 10 parquet shards. `shard_name()` generates the remote file name, `modelscope_file_url()` generates the download URL, and `download_if_needed()` first writes to a temporary file and then atomically replaces the target, so that an interruption does not leave an incomplete shard behind.

The data-loading function hands the local parquet files to Hugging Face Datasets and keeps only the `question` field needed for training:

```python
def load_deepmath(args: argparse.Namespace):
    shard_paths = []

    for index in range(args.num_shards):
        remote_path = shard_name(index)
        local_path = args.dataset_dir / remote_path
        download_if_needed(
            modelscope_file_url(
                args.dataset_repo,
                args.dataset_revision,
                remote_path,
            ),
            local_path,
            args.force_download,
        )
        shard_paths.append(str(local_path))

    dataset = load_dataset(
        "parquet",
        data_files=shard_paths,
        split="train",
        cache_dir=str(
            args.dataset_dir / ".datasets_cache"
        ),
    )

    if "question" not in dataset.column_names:
        raise ValueError(
            "DeepMath must contain 'question', "
            f"got {dataset.column_names}"
        )

    dataset = dataset.shuffle(seed=args.seed)
    if args.sample_size > 0:
        dataset = dataset.select(
            range(min(args.sample_size, len(dataset)))
        )
    return dataset
```

`--num-shards 1` is suitable for a first end-to-end test run, while `--num-shards 10` covers the full dataset. `--sample-size` controls how many samples actually enter the prompt pool after loading.

**(2) Render the question into a Student prompt**

OPD uses prompt-only data and does not read DeepMath's reference reasoning. `build_prompt()` passes the question and the output requirements to the Student tokenizer:

```python
def build_prompt(
    tokenizer,
    question: str,
    suffix: str,
    enable_thinking: bool,
) -> list[int]:
    content = (
        question.strip()
        if not suffix
        else f"{question.strip()}\n\n{suffix}"
    )
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return tokenizer.encode(
        prompt,
        add_special_tokens=False,
    )
```

The prompt is rendered only once. The Teacher then directly evaluates the same set of tokens, so the context on both sides stays exactly the same.

**(3) Create the Student and the Teacher**

The Student uses a LoRA training client, and the Teacher uses a sampling client that does only forward computation:

```python
service_client = trio.ServiceClient()

training_client = (
    service_client.create_lora_training_client(
        base_model=args.base_model,
        rank=args.lora_rank,
        seed=args.seed,
    )
)
tokenizer = training_client.get_tokenizer()

teacher_client = service_client.create_sampling_client(
    base_model=(
        args.teacher_base_model
        or args.base_model
    ),
    model_path=args.teacher_model_path,
)
```

`teacher_model_path` can point to saved sampler weights. The Teacher does not call `forward_backward()` or `optim_step()`, so its parameters stay fixed throughout OPD.

**(4) Have the Teacher evaluate the Student's same trajectory**

The Student rollout returns completion tokens and old-policy logprobs. The Teacher needs to see `prompt_ids + completion_ids`, and then the completion span is sliced out of the returned result:

```python
def completion_teacher_logprobs(
    teacher_client,
    prompt_ids: list[int],
    completion_ids: list[int],
):
    all_ids = prompt_ids + completion_ids
    all_logprobs = teacher_client.compute_logprobs(
        trio.ModelInput.from_ints(all_ids)
    ).result()

    completion_logprobs = all_logprobs[
        len(prompt_ids):
    ]

    if (
        len(completion_logprobs)
        != len(completion_ids)
        or any(
            value is None
            for value in completion_logprobs
        )
    ):
        raise ValueError(
            "Invalid teacher logprobs "
            "for completion tokens"
        )

    return [
        float(value)
        for value in completion_logprobs
    ]
```

The length check and the `None` check protect token alignment. Only after these checks pass can the $t$-th Student completion token, the Student's old logprob and the Teacher logprob form a single training position.

**(5) Build the per-token OPD Datum**

OPD and GRPO use the same `importance_sampling` schema; the difference lies in the advantage. GRPO copies one within-group advantage across the whole completion, whereas OPD writes an independent reverse-KL advantage for each completion token:

```python
def build_opd_datum(
    prompt_ids: list[int],
    completion_ids: list[int],
    old_logprobs,
    advantages,
):
    prompt_loss_len = len(prompt_ids) - 1

    input_ids = (
        prompt_ids
        + completion_ids[:-1]
    )
    target_ids = (
        [0] * prompt_loss_len
        + completion_ids
    )
    padded_logprobs = (
        [0.0] * prompt_loss_len
        + list(old_logprobs)
    )
    padded_advantages = (
        [0.0] * prompt_loss_len
        + list(advantages)
    )

    if not (
        len(input_ids)
        == len(target_ids)
        == len(padded_logprobs)
        == len(padded_advantages)
    ):
        raise ValueError(
            "OPD datum fields must "
            "have the same length"
        )

    return trio.Datum(
        model_input=trio.ModelInput.from_ints(
            input_ids
        ),
        loss_fn_inputs={
            "target_tokens": np.asarray(
                target_ids,
                dtype=np.int64,
            ),
            "logprobs": np.asarray(
                padded_logprobs,
                dtype=np.float32,
            ),
            "advantages": np.asarray(
                padded_advantages,
                dtype=np.float32,
            ),
        },
    )
```

The right-shift rule is exactly the same as in GRPO. In the prompt span, the three loss inputs still use 0 as placeholders; in the completion span, each position holds the Student token, the Student's old logprob and that token's OPD advantage, one to one.

**(6) Generate Student trajectories and compute reverse KL**

At the start of each step, the code refreshes the Student sampler according to `sampler_refresh_steps`. The default value is 1, so every step uses the latest Student weights:

```python
if (
    student_sampler is None
    or step % args.sampler_refresh_steps == 0
):
    student_sampler = (
        training_client
        .save_weights_and_get_sampling_client()
    )
```

The synchronous version processes the questions in the current batch one after another. For each prompt, the Student first generates `group_size` trajectories, and then the Teacher scores each non-empty completion:

```python
result = student_sampler.sample(
    prompt=trio.ModelInput.from_ints(
        prompt_ids
    ),
    num_samples=args.group_size,
    sampling_params=sampling_params,
    return_text=False,
).result()

for sequence in result.sequences:
    completion_ids = sequence.tokens
    if not completion_ids:
        continue

    student_lps = [
        float(value)
        for value in sequence.logprobs
    ]
    teacher_lps = completion_teacher_logprobs(
        teacher_client,
        prompt_ids,
        completion_ids,
    )

    reverse_kl = (
        np.asarray(student_lps)
        - np.asarray(teacher_lps)
    )
    advantages = (
        -args.kl_penalty_coef
        * reverse_kl
    )

    datums.append(
        build_opd_datum(
            prompt_ids,
            completion_ids,
            student_lps,
            advantages,
        )
    )
    reverse_kls.extend(reverse_kl.tolist())
    completion_token_counts.append(
        len(completion_ids)
    )
```

Here, `group_size` widens the coverage of Student states under the same prompt. Each trajectory receives its own Teacher signal independently, with no GRPO-style within-group centering.

**(7) Update the Student, log metrics and save weights**

After the whole batch has been collected, the code runs one forward-backward pass and one optimizer update:

```python
if not datums:
    raise RuntimeError(
        "No OPD datums were built"
    )

fwd_bwd = training_client.forward_backward(
    datums,
    loss_fn="importance_sampling",
)
optim = training_client.optim_step(adam)

fwd_bwd_result = fwd_bwd.result()
optim.result()
```

The `tokens/s` of one OPD step is the total number of completion tokens divided by the time taken by the whole step. This time includes Student sampling, Teacher logprobs, the Student forward-backward pass, the optimizer update and any necessary sampler refresh:

```python
completion_tokens_total = int(
    sum(completion_token_counts)
)
metrics = {
    "data/datums": len(datums),
    "data/completion_tokens_mean": float(
        np.mean(completion_token_counts)
    ),
    "data/completion_tokens_total": (
        completion_tokens_total
    ),
    "data/completion_tokens_per_second": (
        completion_tokens_total
        / step_elapsed_time
    ),
    "opd/reverse_kl_mean": float(
        np.mean(reverse_kls)
    ),
    "opd/reverse_kl_std": float(
        np.std(reverse_kls)
    ),
    "train/learning_rate": args.learning_rate,
    "time/step_elapsed_time": step_elapsed_time,
}
```

After training finishes, the Student's sampler weights are saved:

```python
save_result = (
    training_client
    .save_weights_for_sampler(
        args.save_weights_name
    )
    .result()
)
print(f"Saved weights: {save_result.path}")
```

### 8.2.5 Running the Synchronous and Asynchronous Versions

A one-step trial of the synchronous version automatically downloads one DeepMath shard and draws at most 20 prompts from it:

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

The asynchronous version uses the same algorithm and arguments:

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

Asynchronous OPD takes advantage of two levels of concurrency at once:

| Training stage | Synchronous version | Asynchronous version |
| --- | --- | --- |
| Create Student / Teacher | Synchronous client interface | `create_lora_training_client_async()` and `create_sampling_client_async()` |
| Student rollout | Calls `sample().result()` prompt by prompt | Concurrent `sample_async()` within a batch |
| Teacher scoring | Calls `compute_logprobs().result()` completion by completion | Concurrent `compute_logprobs_async()` within the same prompt |
| Batch aggregation | Appends Datums sequentially | Aggregates everything after two levels of `asyncio.gather()` |
| Student update | Synchronous future + `.result()` | Submits via the async API, then `await`s the future |
| Program entry point | `main(args)` | `asyncio.run(main(args))` |

The main text covers only the synchronous version, because it directly shows the causal order "Student sampling → Teacher scoring → reverse KL → Student update". The asynchronous version keeps the same token alignment, advantage and optimization objective.

By default, the example uses `Qwen/Qwen3.5-4B` as the Student and `Qwen/Qwen3.6-27B` as the Teacher. The models actually available should be taken from the list of supported models returned by the current PyTRIO service. During training, you should monitor the following together:

- `opd/reverse_kl_mean`: the average difference between the Student's sampled tokens and the Teacher's probabilities;
- `opd/reverse_kl_std`: how spread out the distillation signal is across tokens;
- `data/completion_tokens_total`: the total number of tokens scored by the Teacher in this step;
- `data/completion_tokens_per_second`: the effective completion-token throughput of the whole OPD step;
- `trainer/*`: training metrics returned by PyTRIO;
- Separate task evaluations and general-capability evaluations: to confirm how the model's real capabilities change after distillation.

A decreasing reverse KL indicates that the Student's sampling distribution is approaching the Teacher's. Task accuracy still needs to be confirmed with a fixed held-out evaluation. When the loss fluctuates or capabilities degrade, check, in order, the token alignment, Teacher quality, KL coefficient, learning rate, sampler refresh frequency and the proportion of truncated completions.

Up to this point, both GRPO and OPD operate in a single-turn structure where "the model generates an answer and an external module provides the training signal". The next section, Search-R1, changes the shape of the trajectory further: before finishing its answer, the model can pause generation, call a search environment, read the observation, and then decide on its next action.

## 8.3 Search-R1

> **Code source:** This section is based on the complete multi-file implementation in the author's open-source repository [agentic-rl-lab/03-search-r1](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/03-search-r1). The walkthrough follows the actual call chain through `prepare_data.py`, `protocol.py`, `search.py`, `rollout.py`, `reward.py`, `train.py`, `eval.py` and `analyse.py`.

Large language models store a great deal of knowledge internally, but that knowledge is limited by the time span of the training data, the capacity of the parameters and how well the long tail is covered. In retrieval-augmented generation (RAG), the system usually retrieves documents in advance and then hands the results to the model to answer. Search-R1 hands the search decisions to the model instead: the model can decide when to search, what to search for, whether it needs to keep searching in light of new evidence, and when to stop and give an answer.

This change turns a single answer into a multi-turn trajectory:

$$\tau=(x,a_1,o_1,a_2,o_2,\ldots,a_T),$$

where $a_t$ is a search call or the final answer generated by the model, and $o_t$ is the observation returned by the search environment. The model receives the outcome reward only after completing the whole trajectory, and GRPO then separates successful search paths from failed ones.

The accompanying implementation is not an isolated training script but an execution chain made up of 9 files. When reading the code, it helps to first keep the following main thread in mind:

```text
prepare_data.py / data.py
        ↓ read questions and reference answers
protocol.py
        ↓ build the prompt, parse search calls
search.py
        ↓ run retrieval and return the observation
rollout.py
        ↓ generate complete multi-turn trajectories, rewards and within-group advantages
train.py
        ↓ observation mask, Datum, importance_sampling, optim_step
eval.py / analyse.py
        ↓ evaluation in a fixed environment and result aggregation
```

What is actually trained are the Assistant tokens generated by the language model. The search service handled by `search.py` belongs to the environment; `protocol.py` stitches model actions and environment observations into a continuous trajectory; `rollout.py` orchestrates the two; and only at the end does `train.py` convert trajectories into `Datum` objects that PyTRIO can update on. The code is presented below following this actual call chain.

### 8.3.1 From Fixed Retrieval to Autonomous Search

Search-R1 training samples contain only questions and reference answers. They contain no hand-written search queries and no gold search trajectories. `prepare_data.py` unifies the raw fields of NQ and HotpotQA into the following JSONL structure:

```json
{"id":"...","question":"...","answers":["..."],"data_source":"nq"}
```

The cleaning logic keeps only the fields that can be used for the outcome reward:

```python
def normalize_row(row: dict[str, Any]) -> dict[str, Any] | None:
    question = str(row.get("question") or "").strip()
    answers = extract_answers(row)
    if not question or not answers:
        return None
    return {
        "id": str(row.get("id") or ""),
        "question": question,
        "answers": answers,
        "data_source": str(
            row.get("data_source") or "unknown"
        ),
    }
```

`data.py` then converts each row into a `SearchExample`:

```python
@dataclass(frozen=True)
class SearchExample:
    id: str
    question: str
    answers: list[str]
    data_source: str
```

This data boundary matters. The search queries, the number of searches and the way evidence is used must all be explored by the current policy itself during rollout. If the training set already supplied fixed retrieval trajectories, the task would be closer to trajectory imitation than to the Search-R1 discussed in this section.

The differences between fixed RAG and a search agent are shown in Table 8.3.

| Stage | Fixed RAG | Search-R1 |
| --- | --- | --- |
| Whether to retrieve | Decided by an external pipeline | Decided by the model based on the current state |
| Search query | Preset or generated once | Can be rewritten multiple times based on observations |
| Number of retrieval rounds | Usually fixed | Decided by the policy, subject to a maximum count |
| What is trained | The answering model or the retriever | A unified policy over search actions and the final answer |
| Reward | Commonly a supervised answer loss | Outcome reward for the complete trajectory |

For example, the question "In which country was the author of *The Little Prince* born?" requires first identifying the author and then looking up the author's birthplace. The model can form the following trajectory:

```text
User: What country was the author of The Little Prince born in?

Assistant -> search("The Little Prince author")
Tool: Antoine de Saint-Exupéry ...

Assistant -> search("Antoine de Saint-Exupéry birthplace country")
Tool: He was born in Lyon, France ...

Assistant: Answer: France
```

The second search query depends on the first observation, so it cannot be determined all at once before the rollout begins. The training code needs to maintain environment state and parse the model's action after each assistant turn.

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/8-images/8-4-search-r1.png" alt="Multi-turn search and observation mask in Search-R1" width="90%">
    <p>Figure 8.4 Multi-turn search and observation mask in Search-R1</p>
</div>

The Search-R1 paper uses text tags such as `<search>... </search>` and `<information>... </information>` to represent actions and observations. This chapter uses the Qwen3.5-4B model and adopts the model's native function-calling template:

```python
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web for evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
}
```

Both protocols carry the same reinforcement learning semantics: the Assistant outputs trainable actions, and the search backend returns environment observations. Using the native template reduces conflicts between the model's base abilities and the tool format, and it also preserves structured tool messages.

Once the tool is defined, the model output must also be strictly parsed into one of three states: a valid search, a final answer or invalid output. The parser in `protocol.py` requires that a turn contain exactly one complete `search` call, and that no final answer be tacked on after the tool call:

```python
def parse_assistant(text: str) -> ParsedAssistant:
    matches = list(TOOL_CALL_PATTERN.finditer(text))
    if not matches:
        kind = (
            "invalid" if "<tool_call>" in text else "answer"
        )
        return ParsedAssistant(
            kind=kind,
            content=text.strip(),
        )

    if len(matches) != 1 or text[matches[0].end():].strip():
        return ParsedAssistant(
            kind="invalid",
            content=text.strip(),
        )

    query = matches[0].group(1).strip()
    if not query or "<" in query or ">" in query:
        return ParsedAssistant(
            kind="invalid",
            content=text.strip(),
        )

    content = text[:matches[0].start()].strip()
    return ParsedAssistant(
        kind="tool",
        content=content,
        query=query,
    )
```

There is no separately defined "stop" action here. As long as the output contains no valid tool call, the turn ends as the final answer; `reward.py` then checks whether it follows `Answer: <short answer>`. Output containing an incomplete `<tool_call>` is marked `invalid`, which likewise ends the trajectory and receives a format penalty.

### 8.3.2 How a Search Backend Becomes an Environment Observation

`search.py` provides three backends: DeepSeek Search, Wikipedia and Zhihu. Rollout does not depend directly on any HTTP response format; it depends only on the unified `SearchResult`:

```python
@dataclass(frozen=True)
class SearchItem:
    title: str
    content: str
    source: str | None = None
    url: str | None = None


@dataclass(frozen=True)
class SearchResult:
    ok: bool
    items: list[SearchItem]
    latency: float
    status: int | None = None
    error: str | None = None
```

The training entry point creates the backend according to the command-line arguments, and the state machine that follows always calls the same `search(query)` interface:

```python
def create_search_client(
    backend: str,
    env_path: str | Path | None = None,
    *,
    model: str = "deepseek-v4-flash",
    timeout: float | None = None,
) -> SearchClient:
    resolved_timeout = resolve_search_timeout(
        backend, timeout
    )
    if backend == "deepseek":
        return DeepSeekSearchClient.from_env(
            env_path,
            model=model,
            timeout=resolved_timeout,
        )
    if backend == "wikipedia":
        return WikipediaSearchClient(
            timeout=resolved_timeout
        )
    if backend == "zhihu":
        return ZhihuSearchClient.from_env(
            env_path,
            timeout=resolved_timeout,
        )
    raise ValueError(f"Unsupported search backend: {backend}")
```

Take the Wikipedia backend, which needs no key, as an example: the actual retrieval happens in `_request()`. A single API request performs the full-text search and also fetches the title, body extract and URL of the top three pages:

```python
def _request(
    self,
    query: str,
    started: float,
) -> SearchResult:
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": 3,
            "prop": "extracts|info",
            "explaintext": 1,
            "exintro": 1,
            "exchars": 1200,
            "inprop": "url",
            "redirects": 1,
            "utf8": 1,
        }
    )
    request = urllib.request.Request(
        f"{WIKIPEDIA_SEARCH_ENDPOINT}?{params}",
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "User-Agent": WIKIPEDIA_USER_AGENT,
        },
    )
    with urllib.request.urlopen(
        request,
        timeout=self.timeout,
    ) as response:
        body = response.read()
        if response.headers.get(
            "Content-Encoding"
        ) == "gzip":
            body = gzip.decompress(body)
        payload = json.loads(body.decode("utf-8"))
        pages = payload.get("query", {}).get(
            "pages", []
        )
        if not isinstance(pages, list):
            raise TypeError(
                "Wikipedia search response 'pages' is not a list"
            )
        if any(
            not isinstance(page, dict)
            for page in pages
        ):
            raise TypeError(
                "Wikipedia search response 'page' is not an object"
            )
        ordered_pages = sorted(
            pages,
            key=lambda page: int(
                page.get("index", 1_000_000)
            ),
        )
        items = [
            SearchItem(
                title=str(
                    page.get("title") or "Untitled"
                ).strip(),
                content=str(
                    page.get("extract") or ""
                ).strip(),
                source="Wikipedia",
                url=str(
                    page.get("fullurl") or ""
                ).strip(),
            )
            for page in ordered_pages
            if str(page.get("extract") or "").strip()
        ]
        return SearchResult(
            ok=True,
            items=items,
            latency=time.perf_counter() - started,
            status=response.status,
        )
```

The outer `search()` is also responsible for request rate limiting, timeouts and a limited number of retries on 429 and 5xx errors, and it accumulates the success rate and latency. If a request still fails after retries, it does not let the whole rollout crash with an exception; instead it returns `SearchResult(ok=False, error=...)`. `rollout.py` passes the error text to the model as an observation as well, so the model can go on to answer or rewrite its query.

Results from all backends are ultimately formatted as readable evidence:

```text
[1] Title: ...
    Content: ...
    Source: ...
    URL: ...
```

Returning only titles or links is usually not enough, because what the model really needs is `content` that can support the next reasoning step. `max_tool_response_tokens` and `max_trajectory_tokens` limit the length of the evidence once more before the observation is added to the trajectory.

### 8.3.3 The Multi-Turn Rollout State Machine

Single-turn GRPO needs only one `sample()` call. In Search-R1, each trajectory may go through a different number of search calls, so the trainer has to repeatedly run "generate, parse, call the environment, append the observation".

This chapter stores a trajectory as a `Trajectory`:

```python
@dataclass
class AssistantTurn:
    prompt_tokens: list[int]
    completion_tokens: list[int]
    logprobs: list[float]
    text: str


@dataclass
class Trajectory:
    example: SearchExample
    group_index: int
    messages: list[dict[str, Any]]
    next_prompt_tokens: list[int] | None = None
    question_index: int = 0
    turns: list[AssistantTurn] = field(default_factory=list)
    search_calls: int = 0
    final_text: str = ""
    reward: float = -0.1
    advantage: float = 0.0
    valid_format: bool = False
    exact_match: bool = False
    done: bool = False
```

Each `AssistantTurn` records the full prompt before that turn's sampling, the tokens generated by the model, and the corresponding old-policy logprobs. Search results are stored in `messages` and in the next turn's prompt, but they have no rollout logprobs, because those tokens come from the environment.

The function that actually sends requests to PyTRIO is `sample_requests_async()`. Within the same turn, different questions, or different trajectories that have already diverged, can be sampled concurrently:

```python
async def sample_requests_async(
    sampling_client: Any,
    requests: list[SampleRequest],
    config: RolloutConfig,
    tokenizer: Any,
) -> list[Any]:
    tasks = []
    for request in requests:
        params = trio.SamplingParams(
            max_tokens=request.max_tokens,
            seed=request.seed,
            stop=stop_sequences(tokenizer),
            temperature=config.temperature,
            top_p=config.top_p,
        )
        tasks.append(
            sampling_client.sample_async(
                prompt=trio.ModelInput.from_ints(
                    request.prompt_tokens
                ),
                num_samples=request.num_samples,
                sampling_params=params,
                return_text=True,
            )
        )
    return list(await asyncio.gather(*tasks))
```

After sampling returns, `consume_assistant()` first saves the raw tokens and old logprobs, and then calls the protocol parser from the previous section. A valid search becomes a `PendingSearch`; a final answer, invalid formatting or an exhausted budget all end the current trajectory:

```python
def consume_assistant(
    trajectory: Trajectory,
    prompt_tokens: list[int],
    sequence: Any,
    tokenizer: Any,
    config: RolloutConfig,
) -> PendingSearch | None:
    tokens, logprobs, text = read_sequence(
        sequence, tokenizer
    )
    trajectory.turns.append(
        AssistantTurn(
            prompt_tokens,
            tokens,
            logprobs,
            text,
        )
    )
    parsed = parse_assistant(text)

    can_search = (
        parsed.kind == "tool"
        and trajectory.search_calls
        < config.max_search_calls
        and len(trajectory.turns)
        < config.max_assistant_turns
    )
    if not can_search:
        trajectory.messages.append(
            {"role": "assistant", "content": text}
        )
        trajectory.final_text = text
        trajectory.done = True
        return None

    call_id = (
        f"search-{trajectory.question_index}-"
        f"{trajectory.group_index}-"
        f"{trajectory.search_calls + 1}"
    )
    messages_before_assistant = list(
        trajectory.messages
    )
    trajectory.messages.append(
        {"role": "assistant", "content": text}
    )
    return PendingSearch(
        trajectory=trajectory,
        messages_before_assistant=(
            messages_before_assistant
        ),
        assistant_text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=tokens,
        call_id=call_id,
        query=parsed.query or "",
    )
```

`resolve_searches()` uses a thread pool to call the search backend concurrently for different trajectories, and then pairs the results with the pending trajectories in their original order:

```python
def resolve_searches(
    pending_searches: list[PendingSearch],
    search_client: SearchClient,
    tokenizer: Any,
    config: RolloutConfig,
) -> int:
    if not pending_searches:
        return 0
    workers = min(
        config.search_concurrency,
        len(pending_searches),
    )
    with ThreadPoolExecutor(
        max_workers=workers
    ) as pool:
        results = list(
            pool.map(
                search_client.search,
                [
                    pending.query
                    for pending in pending_searches
                ],
            )
        )
    return sum(
        finish_search(
            pending,
            result,
            tokenizer,
            config,
        )
        for pending, result in zip(
            pending_searches,
            results,
            strict=True,
        )
    )
```

Each result is then written back to its original trajectory by `finish_search()`:

```python
def finish_search(
    pending: PendingSearch,
    result: SearchResult,
    tokenizer: Any,
    config: RolloutConfig,
) -> bool:
    fitted = fit_tool_content(
        tokenizer,
        pending.messages_before_assistant,
        pending.assistant_text,
        pending.prompt_tokens,
        pending.completion_tokens,
        pending.call_id,
        result,
        config,
    )
    if fitted is None:
        pending.trajectory.final_text = (
            pending.assistant_text
        )
        pending.trajectory.done = True
        return True

    content, next_prompt_tokens = fitted
    pending.trajectory.messages.append(
        tool_message(pending.call_id, content)
    )
    pending.trajectory.next_prompt_tokens = (
        next_prompt_tokens
    )
    pending.trajectory.search_calls += 1
    return False
```

`fit_tool_content()` adds search evidence one item at a time. As soon as the observation would exceed the per-tool budget or the full-trajectory budget, it stops adding further evidence; if not even a single result fits, the trajectory ends immediately. In this way, how much text the search service returns and how much text the model actually gets to see are two clearly separated boundaries.

In the first turn, a group of GRPO trajectories shares exactly the same prompt. The code generates `group_size` branches with a single request:

```python
first_requests: list[SampleRequest] = []
for index, trajectory in enumerate(roots):
    request = make_request(
        tokenizer,
        trajectory,
        index,
        config.group_size,
        config.seed + index,
        config,
    )
    if request:
        first_requests.append(request)
```

Each sequence in the sampling response is deep-copied from the root trajectory into an independent branch, and each branch's search action is then parsed separately:

```python
responses = asyncio.run(
    sample_requests_async(
        sampling_client,
        first_requests,
        config,
        tokenizer,
    )
)
pending_searches: list[PendingSearch] = []
for request, response in zip(
    first_requests,
    responses,
    strict=True,
):
    root = roots[request.trajectory_index]
    if len(response.sequences) != config.group_size:
        raise ValueError(
            "First-turn sample count does not match group_size"
        )

    for group_index, sequence in enumerate(
        response.sequences
    ):
        branch = copy.deepcopy(root)
        branch.group_index = group_index
        pending = consume_assistant(
            branch,
            request.prompt_tokens,
            sequence,
            tokenizer,
            config,
        )
        trajectories.append(branch)
        if pending is not None:
            pending_searches.append(pending)

resolve_searches(
    pending_searches,
    search_client,
    tokenizer,
    config,
)
```

Once different branches have run their searches, their observations and action histories already differ. In subsequent turns, `num_samples=1` is set separately for each unfinished trajectory:

```python
while any(not item.done for item in trajectories):
    requests: list[SampleRequest] = []
    for index, trajectory in enumerate(trajectories):
        if trajectory.done:
            continue
        request = make_request(
            tokenizer,
            trajectory,
            index,
            1,
            (
                config.seed
                + index
                + len(trajectory.turns) * 10_000
            ),
            config,
        )
        if request:
            requests.append(request)

    if not requests:
        break
    responses = asyncio.run(
        sample_requests_async(
            sampling_client,
            requests,
            config,
            tokenizer,
        )
    )

    pending_searches = []
    for request, response in zip(
        requests, responses, strict=True
    ):
        if len(response.sequences) != 1:
            raise ValueError(
                "Subsequent turns must sample exactly one branch"
            )
        trajectory = trajectories[
            request.trajectory_index
        ]
        pending = consume_assistant(
            trajectory,
            request.prompt_tokens,
            response.sequences[0],
            tokenizer,
            config,
        )
        if pending is not None:
            pending_searches.append(pending)

    resolve_searches(
        pending_searches,
        search_client,
        tokenizer,
        config,
    )
```

This "shared root, then branch" implementation satisfies two requirements at once: trajectories for the same question can be compared within a GRPO group, and each branch can keep acting on its own search results. Multiple PyTRIO sampling requests and search requests can all run concurrently, which reduces the waiting time introduced by multi-turn interaction with the environment.

The state machine must also set explicit boundaries:

- `max_search_calls`: the maximum number of search calls in one trajectory;
- `max_assistant_turns`: the maximum number of Assistant message turns generated;
- `max_assistant_tokens`: the token limit for a single-turn action;
- `max_tool_response_tokens`: the token limit for a single observation;
- `max_trajectory_tokens`: the total token limit for the entire context;
- Search timeout and concurrency: limit the waiting time on, and concurrent load placed on, the external service.

These limits both control training cost and define the scope of the environment the agent can access. The same configuration must be kept during evaluation; otherwise a larger search budget could by itself raise answer accuracy.

### 8.3.4 Tool Protocol and Token Continuity

The most common problem in multi-turn training is that the tokens obtained by re-rendering the chat template do not match the tokens that were actually sampled. The tokenizer may strip whitespace, normalize tool-call text or automatically append end-of-turn markers. If the model's text is re-encoded during training, the old logprobs can no longer be matched one-to-one with the target tokens.

This chapter follows two principles:

1. Assistant actions always keep the original tokens returned by the sampler;
2. Only the newly added end-of-turn markers and tool observation tokens are computed through the chat template.

The full job of `build_next_prompt()` is to extract from the chat template only the incremental tokens for "Assistant end marker + new observation", and then append them after the real completion returned by the sampler:

```python
def build_next_prompt(
    tokenizer: Any,
    messages_before_assistant: list[dict[str, Any]],
    assistant_text: str,
    previous_prompt_tokens: list[int],
    completion_tokens: list[int],
    next_tool_message: dict[str, Any],
) -> list[int]:
    canonical_prompt = build_prompt(
        tokenizer,
        messages_before_assistant,
    )

    empty_assistant_end = _render_chat(
        tokenizer,
        [
            *messages_before_assistant,
            {"role": "assistant", "content": ""},
        ],
        add_generation_prompt=False,
    )
    if empty_assistant_end[
        :len(canonical_prompt)
    ] != canonical_prompt:
        raise ValueError(
            "chat template cannot extract the end boundary from an empty assistant"
        )
    assistant_closing_tokens = empty_assistant_end[
        len(canonical_prompt):
    ]

    assistant_message = {
        "role": "assistant",
        "content": assistant_text,
    }
    messages_with_assistant = [
        *messages_before_assistant,
        assistant_message,
    ]
    canonical_assistant_end = _render_chat(
        tokenizer,
        messages_with_assistant,
        add_generation_prompt=False,
    )
    canonical_next_prompt = build_prompt(
        tokenizer,
        [*messages_with_assistant, next_tool_message],
    )
    if canonical_next_prompt[
        :len(canonical_assistant_end)
    ] != canonical_assistant_end:
        raise ValueError(
            "chat template rewrote history messages after adding the tool observation"
        )

    observation_tokens = canonical_next_prompt[
        len(canonical_assistant_end):
    ]
    overlap = _suffix_prefix_overlap(
        completion_tokens,
        assistant_closing_tokens,
    )
    return [
        *previous_prompt_tokens,
        *completion_tokens,
        *assistant_closing_tokens[overlap:],
        *observation_tokens,
    ]
```

`overlap` handles the case where the sampler has already returned part of the Assistant end marker, so that tokens are not appended twice. `messages` serves as the protocol record and for debugging, while `next_prompt_tokens` is the model input actually submitted to the sampler in the next turn; the two must not be confused.

When the next round of training data is built, this is verified once more:

```python
if turn.prompt_tokens[:len(full_tokens)] != full_tokens:
    raise ValueError(
        "Next-turn prompt is not a prefix extension of the existing trajectory"
    )
```

These two layers of checks catch template rewrites, duplicated end markers and observation-concatenation errors early. For agentic RL, token continuity is part of algorithmic correctness: a trajectory can look perfectly reasonable at the text level and still produce wrong gradients because of misaligned token boundaries.

### 8.3.5 Reward, Within-Group Advantage and the Observation Mask

This chapter uses the short-answer tasks from NQ and HotpotQA. The model must ultimately output a single line:

```text
Answer: <short answer>
```

The rule-based reward has three levels:

$$R(\tau)=\begin{cases}1, & \text{correct format and exact answer match},\\ 0, & \text{correct format but wrong answer},\\ -0.1, & \text{invalid final format}.\end{cases}$$

`reward.py` first extracts the single final-answer line, and then normalizes case, punctuation, English articles and whitespace:

```python
def normalize_answer(text: str) -> str:
    lowered = text.lower()
    without_punctuation = "".join(
        char
        for char in lowered
        if not unicodedata.category(char).startswith("P")
    )
    without_articles = ARTICLE_PATTERN.sub(
        " ", without_punctuation
    )
    return " ".join(without_articles.split())


def extract_answer(text: str) -> str | None:
    matches = ANSWER_PATTERN.findall(text)
    if len(matches) != 1:
        return None
    answer = matches[0].strip()
    return answer or None


def score_answer(
    text: str,
    references: list[str],
) -> RewardResult:
    answer = extract_answer(text)
    if answer is None:
        return RewardResult(
            -0.1, False, False, None
        )
    normalized = normalize_answer(answer)
    exact_match = any(
        normalized == normalize_answer(reference)
        for reference in references
    )
    return RewardResult(
        float(exact_match),
        True,
        exact_match,
        answer,
    )
```

This code evaluates only the final task outcome; it adds no extra score for "how many searches were made" or "the query looks good". For the $G$ complete trajectories of the same question, the GRPO within-group baseline is still used:

$$A_i=R(\tau_i)-\frac{1}{G}\sum_{j=1}^{G}R(\tau_j).$$

The Search-R1 paper discusses several optimization methods, including PPO and GRPO. This chapter chooses GRPO's same-question grouping scheme, so that Search-R1 uses the same advantage construction as Section 8.1.

`rollout_batch()` may run the following function only after every trajectory in the group has finished:

```python
def assign_group_advantages(
    trajectories: list[Trajectory],
) -> int:
    groups: dict[int, list[Trajectory]] = {}
    for trajectory in trajectories:
        groups.setdefault(
            trajectory.question_index, []
        ).append(trajectory)

    degenerate = 0
    for group in groups.values():
        mean_reward = sum(
            item.reward for item in group
        ) / len(group)
        for item in group:
            item.advantage = (
                item.reward - mean_reward
            )
        if all(
            item.advantage == 0.0 for item in group
        ):
            degenerate += 1
    return degenerate
```

When the rewards in a group are all identical, every advantage is 0 and the group produces no useful gradient. The training code skips these trajectories and records the proportion of degenerate groups. The within-group mean must not be computed after the data has been split into micro-batches; otherwise the comparison baseline has already been changed.

This advantage is assigned to the tokens generated by the Assistant in every turn of the trajectory. A search observation only makes up the state needed for the next decision; it is not an action taken by the policy, so both the logprob and the advantage of observation tokens are set to 0:

```python
full_tokens.extend(delta_observation)
old_logprobs_by_token.extend(
    [0.0] * len(delta_observation)
)
advantages_by_token.extend(
    [0.0] * len(delta_observation)
)

full_tokens.extend(turn.completion_tokens)
old_logprobs_by_token.extend(turn.logprobs)
advantages_by_token.extend(
    [trajectory.advantage]
    * len(turn.completion_tokens)
)
```

This observation mask serves two purposes. First, the trainer never asks the model to predict the text returned by the search engine. Second, the observation still stays in the context, so later Assistant tokens can be modeled conditionally on that evidence.

The search calls and the final answer share a single trajectory-level advantage. If the final answer is correct, the query planning, search queries and answer tokens that led to success are all encouraged; if the final answer is wrong, the entire chain of actions is suppressed. This is exactly why an outcome reward can train a tool-use policy.

### 8.3.6 Building Multi-Turn PyTRIO Datums

`build_datum()` in `train.py` brings together the token continuity and observation mask from the previous sections. Here is the complete core implementation:

```python
def build_datum(
    trajectory: Trajectory,
) -> TrainingDatum:
    if not trajectory.turns:
        raise ValueError(
            "Cannot build a training Datum from a trajectory with no assistant turn"
        )

    full_tokens: list[int] = []
    old_logprobs_by_token: list[float] = []
    advantages_by_token: list[float] = []
    assistant_token_count = 0

    for turn_index, turn in enumerate(
        trajectory.turns
    ):
        if len(turn.completion_tokens) != len(
            turn.logprobs
        ):
            raise ValueError(
                f"Assistant turn {turn_index + 1} "
                "has mismatched token and logprob lengths"
            )

        if turn_index == 0:
            delta_observation = turn.prompt_tokens
        elif turn.prompt_tokens[
            :len(full_tokens)
        ] == full_tokens:
            delta_observation = turn.prompt_tokens[
                len(full_tokens):
            ]
        else:
            raise ValueError(
                f"The prompt of assistant turn {turn_index + 1} "
                "is not a prefix extension of the existing trajectory; "
                "cannot safely align sampled logprobs"
            )

        full_tokens.extend(delta_observation)
        full_tokens.extend(turn.completion_tokens)
        old_logprobs_by_token.extend(
            [0.0] * len(delta_observation)
        )
        old_logprobs_by_token.extend(turn.logprobs)
        advantages_by_token.extend(
            [0.0] * len(delta_observation)
        )
        advantages_by_token.extend(
            [trajectory.advantage]
            * len(turn.completion_tokens)
        )
        assistant_token_count += len(
            turn.completion_tokens
        )

    if assistant_token_count == 0:
        raise ValueError(
            "Cannot build a training Datum from a trajectory with no assistant tokens"
        )
    if not (
        len(full_tokens)
        == len(old_logprobs_by_token)
        == len(advantages_by_token)
    ):
        raise ValueError(
            "Full trajectory token, logprob and advantage lengths do not match"
        )

    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    old_logprobs = old_logprobs_by_token[1:]
    advantages = advantages_by_token[1:]
    if not (
        len(input_tokens)
        == len(target_tokens)
        == len(old_logprobs)
        == len(advantages)
    ):
        raise ValueError(
            "Datum input, target, logprobs and advantages lengths do not match"
        )
    if len(input_tokens) > MAX_TRAIN_CONTEXT_TOKENS:
        raise ValueError(
            f"Datum exceeds {MAX_TRAIN_CONTEXT_TOKENS} tokens"
        )

    datum = trio.Datum(
        model_input=trio.ModelInput.from_ints(
            input_tokens
        ),
        loss_fn_inputs={
            "target_tokens": np.asarray(
                target_tokens,
                dtype=np.int64,
            ),
            "logprobs": np.asarray(
                old_logprobs,
                dtype=np.float32,
            ),
            "advantages": np.asarray(
                advantages,
                dtype=np.float32,
            ),
        },
    )
    return TrainingDatum(datum, len(input_tokens))
```

In the first turn, `delta_observation` is the system prompt, the tool definitions and the user question; in later turns, `delta_observation` is the previous turn's end marker, the search results and the prefix of the new Assistant turn. All of these keep their real target tokens, but their old logprobs and advantages are 0. Only the Assistant tokens actually generated by the sampler carry rollout old logprobs and the trajectory advantage.

The rewards for the complete group must all be computed before the correct within-group mean can be obtained. The policy cannot be updated as soon as a single trajectory finishes, because the rewards of the other branches in the same group are still unknown.

Multi-turn trajectories vary greatly in length, and putting all the data directly into one padded batch would waste a large number of tokens. The accompanying code packs `Datum` objects into multiple micro-batches according to sample length, and reweights the loss of each batch by the ratio of its trajectory count to the total number of trajectories in the rollout. This keeps the gradient semantics of a global per-sample average while controlling padding size per batch.

### 8.3.7 From Rollout to a Single Parameter Update

The outer loop in `train.py` chains the preceding modules into one complete training step. The key ordering cannot be changed:

```python
batch = take_batch(
    examples,
    step * args.questions_per_batch,
    args.questions_per_batch,
)

sampling_client = (
    training_client
    .save_weights_and_get_sampling_client()
)

trajectories = rollout_batch(
    sampling_client=sampling_client,
    tokenizer=tokenizer,
    search_client=search_client,
    examples=batch,
    config=rollout_config,
)

datums = build_training_datums(trajectories)
micro_batches = pack_micro_batches(datums)

trainer_results = []
for micro_batch in micro_batches:
    weighted_datums = (
        weight_micro_batch_for_global_mean(
            micro_batch,
            total_samples=len(trajectories),
        )
    )
    result = training_client.forward_backward(
        weighted_datums,
        loss_fn="importance_sampling",
    ).result()
    trainer_results.append(result)

if micro_batches:
    training_client.optim_step(
        adam_params
    ).result()
```

First, `save_weights_and_get_sampling_client()` exports the current LoRA weights at the start of each step, so the old logprobs come from the current policy before this update. Second, `rollout_batch()` has already computed the rewards and advantages for the whole group internally, and `build_training_datums()` skips trajectories whose advantages are all 0. Third, all micro-batches only accumulate gradients, and `optim_step()` is called just once at the end of the entire logical batch.

PyTRIO averages over the samples within one `forward_backward()` call. If the $k$-th micro-batch contains $n_k$ trajectories and the full rollout batch contains $N$ trajectories, the code first multiplies that batch's advantages by $n_k/N$:

$$\sum_k\frac{n_k}{N}\operatorname{mean}(\mathcal L_k)=\operatorname{mean}(\mathcal L_{\mathrm{global}}).$$

The corresponding implementation only rewraps the `Datum` and does not modify the target tokens or old logprobs:

```python
micro_batch_weight = np.float32(
    len(micro_batch) / total_samples
)
weighted_datums = []
for item in micro_batch:
    loss_inputs = item.datum.loss_fn_inputs
    weighted_datums.append(
        trio.Datum(
            model_input=item.datum.model_input,
            loss_fn_inputs={
                "target_tokens": (
                    loss_inputs["target_tokens"]
                    .to_numpy()
                ),
                "logprobs": (
                    loss_inputs["logprobs"]
                    .to_numpy()
                ),
                "advantages": (
                    loss_inputs["advantages"]
                    .to_numpy()
                    * micro_batch_weight
                ),
            },
        )
    )
```

Dynamic batch splitting therefore changes only the padding and GPU memory usage at execution time; it does not change the per-sample average gradient of the logical batch.

### 8.3.8 Running, Evaluation and Experimental Boundaries

The Search-R1 code is split by responsibility into data, protocol, environment, rollout and training:

| File | Purpose |
| --- | --- |
| `prepare_data.py` | Prepare the NQ and HotpotQA data |
| `data.py` | Load the unified JSONL samples |
| `protocol.py` | Define the search tool and the multi-turn message protocol |
| `search.py` | Wrap the search backends and unify the result format |
| `rollout.py` | Run the multi-turn search state machine |
| `reward.py` | Extract the short answer and compute the rule-based reward |
| `train.py` | Build the observation mask and micro-batches, and update the policy |
| `eval.py` | Evaluate the Base Model or a checkpoint in the same environment |
| `analyse.py` | Aggregate evaluation outputs |

Evaluation does not use a separate, simplified search logic; it reuses the same `rollout_batch()`. The main difference is that only one trajectory is kept per question, since the within-group comparison needed during training is no longer required:

```python
config = RolloutConfig(
    group_size=1,
    max_search_calls=args.max_search_calls,
    max_assistant_turns=args.max_assistant_turns,
    max_trajectory_tokens=(
        args.max_trajectory_tokens
    ),
    max_assistant_tokens=args.max_assistant_tokens,
    max_tool_response_tokens=(
        args.max_tool_response_tokens
    ),
    search_concurrency=args.search_concurrency,
    temperature=args.temperature,
    top_p=args.top_p,
    seed=args.seed,
)

batch_trajectories = rollout_batch(
    sampling_client,
    tokenizer,
    search_client,
    batch,
    config,
)
```

When `model_path` is left empty, `eval.py` creates a Base Model sampler; when it is given the path returned by `save_weights_for_sampler()`, it evaluates the trained checkpoint instead. Both models still use the same protocol, search backend, budgets, question set and answer judge.

First, prepare the data:

```bash
python docs/chapter8/search-r1/prepare_data.py
```

The Wikipedia backend needs no API key and is well suited to verifying the full pipeline:

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

Evaluate the Base Model using the same Wikipedia environment:

```bash
python docs/chapter8/search-r1/eval.py \
    --search-backend wikipedia \
    --batch-size 8 \
    --output docs/chapter8/search-r1/eval_result/base.jsonl
```

When evaluating the trained sampler weights, keep all other arguments unchanged and add only the checkpoint path:

```bash
python docs/chapter8/search-r1/eval.py \
    --search-backend wikipedia \
    --batch-size 8 \
    --model-path 'trio://runxxxxxxxxxx' \
    --output docs/chapter8/search-r1/eval_result/checkpoint.jsonl
```

Once the pipeline is confirmed, gradually scale up the number of questions, the group size, the number of searches and the trajectory budget. An experiment should record at least:

- Exact match of the final answer;
- Proportion of valid answer formats;
- Average number of search calls and the proportion of zero-search trajectories;
- Search failure rate, timeout rate and observation truncation rate;
- Average trajectory length and effective training tokens per step;
- Proportion of degenerate groups;
- Time spent in each stage: rollout, search and training.

Search results have external dependencies. Online pages get updated, and the ranking of a search service may also change. When comparing the Base Model with a trained checkpoint, you should fix the question set, search backend, search budget, sampling parameters and evaluation time window. If a strictly reproducible paper-grade experiment is needed, the retrieval corpus and index can be frozen as a read-only snapshot.

This chapter updates only the language model policy; the search backend stays fixed. If the retriever itself also takes part in learning, you would need to additionally define the retrieval action space, index versions and cross-module credit assignment, which goes beyond the scope of this section's example.

Search-R1 shows how GRPO's within-group relative advantage can be extended to multi-turn search trajectories that include external observations. The next section, ReTool, keeps the same multi-turn rollout and observation mask, swaps the environment from a search engine to a code interpreter, and updates the policy with PyTRIO's built-in PPO.

## 8.4 ReTool

> **Code source:** This section is based on the complete multi-file implementation in the author's open-source repository [agentic-rl-lab/05-retool](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/05-retool). The text follows the real call chain through `prepare_data.py`, `protocol.py`, `sandbox.py`, `rollout.py`, `reward.py`, `train.py`, `eval.py`, and `analysis.py`.

A search tool helps the model obtain external knowledge, while a code interpreter helps the model carry out precise computation, symbolic manipulation, and enumeration-based verification. Guided by prompts or supervised data, models can already call Python tools, but "being able to call a tool" is not the same as "calling it at the right moment." Some problems are faster to solve mentally, some are better suited to writing a short program, and in other cases a failed code execution must be fixed based on the error message.

ReTool uses reinforcement learning (RL) to train this kind of strategic tool-use ability. The model decides on its own whether to call the code interpreter, what code to write, how to read the execution results, and when to give the final answer. The environment provides only an outcome reward based on the final mathematical answer, and the tool decisions in successful trajectories are reinforced together with the whole trajectory.

> **Safety note:** The accompanying `sandbox.py` is merely a local subprocess with resource limits; it is not a trustworthy security sandbox. Code generated by the model can still read any file the current account can access, inherit environment variables, and access the network. Real training should run in a disposable container, a low-privilege virtual machine, or a dedicated sandbox service, with PyTRIO, SwanLab, SSH, and cloud-service credentials removed.

ReTool's code pipeline is structurally similar to Search-R1, but the environment is switched from a search service to a Python process:

```text
prepare_data.py / data.py
        ↓ load math problems and reference answers
protocol.py
        ↓ parse code_interpreter calls
sandbox.py
        ↓ execute code, return stdout / stderr / timeout
rollout.py
        ↓ append execution results and continue generating
reward.py
        ↓ check the last \boxed{}
train.py
        ↓ feedback mask, PPO, optim_step
eval.py / analysis.py
        ↓ text-only vs. ReTool comparative evaluation
```

As before, we explain things by following the actual function calls, rather than looking only at the text format of tool calls.

### 8.4.1 From Tool-Call Format to Tool-Use Strategy

A typical ReTool trajectory looks like this:

```text
User: Count the number of integer solutions that satisfy the conditions.

Assistant -> code_interpreter(
    "count = ...\nprint(count)"
)
Tool: 37

Assistant: Based on the enumeration result, the answer is \boxed{37}
```

If the code contains an error, the model can also use the traceback to correct itself:

```text
Assistant -> code_interpreter("print(sum(values))")
Tool: NameError: name 'values' is not defined

Assistant -> code_interpreter(
    "values = [...]\nprint(sum(values))"
)
Tool: 128

Assistant: \boxed{128}
```

From an RL perspective, the code, execution errors, numerical results, and final answer together make up the trajectory. The code interpreter is the environment; both the tool calls and the natural-language answers generated by the Assistant are actions; and stdout or stderr is the observation.

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/8-images/8-5-retool.png" alt="Multi-turn code execution and outcome reward in ReTool" width="90%">
    <p>Figure 8.5 Multi-turn code execution and outcome reward in ReTool</p>
</div>

ReTool does not need to award separate points for "calling a tool." The final-answer reward is propagated back to earlier code actions through the trajectory-level advantage. This way, only the tool-use patterns that genuinely help the task succeed receive a positive signal, while redundant or misleading calls are suppressed along with failed trajectories.

### 8.4.2 Cold Start, Training Data, and the Tool Protocol

The original ReTool consists of two stages:

1. **Cold-start SFT**: use trajectories containing code calls and execution feedback so that the model first masters the protocol and the basic form of interaction;
2. **Reinforcement Learning**: the model performs rollouts on its own and learns when and how to use tools through the final-answer reward.

The purpose of the cold start is to make the initial policy produce a certain proportion of valid tool calls. If the base model cannot output the tool protocol at all, nearly all early trajectories fail, and GRPO can hardly obtain meaningful exploration signals from within-group comparison.

The code in this chapter uses Qwen3.5, which already has native function-calling ability, and directly demonstrates the second-stage PyTRIO training, so no dedicated cold-start SFT pipeline is included. This is an important boundary between the teaching implementation and the original paper's setup. If you switch to a base model that lacks tool-calling ability, you should first prepare SFT data, verify that the rate of valid calls reaches a trainable level, and only then start RL.

The RL data comes from DAPO-Math-17k. Each original problem is wrapped in a uniform template that asks for `Answer: $Answer`, which conflicts with the `\boxed{}` format required in this section. So `prepare_data.py` first strips the outer template and then keeps only the problem and its reference answer:

```python
def normalize_row(
    index: int,
    row: dict[str, Any],
) -> dict[str, Any] | None:
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        return None

    question = strip_dapo_template(
        str(prompt[0].get("content") or "")
    )
    reward_model = row.get("reward_model") or {}
    ground_truth = (
        reward_model.get("ground_truth")
        if isinstance(reward_model, dict)
        else None
    )
    if isinstance(ground_truth, list):
        ground_truth = (
            ground_truth[0] if ground_truth else None
        )
    answer = str(ground_truth or "").strip()
    if not question or not answer:
        return None

    return {
        "id": str(
            row.get("extra_info", {}).get("index")
            or index
        ),
        "question": question,
        "answer": answer,
        "data_source": str(
            row.get("data_source") or "dapo_math"
        ),
    }
```

As with Search-R1, the training samples contain neither reference code nor the number of tool calls. The policy has to explore on its own whether to call the interpreter and what code to generate.

The code interpreter uses a structured tool definition:

```python
CODE_TOOL = {
    "type": "function",
    "function": {
        "name": "code_interpreter",
        "description": (
            "Run Python code and return printed output. "
            "Each execution starts fresh."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
            },
            "required": ["code"],
        },
    },
}
```

`protocol.py` likewise parses tool calls strictly. Python code may contain comparison operators, quotes, and multi-line text, so here we validate only the number of calls, their position, and whether the code is empty:

```python
def parse_assistant(text: str) -> ParsedAssistant:
    matches = list(TOOL_CALL_PATTERN.finditer(text))
    if not matches:
        kind = (
            "invalid" if "<tool_call>" in text else "answer"
        )
        return ParsedAssistant(
            kind=kind,
            content=text.strip(),
        )

    if len(matches) != 1 or text[matches[0].end():].strip():
        return ParsedAssistant(
            kind="invalid",
            content=text.strip(),
        )

    code = matches[0].group(1).strip()
    if not code:
        return ParsedAssistant(
            kind="invalid",
            content=text.strip(),
        )

    content = text[:matches[0].start()].strip()
    return ParsedAssistant(
        kind="tool",
        content=content,
        code=code,
    )
```

A valid output can go to only one of three destinations: `tool` enters the code-execution environment; `answer` ends the trajectory and receives the outcome reward; `invalid` ends the trajectory and is treated as a wrong answer. The system does not attempt speculative repairs of tool calls that are close to the format but incomplete; otherwise, the definition of actions in the training environment would become unstable.

The rollout state machine is similar to Search-R1's, except that the environment action changes from `search(query)` to `code_interpreter(code)`. Every trajectory is subject to the following budgets:

- the maximum number of code calls;
- the maximum number of Assistant turns;
- the number of tokens per generation turn and per execution result;
- the number of tokens in the full trajectory;
- the time limit for a single execution;
- the concurrency of the local executor.

No Python variables or process state are kept between code calls. The model must redefine any data it needs in every call. This makes the semantics of the trajectory clearer and also reduces implicit dependencies between different executions.

### 8.4.3 How Execution Results Are Spliced Back into the Real Token Trajectory

ReTool, too, must not decode the Assistant text and then re-tokenize it. `protocol.py` uses a placeholder Assistant message to extract the template's closing tokens, and then appends the observation tokens corresponding to the actual execution result after the real completion:

```python
def build_next_prompt(
    tokenizer: Any,
    messages_before_assistant: list[dict[str, Any]],
    previous_prompt_tokens: list[int],
    completion_tokens: list[int],
    next_tool_message: dict[str, Any],
) -> list[int]:
    canonical_prompt = build_prompt(
        tokenizer,
        messages_before_assistant,
    )
    placeholder_message = {
        "role": "assistant",
        "content": "x",
    }
    messages_with_assistant = [
        *messages_before_assistant,
        placeholder_message,
    ]
    canonical_assistant_end = _render_chat(
        tokenizer,
        messages_with_assistant,
        add_generation_prompt=False,
    )

    placeholder_tokens = _encoded_text_tokens(
        tokenizer, "x"
    )
    canonical_action = [
        *canonical_prompt,
        *placeholder_tokens,
    ]
    if canonical_assistant_end[
        :len(canonical_action)
    ] != canonical_action:
        raise ValueError(
            "chat template cannot locate the assistant end boundary"
            " (the placeholder content does not match either)"
        )
    assistant_closing_tokens = canonical_assistant_end[
        len(canonical_action):
    ]

    canonical_next_prompt = build_prompt(
        tokenizer,
        [*messages_with_assistant, next_tool_message],
    )
    if canonical_next_prompt[
        :len(canonical_assistant_end)
    ] != canonical_assistant_end:
        raise ValueError(
            "chat template rewrote history messages after adding the tool observation"
        )
    observation_tokens = canonical_next_prompt[
        len(canonical_assistant_end):
    ]

    overlap = _suffix_prefix_overlap(
        completion_tokens,
        assistant_closing_tokens,
    )
    return [
        *previous_prompt_tokens,
        *completion_tokens,
        *assistant_closing_tokens[overlap:],
        *observation_tokens,
    ]
```

Placeholder content is used here because the Qwen chat template may re-segment the reasoning and content in the actually sampled text. The placeholder message is used only to compute the fixed template boundary; the real model actions always come from `completion_tokens`. The next turn's `prompt_tokens` must have the previous turn's complete tokens as a prefix; otherwise `train.py` will refuse to construct the `Datum`.

### 8.4.4 Outcome Reward and the Interpreter Feedback Mask

This chapter extracts the last $\boxed{}$ from the end of the final answer and uses `math_verify` to decide whether the predicted value is mathematically equivalent to the reference answer:

```python
def extract_last_boxed(text: str) -> str | None:
    marker = "\\boxed{"
    index = text.rfind(marker)
    if index < 0:
        return None

    start = index + len(marker)
    depth = 1
    for position in range(start, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[start:position]
    return None


def answers_equivalent(
    prediction: str,
    reference: str,
) -> bool:
    try:
        return bool(
            verify(
                parse(f"${reference}$"),
                parse(f"${prediction}$"),
            )
        )
    except Exception:
        return False


def score_answer(
    text: str,
    reference: str,
) -> RewardResult:
    answer = extract_last_boxed(
        text[-ANSWER_WINDOW_CHARS:]
    )
    if answer is None:
        return RewardResult(
            -1.0, False, False, None
        )

    correct = answers_equivalent(
        answer.strip(),
        reference.strip(),
    )
    return RewardResult(
        1.0 if correct else -1.0,
        correct,
        True,
        answer,
    )
```

The reward function is:

$$R(\tau)=\begin{cases}+1, & \text{final answer is correct},\\ -1, & \text{answer is wrong or format is invalid}.\end{cases}$$

We then compute a centered advantage over the trajectories for the same problem:

$$A_i=R(\tau_i)-\bar R.$$

Code-execution results are environment observations. They should appear in the model's context, but they must not enter the policy loss. The ReTool paper calls this treatment the interpreter feedback mask; this chapter uses the same token-alignment approach as Search-R1:

| Token region | Kept in context? | old logprob | advantage |
| --- | --- | --- | --- |
| Initial prompt | Yes | 0 | 0 |
| Assistant reasoning and code calls | Yes | rollout logprob | trajectory advantage |
| stdout, stderr, or timeout message | Yes | 0 | 0 |
| Final answer | Yes | rollout logprob | trajectory advantage |

The accompanying code uses PyTRIO's PPO loss:

```python
PPO_LOSS_CONFIG = {
    "clip_low_threshold": 0.8,
    "clip_high_threshold": 1.28,
}

training_client.forward_backward(
    datums,
    loss_fn="ppo",
    loss_fn_config=PPO_LOSS_CONFIG,
).result()
```

The RL stage of the original ReTool paper uses PPO. This chapter keeps the multi-turn code trajectories, the outcome reward, and the interpreter feedback mask, while adding same-problem group sampling and centered advantages. The training pipeline can therefore be summarized as "GRPO constructs relative advantages, and PPO clipping controls policy updates."

An asymmetric probability-ratio bound is used here. When the probability ratio relative to the old policy falls below 0.8 or rises above 1.28, PPO limits the corresponding update. Whatever clip parameters are used, the rollout sampler must be refreshed from the current weights before each training step begins.

### 8.4.5 The Code-Execution Environment and Its Safety Boundary

Executing model-generated code carries real risk. The `sandbox.py` in this chapter uses a local subprocess and adds the following engineering limits:

- Python is launched with an argument list, without going through a shell;
- each call creates an independent process;
- a wall-clock timeout and a CPU-time limit are set;
- on timeout, the entire child process group is terminated;
- the thread count of numerical libraries such as BLAS is limited;
- stdout and stderr are written to temporary files, and the amount read back is capped;
- the number of concurrent executions is limited;
- special markers that could break the chat template are removed.

The execution entry point is `LocalPythonSandbox.run_code()`. The model's code is first embedded in a bootstrap that sets the CPU limit, and is then passed to a new Python process through an argument list:

```python
def run_code(self, code: str) -> ExecResult:
    started = time.perf_counter()
    self.stats.calls += 1
    bootstrap = _BOOTSTRAP.format(
        cpu=int(self.timeout) + 5,
        code=repr(code),
    )
    env = {
        **os.environ,
        **_CHILD_ENV_OVERRIDES,
    }

    with self._semaphore:
        with (
            tempfile.TemporaryFile() as stdout_file,
            tempfile.TemporaryFile() as stderr_file,
        ):
            process = subprocess.Popen(
                [
                    self.python,
                    "-B",
                    "-c",
                    bootstrap,
                ],
                stdout=stdout_file,
                stderr=stderr_file,
                env=env,
                start_new_session=True,
            )
            timed_out = False
            try:
                returncode = process.wait(
                    timeout=self.timeout
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(
                    process.pid,
                    signal.SIGKILL,
                )
                returncode = None
                process.wait()

            stdout = _truncate_tail(
                _read_capped(stdout_file)
            )
            stderr = _truncate_tail(
                _read_capped(stderr_file)
            )

    latency = time.perf_counter() - started
    ok = not timed_out and returncode == 0
    if timed_out:
        self.stats.timeouts += 1
    elif ok:
        self.stats.successes += 1
    else:
        self.stats.errors += 1
    self.stats.latency_total += latency
    return ExecResult(
        ok,
        stdout,
        stderr,
        returncode,
        timed_out,
        latency,
    )
```

stdout and stderr are written to temporary files rather than to a `PIPE`, and the parent process reads only a limited number of bytes from the end of each file. So even if the model keeps printing, the entire output is never held in the parent process's memory. On timeout, the independent process group created by `start_new_session=True` is used, and `killpg()` then terminates the whole group of child processes at once.

The execution result is uniformly converted into a single tool observation:

```python
def format_tool_content(
    self,
    result: ExecResult,
) -> str:
    if result.timed_out:
        return (
            "Error: execution timed out after "
            f"{self.timeout:.0f} seconds."
        )
    if result.ok:
        return sanitize_tool_content(result.stdout)
    if result.stderr:
        return sanitize_tool_content(result.stderr)
    return (
        "Error: process exited with code "
        f"{result.returncode}."
    )
```

On success, stdout is returned; syntax errors and runtime exceptions return the raw stderr; and a timeout returns a fixed error text. The current `run_code()` does not automatically add `print()` for the model, so the system prompt explicitly asks the model to print the values it wants to observe. If nothing is printed, the successful observation is simply an empty string.

These measures mainly guard against accidental infinite loops, output floods, and resource consumption. They do not constitute trustworthy security isolation. A local subprocess may still read files accessible to the running account, access the network, read inherited environment variables, or exploit vulnerabilities in the host environment.

Therefore, the following safety requirements must be observed before running ReTool:

1. Do not execute model code directly on a host that holds production credentials, private keys, or sensitive data;
2. Use a disposable container, a low-privilege virtual machine, or a dedicated sandbox service;
3. Disable the network by default, and use a read-only file system with a minimal set of visible directories;
4. Remove irrelevant environment variables and cloud-service credentials;
5. Set hard limits on CPU, memory, number of processes, disk, and execution time;
6. Save the code, exit status, and resource metrics so that anomalous trajectories can be audited.

"Independent execution" in the tool description means that calls do not share Python state; it does not mean host-level security isolation. Even trial runs for learning purposes should be done in an isolated environment first.

### 8.4.6 How a Multi-Turn Code Rollout Advances

`rollout.py` separately stores the trainable Assistant actions, the full trajectory state, and code calls that have been "generated but not yet executed":

```python
@dataclass
class AssistantTurn:
    prompt_tokens: list[int]
    completion_tokens: list[int]
    logprobs: list[float]
    text: str


@dataclass
class Trajectory:
    example: MathExample
    group_index: int
    messages: list[dict[str, Any]]
    next_prompt_tokens: list[int] | None = None
    question_index: int = 0
    turns: list[AssistantTurn] = field(
        default_factory=list
    )
    code_calls: int = 0
    final_text: str = ""
    reward: float = -1.0
    advantage: float = 0.0
    valid_format: bool = False
    correct: bool = False
    done: bool = False


@dataclass(frozen=True)
class PendingExecution:
    trajectory: Trajectory
    code: str
    call_id: str
    messages_before_assistant: list[dict[str, Any]]
    assistant_text: str
    prompt_tokens: list[int]
    completion_tokens: list[int]
```

`begin_advance()` consumes one Assistant output and decides whether the trajectory ends or enters the execution environment:

```python
def begin_advance(
    trajectory: Trajectory,
    prompt_tokens: list[int],
    sequence: Any,
    tokenizer: Any,
    config: RolloutConfig,
) -> PendingExecution | None:
    tokens, logprobs, text = read_sequence(
        sequence, tokenizer
    )
    text = text.strip()
    trajectory.turns.append(
        AssistantTurn(
            prompt_tokens,
            tokens,
            logprobs,
            text,
        )
    )
    parsed = parse_assistant(text)

    can_code = (
        parsed.kind == "tool"
        and trajectory.code_calls
        < config.max_code_calls
        and len(trajectory.turns)
        < config.max_assistant_turns
    )
    if not can_code:
        trajectory.messages.append(
            {"role": "assistant", "content": text}
        )
        trajectory.final_text = text
        trajectory.done = True
        return None

    call_id = (
        f"code-{trajectory.question_index}-"
        f"{trajectory.group_index}-"
        f"{trajectory.code_calls + 1}"
    )
    messages_before_assistant = list(
        trajectory.messages
    )
    trajectory.messages.append(
        {"role": "assistant", "content": text}
    )
    return PendingExecution(
        trajectory,
        parsed.code or "",
        call_id,
        messages_before_assistant,
        text,
        prompt_tokens,
        tokens,
    )
```

Within the same round, many trajectories may request code execution at the same time. They can run concurrently, but each trajectory must still wait for its own observation to come back before it can start the next generation turn:

```python
async def execute_pending_async(
    pendings: list[PendingExecution],
    sandbox: LocalPythonSandbox,
) -> list[ExecResult]:
    return list(
        await asyncio.gather(
            *(
                sandbox.arun_code(p.code)
                for p in pendings
            )
        )
    )


async def advance_round(
    responses: list[Any],
    requests: list[SampleRequest],
    trajectories: list[Trajectory],
    tokenizer: Any,
    sandbox: LocalPythonSandbox,
    config: RolloutConfig,
    progress_callback: Callable[[int], None] | None,
) -> None:
    pendings: list[PendingExecution] = []
    for request, response in zip(
        requests, responses, strict=True
    ):
        trajectory = trajectories[
            request.trajectory_index
        ]
        pending = begin_advance(
            trajectory,
            request.prompt_tokens,
            response.sequences[0],
            tokenizer,
            config,
        )
        if pending is not None:
            pendings.append(pending)
        elif (
            trajectory.done
            and progress_callback is not None
        ):
            progress_callback(1)

    if not pendings:
        return
    results = await execute_pending_async(
        pendings, sandbox
    )
    for pending, result in zip(
        pendings, results, strict=True
    ):
        finish_advance(
            pending,
            result,
            tokenizer,
            sandbox,
            config,
        )
        if (
            pending.trajectory.done
            and progress_callback is not None
        ):
            progress_callback(1)
```

`finish_advance()` calls `format_tool_content()` from the previous section, and then uses `build_next_prompt()` from Section 8.4.3 to construct contiguous tokens. If an execution result is too long, `fit_tool_content()` repeatedly keeps the tail, because tracebacks and final numerical values are usually at the end; if the result still does not fit in the trajectory budget, this trajectory simply ends.

The branching on the first turn is the same as in Search-R1: each problem submits only one prompt, with `num_samples=group_size`. Each returned sequence becomes an independent trajectory through `copy.deepcopy(root)`. Once execution results come back, the prompts of the branches already differ, so all subsequent requests use `num_samples=1`, until the final answer is output or the budget of code calls, turns, and tokens is exhausted.

Only after all trajectories have finished does the state machine call `score_trajectory()` and `assign_group_advantages()` in turn. Thus, a trajectory's traceback can influence the code it generates later, but it cannot cross the trajectory boundary to affect the other branches in the same group.

### 8.4.7 Training ReTool with PyTRIO

First, create a sampling client from the current weights and generate multi-turn trajectories for each problem:

```python
sampling_client = (
    training_client.save_weights_and_get_sampling_client()
)

trajectories = rollout_batch(
    sampling_client=sampling_client,
    tokenizer=tokenizer,
    sandbox=sandbox,
    examples=batch,
    config=rollout_config,
)
```

`rollout_batch()` does the following:

1. On the first turn, sample `group_size` branches for the same problem;
2. Parse the `code_interpreter` call in each branch;
3. Execute the model's code concurrently and append the tool observation;
4. Continue sampling for trajectories that have not yet finished;
5. Extract the final answer and compute the reward and within-group advantage.

Every training step calls `save_weights_and_get_sampling_client()` again, ensuring that the rollout old logprobs come from the current policy before the update. If all trajectories for a problem are correct, or all are wrong, the centered advantages are all 0 and `build_training_datums()` skips that group; if no `Datum` remains, `optim_step()` is not executed for this step.

Next, each complete trajectory is built into a `Datum`. ReTool's `build_datum()` follows the same prefix-extension rules as Section 8.3.6; its core loop is as follows:

```python
for turn_index, turn in enumerate(trajectory.turns):
    if turn_index == 0:
        delta_observation = turn.prompt_tokens
    elif turn.prompt_tokens[
        :len(full_tokens)
    ] == full_tokens:
        delta_observation = turn.prompt_tokens[
            len(full_tokens):
        ]
    else:
        raise ValueError(
            "the next-turn prompt is not a prefix extension of the existing trajectory"
        )

    full_tokens.extend(delta_observation)
    full_tokens.extend(turn.completion_tokens)
    old_logprobs_by_token.extend(
        [0.0] * len(delta_observation)
    )
    old_logprobs_by_token.extend(turn.logprobs)
    advantages_by_token.extend(
        [0.0] * len(delta_observation)
    )
    advantages_by_token.extend(
        [trajectory.advantage]
        * len(turn.completion_tokens)
    )

input_tokens = full_tokens[:-1]
target_tokens = full_tokens[1:]
old_logprobs = old_logprobs_by_token[1:]
advantages = advantages_by_token[1:]

datum = trio.Datum(
    model_input=trio.ModelInput.from_ints(
        input_tokens
    ),
    loss_fn_inputs={
        "target_tokens": np.asarray(
            target_tokens, dtype=np.int64
        ),
        "logprobs": np.asarray(
            old_logprobs, dtype=np.float32
        ),
        "advantages": np.asarray(
            advantages, dtype=np.float32
        ),
    },
)
```

`delta_observation` contains the initial problem, the Assistant end-of-turn token, and the stdout, stderr, or timeout text; their advantages are all 0. The Assistant's reasoning, code calls, and final answer share the trajectory advantage. This is what the interpreter feedback mask actually looks like in code.

Because trajectory lengths vary widely, the code then packs them into dynamic micro-batches:

```python
datums = build_training_datums(trajectories)
micro_batches = pack_micro_batches(datums)

for micro_batch in micro_batches:
    weighted = weight_micro_batch_for_global_mean(
        micro_batch,
        total_samples=len(trajectories),
    )
    training_client.forward_backward(
        weighted,
        loss_fn="ppo",
        loss_fn_config=PPO_LOSS_CONFIG,
    ).result()

if micro_batches:
    training_client.optim_step(adam_params).result()
```

PyTRIO averages over the samples within a single `forward_backward()` call. When micro-batches differ in size, accumulating them directly would inflate the weight of the smaller batches. This chapter therefore scales the advantages by the ratio of each micro-batch's sample count to the global number of trajectories:

$$\sum_k \frac{n_k}{N}\mathrm{mean}(\mathcal L_k) = \mathrm{mean}(\mathcal L_{\mathrm{global}}).$$

In this way, dynamic batch splitting changes only how the computation is executed, not the gradient's meaning as an average over all samples.

### 8.4.8 Running and Evaluating ReTool

The accompanying directory contains the following files:

| File | Purpose |
| --- | --- |
| `prepare_data.py` | Prepares the DAPO-Math-17k training set and dev set |
| `data.py` | Loads unified math samples |
| `protocol.py` | Defines the code tool and the multi-turn message protocol |
| `sandbox.py` | Executes Python code and records resource metrics |
| `rollout.py` | Runs the multi-turn "generate–run–observe" state machine |
| `reward.py` | Extracts $\boxed{}$ and checks mathematical equivalence |
| `train.py` | Builds the feedback mask and micro-batches and runs PPO |
| `eval.py` | Evaluates text-only and ReTool modes in a unified way |
| `analysis.py` | Aggregates metrics across different checkpoints |

The DAPO-Math-17k `train.jsonl` generated by `prepare_data.py` is used for RL, and an additional 50 problems are split off into `dev.jsonl` for quick checks. The formal `eval.py` uses a fixed set of 30 AIME 2025 problems, to avoid reporting results directly on training problems.

In ReTool mode, the `val_n` candidates for each problem are still generated by the same multi-turn state machine:

```python
config = RolloutConfig(
    group_size=args.val_n,
    max_code_calls=args.max_code_calls,
    max_assistant_turns=args.max_assistant_turns,
    max_trajectory_tokens=(
        args.max_trajectory_tokens
    ),
    max_assistant_tokens=args.max_assistant_tokens,
    max_tool_response_tokens=(
        args.max_tool_response_tokens
    ),
    temperature=args.temperature,
    top_p=args.top_p,
    seed=args.seed,
)

trajectories = await rollout_batch_async(
    sampling_client,
    tokenizer,
    sandbox,
    examples,
    config,
)
```

`text-only` mode, by contrast, registers no code tool and directly samples `val_n` single-turn answers for the same problem. Both modes ultimately write out per-problem JSONL and aggregate:

- `Average@N`: the average accuracy over all generated results;
- `Pass@N`: the fraction of problems for which at least one candidate is correct;
- `format_rate`: the fraction from which a valid `\boxed{}` can be extracted;
- `mean_code_calls` and `mean_turns`: tool cost and trajectory length.

Therefore, when comparing checkpoints you need to fix at least the problem set, `val_n`, temperature, top-p, maximum trajectory length, and answer-extraction logic; when comparing text-only with ReTool, you should additionally report execution latency and the number of code calls.

First, prepare the data:

```bash
python docs/chapter8/retool/prepare_data.py
```

After confirming that the code executor is in an isolated environment, do a one-step trial run:

```bash
python docs/chapter8/retool/train.py \
    --max-steps 1 \
    --questions-per-batch 1 \
    --group-size 4 \
    --max-code-calls 2 \
    --sandbox-workers 2 \
    --swanlab-mode disabled
```

Once the local AIME 2025 data is ready, you can first run the no-tool baseline and then run the ReTool evaluation with the same sampling parameters:

```bash
python docs/chapter8/retool/eval.py \
    --mode text-only \
    --val-n 12 \
    --temperature 1.0 \
    --top-p 0.7 \
    --output docs/chapter8/retool/eval-results/aime25-text-only-base.jsonl

python docs/chapter8/retool/eval.py \
    --mode retool \
    --val-n 12 \
    --temperature 1.0 \
    --top-p 0.7 \
    --output docs/chapter8/retool/eval-results/aime25-retool-base.jsonl
```

When evaluating the trained model, add the same `--model-path trio://...` to both commands. Only when the Base Model, checkpoint, text-only, and ReTool all use the same sampling settings can differences be attributed to training or to the tool environment.

Training metrics should cover three levels at once — the task, the policy, and the environment:

- **Task outcomes**: accuracy, fraction of valid $\boxed{}$, fraction of degenerate groups;
- **Tool behavior**: code-call rate, average number of calls per trajectory, accuracy with zero calls;
- **Self-correction**: the fraction of cases where the first execution errors out, the model retries, and it ends up correct;
- **Environment state**: execution success rate, exception rate, timeout rate, average latency;
- **Policy stability**: PPO probability ratio, clip fraction, response length;
- **System efficiency**: time spent in each stage — rollout, executor, remote training, and checkpointing.

During evaluation, you can run text-only and ReTool modes separately. The two must use the same batch of problems, the same sampling parameters, and the same answer checker in order to reveal the net gain brought by the tool environment. Tool mode should also report the extra latency and execution cost, since accuracy gains usually come with more environment calls.

We have now moved from GRPO's single-turn outcome reward and OPD's per-token Teacher feedback to the multi-turn environment interaction of Search-R1 and ReTool. The chapter summary below brings the four modules back together into a single "rollout–feedback–advantage–policy update" data flow.

## Chapter Summary

Starting from GRPO, this chapter has step by step built up a path for reinforcement learning of large language models (LLMs) that can be extended to agents.

GRPO samples the same problem multiple times to construct within-group relative advantages, eliminating the need to train a separate Value Model. It splits the algorithm into three clear layers: the policy generates rollouts, the environment provides rewards, and the update loss optimizes the model based on the probability ratio between the new and old policies. Correct rewards, token alignment, and refreshing the on-policy sampler together determine whether training is effective.

OPD keeps on-policy rollouts but replaces the rule-based reward with the Teacher's per-token probability feedback on the Student's trajectories. Reverse KL provides a dense signal for every action, allowing a small model to learn the Teacher's abilities on the states it will itself visit. The Teacher and Student's tokenizers, reasoning modes, and complementary capabilities need to be verified before training.

Search-R1 and ReTool further extend trajectories into multi-turn environment interaction. Search results and code-execution results enter the context but are excluded from the policy loss through the observation mask; the search queries, code, reasoning, and final answers generated by the model share the trajectory-level outcome signal. Both reuse GRPO's group sampling and relative advantages, replacing only the tool protocol, environment implementation, and reward.

The four examples ultimately converge into the same PyTRIO data flow:

$$\text{current policy}\rightarrow\text{diverse rollouts}\rightarrow\text{environment or Teacher feedback}\rightarrow\text{token-level advantage}\rightarrow\text{policy update}\rightarrow\text{new policy}.$$

Once you have mastered this data flow, a new Agentic RL task can be designed starting from five questions:

1. What actions can the agent take, and what observations will the environment return?
2. Under what conditions does a trajectory end, and how is it limited by budgets?
3. Can the reward stably reflect the final task objective?
4. Which tokens are policy actions, and which tokens must be masked?
5. Are the rollouts, old logprobs, advantages, and current weights strictly aligned?

Only when all five questions have verifiable answers does a tool demo have the foundation to move into reinforcement learning training.

> **Further learning:** Owing to space constraints and the scope of this chapter, we have focused on GRPO, OPD, Search-R1, and ReTool; many other Agentic RL methods and interactive environments could not be covered one by one. If you would like to go on to learn about OPSD, DAPO, GSPO, ALFWorld, and more, visit the author's continuously maintained [agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab). The repository will keep adding the principles, runnable code, and real training processes of related algorithms.

## References

1. Shao, Z. et al. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300). 2024.
2. Schulman, J. et al. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). 2017.
3. Agarwal, R. et al. [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649). 2023.
4. Li, Y. et al. [Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe](https://arxiv.org/abs/2604.13016). 2026.
5. Sun, J. et al. [EasyOPD: An Easy-to-use On-Policy Distillation Framework for Large Language Models](https://arxiv.org/abs/2607.11012). 2026.
6. Jin, B. et al. [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516). 2025.
7. Feng, J. et al. [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536). 2025.
8. PyTRIO. [Loss Function Guide](https://docs.pytrio.com/docs/guide/loss_fn).
9. PyTRIO. [Search-R1 Example](https://docs.pytrio.com/docs/example/search-r1).
10. Agentic-RL Lab (KMnO4-zx). [agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab).
