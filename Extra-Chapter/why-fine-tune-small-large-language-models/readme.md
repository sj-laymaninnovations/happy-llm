# Big models are so powerful, what is the point of fine-tuning a small model of 0.6B?

Everyone uses Deepseek-R1 or Alibaba’s newly released Qwen3 model in their daily lives. Their models are all very capable, and the API servers provided can also meet everyone’s daily or company development needs. But you can also think of a simple question and a few simple questions, as follows:

1. The company's data is sensitive enough, does it need to be kept confidential?
1. Is it difficult to use big models in daily life, and is it necessary to reason chains?
1. What is the concurrency amount of big model API called by the task? How much money is consumed every day?

For question 1, if the company data is sensitive, then I recommend not calling the big model API provided by the vendor. Even if the supplier guarantees that you will not use your data for training, your data is still leaked (there are unnecessary risks). It is recommended to deploy the big model locally.

For question 2, if it is difficult to use a large model scenario problem and a reasoning chain is needed, you can use the supplier's API, which can ensure that the context of the reasoning chain will not explode. If the problem is simple and there is no chain of inference for urgent needs, it is recommended to deploy the small model locally.

For question 3, if the task is simple and the large model API is called is very concurrency, then I suggest fine-tuning a small model for a specific task and deploying it locally. This can satisfy high concurrency and reduce capital consumption. (Local deployment, default hardware environment single card 4090)

Seeing this, I believe everyone has already thought about the above three questions and has the answer in their minds. Then I'll give a small case.

## The requirements of fine-tuning the model

Suppose your company has a task to extract user information from the text of the complaint. For example, you need to extract the user's name, address, email, complaints, etc. from the following text.

> This is just a small case, and the data is also made in batches by me using large models. The real complaint data will not be so "clean and tidy".

INPUT：
```text
龙琳，宁夏回族自治区璐市城东林街g座 955491，邮箱 nafan@example.com。小区垃圾堆积成山，晚上噪音扰人清梦，停车难上加难，简直无法忍受！
```

OUTPUT：
```json
{
    "name": "龙琳",
    "address": "宁夏回族自治区璐市城东林街g座 955491",
    "email": "nafan@example.com",
    "question": "小区垃圾堆积成山，晚上噪音扰人清梦，停车难上加难，简直无法忍受！"
}
```


Then of course you can call Deepseek's most powerful model R1, or call Alibaba's latest and most powerful model Qwen3-235B-A22B, etc. The information extraction effect of these models is also very good.

But there is a problem. If you have millions of such data to process, calling all the latest and best big models may cost tens of thousands of dollars. Moreover, if these complaint data, such as telecommunications complaint data, power grid complaint data, these data are sensitive and cannot be directly placed on the external network.

Therefore, comprehensive data sensitivity and capital consumption. The best choice is to fine-tune a small model (such as Qwen3-0.6B), which can not only ensure high concurrency, ensure data leakage, ensure the effect of model extraction, and save money! ! !

Now, let’s use a small case to take a practical action and fine-tune the Qwen3-0.6B small model to complete the text information extraction task.



## Configure environment Download data

> Colab file address: https://colab.research.google.com/drive/18ByY11KVhIy6zWx1uKUjSzqeHTme-TtU?usp=drive_link

```python
!pip install datasets swanlab -q
```

```python
!wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1a0sf5C209CLW5824TJkUM4olMy0zZWpg' -O fake_sft.json
```

## Processing data

```python
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model
import torch
```

```python
# 将JSON文件转换为CSV文件
df = pd.read_json('fake_sft.json')
ds = Dataset.from_pandas(df)
ds[:3]
```

```python
model_id = "Qwen/Qwen3-0.6B"
```

```python
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
tokenizer
```

The data format for `supervised-finenetuning` (`sft`, supervised fine-tuning) on large language models is as follows:

```json
{
  "instruction": "回答以下用户问题，仅输出答案。",
  "input": "1+1等于几?",
  "output": "2"
}
```

Among them, `instruction` is a user instruction that informs the model of the task it needs to complete; `input` is user input, which is the input content necessary to complete the user instruction; `output` is the output that the model should give.

The goal of supervised fine-tuning is to give the model the ability to understand and follow user instructions. Therefore, when building data sets, we should build data in a targeted manner to our target tasks. For example, if our goal is to fine-tune a large number of characters' dialogue data to get a model that can role-play Zhen Huan's dialogue style, so the data examples in this scenario are as follows:

```json
{
  "instruction": "你父亲是谁？",
  "input": "",
  "output": "家父是大理寺少卿甄远道。"
}
```

The `Chat Template` format used by `Qwen3` is as follows:

Since `Qwen3` is a hybrid inference model, you can manually choose to turn on thinking mode

Not enabled `thinking mode`


```python
messages = [
    {"role": "system", "content": "You are a helpful AI"},
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "I'm fine, think you. and you?"},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)
print(text)
```
```
<|im_start|>system
You are a helpful AI<|im_end|>
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
<think>

</think>

I'm fine, think you. and you?<|im_end|>
<|im_start|>assistant
<think>

</think>
``` 


The data trained by `LoRA` (`Low-Rank Adaptation`) needs to be formatted and encoded before inputting to the model for training. We need to first encode the input text as `input_ids` and encode the output text as `labels`. The result after encoding is a vector. We first define a preprocessing function, which is used to encode its input, output text for each sample and return an encoded dictionary:


```python
def process_func(example):
    MAX_LENGTH = 1024 # 设置最大序列长度为1024个token
    input_ids, attention_mask, labels = [], [], [] # 初始化返回值
    # 适配chat_template
    instruction = tokenizer(
        f"<s><|im_start|>system\n{example['system']}<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n",
        add_special_tokens=False
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    # 将instructio部分和response部分的input_ids拼接，并在末尾添加eos token作为标记结束的token
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    # 注意力掩码，表示模型需要关注的位置
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    # 对于instruction，使用-100表示这些位置不计算loss（即模型不需要预测这部分）
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 超出最大序列长度截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

```


```python
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
tokenized_id
```


```python
tokenizer.decode(tokenized_id[0]['input_ids'])
```

```python
tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[1]["labels"])))
```

## Loading the model

Load the model and configure LoraConfig

```python
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",torch_dtype=torch.bfloat16)
model
```


```
Qwen3ForCausalLM(
  (model): Qwen3Model(
    (embed_tokens): Embedding(151936, 1024)
    (layers): ModuleList(
      (0-27): 28 x Qwen3DecoderLayer(
        (self_attn): Qwen3Attention(
          (q_proj): Linear(in_features=1024, out_features=2048, bias=False)
          (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (v_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (o_proj): Linear(in_features=2048, out_features=1024, bias=False)
          (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
          (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
        )
        (mlp): Qwen3MLP(
          (gate_proj): Linear(in_features=1024, out_features=3072, bias=False)
          (up_proj): Linear(in_features=1024, out_features=3072, bias=False)
          (down_proj): Linear(in_features=3072, out_features=1024, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
        (post_attention_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
      )
    )
    (norm): Qwen3RMSNorm((1024,), eps=1e-06)
    (rotary_emb): Qwen3RotaryEmbedding()
  )
  (lm_head): Linear(in_features=1024, out_features=151936, bias=False)
)
```

```python
model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法
```


## Lora Config

There are many parameters that can be set in the `LoraConfig` class, and the more important ones are as follows

- `task_type`: model type. Most of the models of `decoder_only` are causal language models `CAUSAL_LM`
- `target_modules`: The name of the model layer that needs to be trained is mainly the layer of the `attention` part. The corresponding layers of different models have different names.
- `r`: The rank of `LoRA` determines the dimension of the low rank matrix. A smaller `r` means fewer parameters
- `lora_alpha`: Scaling parameter, together with `r`, determines the strength of the `LoRA` update. The actual scaling is `lora_alpha/r`, which in the current example is `32 / 8 = 4` times
- `lora_dropout`: `dropout rate` applied to the `LoRA` layer, used to prevent overfitting


```python
from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
config
```


```python
model = get_peft_model(model, config)
config
```


```python
model.print_trainable_parameters()  # 模型参数训练量只有0.8395%
```

> trainable params: 5,046,272 || all params: 601,096,192 || trainable%: 0.8395


## Training Arguments

- `output_dir`: The output path of the model
- `per_device_train_batch_size`: `batch_size` on each card
- `gradient_accumulation_steps`: Gradient accumulation
- `num_train_epochs`: As the name implies `epoch`


```python
args = TrainingArguments(
    output_dir="Qwen3_instruct_lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=1,
    num_train_epochs=3,
    save_steps=50,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)
```

## Introduction to SwanLab

[SwanLab](https://github.com/swanhubx/swanlab) is an open source model training and recording tool for AI researchers and provides training visualization, automatic logging, hyperparameter recording, experimental comparison, multi-person collaboration and other functions. On `SwanLab`, researchers can discover training problems based on intuitive visual charts, compare multiple experiments to find research inspiration, and break the barriers of team communication through online link sharing and organization-based multi-person collaborative training.

**Why do you need to record training**

Compared with software development, model training is more like an experimental science. Behind a model with excellent quality is often thousands of experiments. Researchers need to constantly try, record, compare, and accumulate experience to find the best model structure, hyperparameters and data ratios. Among them, how to record and compare efficiently is crucial to improving research efficiency.

`(2) Use an existing SwanLab account` and log in using the private API Key

```python
import swanlab
from swanlab.integration.transformers import SwanLabCallback

# 实例化SwanLabCallback
swanlab_callback = SwanLabCallback(
    project="Qwen3-Lora",  # 注意修改
    experiment_name="Qwen3-8B-LoRA-experiment"  # 注意修改
)
```


```python
import swanlab
from swanlab.integration.transformers import SwanLabCallback

# 实例化SwanLabCallback
swanlab_callback = SwanLabCallback(
    project="Qwen3-Lora",
    experiment_name="Qwen3-0.6B-extarct-lora-2"
)
```


```python
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback]
)
```


```python
trainer.train()
```


## Test text


```python
prompt = "龙琳   ，宁夏回族自治区璐市城东林街g座 955491，nafan@example.com。小区垃圾堆积成山，晚上噪音扰人清梦，停车难上加难，简直无法忍受！太插件了阿萨德看见啊啥的健康仨都会撒娇看到撒谎的、"

messages = [
    {"role": "system", "content": "将文本中的name、address、email、question提取出来，以json格式输出，字段为name、address、email、question，值为文本中提取出来的内容。"},
    {"role": "user", "content": prompt}
]

inputs = tokenizer.apply_chat_template(messages,
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True,
                                       enable_thinking=False).to('cuda')

gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```


```json
{
    "name": "龙琳",
    "address": "宁夏回族自治区璐市城东林街g座 955491",
    "email": "nafan@example.com",
    "question": "小区垃圾堆积成山，晚上噪音扰人清梦，停车难上加难，简直无法忍受！太插件了阿萨德看见啊啥的健康仨都会撒娇看到撒谎的、"
}
```
