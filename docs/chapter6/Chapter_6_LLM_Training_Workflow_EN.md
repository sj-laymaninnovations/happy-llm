# Chapter 6: LLM Training Workflow Practice

## Part 1: Main Chapter Content

# Chapter 6: LLM Training Workflow in Practice

## 6.1 Model pre-training

In the previous chapter, we gradually disassembled the model structure and training process of LLM, implemented the LLaMA model structure and the full Pretrain and SFT pipeline by hand from scratch, and understood the model principles and training details of LLM more deeply. However, in practical applications, hand-written LLM training implementations have the following problems:

- Implementing the LLM structure by hand takes a lot of work, making it difficult to follow up on the structural innovations of the latest models in real time;
- LLM training implemented from scratch cannot support multi-GPU distributed training well, and the training efficiency is low;
- Incompatible with existing pre-trained LLM, pre-trained model parameters cannot be used

Therefore, in this chapter, we will introduce the current mainstream training framework Transformers in the LLM field, and combine the distributed framework deepspeed and efficient fine-tuning framework peft to practice using transformers to perform the entire process of model Pretrain and SFT to better connect with the industry's mainstream LLM technical solutions.

### 6.1.1 Framework Introduction

Transformers is an NLP framework developed by Hugging Face. Through modular design, it achieves unified support for hundreds of mainstream model architectures such as BERT, GPT, LLaMA, T5, and ViT. By using Transformers, developers do not need to repeatedly implement the basic network structure, and can load any pre-trained model in one click through the AutoModel class. Figure 6.1 is the Hugging Face Transformers course homepage:

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/6-images/1-1.png" alt="alt text" width="90%">
    <p>Figure 6.1 Hugging Face Transformers</p>
</div>

At the same time, the framework's built-in Trainer class encapsulates the core logic of distributed training and supports PyTorch native DDP, DeepSpeed, Megatron-LM and other distributed training strategies. By simply configuring the training parameters, hybrid parallel training of data parallelism, model parallelism, and pipeline parallelism can be achieved. Efficient training of 10 billion parameter models can be easily supported on an 8-GPU A100 cluster. In conjunction with components such as SavingPolicy and LoggingCallback, the automated management of the training process is realized. It also supports integration with frameworks such as Deepspeed, peft, wandb, Swanlab, etc., and can be seamlessly connected through parameter settings, so as to quickly and efficiently realize LLM training.

More importantly for NLP researchers in the LLM era, HuggingFace built its huge AI community based on the Transformers framework, opened up hundreds of millions of pre-trained model parameters and 250,000+ different types of data sets. Through multiple frameworks such as Transformers, Dataset, and Evaluate, it can integrate pre-trained models, data sets and evaluation functions, thereby helping developers to use any pre-trained model and conveniently realize the development and application of personal models based on open source models and data sets.

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/6-images/1-2.png" alt="alt text" width="90%">
    <p>Figure 6.2 Hugging Face Transformers Model Community</p>
</div>

In the LLM era, there are fewer and fewer adjustments and re-pre-training of model structures. Instead, most developers' business work consists of using pre-trained LLMs for Post-Training and SFT to support their own downstream business applications. Moreover, due to the large size of the pre-trained model, convenient integration of distributed training frameworks such as deepspeed has gradually become a necessary skill for NLP model training in the LLM era. Therefore, Transformers has gradually become the mainstream framework of NLP technology in academia and industry. Whether it is corporate business development or scientific research, Transformers are gradually chosen to implement the model. At the same time, newly released open source LLMs such as DeepSeek and Qwen will also open up their pre-training weights and model call demos in the Transformers community as soon as possible. By using the Transformers framework, LLM training and development can be completed efficiently and conveniently, achieving industrial-grade output delivery. Next, based on the Transformers framework, we will introduce how to implement Pretrain and SFT of LLM through the Transformers framework.

### 6.1.2 Initialize LLM

We can use transformers' AutoModel class to directly initialize the implemented model. For any pre-trained model, the parameters include the configuration information of the model. If you want to train an LLM from scratch, you can use an existing model architecture to initialize it directly. Here, we take the model architecture of [Qwen-2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main) as an example:

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/6-images/1-3.png" alt="alt text" width="90%">
    <p>Figure 6.3 Qwen-2.5-1.5B</p>
</div>

This interface is the Qwen-2.5-1.5B model parameters in the HuggingFace community. The `config.json` file is the configuration information of the model, including the model's architecture, hidden layer size, model layer number, etc., as shown in Figure 6.4:

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/6-images/1-4.png" alt="alt text" width="90%">
    <p>Figure 6.4 Qwen-2.5-1.5B config.json file</p>
</div>

We can use the configuration information of this model to initialize a Qwen-2.5-1.5B model for training, or we can make changes based on this configuration information, such as modifying the size of the hidden layer, the number of attention heads, etc. to customize a model structure. HuggingFace provides Python tools to easily download the model parameters you want to use:

```python
import os
# Set the environment variable; here we use the HuggingFace mirror site
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# Download the model
os.system('huggingface-cli download --resume-download Qwen/Qwen2.5-1.5B --local-dir your_local_dir')
```

As shown in Figure 6.5, the "Qwen/Qwen2.5-1.5B" here is the identifier of the model to be downloaded. For other models, you can directly copy the model name on HuggingFace:

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/6-images/1-5.png" alt="alt text" width="90%">
    <p>Figure 6.5 Model download identifier</p>
</div>

After the download is completed, you can use the AutoConfig class to directly load the downloaded configuration file:

```python
# Load the defined model configuration - Qwen-2.5-1.5B is used as the example here
# Load it with the transformers Config class
from transformers import AutoConfig

# Local path of the downloaded parameters
model_path = "qwen-1.5b"
config = AutoConfig.from_pretrained(model_name_or_path)
```

You can also customize the configuration file and load it in the same way. You can use the AutoModel class to generate the corresponding model based on the loaded configuration objects:

```python
# Build a model defined by this configuration
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_config(config,trust_remote_code=True)
```

Since LLM is generally a CausalLM architecture, the AutoModelForCausalLM class is used for loading here. If used for classification task training, you can use the AutoModelForSequenceClassification class to load. Looking at the model, Figure 6.6 can see that its architecture and defined configuration files are the same:

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/6-images/1-6.png" alt="alt text" width="70%">
    <p>Figure 6.6 Model structure output results</p>
</div>

This model is a Qwen-2.5-1.5B model initialized from zero. Generally speaking, we rarely initialize LLM from zero for pre-training. The more we do it is to load a pre-trained LLM weight and post-train it on our own corpus. Here, we also introduce how to initialize a pretrained model from the downloaded model parameters.

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(model_name_or_path,trust_remote_code=True)
```

Similarly, just use the from_pretrained method to load. The model_name_or_path here is the local path to the downloaded parameters.

We also need to initialize a tokenizer. Here, we can use the tokenzier parameter corresponding to Qwen-2.5-1.5B directly:

```python
# Load a pre-trained tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
```

The loaded tokenizer can be used directly to tokenize any text.

### 6.1.3 Pre-training data processing

Similar to Chapter 5, we use Mobvoi's open-source Sequence Monkey dataset as the pre-training dataset, and the dataset can be downloaded and decompressed in a consistent manner as in Chapter 5. HuggingFace's datasets library is a third-party library for data download and processing that is supported by the transformers framework. We can directly use the load_dataset function of datasets to load pretrained data:

```python
# Load the pre-training data
from datasets import load_dataset

ds = load_dataset('json', data_files='/mobvoi_seq_monkey_general_open_corpus.jsonl')
```

Note that because the dataset is large, loading may take a long time or run out of memory. For early testing, it is recommended to split off a portion of the pre-training dataset and test on that. The loaded ds is a DatasetDict object. The loaded data will be saved in the value corresponding to the `train` key by default. You can view it through the following code:

```python
ds["train"][0]
```

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/6-images/1-7.png" alt="alt text" width="100%">
    <p>Figure 6.7 Dataset Display</p>
</div>

You can view the characteristics of the dataset (that is, columns) through the feature attribute. Here you need to save the column names of the dataset, because in the subsequent processing, once the text has been tokenized, the original text column needs to be removed:

```python
# Inspect the features
column_names = list(ds["train"].features)
# columnes_name:["text"]
```

Then use the loaded tokenizer to process the data set, and use the map function to batch process:

```python
# Tokenize the dataset
def tokenize_function(examples):
    # Tokenize with the previously loaded tokenizer
    output = tokenizer([item for item in examples["text"]])
    return output

# Batch processing
tokenized_datasets = ds.map(
    tokenize_function,
    batched=True,
    num_proc=10,
    remove_columns=column_names,
    load_from_cache_file=True,
    desc="Running tokenizer on dataset",
)
```

The data set after processing will include two columns 'input_ids' and 'attention_mask', which are the numerical sequence after text tokenize and the attention mask (which marks whether a position is padding). The map method will remove the original 'text' through the remove_columns parameter and will no longer be used during training.

Since pre-training is generally a CLM task, learning the sequence semantics of multiple samples at one time does not affect the model performance, and the training data is large and the training time is long, which requires relatively high training efficiency. During the pre-training process, multiple text segments are generally spliced together and processed into text blocks of uniform length, and then each text block is trained. Here, we implement a splicing function to splice text blocks to 2048 token lengths, and then batch processing is performed through the map method:

```python
# Pre-training usually concatenates texts into fixed-length blocks
from itertools import chain

# Here we use a block length of 2048
block_size = 2048

def group_texts(examples):
    # Concatenate the text segments
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    # Compute the total length of the concatenated text
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # If the length is too long, split it into blocks
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split into chunks of block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    # For the CLM task, labels are the same as the inputs
    result["labels"] = result["input_ids"].copy()
    return result

# Batch processing
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=10,
    load_from_cache_file=True,
    desc=f"Grouping texts in chunks of {block_size}",
    batch_size = 40000,
)
train_dataset = lm_datasets["train"]
```

The processed train_dataset is a pre-trained dataset that can be directly used for CLM Pretrain, with each sample length of 2048 tokens.

### 6.1.4 Training with Trainer

Next, we use the Trainer class provided by transformers for training. Trainer encapsulates the training logic of the model and does good efficiency optimization, visualization and other work, which can efficiently and conveniently complete LLM training.

First, we need to configure the training hyperparameters and instantiate a parameter object using the TrainingArguments class:

```python
from transformers import TrainingArguments
# Configure the training arguments

training_args = TrainingArguments(
    output_dir="output",# Output path for training artifacts
    per_device_train_batch_size=4,# Training batch_size
    gradient_accumulation_steps=4,# Gradient accumulation steps; effective bs = configured bs * accumulation steps
    logging_steps=10,# Step interval for logging the loss
    num_train_epochs=1,# Number of training epochs
    save_steps=100, # Step interval for saving model parameters
    learning_rate=1e-4,# Learning rate
    gradient_checkpointing=True# Enable gradient checkpointing
)
```

Then, based on the initialized model, tokenzier and training_args, pass in the processed training data set and instantiate a trainer object:

```python
from transformers import Trainer, default_data_collator
from torchdata.datapipes.iter import IterableWrapper

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= IterableWrapper(train_dataset),
    eval_dataset= None,
    tokenizer=tokenizer,
    # The default collator is for MLM; use the CLM collator
    data_collator=default_data_collator
)
```

Then use the train method, which will be trained and saved according to the configured training hyperparameters:

```python
trainer.train()
```

> Note: The above code is stored in the `./code/pretrian.ipynb` file.

### 6.1.5 Use DeepSpeed to implement distributed training

Due to the large scale and long time of pre-training, it is generally not recommended to use Jupyter Notebook to run, which is prone to interruption. Moreover, due to the large scale of pre-training, it is generally necessary to use multiple GPUs for distributed training, otherwise the training time will be too long. Here, we introduce how to implement distributed training using the DeepSpeed framework based on the above code to complete the industry-available LLM Pretrain.

For long-term training, generally use bash scripts to set hyperparameters, and then start the written python script to implement training. We use a Python script (`./code/pretrain.py`) to implement the entire training process.

Import the required third-party library first:

```python
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from torchdata.datapipes.iter import IterableWrapper
from itertools import chain
import deepspeed
from typing import Optional,List

import datasets
import pandas as pd
import torch
from datasets import load_dataset
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
import datetime
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
import swanlab
```

First, you need to define several types of hyperparameters to handle the hyperparameter values set in the sh script. Since transformers itself has the TraingArguments class, it includes some essential hyperparameters for training. Here we only need to define the hyperparameters not included in TrainingArguments, which mainly include model-related hyperparameters (defined in ModelArguments) and data-related hyperparameters (defined in DataTrainingArguments):

```python
# Hyperparameter classes
@dataclass
class ModelArguments:
    """
    Model-related arguments
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Used for post-training: path to the pre-trained model parameters"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Used for pre-training: path to the Config file"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Path to the pre-trained tokenizer"}
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Data type used for model training; bfloat16 is recommended"
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Training-related arguments
    """

    train_files: Optional[List[str]]  = field(default=None, metadata={"help": "Path(s) to the training data"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Length of each text block"
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "Number of threads used for preprocessing."},
    )
```

Then a main function can be defined to implement the encapsulation of the above training process. First, use the HfArgumentParser tool provided by transformers to load the hyperparameters set in the sh script:

```python
# Load the script arguments
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
```

In large-scale training, log is generally used to save information about the training process. It is generally not recommended to use print to print directly, which is prone to loss of key training information. Here, we directly use the logging library that comes with python to implement logging. First, log settings need to be performed:

```python
# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Set the log level to INFO
transformers.utils.logging.set_verbosity_info()
log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()
```

Here the log level is set to INFO. The logging log has five levels: DEBUG, INFO, WARNING, ERROR and CRITICAL. To which level the log is set, only the level and information above the level will be output. After the setting is completed, just use logger directly where you need to record the log. The level of logging will be specified when recording, for example:

```python
# Log an overview of the training setup
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
logger.info(f"Training/evaluation parameters {training_args}")
```

I will not repeat the logging in the script in the future.

In large-scale training, interrupts are often inevitable. Training generally saves checkpoints at fixed intervals. After the interruption, training can be restored based on the recent checkpoint. Therefore, we need to first detect if the old checkpoint exists and resume training from the checkpoint:

```python
# Check for an existing checkpoint
last_checkpoint = None
if os.path.isdir(training_args.output_dir):
    # Detect it automatically with transformers' built-in get_last_checkpoint
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        raise ValueError(
            f"Output directory ({training_args.output_dir}) is not empty "
        )
    elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(
            f"Resuming training from {last_checkpoint}"
        )
```

Then the model is initialized in the way described above. Here, the initialization from zero and initialization based on the existing pre-trained model will be packaged together:

```python
# Initialize the model
if model_args.config_name is not None:
    # from scrach
    config = AutoConfig.from_pretrained(model_args.config_name)
    logger.warning("You are initializing a model from scratch")
    logger.info(f"Model config path: {model_args.config_name}")
    logger.info(f"Model config: {config}")
    model = AutoModelForCausalLM.from_config(config,trust_remote_code=True)
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Pre-training a new model - Total size={n_params/2**20:.2f}M params")
elif model_args.model_name_or_path is not None:
    logger.warning("You are initializing a pre-trained model")
    logger.info(f"Model parameter path: {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path,trust_remote_code=True)
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Inheriting a pre-trained model - Total size={n_params/2**20:.2f}M params")
else:
    logger.error("config_name and model_name_or_path cannot both be empty")
    raise ValueError("config_name and model_name_or_path cannot both be empty")
```

Similarly, the tokenizer loading and pre-training data are processed. This part is exactly the same as above and will not be repeated here. Readers can view the details in the code in detail. Similarly, use Trainer for training:

```python
logger.info("Initializing Trainer")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= IterableWrapper(train_dataset),
    tokenizer=tokenizer,
    data_collator=default_data_collator
)

# Load from checkpoint
checkpoint = None
if training_args.resume_from_checkpoint is not None:
    checkpoint = training_args.resume_from_checkpoint
elif last_checkpoint is not None:
        checkpoint = last_checkpoint

logger.info("Starting training")
train_result = trainer.train(resume_from_checkpoint=checkpoint)
trainer.save_model() 
```
Note that since the above checkpoint is detected, resume_from_checkpoint is used here to implement the function of restoring training from checkpoint.

Since it is particularly important to monitor training progress and loss downward trend in large-scale training, in the script, we use swanlab as a training monitoring tool. Swanlab is initialized at the beginning of the script:

```python
# Initialize SwanLab
swanlab.init(project="pretrain", experiment_name="from_scrach")
```

After starting training, the terminal will output the url monitored by swanlab and click to observe the training progress. The details of the use of swanlab will not be detailed here. Readers are welcome to check the relevant information description.

After completing the above code, we use a sh script (`./code/pretrain.sh`) to define the value of the hyperparameter and start training through Deepspeed, thereby achieving efficient multi-GPU distributed training:

```bash
# Set the visible GPUs
CUDA_VISIBLE_DEVICES=0,1

deepspeed pretrain.py \
    --config_name autodl-tmp/qwen-1.5b \
    --tokenizer_name autodl-tmp/qwen-1.5b \
    --train_files autodl-tmp/dataset/pretrain_data/mobvoi_seq_monkey_general_open_corpus_small.jsonl \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --do_train \
    --output_dir autodl-tmp/output/pretrain \
    --evaluation_strategy  no \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --warmup_steps 200 \
    --logging_dir autodl-tmp/output/pretrain/logs \
    --logging_strategy steps \
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps 100 \
    --preprocessing_num_workers 10 \
    --save_total_limit 1 \
    --seed 12 \
    --block_size 2048 \
    --bf16 \
    --gradient_checkpointing \
    --deepspeed ./ds_config_zero2.json \
    --report_to swanlab
    # --resume_from_checkpoint ${output_model}/checkpoint-20400 \
```
After installing the Deepspeed third-party library, you can start multi-GPU training directly through the Deepspeed command. The above script commands mainly define the values of various hyperparameters, which can be used in reference. In Chapter 4, we introduce the principles of DeepSpeed distributed training and the ZeRO stage setting, where we train using ZeRO-2. `ds_config_zero.json` is loaded here as the configuration parameter for DeepSpeed:

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": "auto"
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "none",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 100,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

Finally, run the `pretrain.sh` script in terminal bash to start training.

## 6.2 Supervised Fine-Tuning

In the previous section, we introduced how to use the Transformers framework to quickly and efficiently pre-train models. In this section, we will introduce how to use the Transformers framework to perform supervised fine-tuning of pre-trained models based on the above section.

### 6.2.1 Pretrain VS SFT

First we need to review what the core difference between pre-training and supervised fine-tuning of LLM is. It was mentioned in Chapter 4 that the currently formed LLM is generally trained through the three stages of Pretrain-SFT-RLHF. In the Pretrain stage, a large amount of unsupervised text will be self-supervised to learn text semantic rules and world knowledge in text; in the SFT stage, the pre-trained model is generally instruction-tuned, that is, training the model to complete corresponding tasks according to user instructions, so that the model can follow user instructions and plan, act and output according to user instructions. Therefore, both Pretrain and SFT are modeled using CLM. The core difference is that Pretrain uses a large amount of unsupervised text for training, and the model directly performs the task of "predicting the next token" on the text; while SFT uses constructed instruction-response pairs, and the model models the subsequent output based on the input instruction. Reflected in the specific training implementation, Pretrain will perform loss calculations on all texts, requiring the model to model and predict the entire text; while SFT only performs loss calculations on the output, and does not calculate the loss of the instruction part.

Therefore, compared with the Pretrain code completed in the previous section, the SFT part only needs to modify the data processing link to convert the instruction-pair data into training samples. The rest of the part is completely consistent with Pretrain. This part of the code script is `./code/finetune.py`.

### 6.2.2 Fine-tuning data processing

Also similar to Chapter 5, we use the BelleGroup dataset open-sourced by Beike (Ke Holdings) for SFT here.

During the SFT process, we define a Chat Template, which represents how conversation data is converted into a sequence of text that the model can model and simulate. When we use SFT-based models to fine-tune downstream tasks, we generally need to check the Chat Template of the model and adapt it, so as not to damage the instruction-following ability it learned during SFT. Since we use the Pretrain model for SFT here, we can customize a Chat Template. Since we use the Qwen-2.5-1.5B model structure for Pretrain, here we inherit the Chat Template of Qwen-2.5. If the reader does not have enough resources to perform the Pretrain of the previous part of the model, the official Qwen-2.5-1.5B model can also be used as the base model of the SFT.

We first define several special tokens. Special tokens have special functions in fitting the model, including text sequence start (BOS), text sequence end (EOS), line breaks, etc. Defining special tokens helps avoid semantic confusion in the model during fitting:

```python

# Different tokenizers need their own definitions
# BOS
im_start = tokenizer("<|im_start|>").input_ids
# EOS
im_end = tokenizer("<|im_end|>").input_ids
# PAD
IGNORE_TOKEN_ID = tokenizer.pad_token_id
# Newline
nl_tokens = tokenizer('\n').input_ids
# Role identifiers
_system = tokenizer('system').input_ids + nl_tokens
_user = tokenizer('human').input_ids + nl_tokens
_assistant = tokenizer('assistant').input_ids + nl_tokens
```

The Chat Template in the Qwen series generally has three dialogue characters: System, User and Assistant. System is the system prompt, responsible for activating the model's capabilities. It defaults to "You are a helpful assistant." and is generally not changed during the SFT process. User is the prompt given by the user. Here, since the dialogue role in the dataset is "human", we modify "user" to "human". Assistant is the reply given by LLM, that is, the text that the model needs to fit during the SFT process.

Next, since this dataset is a multi-turn conversation dataset, we need to concatenate the turns of each conversation into a single text sequence:

```python
# Concatenate multi-turn conversations
input_ids, targets = [], []
# Multiple samples
for i in tqdm(range(len(sources))):
    # source is one multi-turn conversation sample
    source = sources[i]
    # Start from the user turn
    if source[0]["from"] != "human":
        source = source[1:]
    # Input and output, respectively
    input_id, target = [], []
    # system: [BOS]system\nYou are a helpful assistant.[EOS]\n
    system = im_start + _system + tokenizer(system_message).input_ids + im_end + nl_tokens
    input_id += system
    # The system part does not need to be fitted
    target += im_start + [IGNORE_TOKEN_ID] * (len(system)-3) + im_end + nl_tokens
    assert len(input_id) == len(target)
    # Concatenate the turns in order
    for j, sentence in enumerate(source):
        # sentence is one turn of the conversation
        role = roles[sentence["from"]]
        # user: <|im_start|>human\ninstruction[EOS]\n
        # assistant: <|im_start|>assistant\nresponse[EOS]\n
        _input_id = tokenizer(role).input_ids + nl_tokens + \
            tokenizer(sentence["value"]).input_ids + im_end + nl_tokens
        input_id += _input_id
        if role == '<|im_start|>human':
            # The user part does not need to be fitted
            _target = im_start + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + im_end + nl_tokens
        elif role == '<|im_start|>assistant':
            # The assistant part needs to be fitted
            _target = im_start + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                _input_id[len(tokenizer(role).input_ids)+1:-2] + im_end + nl_tokens
        else:
            print(role)
            raise NotImplementedError
        target += _target
    assert len(input_id) == len(target)
    # Finally, apply padding
    input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
    target += [IGNORE_TOKEN_ID] * (max_len - len(target))
    input_ids.append(input_id[:max_len])
    targets.append(target[:max_len])
```
The above code follows Qwen's Chat Template logic, and readers can also modify it according to their own preferences. The core point is that the text of the User does not need to be fitted, so the text content corresponding to the User in targets is blocked using IGNORE_TOKEN_ID, while the text content corresponding to Assistant is kept as the original text and is included in the loss calculation. Currently, the mainstream LLM IGNORE_TOKEN_ID is generally set to -100.

After completing the splicing, convert the tokenize numerical sequence into `Torch.tensor`, and then splice it into the dictionary required for Dataset to return:

```python
input_ids = torch.tensor(input_ids)
targets = torch.tensor(targets)

return dict(
    input_ids=input_ids,
    labels=targets,
    attention_mask=input_ids.ne(tokenizer.pad_token_id),
)
```

After completing the above processing logic, you need to customize a Dataset class and call the logic in this class to process the data:

```python
class SupervisedDataset(Dataset):

    def __init__(self, raw_data, tokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()
        # Load and preprocess the data
        sources = [example["conversations"] for example in raw_data]
        # preprocess is the data preprocessing logic defined above
        data_dict = preprocess(sources, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )
```

This class inherits from Torch's Dataset class and can be used directly in Trainer. After completing the data processing, you can modify the data processing logic based on the previous script, and the subsequent model training is almost exactly the same. Here is the main function logic:

```python
# Load the script arguments
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# Initialize SwanLab
swanlab.init(project="sft", experiment_name="qwen-1.5b")

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Set the log level to INFO
transformers.utils.logging.set_verbosity_info()
log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Log an overview of the training setup
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
logger.info(f"Training/evaluation parameters {training_args}")

# Check for an existing checkpoint
last_checkpoint = None
if os.path.isdir(training_args.output_dir):
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        raise ValueError(
            f"Output directory ({training_args.output_dir}) is not empty "
        )
    elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(
            f"Resuming training from {last_checkpoint}"
        )

# Set the random seed.
set_seed(training_args.seed)

# Initialize the model
logger.warning("Loading pre-trained model")
logger.info(f"Model parameter path: {model_args.model_name_or_path}")
model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path,trust_remote_code=True)
n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
logger.info(f"Inheriting a pre-trained model - Total size={n_params/2**20:.2f}M params")

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
logger.info("Finished loading tokenizer")

# Load the fine-tuning data
with open(data_args.train_files) as f:
    lst = [json.loads(line) for line in f.readlines()[:10000]]
logger.info("Finished loading training set")
logger.info(f"Training set path: {data_args.train_files}")
logger.info(f'Total training samples: {len(lst)}')
# logger.info(f"Training set sample: {ds["train"][0]}")

train_dataset = SupervisedDataset(lst, tokenizer=tokenizer, max_len=2048)

logger.info("Initializing Trainer")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= IterableWrapper(train_dataset),
    tokenizer=tokenizer
)

# Load from checkpoint
checkpoint = None
if training_args.resume_from_checkpoint is not None:
    checkpoint = training_args.resume_from_checkpoint
elif last_checkpoint is not None:
        checkpoint = last_checkpoint

logger.info("Starting training")
train_result = trainer.train(resume_from_checkpoint=checkpoint)
trainer.save_model() 
```

The startup method is also started using deepspeed in the sh script. I will not repeat it here. See ./code/finetune.sh in the source code.

## 6.3 Efficient fine-tuning

In the previous sections, we introduce the principles and practice details of Pretraining, SFT and RLHF of the model based on the Transformers framework. However, because the LLM parameters are large and the training data is large, training the model through the above method (mainly SFT and RLHF) requires adjusting all the parameters of the model, and the resource pressure is very high. For enterprises or research groups with limited resources, it is very important to efficiently and quickly fine-tune the domain or task of the model and use LLM to complete the target tasks at a low cost.

### 6.3.1 Efficient fine-tuning solution

There are currently two main solutions to the expensive problem of full-scale fine-tuning:

**Adapt Tuning**. That is, add an Adapter layer to the model, freeze the original parameters during fine-tuning, and only update the Adapter layer.

Specifically, it inserts parameters for downstream tasks in each layer of the pretrained model, i.e. the Adapter module, freezes the model body during fine-tuning, and trains only task-specific parameters, as shown in Figure 6.8.

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/6-images/3-1.png" alt="alt text" width="90%">
    <p>Figure 6.8 Adapt Tuning</p>
</div>

Each Adapter module consists of two feedforward sublayers. The first feedforward sublayer takes the output of the Transformer block as input, projecting the original input dimension $d$ to $m$, limiting the amount of parameters of the Adapter module by controlling the size of $m$, usually $m << d$. In the output stage, the input dimension is restored through the second feedforward sublayer, and $m$ is reprojected to $d$ as the output of the Adapter module (as shown in the structure on the right of the figure above).

LoRA is actually an improved Adapt Tuning method. However, the Adapt Tuning method has a problem of inference delay. Due to the addition of additional parameters and additional calculations, the calculation speed of the model after fine-tuning is slower than that of the original pretrained model.

**Prefix Tuning**. This method freezes the pre-trained LM and adds trainable, task-specific prefixes to it, so that different prefixes can be saved for different tasks, and the fine-tuning cost is also small. Specifically, before each input token, a sequence of task-related virtual tokens is constructed as the prefix, and only the parameters of the prefix part are updated during fine-tuning, while other parameters are frozen unchanged.

P-tuning, another commonly used lightweight fine-tuning method, is actually an improvement of Prefix Tuning. But Prefix Tuning also has a fixed drawback: the model has reduced available sequence length. Due to the addition of virtual tokens, the available sequence length is occupied, so the higher the fine-tuning quality, the lower the available sequence length of the model.

### 6.3.2 LoRA fine tune

If a large language model maps data to a high-dimensional space for processing, we assume that when handling a small, specific task, such a complex model is not needed, and may only need to be solved within a certain subspace range, then there is no need to optimize the full-scale parameters. We can define that when a certain subspace parameter is optimized, it can reach a certain level of the performance of full-parameter optimization (such as 90% accuracy), then the rank of this subspace parameter matrix can be called the intrinsic rank corresponding to the currently to be solved problem.

The pre-trained model itself implicitly lowers the intrinsic rank. When fine-tuning is performed for a specific task, the weight matrix in the model actually has a lower intrinsic rank. At the same time, the simpler the downstream task, the lower the corresponding intrinsic rank. ([Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255)) Therefore, the part of the parameter matrix with the weight update can still be effectively learned despite randomly projecting to a smaller subspace, which can be understood as these weight matrices do not require full rank for specific downstream tasks. We can indirectly train some dense layers in the neural network by optimizing the rank decomposition matrix of the dense layer that changes during the adaptation process, so as to achieve fine-tuning effect by optimizing the rank decomposition matrix of the dense layer only.

For example, suppose that the pretraining parameter is $\theta^D_0$, the intrinsic rank corresponding to the dense layer weight parameter matrix on a specific downstream task is $\theta^d$, and the fine-tuning parameter for a specific downstream task is $\theta^D$, then there is:

$$\theta^D = \theta^D_0 + \theta^d M$$

This $M$ is the LoRA-optimized rank decomposition matrix.

Compared with other efficient fine-tuning methods, LoRA has the following advantages:

1. Small LoRA modules can be built for different downstream tasks, so as to effectively switch downstream tasks based on shared pre-trained model parameters.
2. LoRA uses an Adaptive Optimizer, which does not require calculating gradients or maintaining the optimizer state of most parameters, making training more efficient and has a lower hardware threshold.
3. LoRA uses a simple linear design to merge the trainable matrix with the frozen weights during deployment, without inference delay.
4. LoRA is orthogonal to other methods and can be combined.

Therefore, LoRA has become the mainstream method for efficient fine-tuning of LLM at present. Especially when resources are limited and supervised training data are limited, LoRA fine-tuning is often the preferred method for fine-tuning of LLM.

### 6.3.3 The principle of LoRA fine-tuning

#### (1) Low-rank parameterized update matrix

LoRA assumes that the weight update also has a low intrinsic rank. For the pre-trained weight parameter matrix $W0 \in R^{d \times k}$ ($d$ is the previous layer output dimension and $k$ is the next layer input dimension), a low-rank decomposition is used to represent its update:

$$W_0 + {\Delta}W = W_0 + BA \space\space  where \space B \in R^{d \times r}, A \in R^{r \times k}$$

During training, $W_0$ is frozen and does not update, and $A$ and $B$ contain trainable parameters.

Therefore, the forward pass of LoRA is:

$$h = W_0 x + \Delta W x = W_0 x + B A x$$

At the start of training, use a random Gaussian initialization for $A$, zero initialization for $B$, and then optimize with Adam.

The training ideas are shown in Figure 6.9:

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/6-images/3-2.jpg" alt="alt text" width="90%">
    <p>Figure 6.9 LoRA</p>
</div>

#### (2) Applied to Transformer

In the Transformer structure, LoRA technology is mainly used in the four weight matrices of the attention module: $W_q$, $W_k$, $W_v$, $W_0$, and freezes the weight matrix of MLP.

Through ablation experiments, it was found that simultaneous adjustment of $W_q$ and $W_v$ would produce the best results.

Under the above conditions, the number of training parameters is:

$$\Theta = 2 \times L_{LoRA} \times d_{model} \times r$$

Where $L_{LoRA}$ is the number of weight matrices to which LoRA is applied, $d_{model}$ is the input and output dimension of Transformer, and $r$ is the set LoRA rank.

Generally, r is set to 4, 8, or 16.

### 6.3.4 LoRA code implementation

Currently, the model LoRA fine-tuning is generally achieved through the peft library. The peft library is a third-party library developed by huggingface, which encapsulates a variety of efficient fine-tuning methods including LoRA, Adapt Tuning, P-tuning, etc., which can easily implement LoRA fine-tuning of the model based on this.

This section briefly walks through the LoRA fine-tuning code in the peft library to show how LoRA fine-tuning is implemented.

#### (1) Implementation process

The internal implementation process of LoRA fine-tuning mainly includes the following steps:

1. Determine the layer to use LoRA. The layer types that the peft library currently supports for LoRA are nn.Linear, nn.Embedding, and nn.Conv2d.

2. Replace each layer that should use LoRA with a LoRA layer. The so-called LoRA layer actually adds a bypass to the original result of this layer, and simulates parameter updates through low-rank decomposition (i.e., matrix $A$ and matrix $B$).

3. Freeze the original parameters, perform fine-tuning, and update the LoRA layer parameters.

#### (2) Determine the LoRA layer

When performing LoRA fine-tuning, you first need to determine the LoRA fine-tuning parameters, one of the important parameters is target_modules. target_modules is generally a list of strings, each string is the layer name that requires LoRA, for example:

```python
target_modules = ["q_proj","v_proj"]
```

Here, q_proj is $W_q$ in the attention mechanism, and v_proj is $W_v$ in the attention mechanism. We can customize the layers that require LoRA operations based on model architecture and task requirements.

When creating a LoRA model, this parameter is obtained and the corresponding layer is found in the original model. This operation is mainly achieved by using re to perform regular matching of the layer name:

```python
# Find the model components whose names contain "q_proj" or "v_proj"
target_module_found = re.fullmatch(self.peft_config.target_modules, key)
# Here, key is the name of a model component
```

#### (3) Replace the LoRA layer

For each target layer found, a new LoRA layer is created to replace it.

In terms of specific implementation, the LoRA layer defines a Linear class based on the Lora base class, which inherits both nn.Linear and LoraLayer. LoraLayer is the base class of Lora, which mainly sets up the various hyperparameters of LoRA:

```python
class LoraLayer:
    def __init__(
        self,
        r: int, # LoRA rank
        lora_alpha: int, # Scaling (normalization) parameter
        lora_dropout: float, # Dropout rate of the LoRA layer
        merge_weights: bool, # In eval mode, whether to add the LoRA matrices to the original weight matrix
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False

```
nn.Linear is the linear layer implementation of Pytorch. The Linear class is a specific LoRA layer, and its main implementation is as follows:

```python
class Linear(nn.Linear, LoraLayer):
    # LoRA layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False, 
        merge_weights: bool = True,
        **kwargs,
    ):
        # Call the constructors of both base classes
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            # Parameter matrix A
            self.lora_A = nn.Linear(in_features, r, bias=False)
            # Parameter matrix B
            self.lora_B = nn.Linear(r, out_features, bias=False)
            # Scaling coefficient
            self.scaling = self.lora_alpha / self.r
            # Freeze the original parameters; only update A and B
            self.weight.requires_grad = False
        # Initialize A and B
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

```

When replacing, directly copy the weight and bias of the original layer to the new LoRA layer, and then allocate the new LoRA layer to the specified device.

#### (4) Training

After the replacement of the LoRA layer is implemented, fine-tuning training is performed. Since the original parameters have been frozen in the LoRA layer, only the parameters of A and B will be updated during training, thus achieving efficient fine-tuning. The overall process of training is similar to the original Fine-tune, and will not be described here. Due to the LoRA method, the forward function will also be adjusted accordingly:

```python
    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            if self.r > 0 and self.merged:
                self.weight.data -= (
                    transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.scaling
                )
                self.merged = False

            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        '''Main branch'''
        elif self.r > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            if self.r > 0:
                result += self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
            return result
        else:
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

```
Since the above code considers the parameter merging problem, there are several branches. Here we only need to read the second branch, namely the elif branch. The forward calculation process based on LoRA is shown in the previous formula. First, calculate the product of the original parameter and the input, and then add the product of A and B and the input respectively.

### 6.3.5 Implementing LoRA fine-tuning using peft

peft is well packaged, allowing us to easily and efficiently fine-tune large language models (LLMs). Here, take the LLM SFT in Section 2 as an example, and briefly introduce how to fine-tune an LLM using peft. If it is applied to RLHF, the overall idea is the same.

First load the required library:

```python
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import Trainer
```

Secondly, the original model is loaded with the original tokenizer, which is consistent with the second section:

```python
# Load the base model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_PATH, trust_remote_code=True
)
```

Next, set the peft parameters:

```python
peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
```

Note that LoRA parameters may differ for different models. For example, for ChatGLM, peft can be found by itself without specifying target_modeules; for BaiChuan, it needs to be specified manually. task_type is the task type of the model, and LLMs are generally CAUSAL_LM, that is, traditional (causal) language models.

Then get the LoRA model:

```python
model = get_peft_model(model, peft_config)
```

The underlying operation of get_peft_model here is the specific implementation analyzed above.

Finally, use the Trainer provided by transformers for training, and the GPU memory occupied by training will be greatly reduced:

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= IterableWrapper(train_dataset),
    tokenizer=tokenizer
)
trainer.train()
```

If it is applied to DPO or KTO, then LoRA parameters are added in the same way and a LoRA model is obtained through `get_peft_model`, and no modification is required for the others. However, it should be noted that LoRA fine-tuning can greatly reduce GPU memory usage and can achieve better results in downstream tasks adaptation. However, if it is a task that requires learning the corresponding knowledge, LoRA is difficult to inject knowledge because it only adjusts the low-rank matrix and is generally poor in effect. Therefore, it is not recommended to use LoRA for model pre-training or post-training.

**References**

[1] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. (2019). *Parameter-Efficient Transfer Learning for NLP.* arXiv preprint arXiv:1902.00751.

[2] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv preprint arXiv:2106.09685.

[3] Armen Aghajanyan, Luke Zettlemoyer, and Sonal Gupta. (2020). *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.* arXiv preprint arXiv:2012.13255.

[4] Xiang Lisa Li and Percy Liang. (2021). *Prefix-Tuning: Optimizing Continuous Prompts for Generation.* arXiv preprint arXiv:2101.00190.

## Part 2: Preference Alignment (6.4 WIP)

# 6.4 Preference Alignment through Reinforcement Learning

Before we get into the details of reinforcement learning, let’s take a look at its origins. Reinforcement Learning (RL) is not a new thing. Its theoretical basis can be traced back to behavioral psychology in the early 20th century, especially Edward Thorndike and B.F. Skinner's research on animal learning. Thorndike proposed the "law of effect", that is, if a behavior brings positive results, the probability of such behavior repeating will increase. Skinner further developed this idea, put forward the theory of operant conditioning, and shapes behavior through rewards and punishments.

Reinforcement learning in computer science grew out of these psychological principles. In the 1980s, with the improvement of computing power and the development of mathematical theory, people began to try to apply these biopsychological learning concepts to machines and computer programs, thus developing reinforcement learning in the modern sense.

## 6.4.1 Basic principles of reinforcement learning

Now, we enter the core part - the basic principles of reinforcement learning.

- State: This is the specific situation of a system at a certain moment. For example, in a board game, the state can represent the current arrangement of all chess pieces on the board. For an autonomous vehicle, the state may include the speed, position of the car, and the position of surrounding obstacles.
- Action: Action is an operation that an agent can perform in a given state. Taking a bicycle as an example, actions may include moving forward, stopping, turning, etc. In a complex system, the action set can be very large.
- Reward: This is the feedback obtained by the agent after performing a certain action, usually a numerical value. The reward can be immediate or delayed. A good move may receive a positive reward, while a bad move may receive a negative reward.
- Policy: A policy is a set of rules that guide an agent how to choose actions. Simply put, the policy tells the agent what to do in each state.
- Value Function: This is a policy evaluation tool designed to predict the total rewards that can be obtained in the long run based on the current state. Value functions help agents not only consider the rewards of the current step, but also better weigh short-term and long-term benefits.
- Model: In some reinforcement learning systems, we will build an environmental model to help the agent foresee the results of its actions. This is very useful in many complex computing situations.

![Reinforcement Learning](./images/7.1-1.png)

These elements work together to help agents learn the best action policy through continual trial and error in a virtual environment. In reinforcement learning, agents are the subject of learning and decision-making. It interacts with the environment through the following steps:

1. Observe the state: The agent first observes the current state (State).
2. Select Action: The agent selects an Action based on the observed state and the predetermined policy.
3. Execute Action: The agent performs the selected action.
4. Receive rewards and new states: After performing the action, the agent receives the corresponding rewards (Rewards) and the updated new state (State) from the environment.
5. Update Policy: The agent uses the reward information obtained to adjust the policy to achieve better results in the future.

This process is repeated continuously, and the agent continuously optimizes its policy in repeated interactions, with the goal of making it perform better and better in a given task.

## 6.4.2 Objectives of Reinforcement Learning

The goal of reinforcement learning is very clear: *** By repeatedly testing and learning in a given environment, the agent can select a series of actions to maximize its total cumulative reward. *** This may sound a bit abstract, and we can use playing games as a metaphor. In the game, the player's goal is to win high scores or complete levels through a series of actions (such as walking, jumping, and fighting monsters). In reinforcement learning, this concept of high scores or successful passing levels corresponds to "maximize rewards."

Mathematically, this goal can be expressed as training a policy $\pi$ so that in all states $s$, the agent's selected actions can maximize the expected value of return $R(\tau)$. Specifically, we want to maximize the following expectations:

$$
E(R(\tau))_{\tau \sim P_{\theta}(\tau)} = \sum_{\tau} R(\tau) P_{\theta}(\tau)
$$

where:
- $E(R(\tau))_{\tau \sim P_{\theta}(\tau)}$: represents the expected value of the return $R(\tau)$ of the trajectory $\tau$ under the policy $P_{\theta}(\tau)$.
- $R(\tau)$: The return of the trajectory $\tau$, that is, the sum of all rewards obtained from the start state to the end state.
- $\tau$: represents a trajectory, that is, the state and action sequence of the agent in the environment.
- $P_{\theta}(\tau)$: The probability of generating a trajectory $\tau$ under the parameter $\theta$, usually determined by the policy or policy network.
- $\theta$: The parameter of the policy, which controls the behavior of the policy $P_{\theta}$.

In order to find this policy, we use gradient ascent to continuously update the policy parameter $\theta$, so that $E(R(\tau))_{\tau \sim P_{\theta}(\tau)}$ continues to increase.

This learning method is very effective because it does not rely on a large amount of labeled data, but learns through direct interaction and feedback on the environment. This makes reinforcement learning show great potential in many complex tasks that require adaptation and decision-making, such as robot control, autonomous driving, financial transactions and even games.

The application of reinforcement learning in large-scale models, such as AlphaGo, AlphaZero, etc., has allowed people to see the powerful ability of reinforcement learning in complex tasks. Through reinforcement learning methods, these models continuously optimize their policies, and ultimately defeated top human players in games such as Go and chess, showing the huge potential of reinforcement learning in complex tasks.

Reinforcement learning can also be used for preference alignment problems, such as allowing LLMs to learn to imitate the way humans communicate, and also used in fields such as autonomous driving. The application fields of reinforcement learning are very wide, and there will be more application scenarios in the future.

## 6.4.3 Reward Model

In the field of natural language processing, large language models (such as the Llama series, Qwen series, etc.) have shown strong text understanding and generation capabilities. However, these pre-trained models do not always directly meet specific business needs and human values. To this end, people usually need to perform "instruction tuning" on the pre-trained model, that is, provide the model with specific instructions (prompts) and examples, so that it behaves more in line with human expectations in tasks such as dialogue, question-and-answer, text generation, etc.

After completing the preliminary instruction fine-tuning, we also want to make the model's answers not only correct, but also meet human aesthetics, values and safety standards to the greatest extent. To this end, the concept of Reinforcement Learning from Human Feedback (RLHF) has been introduced. In RLHF, we first obtain preferences for model answers from human annotators (for example, giving multiple model answers to let human annotators rank them), and then use these feedback to guide model learning, thereby continuously improving the fit between model generation content and human preferences.

In order to automatically "score" the model's answers in the RLHF process (give rewards), we need to build a special reward model (Reward Model). This reward model is trained based on human-labeled data and automatically scores the model output independently in actual deployment, thereby reducing the cost and delay of ongoing manual participation.

## 6.4.4 Dataset Construction

Before building a Reward Model, we first need to prepare a high-quality human feedback dataset. The core goal of this dataset is to provide multiple candidate answers (completions) for each given prompt, and human annotators carefully evaluate and rank these answers. By comparing and filtering the answers, we are able to provide clear reference standards for machine models to help them further learn how to generate output that is more in line with human expectations under a given task.

Data collection can be performed as follows:

1. Collect initial answers: First, we need to generate multiple answers for a set of carefully designed prompts from an LLM that has already undergone basic fine-tuning (often a pre-trained model with certain instruction understanding and generation capabilities). These answers will serve as the basis for subsequent human annotation work.


2. Manual annotation and evaluation: After having multiple candidate answers, we invite professional annotation personnel or crowdsourcing annotators to evaluate the quality of each answer. These assessments are usually based on a range of pre-designed evaluation criteria such as the accuracy, completeness, contextual relevance, language fluency, and compliance with ethical and safety guidelines. Comparing and ranking different answers helps us identify the best and worst answers, thus forming valuable training data.

3. Data formatting and organization: After the annotation is completed, we organize and format the data, usually using JSON, CSV or other structured data formats that are convenient for computer processing. The dataset needs to clearly identify each question (prompt), its corresponding multiple answers (completions), and the choice of these answers by human annotators (such as the best answer marked "chosen" and the worse answers "rejected"). These tagged information can be used directly as a supervisory signal for reward model learning, making it automatically tend to generate high-quality answers during training.

Below is a simple data example showing two questions and their corresponding answers and human evaluation results. By comparing the "chosen" and "rejected" fields, we can intuitively see which answer is better.

```json
[
    {
        "question": "What is a list in Python?",
        "chosen": "A list in Python is an ordered, mutable container that can store multiple elements and supports access by index.",
        "rejected": "A list in Python is used to store data."
    },
    {
        "question": "What is a tuple in Python?",
        "chosen": "A tuple in Python is an ordered, immutable container that can store multiple elements and cannot be modified once created.",
        "rejected": "A tuple in Python is used to store data."
    }
]
```

In the above example, human annotators believe that the answers under the "chosen" field are better than the corresponding "rejected" answers in terms of description, accuracy, and information volume. For example, for the definition of a list, the "chosen" reply more clearly explains the characteristics of the list (orderly, variable, and supports index access), rather than just staying in the general description of "used to store data".


## 6.4.5 Reward Model Training

We can use the LLM reinforcement learning framework TRL (Transformer Reinforcement Learning) to train reward models. TRL is a training framework based on reinforcement learning, aiming to generate answers that are more in line with human expectations through human feedback guidance models. In TRL, we use the reward model as a separate component to evaluate the responses generated by the model and give rewards or penalties based on the evaluation results.



## Part 3: Supplementary Readme

# Chapter 6 LLM Training Based on Transformers

Note: The core content of this chapter is to implement LLM pre-training and fine-tuning based on the transformers framework

1. Framework brief description:
   1. transformers
   2. deepspeed
   3. peft
   4. wandb
   5. tokenizers
2. LLM pre-training based on transformers
   1. Tokenizer training
   2. Dataset construction
   3. Model building/inheriting pre-trained models
   4. Construct Trainer for training
3. LLM SFT/downstream task fine-tuning based on transformers
   1. Tokenizer training
   2. Dataset construction
   3. LoRA configuration
   4. Inherit the pre-trained model
   5. Construct Trainer for training