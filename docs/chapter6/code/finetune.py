'''SFT scripts'''

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from torchdata.datapipes.iter import IterableWrapper
from itertools import chain
import deepspeed
from typing import Optional,List,Dict
from torch.utils.data import Dataset
import json


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
from tqdm import tqdm


logger = logging.getLogger(__name__)


# Super ginseng
@dataclass
class ModelArguments:
    """About the parameters of the model"""

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Pre-trained model parameter address"
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Data type used for model training, recommended bfloat16"
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )


@dataclass
class DataTrainingArguments:
    """About training parameters"""

    train_files: Optional[str]  = field(default=None, metadata={"help": "Training data path"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Maximum text block length"
            )
        },
    )

# Instruction text processing
# Reference: https://github.com/QwenLM/Qwen/blob/main/finetune.py
def preprocess(sources, tokenizer, max_len, system_message: str = "You are a helpful assistant."):
    # prompt template
    roles = {"human": "<|im_start|>human", "assistant": "<|im_start|>assistant"}

    # Different tokenizers need to be specifically defined
    # BOS
    im_start = tokenizer("<|im_start|>").input_ids
    # EOS
    im_end = tokenizer("<|im_end|>").input_ids
    # PAD
    IGNORE_TOKEN_ID = tokenizer.pad_token_id
    # Line breaks
    nl_tokens = tokenizer('\n').input_ids
    # Role Identifier
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('human').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Splicing multiple rounds of dialogue
    input_ids, targets = [], []
    for i in tqdm(range(len(sources))):
        source = sources[i]
        # Start with user
        if source[0]["from"] != "human":
            source = source[1:]
        # Input and output
        input_id, target = [], []
        # system: 【BOS】system\nYou are a helpful assistant.【EOS】\n
        system = im_start + _system + tokenizer(system_message).input_ids + im_end + nl_tokens
        input_id += system
        # system does not require fitting
        target += im_start + [IGNORE_TOKEN_ID] * (len(system)-3) + im_end + nl_tokens
        assert len(input_id) == len(target)
        # Splicing in sequence
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            # user：<|im_start|>human\ninstruction【EOS】\n
            # assistant：<|im_start|>assistant\nresponse【EOS】\n
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + im_end + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>human':
                # user does not require fitting
                _target = im_start + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + im_end + nl_tokens
            elif role == '<|im_start|>assistant':
                # assistant requires fit
                _target = im_start + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + im_end + nl_tokens
            else:
                print(role)
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        # Finally, PAD
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    # print(input_ids)
    input_ids = torch.tensor(input_ids)
    targets = torch.tensor(targets)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )
# Customize a Dataset
from typing import Dict

class SupervisedDataset(Dataset):

    def __init__(self, raw_data, tokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()
        # Load and preprocess data
        sources = [example["conversations"] for example in raw_data]
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

                
def main():

    # Load script parameters
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Initialize SwanLab
    swanlab.init(project="sft", experiment_name="qwen-1.5b")
    
    # Setting up logs
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

    # Overall training status record
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output path ({training_args.output_dir}) non-empty"
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Recover training from {last_checkpoint}"
            )

    # Set random number seeds.
    set_seed(training_args.seed)

    # Initialize the model
    logger.warning("Loading pretrained models")
    logger.info(f"Model parameter address: {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path,trust_remote_code=True)
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Inherit a pretrained model - Total size={n_params/2**20:.2f}M params")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    logger.info("Complete tokenzier loading")

    # Load fine-tuning data
    with open(data_args.train_files) as f:
        lst = [json.loads(line) for line in f.readlines()[:10000]]
    logger.info("Complete training set loading")
    logger.info(f"Training set address: {data_args.train_files}")
    logger.info(f'Total number of training samples: {len(lst)}')
    # logger.info(f"train set sampling: {ds["train"][0]}")

    train_dataset = SupervisedDataset(lst, tokenizer=tokenizer, max_len=2048)
    
    logger.info("Initialize Trainer")
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

    logger.info("Start training")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model() 

if __name__ == "__main__":
    main()