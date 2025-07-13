'''Pre-training scripts'''

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


logger = logging.getLogger(__name__)


# Super ginseng
@dataclass
class ModelArguments:
    """About the parameters of the model"""

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Post-training, the parameter address of the pre-trained model"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pre-training, Config file address"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pre-training Tokenizer address"}
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

    train_files: Optional[List[str]]  = field(default=None, metadata={"help": "Training data path"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Set text block length"
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "Preprocessing uses threads."},
    )

                
def main():

    # Load script parameters
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Initialize SwanLab
    swanlab.init(project="pretrain", experiment_name="from_scrach")
    
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
    if model_args.config_name is not None:
        # from scrach
        config = AutoConfig.from_pretrained(model_args.config_name)
        logger.warning("You are initializing a model from zero")
        logger.info(f"Model parameter configuration address: {model_args.config_name}")
        logger.info(f"Model parameters: {config}")
        model = AutoModelForCausalLM.from_config(config,trust_remote_code=True)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Pre-train a new model - Total size={n_params/2**20:.2f}M params")
    elif model_args.model_name_or_path is not None:
        logger.warning("You are initializing a pretrained model")
        logger.info(f"Model parameter address: {model_args.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path,trust_remote_code=True)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Inherit a pretrained model - Total size={n_params/2**20:.2f}M params")
    else:
        logger.error("config_name and model_name_or_path cannot be both empty")
        raise ValueError("config_name and model_name_or_path cannot be both empty")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    logger.info("Complete tokenzier loading")
    logger.info(f"tokenzier configuration address: {model_args.tokenizer_name}")

    # Load pretrained data
    ds = load_dataset('json', data_files=data_args.train_files)
    logger.info("Complete training set loading")
    logger.info(f"Training set address: {data_args.train_files}")
    logger.info(f'Total number of training files: {len(ds["train"])}')
    # logger.info(f"train set sampling: {ds["train"][0]}")

    # Text tokenize
    column_names = list(ds["train"].features)
    logger.info('Training set features:', column_names)
    text_column_name = "text" if "text" in column_names else column_names[0]

    # tokenize function
    def tokenize_function(examples):
        output = tokenizer([item for item in examples[text_column_name]])
        return output

    # Only the main process performs data preprocessing
    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = ds.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset"
        )

    # Text diced
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "tokenizer supports context lengths greater than 1K, and the default setting is 1K"
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The set block length is ({data_args.block_size}), which is greater than the context length of the model"
                f"Set the block length to model context length: {tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    def group_texts(examples):
        # Stitch the text segments together
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        # Calculate the overall length
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # If the length is too long, do chunking
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }    
        result["labels"] = result["input_ids"].copy()
        return result

    with training_args.main_process_first(desc="Text chunking"):
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=True,
            desc=f"Text blocked to {block_size}",
            batch_size = 40000,
        )
        logger.info("Complete data preprocessing")
        train_dataset = lm_datasets["train"]
    
    logger.info("Initialize Trainer")
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

    logger.info("Start training")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model() 

if __name__ == "__main__":
    main()