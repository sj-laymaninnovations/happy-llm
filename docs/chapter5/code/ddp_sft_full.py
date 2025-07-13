import os
import platform
import argparse
import time
import warnings
import math
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext

from transformers import AutoTokenizer

from k_model import ModelConfig, Transformer
from dataset import SFTDataset

import swanlab

# Ignore warnings
warnings.filterwarnings('ignore')


def Logger(content):
    """Logger"""
    print(content)

def get_lr(it, all):
    """Get learning rate"""
    # 1) linear warmup for warmup_iters steps
    # 1) Linear preheating of preheating iteration
    warmup_iters = args.warmup_iters
    lr_decay_iters = all
    min_lr = args.learning_rate / 10

    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    
    # 2) if it > lr_decay_iters, return min learning rate
    # 2) If the number of iterations exceeds the number of learning rate decay iterations, the minimum learning rate will be returned.
    if it > lr_decay_iters:
        return min_lr
    
    # 3) in between, use cosine decay down to min learning rate
    # 3) In between, use cosine attenuation to the minimum learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (args.learning_rate - min_lr)

def train_epoch(epoch):
    """Training an epoch"""
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # Get the learning rate and update the optimizer
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward communication
        with ctx:
            out = model(X, Y)
            loss = out.last_loss / args.accumulation_steps
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()

        # Backpropagation
        scaler.scale(loss).backward()

        # Update weights
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        # Print log
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))
            if args.use_swanlab:
                swanlab.log({
                    "loss": loss.item() * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]['lr']
                })

        # Save the model
        if (step + 1) % args.save_interval == 0:
            model.eval()
            ckp = f'{args.save_dir}/sft_dim{lm_config.dim}_layers{lm_config.n_layers}_vocab_size{lm_config.vocab_size}.pth'

            # Process multiple card saving
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()
        
        # Save the model regularly
        if (step + 1) % 20000 == 0:
            model.eval()
            ckp = f'{args.save_dir}/sft_dim{lm_config.dim}_layers{lm_config.n_layers}_vocab_size{lm_config.vocab_size}_step{step+1}.pth'

            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()


def init_model():
    """Initialize the model"""
    def count_parameters(model):
        """Calculate the model parameter quantity"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Loading word participle
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer_k/')

    # Initialize the model
    model = Transformer(lm_config)

    # Loading pretraining weights
    ckp = './base_model_215M/pretrain_1024_18_6144.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    
    # Multi-card initialization
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        Logger(f"Using {num_gpus} GPUs with DataParallel!")
        model = torch.nn.DataParallel(model)
    
    model = model.to(args.device)
    Logger(f'Total LLM parameter quantity: {count_parameters(model) / 1e6:.3f} million')
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiny-LLM Pretraining")
    parser.add_argument("--out_dir", type=str, default="sft_model_215M", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training rounds")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Equipment used")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data Type")
    parser.add_argument("--use_swanlab", action="store_true", help="Whether to use SwanLab for experimental tracking")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes for data loading")
    parser.add_argument("--data_path", type=str, default="./BelleGroup_sft.jsonl", help="Training data path")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient crop threshold")
    parser.add_argument("--warmup_iters", type=int, default=0, help="Number of preheating iterations")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Model saving interval")
    # Add multiple card parameters
    parser.add_argument("--gpus", type=str, default='0,1,2,3,4,5,6,7', help="Comma-separated GPU ID (e.g. '0,1,2')")

    args = parser.parse_args()

    # Setting up visible GPU
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        # Automatically set the master device to the first GPU
        if torch.cuda.is_available():
            args.device = "cuda:0"
        else:
            args.device = "cpu"

    # Initialize swanlab
    if args.use_swanlab:
        run = swanlab.init(
            project="Happy-LLM",
            experiment_name="SFT-215M",
            config=args,
        )

    # Model configuration
    lm_config = ModelConfig(
        dim=1024,
        n_layers=18,
    )
    max_seq_len = lm_config.max_seq_len
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(42)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # Context Manager
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # Initialize the model and word participle
    model, tokenizer = init_model()
    
    # Create datasets and data loaders
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        num_workers=args.num_workers
    )

    # Scalers and Optimizers
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Start training
    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch)