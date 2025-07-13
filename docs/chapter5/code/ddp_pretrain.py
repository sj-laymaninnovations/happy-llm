# -*- coding: utf-8 -*-
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
from dataset import PretrainDataset

import swanlab

# Ignore warning messages
warnings.filterwarnings('ignore')


def Logger(content):
    """Simple logging function
    
    Args:
        content (str): What to print"""
    print(content)

def get_lr(it, all):
    """Calculate the learning rate of the current iteration and use cosine annealing scheduling strategy
    
    Learning rate scheduling strategy:
    1. Warmup stage: The learning rate grows linearly from 0 to the target learning rate
    2. Cosine annealing stage: The learning rate decays to the minimum learning rate according to the cosine function
    3. After the training steps exceeds: maintain the minimum learning rate
    
    Args:
        it (int): current iteration steps
        all (int): Total iteration steps
        
    Returns:
        float: The learning rate corresponding to the current number of steps"""
    warmup_iters = args.warmup_iters  # Number of preheating iterations
    lr_decay_iters = all  # Total iterations of learning rate decay
    min_lr = args.learning_rate / 10  # Minimum learning rate, 1/10 of the initial learning rate

    # Warmup stage: linear growth
    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    
    # Exceed training steps: Maintain minimum learning rate
    if it > lr_decay_iters:
        return min_lr
    
    # Cosine annealing stage
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Cosine coefficient
    return min_lr + coeff * (args.learning_rate - min_lr)

def train_epoch(epoch):
    """Training an epoch function
    
    A complete training cycle is implemented, including:
    1. Data loading and device transfer
    2. Dynamic learning rate adjustment
    3. Forward propagation and loss calculation
    4. Gradient accumulation and backpropagation
    5. Gradient cropping and optimizer updates
    6. Logging and model saving
    
    Args:
        epoch (int): Current epoch number"""
    start_time = time.time()  # Record the start time
    
    # Iterate through each batch in the data loader
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # Transfer data to the specified device (GPU/CPU)
        X = X.to(args.device)  # Input sequence
        Y = Y.to(args.device)  # Target sequence
        loss_mask = loss_mask.to(args.device)  # Loss mask, used to ignore padding tokens

        # Calculate the learning rate of the current step
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        # Update the learning rate of all parameter groups in the optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Training context using mixed precision
        with ctx:
            # Forward communication
            out = model(X, Y)
            # Calculate the loss and divide by the cumulative number of steps (for gradient accumulation)
            loss = out.last_loss / args.accumulation_steps
            # Flatten loss_mask into one dimension
            loss_mask = loss_mask.view(-1)
            # Apply mask to calculate the effective loss (ignoring padding position)
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()

        # Backpropagation of mixed precision using scaler
        scaler.scale(loss).backward()

        # Perform an optimizer update every accumulation_steps step
        if (step + 1) % args.accumulation_steps == 0:
            # Cancel gradient scaling, prepare for gradient clipping
            scaler.unscale_(optimizer)
            # Gradient cropping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # Perform the optimizer steps
            scaler.step(optimizer)
            # Update scaler's scaler's scaling factor
            scaler.update()

            # Clear the gradient, set_to_none=True can save memory
            optimizer.zero_grad(set_to_none=True)

        # Logs are recorded every log_interval step
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            # Print training progress information
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min;'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,  # Restore the real loss value
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))
            
            # If SwanLab is enabled, record training metrics
            if args.use_swanlab:
                swanlab.log({
                    "loss": loss.item() * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]['lr']
                })

        # Save the model once every save_interval step
        if (step + 1) % args.save_interval == 0:
            model.eval()  # Switch to evaluation mode
            # Build checkpoint file name
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}_{lm_config.n_layers}_{lm_config.vocab_size}.pth'

            # Processing multiple card saving: If it is a DataParallel model, you need to access the .module property
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()  # Switch back to training mode
        
        # Save a checkpoint with step marker every 20,000 steps
        if (step + 1) % 20000 == 0:
            model.eval()
            # Build a checkpoint file name with steps
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}_{lm_config.n_layers}_{lm_config.vocab_size}_step{step+1}.pth'

            # Save Model Status Dictionary
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()


def init_model():
    """Initialize the model and word participle
    
    Functions include:
    1. Load the pre-trained word segmenter
    2. Create a Transformer model
    3. Set up multi-GPU parallel training (if available)
    4. Move the model to the specified device
    5. Statistics and prints the model parameter quantity
    
    Returns:
        tuple: (model, tokenizer) Initialized model and word participle"""
    def count_parameters(model):
        """Statistics the number of trainable parameters in the model
        
        Args:
            model: PyTorch model
            
        Returns:
            int: Total number of trainable parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Loading pretrained word segmenter from local path
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer_k/')

    # Create a Transformer model based on configuration
    model = Transformer(lm_config)
    
    # Multi-card initialization: Check the number of available GPUs and set DataParallel
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        Logger(f"Using {num_gpus} GPUs with DataParallel!")
        # Packaging models with DataParallel to support multi-GPU training
        model = torch.nn.DataParallel(model)
    
    # Move the model to the specified device (GPU or CPU)
    model = model.to(args.device)
    
    # Calculate and print the number of model parameters in millions
    Logger(f'Total LLM parameter quantity: {count_parameters(model) / 1e6:.3f} million')
    return model, tokenizer


if __name__ == "__main__":
    # ====================================================
    parser = argparse.ArgumentParser(description="Tiny-LLM Pretraining")
    
    # Basic training parameters
    parser.add_argument("--out_dir", type=str, default="base_model_215M", help="Model output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training rounds")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Training equipment")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data Type")
    
    # Experiment tracking and data loading parameters
    parser.add_argument("--use_swanlab", action="store_true", help="Whether to use SwanLab for experimental tracking")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes for data loading")
    parser.add_argument("--data_path", type=str, default="./seq_monkey_datawhale.jsonl", help="Training data path")
    
    # Training optimization parameters
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient crop threshold")
    parser.add_argument("--warmup_iters", type=int, default=0, help="Learning rate preheating iterations")
    
    # Log and save parameters
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Model saving interval")
    
    # Multi-GPU training parameters
    parser.add_argument("--gpus", type=str, default='0,1,2,3,4,5,6,7', help="The GPU ID used, separated by commas (for example: '0,1,2')")

    args = parser.parse_args()

    # ========================== GPU environment settings =====================
    # Setting up visible GPU devices
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        # Automatically set the master device to the first available GPU
        if torch.cuda.is_available():
            args.device = "cuda:0"
        else:
            args.device = "cpu"

    # ======================= Experiment tracking initialization ======================
    if args.use_swanlab:
        # Note: You need to log in to swanlab.login(api_key='your key')
        run = swanlab.init(
            project="Happy-LLM",  # Project name
            experiment_name="Pretrain-215M",  # Experiment name
            config=args,  # Save all hyperparameters
        )

    # ========================= Model Configuration =======================
    # Define the configuration parameters of the language model
    lm_config = ModelConfig(
        dim=1024,      # Model Dimension
        n_layers=18,   # Transformer layers
    )

    # ====================================================
    max_seq_len = lm_config.max_seq_len  # Maximum sequence length
    args.save_dir = os.path.join(args.out_dir)  # Model save directory
    
    # Create the necessary directory
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Set random seeds to ensure results are reproducible
    torch.manual_seed(42)
    
    # Determine the device type (used to select the appropriate context manager)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # Set up a context manager for hybrid precision training
    # Use nullcontext when training CPU, and use autocast when training GPU
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # ====================== Model and data initialization =======================
    # Initialize the model and word participle
    model, tokenizer = init_model()
    
    # Create a training dataset
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=max_seq_len)
    
    # Create a data loader
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,  # Batch size
        pin_memory=True,             # Load data into fixed memory to accelerate GPU transmission
        drop_last=False,             # Don't discard the last incomplete batch
        shuffle=True,                # Randomly disrupt data
        num_workers=args.num_workers # Number of parallel worker processes for data loading
    )

    # =========================== Optimizer and training component initialization ====================
    # Initialize a gradient scaler for hybrid precision training
    # Only enabled when using float16 or bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    
    # Initialize the Adam Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # ==================================================
    # Calculate the number of iterations for each epoch
    iter_per_epoch = len(train_loader)
    
    # Start the training cycle
    for epoch in range(args.epochs):
        train_epoch(epoch)