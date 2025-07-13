import torch
import warnings
from transformers import AutoTokenizer
from k_model import Transformer, ModelConfig

warnings.filterwarnings('ignore', category=UserWarning)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def export_model(tokenizer_path, model_config, model_ckpt_path, save_directory):
    # Register custom classes and configurations
    ModelConfig.register_for_auto_class()
    Transformer.register_for_auto_class("AutoModelForCausalLM")

    # Initialize the model
    model = Transformer(model_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading model weights
    state_dict = torch.load(model_ckpt_path, map_location=device)
    # Remove excess prefixes that may exist
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    # Loading weights to model
    model.load_state_dict(state_dict, strict=False)
    print(f'Model parameters: {count_parameters(model)/1e6:.2f}M = {count_parameters(model)/1e9:.2f}B')

    # Loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False
    )

    # Save the complete model and tokenizer
    model.save_pretrained(save_directory, safe_serialization=False)
    tokenizer.save_pretrained(save_directory)
    print(f'The model and tokenizer have been saved to: {save_directory}')


if __name__ == '__main__':
    # Example usage
    config = ModelConfig(
        dim=1024,
        n_layers=18,
    )

    export_model(
        tokenizer_path='./tokenizer_k/',
        model_config=config,
        model_ckpt_path='./BeelGroup_sft_model_215M/sft_dim1024_layers18_vocab_size6144.pth',
        save_directory="k-model-215M"
    )