import os

# Set environment variables
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Download the model
os.system('huggingface-cli download --resume-download Qwen/Qwen2.5-1.5B --local-dir autodl-tmp/qwen-1.5b')