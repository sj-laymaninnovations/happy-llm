import os
import json
from tqdm import tqdm

# Download the pre-training dataset
# os.system("modelscope download --dataset ddzhu123/seq-monkey mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2 --local_dir ./autodl-tmp/dataset/pretrain_data")
# # Unzip the pre-trained dataset
# os.system("tar -xvf ./autodl-tmp/dataset/pretrain_data/mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2")

# Set environment variables
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# Download SFT dataset
os.system(f'huggingface-cli download --repo-type dataset --resume-download BelleGroup/train_3.5M_CN --local-dir ./autodl-tmp/dataset/sft_data/BelleGroup')
