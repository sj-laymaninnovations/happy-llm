import os
import json
from tqdm import tqdm

# pretrain_data is the local path to download pretrain_data when running download_dataset.sh
pretrain_data = 'your local pretrain_data'
output_pretrain_data = 'seq_monkey_datawhale.jsonl'

# sft_data is the local path to download when running download_dataset.sh
sft_data = 'your local sft_data'
output_sft_data = 'BelleGroup_sft.jsonl'

# 1 Process pre-training data
def split_text(text, chunk_size=512):
    """Divide text into chunks by specified length"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

with open(output_pretrain_data, 'a', encoding='utf-8') as pretrain:
    with open(pretrain_data, 'r', encoding='utf-8') as f:
        data = f.readlines()
        for line in tqdm(data, desc=f"Processing lines in {pretrain_data}", leave=False):  # Add a row-level progress bar
            line = json.loads(line)
            text = line['text']
            chunks = split_text(text)
            for chunk in chunks:
                pretrain.write(json.dumps({'text': chunk}, ensure_ascii=False) + '\n')

# 2 Process SFT data
def convert_message(data):
    """Convert raw data to standard format"""
    message = [
        {"role": "system", "content": "You are an AI assistant"},
    ]
    for item in data:
        if item['from'] == 'human':
            message.append({'role': 'user', 'content': item['value']})
        elif item['from'] == 'assistant':
            message.append({'role': 'assistant', 'content': item['value']})
    return message

with open(output_sft_data, 'a', encoding='utf-8') as sft:
    with open(sft_data, 'r') as f:
        data = f.readlines()
        for item in tqdm(data, desc="Processing", unit="lines"):
            item = json.loads(item)
            message = convert_message(item['conversations'])
            sft.write(json.dumps(message, ensure_ascii=False) + '\n')