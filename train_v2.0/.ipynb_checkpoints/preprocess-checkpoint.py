import torch
import random
from tqdm import tqdm
import os
import sys

# --- Dataloader Components ---
en2id = {}
zh2id = {}
pad_id = 2 
unk_id = 3 

def load_data(en_file, zh_file):
    """Loads and preprocesses the English and Chinese data."""
    try:
        fen = open(en_file, encoding='utf8')
        fzh = open(zh_file, encoding='utf8')
    except FileNotFoundError as e:
        print(f"Error: Required file not found: {e.filename}")
        return []

    en_zh = []
    while True:
        lz = fzh.readline() 
        le = fen.readline() 
        if not lz:
            break
        if lz.startswith('<url>'):
            for _ in range(6): # Skip metadata lines
                fzh.readline(); fen.readline()
        else:
            lee = []
            for w in le.split(' '):
                w = w.replace('.', '').replace(',', '').lower()
                if w: lee.append(w)
            lz_stripped = lz.strip()
            if lee and lz_stripped:
                 en_zh.append([lee, list(lz_stripped)])
    fen.close(); fzh.close()
    return en_zh

def process_and_save(dataset, en2id_map, zh2id_map, unk_id_val, output_dir, file_prefix):
    """将文本转换为ID列表，并保存为list of tuples"""
    processed_data = []
    
    zh_unk_id = zh2id_map.get('<unk>', unk_id_val)
    
    for s1, s2 in tqdm(dataset, desc=f"Processing {file_prefix}"):
        # Source
        src_ids = [en2id_map['<sos>']] + [en2id_map.get(w, unk_id_val) for w in s1] + [en2id_map['<eos>']]
        # Target
        trg_ids = [zh2id_map['<sos>']] + [zh2id_map.get(c, zh_unk_id) for c in s2] + [zh2id_map['<eos>']]
        
        processed_data.append((src_ids, trg_ids))

    save_path = os.path.join(output_dir, f'{file_prefix}_data.pt')
    torch.save(processed_data, save_path)
    print(f"Saved {len(processed_data)} samples to {save_path}")

if __name__ == '__main__':
    output_directory = 'preprocessed_attention_data'
    os.makedirs(output_directory, exist_ok=True)
    
    print("Loading raw data...")
    en_zh_data = load_data('train.tags.zh-en.en', 'train.tags.zh-en.zh')

    # 1. 过滤超长句子 (保留 <= 100 的句子)
    # 虽然 V100 显存很大，但太长的句子（如200+）对于 LSTM 来说很难训练，100 是一个合理的“拉满”界限
    MAX_LEN_FILTER = 100 
    en_zh_data = [d for d in en_zh_data if len(d[0]) <= MAX_LEN_FILTER and len(d[1]) <= MAX_LEN_FILTER]
    
    total_samples = len(en_zh_data)
    print(f"Total valid samples: {total_samples}")

    if not en_zh_data: sys.exit(1)
        
    # Build Vocabulary
    en_words = set()
    zh_words = set()
    for s in tqdm(en_zh_data, desc="Building vocabulary"):
        for w in s[0]: en_words.add(w)
        for w in s[1]: zh_words.add(w)

    en_wl = ['<sos>', '<eos>', '<pad>', '<unk>'] + list(en_words)
    zh_wl = ['<sos>', '<eos>', '<pad>', '<unk>'] + list(zh_words)
    
    en2id = {w: i for i, w in enumerate(en_wl)}
    zh2id = {w: i for i, w in enumerate(zh_wl)}
    pad_id = en2id['<pad>']
    unk_id = en2id['<unk>'] 

    # --- 【关键修改】拉满数据 ---
    random.shuffle(en_zh_data)
    
    # 将 98% 的数据用于训练，只留 2% 用于验证
    # 这样既能保证数据利用率最大化，又能防止验证集为0导致的报错
    split_idx = int(total_samples * 0.98) 
    
    train_set = en_zh_data[:split_idx]
    dev_set = en_zh_data[split_idx:]
    
    print(f"Data Split -> Train: {len(train_set)} (98%), Dev: {len(dev_set)} (2%)")

    # Save Data
    process_and_save(train_set, en2id, zh2id, unk_id, output_directory, 'train')
    process_and_save(dev_set, en2id, zh2id, unk_id, output_directory, 'dev')
    
    # Save Vocab
    torch.save(en2id, os.path.join(output_directory, 'en2id.pt'))
    torch.save(zh2id, os.path.join(output_directory, 'zh2id.pt'))
    print("Preprocessing finished. Data utilization maximized.")