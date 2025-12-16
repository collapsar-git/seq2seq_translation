"""
测试脚本：加载已训练好的 Seq2Seq + Attention 模型并进行翻译评估
用法：
    python test_seq2seq.py
要求：
    - best_model_attention.pt 与本脚本在同一目录
    - 训练时使用的词表文件仍然存在
"""

import torch
import torch.nn as nn
import os
from train_seq2seq import Encoder, Decoder, Attention, Seq2Seq, PAD_ID, device

# ---------------- 配置 ----------------
DATA_DIR = 'preprocessed_attention_data'
MODEL_PATH = 'best_model_attention_2.pt'

# ---------------- 加载词表 ----------------
print('Loading vocab...')
en2id = torch.load(os.path.join(DATA_DIR, 'en2id.pt'))
zh2id = torch.load(os.path.join(DATA_DIR, 'zh2id.pt'))
id2zh = {v: k for k, v in zh2id.items()}

# ---------------- 构建模型（必须与训练一致） ----------------
print('Building model...')
attention = Attention(enc_dim=1024, dec_dim=512)
encoder = Encoder(len(en2id), emb_dim=256, hid_dim=512, n_layers=2, dropout=0.2)
decoder = Decoder(len(zh2id), emb_dim=256, hid_dim=512, dropout=0.2, attention=attention)
model = Seq2Seq(encoder, decoder).to(device)

# ---------------- 加载权重 ----------------
print('Loading trained weights...')
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()

# ---------------- 翻译函数 ----------------
def translate(sentence, max_len=50):
    tokens = sentence.lower().split()
    src_ids = [en2id.get(w, en2id['<unk>']) for w in tokens]
    src = torch.tensor([en2id['<sos>']] + src_ids + [en2id['<eos>']]).unsqueeze(1).to(device)

    with torch.no_grad():
        enc_out, hidden, cell = model.encoder(src)
        mask = model.create_mask(src)

        input = torch.tensor([zh2id['<sos>']], device=device)
        result = []

        for _ in range(max_len):
            out, hidden, cell = model.decoder(input, hidden, cell, enc_out, mask)
            top = out.argmax(1).item()
            if top == zh2id['<eos>']:
                break
            result.append(id2zh[top])
            input = torch.tensor([top], device=device)

    return ''.join(result)

# ---------------- 测试样例 ----------------
TEST_SENTENCES = [
    'what is your name',
    'how are you',
    'i love deep learning',
    'machine learning is very interesting',
    'sequence to sequence learning with neural networks'
]

print('\nTranslation Results:')
for s in TEST_SENTENCES:
    print('-' * 60)
    print('EN:', s)
    print('ZH:', translate(s))

print('\nDone.')