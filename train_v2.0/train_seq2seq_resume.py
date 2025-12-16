# 在 train_v2.0.py 基础上：继续训练 10 轮

import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim

from train_seq2seq import (
    DynamicDataset, collate_fn,
    Attention, Encoder, Decoder, Seq2Seq,
    run_epoch, plot_metrics, translate
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAD_ID = 0

if __name__ == '__main__':
    print('Starting resume training...')

    # ========= 1. 加载数据（和原来完全一致） =========
    data_dir = 'preprocessed_attention_data'
    en2id = torch.load(os.path.join(data_dir, 'en2id.pt'))
    zh2id = torch.load(os.path.join(data_dir, 'zh2id.pt'))
    train_data = torch.load(os.path.join(data_dir, 'train_data.pt'))
    dev_data   = torch.load(os.path.join(data_dir, 'dev_data.pt'))

    train_loader = torch.utils.data.DataLoader(
        DynamicDataset(train_data),
        batch_size=128,
        shuffle=True,
        collate_fn=collate_fn
    )
    dev_loader = torch.utils.data.DataLoader(
        DynamicDataset(dev_data),
        batch_size=128,
        shuffle=False,
        collate_fn=collate_fn
    )

    # ========= 2. 构建模型（必须和原来一模一样） =========
    attention = Attention(1024, 512)
    encoder = Encoder(len(en2id), 256, 512, 2, 0.2)
    decoder = Decoder(len(zh2id), 256, 512, 0.2, attention)
    model = Seq2Seq(encoder, decoder).to(device)

    # ========= 3. 加载旧模型参数 =========
    ckpt_path = 'best_model_attention_2.pt'
    print(f'Loading checkpoint: {ckpt_path}')
    model.load_state_dict(torch.load(ckpt_path))

    # ========= 4. 微调设置（省算力 & 稳定） =========
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # 更低 LR
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    extra_epochs = 10
    history = {'train_loss': [], 'val_loss': [], 'train_ppl': [], 'val_ppl': []}
    best_val = float('inf')

    # ========= 5. 继续训练 =========
    for epoch in range(extra_epochs):
        start = time.time()

        # 更低的 teacher forcing
        tf_ratio = max(0.05, 0.3 * (1 - epoch / extra_epochs))

        train_loss = run_epoch(
            model, train_loader, optimizer, criterion,
            tf_ratio=tf_ratio, desc=f'Resume Train {epoch+1}'
        )
        val_loss = run_epoch(
            model, dev_loader, optimizer, criterion,
            desc=f'Resume Val {epoch+1}'
        )

        train_ppl = math.exp(train_loss)
        val_ppl   = math.exp(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_ppl'].append(train_ppl)
        history['val_ppl'].append(val_ppl)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(),
                       'bilstm_attention_mt_v1_resume10.pt')

        mins, secs = divmod(int(time.time() - start), 60)
        print(f'Epoch: {epoch+1:02} | Time: {mins}m {secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | PPL: {train_ppl:.3f}')
        print(f'\t Val. Loss: {val_loss:.3f} | PPL: {val_ppl:.3f}')

    # ========= 6. 画图 =========
    print('Plotting training curves...')
    plot_metrics(history)

    # ========= 7. 测试翻译 =========
    print('\\nTranslation Test:')
    zh_wl = list(zh2id.keys())
    model.load_state_dict(torch.load('bilstm_attention_mt_v1_resume10.pt'))

    tests = [
        'what is your name',
        'how are you',
        'i love deep learning',
        'sequence to sequence learning with neural networks'
    ]

    for s in tests:
        print(s, '->', translate(model, s, en2id, zh2id, zh_wl))
