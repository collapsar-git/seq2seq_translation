# ============================
# Resume Training Seq2Seq + Attention
# 继续训练 + Beam Search 推理
# ============================

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import time
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAD_ID = 0

# ---------------- Dataset ----------------
class DynamicDataset(torch.utils.data.Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_fn(batch):
    src, trg = zip(*batch)
    max_src = max(len(s) for s in src)
    max_trg = max(len(t) for t in trg)
    bs = len(batch)
    src_tensor = torch.full((bs, max_src), PAD_ID)
    trg_tensor = torch.full((bs, max_trg), PAD_ID)
    for i in range(bs):
        src_tensor[i, :len(src[i])] = torch.tensor(src[i])
        trg_tensor[i, :len(trg[i])] = torch.tensor(trg[i])
    return src_tensor, trg_tensor

# ---------------- Attention ----------------
class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.attn = nn.Linear(enc_dim + dec_dim, dec_dim)
        self.v = nn.Linear(dec_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        if hidden.dim() == 3: hidden = hidden[-1]
        src_len = encoder_outputs.size(0)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attn = self.v(energy).squeeze(2)
        attn = attn.masked_fill(mask == 0, -1e10)
        return F.softmax(attn, dim=1)

# ---------------- Encoder ----------------
class Encoder(nn.Module):
    def __init__(self, vocab, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb_dim, padding_idx=PAD_ID)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hid_dim*2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (h, c) = self.rnn(embedded)
        h = torch.tanh(self.fc(torch.cat((h[-2], h[-1]), dim=1)))
        c = torch.zeros_like(h)
        return outputs, h.unsqueeze(0), c.unsqueeze(0)

# ---------------- Decoder ----------------
class Decoder(nn.Module):
    def __init__(self, vocab, emb_dim, hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = vocab
        self.attention = attention
        self.embedding = nn.Embedding(vocab, emb_dim, padding_idx=PAD_ID)
        self.rnn = nn.LSTM(emb_dim + hid_dim*2, hid_dim)
        self.fc = nn.Linear(hid_dim*3, vocab)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs, mask):
        input = input.unsqueeze(0)
        emb = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs, mask).unsqueeze(1)
        enc = encoder_outputs.permute(1,0,2)
        weighted = torch.bmm(a, enc).permute(1,0,2)
        rnn_input = torch.cat((emb, weighted), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        pred = self.fc(torch.cat((output.squeeze(0), weighted.squeeze(0)), dim=1))
        return pred, hidden, cell

# ---------------- Seq2Seq ----------------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def create_mask(self, src): return (src != PAD_ID).permute(1,0)
    def forward(self, src, trg, tf_ratio):
        batch = src.shape[1]
        trg_len = trg.shape[0]
        vocab = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch, vocab).to(device)
        enc_out, hidden, cell = self.encoder(src)
        mask = self.create_mask(src)
        input = trg[0]
        for t in range(1, trg_len):
            out, hidden, cell = self.decoder(input, hidden, cell, enc_out, mask)
            outputs[t] = out
            top1 = out.argmax(1)
            input = trg[t] if random.random() < tf_ratio else top1
        return outputs

# ---------------- Run Epoch ----------------
def run_epoch(model, loader, optimizer, criterion, tf_ratio=None, desc=''):
    is_train = tf_ratio is not None
    model.train() if is_train else model.eval()
    total_loss = 0
    pbar = tqdm(loader, desc=desc, leave=False)
    with torch.set_grad_enabled(is_train):
        for src, trg in pbar:
            src = src.T.to(device)
            trg = trg.T.to(device)
            if is_train: optimizer.zero_grad()
            output = model(src, trg, tf_ratio if is_train else 0)
            output = output[1:].reshape(-1, output.shape[-1])
            trg = trg[1:].reshape(-1)
            loss = criterion(output, trg)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
    return total_loss / len(loader)

# ---------------- Plot ----------------
def plot_metrics(history, save_dir='images'):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.legend(); plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(epochs, history['train_ppl'], label='Train PPL')
    plt.plot(epochs, history['val_ppl'], label='Val PPL')
    plt.legend(); plt.title('Perplexity')
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

# ---------------- Beam Search Translation ----------------
def translate_beam(model, sentence, en2id, zh2id, zh_wl, beam_size=5, max_len=50, alpha=0.6):
    model.eval()
    with torch.no_grad():
        tokens = [en2id.get(w, en2id['<unk>']) for w in sentence.lower().split()]
        src = torch.tensor([en2id['<sos>']] + tokens + [en2id['<eos>']]).unsqueeze(1).to(device)
        enc_out, hidden, cell = model.encoder(src)
        mask = model.create_mask(src)
        beams = [([zh2id['<sos>']], 0.0, hidden, cell)]
        for _ in range(max_len):
            new_beams = []
            for seq, score, h, c in beams:
                if seq[-1] == zh2id['<eos>']:
                    new_beams.append((seq, score, h, c))
                    continue
                input_id = torch.tensor([seq[-1]]).to(device)
                out, h_new, c_new = model.decoder(input_id, h, c, enc_out, mask)
                log_probs = F.log_softmax(out, dim=1).squeeze(0)
                topk_logp, topk_idx = log_probs.topk(beam_size)
                for i in range(beam_size):
                    new_seq = seq + [topk_idx[i].item()]
                    new_score = score + topk_logp[i].item()
                    new_beams.append((new_seq, new_score, h_new, c_new))
            beams = sorted(new_beams, key=lambda x: x[1]/(len(x[0])**alpha), reverse=True)[:beam_size]
        best_seq = beams[0][0]
        words = [zh_wl[idx] for idx in best_seq if idx not in (zh2id['<sos>'], zh2id['<eos>'], PAD_ID)]
        return ''.join(words)

# ---------------- Main ----------------
if __name__ == '__main__':
    data_dir = 'preprocessed_attention_data'
    en2id = torch.load(os.path.join(data_dir, 'en2id.pt'))
    zh2id = torch.load(os.path.join(data_dir, 'zh2id.pt'))
    train_data = torch.load(os.path.join(data_dir, 'train_data.pt'))
    dev_data = torch.load(os.path.join(data_dir, 'dev_data.pt'))

    train_loader = torch.utils.data.DataLoader(
        DynamicDataset(train_data), batch_size=128, shuffle=True, collate_fn=collate_fn)
    dev_loader = torch.utils.data.DataLoader(
        DynamicDataset(dev_data), batch_size=128, shuffle=False, collate_fn=collate_fn)

    attention = Attention(1024, 512)
    encoder = Encoder(len(en2id), 256, 512, 2, 0.2)
    decoder = Decoder(len(zh2id), 256, 512, 0.2, attention)
    model = Seq2Seq(encoder, decoder).to(device)

    # 加载已有模型
    ckpt_path = 'best_model_attention_2.pt'
    print(f'Loading checkpoint: {ckpt_path}')
    model.load_state_dict(torch.load(ckpt_path))

    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # 低 lr
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    history = {'train_loss':[], 'val_loss':[], 'train_ppl':[], 'val_ppl':[]}
    best_val = float('inf')

    extra_epochs = 10
    for epoch in range(extra_epochs):
        start = time.time()
        tf_ratio = max(0.05, 0.3 * (1 - epoch/extra_epochs))
        train_loss = run_epoch(model, train_loader, optimizer, criterion, tf_ratio, desc=f'Resume Train {epoch+1}')
        val_loss = run_epoch(model, dev_loader, optimizer, criterion, desc=f'Resume Val {epoch+1}')

        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_ppl'].append(train_ppl)
        history['val_ppl'].append(val_ppl)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'best_model_attention_resume_beam.pt')

        mins, secs = divmod(int(time.time() - start), 60)
        print(f'Epoch: {epoch+1:02} | Time: {mins}m {secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | PPL: {train_ppl:.3f}')
        print(f'\t Val. Loss: {val_loss:.3f} | PPL: {val_ppl:.3f}')

    print('Plotting training curves...')
    plot_metrics(history)

    # ---------------- Translation Test ----------------
    zh_wl = list(zh2id.keys())
    model.load_state_dict(torch.load('best_model_attention_resume_beam.pt'))

    tests = [
        'what is your name',
        'how are you',
        'i love deep learning',
        'sequence to sequence learning with neural networks'
    ]
    print('\nTranslation Test (Beam Search):')
    for s in tests:
        print(s, '->', translate_beam(model, s, en2id, zh2id, zh_wl, beam_size=5))
