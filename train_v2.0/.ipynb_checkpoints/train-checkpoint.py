import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import time
import os
import sys
import torch.nn.functional as F
import matplotlib.pyplot as plt  # [新增] 引入绘图库
from tqdm import tqdm

# --- Global Config ---
pad_id = 0 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Dataset & Collate Function ---

class DynamicDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def collate_fn(batch):
    src_batch = [item[0] for item in batch]
    trg_batch = [item[1] for item in batch]
    
    max_src_len = max(len(s) for s in src_batch)
    max_trg_len = max(len(t) for t in trg_batch)
    
    batch_size = len(batch)
    
    src_tensor = torch.full((batch_size, max_src_len), pad_id, dtype=torch.long)
    trg_tensor = torch.full((batch_size, max_trg_len), pad_id, dtype=torch.long)
    
    for i, seq in enumerate(src_batch):
        src_tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        
    for i, seq in enumerate(trg_batch):
        trg_tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        
    return src_tensor, trg_tensor

# --- Model Classes ---

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        hidden = hidden.repeat(src_len, 1, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2))) 
        attention = self.v(energy).permute(1, 0, 2)
        return nn.functional.softmax(attention.squeeze(2), dim=1)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, hidden, cell

class AttentionDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim * 2, output_dim) 
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0) 
        embedded = self.dropout(self.embedding(input))
        
        a = self.attention(hidden[-1].unsqueeze(0), encoder_outputs) 
        a = a.unsqueeze(1) 
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2) 
        weighted = torch.bmm(a, encoder_outputs).permute(1, 0, 2) 
        
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        prediction = self.fc_out(torch.cat((output.squeeze(0), weighted.squeeze(0)), dim=1))
        return prediction, hidden, cell

class Seq2SeqAttention(nn.Module):
    def __init__(self, input_word_count, output_word_count, encode_dim, decode_dim, hidden_dim, n_layers, encode_dropout, decode_dropout, device):
        super().__init__()
        attn = Attention(hidden_dim, hidden_dim)
        self.encoder = Encoder(input_word_count, encode_dim, hidden_dim, n_layers, encode_dropout)
        self.decoder = AttentionDecoder(output_word_count, decode_dim, hidden_dim, n_layers, decode_dropout, attn)
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0] if trg is not None else 100
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        input = trg[0, :] if trg is not None else torch.full((batch_size,), zh2id['<sos>'], device=self.device)

        encoder_outputs, hidden, cell = self.encoder(src)

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output
            top1 = output.argmax(1) 
            
            if trg is not None:
                input = trg[t] if random.random() < teacher_forcing_ratio else top1
            else:
                input = top1
        return outputs

# --- Training Utilities ---

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.uniform_(param.data, -0.08, 0.08)
        else:
            nn.init.constant_(param.data, 0)
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    pbar = tqdm(iterator, desc="Training", leave=False)
    
    for i, (src, trg) in enumerate(pbar):
        src, trg = src.transpose(0, 1).to(device), trg.transpose(0, 1).to(device)
        
        optimizer.zero_grad()
        output = model(src, trg)
        
        output_dim = output.shape[-1]
        output = output[1:].contiguous().view(-1, output_dim)
        trg = trg[1:].contiguous().view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.3f}'})
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.transpose(0, 1).to(device), trg.transpose(0, 1).to(device)
            output = model(src, trg, 0)
            output_dim = output.shape[-1]
            output = output[1:].contiguous().view(-1, output_dim)
            trg = trg[1:].contiguous().view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# --- [新增] 绘图函数 ---
def plot_metrics(train_losses, val_losses, train_ppls, val_ppls, save_dir='images'):
    # 确保文件夹存在
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    # 创建画布 (2行1列)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 绘制 Loss 曲线
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-o', label='Val Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 绘制 PPL 曲线
    ax2.plot(epochs, train_ppls, 'b-o', label='Train PPL')
    ax2.plot(epochs, val_ppls, 'r-o', label='Val PPL')
    ax2.set_title('Training & Validation PPL')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Perplexity')
    ax2.legend()
    ax2.grid(True)
    
    # 保存图片
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n[Info] Training curves saved to: {save_path}")
    plt.close() # 释放内存

# --- Beam Search Translation ---

def translate_beam_search(model, sentence, en2id, zh2id, zh_wl, beam_size=3, max_len=50):
    model.eval()
    tokens = [en2id.get(w, en2id.get('<unk>')) for w in sentence.lower().replace('.', '').split()]
    ids = [en2id['<sos>']] + tokens + [en2id['<eos>']]
    src_tensor = torch.LongTensor(ids).unsqueeze(1).to(device)
    
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
        beam = [(0, [zh2id['<sos>']], hidden, cell)]
        
        for _ in range(max_len):
            candidates = []
            all_finished = True
            
            for score, seq, h, c in beam:
                if seq[-1] == zh2id['<eos>']:
                    candidates.append((score, seq, h, c))
                    continue
                
                all_finished = False
                input_token = torch.tensor([seq[-1]]).to(device)
                
                prediction, new_h, new_c = model.decoder(input_token, h, c, encoder_outputs)
                prediction = prediction.squeeze(0).squeeze(0)
                log_probs = F.log_softmax(prediction, dim=0)
                topk_probs, topk_ids = log_probs.topk(beam_size)
                
                for i in range(beam_size):
                    token = topk_ids[i].item()
                    prob = topk_probs[i].item()
                    new_score = score + prob
                    new_seq = seq + [token]
                    candidates.append((new_score, new_seq, new_h, new_c))
            
            beam = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]
            if all_finished: break
        
        best_seq = beam[0][1]
        result = []
        for idx in best_seq:
            if idx == zh2id['<sos>']: continue
            if idx == zh2id['<eos>']: break
            result.append(zh_wl[idx])
        return "".join(result)

# --- Main ---

if __name__ == '__main__':
    # 1. Load Data
    data_dir = 'preprocessed_attention_data'
    if not os.path.exists(data_dir):
        print("Please run preprocess.py first.")
        sys.exit(1)
        
    print(f"Loading data on {device}...")
    try:
        en2id = torch.load(os.path.join(data_dir, 'en2id.pt'))
        zh2id = torch.load(os.path.join(data_dir, 'zh2id.pt'))
        train_data = torch.load(os.path.join(data_dir, 'train_data.pt'))
        dev_data = torch.load(os.path.join(data_dir, 'dev_data.pt'))
    except Exception as e:
        print(f"Error loading files: {e}")
        sys.exit(1)
    
    pad_id = en2id['<pad>']
    zh_wl = list(zh2id.keys())
    
    BATCH_SIZE = 256
    
    train_iterator = torch.utils.data.DataLoader(
        DynamicDataset(train_data), batch_size=BATCH_SIZE, shuffle=True, 
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    
    dev_iterator = torch.utils.data.DataLoader(
        DynamicDataset(dev_data), batch_size=BATCH_SIZE, shuffle=False, 
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    
    print(f"Train batches: {len(train_iterator)}, Dev batches: {len(dev_iterator)}")

    INPUT_DIM = len(en2id)
    OUTPUT_DIM = len(zh2id)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.6
    DEC_DROPOUT = 0.6

    model = Seq2SeqAttention(INPUT_DIM, OUTPUT_DIM, ENC_EMB_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT, DEC_DROPOUT, device).to(device)
    model.apply(init_weights)
    print(f'Model parameters: {count_parameters(model):,}')

    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    N_EPOCHS = 10
    CLIP = 1
    best_valid_loss = float('inf')

    # [新增] 用于存储历史记录的列表
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_ppl': [],
        'val_ppl': []
    }

    print("Starting training...")
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, dev_iterator, criterion)
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(int(end_time - start_time), 60)
        
        # 计算 PPL
        train_ppl = math.exp(train_loss) if train_loss < 100 else float('inf')
        val_ppl = math.exp(valid_loss) if valid_loss < 100 else float('inf')
        
        # [新增] 记录数据
        history['train_loss'].append(train_loss)
        history['val_loss'].append(valid_loss)
        history['train_ppl'].append(train_ppl)
        history['val_ppl'].append(val_ppl)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model_attention.pt')
            
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | PPL: {train_ppl:7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} | PPL: {val_ppl:7.3f}')
        
    # [新增] 训练结束后绘图
    print("Plotting training curves...")
    plot_metrics(history['train_loss'], history['val_loss'], 
                 history['train_ppl'], history['val_ppl'])

    # Test Translation
    print("\nLoading best model for testing...")
    model.load_state_dict(torch.load('best_model_attention.pt'))
    
    test_sentences = [
        "what is your name",
        "how are you",
        "i love deep learning",
        "sequence to sequence learning with neural networks"
    ]
    
    print("\nTranslation Test (Beam Search vs Greedy):")
    for s in test_sentences:
        beam_res = translate_beam_search(model, s, en2id, zh2id, zh_wl, beam_size=3)
        print(f"Input: {s}")
        print(f"Beam(3): {beam_res}\n")