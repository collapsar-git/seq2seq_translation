import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import time
from tqdm import tqdm

# --- Dataloader Components (Moved to Top-Level) ---

# Global Vocabularies are only placeholders now.
en2id = {}
zh2id = {}
pad_id = 2 

class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        s1 = example[0]
        s2 = example[1]
        l1 = len(s1)
        l2 = len(s2)
        return s1, l1, s2, l2, index

# NOTE: the_collate_fn definition is MOVED INTO __main__ to capture vocabulary.
# The definition here is only a placeholder or removed.

# --- Model Classes (Top-Level) ---

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, input_word_count, output_word_count, encode_dim, decode_dim, hidden_dim, n_layers, encode_dropout, decode_dropout, device):
        super().__init__()
        self.encoder = Encoder(input_word_count, encode_dim, hidden_dim, n_layers, encode_dropout)
        self.decoder = Decoder(output_word_count, decode_dim, hidden_dim, n_layers, decode_dropout)
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        
        if trg is not None:
            trg_len = trg.shape[0]
            trg_vocab_size = self.decoder.output_dim
            outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
            input = trg[0, :]
        else:
            # Inference mode setup
            trg_len = 100 
            trg_vocab_size = self.decoder.output_dim
            outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
            # Must assume zh2id is correctly passed or globally accessible (now handled in __main__)
            input = torch.full((batch_size,), zh2id['<sos>']).to(self.device)

        hidden, cell = self.encoder(src)

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)

            outputs[t] = output
            top1 = output.argmax(1) 
            
            if trg is not None:
                teacher_force = random.random() < teacher_forcing_ratio
                input = trg[t] if teacher_force else top1
            else:
                input = top1
                if (input == zh2id['<eos>']).all():
                    return outputs[:t+1] 
        
        return outputs


# --- Utility Functions (Top-Level) ---

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(model, iterator, optimizer, criterion, clip, device):
    """训练函数，支持GPU"""
    model.train()
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        src = batch[0].to(device)
        trg = batch[1].to(device)
        
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    """评估函数，支持GPU"""
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[0].to(device)
            trg = batch[1].to(device)

            output = model(src, trg, 0)  # turn off teacher forcing

            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
            
        return epoch_loss / len(iterator)

def translate(model, en_sentence, device, en2id_map, zh_wl, zh2id_map):
    """Translates an English sentence to Chinese using the trained model."""
    words = []
    for word in en_sentence.strip().split(' '):
        word = word.replace('.', '').replace(',', '').lower()
        if word:
             words.append(word)
    
    # Use the passed-in maps
    ids = [[en2id_map['<sos>']]]
    for w in words:
        if w in en2id_map:
            ids.append([en2id_map[w]]) 
    ids.append([en2id_map['<eos>']])
    
    if len(ids) <= 1:
         return "Translation failed: Empty input after tokenization."

    src = torch.tensor(ids).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(src, None, 0) 
        
    trg_ids = []
    for x in output[1:]:
        predicted_id = x.argmax(1).cpu().item()
        
        # Use the passed-in maps
        if predicted_id == zh2id_map['<eos>']:
            break
            
        trg_ids.append(predicted_id)
        
    result_chars = [zh_wl[i] for i in trg_ids]
    return ''.join(result_chars)


# --- Data Loading Function (Top-Level) ---

def load_data(en_file, zh_file):
    """Loads and preprocesses the English and Chinese data."""
    try:
        fen = open(en_file, encoding='utf8')
        fzh = open(zh_file, encoding='utf8')
    except FileNotFoundError as e:
        print(f"Error: Required file not found: {e.filename}")
        print("Please ensure 'train.tags.zh-en.en' and 'train.tags.zh-en.zh' are in the same directory.")
        return []

    en_zh = []
    
    while True:
        lz = fzh.readline() 
        le = fen.readline() 
        
        if not lz:
            assert not le 
            break
            
        if lz.startswith('<url>'):
            # ... (tag skipping logic) ...
            assert le.startswith('<url>')
            lz = fzh.readline()
            le = fen.readline()
            
            assert lz.startswith('<keywords>')
            assert le.startswith('<keywords>')
            lz = fzh.readline()
            le = fen.readline()
            
            assert lz.startswith('<speaker>')
            assert le.startswith('<speaker>')
            lz = fzh.readline()
            le = fen.readline()
            
            assert lz.startswith('<talkid>')
            assert le.startswith('<talkid>')
            lz = fzh.readline()
            le = fen.readline()
            
            assert lz.startswith('<title>') 
            assert le.startswith('<title>')
            lz = fzh.readline()
            le = fen.readline()
            
            assert lz.startswith('<description>')
            assert le.startswith('<description>')
            
        else:
            lee = []
            for w in le.split(' '):
                w = w.replace('.', '').replace(',', '').lower()
                if w:
                    lee.append(w)
            
            lz_stripped = lz.strip()
            if lee and lz_stripped:
                 en_zh.append([lee, list(lz_stripped)])

    fen.close()
    fzh.close()
    return en_zh


# --- Main execution block ---

if __name__ == '__main__':
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 1. Load data
    print("Loading and preprocessing data...")
    en_zh_data = load_data('train.tags.zh-en.en', 'train.tags.zh-en.zh')

    if not en_zh_data:
        print("Using dummy data for demonstration as data files are not accessible.")
        en_zh_data = [
            [['what', 'is', 'your', 'name'], ['你', '叫', '什', '么', '名', '字']],
            [['sequence', 'to', 'sequence', 'learning', 'with', 'neural', 'networks'], 
             ['序', '列', '到', '序', '列', '神', '经', '网', '络', '学', '习']],
            [['example', 'sentence'], ['示', '例', '句', '子']]
        ] * 10 
        if not en_zh_data:
            print("Data loading failed. Exiting.")
            exit()
        
    en_words = set()
    zh_words = set()
    for s in tqdm(en_zh_data, desc="Building vocabulary"):
        for w in s[0]:
            en_words.add(w)
        for w in s[1]:
            if w:
                zh_words.add(w)

    en_wl = ['<sos>', '<eos>', '<pad>'] + list(en_words)
    zh_wl = ['<sos>', '<eos>', '<pad>'] + list(zh_words)
    
    # Assign global variables HERE, before DataLoader is created
    en2id = {w: i for i, w in enumerate(en_wl)}
    zh2id = {w: i for i, w in enumerate(zh_wl)}
    # pad_id is 2 by default

    random.shuffle(en_zh_data)
    dl = len(en_zh_data)

    # 使用 0.8*0.8 = 0.64 (约 64% 的原始数据) 作为训练集
    train_split_ratio = 0.7 * 0.7
    train_set = en_zh_data[:int(dl * train_split_ratio)]
    
    dev_set = en_zh_data[int(dl * 0.8):]
    print(f"Dataset size: {dl}. Train size: {len(train_set)}, Dev size: {len(dev_set)}")


    # --- Dataloader Setup ---

    batch_size = 8
    # Keep data_workers at 8 for performance, now that variable access is fixed.
    data_workers = 0

    # 重新定义 collate_fn 作为闭包，捕获词汇表映射，确保子进程能正确访问它们
    def the_collate_fn(batch):
        batch_size = len(batch)
        # Use the captured/global variables en2id, zh2id, pad_id
        
        src = [[en2id['<sos>']] * batch_size] 
        tar = [[zh2id['<sos>']] * batch_size] 
        
        src_max_l = max(b[1] for b in batch)
        tar_max_l = max(b[3] for b in batch)
        
        # Pad and convert English (src) to IDs
        for i in range(src_max_l):
            l = []
            for x in batch:
                # Use the captured/global en2id
                if i < x[1] and x[0][i] in en2id:
                    l.append(en2id[x[0][i]]) 
                else:
                    l.append(pad_id)
            src.append(l)
        
        # Pad and convert Chinese (tar) to IDs
        for i in range(tar_max_l):
            l = []
            for x in batch:
                # Use the captured/global zh2id
                if i < x[3] and x[2][i] in zh2id:
                    l.append(zh2id[x[2][i]]) 
                else:
                    l.append(pad_id)
            tar.append(l)
            
        indexs = [b[4] for b in batch]
        
        # Append <eos> token
        src.append([en2id['<eos>']] * batch_size)
        tar.append([zh2id['<eos>']] * batch_size)
        
        s1 = torch.LongTensor(src)
        s2 = torch.LongTensor(tar)
        return s1, s2, indexs

    train_dataset = MyDataSet(train_set)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_workers,
        collate_fn=the_collate_fn,
    )

    dev_dataset = MyDataSet(dev_set)
    dev_data_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_workers,
        collate_fn=the_collate_fn,
    )


    # --- Model Initialization ---

    source_word_count = len(en_wl)
    target_word_count = len(zh_wl)
    encode_dim = 256
    decode_dim = 256
    hidden_dim = 512
    n_layers = 2
    encode_dropout = 0.6
    decode_dropout = 0.6

    model = Seq2Seq(source_word_count, target_word_count, encode_dim, decode_dim, hidden_dim, n_layers, encode_dropout, decode_dropout, device).to(device)

    model.apply(init_weights)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    # optimizer = optim.Adam(model.parameters())
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)


    # --- Training Loop ---

    N_EPOCHS = 10
    CLIP = 1
    best_valid_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(N_EPOCHS):
        
        start_time = time.time()
        
        train_loss = train(model, train_data_loader, optimizer, criterion, CLIP, device)
        valid_loss = evaluate(model, dev_data_loader, criterion, device)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut1-model.pt')
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    # --- Translation Example ---
    
    print("\nExample Translations:")
    # Pass the maps explicitly to the translate function
    result1 = translate(model, 'what is your name', device, en2id, zh_wl, zh2id)
    print(f"what is your name -> {result1}")

    result2 = translate(model, 'Sequence to Sequence Learning with Neural Networks', device, en2id, zh_wl, zh2id)
    print(f"Sequence to Sequence Learning with Neural Networks -> {result2}")