"""
PHASE 8A: BiLSTM with Attention (Text + KG)
==============================================
Trains BiLSTM baseline on the 5000-email dataset.
Features: Tokenized body + phi_G (8-dim KG vector)
"""
import os
import re
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score as sk_f1

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("=" * 65)
print("PHASE 8A — BiLSTM WITH ATTENTION")
print("=" * 65)

# ──────────────────────────────────────────────────────────────────────
# STEP 1 — Build vocabulary and tokenizer
# ──────────────────────────────────────────────────────────────────────
print("\n[STEP 1] Loading data & building vocabulary...")
# Load metadata from Phase 4A
train_meta = pd.read_parquet('data/features/split_train.parquet')
val_meta   = pd.read_parquet('data/features/split_val.parquet')
test_meta  = pd.read_parquet('data/features/split_test.parquet')

all_labeled = pd.read_parquet('data/labeled/emails_labeled_silver.parquet')

train_df = all_labeled[all_labeled['mid'].isin(train_meta['mid'])]
val_df   = all_labeled[all_labeled['mid'].isin(val_meta['mid'])]
test_df  = all_labeled[all_labeled['mid'].isin(test_meta['mid'])]

MAX_VOCAB = 30000
MAX_LEN = 200

# Fallback safely to another column if body_clean missing
body_col = 'body_clean' if 'body_clean' in train_df.columns else 'body_dense' if 'body_dense' in train_df.columns else 'body'

def tokenize(text):
    if not isinstance(text, str): return []
    return re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())[:MAX_LEN]

counter = Counter()
for text in train_df[body_col].fillna(''):
    counter.update(tokenize(text))

vocab = ['<PAD>', '<UNK>'] + [w for w, c in counter.most_common(MAX_VOCAB-2)]
word2idx = {w: i for i, w in enumerate(vocab)}
print(f"Vocab size: {len(vocab)}")

def encode(text):
    tokens = tokenize(text)
    ids = [word2idx.get(t, 1) for t in tokens]  # 1 = UNK
    if len(ids) < MAX_LEN:
        ids = ids + [0] * (MAX_LEN - len(ids))   # 0 = PAD
    return ids[:MAX_LEN]

# ──────────────────────────────────────────────────────────────────────
# STEP 2 — Dataset
# ──────────────────────────────────────────────────────────────────────
LABEL_MAPS = {
    'disclosure_type': {'FINANCIAL':0,'PII':1,'STRATEGIC':2,'LEGAL':3,'RELATIONAL':4,'NONE':5},
    'framing':         {'PROTECTED':0,'UNPROTECTED':1,'NA':2},
    'risk_tier':       {'NONE':0,'LOW':1,'HIGH':2}
}

class EmailDataset(Dataset):
    def __init__(self, df, phi_g_array, include_kg=True):
        self.texts   = [encode(t) for t in df[body_col].fillna('')]
        self.y_type  = [LABEL_MAPS['disclosure_type'].get(str(y), 5) for y in df['disclosure_type']]
        self.y_frame = [LABEL_MAPS['framing'].get(str(y), 2) for y in df['framing']]
        self.y_risk  = [LABEL_MAPS['risk_tier'].get(str(y), 0) for y in df['risk_tier']]
        self.phi_g   = phi_g_array if include_kg else np.zeros_like(phi_g_array)
        
    def __len__(self): return len(self.texts)
    
    def __getitem__(self, i):
        return {
            'input_ids': torch.tensor(self.texts[i], dtype=torch.long),
            'phi_g':     torch.tensor(self.phi_g[i], dtype=torch.float),
            'y_type':    torch.tensor(self.y_type[i],  dtype=torch.long),
            'y_frame':   torch.tensor(self.y_frame[i], dtype=torch.long),
            'y_risk':    torch.tensor(self.y_risk[i],  dtype=torch.long),
        }

phi_g_train = np.load('data/features/phi_g_train.npy')
phi_g_val   = np.load('data/features/phi_g_val.npy')
phi_g_test  = np.load('data/features/phi_g_test.npy')

# ──────────────────────────────────────────────────────────────────────
# STEP 3 — BiLSTM Model with Attention
# ──────────────────────────────────────────────────────────────────────
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)
    
    def forward(self, lstm_out):
        weights = torch.softmax(self.attn(lstm_out), dim=1)
        return (weights * lstm_out).sum(dim=1)

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128,
                 phi_g_dim=8, proj_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                            bidirectional=True, dropout=0.3, num_layers=2)
        self.attention = Attention(hidden_dim)
        # KG projection
        self.kg_proj = nn.Sequential(
            nn.Linear(phi_g_dim, proj_dim), nn.ReLU(), nn.Dropout(0.2)
        )
        combined = hidden_dim * 2 + proj_dim
        self.head_type  = nn.Linear(combined, 6)
        self.head_frame = nn.Linear(combined, 3)
        self.head_risk  = nn.Linear(combined, 3)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, input_ids, phi_g):
        x = self.dropout(self.embedding(input_ids))
        lstm_out, _ = self.lstm(x)
        pooled = self.attention(lstm_out)
        kg_feat = self.kg_proj(phi_g)
        combined = torch.cat([pooled, kg_feat], dim=1)
        return self.head_type(combined), self.head_frame(combined), self.head_risk(combined)

# ──────────────────────────────────────────────────────────────────────
# STEP 4 — Training loop
# ──────────────────────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nTraining on device: {device}")

def train_bilstm(include_kg=True, max_epochs=15, patience=3):
    suffix = 'kg' if include_kg else 'text'
    print(f"\n--- Training BiLSTM ({suffix}) ---")
    
    train_ds = EmailDataset(train_df, phi_g_train, include_kg)
    val_ds   = EmailDataset(val_df,   phi_g_val,   include_kg)
    test_ds  = EmailDataset(test_df,  phi_g_test,  include_kg)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64)
    test_loader  = DataLoader(test_ds,  batch_size=64)
    
    model = BiLSTMClassifier(len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Weighted losses to handle class imbalance
    w_type  = torch.tensor([1.5, 3.0, 2.0, 2.5, 2.0, 0.5], dtype=torch.float).to(device)
    w_frame = torch.tensor([2.0, 1.0, 0.5], dtype=torch.float).to(device)
    w_risk  = torch.tensor([0.5, 2.0, 3.0], dtype=torch.float).to(device)
    
    criterion_type  = nn.CrossEntropyLoss(weight=w_type)
    criterion_frame = nn.CrossEntropyLoss(weight=w_frame)
    criterion_risk  = nn.CrossEntropyLoss(weight=w_risk)
    
    best_val_f1, no_improve = 0.0, 0
    
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            ids   = batch['input_ids'].to(device)
            phi   = batch['phi_g'].to(device)
            y_t   = batch['y_type'].to(device)
            y_f   = batch['y_frame'].to(device)
            y_r   = batch['y_risk'].to(device)
            
            out_t, out_f, out_r = model(ids, phi)
            loss = 0.5 * criterion_type(out_t, y_t) + \
                   0.3 * criterion_frame(out_f, y_f) + \
                   0.2 * criterion_risk(out_r, y_r)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds_type, val_true_type = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(device)
                phi = batch['phi_g'].to(device)
                out_t, _, _ = model(ids, phi)
                val_preds_type.extend(out_t.argmax(1).cpu().numpy())
                val_true_type.extend(batch['y_type'].numpy())
        
        val_f1 = sk_f1(val_true_type, val_preds_type, average='macro', zero_division=0)
        print(f"  Epoch {epoch+1:2d}: loss={total_loss/len(train_loader):.4f}, val_macro_F1={val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f'models/bilstm_{suffix}_best.pt')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Test evaluation
    print("Evaluating best model on Test set...")
    model.load_state_dict(torch.load(f'models/bilstm_{suffix}_best.pt'))
    model.eval()
    all_preds = {'type': [], 'frame': [], 'risk': []}
    all_true  = {'type': [], 'frame': [], 'risk': []}
    
    with torch.no_grad():
        for batch in test_loader:
            ids = batch['input_ids'].to(device)
            phi = batch['phi_g'].to(device)
            out_t, out_f, out_r = model(ids, phi)
            
            all_preds['type'].extend(out_t.argmax(1).cpu().numpy())
            all_preds['frame'].extend(out_f.argmax(1).cpu().numpy())
            all_preds['risk'].extend(out_r.argmax(1).cpu().numpy())
            
            all_true['type'].extend(batch['y_type'].numpy())
            all_true['frame'].extend(batch['y_frame'].numpy())
            all_true['risk'].extend(batch['y_risk'].numpy())
    
    test_results = {}
    for dim in ['type', 'frame', 'risk']:
        f1 = sk_f1(all_true[dim], all_preds[dim], average='macro', zero_division=0)
        test_results[dim] = f1
        print(f"  BiLSTM {suffix} | {dim}: test macro-F1 = {f1:.4f}")
    
    return test_results

if __name__ == '__main__':
    results_bilstm = {}
    results_bilstm['text_only']    = train_bilstm(include_kg=False)
    results_bilstm['kg_augmented'] = train_bilstm(include_kg=True)

    with open('results/bilstm_results.json', 'w') as f:
        json.dump(results_bilstm, f, indent=2)
        
    print("\nPHASE 8A COMPLETE — Results saved.")
