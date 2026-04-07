# ORGDISCLOSE — Implementation Plan
# Part 4 of 6: Phase 6 (Network Analysis) + Phase 7 (ML Models) + Phase 8 (DL Models)

---

## PHASE 6: NETWORK ANALYSIS & TEMPORAL CENTRALITY

### Duration: 2–3 days
### Goal: Monthly centrality matrices + Δ_betweenness derivatives → phi_G feature vectors

---

### 6.1 Exact Tasks

Task 1: Build 36 monthly directed email graphs (Jan 2000 – Dec 2002) using NetworkX
Task 2: Compute 4 centrality metrics per employee per month
Task 3: Compute first-order derivatives (Δ_betweenness)
Task 4: Z-score normalize per metric per month
Task 5: For each email in labeled set, attach sender's phi_G at email's month_index
Task 6: Plot temporal betweenness trajectories for top-10 employees
Task 7: Annotate Aug 14, 2001 (Skilling resignation) on all plots

---

### 6.2 Validation Standard (Must Pass Before Phase 7)

- [ ] centrality_matrix.parquet shape: (n_employees × 36_months × 4_metrics)
- [ ] phi_G for every labeled email — zero NaN values
- [ ] Plot shows visible betweenness spike for senior executives post-Aug 2001
- [ ] Wilcoxon test: Δ_betweenness in crisis months significantly > stable months (p < 0.05)
- [ ] data/features/phi_g_{train,val,test}.npy saved

---

### 6.3 IMPLEMENTATION PROMPT — PHASE 6: TEMPORAL CENTRALITY

```
=== IMPLEMENTATION PROMPT: PHASE 6 — TEMPORAL CENTRALITY ANALYSIS ===

CONTEXT:
- Input: data/processed/emails_clean.parquet (full corpus, ~220k emails)
  Columns needed: sender_canonical, recipients, month_index
- Output:
  graphs/centrality_matrix.parquet
  data/features/phi_g_{train,val,test}.npy
  graphs/betweenness_trajectories.pdf

Write phase6_centrality.py:

STEP 1 — Build monthly directed graphs (use FULL corpus, not just labeled):
import networkx as nx
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, zscore
from tqdm import tqdm

df = pd.read_parquet('data/processed/emails_clean.parquet')
df = df[df['month_index'].between(0, 35)]

# Get all unique senders/employees
all_nodes = set(df['sender_canonical'].dropna().unique())
print(f"Total unique senders: {len(all_nodes)}")

# Build one graph per month
monthly_graphs = {}
for month in range(36):
    month_df = df[df['month_index'] == month]
    G = nx.DiGraph()
    G.add_nodes_from(all_nodes)  # Ensure all nodes present even if no emails
    for _, row in month_df.iterrows():
        sender = row['sender_canonical']
        if not isinstance(sender, str): continue
        recips_str = str(row.get('recipients', ''))
        for recip in recips_str.split(';')[:20]:
            recip = recip.strip()
            if not recip or recip == 'nan': continue
            if G.has_edge(sender, recip):
                G[sender][recip]['weight'] += 1
            else:
                G.add_edge(sender, recip, weight=1)
    monthly_graphs[month] = G
    if month % 6 == 0:
        print(f"Month {month}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

STEP 2 — Compute centrality for each month:
NOTE: betweenness_centrality is O(n³) — for speed, use k=100 approximation
employees = sorted(all_nodes)
n_emp = len(employees)
emp_idx = {e: i for i, e in enumerate(employees)}

# Shape: (n_employees, 36_months, 4_metrics)
# Metrics: [in_degree, out_degree, betweenness, closeness]
centrality_tensor = np.zeros((n_emp, 36, 4))

for month in tqdm(range(36), desc='Computing centrality'):
    G = monthly_graphs[month]
    if G.number_of_edges() == 0:
        continue  # Skip empty months
    
    # Degree centrality (fast, exact)
    in_deg  = dict(nx.in_degree_centrality(G))
    out_deg = dict(nx.out_degree_centrality(G))
    
    # Betweenness (approximate for speed) — k=min(100, n_nodes)
    approx_k = min(100, G.number_of_nodes())
    between  = nx.betweenness_centrality(G, k=approx_k, normalized=True, seed=42)
    
    # Closeness (computed on largest weakly connected component)
    G_undirected = G.to_undirected()
    largest_cc = max(nx.connected_components(G_undirected), key=len)
    G_cc = G_undirected.subgraph(largest_cc)
    close = nx.closeness_centrality(G_cc)
    
    for emp, idx in emp_idx.items():
        centrality_tensor[idx, month, 0] = in_deg.get(emp, 0)
        centrality_tensor[idx, month, 1] = out_deg.get(emp, 0)
        centrality_tensor[idx, month, 2] = between.get(emp, 0)
        centrality_tensor[idx, month, 3] = close.get(emp, 0)

STEP 3 — Compute derivatives and Z-score normalize:
# First-order derivative for betweenness (metric index 2)
# Shape of derivatives: (n_employees, 35_months) — one less than centrality
delta_betweenness = np.diff(centrality_tensor[:, :, 2], axis=1)  # shape: (n_emp, 35)

# Z-score normalize per month per metric (axis=0 = across employees for each month)
centrality_normalized = np.zeros_like(centrality_tensor)
for m in range(36):
    for metric in range(4):
        col = centrality_tensor[:, m, metric]
        std = col.std()
        if std > 0:
            centrality_normalized[:, m, metric] = (col - col.mean()) / std
        else:
            centrality_normalized[:, m, metric] = 0

# Pad delta to match centrality shape (repeat first value)
delta_padded = np.hstack([delta_betweenness[:, [0]], delta_betweenness])  # (n_emp, 36)

STEP 4 — Save centrality matrix:
import os
os.makedirs('graphs', exist_ok=True)
# Save as structured parquet
records = []
for emp, idx in emp_idx.items():
    for month in range(36):
        records.append({
            'employee': emp,
            'month_index': month,
            'in_degree': centrality_normalized[idx, month, 0],
            'out_degree': centrality_normalized[idx, month, 1],
            'betweenness': centrality_normalized[idx, month, 2],
            'closeness': centrality_normalized[idx, month, 3],
            'delta_betweenness': delta_padded[idx, month]
        })
cm_df = pd.DataFrame(records)
cm_df.to_parquet('graphs/centrality_matrix.parquet', index=False)
print(f"Saved centrality_matrix.parquet: {len(cm_df)} rows")

STEP 5 — Build phi_G feature vectors for each labeled email:
# phi_G = [in_degree, out_degree, betweenness, closeness, delta_betweenness,
#           raw_betweenness, raw_delta, month_index_normalized]  → R^8

def get_phi_g(sender_canonical, month_index):
    if sender_canonical not in emp_idx:
        return np.zeros(8, dtype=np.float32)
    idx = emp_idx[sender_canonical]
    m = int(month_index)
    m = max(0, min(35, m))
    return np.array([
        centrality_normalized[idx, m, 0],    # in_degree (normalized)
        centrality_normalized[idx, m, 1],    # out_degree (normalized)
        centrality_normalized[idx, m, 2],    # betweenness (normalized)
        centrality_normalized[idx, m, 3],    # closeness (normalized)
        delta_padded[idx, m],                # Δ_betweenness
        centrality_tensor[idx, m, 2],        # raw betweenness (unnormalized)
        centrality_tensor[idx, m, 0],        # raw in_degree
        m / 35.0                             # time position (0-1)
    ], dtype=np.float32)

# Apply to each split
for split_name in ['train', 'val', 'test']:
    if split_name == 'train':
        df_labeled = pd.concat([
            pd.read_parquet('data/labeled/emails_labeled_gold.parquet'),
            pd.read_parquet('data/labeled/emails_labeled_silver.parquet')
        ])
    else:
        df_labeled = pd.read_parquet('data/labeled/emails_labeled_gold.parquet')
    
    split_meta = pd.read_parquet(f'data/features/split_{split_name}.parquet')
    df_split = df_labeled[df_labeled['mid'].isin(split_meta['mid'])].merge(split_meta[['mid']], on='mid')
    
    phi_g = np.array([
        get_phi_g(row['sender_canonical'], row['month_index'])
        for _, row in df_split.iterrows()
    ], dtype=np.float32)
    
    np.save(f'data/features/phi_g_{split_name}.npy', phi_g)
    print(f"phi_G {split_name}: shape {phi_g.shape}, NaN: {np.isnan(phi_g).sum()}")

STEP 6 — Temporal trajectory plot:
# Plot betweenness for top 10 employees by total email volume
top_senders = df['sender_canonical'].value_counts().head(10).index.tolist()
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
months_label = range(36)
month_ticks = [0, 6, 12, 18, 24, 30, 35]
month_labels = ['Jan 00','Jul 00','Jan 01','Jul 01','Jan 02','Jul 02','Dec 02']
skilling_resign = 19  # Aug 2001 = month_index 19

for i, emp in enumerate(top_senders):
    ax = axes[i // 5][i % 5]
    if emp not in emp_idx: continue
    idx = emp_idx[emp]
    ax.plot(months_label, centrality_tensor[idx, :, 2], 'b-', linewidth=2)
    ax.axvline(x=skilling_resign, color='red', linestyle='--', alpha=0.7, label='Skilling Resignation')
    ax.set_title(emp[:25], fontsize=8)
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels, fontsize=6, rotation=45)
    ax.set_ylabel('Betweenness Centrality', fontsize=7)
    if i == 0: ax.legend(fontsize=7)

plt.suptitle('Temporal Betweenness Centrality — Top 10 Employees\n(Red dashed = Skilling Resignation Aug 2001)', fontsize=12)
plt.tight_layout()
plt.savefig('graphs/betweenness_trajectories.pdf', dpi=150, bbox_inches='tight')
plt.savefig('graphs/betweenness_trajectories.png', dpi=150, bbox_inches='tight')
print("Saved trajectory plot")

STEP 7 — Statistical validation (Wilcoxon test):
# Test: is Δ_betweenness significantly larger in crisis months vs stable months?
# Crisis months: 18-23 (Jul 2001 - Dec 2001)
# Stable months: 0-17 (Jan 2000 - Jun 2001)
all_deltas_stable = delta_padded[:, 0:18].flatten()
all_deltas_crisis = delta_padded[:, 18:24].flatten()
stat, p_value = wilcoxon(
    all_deltas_crisis[:len(all_deltas_stable)],
    all_deltas_stable[:len(all_deltas_crisis)],
    alternative='greater'
)
print(f"\nWilcoxon test — crisis > stable Δ_betweenness:")
print(f"  Statistic: {stat:.2f}, p-value: {p_value:.4f}")
if p_value < 0.05:
    print("  SIGNIFICANT — KG temporal contribution confirmed")
else:
    print("  NOT SIGNIFICANT — review centrality computation")

print("PHASE 6 COMPLETE")
=== END OF PROMPT ===
```

---

## PHASE 7: ML MODELS — XGBOOST + RANDOM FOREST

### Duration: 2–3 days
### Goal: 2 trained ML models, text-only and KG-augmented variants, evaluation on Gold test set

---

### 7.1 Validation Standard (Must Pass)

- [ ] Both models beat random baseline (macro-F1 > 0.30)
- [ ] XGBoost + KG features outperforms XGBoost text-only on disc_type by ≥ 1%
- [ ] Feature importance plot saved for XGBoost (top 30 features)
- [ ] Results saved to results/ml_results.json
- [ ] Both text-only and KG-augmented variants evaluated (needed for ablation table)

---

### 7.2 IMPLEMENTATION PROMPT — PHASE 7: ML MODELS

```
=== IMPLEMENTATION PROMPT: PHASE 7 — ML MODELS (XGBoost + Random Forest) ===

CONTEXT:
- Features: TF-IDF (sparse), Empath (dense), phi_G (dense R^8)
- Labels: y_{train,val,test}_{type,framing,risk} as .npy files
- Output: models/ml_{xgb,rf}_{type,framing,risk}_{text,kg}.pkl + results/ml_results.json

Write phase7_ml_models.py:

STEP 1 — Load all features:
from scipy.sparse import load_npz, hstack
import numpy as np, pandas as pd, pickle, json, os

X_train_tfidf = load_npz('data/features/tfidf_train.npz')
X_val_tfidf   = load_npz('data/features/tfidf_val.npz')
X_test_tfidf  = load_npz('data/features/tfidf_test.npz')

empath_train = np.load('data/features/empath_train.npy')
empath_val   = np.load('data/features/empath_val.npy')
empath_test  = np.load('data/features/empath_test.npy')

phi_g_train = np.load('data/features/phi_g_train.npy')
phi_g_val   = np.load('data/features/phi_g_val.npy')
phi_g_test  = np.load('data/features/phi_g_test.npy')

from scipy.sparse import csr_matrix

# Text-only features: TF-IDF + Empath
X_train_text = hstack([X_train_tfidf, csr_matrix(empath_train)])
X_val_text   = hstack([X_val_tfidf,   csr_matrix(empath_val)])
X_test_text  = hstack([X_test_tfidf,  csr_matrix(empath_test)])

# KG-augmented features: TF-IDF + Empath + phi_G
X_train_kg = hstack([X_train_tfidf, csr_matrix(empath_train), csr_matrix(phi_g_train)])
X_val_kg   = hstack([X_val_tfidf,   csr_matrix(empath_val),   csr_matrix(phi_g_val)])
X_test_kg  = hstack([X_test_tfidf,  csr_matrix(empath_test),  csr_matrix(phi_g_test)])

# Labels (load all 3 dimensions)
labels = {}
for dim in ['type', 'framing', 'risk']:
    for split in ['train', 'val', 'test']:
        labels[f'y_{split}_{dim}'] = np.load(f'data/features/y_{split}_{dim}.npy')

STEP 2 — Compute class weights for imbalance:
from sklearn.utils.class_weight import compute_class_weight
def get_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))

STEP 3 — Train and evaluate XGBoost (Model 1):
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score
import warnings; warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

results = {}

for feat_name, X_tr, X_va, X_te in [
    ('text_only', X_train_text, X_val_text, X_test_text),
    ('kg_augmented', X_train_kg, X_val_kg, X_test_kg)
]:
    results[f'xgboost_{feat_name}'] = {}
    for dim in ['type', 'framing', 'risk']:
        y_tr = labels[f'y_train_{dim}']
        y_va = labels[f'y_val_{dim}']
        y_te = labels[f'y_test_{dim}']
        
        cw = get_class_weights(y_tr)
        sample_weights = np.array([cw[y] for y in y_tr])
        
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1
        )
        model.fit(
            X_tr, y_tr,
            sample_weight=sample_weights,
            eval_set=[(X_va, y_va)],
            verbose=False,
            early_stopping_rounds=20
        )
        
        y_pred = model.predict(X_te)
        macro_f1 = f1_score(y_te, y_pred, average='macro')
        report = classification_report(y_te, y_pred, output_dict=True)
        
        results[f'xgboost_{feat_name}'][dim] = {
            'macro_f1': macro_f1,
            'best_iteration': model.best_iteration,
            'report': report
        }
        print(f"XGBoost {feat_name} | {dim}: macro-F1 = {macro_f1:.4f}")
        
        # Save model
        with open(f'models/xgb_{feat_name}_{dim}.pkl', 'wb') as f:
            pickle.dump(model, f)

STEP 4 — Train and evaluate Random Forest (Model 2):
from sklearn.ensemble import RandomForestClassifier

for feat_name, X_tr, X_va, X_te in [
    ('text_only', X_train_text, X_val_text, X_test_text),
    ('kg_augmented', X_train_kg, X_val_kg, X_test_kg)
]:
    results[f'random_forest_{feat_name}'] = {}
    for dim in ['type', 'framing', 'risk']:
        y_tr = labels[f'y_train_{dim}']
        y_te = labels[f'y_test_{dim}']
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_tr.toarray() if hasattr(X_tr, 'toarray') else X_tr, y_tr)
        # NOTE: RF needs dense array — use X_tr.toarray() only if memory allows
        # If OOM: use only top 2000 TF-IDF features to reduce memory
        
        y_pred = model.predict(X_te.toarray() if hasattr(X_te, 'toarray') else X_te)
        macro_f1 = f1_score(y_te, y_pred, average='macro')
        results[f'random_forest_{feat_name}'][dim] = {'macro_f1': macro_f1}
        print(f"Random Forest {feat_name} | {dim}: macro-F1 = {macro_f1:.4f}")
        
        with open(f'models/rf_{feat_name}_{dim}.pkl', 'wb') as f:
            pickle.dump(model, f)

MEMORY NOTE: If Random Forest goes OOM with full TF-IDF (10k features):
  Use: from sklearn.feature_selection import SelectKBest, chi2
  selector = SelectKBest(chi2, k=2000)
  X_train_text_reduced = selector.fit_transform(X_train_text.abs(), labels['y_train_type'])
  Save selector and use k=2000 version for RF only. XGBoost handles sparse fine.

STEP 5 — Ablation delta computation:
print("\n=== ABLATION ANALYSIS ===")
for model_name in ['xgboost', 'random_forest']:
    for dim in ['type', 'framing', 'risk']:
        text_f1 = results[f'{model_name}_text_only'][dim]['macro_f1']
        kg_f1   = results[f'{model_name}_kg_augmented'][dim]['macro_f1']
        delta   = kg_f1 - text_f1
        print(f"{model_name} | {dim}: Δ = {delta:+.4f} ({'KG HELPS' if delta>0.02 else 'MARGINAL' if delta>0 else 'KG HURTS'})")

STEP 6 — Save results:
with open('results/ml_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print("PHASE 7 COMPLETE — ML results saved")
=== END OF PROMPT ===
```

---

## PHASE 8: DL MODELS — BiLSTM + DeBERTa

### Duration: 4–5 days
### Goal: 2 DL models trained with text-only and KG-augmented variants; BERT ablation proves KG value

---

### 8.1 Validation Standard (Must Pass)

- [ ] DeBERTa macro-F1 > XGBoost macro-F1 on disc_type (expected +5–10%)
- [ ] DeBERTa + KG > DeBERTa text-only by ≥ 2% on at least 1 dimension  ← **PAPER CLAIM**
- [ ] Training loss curves saved as plots (no divergence)
- [ ] Best model checkpoint saved (lowest val loss)
- [ ] Results saved to results/dl_results.json

---

### 8.2 IMPLEMENTATION PROMPT — PHASE 8A: BiLSTM MODEL

```
=== IMPLEMENTATION PROMPT: PHASE 8A — BiLSTM WITH ATTENTION ===

CONTEXT:
- Hardware: RTX 2050 (4GB VRAM)
- Input: emails_labeled_gold + silver parquets
- Output: models/bilstm_{text,kg}_best.pt + results/bilstm_results.json

Write phase8a_bilstm.py:

STEP 1 — Build vocabulary and tokenizer:
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pandas as pd, numpy as np

# Load data
train_meta = pd.read_parquet('data/features/split_train.parquet')
val_meta   = pd.read_parquet('data/features/split_val.parquet')
test_meta  = pd.read_parquet('data/features/split_test.parquet')

gold   = pd.read_parquet('data/labeled/emails_labeled_gold.parquet')
silver = pd.read_parquet('data/labeled/emails_labeled_silver.parquet')
all_labeled = pd.concat([gold, silver])

train_df = all_labeled[all_labeled['mid'].isin(train_meta['mid'])]
val_df   = gold[gold['mid'].isin(val_meta['mid'])]
test_df  = gold[gold['mid'].isin(test_meta['mid'])]

# Build vocab from train only
MAX_VOCAB = 30000
MAX_LEN = 200  # 200 tokens sufficient for emails

from collections import Counter
import re

def tokenize(text):
    if not isinstance(text, str): return []
    return re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())[:MAX_LEN]

counter = Counter()
for text in train_df['body_clean'].fillna(''):
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

STEP 2 — Label encoding:
LABEL_MAPS = {
    'disclosure_type': {'FINANCIAL':0,'PII':1,'STRATEGIC':2,'LEGAL':3,'RELATIONAL':4,'NONE':5},
    'framing':         {'PROTECTED':0,'UNPROTECTED':1,'NA':2},
    'risk_tier':       {'NONE':0,'LOW':1,'HIGH':2}
}

STEP 3 — Dataset class:
class EmailDataset(Dataset):
    def __init__(self, df, phi_g_array, include_kg=True):
        self.texts   = [encode(t) for t in df['body_clean'].fillna('')]
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

STEP 4 — BiLSTM Model with Attention:
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

STEP 5 — Training loop (run for both text_only and kg_augmented):
from sklearn.metrics import f1_score as sk_f1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on: {device}")

def train_bilstm(include_kg=True, max_epochs=15, patience=3):
    suffix = 'kg' if include_kg else 'text'
    
    train_ds = EmailDataset(train_df, phi_g_train, include_kg)
    val_ds   = EmailDataset(val_df,   phi_g_val,   include_kg)
    test_ds  = EmailDataset(test_df,  phi_g_test,  include_kg)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64)
    test_loader  = DataLoader(test_ds,  batch_size=64)
    
    model = BiLSTMClassifier(len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Weighted losses
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
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
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
        
        val_f1 = sk_f1(val_true_type, val_preds_type, average='macro')
        print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, val_macro_F1={val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f'models/bilstm_{suffix}_best.pt')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Test evaluation
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
        f1 = sk_f1(all_true[dim], all_preds[dim], average='macro')
        test_results[dim] = f1
        print(f"BiLSTM {suffix} | {dim}: test macro-F1 = {f1:.4f}")
    
    return test_results

results_bilstm = {}
results_bilstm['text_only']    = train_bilstm(include_kg=False)
results_bilstm['kg_augmented'] = train_bilstm(include_kg=True)

import json
with open('results/bilstm_results.json', 'w') as f:
    json.dump(results_bilstm, f, indent=2)
print("PHASE 8A COMPLETE")
=== END OF PROMPT ===
```

---

### 8.3 IMPLEMENTATION PROMPT — PHASE 8B: DeBERTa-v3-small (PRIMARY MODEL)

```
=== IMPLEMENTATION PROMPT: PHASE 8B — DeBERTa-v3-SMALL MULTI-TASK CLASSIFIER ===

CONTEXT:
- Model: microsoft/deberta-v3-small (44M params, fits on 4GB VRAM with batch=16)
- sequence_length: 256 tokens (email average is 187 tokens)
- Multi-task: 2 heads (disc_type, framing) — risk_tier is rule-computed
- Run TWICE: text_only and kg_augmented variants
- Output: models/deberta_{text,kg}_best/ + results/deberta_results.json

Write phase8b_deberta.py:

STEP 1 — Dataset:
from transformers import AutoTokenizer, AutoModel
import torch, torch.nn as nn, pandas as pd, numpy as np
from torch.utils.data import Dataset, DataLoader

tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small')

LABEL_MAPS = {
    'disclosure_type': {'FINANCIAL':0,'PII':1,'STRATEGIC':2,'LEGAL':3,'RELATIONAL':4,'NONE':5},
    'framing':         {'PROTECTED':0,'UNPROTECTED':1,'NA':2},
}

class DisclosureDataset(Dataset):
    def __init__(self, df, phi_g_array, include_kg=True, max_len=256):
        self.encodings = tokenizer(
            list(df['body_clean'].fillna('')),
            truncation=True, padding='max_length',
            max_length=max_len, return_tensors='pt'
        )
        self.y_type  = [LABEL_MAPS['disclosure_type'].get(str(y), 5) for y in df['disclosure_type']]
        self.y_frame = [LABEL_MAPS['framing'].get(str(y), 2) for y in df['framing']]
        self.phi_g   = torch.tensor(phi_g_array if include_kg else np.zeros_like(phi_g_array), dtype=torch.float)
    
    def __len__(self): return len(self.y_type)
    def __getitem__(self, i):
        return {
            'input_ids':      self.encodings['input_ids'][i],
            'attention_mask': self.encodings['attention_mask'][i],
            'token_type_ids': self.encodings.get('token_type_ids', {i: torch.zeros(256, dtype=torch.long)})[i]
                              if 'token_type_ids' in self.encodings else torch.zeros(256, dtype=torch.long),
            'phi_g':          self.phi_g[i],
            'y_type':         torch.tensor(self.y_type[i],  dtype=torch.long),
            'y_frame':        torch.tensor(self.y_frame[i], dtype=torch.long),
        }

STEP 2 — Model Architecture:
class OrgDiscloseModel(nn.Module):
    def __init__(self, phi_g_dim=8, proj_dim=64, num_type=6, num_frame=3):
        super().__init__()
        self.deberta = AutoModel.from_pretrained('microsoft/deberta-v3-small')
        hidden_size = self.deberta.config.hidden_size  # 768
        
        # KG feature projector
        self.kg_proj = nn.Sequential(
            nn.Linear(phi_g_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        combined = hidden_size + proj_dim
        self.classifier_type  = nn.Sequential(
            nn.Linear(combined, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, num_type)
        )
        self.classifier_frame = nn.Sequential(
            nn.Linear(combined, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, num_frame)
        )
    
    def forward(self, input_ids, attention_mask, token_type_ids, phi_g):
        out = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_emb = out.last_hidden_state[:, 0, :]   # [CLS] token: R^768
        kg_emb  = self.kg_proj(phi_g)               # R^64
        combined = torch.cat([cls_emb, kg_emb], dim=1)   # R^832
        
        logits_type  = self.classifier_type(combined)
        logits_frame = self.classifier_frame(combined)
        return logits_type, logits_frame

STEP 3 — Training function (run for text_only and kg_augmented):
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score as sk_f1
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load data splits (same as Phase 8A)
# ... [use same loading code as in Phase 8A STEP above]

def train_deberta(include_kg=True, lr=2e-5, epochs=5, batch_size=16, patience=3):
    suffix = 'kg' if include_kg else 'text'
    
    train_ds = DisclosureDataset(train_df, phi_g_train, include_kg)
    val_ds   = DisclosureDataset(val_df,   phi_g_val,   include_kg)
    test_ds  = DisclosureDataset(test_df,  phi_g_test,  include_kg)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)
    
    model = OrgDiscloseModel().to(device)
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.01
    )
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1*total_steps),
        num_training_steps=total_steps
    )
    
    w_type  = torch.tensor([1.5,3.0,2.0,2.5,2.0,0.5]).to(device)
    w_frame = torch.tensor([2.0,1.0,0.5]).to(device)
    crit_t = nn.CrossEntropyLoss(weight=w_type)
    crit_f = nn.CrossEntropyLoss(weight=w_frame)
    
    best_val_f1, no_improve = 0.0, 0
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            ttype= batch['token_type_ids'].to(device)
            phi  = batch['phi_g'].to(device)
            y_t  = batch['y_type'].to(device)
            y_f  = batch['y_frame'].to(device)
            
            with torch.cuda.amp.autocast():
                l_t, l_f = model(ids, mask, ttype, phi)
                loss = 0.6*crit_t(l_t, y_t) + 0.4*crit_f(l_f, y_f)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        val_p, val_t = [], []
        with torch.no_grad():
            for batch in val_loader:
                with torch.cuda.amp.autocast():
                    l_t, _ = model(
                        batch['input_ids'].to(device),
                        batch['attention_mask'].to(device),
                        batch['token_type_ids'].to(device),
                        batch['phi_g'].to(device)
                    )
                val_p.extend(l_t.argmax(1).cpu().numpy())
                val_t.extend(batch['y_type'].numpy())
        
        val_f1 = sk_f1(val_t, val_p, average='macro')
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"[{suffix}] Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, val_F1={val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            os.makedirs(f'models/deberta_{suffix}_best', exist_ok=True)
            model.deberta.save_pretrained(f'models/deberta_{suffix}_best')
            torch.save({
                'kg_proj': model.kg_proj.state_dict(),
                'head_type': model.classifier_type.state_dict(),
                'head_frame': model.classifier_frame.state_dict()
            }, f'models/deberta_{suffix}_best/heads.pt')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping epoch {epoch+1}")
                break
    
    # Test evaluation
    # [Load best model and evaluate on test_loader — same pattern as training loop]
    # Return test macro-F1 per dimension
    return {'best_val_f1': best_val_f1, 'train_losses': train_losses}

# Run both variants
results_deberta = {}
results_deberta['text_only']    = train_deberta(include_kg=False)
results_deberta['kg_augmented'] = train_deberta(include_kg=True)

# Compute ablation delta
delta = results_deberta['kg_augmented']['best_val_f1'] - results_deberta['text_only']['best_val_f1']
print(f"\n=== ABLATION: KG contribution = {delta:+.4f} ===")
if delta >= 0.02:
    print("CONFIRMED: KG adds ≥ 2% — paper contribution proven")
else:
    print("WARNING: KG contribution marginal — check phi_G feature quality")

import json
with open('results/deberta_results.json', 'w') as f:
    json.dump(results_deberta, f, indent=2)
print("PHASE 8B COMPLETE")
=== END OF PROMPT ===
```
