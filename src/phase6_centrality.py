"""
PHASE 6: NETWORK ANALYSIS & TEMPORAL CENTRALITY
================================================
Builds 36 monthly directed email graphs (Jan 2000 – Dec 2002).
Computes 4 centrality metrics per employee per month.
Generates phi_G feature vectors for all labeled dataset splits.

Adjusted for Research-Grade pipeline:
- No Gold subset; extracts only from the Silver pool (df_labeled).
- Uses unicode encoding for Python files on Windows.
"""
# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from tqdm import tqdm
from collections import Counter
import os

os.makedirs('graphs', exist_ok=True)
os.makedirs('data/features', exist_ok=True)

print("=" * 65)
print("PHASE 6 — TEMPORAL CENTRALITY ANALYSIS")
print("=" * 65)

# ──────────────────────────────────────────────────────────────────────
# STEP 1 — Build monthly directed graphs
# ──────────────────────────────────────────────────────────────────────
print("\n[STEP 1] Building monthly graphs...")
import sys
sys.path.append(os.path.abspath('src'))
from phase2_preprocess import resolve_alias_with_fuzzy

df = pd.read_parquet('data/processed/emails_clean.parquet')
df = df[df['month_index'].between(0, 35)]

# Get all unique senders (employees)
all_nodes = set(df['sender_canonical'].dropna().unique())
print(f"Total unique senders in corpus: {len(all_nodes)}")

monthly_graphs = {}
memo_canon = {}  # Cache to prevent slow duplicate rapidfuzz calls

for month in tqdm(range(36), desc='Building monthly graphs'):
    month_df = df[df['month_index'] == month][['sender_canonical','recipients']].dropna(subset=['sender_canonical'])
    G = nx.DiGraph()
    G.add_nodes_from(all_nodes)
    
    if len(month_df) == 0:
        monthly_graphs[month] = G
        continue
    
    edge_list = []
    # Using itertuples is faster than iterrows
    for row in month_df.itertuples(index=False):
        sender = row.sender_canonical
        recips_str = str(row.recipients) if pd.notna(row.recipients) else ""
        for recip in recips_str.split(';')[:20]:
            recip = recip.strip()
            if recip and recip != 'nan':
                if recip not in memo_canon:
                    memo_canon[recip] = resolve_alias_with_fuzzy(recip)
                recip_canon = memo_canon[recip]
                
                # Only add edge if the target was successfully resolved to a known employee!
                if recip_canon in all_nodes:
                    edge_list.append((sender, recip_canon))
    
    edge_counts = Counter(edge_list)
    for (u, v), w in edge_counts.items():
        G.add_edge(u, v, weight=w)
    
    monthly_graphs[month] = G

# ──────────────────────────────────────────────────────────────────────
# STEP 2 — Compute centrality for each month
# ──────────────────────────────────────────────────────────────────────
print("\n[STEP 2] Computing centralities...")
employees = sorted(all_nodes)
n_emp = len(employees)
emp_idx = {e: i for i, e in enumerate(employees)}

# Shape: (n_employees, 36_months, 4_metrics)
# Metrics: [in_degree, out_degree, betweenness, closeness]
centrality_tensor = np.zeros((n_emp, 36, 4))

for month in tqdm(range(36), desc='Computing Centrality'):
    G = monthly_graphs[month]
    if G.number_of_edges() == 0:
        continue  
    
    in_deg  = dict(nx.in_degree_centrality(G))
    out_deg = dict(nx.out_degree_centrality(G))
    # Approximation for speed
    approx_k = min(100, G.number_of_nodes())
    between  = nx.betweenness_centrality(G, k=approx_k, normalized=True, seed=42)
    
    # Closeness on largest component to avoid disconnected-node bias
    G_undirected = G.to_undirected()
    components = list(nx.connected_components(G_undirected))
    if components:
        largest_cc = max(components, key=len)
        G_cc = G_undirected.subgraph(largest_cc)
        close = nx.closeness_centrality(G_cc)
    else:
        close = {}
    
    for emp, idx in emp_idx.items():
        centrality_tensor[idx, month, 0] = in_deg.get(emp, 0)
        centrality_tensor[idx, month, 1] = out_deg.get(emp, 0)
        centrality_tensor[idx, month, 2] = between.get(emp, 0)
        centrality_tensor[idx, month, 3] = close.get(emp, 0)

# ──────────────────────────────────────────────────────────────────────
# STEP 3 — Compute derivatives and Z-score normalize
# ──────────────────────────────────────────────────────────────────────
print("\n[STEP 3] Normalizing and Computing Derivatives...")
# Change in betweenness
delta_betweenness = np.diff(centrality_tensor[:, :, 2], axis=1)  # shape: (n_emp, 35)

centrality_normalized = np.zeros_like(centrality_tensor)
for m in range(36):
    for metric in range(4):
        col = centrality_tensor[:, m, metric]
        std = col.std()
        if std > 0:
            centrality_normalized[:, m, metric] = (col - col.mean()) / std
        else:
            centrality_normalized[:, m, metric] = 0

# Pad to match shape 36
delta_padded = np.hstack([delta_betweenness[:, [0]], delta_betweenness])

# ──────────────────────────────────────────────────────────────────────
# STEP 4 — Save centrality matrix
# ──────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────
# STEP 5 — Build phi_G feature vectors for labeled dataset
# ──────────────────────────────────────────────────────────────────────
print("\n[STEP 5] Generating phi_G features...")
def get_phi_g(sender_canonical, month_index):
    if sender_canonical not in emp_idx:
        return np.zeros(8, dtype=np.float32)
    idx = emp_idx[sender_canonical]
    # Ensure m is a valid month index between 0 and 35
    if pd.isna(month_index):
        m = 0
    else:
        m = int(month_index)
    m = max(0, min(35, m))
    
    return np.array([
        centrality_normalized[idx, m, 0],    # in_degree (normalized)
        centrality_normalized[idx, m, 1],    # out_degree (normalized)
        centrality_normalized[idx, m, 2],    # betweenness (normalized)
        centrality_normalized[idx, m, 3],    # closeness (normalized)
        delta_padded[idx, m],                # delta_betweenness
        centrality_tensor[idx, m, 2],        # raw betweenness
        centrality_tensor[idx, m, 0],        # raw in_degree
        m / 35.0                             # time position
    ], dtype=np.float32)

df_labeled = pd.read_parquet('data/labeled/emails_labeled_silver.parquet')

for split_name in ['train', 'val', 'test']:
    meta_path = f'data/features/split_{split_name}.parquet'
    if not os.path.exists(meta_path):
        continue
    
    split_meta = pd.read_parquet(meta_path)
    
    # Merge carefully to keep same order as Phase 4A
    df_indexed = df_labeled.set_index('mid')
    split_indexed = split_meta.set_index('mid')
    df_split = df_indexed.loc[split_indexed.index]
    
    features = []
    for row in df_split.itertuples():
        sender = row.sender_canonical if hasattr(row, 'sender_canonical') else ""
        month = row.month_index if hasattr(row, 'month_index') else 0
        features.append(get_phi_g(sender, month))
        
    phi_g = np.array(features, dtype=np.float32)
    np.save(f'data/features/phi_g_{split_name}.npy', phi_g)
    print(f"phi_G {split_name}: shape {phi_g.shape}, NaN: {np.isnan(phi_g).sum()}")

# ──────────────────────────────────────────────────────────────────────
# STEP 6 — Temporal trajectory plot
# ──────────────────────────────────────────────────────────────────────
print("\n[STEP 6] Saving Trajectories Plot...")
top_senders = df['sender_canonical'].value_counts().head(10).index.tolist()
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
months_label = range(36)
month_ticks = [0, 6, 12, 18, 24, 30, 35]
month_labels = ['Jan 00','Jul 00','Jan 01','Jul 01','Jan 02','Jul 02','Dec 02']
skilling_resign = 19  # Aug 2001

for i, emp in enumerate(top_senders):
    ax = axes[i // 5][i % 5]
    if emp not in emp_idx: continue
    idx = emp_idx[emp]
    ax.plot(months_label, centrality_tensor[idx, :, 2], 'b-', linewidth=2)
    ax.axvline(x=skilling_resign, color='red', linestyle='--', alpha=0.7, label='Skilling Resignation')
    ax.set_title(str(emp)[:25], fontsize=8)
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels, fontsize=6, rotation=45)
    ax.set_ylabel('Betweenness', fontsize=7)
    if i == 0: ax.legend(fontsize=7)

plt.suptitle('Temporal Betweenness Centrality — Top 10 Employees\n(Red dashed = Skilling Resignation Aug 2001)', fontsize=12)
plt.tight_layout()
plt.savefig('graphs/betweenness_trajectories.png', dpi=150, bbox_inches='tight')

# ──────────────────────────────────────────────────────────────────────
# STEP 7 — Statistical validation (Wilcoxon test)
# ──────────────────────────────────────────────────────────────────────
print("\n[STEP 7] Wilcoxon Test...")
# Abs values for change magnitude
all_deltas_stable = np.abs(delta_padded[:, 0:18].flatten())
all_deltas_crisis = np.abs(delta_padded[:, 18:24].flatten())

rng = np.random.default_rng(42)
n_compare = min(len(all_deltas_stable), len(all_deltas_crisis))
stable_sample = rng.choice(all_deltas_stable, size=n_compare, replace=False)
crisis_sample = rng.choice(all_deltas_crisis, size=n_compare, replace=False)

stat, p_value = wilcoxon(crisis_sample, stable_sample, alternative='greater')
print(f"Wilcoxon test — crisis > stable |delta_betweenness|:")
print(f"  n_compare={n_compare}, Statistic: {stat:.2f}, p-value: {p_value:.4f}")
if p_value < 0.05:
    print("  SIGNIFICANT — temporal KG signal confirmed (paper claim holds)")
else:
    print("  NOT SIGNIFICANT — review centrality computation or crisis month boundaries")

print("\nPHASE 6 COMPLETE")
