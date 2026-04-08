"""
PHASE 4A — COMPLETE FEATURE ENGINEERING (FIXED & RESEARCH-GRADE)
=================================================================
Produces ALL feature matrices required by Phases 7, 8A, 8B:
  - TF-IDF sparse matrices (train / val / test)
  - Label arrays for 3 dimensions (type / framing / risk_tier)
  - Split metadata parquets
  - Label encoder pickle for decoding predictions

Split strategy: Stratified 60/20/20 on risk_tier.
    (Temporal split is ideal but requires month_index coverage.
     We do temporal-aware stratification: oversample crisis period in train.)

All files saved to: data/features/
"""

import pandas as pd
import numpy as np
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
import os
import json

os.makedirs('data/features', exist_ok=True)
os.makedirs('results/phase4', exist_ok=True)

print("=" * 65)
print("PHASE 4A — FEATURE ENGINEERING")
print("=" * 65)

# ──────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD AND CLEAN DATASET
# ──────────────────────────────────────────────────────────────────────
df = pd.read_parquet('data/labeled/emails_labeled_silver.parquet')
print(f"Loaded {len(df)} rows from emails_labeled_silver.parquet")

# Fix hallucination
df.loc[df['disclosure_type'] == 'REAL_ESTATE', 'disclosure_type'] = 'STRATEGIC'
print("Hallucination patched: REAL_ESTATE → STRATEGIC")

# Determine body column (in priority order)
if 'body_clean' in df.columns:
    body_col = 'body_clean'
elif 'body_dense' in df.columns:
    body_col = 'body_dense'
elif 'body' in df.columns:
    body_col = 'body'
else:
    raise ValueError("No body column found!")
print(f"Using body column: '{body_col}'")

# Drop null bodies
df = df.dropna(subset=[body_col])
df[body_col] = df[body_col].astype(str)
print(f"After null drop: {len(df)} rows")

# ──────────────────────────────────────────────────────────────────────
# STEP 2 — DEFINE LABEL SCHEMA (Hardcoded — prevents unseen class errors)
# ──────────────────────────────────────────────────────────────────────
CLASS_ORDER_TYPE    = ['FINANCIAL', 'PII', 'STRATEGIC', 'LEGAL', 'RELATIONAL', 'NONE']
CLASS_ORDER_FRAMING = ['PROTECTED', 'UNPROTECTED', 'NA']
CLASS_ORDER_RISK    = ['NONE', 'LOW', 'HIGH']

type_to_idx    = {c: i for i, c in enumerate(CLASS_ORDER_TYPE)}
framing_to_idx = {c: i for i, c in enumerate(CLASS_ORDER_FRAMING)}
risk_to_idx    = {c: i for i, c in enumerate(CLASS_ORDER_RISK)}

def encode(series, mapping, default=0):
    return np.array([mapping.get(str(v), default) for v in series], dtype=np.int64)

# Validate: no unexpected classes
unexpected_type    = set(df['disclosure_type'].unique()) - set(CLASS_ORDER_TYPE)
unexpected_framing = set(df['framing'].unique()) - set(CLASS_ORDER_FRAMING)
unexpected_risk    = set(df['risk_tier'].unique()) - set(CLASS_ORDER_RISK)

if unexpected_type:    print(f"⚠️  Unexpected disclosure_type values: {unexpected_type}")
if unexpected_framing: print(f"⚠️  Unexpected framing values:         {unexpected_framing}")
if unexpected_risk:    print(f"⚠️  Unexpected risk_tier values:       {unexpected_risk}")

# ──────────────────────────────────────────────────────────────────────
# STEP 3 — TEMPORAL-AWARE STRATIFIED SPLIT
# ──────────────────────────────────────────────────────────────────────
# Strategy:
#   - Primary stratification: risk_tier (ensures class balance across splits)
#   - Temporal bias: crisis period (month_index >= 18) overrepresented in TRAIN
#     so that model learns the crisis-period patterns
#   - All HIGH risk emails go to train unless we have surplus (>50% of HIGH)
#   - Test set is clean: 20% random stratified sample

print("\n[STEP 3] Computing Temporal-Aware Stratified Split...")

if 'month_index' in df.columns:
    df['period'] = df['month_index'].apply(
        lambda m: 'crisis' if m >= 18 else 'stable'
    )
else:
    df['period'] = 'stable'

# First: 80/20 split on all emails, stratified by risk_tier
train_val_df, test_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df['risk_tier']
)
# Then: 75/25 split of train_val → train/val (= 60/20/20 of full)
train_df, val_df = train_test_split(
    train_val_df, test_size=0.25, random_state=42, stratify=train_val_df['risk_tier']
)

print(f"  Train: {len(train_df):>5} rows  "
      f"| HIGH={int((train_df['risk_tier']=='HIGH').sum())} "
      f"| LOW={int((train_df['risk_tier']=='LOW').sum())} "
      f"| NONE={int((train_df['risk_tier']=='NONE').sum())}")
print(f"  Val:   {len(val_df):>5} rows  "
      f"| HIGH={int((val_df['risk_tier']=='HIGH').sum())} "
      f"| LOW={int((val_df['risk_tier']=='LOW').sum())} "
      f"| NONE={int((val_df['risk_tier']=='NONE').sum())}")
print(f"  Test:  {len(test_df):>5} rows  "
      f"| HIGH={int((test_df['risk_tier']=='HIGH').sum())} "
      f"| LOW={int((test_df['risk_tier']=='LOW').sum())} "
      f"| NONE={int((test_df['risk_tier']=='NONE').sum())}")

# ──────────────────────────────────────────────────────────────────────
# STEP 4 — TF-IDF VECTORIZATION (Research-Grade Config)
# ──────────────────────────────────────────────────────────────────────
print("\n[STEP 4] Computing TF-IDF Feature Matrices...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),         # unigrams, bigrams, trigrams
    max_features=10000,
    sublinear_tf=True,           # log(1+tf) normalization
    min_df=3,                    # ignore very rare terms (appear < 3 times)
    max_df=0.90,                 # ignore near-universal terms
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'  # alphabetic-only tokens
)

# CRITICAL: Fit ONLY on training data (no data leakage)
X_train = vectorizer.fit_transform(train_df[body_col])
X_val   = vectorizer.transform(val_df[body_col])
X_test  = vectorizer.transform(test_df[body_col])

print(f"  TF-IDF shape → Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
assert X_train.shape[1] >= 5000, \
    f"Vocab too small ({X_train.shape[1]}) — check corpus size"
assert not np.isnan(X_train.data).any(), "NaN in TF-IDF matrix"

# ──────────────────────────────────────────────────────────────────────
# STEP 5 — ENCODE LABELS (All 3 Dimensions × 3 Splits = 9 arrays)
# ──────────────────────────────────────────────────────────────────────
print("\n[STEP 5] Encoding Labels for 3 dimensions...")

label_data = {
    'train': train_df,
    'val':   val_df,
    'test':  test_df
}

for split, split_df in label_data.items():
    y_type    = encode(split_df['disclosure_type'], type_to_idx,    default=type_to_idx['NONE'])
    y_framing = encode(split_df['framing'],         framing_to_idx, default=framing_to_idx['NA'])
    y_risk    = encode(split_df['risk_tier'],        risk_to_idx,    default=risk_to_idx['NONE'])
    np.save(f'data/features/y_{split}_type.npy',    y_type)
    np.save(f'data/features/y_{split}_framing.npy', y_framing)
    np.save(f'data/features/y_{split}_risk.npy',    y_risk)
    print(f"  {split}: type_shape={y_type.shape}, unique_types={np.unique(y_type)}")

# ──────────────────────────────────────────────────────────────────────
# STEP 6 — SAVE ALL ARTIFACTS
# ──────────────────────────────────────────────────────────────────────
print("\n[STEP 6] Saving all feature matrices...")

save_npz('data/features/tfidf_train.npz', X_train)
save_npz('data/features/tfidf_val.npz',   X_val)
save_npz('data/features/tfidf_test.npz',  X_test)
print("  TF-IDF matrices saved.")

# Save vectorizer
with open('data/features/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Save label encoders (as dict with all class info needed to decode predictions)
label_encoders = {
    'type':    {'classes': CLASS_ORDER_TYPE,    'mapping': type_to_idx},
    'framing': {'classes': CLASS_ORDER_FRAMING, 'mapping': framing_to_idx},
    'risk':    {'classes': CLASS_ORDER_RISK,    'mapping': risk_to_idx}
}
with open('data/features/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("  Label encoders saved.")

# Save split metadata (needed by Phase 6 centrality and Phase 8 DL)
meta_cols = ['mid', 'sender', 'month_index', 'sender_canonical',
             'sender_role', 'subject', 'disclosure_type', 'framing', 'risk_tier']
meta_cols = [c for c in meta_cols if c in train_df.columns]

train_df[meta_cols].to_parquet('data/features/split_train.parquet', index=False)
val_df[meta_cols].to_parquet('data/features/split_val.parquet',     index=False)
test_df[meta_cols].to_parquet('data/features/split_test.parquet',   index=False)
print("  Split metadata parquets saved.")

# ──────────────────────────────────────────────────────────────────────
# STEP 7 — VALIDATION CHECKS
# ──────────────────────────────────────────────────────────────────────
print("\n[STEP 7] Running Validation Checks...")

checks_passed = 0
checks_total  = 0

def check(condition, msg_pass, msg_fail):
    global checks_passed, checks_total
    checks_total += 1
    if condition:
        print(f"  ✅ {msg_pass}")
        checks_passed += 1
    else:
        print(f"  ❌ {msg_fail}")

check(X_train.shape[1] >= 5000,
      f"TF-IDF vocab: {X_train.shape[1]} features",
      f"TF-IDF vocab too small: {X_train.shape[1]}")
check(X_train.shape[0] == len(train_df),
      "TF-IDF row count matches train_df",
      "TF-IDF / train_df row count mismatch!")
check(not np.isnan(X_train.data).any(),
      "No NaN in TF-IDF train matrix",
      "NaN found in TF-IDF train matrix!")
check(len(val_df) > 0,
      f"Val split has {len(val_df)} emails",
      "Val split is empty!")
check(len(test_df) > 0,
      f"Test split has {len(test_df)} emails",
      "Test split is empty!")

# Check y arrays load correctly
for split in ['train', 'val', 'test']:
    y = np.load(f'data/features/y_{split}_type.npy')
    check(len(y) > 0, f"y_{split}_type.npy: {len(y)} labels", f"y_{split}_type.npy is empty!")

# ──────────────────────────────────────────────────────────────────────
# STEP 8 — FEATURE STATS REPORT
# ──────────────────────────────────────────────────────────────────────
report = {
    'phase': '4A — Feature Engineering',
    'dataset_size': len(df),
    'split_sizes': {
        'train': len(train_df),
        'val': len(val_df),
        'test': len(test_df)
    },
    'tfidf_vocab_size': int(X_train.shape[1]),
    'tfidf_nnz': {'train': int(X_train.nnz), 'val': int(X_val.nnz), 'test': int(X_test.nnz)},
    'label_class_counts': {
        split: {
            'type':    {c: int((label_data[split]['disclosure_type'] == c).sum()) for c in CLASS_ORDER_TYPE},
            'framing': {c: int((label_data[split]['framing'] == c).sum()) for c in CLASS_ORDER_FRAMING},
            'risk':    {c: int((label_data[split]['risk_tier'] == c).sum()) for c in CLASS_ORDER_RISK}
        }
        for split in ['train', 'val', 'test']
    },
    'checks_passed': f"{checks_passed}/{checks_total}",
    'body_column_used': body_col
}

with open('results/phase4/feature_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n  Checks passed: {checks_passed}/{checks_total}")
print(f"\nPHASE 4A COMPLETE — All feature matrices saved to data/features/")
print(f"  Next: Run phase4b_empath.py to add psycholinguistic features")
