"""
PHASE 4B — EMPATH PSYCHOLINGUISTIC FEATURES
=============================================
Calculates 194 categories (e.g., negative_emotion, money, power)
using the Empath lexicon for all train/val/test splits.

Fixed for Research-Grade pipeline: Since we employ a Weak Supervision
approach (no 'Gold' set), we extract from the unified silver dataset.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from empath import Empath
import os

os.makedirs('data/features', exist_ok=True)
print("=" * 55)
print("PHASE 4B — EMPATH PSYCHOLINGUISTIC FEATURE EXTRACTION")
print("=" * 55)

# STEP 1 — Load Empath and determine actual category count
lexicon = Empath()
_test_result = lexicon.analyze("test", normalize=True)
N_EMPATH = len(_test_result) if _test_result else 194
print(f"Empath categories detected: {N_EMPATH}")

def get_empath_features(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return np.zeros(N_EMPATH, dtype=np.float32)
    result = lexicon.analyze(text, normalize=True)
    if result is None:
        return np.zeros(N_EMPATH, dtype=np.float32)
    return np.array(list(result.values()), dtype=np.float32)

# Load the base labeled dataset
print("\nLoading silver dataset...")
emails_df = pd.read_parquet('data/labeled/emails_labeled_silver.parquet')

# Process each split separately using the metadata split parquets we generated in Phase 4A
print("\nExtracting features...")
for split_name in ['train', 'val', 'test']:
    meta_path = f'data/features/split_{split_name}.parquet'
    if not os.path.exists(meta_path):
        print(f"⚠️ Split metadata missing: {meta_path} — run Phase 4A first!")
        continue
        
    split_meta = pd.read_parquet(meta_path)
    
    # Merge correctly to preserve exact row ordering from Phase 4A's y_ labels
    # We must iterate over split_meta to keep order!
    # A simple way to do this: set index to mid on both, align by split_meta
    emails_df_indexed = emails_df.set_index('mid')
    split_meta_indexed = split_meta.set_index('mid')
    
    # Select only emails in this split, aligned to the meta split's index order
    split_emails = emails_df_indexed.loc[split_meta_indexed.index]
    
    # Use 'body_clean' if available, fallback 'body_dense', fallback 'body'
    body_col = 'body_clean' if 'body_clean' in split_emails.columns else 'body_dense' if 'body_dense' in split_emails.columns else 'body'
    
    features = []
    for body in tqdm(split_emails[body_col].fillna(''), desc=f'Empath {split_name}'):
        features.append(get_empath_features(body))
        
    features = np.array(features, dtype=np.float32)
    np.save(f'data/features/empath_{split_name}.npy', features)
    print(f"Empath {split_name}: shape {features.shape}, NaN count: {np.isnan(features).sum()}")

# STEP 3 — Validate
print("\nValidating...")
for split_name in ['train', 'val', 'test']:
    feat_path = f'data/features/empath_{split_name}.npy'
    if not os.path.exists(feat_path):
        continue
    feat = np.load(feat_path)
    assert feat.shape[1] == N_EMPATH, f"Empath count mismatch: {feat.shape[1]} vs {N_EMPATH}"
    assert not np.isnan(feat).any(), f"NaN in Empath {split_name}"
    print(f"✅ {split_name}: {feat.shape} — OK")

print("\nPHASE 4B COMPLETE — Empath features saved")
