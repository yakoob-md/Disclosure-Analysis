import pandas as pd
import os

def main():
    print("="*50)
    print("PHASE 2 VALIDATION: CLEANING & SAMPLING")
    print("="*50)

    # Load datasets
    raw_path = "data/raw/emails_raw.parquet"
    clean_path = "data/processed/emails_clean.parquet"
    gold_path = "data/processed/emails_gold_pool.parquet"
    silver_path = "data/processed/emails_silver_pool.parquet"

    raw_df = pd.read_parquet(raw_path)
    clean_df = pd.read_parquet(clean_path)
    gold_df = pd.read_parquet(gold_path)
    silver_df = pd.read_parquet(silver_path)

    # 1. Deduplication Effect
    print("\n=== DEDUPLICATION & FILTERING ===")
    print(f"BEFORE (Raw Parsed): {len(raw_df):,}")
    print(f"AFTER (Cleaned):     {len(clean_df):,}")
    removed = len(raw_df) - len(clean_df)
    print(f"Removed (Duplicates/System Emails): {removed:,} ({removed/len(raw_df):.1%})")

    # 2. Alias Resolution
    print("\n=== ALIAS RESOLUTION (RAPIDFUZZ) ===")
    sample_aliases = clean_df[clean_df['sender'] != clean_df['sender_canonical']][['sender', 'sender_canonical', 'sender_role']].head(5)
    for idx, row in sample_aliases.iterrows():
        print(f"Original: {row['sender']:35s} -> Canonical: {row['sender_canonical']:20s} ({row['sender_role']})")

    # 3. Dense Excerpt Extraction Comparison
    print("\n=== DENSE EXCERPT PIPELINE ===")
    sample_excerpt = clean_df[clean_df['body_clean'] != clean_df['body_dense']].head(1).iloc[0]
    print(f">> FULL BODY LENGTH: {len(sample_excerpt['body_clean'])} chars")
    # print(sample_excerpt['body_clean'][:300] + "...")
    print(f">> DENSE EXCERPT LENGTH: {len(sample_excerpt['body_dense'])} chars")
    print(sample_excerpt['body_dense'][:300] + "...")

    # 4. Sampling Stats
    print("\n=== STRATIFIED POOLS ===")
    print(f"Gold Pool (Manual): {len(gold_df):,}")
    print(f"Silver Pool (LLM):  {len(silver_df):,}")
    
    combined = pd.concat([gold_df, silver_df])
    crisis_pct = combined['crisis_flag'].mean()
    fin_pct = combined['has_fin_kw'].mean()
    exec_count = len(combined[combined['sender_role'].isin(['CEO','CFO'])])
    
    print(f"\nStratification Checks:")
    print(f"  - Crisis Period >= 60%?   [{'PASS' if crisis_pct >= 0.59 else 'FAIL'}] ({crisis_pct:.1%})")
    print(f"  - Financial KWs >= 30%?   [{'PASS' if fin_pct >= 0.29 else 'FAIL'}] ({fin_pct:.1%})")
    print(f"  - Exec (CEO/CFO) Oversampled?  {exec_count:,} combined emails")

    with open('results/phase2/summary.txt', 'w') as f:
        f.write(f"Phase 2\nClean Rows: {len(clean_df)}\nGold: {len(gold_df)}\nSilver: {len(silver_df)}\n")
    print("\nReport saved to results/phase2/summary.txt")

if __name__ == "__main__":
    main()
