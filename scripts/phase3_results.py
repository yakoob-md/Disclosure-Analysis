import pandas as pd
import os

def main():
    print("="*50)
    print("PHASE 3 VALIDATION: LLM AUTO-LABELING (SILVER)")
    print("="*50)

    silver_in = "data/processed/emails_silver_pool.parquet"
    silver_out = "data/labeled/emails_labeled_silver.parquet"

    if not os.path.exists(silver_out):
        print(f"Data not generated yet. Waiting for {silver_out}...")
        return

    df_in = pd.read_parquet(silver_in)
    df_out = pd.read_parquet(silver_out)

    print("\n=== LABEL YIELD ===")
    print(f"BEFORE (Unlabeled Silver Pool): {len(df_in):,}")
    print(f"AFTER (High Confidence Labels): {len(df_out):,}")
    print(f"Yield Rate: {len(df_out)/len(df_in):.1%}")

    print("\n=== LABEL DISTRIBUTION ( disclosure_type ) ===")
    dist = df_out['disclosure_type'].value_counts(normalize=True)
    for k, v in dist.items():
        print(f"  {k:15s}: {v:.1%}")

    print("\n=== LABEL DISTRIBUTION ( risk_tier ) ===")
    dist = df_out['risk_tier'].value_counts(normalize=True)
    for k, v in dist.items():
        print(f"  {k:15s}: {v:.1%}")

    print("\n=== SAMPLE LABELED EMAILS ===")
    for cls in ['FINANCIAL', 'STRATEGIC', 'LEGAL']:
        subset = df_out[df_out['disclosure_type'] == cls]
        if len(subset) > 0:
            sample = subset.iloc[0]
            print(f"\n>> TYPE: {cls} | TIER: {sample['risk_tier']} | CONFIDENCE: {sample['confidence']:.2f}")
            print(f"Subject: {sample['subject']}")
            print(f"Body:    {sample['body_dense'][:150]}...")

    with open('results/phase3/summary.txt', 'w') as f:
        f.write(f"Phase 3\nYield: {len(df_out)}\nDist:\n{df_out['disclosure_type'].value_counts().to_string()}\n")
    print("\nReport saved to results/phase3/summary.txt")

if __name__ == "__main__":
    main()
