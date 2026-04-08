import pandas as pd
import os

def main():
    print("="*50)
    print("PHASE 1 VALIDATION: RAW vs PARSED DATA")
    print("="*50)

    # 1. Load Data
    csv_path = "data/raw/emails.csv"
    parquet_path = "data/raw/emails_raw.parquet"

    print("\nLoading datasets (this may take a moment)...")
    if os.path.exists(csv_path):
        raw_df = pd.read_csv(csv_path, nrows=50000) # Load subset for fast validation
        raw_rows = raw_df.shape[0]
        # Total rows is known from OS or assumed ~517k
        import subprocess
        try:
            # Quick row count approximation for large files
            total_raw = sum(1 for line in open(csv_path, 'rb')) - 1
        except:
            total_raw = "Unknown"
    else:
        print(f"Warning: {csv_path} not found.")
        total_raw = "N/A"

    if os.path.exists(parquet_path):
        parsed_df = pd.read_parquet(parquet_path)
        total_parsed = len(parsed_df)
    else:
        print(f"Error: {parquet_path} not found.")
        return

    # 2. Before vs After Stats
    print("\n=== ROW COUNTS ===")
    print(f"BEFORE (CSV String Rows): {total_raw:,}")
    print(f"AFTER  (Structured Parquet): {total_parsed:,}")

    print("\n=== DATA STRUCTURE ===")
    print("BEFORE (Kaggle CSV):")
    print("  Columns: ['file', 'message']")
    print("AFTER (Parsed Format):")
    print(f"  Columns: {list(parsed_df.columns)}")

    print("\n=== MISSING VALUES (AFTER) ===")
    print(parsed_df.isnull().sum())

    print("\n=== UNIQUE COUNTS ===")
    print(f"Unique Senders Identified: {parsed_df['sender'].nunique():,}")

    # 3. Before vs After Example
    print("\n=== SAMPLE TRANSFORMATION ===")
    print(">> BEFORE (Raw 'message' string):")
    if os.path.exists(csv_path):
        sample_raw = raw_df.iloc[0]['message']
        print(str(sample_raw)[:300] + "...")
    
    print("\n>> AFTER (Extracted Fields):")
    sample_parsed = parsed_df.iloc[0]
    print(f"Sender:  {sample_parsed['sender']}")
    print(f"Date:    {sample_parsed['date']}")
    print(f"Subject: {sample_parsed['subject']}")
    print(f"Body:    {sample_parsed['body'][:100]}...")

    # Write log to results
    with open('results/phase1/summary.txt', 'w') as f:
        f.write(f"Phase 1 Validation\nParsed Rows: {total_parsed}\nUnique Senders: {parsed_df['sender'].nunique()}\n")
    print("\nReport saved to results/phase1/summary.txt")

if __name__ == "__main__":
    main()
