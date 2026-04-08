import pandas as pd
import json
import os

def main():
    print("="*50)
    print("PHASE 6 VALIDATION: NETWORK CENTRALITY")
    print("="*50)

    cm_file = "graphs/centrality_matrix.parquet"
    if not os.path.exists(cm_file):
        print(f"Data not generated yet. Waiting for {cm_file}...")
        return

    cm_df = pd.read_parquet(cm_file)
    
    print("\n=== CENTRALITY FEATURES ===")
    print(f"Shape: {cm_df.shape}")
    print("Columns:", list(cm_df.columns))

    print("\n=== TOP CENTRAL NODES (BETWEENNESS) ===")
    avg_bet = cm_df.groupby('employee')['betweenness'].mean().sort_values(ascending=False)
    for emp, score in avg_bet.head(5).items():
        print(f"  {emp:25s}: {score:.4f}")

    print("\n=== WILCOXON STATISTICAL TEST ===")
    wil_file = "results/wilcoxon_centrality.json"
    if os.path.exists(wil_file):
        with open(wil_file, 'r') as f:
            wil = json.load(f)
        print(f"Statistic: {wil.get('stat')}")
        print(f"P-Value:   {wil.get('p_value')}")
        print(f"Significant? {'YES' if wil.get('significant') else 'NO'}")
    else:
        print("Wilcoxon results not found.")

    with open('results/phase6/summary.txt', 'w') as f:
        f.write(f"Phase 6\nShape: {cm_df.shape}\nTop Node: {avg_bet.index[0]}\n")
    print("\nReport saved to results/phase6/summary.txt")

if __name__ == "__main__":
    main()
