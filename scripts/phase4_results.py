import pandas as pd
import json
import os

def main():
    print("="*50)
    print("PHASE 4 VALIDATION: ML / DL MODEL COMPARISONS")
    print("="*50)

    comp_file = "results/comparison_table.csv"
    if not os.path.exists(comp_file):
        print(f"Data not generated yet. Waiting for {comp_file}...")
        return

    comp_df = pd.read_csv(comp_file)
    
    print("\n=== MODEL PERFORMANCE (MACRO F1) ===")
    print(comp_df.to_string(index=False))

    print("\n=== KG ABLATION DELTAS ===")
    ablation_file = "results/ablation_table.json"
    if os.path.exists(ablation_file):
        with open(ablation_file, 'r') as f:
            ablation = json.load(f)
        for model, res in ablation.items():
            print(f"> {model}: Text Only ({res['text_only_f1']:.4f}) -> +KG ({res['kg_augmented_f1']:.4f}) | DELTA: {res['delta']:+.4f}")
    else:
        print("Ablation table not found.")

    with open('results/phase4/summary.txt', 'w') as f:
        f.write(f"Phase 4\nModels tested: {len(comp_df)}\nTop Model: {comp_df.iloc[0]['Model']} (F1: {comp_df.iloc[0]['macro_F1']})\n")
    print("\nReport saved to results/phase4/summary.txt")

if __name__ == "__main__":
    main()
