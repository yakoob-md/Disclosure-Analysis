> **GEMINI 1.5 FLASH CRITICAL INSTRUCTION**: The user has mandated stepping validation. For every phase you execute from this document, you MUST ALSO execute the corresponding `scripts/phaseX_results.py` validation script immediately after and verify the outputs are correct before proceeding to the next phase.

# ORGDISCLOSE — Implementation Plan
# Part 5 of 6: Phase 9 (LLM Baseline) + Phase 10 (Full Evaluation & Ablation)

---

## PHASE 9: LLM BASELINE — PHI-3-MINI FEW-SHOT

### Duration: 1–2 days
### Goal: Phi-3-Mini as a reference "expert" model; evaluated on 200 Gold Test emails

### Why only 200 (not 500)?
Phi-3-Mini inference is ~4 seconds/email on RTX 2050.
200 emails = ~13 minutes. 500 emails = ~33 minutes.
200 is statistically sufficient (confidence interval ±7% at 95% CI).

---

### 9.1 Validation Standard (Must Pass)

- [ ] LLM evaluated on same 200 Gold test emails as all other models
- [ ] Three prompt variants run: zero-shot, few-shot (3 examples), chain-of-thought
- [ ] All three variants evaluated with same macro-F1 metric
- [ ] Prompt consistency test: 50 selected emails re-run 3× each, consistency ≥ 90%
- [ ] Results saved to results/llm_results.json

---

### 9.2 IMPLEMENTATION PROMPT — PHASE 9: LLM BASELINE

```
=== IMPLEMENTATION PROMPT: PHASE 9 — PHI-3-MINI LLM BASELINE ===
# AMENDMENT: Use Entity Overlap instead of BERTScore for LLM evaluation faithfulness where applicable.

CONTEXT:
- Hardware: RTX 2050 (4GB VRAM) — 4-bit quantized Phi-3-mini-4k-instruct
- Evaluate on 200 Gold test emails (same as other models)
- 3 prompt variants: zero-shot, few-shot, chain-of-thought
- Output: results/llm_results.json + results/llm_predictions.parquet

Write phase9_llm_baseline.py:

STEP 1 — Load model (same as Phase 3A):
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch, pandas as pd, numpy as np, json, re
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report

model_name = "microsoft/Phi-3-mini-4k-instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="auto"
)
model.eval()

STEP 2 — Load 200 test emails (fixed random sample for comparability):
gold = pd.read_parquet('data/labeled/emails_labeled_gold.parquet')
test_meta = pd.read_parquet('data/features/split_test.parquet')
test_full = gold[gold['mid'].isin(test_meta['mid'])].reset_index(drop=True)

# Use seeded random sample — NOT .head() which is time-ordered and unrepresentative
test_df = test_full.sample(n=min(200, len(test_full)), random_state=42).reset_index(drop=True)
print(f"Evaluating LLM on {len(test_df)} test emails")

# Save the exact mids used for reproducibility verification
import json
with open('results/llm_test_mids.json', 'w') as f:
    json.dump(test_df['mid'].tolist(), f)
print("Saved LLM test email IDs to results/llm_test_mids.json")

STEP 3 — Define 3 prompt templates:
# 1. ZERO-SHOT PROMPT
ZERO_SHOT = """[INST] You are an expert in organizational email risk analysis.
Classify this email. Reply ONLY in valid JSON.

disclosure_type: one of [FINANCIAL, PII, STRATEGIC, LEGAL, RELATIONAL, NONE]
framing: one of [PROTECTED, UNPROTECTED, NA]
risk_tier: one of [NONE, LOW, HIGH]

Email Subject: {subject}
Email Body: {body}

JSON format: {{"disclosure_type": "...", "framing": "...", "risk_tier": "..."}} [/INST]"""

# 2. FEW-SHOT PROMPT (3 examples hardcoded from training set)
FEW_SHOT_EXAMPLES = """Example 1:
Email: "The write-down of approximately $1.2 billion in equity will be charged against earnings..."
Labels: {{"disclosure_type": "FINANCIAL", "framing": "UNPROTECTED", "risk_tier": "HIGH"}}

Example 2:
Email: "Please keep this strictly between us — the merger with Dynegy is proceeding confidentially..."
Labels: {{"disclosure_type": "STRATEGIC", "framing": "PROTECTED", "risk_tier": "LOW"}}

Example 3:
Email: "Hi John, are we still on for lunch tomorrow?"
Labels: {{"disclosure_type": "NONE", "framing": "NA", "risk_tier": "NONE"}}"""

FEW_SHOT = """[INST] You are an expert in organizational email risk analysis.
Classify the following email using the examples below.

{examples}

Now classify this email. Reply ONLY in valid JSON.
Email Subject: {subject}
Email Body: {body}

JSON format: {{"disclosure_type": "...", "framing": "...", "risk_tier": "..."}} [/INST]"""

# 3. CHAIN-OF-THOUGHT PROMPT
COT = """[INST] You are an expert in organizational email risk analysis.
Analyze the email step by step, then provide your final labels.

Email Subject: {subject}
Email Body: {body}

Step 1: What type of organizational information is disclosed? (FINANCIAL/PII/STRATEGIC/LEGAL/RELATIONAL/NONE)
Step 2: Are there protective markers? (PROTECTED/UNPROTECTED/NA)
Step 3: What is the overall risk? (NONE/LOW/HIGH)

Provide your reasoning, then end with EXACTLY this JSON on the last line:
{{"disclosure_type": "...", "framing": "...", "risk_tier": "..."}} [/INST]"""

STEP 4 — Inference function:
def run_inference(prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                       max_length=768).to('cuda')
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy decoding — deterministic
            # temperature removed: ignored when do_sample=False
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0][inputs['input_ids'].shape[1]:],
                            skip_special_tokens=True)

def parse_json_label(text):
    # Extract last JSON-like structure from response
    matches = list(re.finditer(r'\{[^{}]*"disclosure_type"[^{}]*\}', text, re.DOTALL))
    if not matches: return 'NONE', 'NA', 'NONE'
    try:
        p = json.loads(matches[-1].group())
        valid_dt = ['FINANCIAL','PII','STRATEGIC','LEGAL','RELATIONAL','NONE']
        valid_fr = ['PROTECTED','UNPROTECTED','NA']
        valid_rk = ['NONE','LOW','HIGH']
        dt = p.get('disclosure_type','NONE')
        fr = p.get('framing','NA')
        rk = p.get('risk_tier','NONE')
        return (dt if dt in valid_dt else 'NONE',
                fr if fr in valid_fr else 'NA',
                rk if rk in valid_rk else 'NONE')
    except: return 'NONE', 'NA', 'NONE'

STEP 5 — Run all 3 variants:
all_results = {}

for variant_name, prompt_template in [
    ('zero_shot', ZERO_SHOT),
    ('few_shot', FEW_SHOT),
    ('chain_of_thought', COT)
]:
    preds_type, preds_frame, preds_risk = [], [], []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=variant_name):
        body_excerpt = str(row.get('body_clean',''))[:500]
        subject = str(row.get('subject',''))[:100]
        
        if variant_name == 'few_shot':
            prompt = prompt_template.format(
                examples=FEW_SHOT_EXAMPLES, subject=subject, body=body_excerpt
            )
        elif variant_name == 'chain_of_thought':
            prompt = prompt_template.format(subject=subject, body=body_excerpt)
        else:
            prompt = prompt_template.format(subject=subject, body=body_excerpt)
        
        response = run_inference(prompt, max_new_tokens=150 if variant_name!='chain_of_thought' else 300)
        dt, fr, rk = parse_json_label(response)
        preds_type.append(dt)
        preds_frame.append(fr)
        preds_risk.append(rk)
    
    # Compute F1
    LABEL_MAPS = {
        'type': {'FINANCIAL':0,'PII':1,'STRATEGIC':2,'LEGAL':3,'RELATIONAL':4,'NONE':5},
        'framing': {'PROTECTED':0,'UNPROTECTED':1,'NA':2},
        'risk': {'NONE':0,'LOW':1,'HIGH':2}
    }
    
    true_type    = [LABEL_MAPS['type'].get(y,5) for y in test_df['disclosure_type']]
    true_framing = [LABEL_MAPS['framing'].get(y,2) for y in test_df['framing']]
    true_risk    = [LABEL_MAPS['risk'].get(y,0) for y in test_df['risk_tier']]
    
    pred_type_enc    = [LABEL_MAPS['type'].get(p,5) for p in preds_type]
    pred_framing_enc = [LABEL_MAPS['framing'].get(p,2) for p in preds_frame]
    pred_risk_enc    = [LABEL_MAPS['risk'].get(p,0) for p in preds_risk]
    
    f1_type    = f1_score(true_type,    pred_type_enc,    average='macro', zero_division=0)
    f1_framing = f1_score(true_framing, pred_framing_enc, average='macro', zero_division=0)
    f1_risk    = f1_score(true_risk,    pred_risk_enc,    average='macro', zero_division=0)
    
    all_results[variant_name] = {
        'disc_type_f1': f1_type,
        'framing_f1':   f1_framing,
        'risk_f1':      f1_risk,
        'avg_f1':       (f1_type + f1_framing + f1_risk) / 3
    }
    print(f"\n{variant_name}: type={f1_type:.4f}, frame={f1_framing:.4f}, risk={f1_risk:.4f}")

STEP 6 — Prompt consistency test (forensic reliability):
# Run zero-shot prompt 3 times on same 50 emails
consistency_emails = test_df.head(50)
consistency_runs = [[], [], []]
for run_idx in range(3):
    for _, row in tqdm(consistency_emails.iterrows(), desc=f'Consistency run {run_idx+1}'):
        body_excerpt = str(row.get('body_clean',''))[:500]
        subject = str(row.get('subject',''))
        prompt = ZERO_SHOT.format(subject=subject, body=body_excerpt)
        response = run_inference(prompt, max_new_tokens=80)
        dt, _, _ = parse_json_label(response)
        consistency_runs[run_idx].append(dt)

# Compute agreement: fraction of emails where all 3 runs agree
agreements = [
    consistency_runs[0][i] == consistency_runs[1][i] == consistency_runs[2][i]
    for i in range(50)
]
consistency_score = np.mean(agreements)
print(f"\nPrompt consistency score (3 runs, 50 emails): {consistency_score:.3f}")
if consistency_score >= 0.90:
    print("PASS: LLM is forensically reliable (≥90% consistent)")
else:
    print(f"NOTE: Consistency {consistency_score:.1%} — report in paper as limitation")

all_results['consistency_score'] = consistency_score

# Save results
with open('results/llm_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print("PHASE 9 COMPLETE")
=== END OF PROMPT ===
```

---

## PHASE 10: FULL EVALUATION, ABLATION & STATISTICAL TESTING

### Duration: 2 days
### Goal: Complete comparison table + ablation table + Wilcoxon tests + confusion matrices + error analysis

---

### 10.1 Validation Standard (Paper Acceptance Gate)

- [ ] All 5 models evaluated on same Gold test set
- [ ] Ablation table shows KG contribution (Δ ≥ 2% on DeBERTa)
- [ ] Wilcoxon p < 0.05 for top model vs. second-best
- [ ] Confusion matrices saved for all models for disc_type
- [ ] Top-50 false negatives manually reviewed and categorized
- [ ] All results in one unified results/comparison_table.csv

---

### 10.2 IMPLEMENTATION PROMPT — PHASE 10: UNIFIED EVALUATION

```
=== IMPLEMENTATION PROMPT: PHASE 10 — UNIFIED EVALUATION & COMPARISON ===

CONTEXT:
- All model result files exist in results/ directory
- Gold test set: data/labeled/emails_labeled_gold.parquet (filtered to test split)
- Output: results/comparison_table.csv + results/ablation_table.csv +
          results/confusion_matrices.pdf + results/error_analysis.csv

Write phase10_evaluation.py:

STEP 1 — Load all results and compute master comparison table:
import pandas as pd, numpy as np, json
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import wilcoxon
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                              recall_score, classification_report)

# Load all result files
with open('results/ml_results.json') as f:    ml_res = json.load(f)
with open('results/bilstm_results.json') as f: bilstm_res = json.load(f)
with open('results/deberta_results.json') as f: deberta_res = json.load(f)
with open('results/llm_results.json') as f:   llm_res = json.load(f)

# Master table rows — (model_name, disc_type_f1, framing_f1, risk_f1, avg_f1, has_kg)
rows = []

# Compute random baseline from DummyClassifier (not hardcoded — Enron is imbalanced)
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score as sk_f1
y_test_type  = np.load('data/features/y_test_type.npy')
y_train_type = np.load('data/features/y_train_type.npy')
y_test_frame = np.load('data/features/y_test_framing.npy')
y_test_risk  = np.load('data/features/y_test_risk.npy')
dummy = DummyClassifier(strategy='stratified', random_state=42)
dummy.fit(np.zeros((len(y_train_type),1)), y_train_type)
random_f1_type  = sk_f1(y_test_type,  dummy.predict(np.zeros((len(y_test_type),1))),  average='macro')
random_f1_frame = sk_f1(y_test_frame, dummy.predict(np.zeros((len(y_test_frame),1))), average='macro')
random_f1_risk  = sk_f1(y_test_risk,  dummy.predict(np.zeros((len(y_test_risk),1))),  average='macro')
random_avg = (random_f1_type + random_f1_frame + random_f1_risk) / 3
print(f"Random baseline — type={random_f1_type:.4f}, frame={random_f1_frame:.4f}, risk={random_f1_risk:.4f}")
rows.append({'Model': 'Random Baseline', 'disc_type': random_f1_type,
             'framing': random_f1_frame, 'risk': random_f1_risk,
             'avg': random_avg, 'KG': 'No', 'Tier': 'Baseline'})

# ML models (extract from ml_res structure)
for model_key, tier in [('xgboost','ML'), ('random_forest','ML')]:
    for kg_key, has_kg in [('text_only','No'), ('kg_augmented','Yes')]:
        if f'{model_key}_{kg_key}' in ml_res:
            r = ml_res[f'{model_key}_{kg_key}']
            f1_type  = r.get('type', {}).get('macro_f1', 0.0)
            f1_frame = r.get('framing', {}).get('macro_f1', 0.0)
            f1_risk  = r.get('risk', {}).get('macro_f1', 0.0)
            avg = (f1_type + f1_frame + f1_risk) / 3
            rows.append({'Model': f"{model_key.replace('_',' ').title()} {'+ KG' if has_kg=='Yes' else ''}",
                         'disc_type': f1_type, 'framing': f1_frame,
                         'risk': f1_risk, 'avg': avg, 'KG': has_kg, 'Tier': tier})

# BiLSTM models
for kg_key, has_kg in [('text_only','No'), ('kg_augmented','Yes')]:
    if kg_key in bilstm_res:
        r = bilstm_res[kg_key]
        rows.append({'Model': f"BiLSTM+Attn {'+ KG' if has_kg=='Yes' else ''}",
                     'disc_type': r.get('type', 0.0), 'framing': r.get('frame', 0.0),
                     'risk': r.get('risk', 0.0),
                     'avg': np.mean([r.get('type',0), r.get('frame',0), r.get('risk',0)]),
                     'KG': has_kg, 'Tier': 'DL'})

# DeBERTa models
for kg_key, has_kg in [('text_only','No'), ('kg_augmented','Yes')]:
    if kg_key in deberta_res:
        r = deberta_res[kg_key]
        # Extract F1 scores from your saved format
        f1_type  = r.get('disc_type_f1', r.get('best_val_f1', 0.0))
        f1_frame = r.get('framing_f1', 0.0)
        f1_risk  = r.get('risk_f1', 0.0)
        avg = np.mean([f1_type, f1_frame, f1_risk])
        rows.append({'Model': f"DeBERTa-v3-small {'+ KG' if has_kg=='Yes' else ''}",
                     'disc_type': f1_type, 'framing': f1_frame,
                     'risk': f1_risk, 'avg': avg, 'KG': has_kg, 'Tier': 'DL'})

# LLM models
for variant in ['zero_shot', 'few_shot', 'chain_of_thought']:
    if variant in llm_res:
        r = llm_res[variant]
        rows.append({'Model': f"Phi-3-Mini ({variant.replace('_',' ')})",
                     'disc_type': r.get('disc_type_f1', 0.0),
                     'framing':   r.get('framing_f1', 0.0),
                     'risk':      r.get('risk_f1', 0.0),
                     'avg':       r.get('avg_f1', 0.0),
                     'KG': 'No', 'Tier': 'LLM'})

comparison_df = pd.DataFrame(rows)
comparison_df = comparison_df.sort_values('avg', ascending=False).reset_index(drop=True)
comparison_df.to_csv('results/comparison_table.csv', index=False)
print("\n=== MASTER COMPARISON TABLE ===")
print(comparison_df.to_string(index=False))

STEP 2 — Ablation table:
print("\n=== ABLATION ANALYSIS ===")
ablation_rows = []
for model_base in ['XGBoost', 'Random Forest', 'BiLSTM+Attn', 'DeBERTa-v3-small']:
    text_row = comparison_df[comparison_df['Model'].str.startswith(model_base) &
                              (comparison_df['KG']=='No')]
    kg_row   = comparison_df[comparison_df['Model'].str.startswith(model_base) &
                              (comparison_df['KG']=='Yes')]
    if len(text_row) > 0 and len(kg_row) > 0:
        delta_type = kg_row.iloc[0]['disc_type'] - text_row.iloc[0]['disc_type']
        delta_avg  = kg_row.iloc[0]['avg']        - text_row.iloc[0]['avg']
        verdict = 'PROVES KG' if delta_type >= 0.02 else 'MARGINAL' if delta_type > 0 else 'KG HURTS'
        ablation_rows.append({'Model': model_base,
                               'Text_F1': text_row.iloc[0]['disc_type'],
                               'KG_F1': kg_row.iloc[0]['disc_type'],
                               'Delta_disctype': delta_type,
                               'Delta_avg': delta_avg,
                               'Verdict': verdict})
        print(f"{model_base}: text={text_row.iloc[0]['disc_type']:.4f}, "
              f"KG={kg_row.iloc[0]['disc_type']:.4f}, Δ={delta_type:+.4f} [{verdict}]")

ablation_df = pd.DataFrame(ablation_rows)
ablation_df.to_csv('results/ablation_table.csv', index=False)

STEP 3 — Wilcoxon signed-rank test (actual implementation):
# Load per-sample correct/wrong arrays saved during Phase 8B test evaluation
# These files must be saved from Phase 8B:
#   np.save('results/deberta_kg_per_sample_correct.npy', np.array(correct_kg))
#   np.save('results/deberta_text_per_sample_correct.npy', np.array(correct_text))
# Add this save to Phase 8B after test evaluation:
#   correct_kg = [1 if p==t else 0 for p,t in zip(all_preds['type'], all_true['type'])]
#   np.save('results/deberta_kg_per_sample_correct.npy', np.array(correct_kg))

import os
wilcoxon_done = False
if os.path.exists('results/deberta_kg_per_sample_correct.npy') and \
   os.path.exists('results/deberta_text_per_sample_correct.npy'):
    c_kg   = np.load('results/deberta_kg_per_sample_correct.npy')
    c_text = np.load('results/deberta_text_per_sample_correct.npy')
    # Wilcoxon requires the arrays to differ — if identical, test will error
    if not np.array_equal(c_kg, c_text):
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(c_kg, c_text, alternative='greater')
        print(f"Wilcoxon DeBERTa+KG > DeBERTa-text-only:")
        print(f"  stat={stat:.2f}, p={p:.4f} {'[SIGNIFICANT]' if p<0.05 else '[NOT SIGNIFICANT]'}")
        wilcoxon_done = True
if not wilcoxon_done:
    print("NOTE: Per-sample prediction files not found or identical.")
    print("  Ensure Phase 8B saves: results/deberta_{kg,text}_per_sample_correct.npy")
    print("  Add after test evaluation loop in Phase 8B:")
    print("    correct = [1 if p==t else 0 for p,t in zip(all_preds['type'], all_true['type'])]")
    print("    np.save('results/deberta_kg_per_sample_correct.npy', np.array(correct))")

STEP 4 — Confusion matrices for disc_type (best model):
# Run best model predictions and plot confusion matrices
# This assumes you've saved test predictions from Phase 8B

DISC_TYPE_CLASSES = ['FINANCIAL','PII','STRATEGIC','LEGAL','RELATIONAL','NONE']

# Example: plot confusion matrix from saved predictions
# true_labels and pred_labels should be loaded from saved numpy arrays
# Here we show the plotting code structure:

def plot_confusion_matrix(true_labels, pred_labels, class_names, title, save_path):
    cm = confusion_matrix(true_labels, pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix: {save_path}")

# Call this for each model's predictions:
# plot_confusion_matrix(true_type, pred_type, DISC_TYPE_CLASSES,
#                       'DeBERTa+KG — Disclosure Type', 'results/cm_deberta_kg.pdf')

STEP 5 — Error analysis (top-50 false negatives):
# HIGH-RISK emails that the best model predicted as LOW or NONE
print("\nGenerating error analysis template...")

error_analysis_template = """
For each of the top-50 false negatives, record:
- email_id (mid)
- true_disclosure_type
- predicted_disclosure_type
- true_risk_tier
- predicted_risk_tier
- email_subject
- first_50_words_of_body
- failure_mode: one of [AMBIGUOUS_LANGUAGE, TECHNICAL_JARGON, 
                        IRONIC_OR_CODED, SHORT_CONTEXT, 
                        DOMAIN_MISMATCH, ANNOTATION_NOISE]
- analyst_note: (your manual description of why it failed)

Save as: results/error_analysis.csv
Target: 50 rows minimum
Expected failure mode distribution:
  TECHNICAL_JARGON:   ~35% (Enron-specific financial terms)
  AMBIGUOUS_LANGUAGE: ~25% (borderline disclosures)
  SHORT_CONTEXT:      ~20% (email too short for model)
  IRONIC_OR_CODED:    ~10% (coded language like "Project Braveheart")
  ANNOTATION_NOISE:   ~10% (annotator disagreement)
"""
with open('results/error_analysis_guide.txt', 'w') as f:
    f.write(error_analysis_template)
print("Error analysis template saved to results/error_analysis_guide.txt")

STEP 6 — Print paper-ready summary:
print("\n" + "="*70)
print("PAPER-READY RESULTS SUMMARY")
print("="*70)
print(f"Best model: {comparison_df.iloc[0]['Model']}")
print(f"Best avg macro-F1: {comparison_df.iloc[0]['avg']:.4f}")
print(f"\nKey ablation finding:")
deberta_row = ablation_df[ablation_df['Model']=='DeBERTa-v3-small']
if len(deberta_row) > 0:
    delta = deberta_row.iloc[0]['Delta_disctype']
    print(f"  DeBERTa KG vs text-only: Δ = {delta:+.4f}")
    if delta >= 0.02:
        print("  ✓ KG CONTRIBUTION PROVEN (≥ 2% gain)")
    else:
        print("  ✗ KG contribution marginal — revisit feature engineering")

print(f"\nLLM consistency: {llm_res.get('consistency_score', 'N/A')}")
print("="*70)
print("PHASE 10 COMPLETE — All evaluation artifacts saved")
=== END OF PROMPT ===
```

---

### 10.3 Temporal Analysis Validation

```python
# Run this SEPARATELY after Phase 10 — saves Figure 1 for your paper

# IMPLEMENTATION PROMPT FRAGMENT — TEMPORAL ANALYSIS TABLE:

# Load centrality matrix
cm_df = pd.read_parquet('graphs/centrality_matrix.parquet')
# Load labeled gold emails with risk_tier
gold = pd.read_parquet('data/labeled/emails_labeled_gold.parquet')

# Map: month period → crisis label
def period_label(m):
    if m <= 11: return '1_stable_2000'
    if m <= 17: return '2_pre_crisis_2001'
    if m <= 23: return '3_acute_crisis'
    return '4_post_crisis'

gold['period'] = gold['month_index'].apply(period_label)
cm_df['period'] = cm_df['month_index'].apply(period_label)

# Table: period × avg_delta_betweenness × % HIGH risk emails
for period in sorted(gold['period'].unique()):
    period_gold = gold[gold['period'] == period]
    period_cm   = cm_df[cm_df['period'] == period]
    
    high_pct = (period_gold['risk_tier'] == 'HIGH').mean()
    avg_delta = period_cm['delta_betweenness'].abs().mean()
    n_emails = len(period_gold)
    print(f"{period}: n={n_emails}, HIGH%={high_pct:.1%}, avg|Δ_betweenness|={avg_delta:.4f}")

# Wilcoxon: Is Δ_betweenness significantly larger in crisis vs stable?
stable_delta = cm_df[cm_df['period']=='1_stable_2000']['delta_betweenness'].values
crisis_delta = cm_df[cm_df['period']=='3_acute_crisis']['delta_betweenness'].values
n = min(len(stable_delta), len(crisis_delta))
from scipy.stats import wilcoxon
stat, p = wilcoxon(crisis_delta[:n], stable_delta[:n], alternative='greater')
print(f"\nWilcoxon test crisis > stable Δ_betweenness: stat={stat:.2f}, p={p:.4f}")
```
