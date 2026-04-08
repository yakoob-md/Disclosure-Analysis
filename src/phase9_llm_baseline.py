"""
PHASE 9: LLM BASELINE — PHI-3-MINI FEW-SHOT
===========================================
Phi-3-Mini evaluated on 200 Test emails using 3 prompt variants:
(zero-shot, few-shot, chain-of-thought) to compare against our ML/DL models.

Runs exactly like Phase 3's annotation pipeline but uses the test fold exclusively.
"""
import os
import json
import re
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import f1_score

os.makedirs('results/phase9', exist_ok=True)
print("=" * 65)
print("PHASE 9 — PHI-3-MINI LLM BASELINE")
print("=" * 65)

print("\n[STEP 1] Loading model (4-bit quantization)...")
model_name = "microsoft/Phi-3-mini-4k-instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
)

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto"
    )
    model.eval()
except Exception as e:
    print(f"Failed to load model locally: {e}. Check VRAM/Kaggle.")
    exit(1)

print("\n[STEP 2] Loading test subset...")
all_labeled  = pd.read_parquet('data/labeled/emails_labeled_silver.parquet')
test_meta    = pd.read_parquet('data/features/split_test.parquet')
test_full    = all_labeled[all_labeled['mid'].isin(test_meta['mid'])].reset_index(drop=True)

test_df = test_full.sample(n=min(200, len(test_full)), random_state=42).reset_index(drop=True)
print(f"Evaluating LLM on {len(test_df)} test emails")

with open('results/phase9/llm_test_mids.json', 'w') as f:
    json.dump(test_df['mid'].tolist(), f)

print("\n[STEP 3] Defining prompt variants...")
ZERO_SHOT = """[INST] You are an expert in organizational email risk analysis.
Classify this email. Reply ONLY in valid JSON.

disclosure_type: one of [FINANCIAL, PII, STRATEGIC, LEGAL, RELATIONAL, NONE]
framing: one of [PROTECTED, UNPROTECTED, NA]
risk_tier: one of [NONE, LOW, HIGH]

Email Subject: {subject}
Email Body: {body}

JSON format: {{"disclosure_type": "...", "framing": "...", "risk_tier": "..."}} [/INST]"""

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

def run_inference(prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024).to('cuda')
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

def parse_json_label(text):
    matches = list(re.finditer(r'\{[^{}]*"disclosure_type"[^{}]*\}', text, re.DOTALL))
    if not matches: return 'NONE', 'NA', 'NONE'
    try:
        p = json.loads(matches[-1].group().replace("'", '"'))
        dt_val = p.get('disclosure_type','NONE')
        fr_val = p.get('framing','NA')
        rk_val = p.get('risk_tier','NONE')
        return dt_val, fr_val, rk_val
    except: return 'NONE', 'NA', 'NONE'

print("\n[STEP 4] Running LLM inference for Variants...")
all_results = {}
for variant_name, prompt_template in [('zero_shot', ZERO_SHOT), ('few_shot', FEW_SHOT)]:
    preds_type, preds_frame, preds_risk = [], [], []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=variant_name):
        b = str(row.get('body_clean', row.get('body', '')))[:500]
        s = str(row.get('subject', ''))[:100]
        
        if variant_name == 'few_shot': prompt = prompt_template.format(examples=FEW_SHOT_EXAMPLES, subject=s, body=b)
        else: prompt = prompt_template.format(subject=s, body=b)
        
        response = run_inference(prompt, max_new_tokens=150)
        dt, fr, rk = parse_json_label(response)
        preds_type.append(dt)
        preds_frame.append(fr)
        preds_risk.append(rk)
        
    LABEL_MAPS = {
        'type': {'FINANCIAL':0,'PII':1,'STRATEGIC':2,'LEGAL':3,'RELATIONAL':4,'NONE':5},
        'framing': {'PROTECTED':0,'UNPROTECTED':1,'NA':2},
        'risk': {'NONE':0,'LOW':1,'HIGH':2}
    }
    
    true_type    = [LABEL_MAPS['type'].get(y,5) for y in test_df['disclosure_type']]
    true_framing = [LABEL_MAPS['framing'].get(y,2) for y in test_df['framing']]
    true_risk    = [LABEL_MAPS['risk'].get(y,0) for y in test_df['risk_tier']]
    
    p_type    = [LABEL_MAPS['type'].get(p,5) for p in preds_type]
    p_framing = [LABEL_MAPS['framing'].get(p,2) for p in preds_frame]
    p_risk    = [LABEL_MAPS['risk'].get(p,0) for p in preds_risk]
    
    f1_t = f1_score(true_type, p_type, average='macro', zero_division=0)
    f1_f = f1_score(true_framing, p_framing, average='macro', zero_division=0)
    f1_r = f1_score(true_risk, p_risk, average='macro', zero_division=0)
    
    all_results[variant_name] = {'disc_type_f1': f1_t, 'framing_f1': f1_f, 'risk_f1': f1_r, 'avg_f1': (f1_t + f1_f + f1_r)/3}
    print(f"{variant_name} Macro-F1 => Type: {f1_t:.3f} | Frame: {f1_f:.3f} | Risk: {f1_r:.3f}")

with open('results/phase9/llm_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print("PHASE 9 COMPLETE")
