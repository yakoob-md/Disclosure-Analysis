import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import re
import os
from tqdm import tqdm

def get_phi3_pipeline():
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    print(f"Loading {model_name} in 4-bit...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return tokenizer, model

SYSTEM_PROMPT = """You are an expert in organizational email risk analysis.
Label the following email along 2 dimensions. Reply ONLY with valid JSON. No explanation.

Dimension 1 - disclosure_type: one of [FINANCIAL, PII, STRATEGIC, LEGAL, RELATIONAL, NONE]
  FINANCIAL = earnings, reserves, write-downs, audits, SPEs, off-balance items
  PII = personal identifiers (SSN, address, medical info, individual salary)
  STRATEGIC = mergers, acquisitions, competitive plans, deal terms
  LEGAL = regulatory filings, legal exposure, attorney communications
  RELATIONAL = interpersonal grievances, workplace relationships, social hierarchy
  NONE = no sensitive organizational content

Dimension 2 - framing: one of [PROTECTED, UNPROTECTED, NA]
  PROTECTED = email contains "confidential", "do not forward", "between us", disclaimer
  UNPROTECTED = sensitive content with NO protective language
  NA = only if disclosure_type is NONE

JSON format exactly: {"disclosure_type": "...", "framing": "...", "confidence": 0.0-1.0}"""

def build_prompt(subject, body):
    body_excerpt = str(body)[:600]
    user_text = f"{SYSTEM_PROMPT}\n\nSubject: {subject}\n\nEmail: {body_excerpt}"
    # Phi-3 specific structure
    return f"<|user|>\n{user_text}<|end|>\n<|assistant|>\n"

def parse_label(output_text):
    match = re.search(r'\{.*?\}', output_text, re.DOTALL)
    if not match: return 'NONE', 'NA', 0.0
    try:
        parsed = json.loads(match.group())
        dt = parsed.get('disclosure_type', 'NONE')
        fr = parsed.get('framing', 'NA')
        conf = float(parsed.get('confidence', 0.5))
        
        valid_dt = ['FINANCIAL','PII','STRATEGIC','LEGAL','RELATIONAL','NONE']
        valid_fr = ['PROTECTED','UNPROTECTED','NA']
        if dt not in valid_dt: dt = 'NONE'
        if fr not in valid_fr: fr = 'NA'
        return dt, fr, conf
    except:
        return 'NONE', 'NA', 0.0

def main():
    print("="*50)
    print("PHASE 3A: PHI-3-MINI AUTO-LABELING (SILVER SET)")
    print("="*50)
    
    input_path = 'data/processed/emails_silver_pool.parquet'
    output_path = 'data/labeled/emails_labeled_silver.parquet'
    cache_path = 'data/labeled/silver_cache.json'
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    os.makedirs('data/labeled', exist_ok=True)

    df = pd.read_parquet(input_path)
    # df = df.head(50)  # Remove for 5000 emails, leaving full 5000 for actual production run
    print(f"Loaded {len(df)} emails to label.")

    # Load cache if recovering from crash
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            results_list = json.load(f)
        processed_mids = {r['mid'] for r in results_list}
        print(f"Resuming: found {len(processed_mids)} already processed emails in cache.")
    else:
        results_list = []
        processed_mids = set()

    emails_to_process = [row for _, row in df.iterrows() if row['mid'] not in processed_mids]
    
    if len(emails_to_process) == 0:
        print("All emails already processed!")
        finalize_labels(df, results_list, output_path)
        return

    tokenizer, model = get_phi3_pipeline()

    print(f"Beginning inference for {len(emails_to_process)} remaining emails...")
    
    for i, row in enumerate(tqdm(emails_to_process)):
        subject = str(row.get('subject',''))
        body = str(row.get('body_dense',''))
        prompt = build_prompt(subject, body)
        
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=80, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        dt, fr, conf = parse_label(response)
        
        results_list.append({
            'mid': row['mid'],
            'disclosure_type': dt,
            'framing': fr,
            'confidence': conf,
            'label_source': 'phi3_mini_4bit'
        })
        
        # Save cache every 10 emails
        if (i + 1) % 10 == 0:
            with open(cache_path, 'w') as f:
                json.dump(results_list, f)
                
    # Final save
    with open(cache_path, 'w') as f:
        json.dump(results_list, f)

    finalize_labels(df, results_list, output_path)

def finalize_labels(df, results_list, output_path):
    print("\nMerging and filtering labels...")
    results_df = pd.DataFrame(results_list)
    df_merged = df.merge(results_df, on='mid')
    
    # Keep high confidence
    df_high_conf = df_merged[df_merged['confidence'] >= 0.70].copy()
    
    def compute_risk_tier(row):
        if row['disclosure_type'] == 'NONE': return 'NONE'
        if row['framing'] == 'UNPROTECTED': return 'HIGH'
        return 'LOW'
        
    df_high_conf['risk_tier'] = df_high_conf.apply(compute_risk_tier, axis=1)
    df_high_conf['audience_scope'] = 'PENDING_KG'
    
    df_high_conf.to_parquet(output_path, index=False)
    print(f"Saved {len(df_high_conf)} high-confidence labels to {output_path}")
    print("PHASE 3A COMPLETE")

if __name__ == '__main__':
    main()
