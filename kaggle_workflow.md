# Moving Phase 3 to a Kaggle GPU (Workflow)

This is a fantastic strategy. A free Kaggle **P100 (16GB VRAM)** or **T4 x2 (32GB VRAM)** will drop your processing time from 9 hours down to roughly **15-30 minutes**.

Here is the exact step-by-step guide to do this safely and bring the finished data back to your local project pipeline.

---

### Step 1: Upload Your Data to Kaggle
1. Go to [Kaggle](https://www.kaggle.com) and click **Create -> New Notebook**.
2. On the right side panel, click **Add Data -> Upload (the upload arrow)**.
3. Name your dataset `enron-silver-pool`.
4. Upload exactly this one file from your local computer: 
   `c:\Users\dabaa\OneDrive\Desktop\workplace_nlp\data\processed\emails_silver_pool.parquet`
5. Click **Create** to let it upload.

### Step 2: Configure Kaggle Hardware
1. On the right panel, under **Session Options** -> **Accelerator**, select **GPU T4 x2** or **GPU P100**.
2. Turn ON the accelerator and allow the session to start.

### Step 3: Run the Labeling Engine
Create a single code block in your Kaggle notebook, copy-paste the code below, and press Run. 

*(Note: This is the exact same script we built locally, but I removed the 4-bit restrictors and updated the folder paths for Kaggle's environment so it runs at maximum speed!)*

```python
!pip install -q transformers pandas pyarrow tqdm accelerate

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json, re, os
from tqdm import tqdm

input_path = '/kaggle/input/enron-silver-pool/emails_silver_pool.parquet' # Adjust if Kaggle names it differently
output_path = '/kaggle/working/emails_labeled_silver.parquet'

print("Loading Phi-3-Mini into full GPU VRAM...")
model_name = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16, # Full speed, no 4-bit compression needed here!
    device_map="auto",
    trust_remote_code=True
)
model.eval()

SYSTEM_PROMPT = """You are an expert in organizational email risk analysis.
Label the following email along 2 dimensions. Reply ONLY with valid JSON. No explanation.

Dimension 1 - disclosure_type: one of [FINANCIAL, PII, STRATEGIC, LEGAL, RELATIONAL, NONE]
Dimension 2 - framing: one of [PROTECTED, UNPROTECTED, NA]

JSON format exactly: {"disclosure_type": "...", "framing": "...", "confidence": 0.0-1.0}"""

def build_prompt(subject, body):
    user_text = f"{SYSTEM_PROMPT}\n\nSubject: {subject[:100]}\n\nEmail: {str(body)[:600]}"
    return f"<|user|>\n{user_text}<|end|>\n<|assistant|>\n"

def parse_label(output_text):
    match = re.search(r'\{.*?\}', output_text, re.DOTALL)
    if not match: return 'NONE', 'NA', 0.0
    try:
        parsed = json.loads(match.group())
        dt = parsed.get('disclosure_type', 'NONE')
        fr = parsed.get('framing', 'NA')
        return dt, fr, float(parsed.get('confidence', 0.5))
    except: return 'NONE', 'NA', 0.0

print("Starting inference on 5000 emails...")
df = pd.read_parquet(input_path)
results_list = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    prompt = build_prompt(row.get('subject',''), row.get('body_dense',''))
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=80, pad_token_id=tokenizer.eos_token_id)
        
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    dt, fr, conf = parse_label(response)
    
    results_list.append({
        'mid': row['mid'],
        'disclosure_type': dt,
        'framing': fr,
        'confidence': conf,
        'label_source': 'phi3_mini_kaggle'
    })

results_df = pd.DataFrame(results_list)
df_merged = df.merge(results_df, on='mid')

# Keep high confidence and compute target Risk Tier
df_high_conf = df_merged[df_merged['confidence'] >= 0.70].copy()
df_high_conf['risk_tier'] = df_high_conf.apply(
    lambda r: 'NONE' if r['disclosure_type'] == 'NONE' else ('HIGH' if r['framing'] == 'UNPROTECTED' else 'LOW'), 
    axis=1
)
df_high_conf['audience_scope'] = 'PENDING_KG'
df_high_conf.to_parquet(output_path, index=False)

print(f"DONE! Saved {len(df_high_conf)} labeled emails. Please download the output file.")
```

### Step 4: Bring the Data Back Home
1. Under your Kaggle Notebook's "Output" tab (usually on the right or bottom), you will see **`emails_labeled_silver.parquet`**.
2. Click **Download**.
3. Move that downloaded file into your local project here:
   `c:\Users\dabaa\OneDrive\Desktop\workplace_nlp\data\labeled\emails_labeled_silver.parquet`
4. Once it is physically in that folder, come back to our chat (or open a new local terminal) and simply type: `python scripts/phase3_results.py`. This will seamlessly bridge the gap, log the results, and let us instantly start the remaining local ML pipelines!
