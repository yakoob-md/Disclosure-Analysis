# ORGDISCLOSE — Implementation Plan
# Part 2 of 6: Phase 2 (Preprocessing) + Phase 3 (Annotation)

---

## PHASE 2: DATA PREPROCESSING & CLEANING

### Duration: 2–3 days
### Goal: Clean, deduplicated, alias-resolved email corpus ready for annotation sampling

---

### 2.1 Exact Tasks

Task 1: Deduplicate emails (MD5 hash on sender + date + first 200 chars of body)
Task 2: Resolve email aliases (rapidfuzz fuzzy matching)
Task 3: Parse and clean email body (strip quoted reply text, signatures, HTML)
Task 4: Normalize timestamps to UTC, add month_index column
Task 5: Filter system/auto-generated emails
Task 6: Add sender_role column from org chart lookup
Task 7: Save emails_clean.parquet
Task 8: Sample 5,800 emails using stratified strategy → emails_sample.parquet

---

### 2.2 Validation Standard (Must Pass Before Phase 3)

- [ ] emails_clean.parquet has 200,000–260,000 rows (20–35% reduction from raw)
- [ ] Zero duplicate rows (MD5 hash verified)
- [ ] `month_index` column present, values 0–35 (Jan 2000 = 0, Dec 2002 = 35)
- [ ] `sender_role` column has < 15% null rate
- [ ] emails_sample.parquet has exactly 5,800 rows
- [ ] Stratification verified: print class counts for each stratum
- [ ] No email body shorter than 10 words in the sample

---

### 2.3 IMPLEMENTATION PROMPT — PHASE 2 PREPROCESSING

```
=== IMPLEMENTATION PROMPT: PHASE 2 — PREPROCESSING & SAMPLING ===

CONTEXT:
- Input: data/raw/emails_raw.parquet (517k rows, columns: mid, sender, date,
  subject, body, recipients, recipient_types)
- Output: data/processed/emails_clean.parquet  +  data/processed/emails_sample.parquet
- All code in: phase2_preprocess.py

Write phase2_preprocess.py that does the following in sequence:

STEP 1 — DEDUPLICATION:
import hashlib
def make_hash(row):
    s = str(row['sender']) + str(row['date']) + str(row['body'])[:200]
    return hashlib.md5(s.encode()).hexdigest()
df['content_hash'] = df.apply(make_hash, axis=1)
df = df.drop_duplicates(subset='content_hash')
print(f"After dedup: {len(df)} rows")

STEP 2 — DATE PARSING & MONTH INDEX:
df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
df = df.dropna(subset=['date'])
base_date = pd.Timestamp('2000-01-01', tz='UTC')
df['month_index'] = ((df['date'].dt.year - 2000) * 12 + df['date'].dt.month - 1)
df = df[df['month_index'].between(0, 35)]   # Jan 2000 – Dec 2002
print(f"After date filter: {len(df)} rows, date range: {df['date'].min()} to {df['date'].max()}")

STEP 3 — ALIAS RESOLUTION:
Load the employee list from a hardcoded dict of known aliases (provide at least
these 10 key employees with their canonical forms):
ALIAS_MAP = {
    'ken.lay': 'Kenneth Lay', 'klay': 'Kenneth Lay',
    'jeff.skilling': 'Jeffrey Skilling', 'jskilling': 'Jeffrey Skilling',
    'andrew.fastow': 'Andrew Fastow', 'afastow': 'Andrew Fastow',
    'sherron.watkins': 'Sherron Watkins',
    'richard.causey': 'Richard Causey',
    'louise.kitchen': 'Louise Kitchen',
    'vince.kaminski': 'Vince Kaminski',
    'greg.whalley': 'Greg Whalley',
    'john.lavorato': 'John Lavorato',
    'sally.beck': 'Sally Beck'
}
def resolve_alias(email_str):
    if not isinstance(email_str, str): return email_str
    prefix = email_str.split('@')[0].lower().replace('_', '.').replace('-', '.')
    for alias, canonical in ALIAS_MAP.items():
        if alias in prefix: return canonical
    return email_str
df['sender_canonical'] = df['sender'].apply(resolve_alias)

STEP 4 — BODY CLEANING:
def clean_body(text):
    if not isinstance(text, str): return ''
    import re
    # Remove quoted reply text
    text = re.sub(r'(?m)^>.*$', '', text)
    # Remove "Original Message" blocks
    text = re.sub(r'-+\s*Original Message\s*-+.*', '', text, flags=re.DOTALL|re.IGNORECASE)
    # Remove "From:" forwarded headers
    text = re.sub(r'From:.*?Subject:.*?\n', '', text, flags=re.DOTALL|re.IGNORECASE)
    # Remove email signatures (lines starting with -- or phone patterns)
    text = re.sub(r'--\s*\n.*', '', text, flags=re.DOTALL)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text
df['body_clean'] = df['body'].apply(clean_body)
df['word_count'] = df['body_clean'].apply(lambda x: len(x.split()))
df = df[df['word_count'] >= 10]   # Remove very short emails

STEP 5 — FILTER SYSTEM EMAILS:
system_keywords = ['auto-response', 'out of office', 'delivery failed',
                   'undeliverable', 'mailer-daemon', 'postmaster', 'do not reply']
mask = df['body_clean'].str.lower().str.contains('|'.join(system_keywords), na=False)
df = df[~mask]
print(f"After system email filter: {len(df)} rows")

STEP 6 — SENDER ROLE ASSIGNMENT:
ROLE_MAP = {
    'Kenneth Lay': 'CEO', 'Jeffrey Skilling': 'CEO',
    'Andrew Fastow': 'CFO', 'Richard Causey': 'CFO',
    'Greg Whalley': 'VP', 'Louise Kitchen': 'VP',
    'John Lavorato': 'VP', 'Vince Kaminski': 'VP',
    'Sally Beck': 'Director', 'Sherron Watkins': 'Director'
}
df['sender_role'] = df['sender_canonical'].map(ROLE_MAP).fillna('Analyst')

STEP 7 — SAVE CLEAN FILE:
df.to_parquet('data/processed/emails_clean.parquet', index=False)
print(f"Saved emails_clean.parquet: {len(df)} rows")

STEP 8 — STRATIFIED SAMPLING (5,800 emails):
IMPORTANT: Split the 5,800 into:
  - 800 emails → Gold Set (for human annotation): emails_gold_pool.parquet
  - 5,000 emails → Silver Set (for LLM auto-labeling): emails_silver_pool.parquet

Stratification criteria for the 5,800 sample:
a) time_period: take 60% from month_index >= 18 (crisis period = mid-2001 onward)
   and 40% from month_index < 18 (stable period)
b) sender_role: ensure CEO/CFO emails are oversampled — include ALL available
   CEO/CFO emails (typically 2000–4000), fill rest from VP/Director/Analyst
c) has_financial_kw: flag emails containing keywords:
   ['reserve', 'write-down', 'write down', 'mark-to-market', 'FERC', 'SPE',
    'off-balance', 'audit', 'restatement', 'confidential', 'merger', 'acquisition']
   Ensure 30% of sample has at least one financial keyword
d) word_count bucket: short(<80 words), medium(80-300), long(>300) — equal thirds

Code:
from sklearn.model_selection import train_test_split
# [implement stratified sampling satisfying all 4 criteria above]
# Save both pools
gold_pool.to_parquet('data/processed/emails_gold_pool.parquet', index=False)
silver_pool.to_parquet('data/processed/emails_silver_pool.parquet', index=False)
print(f"Gold pool: {len(gold_pool)}, Silver pool: {len(silver_pool)}")

VALIDATION REPORT at end of script:
Print:
1. Total rows in emails_clean.parquet
2. Reduction % from raw
3. month_index distribution (print value_counts for 0-5, 12-17, 18-23, 24+)
4. sender_role distribution
5. Gold pool stratification counts
6. Silver pool stratification counts
Print "PHASE 2 COMPLETE — ALL CHECKS PASSED" if all validations pass.

OUTPUT FILES:
- data/processed/emails_clean.parquet
- data/processed/emails_gold_pool.parquet   (800 rows for human annotation)
- data/processed/emails_silver_pool.parquet  (5,000 rows for LLM labeling)
- phase2_preprocess.py

=== END OF PROMPT ===
```

---

## PHASE 3: ANNOTATION

### Duration: 10–14 days (parallel annotation work)
### Goal: 800 Gold emails human-labeled with κ ≥ 0.6, 5000 Silver auto-labeled

---

### 3.1 Label Schema (Final — Commit to This)

```
DIMENSION 1: disclosure_type
  Values: FINANCIAL | PII | STRATEGIC | LEGAL | RELATIONAL | NONE
  FINANCIAL  = earnings, reserves, write-downs, audits, SPEs, FERC filings
  PII        = personal identity info (SSN, address, medical, salary of individuals)
  STRATEGIC  = mergers, acquisitions, competitive plans, deal terms, price strategy
  LEGAL      = regulatory filings, compliance violations, attorney communications
  RELATIONAL = interpersonal grievances, social hierarchy, off-record relationships
  NONE       = no sensitive organizational content

DIMENSION 2: framing
  Values: PROTECTED | UNPROTECTED | NA
  PROTECTED   = email contains explicit confidentiality markers:
                "please keep this between us", "do not forward", "confidential",
                "privileged and confidential", disclaimer footer present
  UNPROTECTED = sensitive content present (disc_type ≠ NONE) with NO protective markers
  NA          = disc_type = NONE (no disclosure, framing irrelevant)

DIMENSION 3: audience_scope [AUTO-COMPUTED FROM KG — NOT MANUALLY LABELED]
  Values: INTERNAL_AUTH | INTERNAL_UNAUTH | EXTERNAL
  Computed by querying the role hierarchy in Neo4j.
  Human annotators do NOT label this.

DIMENSION 4: risk_tier [RULE-COMPUTED — NOT MANUALLY LABELED]
  Values: NONE | LOW | HIGH
  Rule:
    if disc_type = NONE → NONE
    if disc_type ≠ NONE AND framing = PROTECTED AND audience = INTERNAL_AUTH → LOW
    if disc_type ≠ NONE AND (framing = UNPROTECTED OR audience = EXTERNAL) → HIGH
    else → LOW

ANNOTATORS ONLY LABEL DIMENSIONS 1 AND 2.
```

---

### 3.2 Annotation Guidelines Summary (for yourself + peer annotator)

```
DECISION TREE FOR disc_type:
1. Does the email mention specific dollar amounts, financial instruments,
   company earnings, losses, reserves, write-downs, or SPEs? → FINANCIAL
2. Does it contain personal identifiers (SSN, address, personal salary)? → PII
3. Does it discuss a merger, acquisition, deal terms, or price strategy? → STRATEGIC
4. Is it about regulatory filings, legal exposure, attorney-client communication? → LEGAL
5. Does it discuss personal relationships, workplace grievances, or social dynamics
   involving organizational hierarchy? → RELATIONAL
6. None of the above? → NONE

RULE: If multiple types apply, pick the PRIMARY (highest risk) type.
Order of precedence: FINANCIAL > LEGAL > PII > STRATEGIC > RELATIONAL > NONE

DECISION TREE FOR framing:
1. Is disc_type = NONE? → NA
2. Does the email text contain any of these phrases?
   "confidential", "do not forward", "please keep", "between us", "off the record",
   "not for distribution", disclaimer footer? → PROTECTED
3. Otherwise → UNPROTECTED
```

---

### 3.3 Validation Standard (Must Pass Before Phase 4)

- [ ] 800 Gold emails manually labeled (400 by annotator A, 400 by annotator B,
      with 150 overlap for κ computation)
- [ ] Cohen's κ ≥ 0.60 on disc_type dimension (hard gate)
- [ ] Cohen's κ ≥ 0.70 on framing dimension (expected — it's more objective)
- [ ] Class distribution: no class < 5% in disc_type (resample if needed)
- [ ] 5,000 Silver emails labeled by local LLM
- [ ] Silver spot-check: 100 random Silver labels reviewed → noise rate ≤ 25%
- [ ] emails_labeled_gold.parquet and emails_labeled_silver.parquet saved

---

### 3.4 IMPLEMENTATION PROMPT — PHASE 3A: LLM AUTO-LABELING (SILVER SET)

```
=== IMPLEMENTATION PROMPT: PHASE 3A — LLM AUTO-LABELING WITH LOCAL MISTRAL-7B ===

CONTEXT:
- Input: data/processed/emails_silver_pool.parquet (5,000 rows)
- Output: data/labeled/emails_labeled_silver.parquet
- Hardware: RTX 2050 (4GB VRAM) — use 4-bit quantization
- No internet API — use local Mistral-7B-Instruct

Write phase3a_autolabel.py that does the following:

STEP 1 — Load model in 4-bit:
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
model.eval()

STEP 2 — Define labeling prompt:
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

JSON format exactly: {"disclosure_type": "...", "framing": "...", "confidence": 0.0-1.0}
The confidence should reflect how certain you are (0.0 = not certain, 1.0 = certain)."""

def build_prompt(subject, body):
    body_excerpt = body[:600]  # Truncate to keep prompt short
    return f"[INST] {SYSTEM_PROMPT}\n\nSubject: {subject}\n\nEmail: {body_excerpt} [/INST]"

STEP 3 — Batch inference with error handling:
import json, re
from tqdm import tqdm

def parse_label(output_text):
    # Extract JSON from response
    match = re.search(r'\{.*?\}', output_text, re.DOTALL)
    if not match: return None, None, 0.0
    try:
        parsed = json.loads(match.group())
        dt = parsed.get('disclosure_type', 'NONE')
        fr = parsed.get('framing', 'NA')
        conf = float(parsed.get('confidence', 0.5))
        # Validate
        valid_dt = ['FINANCIAL','PII','STRATEGIC','LEGAL','RELATIONAL','NONE']
        valid_fr = ['PROTECTED','UNPROTECTED','NA']
        if dt not in valid_dt: dt = 'NONE'
        if fr not in valid_fr: fr = 'NA'
        return dt, fr, conf
    except: return None, None, 0.0

df = pd.read_parquet('data/processed/emails_silver_pool.parquet')
results = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    prompt = build_prompt(str(row.get('subject','')), str(row.get('body_clean','')))
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to('cuda')
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=80, temperature=0.1, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    dt, fr, conf = parse_label(response)
    results.append({'mid': row['mid'], 'disclosure_type': dt, 'framing': fr,
                    'confidence': conf, 'label_source': 'mistral7b_4bit'})

STEP 4 — Merge and filter by confidence:
results_df = pd.DataFrame(results)
df_merged = df.merge(results_df, on='mid')

# Keep only high-confidence labels (≥ 0.70)
df_high_conf = df_merged[df_merged['confidence'] >= 0.70]
df_low_conf  = df_merged[df_merged['confidence'] < 0.70]
print(f"High confidence labels: {len(df_high_conf)} / {len(df_merged)}")
print(f"Low confidence (excluded from training): {len(df_low_conf)}")

# Handle parse failures
df_high_conf = df_high_conf.dropna(subset=['disclosure_type'])

STEP 5 — Add derived labels (rule-computed):
# risk_tier = NONE if disc_type=NONE, HIGH if UNPROTECTED, else LOW
def compute_risk_tier(row):
    if row['disclosure_type'] == 'NONE': return 'NONE'
    if row['framing'] == 'UNPROTECTED': return 'HIGH'
    return 'LOW'
df_high_conf['risk_tier'] = df_high_conf.apply(compute_risk_tier, axis=1)
# audience_scope = placeholder (computed in Phase 6 from KG)
df_high_conf['audience_scope'] = 'PENDING_KG'

STEP 6 — Save:
df_high_conf.to_parquet('data/labeled/emails_labeled_silver.parquet', index=False)
print(f"Saved silver labels: {len(df_high_conf)} rows")

STEP 7 — Label distribution report:
print("\n=== SILVER LABEL DISTRIBUTION ===")
print(df_high_conf['disclosure_type'].value_counts(normalize=True).round(3))
print(df_high_conf['framing'].value_counts(normalize=True).round(3))
print(df_high_conf['risk_tier'].value_counts(normalize=True).round(3))

VALIDATION: Print "PHASE 3A COMPLETE" if:
- At least 3,500 of 5,000 rows have high-confidence labels
- No single class dominates > 60% in disclosure_type

OUTPUT: data/labeled/emails_labeled_silver.parquet
=== END OF PROMPT ===
```

---

### 3.5 IMPLEMENTATION PROMPT — PHASE 3B: INTER-RATER AGREEMENT COMPUTATION

```
=== IMPLEMENTATION PROMPT: PHASE 3B — COHEN'S KAPPA COMPUTATION ===

CONTEXT:
- You have 2 annotators who have labeled 150 emails each (same 150 overlap set)
- These labels are stored in two CSV files:
  data/labeled/annotator_a_150.csv  (columns: mid, disclosure_type, framing)
  data/labeled/annotator_b_150.csv  (columns: mid, disclosure_type, framing)
- The full 800 Gold labels from annotator A are in:
  data/labeled/annotator_a_full_800.csv

Write phase3b_kappa.py that does the following:

STEP 1 — Load and merge overlap sets:
df_a = pd.read_csv('data/labeled/annotator_a_150.csv')
df_b = pd.read_csv('data/labeled/annotator_b_150.csv')
overlap = df_a.merge(df_b, on='mid', suffixes=('_a', '_b'))
print(f"Overlap set size: {len(overlap)}")

STEP 2 — Compute Cohen's Kappa per dimension:
from sklearn.metrics import cohen_kappa_score
kappa_dt = cohen_kappa_score(overlap['disclosure_type_a'], overlap['disclosure_type_b'])
kappa_fr = cohen_kappa_score(overlap['framing_a'], overlap['framing_b'])
print(f"Cohen's κ — disclosure_type: {kappa_dt:.3f}")
print(f"Cohen's κ — framing:         {kappa_fr:.3f}")

STEP 3 — Hard gate check:
if kappa_dt < 0.60:
    print("FAILED: disclosure_type κ < 0.60. Review annotation guide.")
    print("Common disagreement cases:")
    mask = overlap['disclosure_type_a'] != overlap['disclosure_type_b']
    print(overlap[mask][['mid','disclosure_type_a','disclosure_type_b']].head(20))
    print("ACTION: Revise annotation guide for the conflicting classes and re-annotate.")
    exit(1)  # Stop pipeline
if kappa_fr < 0.70:
    print("WARNING: framing κ < 0.70 but above minimum. Check PROTECTED criteria.")

STEP 4 — Build final Gold label set (combining both annotators):
# Use annotator A labels as primary for all 800
# For the 150 overlap emails: use majority vote (A wins on tie since A did all 800)
df_gold_full = pd.read_csv('data/labeled/annotator_a_full_800.csv')
# Add computed dimensions
def compute_risk_tier(row):
    if row['disclosure_type'] == 'NONE': return 'NONE'
    if row['framing'] == 'UNPROTECTED': return 'HIGH'
    return 'LOW'
df_gold_full['risk_tier'] = df_gold_full.apply(compute_risk_tier, axis=1)
df_gold_full['audience_scope'] = 'PENDING_KG'
df_gold_full['label_source'] = 'human_gold'
df_gold_full.to_parquet('data/labeled/emails_labeled_gold.parquet', index=False)

STEP 5 — Print final distribution report:
print("\n=== GOLD LABEL DISTRIBUTION ===")
print(df_gold_full['disclosure_type'].value_counts())
print(df_gold_full['framing'].value_counts())
print(df_gold_full['risk_tier'].value_counts())

STEP 6 — Class imbalance check:
for cls in ['FINANCIAL','PII','STRATEGIC','LEGAL','RELATIONAL','NONE']:
    pct = (df_gold_full['disclosure_type'] == cls).mean()
    if pct < 0.05:
        print(f"WARNING: {cls} has only {pct:.1%} — may need oversampling for training")

VALIDATION: Print "PHASE 3B COMPLETE" only if kappa_dt >= 0.60 and kappa_fr >= 0.65
OUTPUT: data/labeled/emails_labeled_gold.parquet + kappa report printed to console
=== END OF PROMPT ===
```
