# ORGDISCLOSE — AMENDED IMPLEMENTATION FILES
# Two-Track System: Track A (Tomorrow Demo) + Track B (Research Paper)
# ====================================================================
# 
# READ THIS BEFORE TOUCHING ANY CODE
# ====================================================================
#
# TRACK A = demo_pipeline.py         → Run tomorrow. No bugs. No GPU needed.
# TRACK B = impl_part1 through part6 → Research paper. Fix the bugs below first.
#
# This file lists EVERY amendment to the 6 implementation files, organized by:
#   File → Section → Bug/Issue → Exact Fix
# ====================================================================


# ════════════════════════════════════════════════════════════════════
# AMENDMENT 1: impl_part1_novelty_and_phase1.md
# ════════════════════════════════════════════════════════════════════
"""
AMENDMENT 1A — requirements.txt: bitsandbytes version

PROBLEM:
  bitsandbytes==0.43.1 fails on Windows + CUDA 12.x.
  Phase 3A, 8A, 8B, 9, and 11 all fail at import if this is not fixed.

FIX — Replace in requirements.txt:
  REMOVE:  bitsandbytes==0.43.1
  ADD:     # Install manually from Windows wheel (see below)

FIX — Add to phase1_setup.py, after pip install:

import subprocess, sys, platform
if platform.system() == 'Windows':
    print("Installing bitsandbytes Windows wheel...")
    whl_url = (
        "https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/"
        "wheels/bitsandbytes-0.41.3.post2-py3-none-win_amd64.whl"
    )
    subprocess.run([sys.executable, '-m', 'pip', 'install', whl_url], check=True)
else:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'bitsandbytes==0.43.1'],
                   check=True)
"""

"""
AMENDMENT 1B — phase1_setup.py: wrong data format for download URL

PROBLEM:
  The CMU URL (enron_mail_20150507.tgz) is maildir format.
  But the schema referenced (message table, recipientinfo table)
  is from the Adibi SQL version — a different dataset entirely.
  These two are incompatible. Running the SQL queries on the maildir
  tarball will produce zero results or a crash.

FIX — Replace the download task with:

# Option 1: Kaggle CSV (easiest, no SQL needed)
#   kaggle datasets download -d wcukierski/enron-email-dataset
#   → produces emails.csv with 517,401 rows

# Option 2: Adibi SQL version (for SQL schema)
#   URL: https://aws.amazon.com/datasets/enron-email-data/
#   File: enron_mail_20150507.tar.gz → contains .db file

# Option 3: Maildir version (for full headers)
#   URL: https://www.cs.cmu.edu/~enron/enron_mail_20150507.tgz
#   Requires Python email module parsing (see demo_pipeline.py load_maildir())

RECOMMENDATION: Use the Kaggle CSV for Track A and Track B Phase 1.
It is the most accessible and consistent format.

Add this to phase1_setup.py:

import subprocess, sys

def download_enron_kaggle():
    try:
        subprocess.run(['kaggle', 'datasets', 'download',
                        '-d', 'wcukierski/enron-email-dataset',
                        '-p', 'data/raw/', '--unzip'], check=True)
        print("Downloaded emails.csv from Kaggle")
    except FileNotFoundError:
        print("kaggle CLI not found. Download manually:")
        print("  pip install kaggle")
        print("  Set KAGGLE_USERNAME and KAGGLE_KEY in environment")
        print("  kaggle datasets download -d wcukierski/enron-email-dataset")
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        print("Manual alternative: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset")
"""


# ════════════════════════════════════════════════════════════════════
# AMENDMENT 2: impl_part2_phase2_and_phase3.md
# ════════════════════════════════════════════════════════════════════
"""
AMENDMENT 2A — phase2_preprocess.py: Alias resolution covers only 2% of senders

PROBLEM:
  The ALIAS_MAP has 10 entries. The Enron corpus has ~150 unique senders.
  90%+ of senders will fall through as 'Analyst' by default.
  This makes sender_canonical meaningless, which breaks:
    - phi_G feature computation (emp_idx lookup returns 'not found')
    - audience_scope computation (hierarchy check fails)
    - KG edge creation (most SENT edges go to ghost nodes)
  The role features will carry essentially zero signal.

FIX — Replace the ALIAS_MAP and resolve_alias function in phase2_preprocess.py
with the extended version from demo_pipeline.py (40+ employees).
Additionally, add fuzzy matching as fallback:

from rapidfuzz import process, fuzz

# After hardcoded alias lookup fails:
def resolve_alias_with_fuzzy(email_str, canonical_emails_list):
    prefix = email_str.split('@')[0].lower().replace('_','.').replace('-','.')
    # First try exact prefix match (fast)
    for alias, canonical in ALIAS_MAP.items():
        if alias in prefix:
            return canonical
    # Then try fuzzy match against known canonical prefixes (slower, more accurate)
    canonical_prefixes = {
        name.lower().replace(' ','.'): name for name in ROLE_MAP.keys()
    }
    match, score, _ = process.extractOne(
        prefix, list(canonical_prefixes.keys()), scorer=fuzz.token_set_ratio
    )
    if score >= 80:
        return canonical_prefixes[match]
    return email_str   # return as-is rather than defaulting to 'Analyst'
"""

"""
AMENDMENT 2B — phase2_preprocess.py: Body cleaning regex corrupts email content

PROBLEM:
  This regex in clean_body():
    text = re.sub(r'From:.*?Subject:.*?\n', '', text, flags=re.DOTALL|re.IGNORECASE)
  Uses re.DOTALL which makes .* match newlines.
  In a forwarded email: "From: <sender>\nDate: ...\nTo: ...\nSubject: ...\n"
  This will eat everything between the first "From:" and the last "Subject:"
  in the entire email — potentially deleting the entire body.

FIX — Replace that line with:
  # Remove forwarded header blocks by splitting at separator
  text = re.split(r'-{2,}\s*Original Message\s*-{2,}', text,
                  flags=re.IGNORECASE)[0]
  # Remove single forwarded From: lines (not multiline match)
  text = re.sub(r'(?m)^From:.*$', '', text)
  text = re.sub(r'(?m)^Sent:.*$', '', text)
  text = re.sub(r'(?m)^To:.*$', '', text)
"""

"""
AMENDMENT 2C — phase2_preprocess.py: Body truncation loses critical content

PROBLEM:
  In phase3a_autolabel.py STEP 2:
    body_excerpt = body[:600]
  Financial disclosure emails often begin with greetings/pleasantries
  and contain the sensitive content in paragraph 2+.
  Naive head-truncation misses it.

FIX — Replace body_excerpt truncation with information-dense extraction:

from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Pre-compute a small TF-IDF to identify information-dense sentences
DISCLOSURE_TERMS = [
    'reserve', 'write-down', 'audit', 'ferc', 'spe', 'merger',
    'confidential', 'do not forward', 'mark-to-market', 'restatement',
    'privileged', 'settlement', 'compliance', 'ssn', 'acquisition'
]

def extract_dense_excerpt(text, max_chars=800):
    '''Extract the most disclosure-relevant 800 chars from email body.'''
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return text[:max_chars]
    # Score each sentence by disclosure term hits
    scores = []
    for s in sentences:
        score = sum(1 for term in DISCLOSURE_TERMS if term in s.lower())
        scores.append(score)
    # Take top-scoring sentences up to max_chars
    ranked = sorted(zip(scores, range(len(sentences))), reverse=True)
    selected = []
    total_chars = 0
    for score, idx in ranked:
        s = sentences[idx]
        if total_chars + len(s) <= max_chars:
            selected.append((idx, s))
            total_chars += len(s)
    # Re-order by original position (preserve reading order)
    selected.sort(key=lambda x: x[0])
    excerpt = ' '.join([s for _, s in selected])
    return excerpt if excerpt else text[:max_chars]
"""

"""
AMENDMENT 2D — phase3b_kappa.py: Annotator B coverage is inadequate

PROBLEM:
  The plan uses annotator A for all 800 gold emails and annotator B for only
  the 150 overlap subset. This means 650 emails have single-annotator labels.
  Most IEEE/ACM venues require double annotation OR at minimum a statement that
  single-annotator labels were validated against the κ computed on the overlap.
  The current plan is acceptable but must be stated transparently in the paper.

FIX — Add this note to paper Section 3 (Dataset):
  "While Annotator B independently labeled 150 emails (the overlap set) to
  compute inter-rater agreement (κ = X.XX), the remaining 650 gold labels were
  assigned by Annotator A following the verified annotation guide. We consider
  this acceptable given the deterministic nature of our decision tree and the
  high κ on the overlap set."

ADDITIONALLY: In phase3b_kappa.py, add Krippendorff's Alpha as a secondary
agreement measure (more robust for ordinal scales):

import krippendorff   # pip install krippendorff

# Encode labels as integers for Krippendorff
def encode_labels(labels, valid_list):
    return [valid_list.index(l) if l in valid_list else -1 for l in labels]

valid_disc_types = ['FINANCIAL','PII','STRATEGIC','LEGAL','RELATIONAL','NONE']
alpha_data = np.array([
    encode_labels(overlap['disclosure_type_a'].tolist(), valid_disc_types),
    encode_labels(overlap['disclosure_type_b'].tolist(), valid_disc_types)
])
alpha = krippendorff.alpha(reliability_data=alpha_data, level_of_measurement='nominal')
print(f"Krippendorff's α — disclosure_type: {alpha:.3f}")
"""


# ════════════════════════════════════════════════════════════════════
# AMENDMENT 3: impl_part3_phase4_and_phase5.md
# ════════════════════════════════════════════════════════════════════
"""
AMENDMENT 3A — phase5a_kg_build.py: One Neo4j session per email (performance crash)

PROBLEM:
  The SENT/RECEIVED_BY edge creation loop opens and closes a new Neo4j session
  for every email. For 5,800 emails with ~5 recipients each, this opens ~29,000
  TCP connections. At 20-50ms each, this is 10-25 minutes of session overhead
  before a single Cypher query runs. The script will appear to freeze.

FIX — Replace the email-by-email loop in STEP 5 with batched transactions:

BATCH_SIZE = 200

def flush_sent_batch(session, batch):
    '''Insert SENT edges for a batch of emails in one transaction.'''
    session.run(
        "UNWIND $rows AS row "
        "MATCH (emp:Employee {email: row.sender_email}) "
        "MATCH (m:Email {mid: row.mid}) "
        "MERGE (emp)-[:SENT]->(m)",
        rows=batch
    )

def flush_recv_batch(session, batch):
    '''Insert RECEIVED_BY edges for a batch. Handles internal + external.'''
    # Internal
    session.run(
        "UNWIND $rows AS row "
        "MATCH (m:Email {mid: row.mid}) "
        "MERGE (r:Employee {email: row.recip}) "
        "MERGE (m)-[:RECEIVED_BY]->(r)",
        rows=[r for r in batch if r['is_internal']]
    )
    # External
    session.run(
        "UNWIND $rows AS row "
        "MATCH (m:Email {mid: row.mid}) "
        "MERGE (r:ExternalParty {email: row.recip}) "
        "MERGE (m)-[:RECEIVED_BY]->(r)",
        rows=[r for r in batch if not r['is_internal']]
    )

sent_batch, recv_batch = [], []
with driver.session(database="orgdisclose") as session:
    for _, row in tqdm(all_emails.iterrows(), total=len(all_emails)):
        mid = str(row['mid'])
        sender = str(row.get('sender','')).lower()
        # Match sender
        for emp_tuple in EMPLOYEES:
            if emp_tuple[0].split('@')[0] in sender:
                sent_batch.append({'sender_email': emp_tuple[0], 'mid': mid})
                break
        # Parse recipients
        recips_str = str(row.get('recipients',''))
        for recip in recips_str.split(';')[:10]:
            recip = recip.strip().lower()
            if recip and '@' in recip:
                recv_batch.append({
                    'mid': mid,
                    'recip': recip,
                    'is_internal': 'enron.com' in recip
                })
        # Flush in batches
        if len(sent_batch) >= BATCH_SIZE:
            flush_sent_batch(session, sent_batch)
            sent_batch = []
        if len(recv_batch) >= BATCH_SIZE:
            flush_recv_batch(session, recv_batch)
            recv_batch = []
    # Final flush
    if sent_batch:  flush_sent_batch(session, sent_batch)
    if recv_batch:  flush_recv_batch(session, recv_batch)
print("All edges created.")
"""

"""
AMENDMENT 3B — phase5a_kg_build.py: Only 15 employees in KG

PROBLEM:
  The hardcoded EMPLOYEES list has 15 entries. ~135 real Enron employees
  are missing. This means:
    - ~90% of SENT edges cannot be created (no matching Employee node)
    - audience_scope defaults to INTERNAL_AUTH for 98% of emails
    - Novelty Contribution #3 (KG-as-annotation-oracle) cannot be demonstrated

FIX — Expand to all 151 Enron employees using the Shetty-Adibi employee list.
Download from: http://www.cs.cmu.edu/~enron/ (employeelist.csv)

EMPLOYEES_EXTENDED = []
try:
    emp_df = pd.read_csv('data/raw/employeelist.csv')
    for _, row in emp_df.iterrows():
        EMPLOYEES_EXTENDED.append((
            str(row.get('Email_id','')).lower().strip(),
            str(row.get('name','')).strip(),
            str(row.get('role', 'Analyst')).strip(),
            str(row.get('department','Unknown')).strip(),
            int(row.get('seniority_level', 5))
        ))
except FileNotFoundError:
    print("employeelist.csv not found — using hardcoded 40-employee list")
    # Fall back to extended hardcoded list from demo_pipeline.py ROLE_MAP
    EMPLOYEES_EXTENDED = [
        (f"{name.lower().replace(' ','.')}@enron.com", name, role, dept, sl)
        for name, (role, dept, sl) in ROLE_MAP.items()
    ]

# Ingest all in one batch
with driver.session(database="orgdisclose") as session:
    session.run(
        "UNWIND $employees AS emp "
        "MERGE (e:Employee {email: emp[0]}) "
        "SET e.name=emp[1], e.role=emp[2], e.department=emp[3], e.seniority_level=emp[4]",
        employees=EMPLOYEES_EXTENDED
    )
"""

"""
AMENDMENT 3C — phase5a_kg_build.py: audience_scope computation is incomplete

PROBLEM:
  The get_audience_scope() function only checks for ExternalParty recipients.
  All internal emails default to 'INTERNAL_AUTH' without actually checking
  whether the recipient's seniority is appropriate for the disclosure.
  This makes the fourth dimension of the taxonomy a constant.

FIX — Replace the audience_scope function with:

def get_audience_scope_full(mid, sender_email):
    with driver.session(database="orgdisclose") as session:
        # Check external recipients
        ext_count = session.run(
            "MATCH (m:Email {mid:$mid})-[:RECEIVED_BY]->(r:ExternalParty) "
            "RETURN count(r) as c", mid=str(mid)
        ).single()['c']
        if ext_count > 0:
            return 'EXTERNAL'

        # Get sender seniority
        sender_info = session.run(
            "MATCH (e:Employee {email:$email}) RETURN e.seniority_level as sl",
            email=sender_email
        ).data()
        if not sender_info:
            return 'INTERNAL_AUTH'
        sender_sl = sender_info[0]['sl']

        # Check if any recipient has higher seniority (lower number = more senior)
        # This would be UNUSUAL: a junior person sending sensitive info to senior
        # is actually not unauthorized; it's a subordinate to superior, which is normal.
        # INTERNAL_UNAUTH = senior person disclosing to someone outside their reporting chain
        cross_dept = session.run(
            "MATCH (sender:Employee {email:$sender})-[:REPORTS_TO*0..5]->(chain:Employee) "
            "WITH collect(chain.email) as reporting_chain "
            "MATCH (m:Email {mid:$mid})-[:RECEIVED_BY]->(r:Employee) "
            "WHERE NOT r.email IN reporting_chain "
            "RETURN count(r) as c",
            sender=sender_email, mid=str(mid)
        ).single()['c']

        if cross_dept > 0:
            return 'INTERNAL_UNAUTH'   # sent outside reporting chain
        return 'INTERNAL_AUTH'
"""

"""
AMENDMENT 3D — phase4a_tfidf.py: LabelEncoder on test set may fail

PROBLEM:
  The temporal split means the test set (crisis period) may contain classes
  that are rare or absent in the training set. LabelEncoder.transform() on
  unseen classes raises ValueError.

FIX — Replace LabelEncoder with predefined class list:

# Define classes explicitly — do NOT fit from data
CLASS_ORDER = ['FINANCIAL', 'LEGAL', 'PII', 'STRATEGIC', 'RELATIONAL', 'NONE']
class_to_idx = {c: i for i, c in enumerate(CLASS_ORDER)}

def encode_labels(label_series):
    return np.array([class_to_idx.get(str(l), class_to_idx['NONE'])
                     for l in label_series])

y_train = encode_labels(train_df['disclosure_type'])
y_val   = encode_labels(val_df['disclosure_type'])
y_test  = encode_labels(test_df['disclosure_type'])
"""


# ════════════════════════════════════════════════════════════════════
# AMENDMENT 4: impl_part4_phase6_7_8.md
# ════════════════════════════════════════════════════════════════════
"""
AMENDMENT 4A — phase6_centrality.py: delta_padded uses wrong padding value

PROBLEM:
  delta_padded = np.hstack([delta_betweenness[:, [0]], delta_betweenness])
  This copies the first computed delta (month 1) as the value for month 0.
  Month 0 has no previous month — its derivative is undefined. Using the
  first real delta value as a proxy artificially inflates or deflates the
  month 0 feature.

FIX — Pad with zeros (undefined derivative = no change signal):
  delta_padded = np.hstack([np.zeros((n_emp, 1)), delta_betweenness])
"""

"""
AMENDMENT 4B — phase6_centrality.py: Undirected closeness loses directed structure

PROBLEM:
  G_undirected = G.to_undirected()
  close = nx.closeness_centrality(G_cc)  ← computed on undirected largest CC
  This ignores the direction of email flow. For disclosure analysis, what matters
  is how quickly a sender can reach the rest of the network DIRECTIONALLY.
  Converting to undirected treats "A emails B" and "B emails A" as equivalent,
  which they are not for brokerage analysis.

FIX — Use directed closeness with wf_improved (handles disconnected directed graphs):
  # REMOVE these 3 lines:
  # G_undirected = G.to_undirected()
  # largest_cc = max(nx.connected_components(G_undirected), key=len)
  # G_cc = G_undirected.subgraph(largest_cc)
  # close = nx.closeness_centrality(G_cc)

  # REPLACE with:
  close = nx.closeness_centrality(G, wf_improved=True)
  # wf_improved=True uses Wasserman-Faust normalization that properly handles
  # nodes with no outgoing path to others (assigns 0 instead of crashing)
"""

"""
AMENDMENT 4C — phase8b_deberta.py: CRITICAL CRASH — token_type_ids KeyError

PROBLEM:
  The DisclosureDataset.__getitem__ deliberately does NOT include token_type_ids
  (with correct comment that DeBERTa-v3 doesn't support them).
  But the training loop tries to access it:
    ttype = batch['token_type_ids'].to(device)   ← KeyError — crashes immediately
  AND calls model with wrong signature:
    l_t, l_f = model(ids, mask, ttype, phi)      ← model.forward() takes 3 args
  This crash occurs on the FIRST training batch. No training results will be
  produced without this fix. This is the single most critical bug in the plan.

FIX — In the train_deberta() function, make ALL of these changes:

  # CHANGE 1: Remove from training loop
  # DELETE:   ttype = batch['token_type_ids'].to(device)

  # CHANGE 2: Fix model call in training loop
  # CHANGE FROM:
      with torch.cuda.amp.autocast():
          l_t, l_f = model(ids, mask, ttype, phi)
  # CHANGE TO:
      with torch.amp.autocast('cuda'):
          l_t, l_f = model(ids, mask, phi)

  # CHANGE 3: Fix model call in VALIDATION loop
  # CHANGE FROM:
      l_t, _ = model(
          batch['input_ids'].to(device),
          batch['attention_mask'].to(device),
          batch['token_type_ids'].to(device),   ← DELETE THIS LINE
          batch['phi_g'].to(device)
      )
  # CHANGE TO:
      l_t, _ = model(
          batch['input_ids'].to(device),
          batch['attention_mask'].to(device),
          batch['phi_g'].to(device)
      )

  # CHANGE 4: Fix deprecated GradScaler
  # CHANGE FROM:   scaler = torch.cuda.amp.GradScaler()
  # CHANGE TO:     scaler = torch.amp.GradScaler('cuda')
"""

"""
AMENDMENT 4D — phase8b_deberta.py: Missing per-sample prediction save
              (required by Phase 10 Wilcoxon test)

PROBLEM:
  Phase 10 Step 3 requires:
    results/deberta_kg_per_sample_correct.npy
    results/deberta_text_per_sample_correct.npy
  These files are mentioned in a comment in Phase 10 but the save code
  is absent from Phase 8B. Phase 10 will skip the Wilcoxon test silently.

FIX — Add this code block at the end of the test evaluation section in
       train_deberta(), AFTER computing test predictions:

  # Save per-sample correctness for Wilcoxon test in Phase 10
  correct_arr = np.array([
      1 if pred == true else 0
      for pred, true in zip(all_test_preds_type, all_test_true_type)
  ])
  save_name = f'results/deberta_{suffix}_per_sample_correct.npy'
  np.save(save_name, correct_arr)
  print(f"Saved per-sample predictions: {save_name}")

  # Also save full predictions for confusion matrix
  np.save(f'results/deberta_{suffix}_test_preds_type.npy',
          np.array(all_test_preds_type))
  np.save(f'results/deberta_{suffix}_test_true_type.npy',
          np.array(all_test_true_type))
"""

"""
AMENDMENT 4E — phase8b_deberta.py: RTX 2050 OOM prevention

PROBLEM:
  DeBERTa-v3-small with batch_size=16 and max_length=256 requires ~3.5GB VRAM
  on RTX 2050 (4GB). Mixed precision helps but gradient accumulation is needed
  for stability. The current code does not implement gradient accumulation.

FIX — Add gradient accumulation (effective_batch = 16 using accumulation of 2×8):

  # In train_deberta(), change:
  def train_deberta(include_kg=True, lr=2e-5, epochs=5,
                    batch_size=8,          # ← REDUCE from 16 to 8
                    patience=3,
                    grad_accum_steps=2):   # ← ADD: accumulate 2 steps = effective 16

    # In the training loop, change the optimizer step:
    for step, batch in enumerate(train_loader):
        ...
        with torch.amp.autocast('cuda'):
            l_t, l_f = model(ids, mask, phi)
            loss = (0.6*crit_t(l_t, y_t) + 0.4*crit_f(l_f, y_f)) / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

  # Also add gradient checkpointing to save ~40% VRAM:
  model.deberta.gradient_checkpointing_enable()
"""


# ════════════════════════════════════════════════════════════════════
# AMENDMENT 5: impl_part5_phase9_and_phase10.md
# ════════════════════════════════════════════════════════════════════
"""
AMENDMENT 5A — phase9_llm_baseline.py: LLM consistency test is missing

PROBLEM:
  The validation standard requires:
    "Prompt consistency test: 50 selected emails re-run 3× each, consistency ≥ 90%"
  But this code is not implemented anywhere in the file.
  This is listed as a research contribution. Without the code, it won't run.

FIX — Add this function to phase9_llm_baseline.py, after the main evaluation:

def run_consistency_test(test_df, n_samples=50, n_runs=3):
    '''Run the same 50 emails through inference 3 times and measure agreement.'''
    consistency_sample = test_df.sample(n=min(n_samples, len(test_df)),
                                         random_state=42)
    all_run_results = {i: [] for i in range(n_runs)}

    for run_idx in range(n_runs):
        print(f"Consistency run {run_idx + 1}/{n_runs}...")
        for _, row in tqdm(consistency_sample.iterrows(),
                           total=len(consistency_sample)):
            body_excerpt = str(row.get('body_clean',''))[:500]
            subject = str(row.get('subject',''))[:100]
            prompt = ZERO_SHOT.format(subject=subject, body=body_excerpt)
            response = run_inference(prompt, max_new_tokens=100)
            dt, fr, rk = parse_json_label(response)
            all_run_results[run_idx].append({'mid': row['mid'],
                                              'disclosure_type': dt,
                                              'framing': fr,
                                              'risk_tier': rk})

    # Compute consistency: % of emails where all 3 runs agree on disclosure_type
    from itertools import combinations
    agreement_count = 0
    for i in range(len(consistency_sample)):
        labels_across_runs = [all_run_results[r][i]['disclosure_type']
                               for r in range(n_runs)]
        if len(set(labels_across_runs)) == 1:  # all 3 agree
            agreement_count += 1
    consistency_score = agreement_count / len(consistency_sample)

    result = {
        'n_samples': len(consistency_sample),
        'n_runs': n_runs,
        'consistency_score': consistency_score,
        'interpretation': (
            'PASS' if consistency_score >= 0.90
            else f'LOW ({consistency_score:.1%}) — LLM labels are unstable'
        )
    }
    print(f"LLM Consistency: {consistency_score:.1%} [{result['interpretation']}]")
    return result

# Call it (add this at end of main loop):
consistency_result = run_consistency_test(test_df)
with open('results/llm_consistency.json', 'w') as f:
    json.dump(consistency_result, f, indent=2)
"""

"""
AMENDMENT 5B — phase9_llm_baseline.py: Few-shot examples should be from actual data

PROBLEM:
  The few-shot examples are fabricated strings:
    "The write-down of approximately $1.2 billion..."
  These are not from the actual Enron dataset. They are invented.
  A reviewer who cross-references against the dataset will notice.
  For reproducibility, few-shot examples must be real, documented instances.

FIX — Select and hardcode real examples from the training set AFTER labeling:

# Run this once after Phase 3 to get real few-shot examples:
gold = pd.read_parquet('data/labeled/emails_labeled_gold.parquet')
train_meta = pd.read_parquet('data/features/split_train.parquet')
train_gold = gold[gold['mid'].isin(train_meta['mid'])]

examples = []
for label in ['FINANCIAL', 'STRATEGIC', 'NONE']:
    subset = train_gold[
        (train_gold['disclosure_type'] == label) &
        (train_gold['word_count'].between(30, 100))   # short enough for prompt
    ]
    if len(subset) > 0:
        row = subset.iloc[0]
        examples.append({
            'mid': row['mid'],
            'body_excerpt': row['body_clean'][:300],
            'label': {
                'disclosure_type': row['disclosure_type'],
                'framing': row['framing'],
                'risk_tier': row['risk_tier']
            }
        })
        print(f"Few-shot example [{label}]: mid={row['mid']}")

# Save these examples for documentation in paper
with open('data/labeled/few_shot_examples.json', 'w') as f:
    json.dump(examples, f, indent=2)

# Use examples dict to build FEW_SHOT_EXAMPLES string dynamically
FEW_SHOT_EXAMPLES = '\n\n'.join([
    f"Example {i+1}:\nEmail: \"{e['body_excerpt']}\"\n"
    f"Labels: {json.dumps(e['label'])}"
    for i, e in enumerate(examples)
])
"""


# ════════════════════════════════════════════════════════════════════
# AMENDMENT 6: impl_part6_phase11_and_index.md
# ════════════════════════════════════════════════════════════════════
"""
AMENDMENT 6A — phase11_explanations.py: temperature + do_sample contradiction

PROBLEM:
  model.generate(..., temperature=0.1, do_sample=False, ...)
  temperature is silently ignored when do_sample=False.
  For explanation generation, deterministic (greedy) is too rigid —
  you want slightly varied but controlled explanations.

FIX — Choose one of these two options:

  Option A (deterministic, fully reproducible — better for paper):
    max_new_tokens=200, do_sample=False
    # Remove temperature entirely

  Option B (slightly creative, more natural explanations):
    max_new_tokens=200, do_sample=True, temperature=0.3, top_p=0.9

  RECOMMENDATION: Use Option A for the paper. Reproducibility is valued.
  Use Option B only if the deterministic explanations are too formulaic.
"""

"""
AMENDMENT 6B — phase11_explanations.py: BERTScore faithfulness metric is methodologically weak

PROBLEM:
  Using BERTScore with reference = KG context string:
    references = [r['kg_context'] + ' ' + r['centrality_context'] for r in results]
  BERTScore computes semantic similarity between two natural language texts.
  The KG context is structured key-value text ("Sender: Andrew Fastow | Role: CFO |...").
  Comparing a natural language paragraph against a key-value string produces
  scores that measure lexical key-word overlap, not semantic faithfulness.
  A reviewer will reject this metric as invalid.

FIX — Replace with dual-metric faithfulness:

def compute_entity_overlap(explanation, kg_context, centrality_context):
    '''
    Count how many KG entities are explicitly mentioned in the explanation.
    This is interpretable and directly measures grounding.
    '''
    # Extract entities: names, roles, disclosure types, period labels, keywords
    entity_patterns = [
        r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',       # Proper names (e.g., Andrew Fastow)
        r'\b(CEO|CFO|VP|Director|Analyst)\b',   # Roles
        r'\b(FINANCIAL|STRATEGIC|LEGAL|PII|RELATIONAL)\b',  # Disclosure types
        r'\b(HIGH|LOW|NONE)\b',                 # Risk tiers
        r'\b(crisis|bankruptcy|resignation|stable|pre-crisis)\b',  # Period labels
        r'\b(external|internal|confidential|unprotected)\b',  # Key attributes
    ]
    import re
    full_context = (kg_context + ' ' + centrality_context).lower()
    explanation_lower = explanation.lower()
    context_entities = set()
    for pat in entity_patterns:
        matches = re.findall(pat, full_context, re.IGNORECASE)
        context_entities.update([m.lower() if isinstance(m,str) else m[0].lower()
                                  for m in matches])
    if not context_entities:
        return 0.0
    matched = sum(1 for e in context_entities if e in explanation_lower)
    return matched / len(context_entities)

# Apply both metrics:
entity_scores = [
    compute_entity_overlap(r['explanation'], r['kg_context'], r['centrality_context'])
    for r in results
]

P, R, F1_bert = bertscore(
    hypotheses, references,
    model_type='roberta-base',   # ← Use roberta-base not distilbert for better semantics
    lang='en', verbose=False
)
bert_f1_scores = F1_bert.numpy().tolist()

for i, r in enumerate(results):
    r['bertscore_f1']       = bert_f1_scores[i]
    r['entity_overlap_score'] = entity_scores[i]

# Report BOTH metrics
print(f"BERTScore-F1 (semantic): {np.mean(bert_f1_scores):.4f}")
print(f"Entity Overlap (grounding): {np.mean(entity_scores):.4f}")

# Paper statement: "Our explanations achieve X.XX BERTScore-F1 and Y.YY
# entity overlap score, where entity overlap measures the fraction of KG
# entities (sender role, disclosure type, risk tier, temporal period) that
# are explicitly referenced in the generated explanation."
"""

"""
AMENDMENT 6C — Master Index: Missing Phase 4C dependency is undocumented

PROBLEM:
  The execution order shows "Phase 4C → [ADD phi_G after Phase 6]" but
  does not provide the actual implementation file for Phase 4C.
  phi_G cannot be merged into the feature matrices without this step.

FIX — Create phase4c_merge_phi.py with this content:

import numpy as np, pandas as pd
from scipy.sparse import load_npz, hstack, csr_matrix, save_npz

for split in ['train', 'val', 'test']:
    # Load existing TF-IDF features
    X_tfidf = load_npz(f'data/features/tfidf_{split}.npz')
    # Load phi_G
    phi_g   = np.load(f'data/features/phi_g_{split}.npy')
    # Concatenate: TF-IDF (sparse) + phi_G (dense → sparse)
    X_combined = hstack([X_tfidf, csr_matrix(phi_g)])
    save_npz(f'data/features/combined_{split}.npz', X_combined)
    print(f"{split}: TF-IDF {X_tfidf.shape} + phi_G {phi_g.shape} "
          f"→ combined {X_combined.shape}")

print("PHASE 4C COMPLETE — combined feature matrices saved")

# NOTE: Update Phase 7 to load combined_{split}.npz instead of tfidf_{split}.npz
# for the KG-augmented model variants. The text-only variants still use
# tfidf_{split}.npz. This is the ablation comparison.
"""


# ════════════════════════════════════════════════════════════════════
# TWO-TRACK EXECUTION PLAN
# ════════════════════════════════════════════════════════════════════
"""
╔══════════════════════════════════════════════════════════════════╗
║              TWO-TRACK EXECUTION PLAN                           ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  TRACK A — TOMORROW DEMO (single file, no GPU, ~4-5 hours)      ║
║  ──────────────────────────────────────────────────────────────  ║
║  Hour 1:   Download Enron CSV + run demo_pipeline.py STEP 1-2   ║
║  Hour 2:   STEP 3 (keyword labeling) + STEP 4 (TF-IDF split)   ║
║  Hour 2.5: STEP 5 (LR + RF models) — results in <10 minutes    ║
║  Hour 3-4: STEP 6 (centrality) — depends on corpus size        ║
║  Hour 4:   STEP 7-8 (plot + KG) — 30 minutes                   ║
║  Hour 4.5: STEP 9-10 (ablation + demo report)                   ║
║                                                                  ║
║  OUTPUT: All results in results/ + graphs/ — demo ready ✓       ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  TRACK B — RESEARCH PAPER (12 weeks, GPU required for some)     ║
║  ──────────────────────────────────────────────────────────────  ║
║  Week 1:  Apply Amendment 1A+1B → fix Phase 1 setup            ║
║  Week 1:  Apply Amendment 2A+2B → fix alias resolution + regex  ║
║  Week 2:  Apply Amendment 3A+3B → fix KG session bug + expand  ║
║  Week 2:  Run phase3a_autolabel.py (Mistral-7B Silver labels)   ║
║  Week 3:  Human annotation (Gold set — 800 emails, 2 annotators)║
║  Week 4:  Apply Amendment 3D → fix LabelEncoder               ║
║           Run phase3b_kappa.py → κ ≥ 0.60 gate                 ║
║  Week 5:  Run phase5a_kg_build.py (Neo4j, batched)              ║
║           Run phase6_centrality.py (applies Amendment 4A+4B)    ║
║  Week 6:  Run phase7_ml_models.py                               ║
║  Week 7:  Run phase8a_bilstm.py                                 ║
║  Weeks 8-9: Apply Amendment 4C+4D+4E → fix DeBERTa bugs        ║
║           Run phase8b_deberta.py (both text-only + KG-augmented)║
║  Week 10: Apply Amendment 5A+5B → fix LLM baseline              ║
║           Run phase9_llm_baseline.py (200 test emails, 3 prompts)║
║  Week 10: Run phase10_evaluation.py → ablation + Wilcoxon       ║
║  Week 11: Apply Amendment 6A+6B → fix explainability metrics    ║
║           Run phase11_explanations.py                            ║
║  Week 12: Write paper (methodology 2d, results 2d, discussion 1d║
║           polish 2d) → submit                                    ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""


# ════════════════════════════════════════════════════════════════════
# DEMO CHECKLIST — PRINT AND CHECK OFF BEFORE PRESENTING
# ════════════════════════════════════════════════════════════════════
"""
TRACK A — DEMO CHECKLIST
========================
Run: python demo_pipeline.py

After it completes, verify each item:

[ ] data/processed/emails_clean.parquet exists (>50MB)
[ ] data/labeled/emails_labeled.parquet exists
    → Check: pd.read_parquet('data/labeled/emails_labeled.parquet')['disclosure_type'].value_counts()
    → Expected: NONE dominates, FINANCIAL 5-20%, STRATEGIC 3-15%

[ ] results/comparison_table.csv exists
    → Should show 4 models: LR, RF, LR+KG, RF+KG with macro-F1 values
    → KG-augmented should be ≥ text-only

[ ] results/ablation_table.json exists
    → Shows Δ = KG_F1 - text_F1 per model

[ ] graphs/betweenness_trajectories.pdf exists (FIGURE 1 for paper)
    → Open and verify: crisis line visible at month 18
    → Top executives' curves should show visible changes post-crisis

[ ] graphs/centrality_matrix.parquet exists
    → pd.read_parquet('graphs/centrality_matrix.parquet').shape
    → Expected: (n_employees × 36, 7)

[ ] results/wilcoxon_centrality.json exists
    → Check p-value < 0.05 for publishable statistical claim

[ ] graphs/knowledge_graph.pkl exists
    → import pickle; KG = pickle.load(open('graphs/knowledge_graph.pkl','rb'))
    → KG.number_of_nodes() — should be > 100
    → KG.number_of_edges() — should be > 500

[ ] results/sample_predictions.json exists (for demo slides)
    → Shows real classified emails with true vs predicted labels

DEMO SLIDES — what to show:
  Slide 1: Project title + problem definition
  Slide 2: Dataset statistics (# emails, time range, label distribution)
  Slide 3: results/comparison_table.csv as a table
  Slide 4: results/ablation_table.json — "KG adds X% F1"
  Slide 5: graphs/betweenness_trajectories.png (the temporal plot)
  Slide 6: results/centrality_summary.json — top employees' centrality
  Slide 7: results/sample_predictions.json — 3 classified emails
  Slide 8: Next steps (Track B research paper)
"""