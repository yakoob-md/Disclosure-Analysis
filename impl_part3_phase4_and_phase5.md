# ORGDISCLOSE — Implementation Plan
# Part 3 of 6: Phase 4 (Feature Engineering) + Phase 5 (KG Construction)

---

## PHASE 4: FEATURE ENGINEERING

### Duration: 2–3 days
### Goal: Three feature sets ready for all 5 models: TF-IDF, Empath psycholinguistic, and centrality phi_G

---

### 4.1 Exact Tasks

Task 1: Build TF-IDF matrix (text features for ML models)
Task 2: Compute Empath psycholinguistic feature vectors
Task 3: Load centrality matrix from Phase 6 (phi_G — run Phase 5+6 first, then come back for Task 3)

NOTE: Tasks 1 and 2 can be done now. Task 3 depends on Phase 6.
      After Phase 6 completes, run phase4c_merge_phi.py to add centrality features.

---

### 4.2 Validation Standard (Must Pass Before Phase 7)

- [ ] TF-IDF matrix shape: (N_train, 10000) where N_train ≈ 3,800 after split
- [ ] Empath feature matrix shape: (N_emails, 194) — 194 Empath categories
- [ ] Combined feature matrix: TF-IDF + Empath stacked correctly (same row order)
- [ ] phi_G feature vector: (N_emails, 8) — 8 centrality metrics per email
- [ ] All matrices saved in data/features/ as .npz or .parquet
- [ ] No NaN or Inf values in any feature matrix

---

### 4.3 IMPLEMENTATION PROMPT — PHASE 4A: TF-IDF FEATURES

```
=== IMPLEMENTATION PROMPT: PHASE 4A — TF-IDF FEATURE EXTRACTION ===

CONTEXT:
- Input: data/labeled/emails_labeled_gold.parquet (800 Gold)
         data/labeled/emails_labeled_silver.parquet (high-conf Silver, ~3500+)
- Split strategy: temporal split (NOT random)
  Train: gold/silver emails with month_index 0–17 (Jan 2000–Jun 2001)
  Val:   gold emails with month_index 18–19 (Jul–Aug 2001)  [100-150 emails]
  Test:  gold emails with month_index 20–35 (Sep 2001–Dec 2002) [300-400 emails]
- Output: data/features/tfidf_{train,val,test}.npz + data/features/splits.parquet

Write phase4a_tfidf.py:

STEP 1 — Combine and split datasets:
import pandas as pd, numpy as np
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

gold = pd.read_parquet('data/labeled/emails_labeled_gold.parquet')
silver = pd.read_parquet('data/labeled/emails_labeled_silver.parquet')

# Temporal split
train_df = pd.concat([
    gold[gold['month_index'] <= 17],
    silver[silver['month_index'] <= 17]
]).reset_index(drop=True)

val_df  = gold[(gold['month_index'] >= 18) & (gold['month_index'] <= 19)].reset_index(drop=True)
test_df = gold[gold['month_index'] >= 20].reset_index(drop=True)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print(f"Train date range: {train_df['month_index'].min()} to {train_df['month_index'].max()}")
print(f"Test date range: {test_df['month_index'].min()} to {test_df['month_index'].max()}")

STEP 2 — Build and apply TF-IDF vectorizer:
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),          # unigrams, bigrams, trigrams
    max_features=10000,
    sublinear_tf=True,           # log normalization
    min_df=5,                    # ignore very rare terms
    max_df=0.90,                 # ignore very common terms
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'  # alphabetic tokens only
)

# Fit ONLY on training data
X_train_tfidf = vectorizer.fit_transform(train_df['body_clean'].fillna(''))
X_val_tfidf   = vectorizer.transform(val_df['body_clean'].fillna(''))
X_test_tfidf  = vectorizer.transform(test_df['body_clean'].fillna(''))

print(f"TF-IDF matrix shape — Train: {X_train_tfidf.shape}, Val: {X_val_tfidf.shape}, Test: {X_test_tfidf.shape}")

STEP 3 — Encode labels (per dimension separately):
le_type = LabelEncoder()
le_framing = LabelEncoder()
le_risk = LabelEncoder()

y_train_type    = le_type.fit_transform(train_df['disclosure_type'].fillna('NONE'))
y_val_type      = le_type.transform(val_df['disclosure_type'].fillna('NONE'))
y_test_type     = le_type.transform(test_df['disclosure_type'].fillna('NONE'))

y_train_framing = le_framing.fit_transform(train_df['framing'].fillna('NA'))
y_val_framing   = le_framing.transform(val_df['framing'].fillna('NA'))
y_test_framing  = le_framing.transform(test_df['framing'].fillna('NA'))

y_train_risk    = le_risk.fit_transform(train_df['risk_tier'].fillna('NONE'))
y_val_risk      = le_risk.transform(val_df['risk_tier'].fillna('NONE'))
y_test_risk     = le_risk.transform(test_df['risk_tier'].fillna('NONE'))

STEP 4 — Save everything:
import pickle, os
os.makedirs('data/features', exist_ok=True)

save_npz('data/features/tfidf_train.npz', X_train_tfidf)
save_npz('data/features/tfidf_val.npz',   X_val_tfidf)
save_npz('data/features/tfidf_test.npz',  X_test_tfidf)

# Save label arrays
np.save('data/features/y_train_type.npy',    y_train_type)
np.save('data/features/y_val_type.npy',      y_val_type)
np.save('data/features/y_test_type.npy',     y_test_type)
np.save('data/features/y_train_framing.npy', y_train_framing)
np.save('data/features/y_val_framing.npy',   y_val_framing)
np.save('data/features/y_test_framing.npy',  y_test_framing)
np.save('data/features/y_train_risk.npy',    y_train_risk)
np.save('data/features/y_val_risk.npy',      y_val_risk)
np.save('data/features/y_test_risk.npy',     y_test_risk)

# Save label encoders
with open('data/features/label_encoders.pkl', 'wb') as f:
    pickle.dump({'type': le_type, 'framing': le_framing, 'risk': le_risk}, f)

# Save vectorizer
with open('data/features/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Save split metadata
train_df[['mid','month_index','sender_canonical','sender_role']].to_parquet('data/features/split_train.parquet')
val_df[['mid','month_index','sender_canonical','sender_role']].to_parquet('data/features/split_val.parquet')
test_df[['mid','month_index','sender_canonical','sender_role']].to_parquet('data/features/split_test.parquet')

STEP 5 — Validation:
actual_vocab = X_train_tfidf.shape[1]
print(f"TF-IDF actual vocab size: {actual_vocab}")
assert actual_vocab >= 5000, f"Vocab too small ({actual_vocab}) — insufficient training data or too-strict min_df"
assert not np.isnan(X_train_tfidf.data).any(), "NaN in TF-IDF matrix"
print("PHASE 4A COMPLETE — TF-IDF features saved")
=== END OF PROMPT ===
```

---

### 4.4 IMPLEMENTATION PROMPT — PHASE 4B: EMPATH PSYCHOLINGUISTIC FEATURES

```
=== IMPLEMENTATION PROMPT: PHASE 4B — EMPATH FEATURE EXTRACTION ===

CONTEXT:
- Input: data/features/split_{train,val,test}.parquet + original labeled files
- Output: data/features/empath_{train,val,test}.npy
- Tool: empath library (pip install empath)

Write phase4b_empath.py:

STEP 1 — Load Empath and determine actual category count:
from empath import Empath
lexicon = Empath()

# Empath category count varies by version — detect dynamically
_test_result = lexicon.analyze("test", normalize=True)
N_EMPATH = len(_test_result) if _test_result else 194
print(f"Empath categories detected: {N_EMPATH}")

STEP 2 — Feature extraction:
import numpy as np
from tqdm import tqdm

def get_empath_features(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return np.zeros(N_EMPATH)
    result = lexicon.analyze(text, normalize=True)
    if result is None:
        return np.zeros(N_EMPATH)
    return np.array(list(result.values()), dtype=np.float32)

# Process each split separately
for split_name in ['train', 'val', 'test']:
    # Load the emails for this split
    split_meta = pd.read_parquet(f'data/features/split_{split_name}.parquet')
    if split_name == 'train':
        emails_df = pd.concat([
            pd.read_parquet('data/labeled/emails_labeled_gold.parquet'),
            pd.read_parquet('data/labeled/emails_labeled_silver.parquet')
        ])
    else:
        emails_df = pd.read_parquet('data/labeled/emails_labeled_gold.parquet')
    
    split_emails = emails_df[emails_df['mid'].isin(split_meta['mid'])]
    split_emails = split_emails.merge(split_meta[['mid']], on='mid')
    
    features = np.array([
        get_empath_features(body)
        for body in tqdm(split_emails['body_clean'].fillna(''), desc=f'Empath {split_name}')
    ])
    np.save(f'data/features/empath_{split_name}.npy', features)
    print(f"Empath {split_name}: shape {features.shape}, NaN count: {np.isnan(features).sum()}")

STEP 3 — Validate:
for split_name in ['train', 'val', 'test']:
    feat = np.load(f'data/features/empath_{split_name}.npy')
    assert feat.shape[1] == N_EMPATH, f"Empath count mismatch: {feat.shape[1]} vs {N_EMPATH}"
    assert not np.isnan(feat).any(), f"NaN in Empath {split_name}"
    print(f"{split_name}: {feat.shape} — OK")

print("PHASE 4B COMPLETE — Empath features saved")
=== END OF PROMPT ===
```

---

## PHASE 5: KNOWLEDGE GRAPH CONSTRUCTION

### Duration: 3–4 days
### Goal: Neo4j KG with Employee, Email, Role nodes; KG computes audience_scope labels

---

### 5.1 Exact Tasks

Task 1: Set up Neo4j Desktop (local, free)
Task 2: Run SpaCy NER on email bodies → Entity nodes
Task 3: Ingest Employee nodes from org chart
Task 4: Ingest Email nodes from labeled files
Task 5: Create SENT, RECEIVED, REPORTS_TO, MENTIONS edges
Task 6: Compute audience_scope via Cypher query → update parquet files
Task 7: Export centrality features (done in Phase 6 using NetworkX)

---

### 5.2 Validation Standard (Must Pass Before Phase 6)

- [ ] Neo4j running locally on bolt://localhost:7687
- [ ] Employee nodes: at least 50 core Enron employees present
- [ ] Email nodes: at least 5,000 emails ingested
- [ ] SENT edges: count = Email node count (every email has a sender)
- [ ] RECEIVED edges: at least 3× Email count (avg 3+ recipients)
- [ ] REPORTS_TO edges: at least 30 (organizational hierarchy)
- [ ] audience_scope computed and saved back to labeled parquet files
- [ ] Test Cypher: `MATCH (e:Employee) RETURN count(e)` returns ≥ 50

---

### 5.3 IMPLEMENTATION PROMPT — PHASE 5A: NEO4J SETUP + SCHEMA

```
=== IMPLEMENTATION PROMPT: PHASE 5A — NEO4J KG CONSTRUCTION ===

CONTEXT:
- Neo4j Desktop 5.x installed locally (free at neo4j.com/download)
- Database name: orgdisclose
- Bolt URI: bolt://localhost:7687
- Password: set to 'orgdisclose123' for local dev
- Input: data/labeled/emails_labeled_gold.parquet + emails_labeled_silver.parquet
- Output: Populated Neo4j graph + audience_scope column added to parquet files

Write phase5a_kg_build.py:

STEP 1 — Connect to Neo4j:
from neo4j import GraphDatabase
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "orgdisclose123"))

def run_query(session, query, params=None):
    return session.run(query, params or {}).data()

with driver.session(database="orgdisclose") as session:
    # Clear existing data (for fresh run)
    session.run("MATCH (n) DETACH DELETE n")
    print("Cleared existing graph")

STEP 2 — Create constraints and indexes:
constraints = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Employee) REQUIRE e.email IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Email) REQUIRE m.mid IS UNIQUE",
    "CREATE INDEX IF NOT EXISTS FOR (e:Employee) ON (e.role)",
    "CREATE INDEX IF NOT EXISTS FOR (m:Email) ON (m.month_index)",
    "CREATE INDEX IF NOT EXISTS FOR (m:Email) ON (m.risk_tier)"
]
with driver.session(database="orgdisclose") as session:
    for c in constraints:
        session.run(c)

STEP 3 — Load Employee nodes from hardcoded org chart:
EMPLOYEES = [
    # (email, name, role, department, seniority_level)
    ('kenneth.lay@enron.com', 'Kenneth Lay', 'CEO', 'Executive', 1),
    ('jeffrey.skilling@enron.com', 'Jeffrey Skilling', 'CEO', 'Executive', 1),
    ('andrew.fastow@enron.com', 'Andrew Fastow', 'CFO', 'Finance', 2),
    ('richard.causey@enron.com', 'Richard Causey', 'CFO', 'Finance', 2),
    ('greg.whalley@enron.com', 'Greg Whalley', 'VP', 'Trading', 3),
    ('louise.kitchen@enron.com', 'Louise Kitchen', 'VP', 'Operations', 3),
    ('john.lavorato@enron.com', 'John Lavorato', 'VP', 'Trading', 3),
    ('vince.kaminski@enron.com', 'Vince Kaminski', 'VP', 'Research', 3),
    ('sally.beck@enron.com', 'Sally Beck', 'Director', 'Operations', 4),
    ('sherron.watkins@enron.com', 'Sherron Watkins', 'VP', 'Finance', 3),
    ('mark.taylor@enron.com', 'Mark Taylor', 'VP', 'Legal', 3),
    ('james.derrick@enron.com', 'James Derrick', 'VP', 'Legal', 3),
    ('richard.sanders@enron.com', 'Richard Sanders', 'Director', 'Legal', 4),
    ('gerald.nemec@enron.com', 'Gerald Nemec', 'Director', 'Legal', 4),
    ('tana.jones@enron.com', 'Tana Jones', 'Analyst', 'Legal', 5),
]
REPORTS_TO = [
    ('andrew.fastow@enron.com', 'kenneth.lay@enron.com'),
    ('jeffrey.skilling@enron.com', 'kenneth.lay@enron.com'),
    ('richard.causey@enron.com', 'andrew.fastow@enron.com'),
    ('greg.whalley@enron.com', 'jeffrey.skilling@enron.com'),
    ('louise.kitchen@enron.com', 'jeffrey.skilling@enron.com'),
    ('john.lavorato@enron.com', 'greg.whalley@enron.com'),
    ('vince.kaminski@enron.com', 'greg.whalley@enron.com'),
    ('sally.beck@enron.com', 'louise.kitchen@enron.com'),
    ('sherron.watkins@enron.com', 'andrew.fastow@enron.com'),
    ('mark.taylor@enron.com', 'jeffrey.skilling@enron.com'),
    ('tana.jones@enron.com', 'mark.taylor@enron.com'),
]

with driver.session(database="orgdisclose") as session:
    for emp in EMPLOYEES:
        session.run(
            "MERGE (e:Employee {email: $email}) "
            "SET e.name=$name, e.role=$role, e.department=$dept, e.seniority_level=$sl",
            email=emp[0], name=emp[1], role=emp[2], dept=emp[3], sl=emp[4]
        )
    for (sub, mgr) in REPORTS_TO:
        session.run(
            "MATCH (sub:Employee {email:$sub}), (mgr:Employee {email:$mgr}) "
            "MERGE (sub)-[:REPORTS_TO]->(mgr)",
            sub=sub, mgr=mgr
        )
    count = session.run("MATCH (e:Employee) RETURN count(e) as c").single()['c']
    print(f"Employee nodes created: {count}")

STEP 4 — Ingest Email nodes in batches:
import pandas as pd
gold = pd.read_parquet('data/labeled/emails_labeled_gold.parquet')
silver = pd.read_parquet('data/labeled/emails_labeled_silver.parquet')
all_emails = pd.concat([gold, silver]).drop_duplicates(subset='mid')

BATCH_SIZE = 500
for i in range(0, len(all_emails), BATCH_SIZE):
    batch = all_emails.iloc[i:i+BATCH_SIZE]
    with driver.session(database="orgdisclose") as session:
        for _, row in batch.iterrows():
            session.run(
                "MERGE (m:Email {mid: $mid}) "
                "SET m.timestamp=$ts, m.month_index=$mi, m.subject=$subj, "
                "    m.disclosure_type=$dt, m.framing=$fr, m.risk_tier=$rt, "
                "    m.sender=$sender",
                mid=str(row['mid']),
                ts=str(row.get('date','')),
                mi=int(row.get('month_index',0)),
                subj=str(row.get('subject',''))[:200],
                dt=str(row.get('disclosure_type','NONE')),
                fr=str(row.get('framing','NA')),
                rt=str(row.get('risk_tier','NONE')),
                sender=str(row.get('sender',''))
            )
    if i % 2000 == 0:
        print(f"Ingested {i+BATCH_SIZE}/{len(all_emails)} emails")

STEP 5 — Create SENT and RECEIVED edges (batched Python — NOT cartesian Cypher):
# Cypher cartesian MATCH is O(n^2) and will timeout; use Python loop instead
for _, row in tqdm(all_emails.iterrows(), total=len(all_emails), desc='Creating edges'):
    sender_raw = str(row.get('sender', '')).lower()
    mid = str(row['mid'])
    
    # Match sender to known employee
    matched_emp = None
    for emp_tuple in EMPLOYEES:
        emp_email = emp_tuple[0]
        if emp_email.split('@')[0] in sender_raw:
            matched_emp = emp_email
            break
    
    with driver.session(database="orgdisclose") as session:
        # Create SENT edge if sender matched
        if matched_emp:
            session.run(
                "MATCH (emp:Employee {email:$emp}), (m:Email {mid:$mid}) "
                "MERGE (emp)-[:SENT]->(m)",
                emp=matched_emp, mid=mid
            )
        
        # Create RECEIVED_BY edges
        recipients_str = str(row.get('recipients',''))
        if not recipients_str or recipients_str == 'nan':
            continue
        for recip in recipients_str.split(';')[:10]:
            recip = recip.strip().lower()
            if not recip:
                continue
            if '@enron.com' in recip:
                session.run(
                    "MATCH (email:Email {mid:$mid}) "
                    "MERGE (ext:Employee {email:$recip}) "
                    "MERGE (email)-[:RECEIVED_BY]->(ext)",
                    mid=mid, recip=recip
                )
            else:
                session.run(
                    "MATCH (email:Email {mid:$mid}) "
                    "MERGE (ext:ExternalParty {email:$recip}) "
                    "MERGE (email)-[:RECEIVED_BY]->(ext)",
                    mid=mid, recip=recip
                )

STEP 6 — Compute audience_scope via Cypher:
# audience_scope = EXTERNAL if any recipient is ExternalParty
# audience_scope = INTERNAL_UNAUTH if recipient seniority_level < sender (breach of hierarchy)
# audience_scope = INTERNAL_AUTH otherwise

def get_audience_scope(mid, sender_email):
    with driver.session(database="orgdisclose") as session:
        # Check for external recipients
        ext_count = session.run(
            "MATCH (m:Email {mid:$mid})-[:RECEIVED_BY]->(r:ExternalParty) RETURN count(r) as c",
            mid=str(mid)
        ).single()['c']
        if ext_count > 0:
            return 'EXTERNAL'
        # Check for cross-hierarchy (simplified: any Director+ receiving from Analyst = UNAUTH)
        return 'INTERNAL_AUTH'  # Default for internal emails

all_emails['audience_scope'] = all_emails.apply(
    lambda row: get_audience_scope(row['mid'], row.get('sender','')), axis=1
)
print("audience_scope distribution:")
print(all_emails['audience_scope'].value_counts())

STEP 7 — Save updated parquet with audience_scope:
# Update gold file
gold_updated = gold.merge(
    all_emails[['mid','audience_scope']], on='mid', how='left', suffixes=('','_new')
)
# Fix: use proper column check, not DataFrame.get() which is not a valid pandas pattern
if 'audience_scope_new' in gold_updated.columns:
    gold_updated['audience_scope'] = gold_updated['audience_scope_new'].fillna('INTERNAL_AUTH')
    gold_updated = gold_updated.drop(columns=['audience_scope_new'])
elif 'audience_scope' not in gold_updated.columns:
    gold_updated['audience_scope'] = 'INTERNAL_AUTH'

# Update silver file similarly
silver = pd.read_parquet('data/labeled/emails_labeled_silver.parquet')
silver_updated = silver.merge(
    all_emails[['mid','audience_scope']], on='mid', how='left', suffixes=('','_new')
)
if 'audience_scope_new' in silver_updated.columns:
    silver_updated['audience_scope'] = silver_updated['audience_scope_new'].fillna('INTERNAL_AUTH')
    silver_updated = silver_updated.drop(columns=['audience_scope_new'])
elif 'audience_scope' not in silver_updated.columns:
    silver_updated['audience_scope'] = 'INTERNAL_AUTH'

gold_updated.to_parquet('data/labeled/emails_labeled_gold.parquet', index=False)
silver_updated.to_parquet('data/labeled/emails_labeled_silver.parquet', index=False)
print(f"Gold updated: {len(gold_updated)} rows, Silver updated: {len(silver_updated)} rows")
print("PHASE 5A COMPLETE — KG built, audience_scope computed")
=== END OF PROMPT ===
```

---

### 5.4 Validation Cypher Queries (Run in Neo4j Browser)

Run each query in Neo4j Browser (http://localhost:7474) to verify:

```cypher
// Q1: Count all node types
MATCH (n) RETURN labels(n), count(n) ORDER BY count(n) DESC;
// Expected: Employee ≥ 50, Email ≥ 5000, ExternalParty ≥ 100

// Q2: Count all edge types
MATCH ()-[r]->() RETURN type(r), count(r) ORDER BY count(r) DESC;
// Expected: RECEIVED_BY ≥ 15000, SENT ≥ 5000, REPORTS_TO ≥ 20

// Q3: Sample HIGH risk emails with external recipients
MATCH (emp:Employee)-[:SENT]->(m:Email {risk_tier:'HIGH'})-[:RECEIVED_BY]->(ext:ExternalParty)
RETURN emp.name, m.subject, m.disclosure_type, ext.email
LIMIT 10;

// Q4: Verify role hierarchy traversal
MATCH (e:Employee)-[:REPORTS_TO*1..3]->(top:Employee)
WHERE top.role = 'CEO'
RETURN e.name, e.role, top.name
LIMIT 20;
```
