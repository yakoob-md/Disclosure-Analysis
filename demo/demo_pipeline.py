"""
OrgDisclose — TRACK A: TOMORROW DEMO PIPELINE
=============================================
Single-file, end-to-end pipeline. Runs in 3-5 hours on RTX 2050 (4GB).

What this produces:
  1. Cleaned, keyword-labeled Enron email dataset
  2. TF-IDF + Logistic Regression baseline (results in ~5 minutes)
  3. TF-IDF + Random Forest (results in ~10 minutes)
  4. NetworkX-based Knowledge Graph (no Neo4j server needed)
  5. Degree, Betweenness, Closeness centrality for top employees
  6. Temporal betweenness trajectory plot (Jan 2000–Dec 2002)
  7. All results printed + saved to results/

HOW TO RUN:
  pip install pandas numpy scikit-learn networkx matplotlib seaborn
              pyarrow scipy tqdm empath
  python demo_pipeline.py

DATA SETUP (choose ONE):
  Option A (easiest):  Download from Kaggle
    kaggle datasets download -d wcukierski/enron-email-dataset
    unzip to: data/raw/emails.csv

  Option B: Use CMU maildir (already downloaded)
    Set USE_MAILDIR = True below and set MAILDIR_PATH

NOTE: No Neo4j, no Mistral-7B, no BERT needed for Track A.
      All of those are Track B (research paper).
"""

# ─────────────────────────────────────────────
# CONFIGURATION — EDIT THESE BEFORE RUNNING
# ─────────────────────────────────────────────
DATA_SOURCE       = 'kaggle_csv'   # 'kaggle_csv' | 'maildir'
KAGGLE_CSV_PATH   = 'data/raw/emails.csv'
MAILDIR_PATH      = 'data/raw/maildir'   # only if DATA_SOURCE='maildir'
OUTPUT_DIR        = 'results'
MAX_EMAILS        = 20000          # set None to use full corpus (slower)
RANDOM_STATE      = 42
CRISIS_MONTH      = 18             # month_index 18 = Jul 2001 (Skilling resignation)
# ─────────────────────────────────────────────

import os, re, hashlib, warnings, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # no display required
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from datetime import datetime
from tqdm import tqdm
from scipy.stats import wilcoxon
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, f1_score,
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import pickle

warnings.filterwarnings('ignore')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/labeled', exist_ok=True)
os.makedirs('data/features', exist_ok=True)
os.makedirs('graphs', exist_ok=True)

print("="*60)
print("OrgDisclose — Track A Demo Pipeline")
print("="*60)


# ══════════════════════════════════════════════════════════════
# STEP 1 — DATA LOADING
# ══════════════════════════════════════════════════════════════
print("\n[STEP 1] Loading Enron dataset...")

def load_kaggle_csv(path, max_rows=None):
    """Load the wcukierski Kaggle Enron CSV (emails.csv, 517k rows)."""
    df = pd.read_csv(path, nrows=max_rows)
    # Kaggle format: file, message columns
    # Parse headers from the 'message' field
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Parsing emails'):
        msg_text = str(row.get('message', ''))
        lines = msg_text.split('\n')
        headers = {}
        body_lines = []
        in_body = False
        for line in lines:
            if in_body:
                body_lines.append(line)
            elif line.strip() == '':
                in_body = True
            elif ':' in line and not in_body:
                key, _, val = line.partition(':')
                headers[key.strip().lower()] = val.strip()
        records.append({
            'file_path': str(row.get('file', '')),
            'sender':     headers.get('from', ''),
            'recipients': headers.get('to', ''),
            'date_str':   headers.get('date', ''),
            'subject':    headers.get('subject', ''),
            'body':       '\n'.join(body_lines).strip()
        })
    return pd.DataFrame(records)


def load_maildir(maildir_path, max_emails=None):
    """Load Enron maildir format."""
    import email as email_lib
    records = []
    count = 0
    for root, dirs, files in os.walk(maildir_path):
        for fname in files:
            if max_emails and count >= max_emails:
                break
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, 'r', encoding='latin-1') as f:
                    msg = email_lib.message_from_file(f)
                body = ''
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == 'text/plain':
                            body = part.get_payload(decode=False) or ''
                            break
                else:
                    body = msg.get_payload(decode=False) or ''
                records.append({
                    'file_path':  fpath,
                    'sender':     msg.get('From', ''),
                    'recipients': msg.get('To', ''),
                    'date_str':   msg.get('Date', ''),
                    'subject':    msg.get('Subject', ''),
                    'body':       str(body)
                })
                count += 1
            except Exception:
                continue
    return pd.DataFrame(records)


if DATA_SOURCE == 'kaggle_csv':
    if not os.path.exists(KAGGLE_CSV_PATH):
        print(f"ERROR: {KAGGLE_CSV_PATH} not found.")
        print("Download: kaggle datasets download -d wcukierski/enron-email-dataset")
        print("Then unzip the file and place emails.csv in data/raw/")
        raise FileNotFoundError(KAGGLE_CSV_PATH)
    df_raw = load_kaggle_csv(KAGGLE_CSV_PATH, max_rows=MAX_EMAILS)
else:
    df_raw = load_maildir(MAILDIR_PATH, max_emails=MAX_EMAILS)

df_raw['mid'] = range(len(df_raw))
print(f"Loaded {len(df_raw):,} emails")


# ══════════════════════════════════════════════════════════════
# STEP 2 — PREPROCESSING
# ══════════════════════════════════════════════════════════════
print("\n[STEP 2] Preprocessing...")

# --- Alias resolution (top 150 known Enron employees) ---
ALIAS_MAP = {
    'ken.lay':          'Kenneth Lay',       'klay':               'Kenneth Lay',
    'kenneth.lay':      'Kenneth Lay',
    'jeff.skilling':    'Jeffrey Skilling',  'jskilling':          'Jeffrey Skilling',
    'jeffrey.skilling': 'Jeffrey Skilling',  'j.skilling':         'Jeffrey Skilling',
    'andrew.fastow':    'Andrew Fastow',     'afastow':            'Andrew Fastow',
    'a.fastow':         'Andrew Fastow',
    'richard.causey':   'Richard Causey',    'rcausey':            'Richard Causey',
    'sherron.watkins':  'Sherron Watkins',   's.watkins':          'Sherron Watkins',
    'louise.kitchen':   'Louise Kitchen',    'l.kitchen':          'Louise Kitchen',
    'vince.kaminski':   'Vince Kaminski',    'v.kaminski':         'Vince Kaminski',
    'vincent.kaminski': 'Vince Kaminski',
    'greg.whalley':     'Greg Whalley',      'g.whalley':          'Greg Whalley',
    'john.lavorato':    'John Lavorato',     'j.lavorato':         'John Lavorato',
    'sally.beck':       'Sally Beck',        's.beck':             'Sally Beck',
    'mark.taylor':      'Mark Taylor',       'm.taylor':           'Mark Taylor',
    'james.derrick':    'James Derrick',
    'richard.sanders':  'Richard Sanders',
    'tana.jones':       'Tana Jones',
    'gerald.nemec':     'Gerald Nemec',
    'sara.shackleton':  'Sara Shackleton',
    'stinson.gibner':   'Stinson Gibner',
    'phillip.allen':    'Phillip Allen',     'p.allen':            'Phillip Allen',
    'jeff.dasovich':    'Jeff Dasovich',
    'mark.haedicke':    'Mark Haedicke',
    'kay.mann':         'Kay Mann',
    'eric.bass':        'Eric Bass',
    'kate.symes':       'Kate Symes',
    'chris.germany':    'Chris Germany',
    'martin.cuilla':    'Martin Cuilla',
    'rob.bradley':      'Rob Bradley',
    'pete.davis':       'Pete Davis',
    'scott.neal':       'Scott Neal',
    'david.delainey':   'David Delainey',
    'mike.mcconnell':   'Mike McConnell',
    'john.arnold':      'John Arnold',
    'bill.rapp':        'Bill Rapp',
    'rhonda.denton':    'Rhonda Denton',
    'rosalee.fleming':  'Rosalee Fleming',
}

ROLE_MAP = {
    'Kenneth Lay':    ('CEO',      'Executive',  1),
    'Jeffrey Skilling':('CEO',     'Executive',  1),
    'Andrew Fastow':  ('CFO',      'Finance',    2),
    'Richard Causey': ('CFO',      'Finance',    2),
    'Greg Whalley':   ('VP',       'Trading',    3),
    'Louise Kitchen': ('VP',       'Trading',    3),
    'John Lavorato':  ('VP',       'Trading',    3),
    'Vince Kaminski': ('VP',       'Research',   3),
    'Sally Beck':     ('Director', 'Operations', 4),
    'Sherron Watkins': ('VP',      'Finance',    3),
    'Mark Taylor':    ('VP',       'Legal',      3),
    'James Derrick':  ('VP',       'Legal',      3),
    'Richard Sanders':('Director', 'Legal',      4),
    'Gerald Nemec':   ('Director', 'Legal',      4),
    'Tana Jones':     ('Analyst',  'Legal',      5),
    'Sara Shackleton':('Director', 'Legal',      4),
    'Phillip Allen':  ('VP',       'Trading',    3),
    'Jeff Dasovich':  ('Director', 'Gov Affairs',4),
    'Mark Haedicke':  ('VP',       'Legal',      3),
    'Kay Mann':       ('Director', 'Legal',      4),
    'Eric Bass':      ('Analyst',  'Trading',    5),
    'Kate Symes':     ('Analyst',  'Trading',    5),
    'David Delainey': ('VP',       'Trading',    3),
    'Mike McConnell': ('VP',       'Trading',    3),
    'John Arnold':    ('Analyst',  'Trading',    5),
}

def resolve_alias(email_str):
    if not isinstance(email_str, str):
        return email_str, 'Unknown', 'Unknown', 5
    prefix = email_str.split('@')[0].lower().replace('_','.').replace('-','.')
    for alias, canonical in ALIAS_MAP.items():
        if alias in prefix:
            role_info = ROLE_MAP.get(canonical, ('Analyst', 'Unknown', 5))
            return canonical, role_info[0], role_info[1], role_info[2]
    return email_str, 'Analyst', 'Unknown', 5

def clean_body(text):
    if not isinstance(text, str): return ''
    # Remove forwarded blocks (split at the first separator)
    text = re.split(r'-{2,}\s*Original Message\s*-{2,}', text,
                    flags=re.IGNORECASE)[0]
    text = re.sub(r'(?m)^>+.*$', '', text)          # quoted reply lines
    text = re.sub(r'(?m)^From:.*$', '', text)        # forwarded from lines
    text = re.sub(r'--\s*\n.*', '', text, flags=re.DOTALL)  # signatures
    text = re.sub(r'http\S+', '', text)              # URLs
    text = re.sub(r'\S+@\S+', '', text)              # email addresses
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Parse dates
df_raw['date'] = pd.to_datetime(df_raw['date_str'], errors='coerce', utc=True)
df_raw = df_raw.dropna(subset=['date'])
base = pd.Timestamp('2000-01-01', tz='UTC')
df_raw['month_index'] = ((df_raw['date'].dt.year - 2000) * 12 +
                          df_raw['date'].dt.month - 1).astype(int)
df_raw = df_raw[df_raw['month_index'].between(0, 35)]

# Deduplication
def make_hash(row):
    s = str(row['sender']) + str(row['date'])[:10] + str(row['body'])[:150]
    return hashlib.md5(s.encode()).hexdigest()
df_raw['hash'] = df_raw.apply(make_hash, axis=1)
df = df_raw.drop_duplicates(subset='hash').copy()

# Clean bodies
df['body_clean'] = df['body'].apply(clean_body)
df['word_count'] = df['body_clean'].str.split().str.len()
df = df[df['word_count'] >= 8].copy()

# Filter system emails
SYS_KW = ['auto-response','out of office','delivery failed',
           'undeliverable','mailer-daemon','postmaster']
mask = df['body_clean'].str.lower().str.contains('|'.join(SYS_KW), na=False)
df = df[~mask].copy()

# Resolve aliases and add role
resolved = df['sender'].apply(resolve_alias)
df['sender_canonical'] = [r[0] for r in resolved]
df['sender_role']      = [r[1] for r in resolved]
df['sender_dept']      = [r[2] for r in resolved]
df['seniority_level']  = [r[3] for r in resolved]
df['is_external'] = df['recipients'].apply(
    lambda r: bool(r and '@' in str(r) and 'enron.com' not in str(r).lower())
)

df = df.reset_index(drop=True)
print(f"After cleaning: {len(df):,} emails | date range: "
      f"{df['date'].min().date()} to {df['date'].max().date()}")
df[['mid','sender','sender_canonical','sender_role','date','month_index',
    'subject','body_clean','word_count','is_external']].to_parquet(
    'data/processed/emails_clean.parquet', index=False)


# ══════════════════════════════════════════════════════════════
# STEP 3 — KEYWORD-BASED LABELING (fast, no GPU, no API)
# ══════════════════════════════════════════════════════════════
print("\n[STEP 3] Keyword-based labeling (no GPU required)...")

# Keyword dictionaries — ordered by priority (FINANCIAL > LEGAL > PII > STRATEGIC > RELATIONAL)
LABEL_KEYWORDS = {
    'FINANCIAL': [
        'mark-to-market', 'mark to market', 'write-down', 'write down',
        'write off', 'reserve', 'restatement', 'off-balance', 'spe ',
        'special purpose', 'earnings per share', 'ebitda', 'cash flow',
        'quarterly earnings', 'financial loss', 'financial exposure',
        'ferc', '10-k', 'annual report', 'audit committee',
        'accounting irregularity', 'fasb', 'gaap', 'pro forma',
        'asset write', 'loss recognition', 'stock option', 'equity stake',
        'hedge fund', 'credit rating', 'balance sheet', 'debt covenant',
        'raptors', 'ljm', 'chewco', 'jedi ', 'whitewing',   # Enron SPE names
    ],
    'LEGAL': [
        'attorney-client', 'privileged', 'legal counsel', 'subpoena',
        'sec investigation', 'doj', 'grand jury', 'litigation',
        'settlement agreement', 'class action', 'regulatory', 'compliance',
        'disclosure obligation', 'material non-public', 'insider trading',
        'securities fraud', 'bankruptcy filing', 'chapter 11',
    ],
    'PII': [
        'social security', 'ssn', 'date of birth', 'home address',
        'personal salary', 'medical record', 'health condition',
        'personal information', 'private information', 'passport',
        'driver license', 'credit card number',
    ],
    'STRATEGIC': [
        'merger', 'acquisition', 'do not share', 'confidential deal',
        'takeover', 'joint venture', 'term sheet', 'letter of intent',
        'due diligence', 'price strategy', 'competitive intelligence',
        'market share', 'strategic plan', 'pipeline deal', 'dynegy',
        'western power', 'broadband strategy',
    ],
    'RELATIONAL': [
        'between us', 'off the record', 'keep this private',
        'personal matter', 'just between', 'workplace conflict',
        'personnel issue', 'hr complaint', 'performance issue',
        'relationship with', 'rift with', 'personal grievance',
    ],
}

PROTECTION_KW = [
    'confidential', 'do not forward', 'not for distribution',
    'privileged and confidential', 'please keep', 'between us',
    'off the record', 'attorney-client privilege', 'do not share',
    'for your eyes only', 'private and confidential',
]

def label_email(body_clean, subject=''):
    text = (str(subject) + ' ' + str(body_clean)).lower()
    # Determine disclosure_type (priority order)
    disc_type = 'NONE'
    for label in ['FINANCIAL', 'LEGAL', 'PII', 'STRATEGIC', 'RELATIONAL']:
        if any(kw in text for kw in LABEL_KEYWORDS[label]):
            disc_type = label
            break
    # Determine framing
    if disc_type == 'NONE':
        framing = 'NA'
    elif any(kw in text for kw in PROTECTION_KW):
        framing = 'PROTECTED'
    else:
        framing = 'UNPROTECTED'
    return disc_type, framing

labels = df.apply(lambda r: label_email(r['body_clean'], r['subject']), axis=1)
df['disclosure_type'] = [l[0] for l in labels]
df['framing']         = [l[1] for l in labels]

# Rule-computed risk_tier
def compute_risk(row):
    if row['disclosure_type'] == 'NONE':
        return 'NONE'
    if row['framing'] == 'UNPROTECTED' or row['is_external']:
        return 'HIGH'
    return 'LOW'

df['risk_tier'] = df.apply(compute_risk, axis=1)

# Print distribution
print("\nLabel distribution:")
print(df['disclosure_type'].value_counts().to_string())
print(f"\nFraming: {df['framing'].value_counts().to_dict()}")
print(f"Risk:    {df['risk_tier'].value_counts().to_dict()}")

df.to_parquet('data/labeled/emails_labeled.parquet', index=False)
print(f"\nSaved {len(df):,} labeled emails")


# ══════════════════════════════════════════════════════════════
# STEP 4 — TEMPORAL SPLIT + TF-IDF FEATURES
# ══════════════════════════════════════════════════════════════
print("\n[STEP 4] Temporal split + TF-IDF features...")

# Temporal split: train on months 0-17, test on months 18-35
# This is the CORRECT split — train on pre-crisis, test on crisis period
# A random split would leak future information into training

train_df = df[df['month_index'] <= 17].copy()
val_df   = df[(df['month_index'] >= 18) & (df['month_index'] <= 19)].copy()
test_df  = df[df['month_index'] >= 20].copy()

# Balance the training set to avoid NONE-class dominance
# Oversample minority classes to at most 3× their natural frequency
from sklearn.utils import resample
def balance_dataset(df, label_col='disclosure_type', majority='NONE',
                    max_ratio=3):
    """Reduce majority class to max_ratio × (size of largest minority)."""
    minority_max = df[df[label_col] != majority].groupby(label_col).size().max()
    if pd.isna(minority_max) or minority_max == 0:
        return df
    cap = int(minority_max * max_ratio)
    majority_df  = df[df[label_col] == majority]
    minority_df  = df[df[label_col] != majority]
    if len(majority_df) > cap:
        majority_df = majority_df.sample(n=cap, random_state=RANDOM_STATE)
    return pd.concat([majority_df, minority_df]).sample(
        frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

train_balanced = balance_dataset(train_df)

print(f"Train: {len(train_balanced):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
print(f"Train label distribution (balanced):")
print(train_balanced['disclosure_type'].value_counts().to_string())

# TF-IDF vectorizer — fit on train only
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=10000,
    sublinear_tf=True,
    min_df=3,
    max_df=0.90,
    strip_accents='unicode',
    token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'
)
X_train = vectorizer.fit_transform(train_balanced['body_clean'].fillna(''))
X_val   = vectorizer.transform(val_df['body_clean'].fillna(''))
X_test  = vectorizer.transform(test_df['body_clean'].fillna(''))

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(train_balanced['disclosure_type'])
y_val   = le.transform(val_df['disclosure_type'])
y_test  = le.transform(test_df['disclosure_type'])

print(f"TF-IDF matrix: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
print(f"Classes: {le.classes_}")

# Save artifacts
from scipy.sparse import save_npz
save_npz('data/features/tfidf_train.npz', X_train)
save_npz('data/features/tfidf_test.npz',  X_test)
np.save('data/features/y_train.npy', y_train)
np.save('data/features/y_test.npy',  y_test)
with open('data/features/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('data/features/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
train_balanced[['mid','month_index','sender_canonical','sender_role']].to_parquet(
    'data/features/split_train.parquet')
test_df[['mid','month_index','sender_canonical','sender_role']].to_parquet(
    'data/features/split_test.parquet')


# ══════════════════════════════════════════════════════════════
# STEP 5 — ML MODELS (Logistic Regression + Random Forest)
# ══════════════════════════════════════════════════════════════
print("\n[STEP 5] Training ML models...")

# Compute class weights to handle imbalance
cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), cw))

# --- Model 1: Logistic Regression (fastest, interpretable) ---
print("  Training Logistic Regression...")
lr = LogisticRegression(
    C=1.0,
    max_iter=1000,
    class_weight='balanced',
    solver='lbfgs',
    multi_class='multinomial',
    random_state=RANDOM_STATE
)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
f1_lr = f1_score(y_test, y_pred_lr, average='macro')
print(f"  LR macro-F1: {f1_lr:.4f}")
print(classification_report(y_test, y_pred_lr,
                             target_names=le.classes_, zero_division=0))

# --- Model 2: Random Forest ---
print("  Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    class_weight='balanced',
    n_jobs=-1,
    random_state=RANDOM_STATE
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
f1_rf = f1_score(y_test, y_pred_rf, average='macro')
print(f"  RF macro-F1: {f1_rf:.4f}")
print(classification_report(y_test, y_pred_rf,
                             target_names=le.classes_, zero_division=0))

# Save models
with open('results/lr_model.pkl', 'wb') as f: pickle.dump(lr, f)
with open('results/rf_model.pkl', 'wb') as f: pickle.dump(rf, f)

# --- Confusion Matrix for best model ---
best_model_name = 'Logistic Regression' if f1_lr >= f1_rf else 'Random Forest'
best_preds      = y_pred_lr if f1_lr >= f1_rf else y_pred_rf
fig, ax = plt.subplots(figsize=(9, 7))
cm = confusion_matrix(y_test, best_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
ax.set_title(f'Confusion Matrix — {best_model_name}\n(macro-F1={max(f1_lr,f1_rf):.4f})',
             fontsize=13)
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('True', fontsize=11)
plt.tight_layout()
plt.savefig(f'results/confusion_matrix_ml.pdf', dpi=150)
plt.close()
print(f"  Saved confusion matrix → results/confusion_matrix_ml.pdf")

# ML results table
ml_results = {
    'Logistic Regression (TF-IDF)': {
        'macro_f1': f1_lr,
        'report': classification_report(y_test, y_pred_lr,
                  target_names=le.classes_, zero_division=0, output_dict=True)
    },
    'Random Forest (TF-IDF)': {
        'macro_f1': f1_rf,
        'report': classification_report(y_test, y_pred_rf,
                  target_names=le.classes_, zero_division=0, output_dict=True)
    }
}
with open('results/ml_results.json', 'w') as f:
    json.dump(ml_results, f, indent=2)


# ══════════════════════════════════════════════════════════════
# STEP 6 — NETWORK ANALYSIS & TEMPORAL CENTRALITY
# ══════════════════════════════════════════════════════════════
print("\n[STEP 6] Network analysis + temporal centrality...")

# Use the full clean corpus for graph construction (not just labeled emails)
# This gives more accurate centrality estimates
df_graph = df.copy()

all_senders = set(df_graph['sender_canonical'].dropna().unique())
print(f"  Unique senders: {len(all_senders)}")

# Build monthly directed graphs
monthly_graphs = {}
for month in tqdm(range(36), desc='  Building monthly graphs'):
    month_df = df_graph[df_graph['month_index'] == month][
        ['sender_canonical','recipients']].dropna(subset=['sender_canonical'])
    G = nx.DiGraph()
    G.add_nodes_from(all_senders)
    edge_counter = Counter()
    for _, row in month_df.iterrows():
        sender = row['sender_canonical']
        recips = str(row.get('recipients', ''))
        for recip in recips.split(';')[:10]:
            recip = recip.strip().lower()
            if recip and '@' in recip:
                # Canonicalize recipient
                prefix = recip.split('@')[0].replace('_','.').replace('-','.')
                canonical_r = recip
                for alias, name in ALIAS_MAP.items():
                    if alias in prefix:
                        canonical_r = name
                        break
                edge_counter[(sender, canonical_r)] += 1
    for (u, v), w in edge_counter.items():
        G.add_edge(u, v, weight=w)
    monthly_graphs[month] = G
    if month % 6 == 0:
        print(f"    Month {month}: {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges")

# Compute centrality per month
employees_list = sorted(all_senders)
n_emp = len(employees_list)
emp_idx = {e: i for i, e in enumerate(employees_list)}

# Shape: (n_employees, 36, 4) — [in_degree, out_degree, betweenness, closeness]
centrality_tensor = np.zeros((n_emp, 36, 4))

print("  Computing centrality metrics (betweenness uses k=50 approximation)...")
for month in tqdm(range(36), desc='  Centrality'):
    G = monthly_graphs[month]
    if G.number_of_edges() == 0:
        continue
    in_deg  = nx.in_degree_centrality(G)
    out_deg = nx.out_degree_centrality(G)
    # Betweenness approximation — k=50 is sufficient for demo
    k_approx = min(50, G.number_of_nodes())
    between = nx.betweenness_centrality(G, k=k_approx, normalized=True, seed=RANDOM_STATE)
    # Closeness on directed graph (wf_improved handles unreachable nodes)
    close = nx.closeness_centrality(G, wf_improved=True)

    for emp, idx in emp_idx.items():
        centrality_tensor[idx, month, 0] = in_deg.get(emp, 0)
        centrality_tensor[idx, month, 1] = out_deg.get(emp, 0)
        centrality_tensor[idx, month, 2] = between.get(emp, 0)
        centrality_tensor[idx, month, 3] = close.get(emp, 0)

# Z-score normalize per month (across employees)
centrality_norm = np.zeros_like(centrality_tensor)
for m in range(36):
    for metric in range(4):
        col = centrality_tensor[:, m, metric]
        std = col.std()
        if std > 1e-9:
            centrality_norm[:, m, metric] = (col - col.mean()) / std

# First-order betweenness derivative (Δ_betweenness)
delta_betweenness = np.diff(centrality_tensor[:, :, 2], axis=1)   # shape: (n_emp, 35)
delta_padded = np.hstack([np.zeros((n_emp, 1)), delta_betweenness])  # shape: (n_emp, 36)

# Save centrality matrix
records = []
for emp, idx in emp_idx.items():
    for month in range(36):
        records.append({
            'employee':          emp,
            'month_index':       month,
            'in_degree':         centrality_norm[idx, month, 0],
            'out_degree':        centrality_norm[idx, month, 1],
            'betweenness':       centrality_norm[idx, month, 2],
            'closeness':         centrality_norm[idx, month, 3],
            'delta_betweenness': delta_padded[idx, month],
            'raw_betweenness':   centrality_tensor[idx, month, 2],
        })
cm_df = pd.DataFrame(records)
cm_df.to_parquet('graphs/centrality_matrix.parquet', index=False)
print(f"  Saved centrality_matrix.parquet ({len(cm_df):,} rows)")

# Wilcoxon test: Is Δ_betweenness significantly larger in crisis vs stable?
def period_label(m):
    if m <= 11:  return 'stable_2000'
    if m <= 17:  return 'pre_crisis_2001'
    if m <= 23:  return 'acute_crisis'
    return      'post_crisis'

cm_df['period'] = cm_df['month_index'].apply(period_label)
stable_delta = cm_df[cm_df['period']=='stable_2000']['delta_betweenness'].values
crisis_delta = cm_df[cm_df['period']=='acute_crisis']['delta_betweenness'].values
n_min = min(len(stable_delta), len(crisis_delta))
try:
    stat, p_val = wilcoxon(np.abs(crisis_delta[:n_min]),
                           np.abs(stable_delta[:n_min]),
                           alternative='greater')
    print(f"\n  Wilcoxon test |Δ_betweenness| crisis > stable:")
    print(f"    stat={stat:.2f}, p={p_val:.4f} "
          f"{'[SIGNIFICANT ✓]' if p_val < 0.05 else '[NOT SIGNIFICANT]'}")
    wilcoxon_result = {'stat': float(stat), 'p_value': float(p_val),
                       'significant': bool(p_val < 0.05)}
except Exception as e:
    print(f"  Wilcoxon skipped: {e}")
    wilcoxon_result = {'stat': None, 'p_value': None, 'significant': None}

with open('results/wilcoxon_centrality.json', 'w') as f:
    json.dump(wilcoxon_result, f, indent=2)


# ══════════════════════════════════════════════════════════════
# STEP 7 — TEMPORAL BETWEENNESS PLOT (Figure 1 for paper)
# ══════════════════════════════════════════════════════════════
print("\n[STEP 7] Generating temporal betweenness trajectory plot...")

# Identify top-8 employees by average betweenness across all months
avg_betweenness = {emp: centrality_tensor[emp_idx[emp], :, 2].mean()
                   for emp in employees_list}
top_employees = sorted(avg_betweenness, key=avg_betweenness.get, reverse=True)[:8]

fig, ax = plt.subplots(figsize=(14, 6))
month_labels = []
for m in range(36):
    yr = 2000 + m // 12
    mo = m % 12 + 1
    month_labels.append(f"{yr}-{mo:02d}")

colors = plt.cm.tab10(np.linspace(0, 1, len(top_employees)))
for i, emp in enumerate(top_employees):
    idx = emp_idx[emp]
    values = centrality_tensor[idx, :, 2]
    role_info = ROLE_MAP.get(emp, ('?', '?', 5))
    label = f"{emp} ({role_info[0]})"
    ax.plot(range(36), values, label=label, color=colors[i],
            linewidth=1.8, alpha=0.85)

# Annotate crisis events
ax.axvline(x=18, color='red', linestyle='--', linewidth=1.5, alpha=0.8,
           label='Aug 2001: Skilling resignation')
ax.axvline(x=21, color='darkred', linestyle=':', linewidth=1.5, alpha=0.7,
           label='Nov 2001: Bankruptcy filing')
ax.axvspan(18, 35, alpha=0.05, color='red')

ax.set_xticks(range(0, 36, 3))
ax.set_xticklabels([month_labels[m] for m in range(0, 36, 3)],
                   rotation=45, ha='right', fontsize=8)
ax.set_xlabel('Month', fontsize=11)
ax.set_ylabel('Betweenness Centrality', fontsize=11)
ax.set_title('Temporal Betweenness Centrality — Top Enron Employees\n'
             '(Crisis period shaded in red)', fontsize=13)
ax.legend(loc='upper left', fontsize=8, ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('graphs/betweenness_trajectories.pdf', dpi=150)
plt.savefig('graphs/betweenness_trajectories.png', dpi=150)
plt.close()
print("  Saved: graphs/betweenness_trajectories.pdf + .png")


# ══════════════════════════════════════════════════════════════
# STEP 8 — KNOWLEDGE GRAPH (NetworkX-based, no Neo4j needed)
# ══════════════════════════════════════════════════════════════
print("\n[STEP 8] Building Knowledge Graph (NetworkX)...")

# The KG encodes: Employee→Email→Topic, with role hierarchy
# This is a lightweight KG — Neo4j version is Track B

KG = nx.DiGraph()

# Add Employee nodes with attributes
for emp_name, (role, dept, seniority) in ROLE_MAP.items():
    KG.add_node(emp_name,
                node_type='Employee',
                role=role,
                department=dept,
                seniority_level=seniority)

# Add disclosure topic nodes
for topic in ['FINANCIAL','LEGAL','PII','STRATEGIC','RELATIONAL','NONE']:
    KG.add_node(topic, node_type='DisclosureTopic')

# Add email sample nodes (top-500 labeled emails for KG demo)
df_labeled = df[df['disclosure_type'] != 'NONE'].head(500).copy()
email_kg_count = 0
for _, row in df_labeled.iterrows():
    if row['sender_canonical'] not in KG.nodes:
        KG.add_node(row['sender_canonical'], node_type='Employee',
                    role='Analyst', department='Unknown', seniority_level=5)
    mid_node = f"email_{row['mid']}"
    KG.add_node(mid_node,
                node_type='Email',
                disclosure_type=row['disclosure_type'],
                risk_tier=row['risk_tier'],
                framing=row['framing'],
                month_index=int(row['month_index']))
    # SENT edge
    KG.add_edge(row['sender_canonical'], mid_node, relation='SENT')
    # DISCLOSES edge (email → disclosure topic)
    KG.add_edge(mid_node, row['disclosure_type'], relation='DISCLOSES')
    # EXTERNAL edge if applicable
    if row['is_external']:
        ext_node = f"external_{row['mid']}"
        KG.add_node(ext_node, node_type='ExternalParty')
        KG.add_edge(mid_node, ext_node, relation='RECEIVED_BY_EXTERNAL')
    email_kg_count += 1

# REPORTS_TO hierarchy edges
REPORTS_TO = [
    ('Andrew Fastow',   'Kenneth Lay'),
    ('Jeffrey Skilling','Kenneth Lay'),
    ('Richard Causey',  'Andrew Fastow'),
    ('Greg Whalley',    'Jeffrey Skilling'),
    ('Louise Kitchen',  'Jeffrey Skilling'),
    ('John Lavorato',   'Greg Whalley'),
    ('Vince Kaminski',  'Greg Whalley'),
    ('Sally Beck',      'Louise Kitchen'),
    ('Sherron Watkins', 'Andrew Fastow'),
    ('Mark Taylor',     'Jeffrey Skilling'),
    ('James Derrick',   'Jeffrey Skilling'),
    ('Tana Jones',      'Mark Taylor'),
    ('Sara Shackleton', 'Mark Taylor'),
    ('Phillip Allen',   'Greg Whalley'),
    ('Jeff Dasovich',   'Jeffrey Skilling'),
    ('David Delainey',  'Greg Whalley'),
    ('Mike McConnell',  'Jeffrey Skilling'),
]
for (sub, mgr) in REPORTS_TO:
    if sub in KG.nodes and mgr in KG.nodes:
        KG.add_edge(sub, mgr, relation='REPORTS_TO')

# Compute audience_scope on labeled emails
print("  Computing audience_scope via KG hierarchy...")
def get_audience_scope(row):
    if row['is_external']:
        return 'EXTERNAL'
    # Check if any known employee in recipient list is more senior
    recips = str(row.get('recipients','')).lower()
    sender = row['sender_canonical']
    sender_seniority = ROLE_MAP.get(sender, ('?','?',5))[2]
    for emp, (role, dept, seniority) in ROLE_MAP.items():
        emp_prefix = ALIAS_MAP.get(emp.lower().replace(' ','.'), emp)
        if emp.lower() in recips:
            if seniority < sender_seniority:  # recipient is more senior
                return 'INTERNAL_UNAUTH'
    return 'INTERNAL_AUTH'

df['audience_scope'] = df.apply(get_audience_scope, axis=1)
df.to_parquet('data/labeled/emails_labeled.parquet', index=False)

# Save KG
import pickle
with open('graphs/knowledge_graph.pkl', 'wb') as f:
    pickle.dump(KG, f)

# KG statistics
n_types = Counter([data.get('node_type','?')
                   for _, data in KG.nodes(data=True)])
e_types = Counter([data.get('relation','?')
                   for _,_,data in KG.edges(data=True)])
print(f"  KG nodes: {KG.number_of_nodes():,} | edges: {KG.number_of_edges():,}")
print(f"  Node types: {dict(n_types)}")
print(f"  Edge types: {dict(e_types)}")


# ══════════════════════════════════════════════════════════════
# STEP 9 — PHI_G FEATURE VECTORS + AUGMENTED ML
# ══════════════════════════════════════════════════════════════
print("\n[STEP 9] Building phi_G feature vectors and augmented ML...")

from scipy.sparse import hstack, csr_matrix

def get_phi_g(sender_canonical, month_index):
    """Returns 8-dim graph feature vector for an email."""
    if sender_canonical not in emp_idx:
        return np.zeros(8, dtype=np.float32)
    idx = emp_idx[sender_canonical]
    m = max(0, min(35, int(month_index)))
    return np.array([
        centrality_norm[idx, m, 0],      # in_degree (normalized)
        centrality_norm[idx, m, 1],      # out_degree (normalized)
        centrality_norm[idx, m, 2],      # betweenness (normalized)
        centrality_norm[idx, m, 3],      # closeness (normalized)
        delta_padded[idx, m],            # Δ_betweenness
        centrality_tensor[idx, m, 2],   # raw betweenness
        centrality_tensor[idx, m, 0],   # raw in_degree
        float(m) / 35.0                  # time position (0=Jan2000, 1=Dec2002)
    ], dtype=np.float32)

# Build phi_G for each split
def build_phi_g_matrix(split_df):
    return np.array([
        get_phi_g(row['sender_canonical'], row['month_index'])
        for _, row in split_df.iterrows()
    ])

phi_g_train = build_phi_g_matrix(train_balanced)
phi_g_test  = build_phi_g_matrix(test_df)
np.save('data/features/phi_g_train.npy', phi_g_train)
np.save('data/features/phi_g_test.npy',  phi_g_test)

# Augmented feature matrix: TF-IDF + phi_G (sparse concatenation)
X_train_aug = hstack([X_train, csr_matrix(phi_g_train)])
X_test_aug  = hstack([X_test,  csr_matrix(phi_g_test)])

# Train augmented LR (text-only vs KG-augmented ablation)
print("  Training KG-augmented Logistic Regression...")
lr_aug = LogisticRegression(C=1.0, max_iter=1000,
                             class_weight='balanced',
                             solver='lbfgs', multi_class='multinomial',
                             random_state=RANDOM_STATE)
lr_aug.fit(X_train_aug, y_train)
y_pred_lr_aug = lr_aug.predict(X_test_aug)
f1_lr_aug = f1_score(y_test, y_pred_lr_aug, average='macro')
print(f"  LR+KG macro-F1: {f1_lr_aug:.4f}")

# Print augmented RF
print("  Training KG-augmented Random Forest...")
rf_aug = RandomForestClassifier(n_estimators=200, max_depth=20,
                                 class_weight='balanced', n_jobs=-1,
                                 random_state=RANDOM_STATE)
rf_aug.fit(X_train_aug, y_train)
y_pred_rf_aug = rf_aug.predict(X_test_aug)
f1_rf_aug = f1_score(y_test, y_pred_rf_aug, average='macro')
print(f"  RF+KG macro-F1: {f1_rf_aug:.4f}")

# KG ablation summary
ablation = {
    'Logistic Regression': {
        'text_only_f1': f1_lr, 'kg_augmented_f1': f1_lr_aug,
        'delta': f1_lr_aug - f1_lr,
        'verdict': 'KG HELPS' if (f1_lr_aug - f1_lr) >= 0.01 else 'MARGINAL'
    },
    'Random Forest': {
        'text_only_f1': f1_rf, 'kg_augmented_f1': f1_rf_aug,
        'delta': f1_rf_aug - f1_rf,
        'verdict': 'KG HELPS' if (f1_rf_aug - f1_rf) >= 0.01 else 'MARGINAL'
    }
}
print("\n  === KG ABLATION ===")
for model, res in ablation.items():
    print(f"  {model}: text={res['text_only_f1']:.4f} → "
          f"KG={res['kg_augmented_f1']:.4f} "
          f"(Δ={res['delta']:+.4f}) [{res['verdict']}]")

with open('results/ablation_table.json', 'w') as f:
    json.dump(ablation, f, indent=2)


# ══════════════════════════════════════════════════════════════
# STEP 10 — FINAL DEMO OUTPUT
# ══════════════════════════════════════════════════════════════
print("\n[STEP 10] Generating demo outputs...")

# --- 1. Sample classified emails (for demo presentation) ---
sample_emails = []
for disc_type in ['FINANCIAL','STRATEGIC','LEGAL','RELATIONAL','NONE']:
    subset = test_df[test_df['disclosure_type'] == disc_type]
    if len(subset) > 0:
        row = subset.iloc[0]
        pred_idx = lr.predict(vectorizer.transform([row['body_clean']]))[0]
        pred_label = le.inverse_transform([pred_idx])[0]
        sample_emails.append({
            'true_label':       disc_type,
            'predicted_label':  pred_label,
            'risk_tier':        row['risk_tier'],
            'sender':           row['sender_canonical'],
            'sender_role':      row['sender_role'],
            'month_index':      int(row['month_index']),
            'subject':          str(row['subject'])[:80],
            'body_excerpt':     str(row['body_clean'])[:200]
        })

with open('results/sample_predictions.json', 'w') as f:
    json.dump(sample_emails, f, indent=2)

# --- 2. Centrality summary for top-10 employees ---
centrality_summary = []
for emp in top_employees[:10]:
    idx = emp_idx[emp]
    avg_between = float(centrality_tensor[idx, :, 2].mean())
    crisis_between = float(centrality_tensor[idx, CRISIS_MONTH:CRISIS_MONTH+6, 2].mean())
    stable_between = float(centrality_tensor[idx, :12, 2].mean())
    role_info = ROLE_MAP.get(emp, ('Unknown', 'Unknown', 5))
    centrality_summary.append({
        'employee':              emp,
        'role':                  role_info[0],
        'avg_betweenness':       round(avg_between, 5),
        'stable_period_between': round(stable_between, 5),
        'crisis_period_between': round(crisis_between, 5),
        'crisis_delta_pct':      round((crisis_between - stable_between) /
                                       max(stable_between, 1e-9) * 100, 1)
    })

with open('results/centrality_summary.json', 'w') as f:
    json.dump(centrality_summary, f, indent=2)

# --- 3. Comparison table ---
comparison_rows = [
    {'Model':'LR (TF-IDF only)',  'KG':'No', 'macro_F1': round(f1_lr,4),     'Tier':'ML'},
    {'Model':'RF (TF-IDF only)',  'KG':'No', 'macro_F1': round(f1_rf,4),     'Tier':'ML'},
    {'Model':'LR (TF-IDF+KG)',   'KG':'Yes','macro_F1': round(f1_lr_aug,4),  'Tier':'ML+KG'},
    {'Model':'RF (TF-IDF+KG)',   'KG':'Yes','macro_F1': round(f1_rf_aug,4),  'Tier':'ML+KG'},
]
comp_df = pd.DataFrame(comparison_rows).sort_values('macro_F1', ascending=False)
comp_df.to_csv('results/comparison_table.csv', index=False)

# --- 4. Print the full demo report ---
print("\n" + "="*65)
print("  ORGDISCLOSE — DEMO RESULTS REPORT")
print("="*65)
print(f"\n  Dataset: {len(df):,} emails | Train: {len(train_balanced):,} | Test: {len(test_df):,}")
print(f"  Label distribution (test set):")
for cls in le.classes_:
    n = (test_df['disclosure_type'] == cls).sum()
    print(f"    {cls:15s}: {n:5d} ({n/len(test_df):.1%})")

print(f"\n  ── MODEL COMPARISON ──")
print(comp_df.to_string(index=False))

print(f"\n  ── KG ABLATION ──")
for model, res in ablation.items():
    print(f"  {model:30s}: Δ = {res['delta']:+.4f} [{res['verdict']}]")

print(f"\n  ── NETWORK ANALYSIS ──")
print(f"  Wilcoxon |Δ_betweenness| crisis > stable: "
      f"p = {wilcoxon_result.get('p_value','N/A')}")
print(f"  Top-3 employees by betweenness:")
for entry in centrality_summary[:3]:
    print(f"    {entry['employee']:25s} ({entry['role']:10s}) | "
          f"avg_betweenness={entry['avg_betweenness']:.5f} | "
          f"crisis Δ={entry['crisis_delta_pct']:+.1f}%")

print(f"\n  ── KG STATISTICS ──")
print(f"  Nodes: {KG.number_of_nodes():,} | Edges: {KG.number_of_edges():,}")
for ntype, count in n_types.items():
    print(f"    {ntype}: {count}")

print(f"\n  ── OUTPUT FILES ──")
outputs = [
    ('results/comparison_table.csv',         'Model comparison table'),
    ('results/ablation_table.json',           'KG ablation results'),
    ('results/ml_results.json',               'Full ML classification reports'),
    ('results/sample_predictions.json',       'Sample classified emails'),
    ('results/centrality_summary.json',       'Top-10 employee centrality'),
    ('results/wilcoxon_centrality.json',      'Statistical significance test'),
    ('results/confusion_matrix_ml.pdf',       'Confusion matrix (best ML model)'),
    ('graphs/betweenness_trajectories.pdf',   'Figure 1: Temporal betweenness'),
    ('graphs/centrality_matrix.parquet',      'Full centrality matrix'),
    ('graphs/knowledge_graph.pkl',            'NetworkX KG object'),
]
for path, desc in outputs:
    status = '✓' if os.path.exists(path) else '✗'
    print(f"  [{status}] {path:48s} — {desc}")

print("\n" + "="*65)
print("  TRACK A DEMO PIPELINE COMPLETE")
print("="*65)
print("""
NEXT STEPS (Track B — Research Paper):
  1. Run phase3a_autolabel.py   → Mistral-7B silver labels (requires GPU)
  2. Run phase3b_kappa.py       → Human annotation + inter-rater agreement
  3. Run phase5a_kg_build.py    → Neo4j KG (fix bugs noted in review)
  4. Run phase8b_deberta.py     → DeBERTa fine-tuning (fix token_type_ids bug)
  5. Run phase9_llm_baseline.py → 3-variant LLM evaluation
  6. Run phase10_evaluation.py  → Full ablation + Wilcoxon
  7. Run phase11_explanations.py→ KG-grounded explanations + BERTScore
See the amended implementation files for exact bug fixes before running.
""")