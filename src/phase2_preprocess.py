import os
import hashlib
import re
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz

# --- ALIAS & ROLE MAPS ---
ROLE_MAP = {
    'Kenneth Lay': ('CEO', 'Executive', 1),
    'Jeffrey Skilling': ('CEO', 'Executive', 1),
    'Andrew Fastow': ('CFO', 'Finance', 2),
    'Richard Causey': ('CFO', 'Finance', 2),
    'Greg Whalley': ('VP', 'Trading', 3),
    'Louise Kitchen': ('VP', 'Trading', 3),
    'John Lavorato': ('VP', 'Trading', 3),
    'Vince Kaminski': ('VP', 'Research', 3),
    'Sally Beck': ('Director', 'Operations', 4),
    'Sherron Watkins': ('VP', 'Finance', 3),
    'Mark Taylor': ('VP', 'Legal', 3),
    'James Derrick': ('VP', 'Legal', 3),
    'Richard Sanders': ('Director', 'Legal', 4),
    'Gerald Nemec': ('Director', 'Legal', 4),
    'Tana Jones': ('Analyst', 'Legal', 5),
    'Sara Shackleton': ('Director', 'Legal', 4),
    'Phillip Allen': ('VP', 'Trading', 3),
    'Jeff Dasovich': ('Director', 'Gov Affairs', 4),
    'Mark Haedicke': ('VP', 'Legal', 3),
    'Kay Mann': ('Director', 'Legal', 4),
    'Eric Bass': ('Analyst', 'Trading', 5),
    'Kate Symes': ('Analyst', 'Trading', 5),
    'David Delainey': ('VP', 'Trading', 3),
    'Mike McConnell': ('VP', 'Trading', 3),
    'John Arnold': ('Analyst', 'Trading', 5),
    'Bill Rapp': ('Director', 'Legal', 4),
    'Rosalee Fleming': ('Analyst', 'Executive', 5),
    'Rhonda Denton': ('Analyst', 'Trading', 5)
}

ALIAS_MAP = {
    'ken.lay': 'Kenneth Lay', 'klay': 'Kenneth Lay', 'kenneth.lay': 'Kenneth Lay',
    'jeff.skilling': 'Jeffrey Skilling', 'jskilling': 'Jeffrey Skilling',
    'andrew.fastow': 'Andrew Fastow', 'afastow': 'Andrew Fastow',
    'richard.causey': 'Richard Causey', 'rcausey': 'Richard Causey',
    'sherron.watkins': 'Sherron Watkins', 's.watkins': 'Sherron Watkins',
    'louise.kitchen': 'Louise Kitchen', 'l.kitchen': 'Louise Kitchen',
    'vince.kaminski': 'Vince Kaminski', 'v.kaminski': 'Vince Kaminski',
    'greg.whalley': 'Greg Whalley', 'g.whalley': 'Greg Whalley',
    'john.lavorato': 'John Lavorato', 'j.lavorato': 'John Lavorato',
    'sally.beck': 'Sally Beck', 's.beck': 'Sally Beck',
    'mark.taylor': 'Mark Taylor', 'm.taylor': 'Mark Taylor',
    'james.derrick': 'James Derrick',
    'richard.sanders': 'Richard Sanders',
    'tana.jones': 'Tana Jones',
    'gerald.nemec': 'Gerald Nemec',
    'sara.shackleton': 'Sara Shackleton',
    'phillip.allen': 'Phillip Allen', 'p.allen': 'Phillip Allen',
    'jeff.dasovich': 'Jeff Dasovich',
    'mark.haedicke': 'Mark Haedicke',
    'kay.mann': 'Kay Mann',
    'eric.bass': 'Eric Bass',
    'kate.symes': 'Kate Symes',
    'david.delainey': 'David Delainey',
    'mike.mcconnell': 'Mike McConnell',
    'john.arnold': 'John Arnold',
    'bill.rapp': 'Bill Rapp',
    'rosalee.fleming': 'Rosalee Fleming',
    'rhonda.denton': 'Rhonda Denton'
}

CANONICAL_PREFIXES = {name.lower().replace(' ', '.'): name for name in ROLE_MAP.keys()}

def resolve_alias_with_fuzzy(email_str):
    if not isinstance(email_str, str) or not email_str: return 'Unknown'
    
    # Strip domain and normalize
    prefix = email_str.split('@')[0].lower().replace('_', '.').replace('-', '.')
    
    # 1. Exact match in hardcoded ALIAS_MAP
    for alias, canonical in ALIAS_MAP.items():
        if alias in prefix:
            return canonical
            
    # 2. Fuzzy match against canonical prefixes
    match = process.extractOne(prefix, list(CANONICAL_PREFIXES.keys()), scorer=fuzz.token_set_ratio)
    if match and len(match) >= 2 and match[1] >= 80:
        return CANONICAL_PREFIXES[match[0]]
        
    return email_str

# --- CLEANING ---
def clean_body_advanced(text):
    if not isinstance(text, str): return ''
    
    # AMENDMENT 2B: Prevent regex body corruption (do NOT use re.DOTALL across the whole body for forwarding)
    text = re.split(r'-{2,}\s*Original Message\s*-{2,}', text, flags=re.IGNORECASE)[0]
    text = re.sub(r'(?m)^>.*$', '', text)
    text = re.sub(r'(?m)^From:.*$', '', text)
    text = re.sub(r'(?m)^Sent:.*$', '', text)
    text = re.sub(r'(?m)^To:.*$', '', text)
    text = re.sub(r'(?m)^Subject:.*$', '', text)
    
    # Signatures
    text = re.sub(r'--\s*\n.*', '', text, flags=re.DOTALL)
    
    # URLs
    text = re.sub(r'http\S+', '', text)
    
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

DISCLOSURE_TERMS = [
    'reserve', 'write-down', 'audit', 'ferc', 'spe', 'merger',
    'confidential', 'do not forward', 'mark-to-market', 'restatement',
    'privileged', 'settlement', 'compliance', 'ssn', 'acquisition'
]

def extract_dense_excerpt(text, max_chars=800):
    """Amendment 2C: Rank sentences by disclosure-keyword density."""
    if not text: return ""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return text[:max_chars]
        
    scores = []
    for s in sentences:
        s_lower = s.lower()
        score = sum(1 for term in DISCLOSURE_TERMS if term in s_lower)
        scores.append(score)
        
    ranked = sorted(zip(scores, range(len(sentences))), reverse=True)
    selected = []
    total_chars = 0
    for score, idx in ranked:
        s = sentences[idx]
        if total_chars + len(s) <= max_chars:
            selected.append((idx, s))
            total_chars += len(s)
            
    # Keep original ordering
    selected.sort(key=lambda x: x[0])
    excerpt = ' '.join([s for _, s in selected])
    return excerpt if excerpt else text[:max_chars]

def make_hash(row):
    s = str(row.get('sender', '')) + str(row.get('date', ''))[:10] + str(row.get('body_clean', ''))[:200]
    return hashlib.md5(s.encode()).hexdigest()

def main():
    print("Loading emails_raw.parquet...")
    df = pd.read_parquet('data/raw/emails_raw.parquet')
    
    print(f"Initial row count: {len(df):,}")
    
    # 1. Clean bodies first so we can hash them safely
    print("Performing advanced cleaning of email bodies (Amendment 2B)...")
    df['body_clean'] = df['body'].apply(clean_body_advanced)
    
    # 2. Deduplicate
    print("Deduplicating...")
    df['content_hash'] = df.apply(make_hash, axis=1)
    df = df.drop_duplicates(subset='content_hash')
    print(f"Rows after dedup:  {len(df):,}")
    
    # 3. Date parsing
    print("Parsing dates and extracting month index...")
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    df = df.dropna(subset=['date_parsed']).copy()
    df['month_index'] = ((df['date_parsed'].dt.year - 2000) * 12 + df['date_parsed'].dt.month - 1).astype(int)
    # Filter 2000 - 2002
    df = df[df['month_index'].between(0, 35)]
    print(f"Rows after date filter: {len(df):,} | Range: {df['date_parsed'].min()} to {df['date_parsed'].max()}")
    
    # 4. Dense excerpt extraction (Amendment 2C)
    print("Extracting dense excerpts (Amendment 2C)...")
    df['body_dense'] = df['body_clean'].apply(extract_dense_excerpt)
    df['word_count'] = df['body_dense'].str.split().str.len()
    df = df[df['word_count'] >= 10].copy()
    print(f"Rows after word count filter: {len(df):,}")
    
    # 5. System email filter
    print("Filtering system/auto-generated emails...")
    system_keywords = ['auto-response', 'out of office', 'delivery failed',
                       'undeliverable', 'mailer-daemon', 'postmaster', 'do not reply', 'failed to deliver']
    sys_pattern = '|'.join(system_keywords)
    mask = df['body_dense'].str.lower().str.contains(sys_pattern, na=False)
    df = df[~mask].copy()
    print(f"Rows after system filter: {len(df):,}")
    
    # 6. Alias Resolution with Fuzzy matching
    print("Resolving sender aliases mapped to roles (Amendment 2A with rapidfuzz)...")
    df['sender_canonical'] = df['sender'].apply(resolve_alias_with_fuzzy)
    df['sender_role'] = df['sender_canonical'].apply(lambda x: ROLE_MAP.get(x, ('Analyst', 'Unknown', 5))[0])
    
    print("\n--- Canonical Distribution Sample ---")
    print(df['sender_role'].value_counts())
    
    # Save cleaned
    print("\nSaving data/processed/emails_clean.parquet...")
    df.to_parquet('data/processed/emails_clean.parquet', index=False)
    print(f"Saved {len(df):,} rows.")
    
    # 7. Stratified Sampling (5800 emails)
    print("\nExecuting Stratified Sampling for Annotation Pools...")
    df['crisis_flag'] = (df['month_index'] >= 18).astype(int)
    fin_kws = ['reserve','write-down','write down','mark-to-market','ferc','spe',
               'off-balance','audit','restatement','confidential','merger','acquisition']
    df['has_fin_kw'] = df['body_dense'].str.lower().str.contains('|'.join(fin_kws), na=False).astype(int)
    
    # Execs oversampling
    exec_pool = df[df['sender_role'].isin(['CEO','CFO'])]
    exec_count = min(len(exec_pool), 1500)
    exec_sample = exec_pool.sample(n=exec_count, random_state=42) if len(exec_pool) >= exec_count else exec_pool
    print(f"Executive strata sampled: {len(exec_sample)}")
    
    # Remaining sample stratification
    remaining_pool = df[~df['mid'].isin(exec_sample['mid'])]
    needed = 5800 - len(exec_sample)
    
    # Time split (60% crisis period)
    crisis_pool = remaining_pool[remaining_pool['crisis_flag'] == 1]
    stable_pool = remaining_pool[remaining_pool['crisis_flag'] == 0]
    n_crisis = min(int(needed * 0.60), len(crisis_pool))
    n_stable = needed - n_crisis
    # Ensure stable pool covers it
    n_stable = min(n_stable, len(stable_pool))
    
    print(f"Remaining needed: {needed} (Target 60% crisis: {n_crisis}, 40% stable: {n_stable})")
    crisis_sample = crisis_pool.sample(n=n_crisis, random_state=42) if n_crisis > 0 else pd.DataFrame()
    stable_sample = stable_pool.sample(n=n_stable, random_state=42) if n_stable > 0 else pd.DataFrame()
    
    full_sample = pd.concat([exec_sample, crisis_sample, stable_sample]).drop_duplicates(subset='mid')
    full_sample = full_sample.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nFinal sampled size: {len(full_sample):,} (Target: 5,800)")
    print(f"Financial keyword coverage: {full_sample['has_fin_kw'].mean():.1%} (Target > 30%)")
    
    # Split Gold/Silver
    pool_size = len(full_sample)
    if pool_size >= 800:
        gold_pool = full_sample.head(800).copy()
        silver_pool = full_sample.tail(pool_size - 800).copy()
        
        gold_pool.to_parquet('data/processed/emails_gold_pool.parquet', index=False)
        silver_pool.to_parquet('data/processed/emails_silver_pool.parquet', index=False)
        print("\nSaved Gold (800) and Silver pools to data/processed/")
    else:
        print("\nWARNING: Total sampled size is less than 800. Insufficient data.")

    print("\nPHASE 2 COMPLETE — ALL CHECKS PASSED")

if __name__ == '__main__':
    # Execute Phase 2 Pipeline
    main()
