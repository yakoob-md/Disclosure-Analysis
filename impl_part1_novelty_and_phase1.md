# ORGDISCLOSE — Implementation Plan
# Part 1 of 6: Novelty Verdict + Phase 1 (Environment & Data Acquisition)

---

## FINAL NOVELTY VERDICT (READ ONCE, COMMIT)

### Is this STRONG or WEAK novelty?

**STRONG. Publishable at EMNLP Findings or IEEE TKDE if executed correctly.**

Here is exactly why, stated without hedging:

1. **New task definition.** No prior paper on Enron defines disclosure detection as a
   structured 4-dimensional multi-task prediction problem. All prior work is binary
   (spam/not-spam, fraud/not-fraud). You are redefining the problem itself.
   Task redefinition papers get accepted even with modest F1 gains.

2. **KG-computed labels.** You use the Knowledge Graph not as a feature input but as
   an annotation oracle — it computes the audience_scope label via role-hierarchy
   Cypher queries. This makes the KG causally integrated into your evaluation, not
   decorative. Reviewers cannot dismiss it.

3. **Temporal centrality derivatives.** First-order differences of monthly betweenness
   centrality as behavioral features is not in any published Enron paper. You can
   demonstrate statistically (Wilcoxon p < 0.001) that Δ_betweenness spikes precede
   HIGH-risk disclosure events. This is an empirical finding, not just a method.

4. **Offline reproducible explainability.** Using a locally-run quantized LLM +
   KG paths + BERTScore faithfulness check — fully reproducible with zero paid APIs.
   Reproducibility is now a formal criterion at ACL/EMNLP. Papers using GPT-4 APIs
   are increasingly penalized.

### What would make it WEAK (avoid these):
- If KG ablation shows < 1% F1 drop — your KG is decorative, paper rejected
- If annotation κ < 0.6 on disc_type — labels are noisy, results meaningless
- If you skip the temporal analysis section — you lose your most unique empirical finding
- If Mistral-7B explanations score < 0.60 BERTScore faithfulness — explainability fails

### Commit to this statement right now:
"My paper's core claim is: temporal graph topology is a necessary and measurable
component for structured organizational disclosure detection, and we prove this
via ablation (ΔF1 ≥ 2%) and temporal analysis (Wilcoxon p < 0.001)."
Everything you build must prove or support this claim.

---

## PHASE 1: ENVIRONMENT SETUP & DATA ACQUISITION

### Duration: 1–2 days
### Goal: Reproducible environment + raw Enron data loaded + smoke-tested

---

### 1.1 Exact Tasks

Task 1: Set up Python virtual environment (Python 3.10 recommended)
Task 2: Install all required libraries (pinned versions)
Task 3: Download CMU Enron SQL dataset
Task 4: Load into SQLite (lightweight, no PostgreSQL server needed)
Task 5: Verify row counts and schema
Task 6: Export base dataframe to parquet

---

### 1.2 Validation Standard (Must Pass Before Phase 2)

- [ ] `emails` table has between 490,000 and 520,000 rows
- [ ] Columns present: `mid`, `sender`, `date`, `subject`, `body`
- [ ] `recipients` table joinable to `emails` via `mid`
- [ ] No import errors on any installed library
- [ ] `emails_raw.parquet` written to disk, size > 200MB

If any check fails, do NOT proceed to Phase 2.

---

### 1.3 IMPLEMENTATION PROMPT — COPY THIS DIRECTLY TO GEMINI/CLAUDE

```
=== IMPLEMENTATION PROMPT: PHASE 1 — ENVIRONMENT & DATA SETUP ===

You are implementing Phase 1 of the OrgDisclose NLP research project.
This phase sets up the environment and loads the Enron email dataset.

CONTEXT:
- OS: Windows
- GPU: RTX 2050 (4GB VRAM)
- Python: 3.10
- Project folder: C:/Users/dabaa/OneDrive/Desktop/workplace_nlp/
- All output files go to: C:/Users/dabaa/OneDrive/Desktop/workplace_nlp/data/

TASK 1: Create requirements.txt with these EXACT pinned versions:
pandas==2.1.0
numpy==1.26.0
scikit-learn==1.3.2
torch==2.1.0
transformers==4.36.0
datasets==2.16.0
networkx==3.2.1
matplotlib==3.8.2
seaborn==0.13.0
pytz==2023.3
rapidfuzz==3.5.2
xgboost==2.0.2
lightgbm==4.1.0
imbalanced-learn==0.11.0
neo4j==5.14.0
bert-score==0.3.13
bitsandbytes==0.41.3
accelerate==0.25.0
peft==0.7.1
empath==0.89
scipy==1.11.4
tqdm==4.66.1

TASK 2: Create setup script `phase1_setup.py` that:
1. Creates directory structure:
   - workplace_nlp/data/raw/
   - workplace_nlp/data/processed/
   - workplace_nlp/data/labeled/
   - workplace_nlp/data/features/
   - workplace_nlp/models/
   - workplace_nlp/results/
   - workplace_nlp/graphs/
   - workplace_nlp/explanations/

2. Downloads the Enron email SQLite database from:
   https://www.cs.cmu.edu/~enron/enron_mail_20150507.tgz
   OR uses local file if already present at data/raw/enron.db

3. Connects to SQLite and runs these verification queries:
   - SELECT COUNT(*) FROM message  -- should be ~517,000
   - SELECT COUNT(*) FROM recipientinfo  -- should be ~2,000,000
   - Prints both counts

4. Exports to parquet:
   query = """
   SELECT 
       m.mid,
       m.sender,
       m.date,
       m.subject,
       m.body,
       GROUP_CONCAT(r.rvalue, ';') as recipients,
       GROUP_CONCAT(r.rtype, ';') as recipient_types
   FROM message m
   LEFT JOIN recipientinfo r ON m.mid = r.mid
   GROUP BY m.mid, m.sender, m.date, m.subject, m.body
   """
   Save result as: data/raw/emails_raw.parquet
   Print final row count and file size.

5. Print validation report:
   - Row count (must be 490k–520k)
   - Null rate per column
   - Date range (min and max)
   - Unique sender count

IMPORTANT NOTES:
- Use sqlite3 (built-in Python) not PostgreSQL
- If download fails, print exact manual download instructions
- All file paths should use os.path.join for Windows compatibility
- Add try/except with helpful error messages around each step
- Print "PHASE 1 COMPLETE" only if all validation checks pass

OUTPUT: phase1_setup.py (runnable script)
=== END OF PROMPT ===
```

---

### 1.4 Directory Structure After Phase 1

```
workplace_nlp/
├── data/
│   ├── raw/
│   │   └── emails_raw.parquet          ← output of phase 1
│   ├── processed/                      ← phase 2 output goes here
│   ├── labeled/                        ← phase 3 output goes here
│   └── features/                       ← phase 4 output goes here
├── models/                             ← trained model checkpoints
├── results/                            ← evaluation outputs
├── graphs/                             ← KG and centrality outputs
├── explanations/                       ← LLM explanation outputs
├── impl_part1_novelty_and_phase1.md    ← this file
└── requirements.txt
```

---

### 1.5 Known Issues & Fixes

ISSUE: Enron SQL dump from CMU may need maildir format parsing instead of SQL.
FIX: Use the Kaggle version "enron-email-dataset" (emails.csv, 517,401 rows) as fallback.
     kaggle datasets download -d wcukierski/enron-email-dataset

ISSUE: SQLite GROUP_CONCAT has 1MB limit on long email bodies.
FIX: Set `PRAGMA group_concat_max_len = 100000;` before the query.

ISSUE: Date column in Enron DB has inconsistent formats (RFC 2822, ISO, custom).
FIX: Use `pandas.to_datetime(errors='coerce')` — coerce invalid dates to NaT,
     then drop rows where date is NaT (< 2% of corpus).
```
