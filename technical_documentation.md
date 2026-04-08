# Technical Documentation

## 1. Project Title

**Computational Analysis of Workplace Communication and Knowledge Graph Construction using the Enron Email Corpus**

## 2. Purpose

This project studies workplace communication patterns in the Enron Email Corpus and converts unstructured email data into structured analytical assets. The end goal is to build a communication knowledge graph and compare multiple machine learning, deep learning, and large language model baselines for organizational signal detection.

The project is designed for research use. The methodology prioritizes data quality, reproducibility, and interpretability before advanced modeling.

## 3. Research Objectives

The project aims to:

1. Parse raw Enron emails into structured fields.
2. Clean and normalize email content for downstream analysis.
3. Extract communication and linguistic features.
4. Build a weighted employee communication graph.
5. Compute graph-based organizational metrics.
6. Compare classical ML, deep learning, and zero-shot LLM baselines.
7. Identify communication patterns associated with work culture, performance, bias, and attrition-related behavior.

## 4. Data Source

### 4.1 Enron Email Corpus

The primary dataset is the Enron Email Corpus, stored at:

`data/raw/enron_emails.csv`

The corpus contains roughly:

- 500,000 emails
- 150 employees / major email actors
- Message metadata such as sender, recipients, timestamp, subject, and email body

### 4.2 Current Dataset Shape in This Project

The current raw CSV in the workspace has two columns:

- `file`
- `message`

The `message` column contains the full raw email text and is parsed into structured fields during preprocessing.

## 5. Repository Layout

```text
workplace-communication-kg-analysis/
├── data/
│   ├── raw/
│   │   └── enron_emails.csv
│   └── processed/
├── docs/
│   └── technical_documentation.md
├── src/
│   ├── preprocessing/
│   ├── models/
│   └── graph/
├── README.md
├── requirements.txt
└── .gitignore
```

## 6. Current Implementation Status

### 6.1 Completed

The project currently has a working Phase 1 preprocessing pipeline:

- `src/preprocessing/loader.py`
  - Loads the raw CSV dataset.
  - Resolves the project root reliably from notebook or script execution.

- `src/preprocessing/parser.py`
  - Parses RFC-style raw email messages.
  - Extracts `sender`, `recipients`, `date`, `subject`, and `body`.

- `src/preprocessing/cleaner.py`
  - Removes invalid rows with missing or unusable critical fields.
  - Produces cleaning statistics.

- `src/preprocessing/eda.py`
  - Computes communication statistics.
  - Extracts sender and recipient identities.
  - Produces basic counts and descriptive summaries.

- `src/preprocessing/phase1_eda.py`
  - Orchestrates the Phase 1 workflow.
  - Saves the cleaned dataset to `data/processed/enron_cleaned.csv`.

### 6.2 Output Produced So Far

The Phase 1 pipeline currently produces:

- Cleaned email dataset
- Unique sender and recipient counts
- Top senders and recipients by volume
- Basic descriptive communication statistics

### 6.3 Not Yet Implemented

The following components are planned but not yet implemented:

- Signature and quoted-reply removal
- Thread reconstruction
- POS tagging and NER pipeline
- Feature engineering module
- Graph construction module
- Graph analytics module
- Model training modules
- Evaluation and comparison scripts
- Main pipeline runner

## 7. Technical Architecture

The project follows a layered architecture:

1. **Data Engineering Layer**
   - Load raw CSV data.
   - Parse messages into structured email records.
   - Validate and clean malformed records.

2. **Semantic NLP Layer**
   - Normalize body text.
   - Remove signatures and reply chains.
   - Extract linguistic features such as tokens, POS tags, and named entities.

3. **Graph Engineering Layer**
   - Build a directed weighted communication graph.
   - Derive graph metrics and communities.

4. **Modeling Layer**
   - Train classical ML, DL, and zero-shot baselines.
   - Compare models on a consistent evaluation protocol.

5. **Insight Layer**
   - Translate model outputs into communication-pattern indicators.
   - Interpret organizational behavior from graph and text signals.

## 8. Phase 1: Data Engineering and Structural Parsing

This phase is already implemented and is the foundation for all later steps.

### 8.1 Data Ingestion

The raw CSV is loaded into pandas. Because notebook execution paths can vary, the loader resolves the project root dynamically rather than relying on a fixed relative path.

### 8.2 Message Parsing

Each raw message is parsed as an email message. The parser extracts:

- Sender
- Recipients
- Date
- Subject
- Body

This parsing step is essential because the knowledge graph depends on sender-recipient identity, while NLP modeling depends on the email body.

### 8.3 Data Quality Audit

The pipeline checks:

- Null values
- Parse failures
- Unknown senders
- Unknown recipients
- Empty subjects
- Empty bodies

This stage acts as a quality gate before downstream analysis.

### 8.4 Cleaning Rules

Rows are removed when critical fields are unusable. The current logic removes records with:

- `sender = UNKNOWN`
- `sender = PARSE_ERROR`
- `recipients = UNKNOWN`
- `recipients = PARSE_ERROR`
- empty body
- body parse error

The cleaned dataset is stored in `data/processed/enron_cleaned.csv`.

### 8.5 Descriptive Communication Statistics

The Phase 1 EDA computes:

- Unique senders
- Unique recipients
- Top senders by email count
- Top recipients by email count
- Average email length
- Average subject length

These statistics provide a first look at the communication network and help identify major actors.

## 9. Phase 2: Semantic NLP Pipeline

This phase is planned and will extend the cleaned dataset.

### 9.1 Body Normalization

The email body will be normalized by removing:

- Signature blocks
- Forwarded chains
- Quoted replies
- Boilerplate legal or disclaimer text
- Special characters and excess whitespace

### 9.2 Linguistic Analysis

The normalized text will be processed with spaCy for:

- Tokenization
- Part-of-speech tagging
- Named entity recognition

### 9.3 Embedding Extraction

BERT-based embeddings will be used to represent contextual semantics in the message body. These embeddings will later be used in supervised or zero-shot experiments.

## 10. Phase 3: Knowledge Graph Construction

The graph layer will model communication as a directed weighted network.

### 10.1 Graph Definition

- **Nodes**: employees or email actors
- **Edges**: sender-to-recipient email interactions
- **Edge weights**: frequency of communication

### 10.2 Graph Construction Rules

A directed edge will be created from a sender to a recipient whenever an email is observed. If multiple messages occur between the same pair, the edge weight will increase accordingly.

### 10.3 Graph Metrics

The project will compute:

- Degree centrality
- Betweenness centrality
- Community detection

These metrics are used to estimate communication prominence, brokerage position, and group structure.

## 11. Phase 4: Modeling and Evaluation

This phase compares multiple model families on communication-related prediction tasks.

### 11.1 Machine Learning Baselines

- Logistic Regression
- Support Vector Machine (SVM)

These models provide interpretable and strong classical baselines, especially for engineered or sparse features.

### 11.2 Deep Learning Models

- BiLSTM
- BERT

BiLSTM will be used as a sequence-aware baseline, while BERT will serve as the stronger contextual language model.

### 11.3 LLM Reference Baseline

A zero-shot intent classification approach will be used as a label-light baseline.

### 11.4 Evaluation Metrics

The project will report:

- Accuracy
- Precision
- Recall
- F1-score

For imbalanced tasks, macro-averaged and per-class metrics will be emphasized.

### 11.5 Experimental Comparisons

The final analysis should compare:

- Text-only features
- Graph-only features
- Hybrid graph + text features

This ablation design helps identify the contribution of each signal source.

## 12. Data Artifacts and File Conventions

### 12.1 Raw Data

- `data/raw/enron_emails.csv`

### 12.2 Processed Data

- `data/processed/enron_cleaned.csv`

Future processed artifacts should follow the same convention:

- `data/processed/<stage_name>.csv`
- `data/processed/<stage_name>.pkl`
- `data/processed/<stage_name>.json`

### 12.3 Code Modules

Future modules should remain organized as follows:

- `src/preprocessing/` for parsing, cleaning, NLP preprocessing, and feature extraction
- `src/graph/` for graph construction and network analytics
- `src/models/` for ML, DL, and LLM experiments

## 13. Reproducibility and Research Controls

To keep the project reproducible, the following practices should be maintained:

1. Keep the raw dataset unchanged.
2. Save intermediate artifacts in `data/processed/`.
3. Use explicit cleaning rules and document them in code.
4. Use time-aware data splits for modeling.
5. Fix random seeds where applicable.
6. Record package versions in `requirements.txt`.
7. Avoid training on data that leaks future information into past predictions.

## 14. Interpretation Guidelines

Results should be interpreted as computational indicators rather than causal proof.

For example:

- High betweenness centrality may indicate brokerage or coordination role.
- Frequent recipient status may indicate visibility or managerial load.
- Strong intent signals may reflect communication pressure, escalation, or coordination demand.

These are useful research signals, but they do not by themselves establish organizational causality.

## 15. Recommended Next Implementation Steps

The next technical milestones should be:

1. Add body-cleaning utilities for signatures and quoted text.
2. Implement a feature extraction module.
3. Create graph construction code in `src/graph/`.
4. Add a main pipeline runner.
5. Implement baseline ML training scripts.
6. Add deep learning and BERT pipelines.
7. Define evaluation and reporting utilities.

## 16. Status Summary

This project is currently in the **Phase 1 complete / Phase 2 preparation** stage.

Completed:

- Raw data ingestion
- Email parsing
- Data cleaning
- Initial communication EDA

Next:

- NLP normalization
- Feature engineering
- Graph construction
- Model training and evaluation

## 17. References

- Klimt, B., & Yang, Y. (2004). *The Enron Corpus: A New Dataset for Email Classification Research.*
- Diesner, J., & Carley, K. (2005). *Exploration of communication networks from the Enron email corpus.*
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*
- Hochreiter, S., & Schmidhuber, S. (1997). *Long Short-Term Memory.*

## 18. Maintenance Notes

This document should be updated as the project evolves. Recommended updates include:

- New modules and functions
- Changes in preprocessing rules
- Final graph schema
- Final model list and evaluation results
- Any changes to the output artifacts or directory structure
