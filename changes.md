# ORGDISCLOSE — QUICK REFERENCE CARD
# Two-Track System: Tomorrow Demo vs Research Paper
# ===================================================

## TRACK A — TOMORROW (demo_pipeline.py)
## ========================================

### Setup (~15 min)
```bash
pip install pandas numpy scikit-learn networkx matplotlib seaborn pyarrow scipy tqdm rapidfuzz
kaggle datasets download -d wcukierski/enron-email-dataset -p data/raw/ --unzip
python demo_pipeline.py
```

### What you'll get (~4-5 hours total)
```
results/
  comparison_table.csv      ← ML model comparison (4 models)
  ablation_table.json       ← KG contribution delta
  ml_results.json           ← Full classification reports
  sample_predictions.json   ← 5 real classified emails
  centrality_summary.json   ← Top-10 employee centrality
  wilcoxon_centrality.json  ← Statistical significance test
  confusion_matrix_ml.pdf   ← Confusion matrix

graphs/
  betweenness_trajectories.pdf  ← FIGURE 1 (temporal plot)
  betweenness_trajectories.png  ← Same (for slides)
  centrality_matrix.parquet     ← Full centrality data
  knowledge_graph.pkl           ← NetworkX KG
```

---

## TRACK B — RESEARCH PAPER (amended phase files)
## =================================================

### CRITICAL BUGS TO FIX BEFORE RUNNING ANY TRACK B FILE

| File                  | Bug                              | Amendment | Priority |
|-----------------------|----------------------------------|-----------|----------|
| phase1_setup.py       | Wrong bitsandbytes version (Windows) | 1A   | HIGH     |
| phase1_setup.py       | Wrong data format (maildir vs SQL)   | 1B   | HIGH     |
| phase2_preprocess.py  | Alias resolution: only 10 employees  | 2A   | CRITICAL |
| phase2_preprocess.py  | Regex corrupts email bodies          | 2B   | HIGH     |
| phase3a_autolabel.py  | Body truncation loses key content    | 2C   | MEDIUM   |
| phase3b_kappa.py      | Add Krippendorff's alpha             | 2D   | LOW      |
| phase5a_kg_build.py   | One session per email (hangs)        | 3A   | CRITICAL |
| phase5a_kg_build.py   | Only 15 employees in KG              | 3B   | CRITICAL |
| phase5a_kg_build.py   | audience_scope defaults to constant  | 3C   | HIGH     |
| phase4a_tfidf.py      | LabelEncoder on unseen classes fails | 3D   | HIGH     |
| phase6_centrality.py  | Wrong delta_padded initialization    | 4A   | MEDIUM   |
| phase6_centrality.py  | Undirected closeness loses direction | 4B   | MEDIUM   |
| phase8b_deberta.py    | token_type_ids KeyError → CRASHES    | 4C   | CRITICAL |
| phase8b_deberta.py    | Missing per-sample prediction save   | 4D   | HIGH     |
| phase8b_deberta.py    | OOM on 4GB VRAM (need grad accum)    | 4E   | HIGH     |
| phase9_llm_baseline.py| Consistency test not implemented     | 5A   | HIGH     |
| phase9_llm_baseline.py| Few-shot examples are fabricated     | 5B   | MEDIUM   |
| phase11_explanations.py| temp + do_sample contradiction      | 6A   | LOW      |
| phase11_explanations.py| BERTScore metric is invalid          | 6B   | HIGH     |
| (missing file)        | phase4c_merge_phi.py doesn't exist   | 6C   | HIGH     |

### Apply fixes in this order (from amended_implementation.py):
```
1. Amendment 2A → fix alias resolution (prerequisite for everything)
2. Amendment 3B → expand KG to 151 employees (prerequisite for KG claims)
3. Amendment 4C → fix DeBERTa token_type_ids crash (prerequisite for DL results)
4. Amendment 3A → fix Neo4j session batching (prevents 25-minute hang)
5. Amendment 4D → add per-sample save (prerequisite for Wilcoxon in Phase 10)
6. Amendment 5A → implement consistency test (listed as research contribution)
7. Amendment 6B → fix faithfulness metric (prevents reviewer rejection)
8. Amendment 6C → create phase4c_merge_phi.py (missing file)
9. All others   → apply before running respective phases
```

---

## GPU STRATEGY (RTX 2050, 4GB VRAM)
## =====================================

### Models that fit on 4GB:
| Model                    | VRAM Usage | Batch Size | Est. Time |
|--------------------------|-----------|------------|-----------|
| TF-IDF + Logistic Reg    | 0 GB      | N/A        | 5 min     |
| TF-IDF + Random Forest   | 0 GB      | N/A        | 10 min    |
| Mistral-7B 4-bit (label) | ~3.8 GB   | 1          | 5h/5000   |
| DeBERTa-v3-small+KG      | ~3.5 GB   | 8+gradaccum| 3h/5ep    |
| Mistral-7B 4-bit (infer) | ~3.8 GB   | 1          | 13m/200   |
| BERTScore (roberta-base) | ~1.5 GB   | auto       | 5 min     |

### What to AVOID on 4GB:
- DeBERTa-v3-base (768 hidden × 12 layers) → use v3-SMALL
- Batch size > 8 for any transformer
- Loading two models at the same time (always del + gc.collect first)
- Full precision (fp32) transformer training → always use fp16/autocast

---

## PAPER-READY RESULT TARGETS
## =============================

| Metric                              | Target    | Where Produced      |
|-------------------------------------|-----------|---------------------|
| Cohen's κ — disclosure_type         | ≥ 0.60    | phase3b_kappa.py    |
| DeBERTa+KG macro-F1 (disc_type)     | ≥ 0.70    | phase8b_deberta.py  |
| KG ablation Δ (disc_type F1)        | ≥ +0.02   | phase10_evaluation.py|
| Wilcoxon p (best vs 2nd-best model) | < 0.05    | phase10_evaluation.py|
| Wilcoxon p (crisis > stable Δbetw.) | < 0.05    | phase6_centrality.py|
| BERTScore faithfulness              | ≥ 0.60    | phase11_explanations |
| Entity overlap grounding            | ≥ 0.70    | phase11_explanations |
| LLM consistency (3 runs)           | ≥ 90%     | phase9_llm_baseline  |

### If any target is missed, escalate:
- κ < 0.60 → Revise annotation guide, add 10 more examples per class, re-annotate
- KG Δ < 0.02 → Add PageRank + clustering coefficient to phi_G (6 more features)
- Wilcoxon p > 0.05 → Report as "trend" not "significant" — still publishable
- BERTScore < 0.60 → Report entity overlap only; note BERTScore limitation in paper
- LLM consistency < 90% → Report it honestly; this itself is a finding (LLMs are inconsistent)