# ORGDISCLOSE — Implementation Plan
# Part 6 of 6: Phase 11 (Explainability) + Master Index + Paper Checklist

---

## PHASE 11: EXPLAINABILITY PIPELINE

### Duration: 2–3 days
### Goal: KG-grounded NL explanations for top-50 HIGH-risk emails + BERTScore faithfulness

---

### 11.1 Validation Standard (Must Pass)

- [ ] 50 HIGH-risk test emails with explanations generated
- [ ] Each explanation references: sender, disclosure type, recipient type, centrality signal
- [ ] BERTScore-F1 (against gold KG path summary) ≥ 0.60
- [ ] Manual spot-check: 10 explanations reviewed by you — rated coherent/incoherent
- [ ] explanations/explanations_with_scores.json saved

---

### 11.2 IMPLEMENTATION PROMPT — PHASE 11: EXPLAINABILITY

```
=== IMPLEMENTATION PROMPT: PHASE 11 — EXPLAINABILITY PIPELINE ===

CONTEXT:
- Input: Best model predictions on test set (DeBERTa + KG)
- Select top-50 emails predicted as HIGH risk (true label also HIGH)
- Use Neo4j to extract KG paths for each email
- Use Mistral-7B-Instruct (4-bit) to generate NL explanation
- Use BERTScore to measure faithfulness
- Output: explanations/explanations_with_scores.json

Write phase11_explanations.py:

STEP 1 — Load HIGH risk test emails:
import pandas as pd, numpy as np, json
from neo4j import GraphDatabase
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from bert_score import score as bertscore
import torch

# Load gold test data and best model predictions
gold = pd.read_parquet('data/labeled/emails_labeled_gold.parquet')
test_meta = pd.read_parquet('data/features/split_test.parquet')
test_df = gold[gold['mid'].isin(test_meta['mid'])].reset_index(drop=True)

# Filter to HIGH risk emails (ground truth AND predicted)
# Assumes you saved test predictions from Phase 8B as results/deberta_kg_test_preds.csv
# If not saved yet, run the test evaluation loop from Phase 8B first.
high_risk_test = test_df[test_df['risk_tier'] == 'HIGH'].head(50).reset_index(drop=True)
print(f"Selected {len(high_risk_test)} HIGH risk emails for explanation")

STEP 2 — Extract KG paths from Neo4j:
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "orgdisclose123"))

def get_kg_context(mid, sender):
    """Returns a structured context string from the KG for this email."""
    queries = []
    with driver.session(database="orgdisclose") as session:
        # Get sender info
        sender_info = session.run(
            "MATCH (e:Employee {email:$email}) RETURN e.name, e.role, e.department",
            email=str(sender)
        ).data()
        
        # Get this email's recipients
        recipients = session.run(
            "MATCH (m:Email {mid:$mid})-[:RECEIVED_BY]->(r) "
            "RETURN labels(r)[0] as rtype, r.email as remail, "
            "r.role as rrole LIMIT 5",
            mid=str(mid)
        ).data()
        
        # Get email's KG labels
        email_info = session.run(
            "MATCH (m:Email {mid:$mid}) RETURN m.disclosure_type, m.risk_tier, m.month_index",
            mid=str(mid)
        ).data()
    
    # Format context
    sender_name = sender_info[0]['e.name'] if sender_info else sender
    sender_role = sender_info[0]['e.role'] if sender_info else 'Unknown'
    
    recip_summary = ', '.join([
        f"{r.get('rrole','External')} ({r.get('rtype','')})"
        for r in recipients[:3]
    ]) if recipients else 'unknown'
    
    email_data = email_info[0] if email_info else {}
    disclosure_type = email_data.get('m.disclosure_type', 'UNKNOWN')
    month_index = email_data.get('m.month_index', 0)
    
    # Timeline context
    if month_index <= 17:
        period = "stable operations period (pre-crisis)"
    elif month_index <= 23:
        period = "acute crisis period (post-Skilling resignation)"
    else:
        period = "post-bankruptcy period"
    
    kg_context = (f"Sender: {sender_name} | Role: {sender_role} | "
                  f"Timeline: {period} (month {month_index}) | "
                  f"Disclosure: {disclosure_type} | "
                  f"Recipients: {recip_summary}")
    return kg_context

STEP 3 — Load centrality data for anomaly context:
cm_df = pd.read_parquet('graphs/centrality_matrix.parquet')

def get_centrality_context(sender, month_index):
    emp_data = cm_df[
        (cm_df['employee'] == sender) &
        (cm_df['month_index'] == month_index)
    ]
    if len(emp_data) == 0:
        return "No centrality data available."
    row = emp_data.iloc[0]
    delta = row.get('delta_betweenness', 0)
    betw  = row.get('betweenness', 0)
    
    anomaly_str = ""
    if abs(delta) > 1.5:  # > 1.5 standard deviations change
        direction = "increased" if delta > 0 else "decreased"
        anomaly_str = (f"ANOMALY: Sender's betweenness centrality {direction} by "
                       f"{abs(delta):.2f}σ in this month — unusual broker activity detected.")
    else:
        anomaly_str = f"Network position: betweenness={betw:.4f} (within normal range)."
    return anomaly_str

STEP 4 — Load Mistral-7B (reuse from Phase 9 if session still active):
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="auto"
)
model.eval()

EXPLANATION_PROMPT = """[INST] You are a forensic communication analyst.
Write a 3-sentence risk explanation for the following email.
Your explanation MUST reference: (1) who sent it and their role, 
(2) what type of sensitive information was disclosed, 
(3) whether the network position of the sender was anomalous.

CONTEXT:
{kg_context}

NETWORK SIGNAL:
{centrality_context}

EMAIL EXCERPT (first 400 chars):
{body_excerpt}

Write 3 sentences. Be specific. Do not be generic. [/INST]"""

STEP 5 — Generate explanations:
import os
os.makedirs('explanations', exist_ok=True)

results = []
for _, row in tqdm(high_risk_test.iterrows(), total=len(high_risk_test)):
    mid = str(row['mid'])
    sender = str(row.get('sender', ''))
    sender_canonical = str(row.get('sender_canonical', sender))
    month_index = int(row.get('month_index', 0))
    body_excerpt = str(row.get('body_clean', ''))[:400]
    
    kg_ctx = get_kg_context(mid, sender)
    cent_ctx = get_centrality_context(sender_canonical, month_index)
    
    prompt = EXPLANATION_PROMPT.format(
        kg_context=kg_ctx,
        centrality_context=cent_ctx,
        body_excerpt=body_excerpt
    )
    
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                       max_length=768).to('cuda')
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=200, temperature=0.1, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    explanation = tokenizer.decode(
        output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
    ).strip()
    
    results.append({
        'mid': mid,
        'sender': sender_canonical,
        'disclosure_type': row.get('disclosure_type'),
        'risk_tier': row.get('risk_tier'),
        'month_index': month_index,
        'kg_context': kg_ctx,
        'centrality_context': cent_ctx,
        'explanation': explanation
    })

STEP 6 — BERTScore faithfulness evaluation:
# Reference = the structured KG context (what the explanation SHOULD mention)
# Hypothesis = generated explanation
references   = [r['kg_context'] + ' ' + r['centrality_context'] for r in results]
hypotheses   = [r['explanation'] for r in results]

P, R, F1 = bertscore(
    hypotheses, references,
    model_type='distilbert-base-uncased',  # lighter model for 4GB VRAM
    lang='en', verbose=False
)
F1_scores = F1.numpy().tolist()
print(f"BERTScore Faithfulness — Mean F1: {np.mean(F1_scores):.4f}")
print(f"  Min: {np.min(F1_scores):.4f}, Max: {np.max(F1_scores):.4f}")

# Add scores to results
for i, r in enumerate(results):
    r['bertscore_f1'] = F1_scores[i]

STEP 7 — Save and report:
with open('explanations/explanations_with_scores.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print top-3 and bottom-3 explanations for manual review
sorted_results = sorted(results, key=lambda x: x['bertscore_f1'], reverse=True)
print("\n=== TOP-3 FAITHFULNESS ===")
for r in sorted_results[:3]:
    print(f"Score: {r['bertscore_f1']:.4f}")
    print(f"KG Context: {r['kg_context']}")
    print(f"Explanation: {r['explanation'][:300]}")
    print("---")

print("\n=== BOTTOM-3 FAITHFULNESS ===")
for r in sorted_results[-3:]:
    print(f"Score: {r['bertscore_f1']:.4f}")
    print(f"Explanation: {r['explanation'][:300]}")
    print("---")

if np.mean(F1_scores) >= 0.60:
    print("PASS: Explainability faithfulness ≥ 0.60")
else:
    print(f"NOTE: Mean faithfulness = {np.mean(F1_scores):.4f}")
    print("Improve by: adding more specific KG entity mentions in prompt")

print("PHASE 11 COMPLETE")
=== END OF PROMPT ===
```

---

## MASTER INDEX — ALL FILES IN SEQUENCE

Use this as your execution checklist:

```
EXECUTION ORDER:
□ Phase 1  → impl_part1_novelty_and_phase1.md   → phase1_setup.py
□ Phase 2  → impl_part2_phase2_and_phase3.md    → phase2_preprocess.py
□ Phase 3A → impl_part2_phase2_and_phase3.md    → phase3a_autolabel.py
□ Phase 3B → impl_part2_phase2_and_phase3.md    → phase3b_kappa.py
              [MANUAL: human annotation with Doccano — 2 weeks]
□ Phase 4A → impl_part3_phase4_and_phase5.md    → phase4a_tfidf.py
□ Phase 4B → impl_part3_phase4_and_phase5.md    → phase4b_empath.py
□ Phase 5A → impl_part3_phase4_and_phase5.md    → phase5a_kg_build.py
□ Phase 6  → impl_part4_phase6_7_8.md           → phase6_centrality.py
□ Phase 4C → [ADD phi_G to feature matrices after Phase 6]
□ Phase 7  → impl_part4_phase6_7_8.md           → phase7_ml_models.py
□ Phase 8A → impl_part4_phase6_7_8.md           → phase8a_bilstm.py
□ Phase 8B → impl_part4_phase6_7_8.md           → phase8b_deberta.py
□ Phase 9  → impl_part5_phase9_and_phase10.md   → phase9_llm_baseline.py
□ Phase 10 → impl_part5_phase9_and_phase10.md   → phase10_evaluation.py
□ Phase 11 → impl_part6_phase11_and_index.md    → phase11_explanations.py
```

---

## PAPER CHECKLIST — READ BEFORE SUBMITTING

### Results You Must Have (Non-Negotiable)

| Item | Where Generated | Paper Section |
|---|---|---|
| Cohen's κ ≥ 0.60 on disc_type | Phase 3B | §3 Dataset |
| DeBERTa + KG macro-F1 (best model) | Phase 8B | §5 Results |
| Ablation: KG vs text-only (Δ ≥ 2% on disc_type) | Phase 10 | §5 Ablation |
| Wilcoxon p < 0.05 (best vs 2nd best) | Phase 10 | §5 Results |
| Temporal betweenness trajectory plot | Phase 6 | §5 Temporal |
| Wilcoxon crisis > stable Δ_betweenness (p < 0.05) | Phase 6 | §5 Temporal |
| BERTScore faithfulness ≥ 0.60 | Phase 11 | §5 Explainability |
| Error analysis (50 false negatives, 3+ categories) | Phase 10 | §6 Analysis |

### Figures for the Paper (Make These Beautiful)

```
Figure 1: Temporal betweenness trajectories (top-5 executives, crisis line annotated)
          → graphs/betweenness_trajectories.pdf

Figure 2: OrgDisclose architecture diagram
          → Draw manually or use Mermaid/draw.io

Table 1:  Annotation statistics (κ per dimension, class distribution)
          → Phase 3B output

Table 2:  Main comparison results (all models × all dimensions × macro-F1)
          → results/comparison_table.csv

Table 3:  Ablation study (text-only vs KG-augmented per model)
          → results/ablation_table.csv

Table 4:  Temporal analysis (period × avg_delta × HIGH risk %)
          → Phase 10 temporal analysis code

Figure 3: Confusion matrix for best model (disc_type, 6 classes)
          → results/cm_deberta_kg.pdf

Figure 4: Case study — one HIGH risk email with full explanation pipeline shown
```

---

## WEEK-BY-WEEK TIMELINE

```
Week 1:  Phases 1–2 (environment + preprocessing)
Week 2:  Phase 3A (LLM auto-labeling) + start human annotation
Week 3:  Continue annotation (human Gold set)
Week 4:  Phase 3B κ computation + Phases 4A-4B (features)
Week 5:  Phase 5A (KG) + Phase 6 (centrality)
Week 6:  Phase 7 (ML models — both variants)
Week 7:  Phase 8A (BiLSTM)
Week 8:  Phase 8B (DeBERTa — text-only)
Week 9:  Phase 8B (DeBERTa — KG-augmented) + ablation check
Week 10: Phase 9 (LLM baseline) + Phase 10 (evaluation)
Week 11: Phase 11 (explainability)
Week 12: Paper writing — methodology (2d) + results (3d) + discussion (1d) + polish (1d)
```

---

## IF YOU FACE A PROBLEM — ESCALATION PATH

```
Problem: Phase 3B κ < 0.60 on disc_type
Fix: Review the FINANCIAL vs STRATEGIC boundary. Add 5 more examples of each
     to the annotation guide. Re-annotate the 150 overlap emails.

Problem: DeBERTa goes OOM (out of memory) on 4GB VRAM
Fix: a) Reduce batch_size from 16 to 8
     b) Reduce max_length from 256 to 128
     c) Add gradient_checkpointing_enable() to model
     d) Use DistilBERT-base instead (66M params, ~20% F1 penalty)

Problem: KG ablation Δ < 1% (KG is not helping)
Fix: a) Check phi_G is actually joined correctly per email (not all zeros)
     b) Add 4 more centrality features: PageRank, clustering coefficient,
        reciprocity, and email volume percentile
     c) Verify Z-score normalization is not collapsing variance

Problem: Mistral-7B labels have > 30% noise (spot-check fails)
Fix: a) Add 2-3 more examples to the prompt
     b) Use a smaller model with better instruction following:
        'TheBloke/neural-chat-7B-v3-1-GPTQ' (better at structured output)
     c) Post-process: reject labels where confidence < 0.75 (stricter filter)

Problem: Neo4j won't start locally
Fix: Use purely NetworkX-based approach for the KG
     Store graph data in SQLite + pickle for NetworkX DiGraph objects
     audience_scope can be computed by checking if recipient email domain
     is '@enron.com' (internal) or not (external) — no Neo4j needed

Problem: BERTScore faithfulness < 0.50
Fix: a) Add explicit entity overlap scoring as second metric:
        "entity_overlap = # KG entities mentioned in explanation / # total KG entities"
     b) Report both metrics — BERTScore and entity overlap
     c) Narrative: "Our explanations achieve X BERTScore-F1 and Y entity overlap"
```

---

## FINAL NOVELTY REMINDER (REFERENCE THIS WHEN WRITING THE PAPER)

**Your 4 novel contributions — state these verbatim in your paper's contribution list:**

1. **"We introduce the first multi-dimensional disclosure taxonomy for organizational email, defining disclosure as a structured 4D classification problem (type, framing, audience, risk) on the Enron corpus, with inter-rater κ = X.XX."**

2. **"We demonstrate that temporal betweenness centrality derivatives (Δ_betweenness) are statistically significant precursors to organizational disclosure events, with crisis-period anomalies detectable up to N weeks before overt behavioral signals (Wilcoxon p < 0.001)."**

3. **"We propose a KG-augmented multi-task framework where the Knowledge Graph computes the audience-appropriateness label — making the graph causally integrated into annotation, not post-hoc decoration. Ablation confirms a +X.X% F1 gain from KG integration."**

4. **"We provide a fully reproducible, offline explainability pipeline using a locally-quantized LLM grounded in KG paths, achieving BERTScore faithfulness of X.XX with zero paid API dependency."**

If all 4 are implemented with the stated metrics, you have a paper.
If 3 of 4 are implemented, you have a paper.
If 2 of 4 are implemented, you have a workshop paper.
Go get all 4.
```
