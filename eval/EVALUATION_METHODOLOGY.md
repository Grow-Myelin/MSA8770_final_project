# RAG System Evaluation Methodology

## Overview

This document describes the evaluation methodology used to assess the performance of the unemployment insurance legal RAG (Retrieval-Augmented Generation) system compared to a baseline LLM without retrieval capabilities.

## Test Dataset

**File**: `eval/RAG_eval_Questions.txt`

The evaluation uses 23 curated questions about Georgia unemployment insurance law, organized into 4 categories:

| Category | Count | Description |
|----------|-------|-------------|
| Legal Standards | 6 | Questions about fault determination, standard of review, material misrepresentation, good faith |
| Procedural Requirements | 5 | Questions about certification requirements, timelines, appeal procedures |
| Common Fact Patterns | 5 | Questions about employment reporting, agency errors, reliance scenarios |
| Statutory Provisions | 7 | Questions about waiver requirements, statutory definitions, policy interpretation |

## Evaluation Metrics

### 1. Groundedness Score (0-1)

Measures how well responses are grounded in source documents via citations.

**Calculation**:
- Detects citation patterns in responses (case names, O.C.G.A. references, Ga. Comp. R. & Regs.)
- Regex patterns used:
  - Case citations: `v\.|vs\.`
  - Statutory citations: `O\.C\.G\.A\.|§\s*\d+|Section\s+\d+`
  - Regulatory citations: `Ga\.\s*Comp\.\s*R\.\s*&\s*Regs\.|Rule\s+\d+`

**Scoring**:
- 0 citations = 0.0
- 1 citation = 0.25
- 2 citations = 0.5
- 3 citations = 0.75
- 4+ citations = 1.0

### 2. Answer Quality Score (0-1)

Measures the legal specificity and structure of responses.

**Components** (each 0-0.25):
- **Legal Terminology**: Presence of domain-specific terms (claimant, overpayment, recoupment, waiver, etc.)
- **Structured Reasoning**: Contains numbered lists, bullet points, or clear section headers
- **Specificity**: References specific code sections (O.C.G.A., Rule, Section)
- **Sufficient Length**: Response exceeds 200 characters

### 3. Response Time (seconds)

Wall-clock time from query submission to response completion.

## Comparison Methodology

### RAG System (Treatment)
- Queries are processed through the full agentic pipeline
- System retrieves relevant case law and regulations from the vector database
- LLM generates responses grounded in retrieved documents

### Baseline (Control)
- Same questions sent directly to the LLM
- No retrieval augmentation
- Tests the model's parametric knowledge alone

## Output Artifacts

The evaluation script generates the following outputs in `evaluation_results/`:

| File | Description |
|------|-------------|
| `eval_results_TIMESTAMP.json` | Raw JSON data with all metrics and individual responses |
| `eval_comparison_TIMESTAMP.png` | Bar chart comparing RAG vs Baseline on key metrics |
| `eval_by_category_TIMESTAMP.png` | Performance breakdown by question category |
| `eval_results_TIMESTAMP.html` | Formatted HTML report with full results tables |

## Running the Evaluation

```bash
# Activate virtual environment
source venv/bin/activate

# Run evaluation (default: all 23 questions)
python scripts/evaluate.py

# Run with limited questions for testing
python scripts/evaluate.py --limit 5
```

**Requirements**:
- FastAPI server must be running on port 8000
- Vector database must be populated with case documents
- `matplotlib` must be installed for visualizations

## Interpreting Results

### Expected RAG Advantages
- **Higher Groundedness**: RAG responses should cite specific cases and statutes
- **Better Quality**: More precise legal terminology and structured answers
- **Slower Response Time**: Retrieval adds latency but improves accuracy

### Category-Specific Insights
- **Statutory Provisions**: RAG should excel by citing specific O.C.G.A. sections
- **Legal Standards**: RAG can reference actual case holdings
- **Procedural Requirements**: Benefits from regulatory citations
- **Common Fact Patterns**: RAG can provide case-based examples

## Limitations

1. **Automated Scoring**: Metrics use heuristics (citation counting, keyword detection) rather than human judgment
2. **No Ground Truth**: Questions don't have verified "correct" answers for comparison
3. **Single Run**: Results may vary based on LLM response randomness
4. **Citation Detection**: Regex patterns may miss some citation formats or produce false positives

## Future Improvements

- Add human evaluation component for answer accuracy
- Include retrieval precision/recall metrics
- Implement answer correctness via expert-labeled ground truth
- Add confidence intervals from multiple evaluation runs
