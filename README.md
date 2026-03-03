# Proteomics SDRF Metadata Extraction via Multi-Model Deliberation

**Kaggle Competition**: [Harmonizing the Data of your Data](https://www.kaggle.com/competitions/harmonizing-the-data-of-your-data)

An LLM-based pipeline that extracts standardized Sample and Data Relationship Format (SDRF) metadata from proteomics publications. Three frontier models independently extract metadata from paper text, then reconcile through structured deliberation rounds before a grounded judge produces the final ontology-compliant output.

**Best Score**: 0.318 mean F1 (top-tier on leaderboard)

---

## Problem Statement

The [SDRF-Proteomics](https://github.com/bigbio/proteomics-sample-metadata) format standardizes how proteomics experiments are annotated — capturing 81 columns of metadata including organism, instrument, post-translational modifications, labeling strategy, and sample characteristics. Manually curating SDRF files from published papers is labor-intensive and error-prone. This competition challenges participants to automate SDRF generation from 15 test proteomics publications deposited in [PRIDE](https://www.ebi.ac.uk/pride/).

**Scoring**: For each (PXD, column) pair, the submission's unique values are compared against gold-standard values using agglomerative clustering at 0.80 string similarity. Per-pair F1 scores are averaged across all PXDs and columns.

---

## Architecture

```
                    ┌──────────────┐
                    │  Paper Text  │
                    │  + Raw Files │
                    │  + PRIDE API │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Claude   │ │ GPT-4.1  │ │ DeepSeek │   R1: Independent
        │ Opus 4.5 │ │          │ │  V3.2    │   Extraction
        └────┬─────┘ └────┬─────┘ └────┬─────┘   (6 calls: 2 specialist
             │            │            │           prompts per model)
             └────────────┼────────────┘
                          │ All 3 extractions
              ┌───────────┼───────────┐
              ▼           ▼           ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Claude   │ │ GPT-4.1  │ │ DeepSeek │   R2: Cross-Pollinated
        │ Opus 4.5 │ │          │ │  V3.2    │   Deliberation
        └────┬─────┘ └────┬─────┘ └────┬─────┘   (3 calls, with
             │            │            │           confidence scores)
             └────────────┼────────────┘
                          │ All 3 reconciliations
                          ▼
                   ┌─────────────┐
                   │ Claude Opus │                R3: Grounded Judge
                   │  4.5 Judge  │◄── Ontology    (1 call, with training
                   │             │    Dicts +      SDRF few-shot examples)
                   └──────┬──────┘    Few-Shot
                          │           Examples
                          ▼
                    ┌───────────┐
                    │ Validate  │──fail──► Refine ──► R3 Judge
                    └─────┬─────┘         (up to 2×)
                        pass
                          │
                          ▼
                  ┌───────────────┐
                  │  Format SDRF  │──► submission.csv
                  │  (81 columns) │
                  └───────────────┘
```

### Round 1 — Specialist Extraction (6 API calls)

Each of the three models runs **two focused prompts** instead of one monolithic extraction:

- **Biological prompt**: Organism, tissue, disease, cell type, cell line, sex, age, developmental stage, strain, ancestry, treatment, compound, label type, material type (~25 fields)
- **Analytical prompt**: Instrument, cleavage agent, fragmentation method, MS2 mass analyzer, separation, collision energy, mass tolerances, modifications, enrichment, fractionation (~20 fields)

This domain decomposition yields measurably better per-field accuracy than a single 50-field prompt (+0.008 F1).

### Round 2 — Cross-Pollinated Deliberation (3 API calls)

Each model receives **all three** Round 1 extractions alongside the original paper. Each produces a reconciled extraction with per-field confidence scores (0.0–1.0). This is where cross-pollination occurs: a model that missed the instrument in its own extraction can adopt it from another model's output.

### Round 3 — Grounded Judge (1 API call)

Claude Opus 4.5 acts as final judge, receiving all Round 2 reconciliations plus grounding context:

- **Training SDRF few-shot examples**: The 2 most similar training SDRFs (by organism + instrument Jaccard overlap) are injected as formatting calibration
- **Complete ontology dictionaries**: 700+ instruments, 60+ modifications, cleavage agents, fragmentation methods — all with correct accession codes
- **Competition scoring specification**: The judge understands how string similarity clustering works at the 0.80 threshold
- **Cross-field consistency checks**: Organism vs. material type, label type vs. expected row multiplicity

### Validation and Refinement

The judge output is validated for critical field presence (organism, instrument, cleavage agent, label type), row count consistency with labeling strategy, and organism binomial format. Failed validations trigger targeted re-prompting (up to 2 iterations).

### Ensemble Averaging

The full pipeline runs twice at different temperatures (0.1 and 0.3). Results are merged per-PXD:
- **Agreement** (both runs same value): kept with high confidence
- **One-sided** (one run found a value, the other didn't): value is kept
- **Conflict** (different values): lower temperature run wins
- **Modifications**: union of both runs (maximizes recall)

---

## Key Engineering Decisions

### Bayesian Column Gating

Training data analysis of 103 gold-standard SDRFs revealed that many columns are active in fewer than half of all studies:

| Column | Active Rate | Strategy |
|--------|-------------|----------|
| DevelopmentalStage | 8% | Suppress unless specific value |
| AncestryCategory | 3% | Suppress all guesses |
| Age | 47% | Suppress all guesses |
| Sex | 58% | Suppress unless high confidence |

Rather than letting LLMs hallucinate values for rarely-populated columns, the pipeline suppresses them based on training-set activation frequencies. This reduces false positives that inflate the scorer's denominator.

### PRIDE API Augmentation

For each test PXD, the pipeline fetches structured metadata from the [PRIDE REST API](https://www.ebi.ac.uk/pride/ws/archive/v2/):
- Instruments (with ontology accessions)
- Organisms and organism parts
- Diseases and modifications
- Project-level metadata (title, keywords, submitters)
- Raw file listings (used for row count determination and fraction parsing)

PRIDE data serves as a reliability anchor — LLM extractions are cross-checked against it, and well-structured PRIDE values are preferred over ambiguous LLM guesses.

### Ontology Compliance

SDRF values must follow specific ontology formats: `NT=<human-readable>;AC=<accession>`. The pipeline embeds complete dictionaries for:

- **Instruments**: 700+ mass spectrometers mapped to PSI-MS accessions
- **Modifications**: UNIMOD-indexed post-translational modifications with position and target amino acid
- **Cleavage agents**: Enzyme names with MS ontology codes
- **Fragmentation methods**: HCD, CID, ETD, EThcD with correct accessions
- **Labels**: TMT, iTRAQ, SILAC, label free — with channel-level row expansion

### Fraction Identifier Parsing

Raw file names encode experimental structure. The pipeline parses fraction identifiers using multi-pattern regex:
1. Explicit fraction prefixes (`F12`, `frac06`, `Fr3`)
2. Temperature-based fractionation patterns (`65C_12.raw`)

With strict lookaheads to avoid false positives from mutations (`F198S`), temperatures (`30min`), or other numeric patterns.

---

## Score Progression

| Version | F1 Score | Key Changes |
|---------|----------|-------------|
| v1 | 0.239 | Baseline: single-model extraction |
| v2 | 0.276 | 3-model deliberation pipeline |
| v3 | 0.293 | Ontology grounding + PRIDE augmentation |
| v4 | 0.305 | Format compliance (NT-first instruments) |
| v5 | 0.314 | Bayesian column gating |
| v6 | **0.318** | Specialist prompts + ensemble + few-shot examples |

---

## Repository Structure

```
.
├── pipeline_merged.py          # Main pipeline (2,377 lines)
│   ├── Ontology dictionaries   # Instruments, mods, enzymes, analyzers
│   ├── PRIDE API client        # Fetches structured metadata per PXD
│   ├── R1 extraction           # Specialist BIO + ANALYTICAL prompts
│   ├── R2 deliberation         # Cross-pollinated reconciliation
│   ├── R3 judge                # Grounded judge with few-shot examples
│   ├── Validation/refinement   # Cross-field consistency checks
│   ├── SDRF formatting         # Row expansion, ontology mapping
│   └── Ensemble merge          # Multi-run aggregation
├── scoring.py                  # Competition scoring function
├── evaluate.py                 # Local evaluation harness
├── data/
│   ├── TestPubText/            # 15 test paper texts (JSON + TXT)
│   ├── TrainingPubText/        # 103 training paper texts
│   ├── TrainingSDRFs/          # 103 gold-standard SDRF files
│   └── SampleSubmission.csv    # Competition template (1,659 rows × 81 cols)
└── submission_v6.csv           # Best submission (F1 = 0.318)
```

---

## Usage

```bash
# Install dependencies
pip install langgraph langchain-core requests

# Set API credentials
export AZURE_AI_ENDPOINT="https://your-endpoint.services.ai.azure.com"
export AZURE_AI_KEY="your-api-key"

# Test model connectivity
python pipeline_merged.py --data_dir ./data --test

# Process a single PXD
python pipeline_merged.py --data_dir ./data --single-pxd PXD004010

# Full run (15 PXDs, single pass)
python pipeline_merged.py --data_dir ./data --output submission.csv

# Full run with ensemble (2 passes at different temperatures)
python pipeline_merged.py --data_dir ./data --output submission.csv --ensemble 2

# Evaluate against training gold standard
python evaluate.py --data_dir ./data
```

---

## Technical Stack

- **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph) (StateGraph with conditional edges)
- **Models**: Claude Opus 4.5, GPT-4.1, DeepSeek-V3.2 via Azure AI Foundry
- **Judge**: Claude Opus 4.5 (highest reasoning capability for cross-field consistency)
- **External API**: PRIDE Archive REST API v2
- **Ontologies**: PSI-MS (instruments, analyzers), UNIMOD (modifications), PRIDE (organisms, diseases)

---

## Cost

| Configuration | API Calls/PXD | Total (15 PXDs) |
|---------------|---------------|-----------------|
| Single run | 10 (no errors) – 16 (2 refinements) | ~$4.50 |
| Ensemble (2×) | 20 – 32 | ~$9.00 |

---

## Limitations and Future Work

**Current limitation**: The pipeline extracts a single canonical value per metadata field and replicates it across all rows. Gold-standard SDRFs often contain per-sample variation — e.g., `Disease = {cancer, normal}` across different experimental groups. This architectural constraint caps achievable F1 for any column with multi-valued content.

**Planned improvements**:
- **Multi-value extraction**: Extract ALL distinct values per column, then map raw files to experimental groups for per-row variation
- **Local evaluation loop**: Score against training SDRFs to identify column-level F1 losses without consuming Kaggle submission attempts
- **PRIDE SDRF fallback**: Check if PRIDE hosts a submitted SDRF file before running LLM extraction

---

## License

MIT
