# Automated SDRF Metadata Extraction for Proteomics

**Kaggle Competition**: [Harmonizing the Data of your Data](https://www.kaggle.com/competitions/harmonizing-the-data-of-your-data)

An end-to-end system for extracting standardised [SDRF](https://github.com/bigbio/proteomics-sample-metadata) metadata from proteomics publications. The project evolved through three architectural phases, culminating in a tool-augmented agentic system that combines a fine-tuned language model with authoritative ontology lookup tools.

---

## Problem

The [SDRF-Proteomics](https://github.com/bigbio/proteomics-sample-metadata) format standardises how proteomics experiments are annotated — capturing 81 columns of metadata across biology (organism, disease, cell type), sample preparation (cleavage enzyme, modifications, labelling), and instrument configuration (MS analyser, fragmentation, tolerances). Values must follow exact ontology formats (e.g., `NT=Q Exactive HF;AC=MS:1002523`).

Manually curating SDRF files from published papers is labour-intensive and error-prone. This competition challenges participants to automate SDRF generation from 15 test proteomics publications deposited in [PRIDE](https://www.ebi.ac.uk/pride/).

**Scoring**: For each (PXD, column) pair, the submission's unique values are compared against gold-standard values using agglomerative clustering at a 0.80 SequenceMatcher threshold. Per-pair F1 scores are macro-averaged across all PXDs and columns.

---

## Results

| Version | F1 Score | Architecture |
|---------|----------|-------------|
| v1 | 0.239 | Single-model extraction |
| v6 | **0.318** | Multi-round 3-model deliberation |
| v15 | 0.402* | Defaults-first + PRIDE API + single targeted LLM call |
| Agentic | 0.65-0.80** | Fine-tuned 7B model + MCP tool augmentation |

\* Local validation on 10 training PXDs
\** Projected based on component analysis

---

## Architecture Evolution

### Phase 1: Multi-Round Deliberation (v1-v6)

Three frontier models (GPT-4.1, DeepSeek-V3.2, Claude) independently extract metadata, then reconcile through structured deliberation rounds. A grounded judge produces the final ontology-compliant output with few-shot examples from training SDRFs.

```
Paper Text ──► 3 Models (independent extraction)
             ──► 3 Models (cross-pollinated deliberation)
             ──► Judge (grounded with ontology dicts + few-shot)
             ──► Validation/refinement loop
             ──► submission.csv
```

**Key insight**: While architecturally elegant, the multi-round approach scored 0.318 — lower than smart defaults alone. The LLM hallucinated ontology codes, overwrote correct defaults (Trypsin, RPLC) with wrong guesses, and couldn't reliably produce the exact format strings the scorer expects.

See [`pipeline_merged_v6.py`](pipeline_merged_v6.py) for the full implementation.

### Phase 2: Defaults-First Pipeline (v15)

A fundamental rethink: instead of asking the LLM to do everything, use structured data sources for what they're good at and restrict the LLM to biology-only extraction.

```
Layer 1: Protected defaults (CleavageAgent=Trypsin, Separation=RPLC) ──── never overridden
Layer 2: PRIDE API (Instrument, Organism, Label type) ─────────────────── authoritative
Layer 3: Single LLM call (biology columns only) ───────────────────────── vocabulary-constrained
Layer 4: Technical overrides (only if LLM output matches gold vocabulary)
```

**Key findings**:
- Defaults + PRIDE alone score 0.372, beating v6's 0.318
- The LLM is net-positive (+0.03) when restricted to biology extraction only
- BiologicalReplicate = "1" for all rows matches the gold majority (F1 0.073 → 0.639)
- "Not Applicable" in both gold and submission = F1 1.0 (a match, not excluded)

See [`pipeline_v15.py`](pipeline_v15.py) for the implementation.

### Phase 3: Agentic System with MCP Tools

The final architecture addresses the root cause of LLM failures: ontology format hallucination. A fine-tuned 7B model uses tool calls to look up exact accession codes instead of guessing.

```
                    ┌──────────────────────────────┐
                    │   Fine-tuned Qwen 2.5-7B     │
                    │                              │
                    │   Reads paper → extracts     │
                    │   biology → calls tools for  │
                    │   exact ontology codes        │
                    └──────────┬───────────────────┘
                               │ tool calls
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │  PRIDE API    │   │  Ontology DB  │   │  Format DB   │
  │              │   │              │   │              │
  │ instruments  │   │ PSI-MS OBO   │   │ 374 gold     │
  │ organisms    │   │ UNIMOD XML   │   │ SDRFs as     │
  │ mods/quant   │   │              │   │ reference    │
  └──────────────┘   └──────────────┘   └──────────────┘
```

**5 MCP Tools**:
| Tool | Source | Purpose |
|------|--------|---------|
| `pride_lookup` | PRIDE REST API | Authoritative instrument, organism, modification data |
| `ms_ontology_lookup` | PSI-MS OBO (600 terms) | Exact accession codes for instruments, fragmentation, analysers |
| `unimod_lookup` | UNIMOD XML (1,560 mods) | Formatted modification strings with targets and types |
| `paper_fetch` | Local + EuropePMC | Full paper text retrieval with JATS XML parsing |
| `sdrf_format_reference` | 374 gold SDRFs (5,965 values) | Canonical format strings with frequency data |

**Training data**: 103 synthetic tool-calling traces generated from gold SDRFs with real tool responses — the model learns to trust tool outputs rather than memorising accession codes.

See [`docs/AGENT_DESIGN.md`](docs/AGENT_DESIGN.md) for the full system design and [`sdrf-mcp-server/`](sdrf-mcp-server/) for the implementation.

---

## Key Technical Insights

### Scoring Asymmetry

The scorer only iterates over categories in the gold standard. This creates an asymmetry:
- **False positives** (filling columns gold has as "Not Applicable"): **zero cost**
- **False negatives** (leaving columns empty that gold has filled): **F1 = 0** for that pair
- **Both "Not Applicable"**: **excluded** from scoring entirely

This means aggressive column filling is always better than conservative defaults.

### The "Not Applicable" Trap

Changing a column from `"Not Applicable"` to `"not available"` seems harmless, but:
- If gold also has `"Not Applicable"` → both are excluded, **F1 = 1.0**
- If you change to `"not available"` → new scored pair where gold's `"Not Applicable"` doesn't match → **F1 = 0**

This single insight explains why v6's aggressive LLM extraction (0.318) scored worse than passive defaults + PRIDE (0.372).

### Protected Defaults

Columns where defaults score perfectly should never be overridden by LLM output:

| Column | Default | Training F1 |
|--------|---------|-------------|
| CleavageAgent | `AC=MS:1001251;NT=Trypsin` | 1.000 |
| Separation | `AC=PRIDE:0000563;NT=Reversed-phase chromatography` | 1.000 |
| AlkylationReagent | `iodoacetamide` | 1.000 |
| ReductionReagent | `DTT` | 1.000 |

### Gold SDRF Corpus Analysis

Analysis of 374 gold SDRFs (103 training + 271 bigbio) revealed constrained vocabularies:
- 14 unique instruments, 14 enzymes, 36 modifications, 10 fragmentation methods
- Format inconsistencies exist (`AC=...;NT=...` vs `NT=...;AC=...`) but the 0.80 fuzzy threshold handles them
- The gold corpus serves as a vocabulary constraint, not a template source

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── pipeline_merged_v6.py          # v6: Multi-round 3-model deliberation (0.318 Kaggle)
├── pipeline_v15.py                # v15: Defaults-first + PRIDE + single LLM (0.402 local)
├── scoring.py                     # Competition scoring metric implementation
├── evaluate.py                    # Local evaluation against training gold SDRFs
├── error_analysis.py              # Strategy ladder analysis and per-column diagnostics
│
├── validate_v15.py                # Full LLM validation on training PXDs
├── validate_v15_nollm.py          # No-LLM baseline (just defaults + PRIDE)
│
├── submission_v6.csv              # Best Kaggle submission (F1 = 0.318)
├── submission_v15.csv             # v15 output (F1 = 0.402 local)
│
├── sdrf-mcp-server/               # Agentic system implementation
│   ├── server.py                  # MCP server with 5 tools
│   ├── build_databases.py         # Ontology DB builder (PSI-MS + UNIMOD + gold SDRFs)
│   ├── generate_training_data.py  # Training trace generator (103 examples)
│   ├── inference_pipeline.py      # Agentic inference loop (model + tools)
│   ├── test_tools.py              # Tool validation suite
│   ├── requirements.txt
│   └── Dockerfile
│
└── docs/
    └── AGENT_DESIGN.md            # Agentic system design document
```

---

## Usage

### Pipeline v15 (defaults-first)

```bash
pip install openai pandas requests

export AZURE_OPENAI_ENDPOINT="https://your-endpoint.services.ai.azure.com"
export AZURE_OPENAI_API_KEY="your-key"

# Full run on test PXDs
python pipeline_v15.py --data_dir ./data --output submission.csv

# Single PXD
python pipeline_v15.py --data_dir ./data --single-pxd PXD004010
```

### Agentic System

```bash
cd sdrf-mcp-server

# Build ontology databases (downloads PSI-MS OBO + UNIMOD XML, parses gold SDRFs)
python build_databases.py

# Test all 5 MCP tools
python test_tools.py

# Generate training data (103 tool-calling traces from gold SDRFs)
python generate_training_data.py

# Run agentic inference with GPT-4.1
python inference_pipeline.py --model gpt4 --submit

# Or start the MCP server for external model access
python server.py                     # stdio transport
python server.py --transport http    # HTTP transport on port 8080
```

### Local Evaluation

```bash
# Evaluate against training gold standard
python evaluate.py --data_dir data

# No-LLM baseline (defaults + PRIDE only)
python validate_v15_nollm.py

# Full LLM validation
python validate_v15.py
```

---

## Technical Stack

- **Models**: GPT-4.1, DeepSeek-V3.2 via Azure AI Foundry; Qwen 2.5-7B-Instruct (fine-tuning target)
- **Tools**: [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) with FastMCP server
- **Ontologies**: [PSI-MS](https://github.com/HUPO-PSI/psi-ms-CV) (instruments/analysers), [UNIMOD](http://www.unimod.org/) (modifications)
- **APIs**: [PRIDE Archive](https://www.ebi.ac.uk/pride/ws/archive/v2/), [EuropePMC](https://europepmc.org/RestfulWebService)
- **Storage**: SQLite (ontology databases, gold SDRF index)
- **Serving**: RunPod Serverless with SGLang (for fine-tuned model deployment)

---

## License

MIT
