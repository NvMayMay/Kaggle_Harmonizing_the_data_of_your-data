# Agentic SDRF Extraction System

## Problem

Given a proteomics dataset accession (PXD), produce a complete SDRF metadata file with ~25 structured columns covering biology (organism, disease, cell type), sample preparation (cleavage, modifications, labeling), and instrument configuration (MS analyzer, fragmentation, tolerances). Values must match exact ontology formats (UNIMOD codes, MS accession numbers, PRIDE CV terms).

**Current best**: v15 pipeline scores F1=0.40 locally using smart defaults + PRIDE API + single GPT-4.1 call. The LLM guesses ontology formats and gets them wrong ~40% of the time.

**Target**: F1 0.75-0.85 by combining a fine-tuned extraction model with authoritative ontology lookup tools.

---

## Architecture

```
                    ┌──────────────────────────────┐
                    │   Fine-tuned 7B Model         │
                    │   (Qwen 2.5-7B-Instruct)      │
                    │                                │
                    │   Trained on 374 SDRFs to:     │
                    │   - Read papers → extract      │
                    │   - Know when to call tools     │
                    │   - Know "not available" vs     │
                    │     "Not Applicable"            │
                    │   - Output structured JSON      │
                    └──────────┬───────────────────┘
                               │ tool calls
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │  PRIDE API    │   │  Ontology DB  │   │  Format DB   │
  │              │   │              │   │              │
  │ instruments  │   │ PSI-MS OBO   │   │ 374 gold     │
  │ organisms    │   │ UNIMOD XML   │   │ SDRFs as     │
  │ mods/quant   │   │ PRIDE CV     │   │ reference    │
  │ protocols    │   │              │   │ values       │
  └──────────────┘   └──────────────┘   └──────────────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               ▼
                    ┌──────────────────────────────┐
                    │   Row Builder (deterministic) │
                    │   - TMT/SILAC expansion       │
                    │   - Fraction parsing           │
                    │   - Replicate assignment        │
                    │   - ID mapping                  │
                    └──────────────────────────────┘
                               │
                               ▼
                         submission.csv
```

---

## MCP Tools (5 tools)

### Tool 1: `pride_lookup`

**Purpose**: Fetch authoritative metadata from PRIDE REST API.

```
Input:  { "pxd": "PXD000070" }
Output: {
  "instruments": [
    {"name": "LTQ Orbitrap Velos", "accession": "MS:1001742"}
  ],
  "organisms": [
    {"name": "Plasmodium falciparum", "accession": "NEWT:5833"}
  ],
  "modifications": [
    {"name": "Oxidation", "accession": "MOD:00696"},
    {"name": "Carbamidomethyl", "accession": "MOD:01060"}
  ],
  "quantification": "label-free",
  "sample_protocol": "...",
  "data_protocol": "..."
}
```

**Source**: `https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{pxd}`

**Why the model needs this**: PRIDE is authoritative for instrument and organism. The model reads the paper independently and cross-references. If PRIDE says "Q Exactive" but the paper says "Orbitrap Velos", the model can resolve the conflict by checking if PRIDE listed multiple instruments.

---

### Tool 2: `ms_ontology_lookup`

**Purpose**: Look up exact accession codes for MS instruments, fragmentation methods, analyzers, and other PSI-MS controlled vocabulary terms.

```
Input:  { "term": "Q Exactive HF", "category": "instrument" }
Output: {
  "accession": "MS:1002523",
  "name": "Q Exactive HF",
  "formatted": "NT=Q Exactive HF;AC=MS:1002523",
  "parent": "MS:1000494 (Thermo Fisher Scientific instrument model)"
}
```

```
Input:  { "term": "HCD", "category": "fragmentation" }
Output: {
  "accession": "MS:1000422",
  "name": "beam-type collision-induced dissociation",
  "formatted": "AC=MS:1000422;NT=HCD",
  "synonyms": ["HCD", "beam-type CID", "higher-energy collisional dissociation"]
}
```

**Source**: PSI-MS OBO file (`https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo`), parsed into a searchable index.

**Categories supported**:
- `instrument` — ~200 instrument models with MS: accession codes
- `fragmentation` — HCD, CID, ETD, EThcD, ETciD (~10 methods)
- `analyzer` — Orbitrap, ion trap, TOF, FTICR (~8 types)
- `ionization` — ESI, MALDI, nanoESI (~5 types)
- `separation` — RP, HILIC, SCX, SAX (~10 methods via PRIDE CV)

**Why the model needs this**: Eliminates accession code hallucination entirely. The model extracts the instrument name from the paper, calls the lookup, gets the guaranteed-correct formatted string.

---

### Tool 3: `unimod_lookup`

**Purpose**: Look up exact UNIMOD modification entries with accession codes, target residues, and modification types.

```
Input:  { "modification": "phosphorylation" }
Output: {
  "accession": "UNIMOD:21",
  "name": "Phospho",
  "targets": ["S", "T", "Y"],
  "type": "Variable",
  "formatted": "NT=Phospho;AC=UNIMOD:21;TA=S,T,Y;MT=Variable",
  "monoisotopic_mass": 79.966331
}
```

```
Input:  { "modification": "carbamidomethyl" }
Output: {
  "accession": "UNIMOD:4",
  "name": "Carbamidomethyl",
  "targets": ["C"],
  "type": "Fixed",
  "formatted": "NT=Carbamidomethyl;AC=UNIMOD:4;TA=C;MT=Fixed",
  "monoisotopic_mass": 57.021464
}
```

**Source**: UNIMOD XML (`http://www.unimod.org/xml/unimod.xml`), parsed into a searchable index.

**Coverage**: ~2000 modifications. The 36 most common in gold SDRFs cover 99%+ of cases:
- Fixed: Carbamidomethyl, Propionamide
- Variable: Oxidation, Acetyl, Phospho, Deamidated, GlyGly, Methyl, Dimethyl, Hydroxylation
- Labels-as-mods: TMT6plex, TMTpro, iTRAQ4plex, SILAC (13C6, 13C6-15N2, 13C6-15N4)

**Why the model needs this**: Modification formatting is the hardest column (F1=0.37 currently). The model reads "phosphorylation on serine and threonine residues" from the paper, calls the tool, gets the exact UNIMOD string. No format guessing.

---

### Tool 4: `paper_fetch`

**Purpose**: Retrieve full paper text when not available locally, or fetch additional sections.

```
Input:  { "pxd": "PXD000070" }
Output: {
  "title": "...",
  "abstract": "...",
  "methods": "...",
  "results": "...",
  "doi": "10.1234/...",
  "pmid": "12345678"
}
```

**Resolution chain**:
1. Check local `TestPubText/{pxd}_PubText.json`
2. Query PRIDE API for publication DOI/PMID
3. Fetch from EuropePMC: `https://www.ebi.ac.uk/europepmc/webservices/rest/{PMCID}/fullTextXML`
4. Parse JATS XML → extract METHODS section (most relevant for SDRF)

**Why the model needs this**: Some papers have truncated text. The model can request the full methods section for detailed protocol information (cleavage enzyme, fractionation details, mass spec parameters).

---

### Tool 5: `sdrf_format_reference`

**Purpose**: Look up canonical SDRF-formatted values from the gold corpus of 374 annotated SDRFs.

```
Input:  { "column": "Comment[Instrument]", "query": "Q Exactive HF" }
Output: {
  "exact_matches": ["NT=Q Exactive HF;AC=MS:1002523"],
  "fuzzy_matches": [
    {"value": "NT=Q Exactive HF-X;AC=MS:1002877", "similarity": 0.85},
    {"value": "NT=Q Exactive Plus;AC=MS:1002634", "similarity": 0.72}
  ],
  "frequency": 6068,
  "total_occurrences": 36280
}
```

```
Input:  { "column": "Comment[CollisionEnergy]", "query": "30" }
Output: {
  "exact_matches": ["30% NCE", "30 NCE"],
  "frequency_map": {"30% NCE": 3773, "30 NCE": 412},
  "recommendation": "30% NCE"
}
```

**Source**: Pre-parsed index of all 374 gold SDRFs (103 training + 271 bigbio).

**Coverage per column** (unique canonical values):

| Column | Unique Values | Notes |
|--------|--------------|-------|
| Instrument | 14 | All with MS: accession codes |
| CleavageAgent | 14 | Trypsin dominates (81%) |
| FragmentationMethod | 10 | HCD, CID, ETD, EThcD, ETciD + variants |
| MS2MassAnalyzer | 5 | Orbitrap, ion trap, TOF, FTICR, Astral |
| Separation | 3 | RP chromatography, HPLC, SAX |
| Modification | 36 | Normalized by NT= name |
| Label | 36 | label-free, TMT channels, SILAC, iTRAQ |
| MaterialType | 22 | tissue, cell, organism part, etc. |
| FractionationMethod | 15 | ~9 after deduplication |
| EnrichmentMethod | 7 | Phospho, IMAC, TiO2, none |
| CollisionEnergy | 17 | Format: "NN NCE" or "NN% NCE" |
| PrecursorMassTolerance | 13 | Format: "N ppm" or "N Da" |
| FragmentMassTolerance | 25 | Format: "N Da", "N ppm", "N m/z" |

**Why the model needs this**: After extracting a raw value, the model queries this tool to find the exact gold-standard format string. This guarantees scorer compatibility (the 0.80 fuzzy threshold becomes irrelevant when you output exact matches).

---

## Fine-Tuned Model Design

### Base Model

**Qwen 2.5-7B-Instruct** — chosen because:
- 7B parameters: fast inference, single GPU serving
- Strong instruction following and JSON output
- Good at structured extraction tasks
- Compatible with existing training infrastructure (train_v2.py, RunPod H100s)

### Training Data Generation

**Source**: 374 gold SDRFs (103 training + 271 bigbio) with their associated papers.

**Training format** (ChatML with tool calls):

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an SDRF metadata extraction agent. Extract proteomics experiment metadata from papers. You have access to tools: pride_lookup, ms_ontology_lookup, unimod_lookup, paper_fetch, sdrf_format_reference. Call tools when you need authoritative data. Output final results as JSON."
    },
    {
      "role": "user",
      "content": "Extract SDRF metadata for PXD000070.\n\nPaper text:\n{paper_methods_section}"
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [{"function": {"name": "pride_lookup", "arguments": "{\"pxd\": \"PXD000070\"}"}}]
    },
    {
      "role": "tool",
      "name": "pride_lookup",
      "content": "{\"instruments\": [{\"name\": \"LTQ Orbitrap Velos\", \"accession\": \"MS:1001742\"}], ...}"
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [{"function": {"name": "ms_ontology_lookup", "arguments": "{\"term\": \"LTQ Orbitrap Velos\", \"category\": \"instrument\"}"}}]
    },
    {
      "role": "tool",
      "name": "ms_ontology_lookup",
      "content": "{\"accession\": \"MS:1001742\", \"formatted\": \"NT=LTQ Orbitrap Velos;AC=MS:1001742\"}"
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [{"function": {"name": "unimod_lookup", "arguments": "{\"modification\": \"carbamidomethyl\"}"}}]
    },
    {
      "role": "tool",
      "name": "unimod_lookup",
      "content": "{\"formatted\": \"NT=Carbamidomethyl;AC=UNIMOD:4;TA=C;MT=Fixed\"}"
    },
    {
      "role": "assistant",
      "content": "```json\n{\"organism\": \"Plasmodium falciparum\", \"instrument\": \"NT=LTQ Orbitrap Velos;AC=MS:1001742\", \"cleavage_agent\": \"AC=MS:1001251;NT=Trypsin\", \"fragmentation\": \"AC=MS:1000422;NT=HCD\", \"modifications\": [\"NT=Carbamidomethyl;AC=UNIMOD:4;TA=C;MT=Fixed\", \"NT=Oxidation;AC=UNIMOD:35;MT=Variable;TA=M\"], \"organism_part\": \"not available\", \"disease\": \"not available\", \"cell_type\": \"not available\", \"material_type\": \"cell\", \"label\": \"AC=MS:1002038;NT=label free sample\", ...}\n```"
    }
  ]
}
```

### Training Data Pipeline

```
For each of 374 gold SDRFs:
  1. Load gold SDRF → extract per-column values
  2. Load paper text (training: local files, bigbio: fetch via EuropePMC)
  3. Fetch PRIDE metadata for the PXD
  4. Generate synthetic tool-call trace:
     - Always: pride_lookup → ms_ontology_lookup (for instrument)
     - Always: unimod_lookup (for each modification in gold)
     - Sometimes: sdrf_format_reference (for tolerance values, collision energy)
     - Sometimes: paper_fetch (simulate fetching additional context)
  5. Final assistant message: JSON with all gold values in exact format
  6. Save as ChatML conversation
```

**Key design choice**: The tool responses in training data are **real** — we actually call the ontology lookup during data generation and record the real response. This means the model learns to trust tool outputs, not to memorize accession codes.

### What the Model Learns vs What Tools Handle

| Capability | Model | Tools |
|---|---|---|
| Reading papers for biology | Primary | - |
| Deciding which tools to call | Primary | - |
| When to say "not available" | Primary (calibrated from 374 examples) | - |
| Instrument accession codes | - | ms_ontology_lookup |
| UNIMOD modification strings | - | unimod_lookup |
| Cross-referencing PRIDE vs paper | Primary (conflict resolution) | pride_lookup |
| Output format validation | - | sdrf_format_reference |
| Row construction (TMT, fractions) | - | Deterministic code |

### Training Configuration

- **Model**: Qwen 2.5-7B-Instruct
- **Method**: LoRA r=32, alpha=64 (reuse train_v2.py infra)
- **Data**: ~374 examples with tool-call traces (~1500 messages total)
- **Epochs**: 3-5 (small dataset, need multiple passes)
- **Max seq length**: 8192 (papers can be long)
- **GPU**: 1x H100 on RunPod (7B fits easily)
- **Training time**: ~1-2 hours

---

## MCP Server Implementation

### Option A: Single Python MCP Server (Recommended)

One MCP server exposing all 5 tools, deployed as a Container App (reuse existing CosmosDB MCP infra pattern).

```
sdrf-tools-mcp/
  server.py           # FastMCP server with 5 tools
  ontology/
    psi_ms.db          # SQLite: parsed PSI-MS OBO (~5000 terms)
    unimod.db          # SQLite: parsed UNIMOD XML (~2000 mods)
    gold_sdrf.db       # SQLite: parsed 374 gold SDRFs (value index)
  Dockerfile
  requirements.txt
```

**Tech stack**:
- `mcp` Python SDK (Streamable HTTP transport)
- SQLite for ontology data (fast, no external dependencies)
- `requests` for PRIDE API and EuropePMC calls
- Deploy as Azure Container App (same pattern as CosmosDB MCP toolkit)

### Tool Implementation Details

**`psi_ms.db` schema**:
```sql
CREATE TABLE terms (
    accession TEXT PRIMARY KEY,  -- e.g., "MS:1001742"
    name TEXT,                   -- e.g., "LTQ Orbitrap Velos"
    category TEXT,               -- "instrument", "fragmentation", "analyzer", etc.
    parent TEXT,                 -- parent accession
    synonyms TEXT                -- JSON array of synonyms
);
CREATE INDEX idx_name ON terms(name COLLATE NOCASE);
CREATE INDEX idx_category ON terms(category);
```

**`unimod.db` schema**:
```sql
CREATE TABLE modifications (
    accession TEXT PRIMARY KEY,  -- e.g., "UNIMOD:21"
    name TEXT,                   -- e.g., "Phospho"
    full_name TEXT,              -- e.g., "Phosphorylation"
    targets TEXT,                -- e.g., "S,T,Y"
    mod_type TEXT,               -- "Fixed" or "Variable"
    mono_mass REAL,
    formatted TEXT               -- pre-built SDRF string
);
CREATE INDEX idx_mod_name ON modifications(name COLLATE NOCASE);
```

**`gold_sdrf.db` schema**:
```sql
CREATE TABLE values (
    id INTEGER PRIMARY KEY,
    column_name TEXT,            -- e.g., "Comment[Instrument]"
    value TEXT,                  -- e.g., "NT=Q Exactive HF;AC=MS:1002523"
    frequency INTEGER,          -- how many times in gold corpus
    pxd TEXT                     -- source PXD
);
CREATE INDEX idx_col_val ON values(column_name, value);
```

### Option B: Reuse CosmosDB MCP Toolkit

Store ontology data in CosmosDB containers alongside the existing ICH guidelines data. Use `vector_search` for fuzzy matching and `text_search` for exact lookups.

**Pros**: Reuses existing infrastructure, no new deployment needed.
**Cons**: Slower than SQLite, overkill for small datasets, adds latency per tool call.

**Recommendation**: Option A (dedicated lightweight server). The ontology data is tiny (<10MB total) and latency matters for agentic loops.

---

## Inference Pipeline

```python
async def process_pxd(pxd: str, paper_text: str) -> dict:
    """Agentic extraction for one PXD."""

    # 1. Model reads paper + decides first tool calls
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract SDRF for {pxd}.\n\n{paper_text}"}
    ]

    # 2. Agentic loop (max 5 iterations)
    for _ in range(5):
        response = model.generate(messages, tools=MCP_TOOLS)

        if response.has_tool_calls:
            # Execute tool calls via MCP
            for call in response.tool_calls:
                result = await mcp_client.call_tool(call.name, call.arguments)
                messages.append({"role": "tool", "content": result})
        else:
            # Model produced final JSON output
            return parse_json(response.content)

    # 3. Fallback: return defaults if model didn't converge
    return DEFAULT_VALUES
```

**Expected tool calls per PXD**: 3-6
- 1x `pride_lookup` (always)
- 1x `ms_ontology_lookup` for instrument (always)
- 1-3x `unimod_lookup` for modifications (per unique mod)
- 0-1x `sdrf_format_reference` for format validation

**Latency**: ~5-10s per PXD (7B model inference + tool calls)

---

## Serving Infrastructure

### Model Serving

- **RunPod Serverless** with SGLang (same pattern as Mixtral deployment)
- **GPU**: 1x A100 40GB or H100 80GB (7B model fits in ~14GB with bf16)
- **Endpoint**: OpenAI-compatible API at `https://api.runpod.ai/v2/{endpoint_id}/openai/v1`
- **Model name**: `default`
- **Context**: 8192 tokens
- **Tool calling**: Native Qwen 2.5 tool calling format

### MCP Server Serving

- **Azure Container App** (same deployment pattern as CosmosDB MCP toolkit)
- **Region**: East US 2 (colocate with Foundry endpoint) or Switzerland North (colocate with CosmosDB)
- **Transport**: Streamable HTTP at `/mcp`
- **Auth**: API key (simple, no EasyAuth complexity)

### Alternative: All-Local

For development/testing, run everything locally:
- Model: `ollama run qwen2.5:7b-instruct` or vLLM on local GPU
- MCP: `python server.py` (localhost:8080)
- No cloud dependencies needed

---

## Implementation Status

### Phase 1: Data & Ontologies -- COMPLETE

| Artifact | Records | Source |
|----------|---------|--------|
| `psi_ms.db` | 600 terms (493 instruments, 23 fragmentation, 12 analysers, 49 ionisation, 23 cleavage agents) | PSI-MS OBO |
| `unimod.db` | 1,560 modifications with formatted SDRF strings | UNIMOD XML |
| `gold_sdrf.db` | 5,965 unique (column, value) pairs from 281 PXDs | 103 training + 298 bigbio SDRFs |

Built by `build_databases.py` which downloads source files and parses them into SQLite.

### Phase 2: MCP Server -- COMPLETE

All 5 tools implemented in `server.py` and validated via `test_tools.py`:
- `pride_lookup` -- returns instruments with MS accessions, organisms, label type detection
- `ms_ontology_lookup` -- exact name/synonym/fuzzy matching + gold corpus cross-reference
- `unimod_lookup` -- common target conventions (C for Carbamidomethyl, M for Oxidation) + gold format reference
- `paper_fetch` -- local files, EuropePMC full text (JATS XML parsing), abstract fallback
- `sdrf_format_reference` -- exact/substring/fuzzy matching with frequency-based recommendations

### Phase 3: Training Data Generation -- COMPLETE

103 tool-calling traces generated from training SDRFs with real tool responses:
- 1,391 total messages, 541 tool calls
- Average 13.5 messages, 5.3 tool calls per example
- Output: `training_data/sdrf_finetune.jsonl` (ChatML format, ready for fine-tuning)

### Phase 4: Fine-Tuning -- PENDING

Target: Qwen 2.5-7B-Instruct with LoRA (r=32, alpha=64, 3-5 epochs)

### Phase 5: Integration -- COMPLETE

Inference pipeline (`inference_pipeline.py`) supports:
- GPT-4.1 via Azure OpenAI (immediate use, no fine-tuning needed)
- RunPod endpoint (for fine-tuned model)
- Local model via OpenAI-compatible API

### Phase 6: Deployment -- PENDING

Dockerfile provided for containerised MCP server deployment.

---

## Expected Performance

| Component | v15 (GPT-4.1) | Agentic (fine-tuned + tools) | Why |
|---|---|---|---|
| Instrument | 0.533 | **0.90+** | ms_ontology_lookup eliminates format guessing; model cross-references PRIDE vs paper |
| Modifications | 0.365 | **0.80+** | unimod_lookup gives exact UNIMOD strings |
| Fragmentation | 0.633 | **0.85+** | ms_ontology_lookup for AC codes |
| Tolerance | 0.600 | **0.75+** | sdrf_format_reference for exact format |
| CollisionEnergy | 0.667 | **0.80+** | sdrf_format_reference normalizes format |
| Label | 0.840 | **0.90+** | Model learned from 374 examples |
| Biology columns | 0.00-0.20 | **0.50-0.70** | Fine-tuned model is calibrated on real extraction patterns |
| "not available" calibration | Poor | **Good** | Model learns when to use it from 374 gold examples |
| **Overall F1** | **0.40** | **0.65-0.80** | |

### Risk Factors

1. **374 training examples may be insufficient** for the model to generalize to unseen PXDs. Mitigation: aggressive data augmentation (rephrase papers, vary tool call order).

2. **Tool-calling training format** may not transfer perfectly to inference. Mitigation: use Qwen's native tool-calling format which is well-tested.

3. **Bigbio papers may be hard to fetch** (some behind paywalls, some very old). Mitigation: use abstracts when full text unavailable.

4. **Latency** from multiple tool calls per PXD. Mitigation: batch tool calls where possible, cache PRIDE responses.

---

## Cost Estimate

| Component | Cost |
|---|---|
| RunPod fine-tuning (1x H100, 2 hours) | ~$8 |
| RunPod serverless serving (15 PXDs x 10s) | ~$1 |
| Azure Container App (MCP server) | ~$5/month |
| PRIDE API | Free |
| EuropePMC API | Free |
| **Total per submission** | **~$14** |

Compared to GPT-4.1 approach (~$2-5 per run in API costs), this is comparable but produces much better results.
