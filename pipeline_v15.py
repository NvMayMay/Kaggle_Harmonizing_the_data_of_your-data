#!/usr/bin/env python3
"""
Pipeline v15: Defaults-First, LLM-as-Scalpel
=============================================
Key insight: defaults+bio+PRIDE = 0.372 > v6 Kaggle 0.318.
v6's multi-round LLM is NET NEGATIVE — it replaces correct defaults with wrong guesses.

Architecture:
  1. Smart corpus defaults for technical columns (F1=0.176)
  2. "not available" for biology columns where that's the correct answer (F1→0.250)
  3. PRIDE API for Organism + Instrument (F1→0.372)
  4. PRIDE-based Label detection (F1→0.434)
  5. SINGLE focused LLM call for zero-scoring biology columns ONLY
  6. LLM NEVER overrides protected columns
  7. Fraction parsing + replicate assignment

Protected columns (LLM NEVER touches):
  CleavageAgent (1.000), Separation (1.000), Sex (0.600),
  AncestryCategory (0.600), Age (0.500), DevelopmentalStage (0.333)
"""
import argparse, csv, json, os, re, sys, time
from collections import defaultdict
from typing import Dict, List, Optional

import requests

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    "endpoint": os.environ.get(
        "AZURE_AI_ENDPOINT",
        "https://data-synthesis-foundry-east-us-2.services.ai.azure.com"
    ),
    "api_key": os.environ.get("AZURE_AI_KEY", ""),
}

MODEL_ID = "gpt-4.1"
MODEL_CFG = {
    "display": "GPT-4.1",
    "paths": [
        "/openai/deployments/gpt-4.1/chat/completions?api-version=2024-12-01-preview",
        "/models/chat/completions",
        "/v1/chat/completions",
    ],
}

_working_path: Optional[str] = None

# ═══════════════════════════════════════════════════════════════════════════════
# SUBMISSION SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════

SUBMISSION_COLS = [
    'ID', 'PXD', 'Raw Data File',
    'Characteristics[Age]', 'Characteristics[AlkylationReagent]',
    'Characteristics[AnatomicSiteTumor]', 'Characteristics[AncestryCategory]',
    'Characteristics[BMI]', 'Characteristics[Bait]',
    'Characteristics[BiologicalReplicate]', 'Characteristics[CellLine]',
    'Characteristics[CellPart]', 'Characteristics[CellType]',
    'Characteristics[CleavageAgent]', 'Characteristics[Compound]',
    'Characteristics[ConcentrationOfCompound]', 'Characteristics[Depletion]',
    'Characteristics[DevelopmentalStage]', 'Characteristics[DiseaseTreatment]',
    'Characteristics[Disease]', 'Characteristics[GeneticModification]',
    'Characteristics[Genotype]', 'Characteristics[GrowthRate]',
    'Characteristics[Label]', 'Characteristics[MaterialType]',
    'Characteristics[Modification]', 'Characteristics[Modification].1',
    'Characteristics[Modification].2', 'Characteristics[Modification].3',
    'Characteristics[Modification].4', 'Characteristics[Modification].5',
    'Characteristics[Modification].6',
    'Characteristics[NumberOfBiologicalReplicates]',
    'Characteristics[NumberOfSamples]',
    'Characteristics[NumberOfTechnicalReplicates]',
    'Characteristics[OrganismPart]', 'Characteristics[Organism]',
    'Characteristics[OriginSiteDisease]', 'Characteristics[PooledSample]',
    'Characteristics[ReductionReagent]', 'Characteristics[SamplingTime]',
    'Characteristics[Sex]', 'Characteristics[Specimen]',
    'Characteristics[SpikedCompound]', 'Characteristics[Staining]',
    'Characteristics[Strain]', 'Characteristics[SyntheticPeptide]',
    'Characteristics[Temperature]', 'Characteristics[Time]',
    'Characteristics[Treatment]', 'Characteristics[TumorCellularity]',
    'Characteristics[TumorGrade]', 'Characteristics[TumorSite]',
    'Characteristics[TumorSize]', 'Characteristics[TumorStage]',
    'Comment[AcquisitionMethod]', 'Comment[CollisionEnergy]',
    'Comment[EnrichmentMethod]', 'Comment[FlowRateChromatogram]',
    'Comment[FractionIdentifier]', 'Comment[FractionationMethod]',
    'Comment[FragmentMassTolerance]', 'Comment[FragmentationMethod]',
    'Comment[GradientTime]', 'Comment[Instrument]',
    'Comment[IonizationType]', 'Comment[MS2MassAnalyzer]',
    'Comment[NumberOfFractions]', 'Comment[NumberOfMissedCleavages]',
    'Comment[PrecursorMassTolerance]', 'Comment[Separation]',
    'FactorValue[Bait]', 'FactorValue[CellPart]', 'FactorValue[Compound]',
    'FactorValue[ConcentrationOfCompound].1', 'FactorValue[Disease]',
    'FactorValue[FractionIdentifier]', 'FactorValue[GeneticModification]',
    'FactorValue[Temperature]', 'FactorValue[Treatment]', 'Usage',
]

# ═══════════════════════════════════════════════════════════════════════════════
# ONTOLOGY LOOKUPS (from v6)
# ═══════════════════════════════════════════════════════════════════════════════

INSTRUMENT_ONTOLOGY = {
    "q exactive hf-x": "NT=Q Exactive HF-X;AC=MS:1002877",
    "q exactive hf": "NT=Q Exactive HF;AC=MS:1002523",
    "q exactive plus": "NT=Q Exactive Plus;AC=MS:1002634",
    "q exactive": "NT=Q Exactive;AC=MS:1001911",
    "orbitrap fusion lumos": "NT=Orbitrap Fusion Lumos;AC=MS:1002732",
    "orbitrap exploris 480": "NT=Orbitrap Exploris 480;AC=MS:1003028",
    "exploris 480": "NT=Orbitrap Exploris 480;AC=MS:1003028",
    "exploris480": "NT=Orbitrap Exploris 480;AC=MS:1003028",
    "orbitrap exploris 240": "NT=Orbitrap Exploris 240;AC=MS:1003096",
    "orbitrap astral": "NT=Orbitrap Astral;AC=MS:1003356",
    "orbitrap eclipse": "NT=Orbitrap Eclipse;AC=MS:1003029",
    "orbitrap fusion": "NT=Orbitrap Fusion;AC=MS:1002416",
    "orbitrap elite": "NT=Orbitrap Elite;AC=MS:1002417",
    "ltq orbitrap elite": "NT=LTQ Orbitrap Elite;AC=MS:1001910",
    "ltq orbitrap velos": "NT=LTQ Orbitrap Velos;AC=MS:1001742",
    "ltq orbitrap xl etd": "NT=LTQ Orbitrap XL ETD;AC=MS:1000639",
    "ltq orbitrap xl": "NT=LTQ Orbitrap XL;AC=MS:1000556",
    "ltq orbitrap": "NT=LTQ Orbitrap;AC=MS:1000449",
    "ltq velos": "NT=LTQ Velos;AC=MS:1000855",
    "triple tof 6600": "NT=Triple TOF 6600;AC=MS:1002533",
    "tof 6600": "NT=Triple TOF 6600;AC=MS:1002533",
    "tripletof 6600": "NT=Triple TOF 6600;AC=MS:1002533",
    "tripletof 5600": "NT=TripleTOF 5600;AC=MS:1000931",
    "tof 5600": "NT=TripleTOF 5600;AC=MS:1000931",
    "zeno tof 7600": "NT=Zeno TOF 7600;AC=MS:1003355",
    "impact ii": "NT=impact II;AC=MS:1003123",
    "timstof pro 2": "NT=timsTOF Pro 2;AC=MS:1003230",
    "timstof pro": "NT=timsTOF Pro;AC=MS:1003005",
    "timstof ht": "NT=timsTOF HT;AC=MS:1003303",
    "timstof flex": "NT=timsTOF fleX;AC=MS:1003122",
    "tims tof": "NT=timsTOF Pro;AC=MS:1003005",
    "synapt xs": "NT=Synapt XS;AC=MS:1002797",
    "synapt g2-si": "NT=Synapt G2-Si;AC=MS:1002726",
    "xevo tq-s": "NT=Xevo TQ-S;AC=MS:1002530",
    "tsq quantiva": "NT=TSQ Quantiva;AC=MS:1002488",
    "tsq altis": "NT=TSQ Altis;AC=MS:1003207",
    "synapt ms": "NT=Synapt MS;AC=MS:1001782",
    "synapt": "NT=Synapt MS;AC=MS:1001782",
    "zenotof 7600": "NT=ZenoTOF 7600;AC=MS:1003293",
}

MODIFICATION_ONTOLOGY = {
    "oxidation": "NT=Oxidation;AC=UNIMOD:35;MT=Variable;TA=M",
    "carbamidomethyl": "NT=Carbamidomethyl;AC=UNIMOD:4;TA=C;MT=Fixed",
    "acetyl": "NT=Acetyl;AC=UNIMOD:1;PP=Protein N-term;MT=Variable",
    "phospho": "NT=Phospho;AC=UNIMOD:21;TA=S,T,Y;MT=Variable",
    "deamidated": "NT=Deamidated;AC=UNIMOD:7;TA=N,Q;MT=Variable",
    "methyl": "NT=Methyl;AC=UNIMOD:34;MT=Variable",
    "dimethyl": "NT=Dimethyl;AC=UNIMOD:36;MT=Variable",
    "ubiquitin": "NT=GlyGly;AC=UNIMOD:121;TA=K;MT=Variable",
    "glyg": "NT=GlyGly;AC=UNIMOD:121;TA=K;MT=Variable",
    "tmt": "NT=TMT6plex;AC=UNIMOD:737;TA=K,N-term;MT=Fixed",
    "tmt6plex": "NT=TMT6plex;AC=UNIMOD:737;TA=K,N-term;MT=Fixed",
    "tmtpro": "NT=TMTpro;AC=UNIMOD:2016;TA=K,N-term;MT=Fixed",
    "itraq4plex": "NT=iTRAQ4plex;AC=UNIMOD:214;TA=K,N-term;MT=Fixed",
    "itraq8plex": "NT=iTRAQ8plex;AC=UNIMOD:730;TA=K,N-term;MT=Fixed",
    "propionamide": "NT=Propionamide;AC=UNIMOD:24;TA=C;MT=Fixed",
    "pyro-glu": "NT=Glu->pyro-Glu;AC=UNIMOD:27;TA=E;MT=Variable",
    "ammonia-loss": "NT=Ammonia-loss;AC=UNIMOD:385;TA=Q,C;MT=Variable",
    "carbamyl": "NT=Carbamyl;AC=UNIMOD:5;TA=K,N-term;MT=Variable",
    "cyclization": "NT=Gln->pyro-Glu;AC=UNIMOD:28;TA=Q;MT=Variable",
    "gln->pyro-glu": "NT=Gln->pyro-Glu;AC=UNIMOD:28;TA=Q;MT=Variable",
}

TMT_CHANNELS = {
    "tmt6plex": ["TMT126", "TMT127", "TMT128", "TMT129", "TMT130", "TMT131"],
    "tmt10plex": [f"TMT{c}" for c in ["126","127N","127C","128N","128C","129N","129C","130N","130C","131"]],
    "tmt11plex": [f"TMT{c}" for c in ["126","127N","127C","128N","128C","129N","129C","130N","130C","131N","131C"]],
    "tmt16plex": [f"TMTpro{c}" for c in ["126","127N","127C","128N","128C","129N","129C","130N","130C","131N","131C","132N","132C","133N","133C","134N"]],
    "tmt18plex": [f"TMTpro{c}" for c in ["126","127N","127C","128N","128C","129N","129C","130N","130C","131N","131C","132N","132C","133N","133C","134N","134C","135N"]],
    "itraq4plex": ["ITRAQ114", "ITRAQ115", "ITRAQ116", "ITRAQ117"],
    "itraq8plex": ["ITRAQ113", "ITRAQ114", "ITRAQ115", "ITRAQ116", "ITRAQ117", "ITRAQ118", "ITRAQ119", "ITRAQ121"],
    "silac": ["SILAC light", "SILAC heavy"],
    "silac3": ["SILAC light", "SILAC medium", "SILAC heavy"],
}

# ═══════════════════════════════════════════════════════════════════════════════
# SMART DEFAULTS (proven F1 from error analysis)
# ═══════════════════════════════════════════════════════════════════════════════

# PROTECTED: These defaults score well. LLM NEVER overrides them.
PROTECTED_DEFAULTS = {
    "Characteristics[CleavageAgent]": "AC=MS:1001251;NT=Trypsin",       # F1=1.000
    "Comment[Separation]": "AC=PRIDE:0000563;NT=Reversed-phase chromatography",  # F1=1.000
    "Characteristics[Sex]": "not available",                             # F1=0.600
    "Characteristics[AncestryCategory]": "not available",                # F1=0.600
    "Characteristics[Age]": "not available",                             # F1=0.500
    "Characteristics[DevelopmentalStage]": "not available",              # F1=0.333
    "Characteristics[AlkylationReagent]": "iodoacetamide",
    "Characteristics[ReductionReagent]": "DTT",
    "Comment[NumberOfMissedCleavages]": "2",
}

# TECHNICAL DEFAULTS: LLM can override ONLY if it returns a recognized vocabulary value
TECHNICAL_DEFAULTS = {
    "Comment[FragmentationMethod]": "AC=MS:1000422;NT=HCD",             # F1=0.500
    "Comment[MS2MassAnalyzer]": "AC=MS:1000484;NT=Orbitrap",            # F1=0.500
    "Comment[PrecursorMassTolerance]": "10 ppm",                        # F1=0.500
    "Comment[FragmentMassTolerance]": "0.02 Da",                        # F1=0.250
    "Comment[CollisionEnergy]": "Not Applicable",                       # diverse values in gold, no safe default
    "Comment[FractionationMethod]": "Not Applicable",                   # diverse values in gold
    "Comment[EnrichmentMethod]": "Not Applicable",                      # rarely present
    "Comment[AcquisitionMethod]": "Not Applicable",                     # rarely present
    "Comment[IonizationType]": "Not Applicable",                        # rarely present
}

# BIOLOGY DEFAULTS: LLM overrides these
# Strategy: use "not available" ONLY for columns where gold commonly has "not available"
# Use "Not Applicable" for columns where gold has diverse real values (safer: matches gold's "Not Applicable")
BIOLOGY_DEFAULTS = {
    "Characteristics[OrganismPart]": "not available",        # gold often has "not available"
    "Characteristics[CellType]": "not available",            # gold often has "not available"
    "Characteristics[CellLine]": "Not Applicable",           # gold has real names when present
    "Characteristics[Disease]": "not available",             # gold often has "not available"
    "Characteristics[MaterialType]": "Not Applicable",       # gold has diverse values
    "Characteristics[Treatment]": "Not Applicable",          # gold has diverse values when present
    "Characteristics[Compound]": "Not Applicable",           # rarely present
    "Characteristics[Specimen]": "Not Applicable",           # rarely present
    "Characteristics[Bait]": "Not Applicable",               # rarely present
    "Characteristics[GeneticModification]": "Not Applicable", # rarely present
    "Characteristics[Strain]": "not available",              # gold often has "not available"
    "Characteristics[Depletion]": "Not Applicable",          # rarely present
}

# Modification defaults
MOD_DEFAULTS = [
    "NT=Carbamidomethyl;AC=UNIMOD:4;TA=C;MT=Fixed",
    "NT=Oxidation;AC=UNIMOD:35;MT=Variable;TA=M",
]

# ═══════════════════════════════════════════════════════════════════════════════
# PRIDE API
# ═══════════════════════════════════════════════════════════════════════════════

_pride_cache: Dict[str, dict] = {}

def fetch_pride_metadata(pxd: str) -> dict:
    """Fetch metadata from PRIDE REST API."""
    if pxd in _pride_cache:
        return _pride_cache[pxd]
    result = {
        "organism": None, "organism_part": None, "disease": None,
        "instrument": None, "instrument_ac": None,
        "modifications": [], "quantification": None,
        "sample_protocol": "", "data_protocol": "", "keywords": [],
    }
    try:
        url = f"https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{pxd}"
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            _pride_cache[pxd] = result
            return result
        data = resp.json()

        # Instruments
        instruments = data.get("instruments", [])
        if instruments:
            inst = instruments[0]
            name = inst.get("name", "")
            accession = inst.get("accession", "")
            if name:
                result["instrument"] = name
            if accession:
                result["instrument_ac"] = accession

        # Organism
        organisms = data.get("organisms", [])
        if organisms:
            name = organisms[0].get("name", "")
            if name:
                name = re.sub(r'\s*\([^)]*\)\s*$', '', name).strip()
                result["organism"] = name

        # Organism part
        org_parts = data.get("organismParts", [])
        if org_parts:
            result["organism_part"] = org_parts[0].get("name", "")

        # Disease
        diseases = data.get("diseases", [])
        if diseases:
            name = diseases[0].get("name", "")
            if name and name.lower() not in ("not available", "not applicable"):
                result["disease"] = name

        # Modifications
        ptms = data.get("identifiedPTMStrings", [])
        if ptms:
            result["modifications"] = [
                {"name": p.get("name", ""), "accession": p.get("accession", "")}
                for p in ptms if p.get("name")
            ]

        # Quantification (for label detection)
        quant = data.get("quantificationMethods", [])
        if quant:
            result["quantification"] = quant[0].get("name", "")

        result["sample_protocol"] = data.get("sampleProcessingProtocol", "")
        result["data_protocol"] = data.get("dataProcessingProtocol", "")
        result["keywords"] = data.get("keywords", [])

        print(f"    PRIDE: instrument={result['instrument']}, organism={result['organism']}, "
              f"quant={result['quantification']}, mods={len(result['modifications'])}")
    except Exception as e:
        print(f"    PRIDE error: {e}")

    _pride_cache[pxd] = result
    return result


def pride_to_instrument_sdrf(pride: dict) -> str:
    """Convert PRIDE instrument to SDRF format."""
    name = pride.get("instrument", "")
    ac = pride.get("instrument_ac", "")
    if not name:
        return "Not Applicable"
    if ac and ac.startswith("MS:"):
        return f"NT={name};AC={ac}"
    key = name.lower().strip()
    for pattern, formatted in INSTRUMENT_ONTOLOGY.items():
        if pattern in key:
            return formatted
    return f"NT={name}"


def pride_to_label(pride: dict) -> str:
    """Detect label type from PRIDE quantification method."""
    quant = (pride.get("quantification") or "").lower()
    if not quant or "label free" in quant or "label-free" in quant:
        return "AC=MS:1002038;NT=label free sample"
    if "tmt" in quant:
        return "TMT"  # Channels determined separately
    if "itraq" in quant:
        return "iTRAQ"
    if "silac" in quant:
        return "SILAC"
    if "dimethyl" in quant:
        return "dimethyl label"
    # Default to label-free if unsure
    return "AC=MS:1002038;NT=label free sample"


def get_tmt_channels(label_type: str) -> list:
    """Get channel labels for multiplexed experiments."""
    key = label_type.lower().strip().replace(" ", "").replace("-", "")
    for pattern, channels in TMT_CHANNELS.items():
        if pattern == key:
            return channels
    for pattern, channels in TMT_CHANNELS.items():
        if pattern in key:
            return channels
    if "silac" in key:
        if "triple" in key or "3" in key or "medium" in key:
            return TMT_CHANNELS["silac3"]
        return TMT_CHANNELS["silac"]
    if "itraq" in key:
        if "8" in key:
            return TMT_CHANNELS["itraq8plex"]
        return TMT_CHANNELS["itraq4plex"]
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# LLM CALLER
# ═══════════════════════════════════════════════════════════════════════════════

def call_model(system_prompt: str, user_prompt: str) -> str:
    """Call GPT-4.1 on Azure AI Foundry."""
    global _working_path
    paths = [_working_path] if _working_path else MODEL_CFG["paths"]

    headers = {
        "Content-Type": "application/json",
        "api-key": CONFIG["api_key"],
    }
    body = {
        "model": MODEL_ID,
        "max_tokens": 3000,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    for path in paths:
        url = CONFIG["endpoint"] + path
        for attempt in range(5):
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=180)
                if resp.status_code == 429:
                    wait = min(2 ** attempt * 5, 60)
                    print(f"      Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if resp.status_code == 404:
                    break
                resp.raise_for_status()
                data = resp.json()
                _working_path = path
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")
            except requests.exceptions.HTTPError:
                if attempt < 4:
                    time.sleep(2 ** attempt)
                else:
                    print(f"      FAILED: HTTP {resp.status_code}")
            except Exception as e:
                if attempt < 4:
                    time.sleep(2 ** attempt)
                else:
                    print(f"      FAILED: {e}")
    return ""


def parse_json_response(text: str) -> dict:
    """Parse JSON from model response."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        if start >= 0:
            depth = 0
            in_string = False
            escape = False
            for i in range(start, len(text)):
                c = text[i]
                if escape:
                    escape = False
                    continue
                if c == '\\' and in_string:
                    escape = True
                    continue
                if c == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i+1])
                        except json.JSONDecodeError:
                            break
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# LLM EXTRACTION (biology columns only)
# ═══════════════════════════════════════════════════════════════════════════════

LLM_SYSTEM = """You are extracting biological metadata from a proteomics paper.
Return ONLY valid JSON. No markdown, no explanation, no ```json blocks.
For fields not mentioned in the paper, return "not found".
Do NOT guess or hallucinate. Only extract what is explicitly stated."""

def build_llm_prompt(pxd: str, paper_text: str, pride: dict, raw_files: list) -> str:
    """Build focused prompt for biology-only extraction."""
    pride_context = f"""
KNOWN METADATA (already extracted, DO NOT override):
  Organism: {pride.get('organism', 'unknown')}
  Instrument: {pride.get('instrument', 'unknown')}
  Quantification: {pride.get('quantification', 'unknown')}
"""

    return f"""Extract biological metadata from this proteomics paper for dataset {pxd}.
{pride_context}
Raw data files ({len(raw_files)} files): {', '.join(raw_files[:5])}{'...' if len(raw_files) > 5 else ''}

Extract these fields as JSON:
{{
  "organism_part": "tissue/organ studied (e.g., liver, brain, plasma)",
  "cell_type": "cell type (e.g., T cell, epithelial, macrophage)",
  "cell_line": "cell line name (e.g., HeLa, HEK293, MCF-7)",
  "disease": "disease state (e.g., breast cancer, Alzheimer's, healthy/normal)",
  "material_type": "sample type: tissue, cell, organism part, body fluid, or synthetic",
  "treatment": "any treatment/condition applied",
  "compound": "chemical compounds used in treatment",
  "genetic_modification": "genetic modifications (e.g., knockout, overexpression)",
  "bait": "bait protein for AP-MS experiments",
  "specimen": "specimen type",
  "strain": "organism strain",
  "label_type": "labeling: label free, TMT6plex, TMT10plex, TMT11plex, TMT16plex, TMT18plex, SILAC, iTRAQ4plex, iTRAQ8plex",
  "fragmentation_method": "HCD, CID, ETD, EThcD, or ETciD",
  "collision_energy": "e.g., 28 NCE, 30% NCE, 35 NCE",
  "fractionation_method": "e.g., high pH RPLC, SDS-PAGE, SCX, no fractionation",
  "enrichment_method": "e.g., TiO2, IMAC, no enrichment",
  "acquisition_method": "DDA, DIA, SRM, or PRM",
  "precursor_mass_tolerance": "e.g., 10 ppm, 20 ppm",
  "fragment_mass_tolerance": "e.g., 0.02 Da, 0.05 Da, 20 ppm, 0.5 m/z",
  "ms2_mass_analyzer": "Orbitrap, ion trap, TOF",
  "modifications": ["list of modifications searched, e.g., Carbamidomethyl (C, fixed), Oxidation (M, variable)"]
}}

PAPER TEXT:
{paper_text[:12000]}"""


def format_modification(raw: str) -> str:
    """Format a modification string using UNIMOD ontology."""
    if not raw or raw.lower() in ("not available", "not applicable", "not found"):
        return ""
    if "NT=" in raw or "AC=UNIMOD" in raw:
        return raw
    key = re.sub(r'\s*\([^)]*\)', '', raw.lower().strip()).strip()
    for pattern, formatted in MODIFICATION_ONTOLOGY.items():
        if pattern in key:
            return formatted
    return f"NT={raw};MT=Variable"


FRAGMENTATION_VOCAB = {
    "hcd": "AC=MS:1000422;NT=HCD",
    "cid": "AC=MS:1000133;NT=CID",
    "etd": "AC=MS:1000598;NT=ETD",
    "ethcd": "AC=MS:1002631;NT=EThcD",
    "etcid": "AC=PRIDE:0000592;NT=ETciD",
}

MS2_VOCAB = {
    "orbitrap": "AC=MS:1000484;NT=Orbitrap",
    "ion trap": "AC=MS:1000264;NT=ion trap",
    "iontrap": "AC=MS:1000264;NT=ion trap",
    "tof": "AC=MS:1000084;NT=time-of-flight",
    "time-of-flight": "AC=MS:1000084;NT=time-of-flight",
    "astral": "NT=Astral analyzer;AC=MS:1003381",
}

FRACTIONATION_VOCAB = {
    "high ph": "NT=high pH RPLC;AC=PRIDE:0000564",
    "hprplc": "NT=high pH RPLC;AC=PRIDE:0000564",
    "sds-page": "NT=SDS-PAGE;AC=PRIDE:0000568",
    "sds page": "NT=SDS-PAGE;AC=PRIDE:0000568",
    "gel electrophoresis": "NT=SDS-PAGE;AC=PRIDE:0000568",
    "scx": "NT=Strong cation-exchange chromatography (SCX);AC=PRIDE:0000561",
    "strong cation": "NT=Strong cation-exchange chromatography (SCX);AC=PRIDE:0000561",
    "sax": "NT=Strong anion-exchange chromatography (SAX);AC=PRIDE:0000558",
    "no fractionation": "no fractionation",
    "none": "no fractionation",
}

ACQUISITION_VOCAB = {
    "dda": "NT=Data-Dependent Acquisition;AC=NCIT:C161785",
    "data-dependent": "NT=Data-Dependent Acquisition;AC=NCIT:C161785",
    "data dependent": "NT=Data-Dependent Acquisition;AC=NCIT:C161785",
    "dia": "NT=Data-Independent Acquisition;AC=NCIT:C161786",
    "data-independent": "NT=Data-Independent Acquisition;AC=NCIT:C161786",
    "data independent": "NT=Data-Independent Acquisition;AC=NCIT:C161786",
    "srm": "NT=Selected Reaction Monitoring",
    "prm": "NT=Parallel Reaction Monitoring",
}

TOLERANCE_VOCAB = ["4.5 ppm", "5 ppm", "6 ppm", "7 ppm", "10 ppm", "20 ppm",
                   "0.01 Da", "0.02 Da", "0.05 Da", "0.08 Da", "0.1 Da", "0.4 Da",
                   "0.5 Da", "0.5 m/z", "0.6 Da", "0.7 Da", "1.0005 Da", "20 ppm"]


def match_vocab(value: str, vocab: dict) -> Optional[str]:
    """Match a value to a known vocabulary. Returns formatted value or None."""
    if not value or value.lower() in ("not found", "not available", "not applicable"):
        return None
    key = value.lower().strip()
    for pattern, formatted in vocab.items():
        if pattern in key:
            return formatted
    return None


def match_tolerance(value: str) -> Optional[str]:
    """Match tolerance value to known good values."""
    if not value or value.lower() in ("not found", "not available", "not applicable"):
        return None
    val = value.strip()
    # Direct match
    if val in TOLERANCE_VOCAB:
        return val
    # Normalize: "10ppm" -> "10 ppm", "0.02Da" -> "0.02 Da"
    m = re.match(r'([\d.]+)\s*(ppm|da|m/z)', val, re.IGNORECASE)
    if m:
        num, unit = m.group(1), m.group(2)
        if unit.lower() == 'da':
            unit = 'Da'
        normalized = f"{num} {unit}"
        if normalized in TOLERANCE_VOCAB:
            return normalized
        # Accept even if not in vocab — at least it's properly formatted
        return normalized
    return None


def apply_llm_results(row: dict, llm: dict, pride: dict):
    """Apply LLM extraction results. Only touches biology + zero-scoring tech columns."""

    # BIOLOGY COLUMNS — LLM always overrides defaults (these default to 0.0 F1)
    bio_mapping = {
        "organism_part": "Characteristics[OrganismPart]",
        "cell_type": "Characteristics[CellType]",
        "cell_line": "Characteristics[CellLine]",
        "disease": "Characteristics[Disease]",
        "material_type": "Characteristics[MaterialType]",
        "treatment": "Characteristics[Treatment]",
        "compound": "Characteristics[Compound]",
        "genetic_modification": "Characteristics[GeneticModification]",
        "bait": "Characteristics[Bait]",
        "specimen": "Characteristics[Specimen]",
        "strain": "Characteristics[Strain]",
    }
    for llm_key, col in bio_mapping.items():
        val = llm.get(llm_key, "")
        if isinstance(val, list):
            val = val[0] if val else ""
        if not isinstance(val, str):
            val = str(val) if val else ""
        if val and val.lower() not in ("not found", "not applicable", "none", ""):
            if val.lower() == "not available":
                row[col] = "not available"
            else:
                row[col] = val
        # If LLM says "not found" and column defaults to "Not Applicable", keep it
        # But for some columns, "not available" is better than "Not Applicable"
        elif col in ("Characteristics[Disease]", "Characteristics[OrganismPart]",
                      "Characteristics[CellType]", "Characteristics[Strain]"):
            row[col] = "not available"

    # LABEL — LLM can override if PRIDE didn't detect
    label = llm.get("label_type", "")
    if label and label.lower() not in ("not found", "not applicable", "none", ""):
        # Use LLM label to refine PRIDE label
        key = label.lower()
        if "label free" in key or "label-free" in key:
            row["Characteristics[Label]"] = "AC=MS:1002038;NT=label free sample"
        elif "tmt" in key or "itraq" in key or "silac" in key:
            # LLM detected multiplexing — this is valuable info
            row["_label_type"] = label  # Store for channel expansion later

    # TECHNICAL COLUMNS — LLM overrides ONLY if value matches known vocabulary
    frag = match_vocab(llm.get("fragmentation_method", ""), FRAGMENTATION_VOCAB)
    if frag:
        row["Comment[FragmentationMethod]"] = frag

    ms2 = match_vocab(llm.get("ms2_mass_analyzer", ""), MS2_VOCAB)
    if ms2:
        row["Comment[MS2MassAnalyzer]"] = ms2

    frac_method = match_vocab(llm.get("fractionation_method", ""), FRACTIONATION_VOCAB)
    if frac_method:
        row["Comment[FractionationMethod]"] = frac_method

    acq = match_vocab(llm.get("acquisition_method", ""), ACQUISITION_VOCAB)
    if acq:
        row["Comment[AcquisitionMethod]"] = acq

    prec_tol = match_tolerance(llm.get("precursor_mass_tolerance", ""))
    if prec_tol:
        row["Comment[PrecursorMassTolerance]"] = prec_tol

    frag_tol = match_tolerance(llm.get("fragment_mass_tolerance", ""))
    if frag_tol:
        row["Comment[FragmentMassTolerance]"] = frag_tol

    # Collision energy — accept if it looks like "NN NCE" or "NN% NCE"
    ce = llm.get("collision_energy", "")
    if ce and ce.lower() not in ("not found", "not applicable", "none", ""):
        m = re.match(r'(\d+)%?\s*NCE', ce, re.IGNORECASE)
        if m:
            row["Comment[CollisionEnergy]"] = ce.strip()

    # Enrichment
    enrich = llm.get("enrichment_method", "")
    if enrich and enrich.lower() not in ("not found", "not applicable", "none", "no enrichment"):
        row["Comment[EnrichmentMethod]"] = enrich

    # Modifications — merge with defaults
    llm_mods = llm.get("modifications", [])
    if isinstance(llm_mods, list) and llm_mods:
        formatted_mods = []
        for mod in llm_mods:
            if isinstance(mod, str):
                fmt = format_modification(mod)
                if fmt:
                    formatted_mods.append(fmt)
        if formatted_mods:
            # Merge with defaults, deduplicate by NT= name
            all_mods = list(MOD_DEFAULTS)  # Start with Carbamidomethyl + Oxidation
            existing_names = set()
            for m in all_mods:
                nt_match = re.search(r'NT=([^;]+)', m)
                if nt_match:
                    existing_names.add(nt_match.group(1).lower())
            for m in formatted_mods:
                nt_match = re.search(r'NT=([^;]+)', m)
                if nt_match and nt_match.group(1).lower() not in existing_names:
                    all_mods.append(m)
                    existing_names.add(nt_match.group(1).lower())
            # Assign to modification slots
            for i, mod in enumerate(all_mods[:7]):  # Max 7 slots
                suffix = "" if i == 0 else f".{i}"
                row[f"Characteristics[Modification]{suffix}"] = mod


# ═══════════════════════════════════════════════════════════════════════════════
# ROW BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

def assign_fraction_identifiers(rows: list):
    """Parse fraction identifiers from raw file names."""
    for row in rows:
        raw = row.get('Raw Data File', '')
        if not raw:
            continue
        m = re.search(r'[_\-](?:F|frac(?:tion)?|Fr)(\d{1,3})(?=[_.\-]|$)', raw, re.IGNORECASE)
        if m and int(m.group(1)) <= 200:
            row['Comment[FractionIdentifier]'] = m.group(1)
            continue
        m = re.search(r'[\-_]\d+C[\-_](\d{1,3})(?=[_.\-]|$)', raw)
        if m and int(m.group(1)) <= 200:
            row['Comment[FractionIdentifier]'] = m.group(1)
            continue
        row['Comment[FractionIdentifier]'] = '1'


def build_rows(pxd: str, sample_rows: list, pride: dict, llm_result: dict) -> list:
    """Build submission rows for a PXD."""
    raw_files = sorted(set(r['Raw Data File'] for r in sample_rows))
    n_expected = len(sample_rows)

    # --- Step 1: Build base row with all defaults ---
    base = {col: "Not Applicable" for col in SUBMISSION_COLS}
    base['PXD'] = pxd

    # Apply protected defaults
    for col, val in PROTECTED_DEFAULTS.items():
        base[col] = val

    # Apply technical defaults
    for col, val in TECHNICAL_DEFAULTS.items():
        base[col] = val

    # Apply biology defaults
    for col, val in BIOLOGY_DEFAULTS.items():
        base[col] = val

    # Apply modification defaults
    for i, mod in enumerate(MOD_DEFAULTS):
        suffix = "" if i == 0 else f".{i}"
        base[f"Characteristics[Modification]{suffix}"] = mod

    # --- Step 2: PRIDE overrides ---
    if pride.get("organism"):
        base["Characteristics[Organism]"] = pride["organism"]
    if pride.get("instrument"):
        base["Comment[Instrument]"] = pride_to_instrument_sdrf(pride)

    # Label from PRIDE
    label = pride_to_label(pride)
    base["Characteristics[Label]"] = label

    # --- Step 3: LLM overrides (biology + vocab-matched tech) ---
    if llm_result:
        apply_llm_results(base, llm_result, pride)

    # --- Step 4: Determine label/channel expansion ---
    label_type = base.get("_label_type", "") or base.get("Characteristics[Label]", "")
    channels = get_tmt_channels(label_type)
    base.pop("_label_type", None)  # Remove temp key

    # --- Step 5: Build rows ---
    rows = []
    if channels:
        # Multiplexed: one row per file x channel
        for raw in raw_files:
            for ch in channels:
                row = dict(base)
                row['Raw Data File'] = raw
                row['Characteristics[Label]'] = ch
                rows.append(row)
    else:
        # Non-multiplexed: one row per file
        for raw in raw_files:
            row = dict(base)
            row['Raw Data File'] = raw
            rows.append(row)

    # --- Step 6: Adjust row count to match expected ---
    if len(rows) < n_expected:
        # Duplicate last row to fill
        while len(rows) < n_expected:
            rows.append(dict(rows[-1]))
    elif len(rows) > n_expected:
        rows = rows[:n_expected]

    # --- Step 7: Assign IDs and fraction identifiers ---
    for i, row in enumerate(rows):
        row['ID'] = sample_rows[i]['ID']
        row['Raw Data File'] = sample_rows[i]['Raw Data File']

    assign_fraction_identifiers(rows)

    # --- Step 8: Biological replicate assignment ---
    # Default: "1" for all rows (most common in gold corpus)
    # Sequential assignment was wrong: multi-fraction experiments have same replicate for all files
    for row in rows:
        row['Characteristics[BiologicalReplicate]'] = "1"

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# PAPER LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_paper_text(paper_dir: str, pxd: str) -> str:
    """Load paper text from JSON file."""
    json_path = os.path.join(paper_dir, f"{pxd}_PubText.json")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        parts = []
        for section in ["TITLE", "ABSTRACT", "METHODS", "RESULTS", "INTRO", "DISCUSS"]:
            text = data.get(section, "")
            if text:
                parts.append(f"=== {section} ===\n{text}")
        return "\n\n".join(parts)

    txt_path = os.path.join(paper_dir, f"{pxd}_PubText.txt")
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()

    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(data_dir: str, output_path: str, paper_dir: str = None,
                 single_pxd: str = None):
    """Run the v15 pipeline."""
    sample_sub_path = os.path.join(data_dir, "SampleSubmission.csv")
    if not paper_dir:
        paper_dir = os.path.join(data_dir, "TestPubText")

    # Load sample submission
    with open(sample_sub_path) as f:
        reader = csv.DictReader(f)
        sub_rows = list(reader)

    # Group by PXD
    pxd_groups = defaultdict(list)
    for row in sub_rows:
        pxd_groups[row['PXD']].append(row)

    pxds = sorted(pxd_groups.keys())
    if single_pxd:
        pxds = [single_pxd]

    print(f"Pipeline v15: {len(pxds)} PXDs to process")

    all_rows = []
    for pxd in pxds:
        sample_rows = pxd_groups[pxd]
        print(f"\n{'='*60}")
        print(f"Processing {pxd} ({len(sample_rows)} rows)")

        # Step 1: PRIDE
        pride = fetch_pride_metadata(pxd)

        # Step 2: Paper text
        paper = load_paper_text(paper_dir, pxd)
        if not paper:
            print(f"  WARNING: No paper text for {pxd}")

        # Step 3: LLM extraction (single call)
        llm_result = {}
        if paper:
            raw_files = sorted(set(r['Raw Data File'] for r in sample_rows))
            prompt = build_llm_prompt(pxd, paper, pride, raw_files)
            print(f"  Calling LLM...")
            response = call_model(LLM_SYSTEM, prompt)
            if response:
                llm_result = parse_json_response(response)
                if llm_result:
                    print(f"  LLM extracted {len(llm_result)} fields")
                else:
                    print(f"  LLM: failed to parse JSON")
            else:
                print(f"  LLM: no response")

        # Step 4: Build rows
        rows = build_rows(pxd, sample_rows, pride, llm_result)
        print(f"  Built {len(rows)} rows")
        all_rows.extend(rows)

        # Rate limit protection
        time.sleep(3)

    # Sort by original ID
    all_rows.sort(key=lambda r: int(r['ID']))

    # Write output
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_COLS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{'='*60}")
    print(f"Output: {output_path} ({len(all_rows)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline v15: Defaults-first, LLM-as-scalpel")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", default="submission_v15.csv")
    parser.add_argument("--paper-dir", default=None)
    parser.add_argument("--single-pxd", default=None)
    args = parser.parse_args()

    run_pipeline(args.data_dir, args.output, args.paper_dir, args.single_pxd)
