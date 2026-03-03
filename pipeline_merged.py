#!/usr/bin/env python3
"""
SDRF Deliberation Pipeline -- LangGraph Edition
================================================
3-round multi-model deliberation for proteomics SDRF metadata extraction.

Round 1: Independent extraction (3 models × 1 call = 3 calls)
Round 2: Cross-pollinated reconciliation with confidence scores (3 calls)
Round 3: Grounded judge reconciliation (1 call, Claude Opus)

Graph:
    extract_r1 -> deliberate_r2 -> judge_r3 -> validate ──┬──► format -> END
                                     ▲                  │
                                     └── refine ◄───────┘

Requirements: pip install langgraph langchain-core requests
"""
import argparse, csv, json, os, re, sys, time
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Tuple, TypedDict

import requests
from langgraph.graph import END, StateGraph

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

MODELS = {
    "claude-opus-4-5": {
        "display": "Claude Opus 4.5",
        "paths": [
            "/anthropic/v1/messages",
            "/models/chat/completions",
        ],
        "format": "anthropic",
    },
    "gpt-4.1": {
        "display": "GPT-4.1",
        "paths": [
            "/openai/deployments/gpt-4.1/chat/completions?api-version=2024-12-01-preview",
            "/models/chat/completions",
            "/v1/chat/completions",
        ],
        "format": "openai",
    },
    "DeepSeek-V3.2": {
        "display": "DeepSeek-V3.2",
        "paths": [
            "/v1/chat/completions",
            "/models/chat/completions",
            "/openai/deployments/DeepSeek-V3.2/chat/completions?api-version=2024-12-01-preview",
        ],
        "format": "openai",
    },
}

TIEBREAKER_MODEL = {
    "gpt-4.1-mini": {
        "display": "GPT-4.1-mini",
        "paths": [
            "/openai/deployments/gpt-4.1-mini/chat/completions?api-version=2024-12-01-preview",
            "/models/chat/completions",
            "/v1/chat/completions",
        ],
        "format": "openai",
    },
}

JUDGE_MODEL_ID = "claude-opus-4-5"

_working_paths: Dict[str, str] = {}
MAX_REFINEMENT_ITERATIONS = 2

# ═══════════════════════════════════════════════════════════════════════════════
# PRIDE API METADATA FETCHER
# ═══════════════════════════════════════════════════════════════════════════════

_pride_cache: Dict[str, dict] = {}

def fetch_pride_metadata(pxd: str) -> dict:
    """Fetch structured metadata from PRIDE REST API for a PXD accession.
    Returns a dict with normalized SDRF-relevant fields."""
    if pxd in _pride_cache:
        return _pride_cache[pxd]

    result = {
        "organism": None,
        "organism_part": None,
        "disease": None,
        "instrument": None,
        "instrument_ac": None,
        "modifications": [],
        "quantification": None,
        "sample_protocol": None,
        "data_protocol": None,
        "keywords": [],
    }

    try:
        url = f"https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{pxd}"
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            print(f"    PRIDE API: HTTP {resp.status_code} for {pxd}")
            _pride_cache[pxd] = result
            return result

        data = resp.json()

        # Instruments — most reliable field
        instruments = data.get("instruments", [])
        if instruments:
            inst = instruments[0]  # primary instrument
            name = inst.get("name", "")
            accession = inst.get("accession", "")
            if name:
                result["instrument"] = name
            if accession:
                result["instrument_ac"] = accession

        # Organism
        organisms = data.get("organisms", [])
        if organisms:
            org = organisms[0]
            name = org.get("name", "")
            if name:
                # Clean up "(Human)" suffix
                name = re.sub(r'\s*\([^)]*\)\s*$', '', name).strip()
                result["organism"] = name

        # Organism part
        org_parts = data.get("organismParts", [])
        if org_parts:
            result["organism_part"] = org_parts[0].get("name", "")

        # Disease
        diseases = data.get("diseases", [])
        if diseases:
            d = diseases[0]
            name = d.get("name", "")
            if name and name.lower() not in ("not available", "not applicable"):
                result["disease"] = name

        # Modifications (partial — no fixed/variable distinction)
        ptms = data.get("identifiedPTMStrings", [])
        if ptms:
            result["modifications"] = [
                {"name": p.get("name", ""), "accession": p.get("accession", "")}
                for p in ptms if p.get("name")
            ]

        # Quantification method
        quant = data.get("quantificationMethods", [])
        if quant:
            result["quantification"] = quant[0].get("name", "")

        # Free-text protocols
        result["sample_protocol"] = data.get("sampleProcessingProtocol", "")
        result["data_protocol"] = data.get("dataProcessingProtocol", "")

        # Keywords
        result["keywords"] = data.get("keywords", [])

        print(f"    PRIDE API: instrument={result['instrument']}, "
              f"organism={result['organism']}, "
              f"mods={len(result['modifications'])}")

    except Exception as e:
        print(f"    PRIDE API error: {e}")

    _pride_cache[pxd] = result
    return result


def pride_instrument_to_sdrf(pride_meta: dict) -> str:
    """Convert PRIDE instrument metadata to SDRF format.
    Uses the accession code from PRIDE for exact ontology matching."""
    name = pride_meta.get("instrument", "")
    ac = pride_meta.get("instrument_ac", "")

    if not name:
        return ""

    # If we have the MS accession from PRIDE, construct NT-first format (matches gold majority)
    if ac and ac.startswith("MS:"):
        return f"NT={name};AC={ac}"

    # Fall back to our ontology lookup
    return format_instrument(name)

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

METADATA_COLS = [c for c in SUBMISSION_COLS if c not in ('ID', 'PXD', 'Raw Data File')]

EXTRACTION_FIELDS = [
    "organism", "organism_part", "cell_type", "cell_line", "disease",
    "sex", "age", "strain", "instrument", "fragmentation_method",
    "cleavage_agent", "label_type", "modifications", "material_type",
    "enrichment_method", "fractionation_method", "alkylation_reagent",
    "reduction_reagent", "precursor_mass_tolerance", "fragment_mass_tolerance",
    "missed_cleavages", "compound", "treatment", "genetic_modification", "bait",
    "acquisition_method", "separation", "ionization_type", "ms2_mass_analyzer",
    "number_of_fractions", "collision_energy", "gradient_time", "flow_rate",
    "specimen", "pooled_sample",
    # Additional fields for completeness
    "developmental_stage", "ancestry_category", "depletion",
    "spiked_compound", "cell_part", "temperature", "genotype",
    "number_of_biological_replicates", "number_of_technical_replicates",
    "number_of_samples", "concentration_of_compound", "sampling_time",
    "staining", "synthetic_peptide", "disease_treatment",
    "factor_value_type", "factor_values",
    "biological_replicate_scheme", "fraction_scheme", "sample_to_file_mapping",
]

CRITICAL_FIELDS = ["organism", "instrument", "cleavage_agent", "label_type"]
IMPORTANT_FIELDS = ["organism_part", "disease", "material_type", "fragmentation_method", "modifications"]

# ── Specialist prompt field splits ──
FIELDS_BIO = [
    "organism", "organism_part", "cell_type", "cell_line", "disease",
    "sex", "age", "strain", "label_type", "material_type",
    "developmental_stage", "ancestry_category", "depletion",
    "spiked_compound", "cell_part", "temperature", "genotype",
    "number_of_biological_replicates", "number_of_technical_replicates",
    "number_of_samples", "concentration_of_compound", "sampling_time",
    "staining", "synthetic_peptide", "disease_treatment",
    "compound", "treatment", "genetic_modification", "bait",
    "specimen", "pooled_sample",
    "factor_value_type", "factor_values",
    "biological_replicate_scheme", "sample_to_file_mapping",
]

FIELDS_ANALYTICAL = [
    "instrument", "fragmentation_method", "cleavage_agent", "modifications",
    "enrichment_method", "fractionation_method", "alkylation_reagent",
    "reduction_reagent", "precursor_mass_tolerance", "fragment_mass_tolerance",
    "missed_cleavages", "acquisition_method", "separation", "ionization_type",
    "ms2_mass_analyzer", "number_of_fractions", "collision_energy",
    "gradient_time", "flow_rate", "fraction_scheme",
]

# ═══════════════════════════════════════════════════════════════════════════════
# ONTOLOGY LOOKUPS
# ═══════════════════════════════════════════════════════════════════════════════

INSTRUMENT_ONTOLOGY = {
    # NT-first format — matches majority of gold standard training SDRFs
    # String similarity between NT-first and AC-first is ~0.48, below the 0.80 clustering threshold
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
    "vanquish neo": "NT=Vanquish Neo UHPLC",
    "echo ms": "NT=Echo MS",
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
    "silac light": "NT=Label:13C(6)15N(2);AC=UNIMOD:259;TA=K;MT=Fixed",
    "silac heavy": "NT=Label:13C(6)15N(4);AC=UNIMOD:267;TA=R;MT=Fixed",
    "cyclization": "NT=Gln->pyro-Glu;AC=UNIMOD:28;TA=Q;MT=Variable",
    "pyro-glutamic": "NT=Gln->pyro-Glu;AC=UNIMOD:28;TA=Q;MT=Variable",
    "gln->pyro-glu": "NT=Gln->pyro-Glu;AC=UNIMOD:28;TA=Q;MT=Variable",
    "succinyl": "NT=Succinyl;AC=UNIMOD:64;TA=K;MT=Variable",
    "citrullination": "NT=Citrullination;AC=UNIMOD:7;TA=R;MT=Variable",
    "crotonyl": "NT=Crotonyl;AC=UNIMOD:1363;TA=K;MT=Variable",
    "hydroxyl": "NT=Hydroxylation;AC=UNIMOD:35;MT=Variable",
    "formyl": "NT=Formyl;AC=UNIMOD:122;PP=Protein N-term;MT=Variable",
    "nitrosyl": "NT=Nitrosylation;AC=UNIMOD:275;TA=C;MT=Variable",
    "sulfo": "NT=Sulfo;AC=UNIMOD:40;TA=S,T,Y;MT=Variable",
    "cysteinyl": "NT=Cysteinylation;AC=UNIMOD:312;TA=C;MT=Variable",
}

FRAGMENTATION_ONTOLOGY = {
    "hcd": "AC=MS:1000422;NT=HCD",
    "cid": "AC=MS:1000133;NT=CID",
    "etd": "AC=MS:1000598;NT=ETD",
    "ethcd": "AC=MS:1002631;NT=EThcD",
    "etcid": "AC=PRIDE:0000592;NT=ETciD",
    "ecd": "AC=MS:1000250;NT=ECD",
}

CLEAVAGE_ONTOLOGY_EXTRA = {
    "glutamyl endopeptidase": "NT=Glutamyl endopeptidase;AC=MS:1001917",
    "v8-de": "AC=MS:1001314;NT=V8-DE",
    "leukocyte elastase": "NT=leukocyte elastase;AC=MS:1001915",
}

MS2_ANALYZER_ONTOLOGY = {
    "orbitrap": "AC=MS:1000484;NT=Orbitrap",
    "ion trap": "AC=MS:1000264;NT=ion trap",
    "iontrap": "AC=MS:1000264;NT=ion trap",
    "tof": "AC=MS:1000084;NT=time-of-flight",
    "time-of-flight": "AC=MS:1000084;NT=time-of-flight",
    "quadrupole": "AC=MS:1000081;NT=quadrupole",
    "astral": "NT=Astral analyzer;AC=MS:1003381",
    "ftms": "AC=MS:1000079;NT=fourier transform ion cyclotron resonance mass spectrometer",
}

FRACTIONATION_ONTOLOGY = {
    "high ph": "NT=high pH RPLC;AC=PRIDE:0000564",
    "hprplc": "NT=high pH RPLC;AC=PRIDE:0000564",
    "hprp": "NT=high pH RPLC;AC=PRIDE:0000564",
    "basic rp": "NT=high pH RPLC;AC=PRIDE:0000564",
    "sds-page": "AC=PRIDE:0000124;NT=Sodium dodecyl sulfate polyacrylamide gel electrophoresis",
    "sds page": "AC=PRIDE:0000124;NT=Sodium dodecyl sulfate polyacrylamide gel electrophoresis",
    "gel electrophoresis": "AC=PRIDE:0000124;NT=Sodium dodecyl sulfate polyacrylamide gel electrophoresis",
    "page": "AC=PRIDE:0000124;NT=Sodium dodecyl sulfate polyacrylamide gel electrophoresis",
    "scx": "NT=Strong cation-exchange chromatography (SCX);AC=PRIDE:0000561",
    "strong cation": "NT=Strong cation-exchange chromatography (SCX);AC=PRIDE:0000561",
    "sax": "NT=Strong anion-exchange chromatography (SAX);AC=PRIDE:0000558",
    "strong anion": "NT=Strong anion-exchange chromatography (SAX);AC=PRIDE:0000558",
    "rp ": "NT=Reversed-phase chromatography (RP);AC=PRIDE:0000563",
    "reversed-phase": "NT=Reversed-phase chromatography (RP);AC=PRIDE:0000563",
    "offgel": "NT=Offgel electrophoresis;AC=PRIDE:0000570",
    "isoelectric focusing": "NT=Isoelectric focusing;AC=PRIDE:0000570",
    "ief": "NT=Isoelectric focusing;AC=PRIDE:0000570",
    "hilic": "NT=Hydrophilic interaction liquid chromatography (HILIC);AC=PRIDE:0000565",
    "no fractionation": "no fractionation",
    "none": "no fractionation",
}

ACQUISITION_ONTOLOGY = {
    "dda": "NT=Data-Dependent Acquisition;AC=NCIT:C161785",
    "data-dependent": "NT=Data-Dependent Acquisition;AC=NCIT:C161785",
    "data dependent": "NT=Data-Dependent Acquisition;AC=NCIT:C161785",
    "dia": "NT=Data-Independent Acquisition;AC=NCIT:C161786",
    "data-independent": "NT=Data-Independent Acquisition;AC=NCIT:C161786",
    "data independent": "NT=Data-Independent Acquisition;AC=NCIT:C161786",
    "srm": "NT=Selected Reaction Monitoring",
    "prm": "NT=Parallel Reaction Monitoring",
}

SEPARATION_ONTOLOGY = {
    "reversed-phase": "AC=PRIDE:0000563;NT=Reversed-phase chromatography",
    "reverse phase": "AC=PRIDE:0000563;NT=Reversed-phase chromatography",
    "rp-hplc": "AC=PRIDE:0000563;NT=Reversed-phase chromatography",
    "rplc": "AC=PRIDE:0000563;NT=Reversed-phase chromatography",
    "hplc": "NT=High-performance liquid chromatography;AC=PRIDE:0000565",
    "uhplc": "NT=High-performance liquid chromatography;AC=PRIDE:0000565",
    "nanolc": "AC=PRIDE:0000563;NT=Reversed-phase chromatography",
    "nano-lc": "AC=PRIDE:0000563;NT=Reversed-phase chromatography",
    "microlc": "NT=MicroLC",
    "capillary lc": "NT=Capillary LC",
}

CLEAVAGE_ONTOLOGY = {
    "trypsin/p": "AC=MS:1001313;NT=Trypsin/P",
    "trypsin": "AC=MS:1001251;NT=Trypsin",
    "lys-c": "AC=MS:1001309;NT=Lys-C",
    "lysc": "AC=MS:1001309;NT=Lys-C",
    "arg-c": "AC=MS:1001303;NT=Arg-C",
    "asp-n": "AC=MS:1001304;NT=Asp-N",
    "glu-c": "AC=MS:1001917;NT=Glu-C",
    "glutamyl endopeptidase": "NT=Glutamyl endopeptidase;AC=MS:1001917",
    "chymotrypsin": "AC=MS:1001306;NT=Chymotrypsin",
    "cnbr": "AC=MS:1001307;NT=CNBr",
    "v8-de": "AC=MS:1001314;NT=V8-DE",
    "pepsin": "AC=MS:1001311;NT=Pepsin",
    "proteinase k": "AC=MS:1001915;NT=Proteinase K",
}

IONIZATION_ONTOLOGY = {
    # More specific patterns first to avoid "esi" matching "nanoesi"
    "nanoesi": "AC=MS:1000398;NT=nanoelectrospray",
    "nano-esi": "AC=MS:1000398;NT=nanoelectrospray",
    "nano-electrospray": "AC=MS:1000398;NT=nanoelectrospray",
    "nanoelectrospray": "AC=MS:1000398;NT=nanoelectrospray",
    "nsi": "AC=MS:1000398;NT=nanoelectrospray",
    "electrospray ionization": "AC=MS:1000073;NT=electrospray ionization",
    "electrospray": "AC=MS:1000073;NT=electrospray ionization",
    "esi": "AC=MS:1000073;NT=electrospray ionization",
    "maldi": "AC=MS:1000075;NT=matrix-assisted laser desorption ionization",
    "apci": "AC=MS:1000070;NT=atmospheric pressure chemical ionization",
}

LABEL_ONTOLOGY = {
    "label free": "AC=MS:1002038;NT=label free sample",
    "label-free": "AC=MS:1002038;NT=label free sample",
    "tmt": "TMT",
    "tmtpro": "TMTpro",
    "itraq": "iTRAQ",
    "silac": "SILAC",
    "dimethyl": "dimethyl label",
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
# GROUNDING CONTEXT (embedded in judge prompt)
# ═══════════════════════════════════════════════════════════════════════════════

TRAINING_EXAMPLE = """Example of correctly formatted SDRF for PXD000070 (phosphoproteomics study):

Row 1:
  Organism: plasmodium falciparum
  OrganismPart: human erythrocytes
  CellType: schizont
  BiologicalReplicate: 1
  Label: AC=MS:1002038;NT=label free sample
  Instrument: AC=MS:1001742;NT=LTQ Orbitrap Velos
  FragmentationMethod: AC=MS:1000598;NT=ETD
  CleavageAgent: AC=MS:1001251;NT=Trypsin
  Modification: NT=Carbamidomethyl;AC=UNIMOD:4;TA=C;MT=Fixed
  Modification.1: NT=Oxidation;AC=UNIMOD:35;TA=M,W,H;MT=Variable
  Modification.2: NT=Acetyl;AC=UNIMOD:1;PP=Protein N-term;MT=Variable
  Modification.3: NT=Phospho;AC=UNIMOD:21;TA=S,T,Y,A;MT=Variable
  MaterialType: tissue
  PrecursorMassTolerance: 20 ppm
  FragmentMassTolerance: 1.0005 Da
  AcquisitionMethod: NT=Data-Dependent Acquisition;AC=NCIT:C161785
  Separation: AC=PRIDE:0000563;NT=Reversed-phase chromatography
  MS2MassAnalyzer: AC=MS:1000484;NT=orbitrap
  EnrichmentMethod: NT=Phospho;AC=PRIDE:0000590
  FractionationMethod: no fractionation
  NumberOfMissedCleavages: 2
  FractionIdentifier: 1
  NumberOfFractions: 1

Key format conventions:
- Instruments MUST use AC=MS:XXXXXXX;NT=Name format
- Modifications MUST use NT=Name;AC=UNIMOD:XX;TA=AminoAcids;MT=Fixed/Variable
- Cleavage agents MUST use AC=MS:XXXXXXX;NT=Name format
- Label: use "AC=MS:1002038;NT=label free sample" for label-free, "TMT126" etc for TMT channels
- MaterialType: use lowercase (tissue, cell, organism part, cell line, lysate)
- AcquisitionMethod: NT=Data-Dependent Acquisition;AC=NCIT:C161785 or NT=Data-Independent Acquisition;AC=NCIT:C161786
- Separation: AC=PRIDE:0000563;NT=Reversed-phase chromatography
- MS2MassAnalyzer: AC=MS:1000484;NT=orbitrap or AC=MS:1000264;NT=ion trap
- FractionationMethod: "no fractionation" or NT=High-pH reversed-phase chromatography (hpHRP);AC=PRIDE:0000564
- "Not Applicable" = use for ALL fields where the paper doesn't mention the value or the field doesn't apply
- One row per raw file for label-free; one row per (raw file × channel) for multiplexed
- factor_value_type: the column name of the primary experimental variable (e.g., "treatment", "disease", "bait")
- factor_values: list of distinct values for that factor across samples"""

SCORING_DESCRIPTION = """Scoring function:
1. For each (PXD, column), unique values from submission and solution are collected
2. Values with NT= prefixes have the NT= value extracted for comparison
3. All unique values are clustered by string similarity (difflib SequenceMatcher, threshold 0.80)
4. Agglomerative clustering groups similar strings into the same cluster ID
5. Precision, recall, and F1 are computed on cluster membership (macro-averaged)
6. Final score = mean F1 across all (PXD, column) pairs

Implications:
- Getting the CANONICAL ontology term right is critical (matches cluster with solution)
- Minor spelling variations within 80% similarity still match, but wrong instruments don't
- "Not Applicable" is used for unknown/inapplicable fields -- columns with only "Not Applicable" are EXCLUDED from scoring (no penalty)
- "not available" is SCORED as a real value -- only use it if you're confident the solution also has "not available"
- Missing a value that the solution has hurts recall
- Including a value the solution doesn't have hurts precision
- Use "Not Applicable" for unknown fields to avoid scoring penalties"""


def build_ontology_reference() -> str:
    """Build a compact ontology reference for the judge."""
    lines = ["Valid ontology terms:"]
    lines.append("\nInstruments:")
    for k, v in INSTRUMENT_ONTOLOGY.items():
        lines.append(f"  {k} -> {v}")
    lines.append("\nModifications:")
    for k, v in MODIFICATION_ONTOLOGY.items():
        lines.append(f"  {k} -> {v}")
    lines.append("\nFragmentation:")
    for k, v in FRAGMENTATION_ONTOLOGY.items():
        lines.append(f"  {k} -> {v}")
    lines.append("\nCleavage agents:")
    for k, v in CLEAVAGE_ONTOLOGY.items():
        lines.append(f"  {k} -> {v}")
    lines.append("\nLabel types:")
    for k, v in TMT_CHANNELS.items():
        lines.append(f"  {k}: {len(v)} channels ({v[0]}...{v[-1]})")
    lines.append("\nAcquisition methods:")
    for k, v in ACQUISITION_ONTOLOGY.items():
        lines.append(f"  {k} -> {v}")
    lines.append("\nMS2 mass analyzers:")
    for k, v in MS2_ANALYZER_ONTOLOGY.items():
        lines.append(f"  {k} -> {v}")
    lines.append("\nFractionation methods:")
    for k, v in FRACTIONATION_ONTOLOGY.items():
        lines.append(f"  {k} -> {v}")
    lines.append("\nIonization types:")
    for k, v in IONIZATION_ONTOLOGY.items():
        lines.append(f"  {k} -> {v}")
    lines.append("\nSeparation:")
    for k, v in SEPARATION_ONTOLOGY.items():
        lines.append(f"  {k} -> {v}")
    return "\n".join(lines)


ONTOLOGY_REFERENCE = build_ontology_reference()

# ═══════════════════════════════════════════════════════════════════════════════
# API LAYER
# ═══════════════════════════════════════════════════════════════════════════════

def call_model(model_id: str, system_prompt: str, user_prompt: str,
               max_tokens: int = 4000, temperature: float = 0.1) -> str:
    """Call a model on Azure AI Foundry. Tries multiple endpoint paths."""
    cfg = MODELS.get(model_id) or TIEBREAKER_MODEL.get(model_id)
    if not cfg:
        raise ValueError(f"Unknown model: {model_id}")

    paths_to_try = [_working_paths[model_id]] if model_id in _working_paths else cfg["paths"]

    headers = {
        "Content-Type": "application/json",
        "api-key": CONFIG["api_key"],
    }

    for path in paths_to_try:
        url = CONFIG["endpoint"] + path

        if cfg["format"] == "anthropic" and "/anthropic/" in path:
            body = {
                "model": model_id,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }
            h = dict(headers)
            h["x-api-key"] = CONFIG["api_key"]
            h["anthropic-version"] = "2023-06-01"
        else:
            body = {
                "model": model_id,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            h = dict(headers)

        for attempt in range(3):
            try:
                resp = requests.post(url, headers=h, json=body, timeout=180)
                if resp.status_code == 429:
                    wait = min(2 ** attempt * 5, 60)
                    print(f"      Rate limited on {cfg['display']}, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if resp.status_code == 404:
                    break
                resp.raise_for_status()
                data = resp.json()
                _working_paths[model_id] = path

                if cfg["format"] == "anthropic" and "/anthropic/" in path:
                    return data.get("content", [{}])[0].get("text", "")
                else:
                    return data.get("choices", [{}])[0].get("message", {}).get("content", "")
            except requests.exceptions.HTTPError:
                if attempt < 2:
                    print(f"      HTTP {resp.status_code} on {cfg['display']}, retrying...")
                    time.sleep(2 ** attempt)
                else:
                    print(f"      FAILED {cfg['display']}: HTTP {resp.status_code}")
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    print(f"      FAILED {cfg['display']}: {type(e).__name__}")
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    print(f"      FAILED {cfg['display']}: {e}")

    print(f"    [FAIL] All paths failed for {cfg['display']}")
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# JSON PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def parse_json_response(text: str) -> dict:
    """Parse JSON from model response using balanced-brace extraction."""
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
    print(f"        WARNING: Could not parse JSON")
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

R1_SYSTEM = """You are an expert bioinformatician specializing in proteomics SDRF metadata extraction.
Return ONLY valid JSON. No markdown, no explanation, no ```json blocks.
Use "Not Applicable" for fields where the value is unknown, not mentioned, or doesn't apply.

CRITICAL FORMAT RULES (these exact formats are required for scoring):
- organism: binomial nomenclature, e.g. "Homo sapiens", "Mus musculus", "Rattus norvegicus"
- instrument: AC=MS:xxx;NT=Name format, e.g. "AC=MS:1002523;NT=Q Exactive HF"
- cleavage_agent: AC=MS:xxx;NT=Name format, e.g. "AC=MS:1001251;NT=Trypsin"
- modifications: list of ALL modifications searched in the experiment, each in NT=Name;AC=UNIMOD:xx;TA=aa;MT=Fixed/Variable format
  - ALWAYS include NT=Carbamidomethyl;AC=UNIMOD:4;TA=C;MT=Fixed if iodoacetamide/IAA alkylation is used
  - ALWAYS include NT=Oxidation;AC=UNIMOD:35;TA=M;MT=Variable (standard in virtually every experiment)
  - Common variable mods to look for: Acetyl (protein N-term), Phospho (S,T,Y), Deamidated (N,Q), GlyGly (K for ubiquitin)
  - List Fixed modifications first, then Variable
  - Extract ALL modifications mentioned in the methods section — typically 3-8 modifications per study
- label_type: "AC=MS:1002038;NT=label free sample" for label-free, or "tmt10plex"/"tmt16plex" for TMT
- material_type: lowercase — "tissue", "cell", "organism part", "cell line", "lysate"
- acquisition_method: "DDA" for data-dependent, "DIA" for data-independent, "SWATH" for SWATH-MS
- ms2_mass_analyzer: "orbitrap", "ion trap", "time-of-flight", "quadrupole" (lowercase)
- ionization_type: "ESI", "nanoESI", "MALDI"
- fragmentation_method: "HCD", "CID", "ETD", "EThcD"
- separation: "Reversed-phase chromatography", "HPLC", "nanoLC"
- developmental_stage: "adult", "embryonic", "neonatal", etc. Use "Not Applicable" if not mentioned.
- ancestry_category: ethnic/racial background if mentioned. Use "Not Applicable" if not mentioned.
- depletion: protein depletion method if mentioned (e.g., "Top-14 depletion", "immunodepletion")
- spiked_compound: spike-in standard if mentioned (e.g., "UPS1", "iRT peptides", "AQUA peptides")
- cell_part: subcellular fraction (e.g., "cytoplasm", "membrane", "nucleus", "mitochondria")
- temperature: experimental temperature if mentioned (e.g., "37 C", "25 C")
- genotype: genetic background/mutant (e.g., "wild-type", "knockout", "C57BL/6")
- number_of_biological_replicates: total number of biological replicates (e.g., "3")
- number_of_technical_replicates: total number of technical replicates (e.g., "2")
- number_of_samples: total number of unique samples (e.g., "6")
- concentration_of_compound: drug/compound concentration if mentioned (e.g., "10 uM")
- sampling_time: time of sample collection if mentioned
- disease_treatment: treatment for disease if mentioned
- staining: staining method if mentioned
- synthetic_peptide: "Yes" if synthetic peptides used, "Not Applicable" otherwise
- factor_value_type: the column name of the primary experimental variable (treatment, disease, bait, compound, genetic_modification, temperature, cell_part, genotype)
- factor_values: list of ALL distinct conditions/groups in the experiment"""


R1_SYSTEM_BIO = """You are an expert bioinformatician specializing in biological sample metadata extraction from proteomics papers.
Return ONLY valid JSON. No markdown, no explanation, no ```json blocks.
Use "Not Applicable" for fields where the value is unknown, not mentioned, or doesn't apply.

Focus on BIOLOGICAL SAMPLE CHARACTERISTICS — the organisms, tissues, cells, and conditions studied.

FORMAT RULES:
- organism: binomial nomenclature, e.g. "Homo sapiens", "Mus musculus"
- material_type: lowercase — "tissue", "cell", "organism part", "cell line", "lysate"
- label_type: "AC=MS:1002038;NT=label free sample" for label-free, or "tmt10plex"/"tmt16plex" for TMT
- developmental_stage: "adult", "embryonic", "neonatal", etc. Use "Not Applicable" if not mentioned.
- ancestry_category: ethnic/racial background if mentioned. Use "Not Applicable" if not mentioned.
- cell_part: subcellular fraction (e.g., "cytoplasm", "membrane", "nucleus")
- genotype: genetic background/mutant (e.g., "wild-type", "knockout", "C57BL/6")
- factor_value_type: the column name of the primary experimental variable (treatment, disease, bait, compound, genetic_modification, temperature)
- factor_values: list of ALL distinct conditions/groups in the experiment
- number_of_biological_replicates: total number of biological replicates (e.g., "3")"""


R1_SYSTEM_ANALYTICAL = """You are an expert mass spectrometrist specializing in analytical method metadata extraction from proteomics papers.
Return ONLY valid JSON. No markdown, no explanation, no ```json blocks.
Use "Not Applicable" for fields where the value is unknown, not mentioned, or doesn't apply.

Focus on ANALYTICAL/INSTRUMENT PARAMETERS — the instruments, methods, and settings used.

FORMAT RULES:
- instrument: AC=MS:xxx;NT=Name format, e.g. "AC=MS:1002523;NT=Q Exactive HF"
- cleavage_agent: AC=MS:xxx;NT=Name format, e.g. "AC=MS:1001251;NT=Trypsin"
- modifications: list of ALL modifications searched, each in NT=Name;AC=UNIMOD:xx;TA=aa;MT=Fixed/Variable format
  - ALWAYS include NT=Carbamidomethyl;AC=UNIMOD:4;TA=C;MT=Fixed if iodoacetamide/IAA alkylation is used
  - ALWAYS include NT=Oxidation;AC=UNIMOD:35;TA=M;MT=Variable (standard in virtually every experiment)
  - Common variable mods: Acetyl (protein N-term), Phospho (S,T,Y), Deamidated (N,Q), GlyGly (K for ubiquitin)
  - List Fixed modifications first, then Variable
  - Extract ALL modifications mentioned — typically 3-8 per study
- acquisition_method: "DDA" for data-dependent, "DIA" for data-independent, "SWATH" for SWATH-MS
- ms2_mass_analyzer: "orbitrap", "ion trap", "time-of-flight", "quadrupole" (lowercase)
- ionization_type: "ESI", "nanoESI", "MALDI"
- fragmentation_method: "HCD", "CID", "ETD", "EThcD"
- separation: "Reversed-phase chromatography", "HPLC", "nanoLC"
- precursor_mass_tolerance: e.g. "10 ppm", "20 ppm"
- fragment_mass_tolerance: e.g. "0.02 Da", "0.5 Da"
- fractionation_method: e.g. "High-pH reversed-phase", "SCX", "SDS-PAGE"
- collision_energy: e.g. "30 NCE", "35% NCE" """


def build_r1_prompt_specialist(paper: dict, raw_files: list, pxd: str,
                               fields: list, pride_meta: dict = None) -> str:
    """Build R1 extraction prompt for a specific field subset."""
    paper_text = ""
    for section in ['TITLE', 'ABSTRACT', 'METHODS', 'RESULTS', 'INTRO', 'DISCUSS']:
        text = paper.get(section, "").strip()
        if text:
            paper_text += f"\n=== {section} ===\n{text}\n"

    fields_schema = ", ".join(f'"{f}"' for f in fields)

    pride_block = ""
    if pride_meta:
        pride_lines = ["\n=== PRIDE/PROTEOMEXCHANGE REPOSITORY METADATA (high-confidence, from submitter) ==="]
        if pride_meta.get("instrument"):
            ac = pride_meta.get("instrument_ac", "")
            pride_lines.append(f"Instrument: {pride_meta['instrument']}" +
                             (f" (accession: {ac})" if ac else ""))
        if pride_meta.get("organism"):
            pride_lines.append(f"Organism: {pride_meta['organism']}")
        if pride_meta.get("organism_part"):
            pride_lines.append(f"Organism part: {pride_meta['organism_part']}")
        if pride_meta.get("disease"):
            pride_lines.append(f"Disease: {pride_meta['disease']}")
        if pride_meta.get("quantification"):
            pride_lines.append(f"Quantification: {pride_meta['quantification']}")
        if pride_meta.get("modifications"):
            mod_names = [m["name"] for m in pride_meta["modifications"]]
            pride_lines.append(f"Identified PTMs: {', '.join(mod_names)}")
        if pride_meta.get("sample_protocol"):
            proto = pride_meta["sample_protocol"][:500]
            pride_lines.append(f"Sample protocol: {proto}")
        if pride_meta.get("data_protocol"):
            proto = pride_meta["data_protocol"][:500]
            pride_lines.append(f"Data processing: {proto}")
        pride_block = "\n".join(pride_lines) + "\n"

    mods_note = ""
    if "modifications" in fields:
        mods_note = '\nFor "modifications", return a list of strings.'

    return f"""Extract SDRF metadata from this proteomics paper for dataset {pxd}.
Raw data files: {json.dumps(raw_files)}

Return a JSON object with ONLY these keys: [{fields_schema}]
{mods_note}
For all other fields, return a single string value.
{pride_block}
PAPER TEXT:
{paper_text}

Return ONLY the JSON object."""


def build_r1_prompt(paper: dict, raw_files: list, pxd: str,
                    pride_meta: dict = None) -> str:
    paper_text = ""
    for section in ['TITLE', 'ABSTRACT', 'METHODS', 'RESULTS', 'INTRO', 'DISCUSS']:
        text = paper.get(section, "").strip()
        if text:
            paper_text += f"\n=== {section} ===\n{text}\n"

    fields_schema = ", ".join(f'"{f}"' for f in EXTRACTION_FIELDS)

    pride_block = ""
    if pride_meta:
        pride_lines = ["\n=== PRIDE/PROTEOMEXCHANGE REPOSITORY METADATA (high-confidence, from submitter) ==="]
        if pride_meta.get("instrument"):
            ac = pride_meta.get("instrument_ac", "")
            pride_lines.append(f"Instrument: {pride_meta['instrument']}" +
                             (f" (accession: {ac})" if ac else ""))
        if pride_meta.get("organism"):
            pride_lines.append(f"Organism: {pride_meta['organism']}")
        if pride_meta.get("organism_part"):
            pride_lines.append(f"Organism part: {pride_meta['organism_part']}")
        if pride_meta.get("disease"):
            pride_lines.append(f"Disease: {pride_meta['disease']}")
        if pride_meta.get("quantification"):
            pride_lines.append(f"Quantification: {pride_meta['quantification']}")
        if pride_meta.get("modifications"):
            mod_names = [m["name"] for m in pride_meta["modifications"]]
            pride_lines.append(f"Identified PTMs: {', '.join(mod_names)}")
        if pride_meta.get("sample_protocol"):
            proto = pride_meta["sample_protocol"][:500]
            pride_lines.append(f"Sample protocol: {proto}")
        if pride_meta.get("data_protocol"):
            proto = pride_meta["data_protocol"][:500]
            pride_lines.append(f"Data processing: {proto}")
        pride_block = "\n".join(pride_lines) + "\n"

    return f"""Extract SDRF metadata from this proteomics paper for dataset {pxd}.
Raw data files: {json.dumps(raw_files)}

Return a JSON object with these keys: [{fields_schema}]

For "modifications", return a list of strings.
For all other fields, return a single string value.
{pride_block}
PAPER TEXT:
{paper_text}

Return ONLY the JSON object."""


R2_SYSTEM = """You are an expert bioinformatician reviewing SDRF metadata extractions.
You will see 3 independent extractions from the same paper. Your job:
1. Compare all 3 extractions carefully
2. For each field, pick the best value or synthesize a better one
3. Assign a confidence score (0.0 to 1.0) for each field
4. If you spot something one extractor found that others missed, use it

Return ONLY valid JSON. No markdown, no explanation."""


def build_r2_prompt(paper: dict, raw_files: list, pxd: str,
                    r1_extractions: List[Tuple[str, dict]]) -> str:
    paper_text = ""
    for section in ['TITLE', 'ABSTRACT', 'METHODS', 'RESULTS', 'INTRO', 'DISCUSS']:
        text = paper.get(section, "").strip()
        if text:
            paper_text += f"\n=== {section} ===\n{text}\n"

    extractions_text = ""
    for i, (model_name, ext) in enumerate(r1_extractions):
        extractions_text += f"\n--- Extraction {i+1} (from {model_name}) ---\n"
        extractions_text += json.dumps(ext, indent=2) + "\n"

    return f"""Dataset {pxd} has {len(raw_files)} raw files.

Here are 3 independent extractions from the same paper:
{extractions_text}

Re-read the paper and produce a reconciled extraction. For each field, return:
{{"value": "your best answer", "confidence": 0.0-1.0}}

For "modifications", return:
{{"value": ["list", "of", "mods"], "confidence": 0.0-1.0}}

Confidence guide:
- 1.0: Explicitly stated in paper, all extractors agree
- 0.8: Clearly in paper, minor disagreements in wording
- 0.5: Implied or only partially stated
- 0.2: Guessed or inferred from limited context
- 0.0: Not found in paper at all

Return a JSON object where every key maps to {{"value": ..., "confidence": ...}}.
Keys: {json.dumps(EXTRACTION_FIELDS)}

PAPER TEXT:
{paper_text}

Return ONLY the JSON object."""


# ── Training SDRF few-shot index ──
_training_index = None  # built lazily

def _build_training_index(data_dir: str) -> list:
    """Build an index of training SDRFs for few-shot retrieval."""
    global _training_index
    if _training_index is not None:
        return _training_index

    sdrf_dir = os.path.join(data_dir, "TrainingSDRFs")
    if not os.path.isdir(sdrf_dir):
        _training_index = []
        return _training_index

    KEY_COLS = ["Organism", "OrganismPart", "Instrument", "CleavageAgent", "Label",
                "Disease", "MaterialType", "Modification", "FragmentationMethod",
                "Separation", "MS2MassAnalyzer"]

    index = []
    for fname in os.listdir(sdrf_dir):
        if not fname.endswith(".sdrf.tsv"):
            continue
        pxd = fname.split("_")[0]
        fpath = os.path.join(sdrf_dir, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                rows = list(reader)
            if not rows:
                continue
            # Extract unique values for key columns
            col_vals = {}
            for col in KEY_COLS:
                vals = set()
                for r in rows:
                    v = r.get(col, "").strip()
                    if v and v.lower() not in ("not applicable", "not available"):
                        vals.add(v)
                if vals:
                    col_vals[col] = vals
            # Build compact excerpt (first row, key columns only)
            excerpt_cols = [c for c in KEY_COLS if c in col_vals]
            excerpt_line = " | ".join(f"{c}={list(col_vals[c])[0]}" for c in excerpt_cols[:8])
            index.append({
                "pxd": pxd,
                "organisms": col_vals.get("Organism", set()),
                "instruments": col_vals.get("Instrument", set()),
                "labels": col_vals.get("Label", set()),
                "excerpt": excerpt_line,
                "col_vals": col_vals,
                "n_rows": len(rows),
            })
        except Exception:
            continue

    _training_index = index
    print(f"  [FEW-SHOT] Built training index: {len(index)} SDRFs")
    return _training_index


def _get_few_shot_block(extraction: dict, data_dir: str) -> str:
    """Find 2 most similar training SDRFs and format as few-shot examples."""
    index = _build_training_index(data_dir)
    if not index:
        return ""

    # Extract organism and instrument from the current extraction
    ext_org = str(extraction.get("organism", "")).lower()
    ext_inst = str(extraction.get("instrument", "")).lower()
    ext_label = str(extraction.get("label_type", "")).lower()

    def similarity(entry):
        score = 0
        # Organism match (highest weight)
        for org in entry["organisms"]:
            if org.lower() in ext_org or ext_org in org.lower():
                score += 3
                break
        # Instrument match
        for inst in entry["instruments"]:
            if any(w in ext_inst for w in inst.lower().split()):
                score += 2
                break
        # Label match
        for lbl in entry["labels"]:
            if "label free" in ext_label and "label free" in lbl.lower():
                score += 1
            elif "tmt" in ext_label and "tmt" in lbl.lower():
                score += 1
        return score

    ranked = sorted(index, key=similarity, reverse=True)
    top = ranked[:2]

    if not top:
        return ""

    lines = ["\n=== SIMILAR TRAINING SDRF EXAMPLES (gold standard format) ==="]
    for entry in top:
        lines.append(f"\n{entry['pxd']} ({entry['n_rows']} rows):")
        lines.append(f"  {entry['excerpt']}")
        # Show unique values per key column
        for col, vals in entry['col_vals'].items():
            unique = sorted(vals)[:3]
            suffix = f" (+{len(vals)-3} more)" if len(vals) > 3 else ""
            lines.append(f"  {col}: {', '.join(unique)}{suffix}")

    return "\n".join(lines) + "\n"


def build_judge_system(few_shot_block: str = "") -> str:
    """Build the judge system prompt, optionally with few-shot examples."""
    return f"""You are the final judge for SDRF metadata extraction in a proteomics competition.
You will see 3 reconciled extractions with confidence scores. Your job is to produce the
single best extraction, checking for cross-field consistency and ontology correctness.

GROUNDING CONTEXT:

{TRAINING_EXAMPLE}

{ONTOLOGY_REFERENCE}

{SCORING_DESCRIPTION}
{few_shot_block}
RULES:
1. Return ONLY valid JSON with flat key-value pairs (no confidence scores, just final values).
2. Check cross-field consistency: organism vs material_type, label_type vs expected row count, etc.
3. Use canonical ontology terms wherever possible -- they score higher.
4. For "modifications", return a COMPLETE list of ALL modifications (typically 3-8 per study: fixed + variable). Check the paper's methods section for all searched modifications.
5. If all 3 reconciliations have low confidence on a field and you can't verify from the paper, use "Not Applicable".
6. Prefer specificity: "Homo sapiens" over "human", "Q Exactive HF" over "Orbitrap".
7. Use "Not Applicable" for unknown or inapplicable fields.
8. For factor_value_type, identify the PRIMARY experimental variable (treatment, disease, bait, compound, genetic_modification, temperature).
9. For factor_values, list ALL distinct conditions/groups in the experiment."""

# Keep a static version for backward compatibility
JUDGE_SYSTEM = build_judge_system()


def build_judge_prompt(paper: dict, raw_files: list, pxd: str,
                       r2_reconciliations: List[Tuple[str, dict]],
                       validation_feedback: str = "") -> str:
    paper_text = ""
    for section in ['TITLE', 'ABSTRACT', 'METHODS']:
        text = paper.get(section, "").strip()
        if text:
            paper_text += f"\n=== {section} ===\n{text}\n"

    recon_text = ""
    for i, (model_name, recon) in enumerate(r2_reconciliations):
        recon_text += f"\n--- Reconciliation {i+1} (from {model_name}) ---\n"
        recon_text += json.dumps(recon, indent=2) + "\n"

    feedback_block = ""
    if validation_feedback:
        feedback_block = f"""
PREVIOUS VALIDATION FOUND THESE ISSUES -- you must fix them:
{validation_feedback}
"""

    return f"""Produce the final SDRF extraction for dataset {pxd}.
Raw files: {json.dumps(raw_files)} ({len(raw_files)} files)

Here are 3 reconciled extractions with confidence scores:
{recon_text}
{feedback_block}
PAPER TEXT (for fact-checking):
{paper_text}

Return ONLY a flat JSON object with keys: {json.dumps(EXTRACTION_FIELDS)}
"modifications" should be a list. All other fields should be strings."""


def build_refine_prompt(paper: dict, raw_files: list, pxd: str,
                        judge_output: dict, feedback: str) -> str:
    """Prompt for refinement round: models correct specific issues."""
    paper_text = ""
    for section in ['TITLE', 'ABSTRACT', 'METHODS', 'RESULTS', 'INTRO', 'DISCUSS']:
        text = paper.get(section, "").strip()
        if text:
            paper_text += f"\n=== {section} ===\n{text}\n"

    return f"""The judge's extraction for {pxd} has validation errors. Fix them.

CURRENT EXTRACTION:
{json.dumps(judge_output, indent=2)}

ERRORS TO FIX:
{feedback}

Raw files: {json.dumps(raw_files)}

Re-read the paper carefully. Return a corrected extraction with confidence scores.
For each field: {{"value": "...", "confidence": 0.0-1.0}}
For "modifications": {{"value": ["list"], "confidence": 0.0-1.0}}

Keys: {json.dumps(EXTRACTION_FIELDS)}

PAPER TEXT:
{paper_text}

Return ONLY the JSON object."""


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_extraction(extraction: dict, raw_files: list, expected_rows: int) -> List[str]:
    """Validate a final extraction. Returns list of error strings (empty = valid)."""
    errors = []

    for field in CRITICAL_FIELDS:
        val = extraction.get(field, "")
        if not val or str(val).lower() in ("not available", "not applicable", "n/a", "none", ""):
            errors.append(
                f"CRITICAL: '{field}' is missing or 'not available'. "
                f"This field almost always exists in proteomics papers. "
                f"Re-read the Methods section carefully."
            )

    for field in IMPORTANT_FIELDS:
        val = extraction.get(field, "")
        if field == "modifications":
            if not val or (isinstance(val, list) and len(val) == 0):
                errors.append(
                    f"WARNING: No modifications found. Most experiments use at least "
                    f"Carbamidomethyl (C) as fixed and Oxidation (M) as variable."
                )
        elif not val or str(val).lower() in ("not available", "not applicable", "n/a", "none", ""):
            errors.append(f"WARNING: '{field}' is 'not available'.")

    organism = extraction.get("organism", "")
    if organism and organism.lower() not in ("not available", "not applicable"):
        if len(organism.strip().split()) < 2:
            errors.append(f"FORMAT: Organism '{organism}' should be binomial (e.g., 'Homo sapiens').")

    label_type = extraction.get("label_type", "label free")
    channels = get_tmt_channels(label_type)
    if channels:
        expected_from_label = len(raw_files) * len(channels)
        if expected_from_label != expected_rows:
            errors.append(
                f"ROW_COUNT: '{label_type}' gives {len(channels)} channels × "
                f"{len(raw_files)} files = {expected_from_label}, but expected {expected_rows} rows."
            )
    elif len(raw_files) != expected_rows and expected_rows > len(raw_files):
        errors.append(
            f"ROW_COUNT: {len(raw_files)} files but {expected_rows} expected rows. "
            f"Is this multiplexed (TMT/iTRAQ)?"
        )

    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# ONTOLOGY FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

def format_instrument(raw: str) -> str:
    if not raw or raw.lower() in ("not available", "not applicable"):
        return "Not Applicable"
    if "AC=MS:" in raw:
        # Normalize to NT-first format if currently AC-first
        s = raw.strip()
        if s.startswith("AC="):
            import re
            m = re.match(r'(AC=MS:\d+);(NT=.+)', s)
            if m:
                return f"{m.group(2)};{m.group(1)}"
        return raw
    key = raw.lower().strip()
    for pattern, formatted in INSTRUMENT_ONTOLOGY.items():
        if pattern in key:
            return formatted
    return f"NT={raw}"


def format_modification(raw: str) -> str:
    if not raw or raw.lower() in ("not available", "not applicable"):
        return ""
    if "NT=" in raw or "AC=UNIMOD" in raw:
        return raw
    key = re.sub(r'\s*\([^)]*\)', '', raw.lower().strip()).strip()
    for pattern, formatted in MODIFICATION_ONTOLOGY.items():
        if pattern in key:
            return formatted
    return f"NT={raw};MT=Variable"


def format_fragmentation(raw: str) -> str:
    if not raw or raw.lower() in ("not available", "not applicable"):
        return "Not Applicable"
    if "AC=MS:" in raw:
        return raw
    key = raw.lower().strip()
    for pattern, formatted in FRAGMENTATION_ONTOLOGY.items():
        if pattern in key:
            return formatted
    return f"NT={raw}"


def format_cleavage(raw: str) -> str:
    if not raw or raw.lower() in ("not available", "not applicable"):
        return "Not Applicable"
    if "AC=MS:" in raw:
        return raw
    key = raw.lower().strip()
    for pattern, formatted in CLEAVAGE_ONTOLOGY.items():
        if pattern in key:
            return formatted
    return f"NT={raw}"


def format_label(raw: str) -> str:
    if not raw or raw.lower() in ("not available", "not applicable"):
        return "Not Applicable"
    key = raw.lower().strip()
    # Already formatted (AC=... or NT=...)
    if "AC=" in raw or "NT=" in raw:
        return raw
    # Label free variations
    if "label free" in key or "label-free" in key or key == "unlabeled sample" or key == "label free sample":
        return "AC=MS:1002038;NT=label free sample"
    # Preserve specific channel labels — do NOT collapse to generic types
    if key.startswith("tmt") and len(key) > 3:
        return raw  # TMT126, TMT127N, TMTpro126, etc.
    if key.startswith("itraq") and len(key) > 5:
        return raw  # ITRAQ114, ITRAQ115, etc.
    if "silac" in key and any(ch in key for ch in ("heavy", "medium", "light")):
        return raw  # SILAC heavy, SILAC medium, SILAC light
    # Generic type mappings
    for pattern, formatted in LABEL_ONTOLOGY.items():
        if pattern in key:
            return formatted
    return raw


def get_tmt_channels(label_type: str) -> list:
    key = label_type.lower().strip().replace(" ", "").replace("-", "")
    # Check exact matches first for specificity
    for pattern, channels in TMT_CHANNELS.items():
        if pattern == key:
            return channels
    # Then substring matches
    for pattern, channels in TMT_CHANNELS.items():
        if pattern in key:
            return channels
    # Detect SILAC without explicit plex count
    if "silac" in key:
        if "triple" in key or "3" in key or "medium" in key:
            return TMT_CHANNELS["silac3"]
        return TMT_CHANNELS["silac"]  # Default to 2-channel SILAC
    # Detect iTRAQ without explicit plex count
    if "itraq" in key:
        if "8" in key:
            return TMT_CHANNELS["itraq8plex"]
        return TMT_CHANNELS["itraq4plex"]  # Default to 4-plex
    return []


def format_acquisition(raw: str) -> str:
    if not raw or raw.lower() in ("not available", "not applicable"):
        return "Not Applicable"
    if "AC=" in raw or "NT=" in raw:
        return raw
    key = raw.lower().strip()
    for pattern, formatted in ACQUISITION_ONTOLOGY.items():
        if pattern in key:
            return formatted
    return raw


def format_ms2_analyzer(raw: str) -> str:
    if not raw or raw.lower() in ("not available", "not applicable"):
        return "Not Applicable"
    if "AC=MS:" in raw:
        return raw
    key = raw.lower().strip()
    for pattern, formatted in MS2_ANALYZER_ONTOLOGY.items():
        if pattern in key:
            return formatted
    return raw


def format_fractionation(raw: str) -> str:
    if not raw or raw.lower() in ("not available", "not applicable"):
        return "Not Applicable"
    if "AC=" in raw or "NT=" in raw:
        return raw
    key = raw.lower().strip()
    for pattern, formatted in FRACTIONATION_ONTOLOGY.items():
        if pattern in key:
            return formatted
    return raw


def format_separation(raw: str) -> str:
    if not raw or raw.lower() in ("not available", "not applicable"):
        return "Not Applicable"
    if "AC=" in raw or "NT=" in raw:
        return raw
    key = raw.lower().strip()
    for pattern, formatted in SEPARATION_ONTOLOGY.items():
        if pattern in key:
            return formatted
    return raw


def format_ionization(raw: str) -> str:
    if not raw or raw.lower() in ("not available", "not applicable"):
        return "Not Applicable"
    if "AC=MS:" in raw:
        return raw
    key = raw.lower().strip()
    for pattern, formatted in IONIZATION_ONTOLOGY.items():
        if pattern in key:
            return formatted
    return raw


def format_collision_energy(raw: str) -> str:
    """Normalize collision energy to standard format: 'X NCE', 'X eV', etc."""
    if not raw or str(raw).lower() in ("not applicable", "not available", "none", "", "unknown",
                                        "not mentioned", "not specified"):
        return "Not Applicable"
    s = str(raw).strip()
    # Already well-formatted
    if s.endswith(" NCE") or s.endswith(" eV") or s.endswith(" V"):
        return s
    import re
    # "NCE 28" → "28 NCE"
    m = re.match(r'^NCE\s*(\d+(?:\.\d+)?)', s, re.IGNORECASE)
    if m:
        return f"{m.group(1)} NCE"
    # "35%" → "35 NCE" (percentage is typically NCE)
    m = re.match(r'^(\d+(?:\.\d+)?)\s*%$', s)
    if m:
        return f"{m.group(1)} NCE"
    # "35" (bare number) → "35 NCE"
    m = re.match(r'^(\d+(?:\.\d+)?)$', s)
    if m:
        return f"{m.group(1)} NCE"
    # "35 nce", "35nce"
    m = re.match(r'^(\d+(?:\.\d+)?)\s*nce$', s, re.IGNORECASE)
    if m:
        return f"{m.group(1)} NCE"
    # "35 ev", "35eV"
    m = re.match(r'^(\d+(?:\.\d+)?)\s*ev$', s, re.IGNORECASE)
    if m:
        return f"{m.group(1)} eV"
    # Complex descriptions: pass through
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# ROW BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_sdrf_rows(pxd: str, raw_files: list, extraction: dict,
                    sample_rows: list) -> list:
    NA = "Not Applicable"
    NAV = "Not Applicable"

    label_type = extraction.get("label_type", "label free")
    channels = get_tmt_channels(label_type)
    is_multiplexed = len(channels) > 0

    instrument = format_instrument(extraction.get("instrument", NAV))
    frag_method = format_fragmentation(extraction.get("fragmentation_method", ""))
    cleavage = format_cleavage(extraction.get("cleavage_agent", ""))

    raw_mods = extraction.get("modifications", [])
    if isinstance(raw_mods, str):
        raw_mods = [raw_mods]
    formatted_mods = [format_modification(m) for m in raw_mods if m]
    formatted_mods = [m for m in formatted_mods if m]
    while len(formatted_mods) < 7:
        formatted_mods.append(NA)

    def norm(val, default=NAV):
        if not val or str(val).lower() in ("none", "n/a", "", "not mentioned", "not specified", "not available", "unknown"):
            return default
        if str(val).lower() == "not applicable":
            return "Not Applicable"
        return val

    base = {col: NA for col in METADATA_COLS}
    base['Characteristics[Organism]'] = norm(extraction.get("organism", NAV))
    base['Characteristics[OrganismPart]'] = norm(extraction.get("organism_part", NAV))
    base['Characteristics[CellType]'] = norm(extraction.get("cell_type", NAV))
    base['Characteristics[CellLine]'] = norm(extraction.get("cell_line", NA), NA)
    base['Characteristics[Disease]'] = norm(extraction.get("disease", NAV))
    base['Characteristics[Sex]'] = norm(extraction.get("sex", NAV))
    base['Characteristics[Age]'] = norm(extraction.get("age", NAV))
    base['Characteristics[Strain]'] = norm(extraction.get("strain", NA), NA)
    base['Characteristics[CleavageAgent]'] = cleavage
    base['Characteristics[MaterialType]'] = norm(extraction.get("material_type", NAV))
    base['Characteristics[AlkylationReagent]'] = norm(extraction.get("alkylation_reagent", NA), NA)
    base['Characteristics[ReductionReagent]'] = norm(extraction.get("reduction_reagent", NA), NA)
    base['Characteristics[Compound]'] = norm(extraction.get("compound", NA), NA)
    base['Characteristics[Treatment]'] = norm(extraction.get("treatment", NA), NA)
    base['Characteristics[GeneticModification]'] = norm(extraction.get("genetic_modification", NA), NA)
    base['Characteristics[Bait]'] = norm(extraction.get("bait", NA), NA)

    for i, mod in enumerate(formatted_mods[:7]):
        suffix = "" if i == 0 else f".{i}"
        col = f"Characteristics[Modification]{suffix}"
        if col in base:
            base[col] = mod

    base['Comment[Instrument]'] = instrument
    base['Comment[FragmentationMethod]'] = frag_method
    base['Comment[EnrichmentMethod]'] = norm(extraction.get("enrichment_method", NA), NA)
    base['Comment[FractionationMethod]'] = format_fractionation(extraction.get("fractionation_method", ""))
    base['Comment[PrecursorMassTolerance]'] = norm(extraction.get("precursor_mass_tolerance", NAV))
    base['Comment[FragmentMassTolerance]'] = norm(extraction.get("fragment_mass_tolerance", NAV))
    base['Comment[NumberOfMissedCleavages]'] = norm(extraction.get("missed_cleavages", NAV))
    base['Comment[AcquisitionMethod]'] = format_acquisition(extraction.get("acquisition_method", ""))
    base['Comment[Separation]'] = format_separation(extraction.get("separation", ""))
    base['Comment[IonizationType]'] = format_ionization(extraction.get("ionization_type", NA))
    base['Comment[MS2MassAnalyzer]'] = format_ms2_analyzer(extraction.get("ms2_mass_analyzer", ""))
    base['Comment[NumberOfFractions]'] = norm(extraction.get("number_of_fractions", NA), NA)
    base['Comment[CollisionEnergy]'] = format_collision_energy(extraction.get("collision_energy", ""))
    base['Comment[GradientTime]'] = norm(extraction.get("gradient_time", NA), NA)
    base['Comment[FlowRateChromatogram]'] = norm(extraction.get("flow_rate", NA), NA)
    base['Characteristics[Specimen]'] = norm(extraction.get("specimen", NA), NA)
    base['Characteristics[PooledSample]'] = norm(extraction.get("pooled_sample", NA), NA)

    # Additional characteristics columns
    base['Characteristics[DevelopmentalStage]'] = norm(extraction.get("developmental_stage", NA), NA)
    base['Characteristics[AncestryCategory]'] = norm(extraction.get("ancestry_category", NA), NA)
    base['Characteristics[Depletion]'] = norm(extraction.get("depletion", NA), NA)
    base['Characteristics[SpikedCompound]'] = norm(extraction.get("spiked_compound", NA), NA)
    base['Characteristics[CellPart]'] = norm(extraction.get("cell_part", NA), NA)
    base['Characteristics[Temperature]'] = norm(extraction.get("temperature", NA), NA)
    base['Characteristics[Genotype]'] = norm(extraction.get("genotype", NA), NA)
    base['Characteristics[NumberOfBiologicalReplicates]'] = norm(extraction.get("number_of_biological_replicates", NA), NA)
    base['Characteristics[NumberOfTechnicalReplicates]'] = norm(extraction.get("number_of_technical_replicates", NA), NA)
    base['Characteristics[NumberOfSamples]'] = norm(extraction.get("number_of_samples", NA), NA)
    base['Characteristics[ConcentrationOfCompound]'] = norm(extraction.get("concentration_of_compound", NA), NA)
    base['Characteristics[SamplingTime]'] = norm(extraction.get("sampling_time", NA), NA)
    base['Characteristics[Staining]'] = norm(extraction.get("staining", NA), NA)
    base['Characteristics[SyntheticPeptide]'] = norm(extraction.get("synthetic_peptide", NA), NA)
    base['Characteristics[DiseaseTreatment]'] = norm(extraction.get("disease_treatment", NA), NA)
    base['Characteristics[Time]'] = norm(extraction.get("time", NA), NA)

    # ── Bayesian column gating ──
    # For low-prior columns (rare in training SDRFs), suppress LLM guesses
    # to avoid adding wrong values that drag down F1 average.
    # Training priors: DevelopmentalStage=39%, AncestryCategory=23%, Age=47%
    # Format: {submission_col: set of generic defaults to suppress (empty=suppress all)}
    _SUPPRESS_COLS = {
        'Characteristics[DevelopmentalStage]': {'adult', 'not determined'},  # 39% active
        'Characteristics[AncestryCategory]': set(),  # 23% active — suppress all
        'Characteristics[Age]': set(),  # 47% active — suppress all guesses
        'Characteristics[Sex]': {'male', 'female', 'mixed', 'not determined'},  # 58% active
    }
    for col, defaults in _SUPPRESS_COLS.items():
        val = base.get(col, NA)
        if val in (NA, NAV, "Not Applicable"):
            continue
        if len(defaults) == 0 or val.lower().strip() in {d.lower() for d in defaults}:
            base[col] = NA

    # FactorValue mirroring (Change 3)
    fv_type = extraction.get("factor_value_type", "")
    if fv_type and str(fv_type).lower() not in ("not applicable", "not available", "none", ""):
        fv_type_lower = fv_type.lower().strip()
        fv_mapping = {
            "treatment": ("FactorValue[Treatment]", "Characteristics[Treatment]"),
            "disease": ("FactorValue[Disease]", "Characteristics[Disease]"),
            "bait": ("FactorValue[Bait]", "Characteristics[Bait]"),
            "compound": ("FactorValue[Compound]", "Characteristics[Compound]"),
            "genetic_modification": ("FactorValue[GeneticModification]", "Characteristics[GeneticModification]"),
            "temperature": ("FactorValue[Temperature]", "Characteristics[Temperature]"),
            "cell_part": ("FactorValue[CellPart]", "Characteristics[CellPart]"),
            "fraction_identifier": ("FactorValue[FractionIdentifier]", "Comment[FractionIdentifier]"),
            "genotype": ("FactorValue[GeneticModification]", "Characteristics[Genotype]"),
            "concentration": ("FactorValue[ConcentrationOfCompound].1", "Characteristics[ConcentrationOfCompound]"),
        }
        for fv_key, (fv_col, src_col) in fv_mapping.items():
            if fv_key in fv_type_lower:
                src_val = base.get(src_col, NA)
                if src_val != NA:
                    base[fv_col] = src_val
                break

    # Determine biological replicate scheme
    # If LLM extracted a specific number of biological replicates, use that
    # Otherwise default to "1" for all rows (most common in gold standard)
    n_bio_rep = extraction.get("number_of_biological_replicates", "")
    bio_rep_scheme = extraction.get("biological_replicate_scheme", "")
    try:
        n_bio = int(str(n_bio_rep).strip()) if n_bio_rep and str(n_bio_rep).strip().isdigit() else 0
    except (ValueError, TypeError):
        n_bio = 0

    rows = []
    if is_multiplexed:
        for raw_file in raw_files:
            for ch_idx, channel in enumerate(channels):
                row = dict(base)
                row['PXD'] = pxd
                row['Raw Data File'] = raw_file
                row['Characteristics[Label]'] = channel
                # For multiplexed, each channel is a different sample, use channel index
                row['Characteristics[BiologicalReplicate]'] = str(ch_idx + 1)
                rows.append(row)
    else:
        for file_idx, raw_file in enumerate(raw_files):
            row = dict(base)
            row['PXD'] = pxd
            row['Raw Data File'] = raw_file
            row['Characteristics[Label]'] = format_label(label_type)
            # Default: all rows get replicate "1" (most common in gold standard)
            # Only assign sequential replicates if LLM identified multiple bio replicates
            if n_bio > 1 and len(raw_files) >= n_bio:
                # Distribute files across biological replicates
                files_per_rep = max(1, len(raw_files) // n_bio)
                rep_idx = min(file_idx // files_per_rep, n_bio - 1) + 1
                row['Characteristics[BiologicalReplicate]'] = str(rep_idx)
            else:
                row['Characteristics[BiologicalReplicate]'] = "1"
            rows.append(row)

    expected = len(sample_rows)
    if len(rows) != expected and expected > 0:
        print(f"      Row adjustment: {len(rows)} -> {expected}")
        if len(rows) > expected:
            rows = rows[:expected]
        else:
            while len(rows) < expected:
                row = dict(rows[-1])
                # Keep same replicate as last row (don't create false unique values)
                rows.append(row)

    for i, row in enumerate(rows):
        if i < len(sample_rows):
            row['ID'] = sample_rows[i]['ID']
            row['Raw Data File'] = sample_rows[i]['Raw Data File']
        else:
            row['ID'] = str(i + 1)

    # Per-row FractionIdentifier from raw file names
    _assign_fraction_identifiers(rows)

    return rows


def _assign_fraction_identifiers(rows: list) -> None:
    """Parse fraction identifiers from raw file names and assign to rows."""
    for row in rows:
        raw = row.get('Raw Data File', '')
        if not raw:
            continue
        # Pattern 1: Explicit fraction prefixes (F, frac, fraction, Fr)
        # Lookahead prevents matching mutations like F198S (letter after digits)
        m = re.search(r'[_\-](?:F|frac(?:tion)?|Fr)(\d{1,3})(?=[_.\-]|$)', raw, re.IGNORECASE)
        if m and int(m.group(1)) <= 200:
            row['Comment[FractionIdentifier]'] = m.group(1)
            continue
        # Pattern 2: High-pH/temperature fractionation: ...{temp}C_{fraction}.raw
        # Strict lookahead: fraction must be followed by separator/end (prevents 30min backtrack)
        m = re.search(r'[\-_]\d+C[\-_](\d{1,3})(?=[_.\-]|$)', raw)
        if m and int(m.group(1)) <= 200:
            row['Comment[FractionIdentifier]'] = m.group(1)
            continue
        # Default: no fractionation detected — gold standard uses "1" (99% of cases)
        row['Comment[FractionIdentifier]'] = '1'


# ═══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH STATE
# ═══════════════════════════════════════════════════════════════════════════════

class PXDState(TypedDict):
    pxd: str
    paper: dict
    raw_files: list
    sample_rows: list
    model_ids: list
    pride_meta: dict           # PRIDE API metadata
    data_dir: str              # for few-shot training SDRF lookup
    temperature: float         # model temperature for this run
    # Round 1: independent extractions
    r1_extractions: list       # [(model_name, flat_dict), ...]
    # Round 2: reconciliations with confidence
    r2_reconciliations: list   # [(model_name, {field: {value, confidence}}), ...]
    # Round 3: judge output
    judge_output: dict         # flat dict
    # Validation
    validation_errors: list
    refinement_feedback: str
    iteration: int
    max_iterations: int
    # Output
    sdrf_rows: list


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH NODES
# ═══════════════════════════════════════════════════════════════════════════════

def extract_r1_node(state: PXDState) -> dict:
    """Round 1: Independent extraction from each model (single prompt)."""
    pxd = state["pxd"]
    paper = state["paper"]
    raw_files = state["raw_files"]
    model_ids = state["model_ids"]
    pride_meta = state.get("pride_meta", {})
    temp = state.get("temperature", 0.1)

    print(f"\n  [R1 EXTRACT] Single-prompt extraction -- {len(model_ids)} models (temp={temp})")

    prompt = build_r1_prompt(paper, raw_files, pxd, pride_meta)
    extractions = []

    for model_id in model_ids:
        cfg = MODELS.get(model_id) or TIEBREAKER_MODEL.get(model_id)
        if not cfg:
            print(f"    WARNING: Unknown model '{model_id}', skipping")
            continue

        print(f"    {cfg['display']}...", end=" ", flush=True)
        response = call_model(model_id, R1_SYSTEM, prompt, temperature=temp)
        parsed = parse_json_response(response) if response else {}
        if parsed:
            print(f"[OK] {len(parsed)} fields")
            extractions.append((cfg['display'], parsed))
        else:
            print("[FAIL]")

    return {"r1_extractions": extractions}


def deliberate_r2_node(state: PXDState) -> dict:
    """Round 2: Each model sees all R1 outputs and reconciles with confidence scores."""
    pxd = state["pxd"]
    paper = state["paper"]
    raw_files = state["raw_files"]
    model_ids = state["model_ids"]
    r1_extractions = state["r1_extractions"]
    temp = state.get("temperature", 0.1)

    print(f"\n  [R2 DELIBERATE] Cross-pollinated reconciliation -- {len(model_ids)} models")

    if not r1_extractions:
        print("    No R1 extractions to deliberate on!")
        return {"r2_reconciliations": []}

    prompt = build_r2_prompt(paper, raw_files, pxd, r1_extractions)
    reconciliations = []

    for model_id in model_ids:
        cfg = MODELS.get(model_id) or TIEBREAKER_MODEL.get(model_id)
        if not cfg:
            continue
        print(f"    {cfg['display']}...", end=" ", flush=True)
        response = call_model(model_id, R2_SYSTEM, prompt, max_tokens=5000, temperature=temp)
        if response:
            parsed = parse_json_response(response)
            if parsed:
                reconciliations.append((cfg['display'], parsed))
                # Count high-confidence fields
                high_conf = sum(
                    1 for v in parsed.values()
                    if isinstance(v, dict) and v.get("confidence", 0) >= 0.8
                )
                print(f"[OK] {high_conf}/{len(parsed)} high-confidence")
            else:
                print("[FAIL] parse failed")
        else:
            print("[FAIL] no response")

    return {"r2_reconciliations": reconciliations}


def judge_r3_node(state: PXDState) -> dict:
    """Round 3: Grounded judge produces final extraction."""
    pxd = state["pxd"]
    paper = state["paper"]
    raw_files = state["raw_files"]
    r2_reconciliations = state["r2_reconciliations"]
    feedback = state.get("refinement_feedback", "")
    data_dir = state.get("data_dir", "")

    print(f"\n  [R3 JUDGE] Final reconciliation (Claude Opus)")

    if not r2_reconciliations:
        print("    No reconciliations to judge!")
        return {"judge_output": {}}

    # Build few-shot block from similar training SDRFs
    # Use the first reconciliation's values to find similar examples
    few_shot_block = ""
    if data_dir:
        # Extract a rough extraction dict from reconciliations for similarity matching
        rough_ext = {}
        for _, recon in r2_reconciliations:
            for field, val in recon.items():
                if field not in rough_ext:
                    if isinstance(val, dict):
                        rough_ext[field] = val.get("value", "")
                    else:
                        rough_ext[field] = val
        few_shot_block = _get_few_shot_block(rough_ext, data_dir)
        if few_shot_block:
            print(f"    Few-shot examples injected")

    judge_system = build_judge_system(few_shot_block) if few_shot_block else JUDGE_SYSTEM

    temp = state.get("temperature", 0.1)
    prompt = build_judge_prompt(paper, raw_files, pxd, r2_reconciliations, feedback)
    print(f"    Calling {JUDGE_MODEL_ID}...", end=" ", flush=True)
    response = call_model(JUDGE_MODEL_ID, judge_system, prompt, max_tokens=4000, temperature=temp)

    if response:
        parsed = parse_json_response(response)
        if parsed:
            print(f"[OK] {len(parsed)} fields")
            print(f"    organism={parsed.get('organism', '?')}, "
                  f"instrument={parsed.get('instrument', '?')}, "
                  f"label={parsed.get('label_type', '?')}")
            return {"judge_output": parsed}
        else:
            print("[FAIL] parse failed")
    else:
        print("[FAIL] no response")

    return {"judge_output": {}}


def validate_node(state: PXDState) -> dict:
    """Validate the judge's output."""
    judge_output = state["judge_output"]
    raw_files = state["raw_files"]
    expected_rows = len(state["sample_rows"])
    iteration = state["iteration"]

    print(f"\n  [VALIDATE] Iteration {iteration}")

    if not judge_output:
        return {
            "validation_errors": ["CRITICAL: Judge produced no output"],
            "refinement_feedback": "The judge produced no output. All models must re-extract.",
        }

    errors = validate_extraction(judge_output, raw_files, expected_rows)

    if errors:
        critical = [e for e in errors if e.startswith("CRITICAL") or e.startswith("ROW_COUNT")]
        warnings = [e for e in errors if not e.startswith("CRITICAL") and not e.startswith("ROW_COUNT")]
        print(f"    {len(critical)} critical, {len(warnings)} warnings")
        for e in errors:
            print(f"      * {e[:120]}")
        feedback = "Fix these issues:\n" + "\n".join(f"- {e}" for e in errors)
    else:
        print(f"    [OK] Passed")
        feedback = ""

    return {"validation_errors": errors, "refinement_feedback": feedback}


def refine_node(state: PXDState) -> dict:
    """Re-prompt models with judge output + errors, producing new R2-style reconciliations."""
    pxd = state["pxd"]
    paper = state["paper"]
    raw_files = state["raw_files"]
    model_ids = state["model_ids"]
    judge_output = state["judge_output"]
    feedback = state["refinement_feedback"]
    iteration = state["iteration"] + 1

    print(f"\n  [REFINE] Iteration {iteration} -- re-prompting with targeted feedback")

    prompt = build_refine_prompt(paper, raw_files, pxd, judge_output, feedback)
    reconciliations = []

    for model_id in model_ids:
        cfg = MODELS.get(model_id) or TIEBREAKER_MODEL.get(model_id)
        if not cfg:
            continue
        print(f"    {cfg['display']}...", end=" ", flush=True)
        response = call_model(model_id, R2_SYSTEM, prompt, max_tokens=5000)
        if response:
            parsed = parse_json_response(response)
            if parsed:
                reconciliations.append((cfg['display'], parsed))
                print(f"[OK]")
            else:
                print("[FAIL] parse failed")
        else:
            print("[FAIL] no response")

    return {"r2_reconciliations": reconciliations, "iteration": iteration}


# ═══════════════════════════════════════════════════════════════════════════════
# BAYESIAN PRIORS (computed from 103 training SDRFs)
# ═══════════════════════════════════════════════════════════════════════════════

# Column activity rates: P(column has a real non-NA value) across 103 training PXDs
# Keys are extraction field names (not submission column names)
COLUMN_ACTIVITY_PRIOR = {
    "organism": 1.00,
    "label_type": 1.00,
    "instrument": 1.00,
    "cleavage_agent": 1.00,
    "modifications": 0.97,
    "precursor_mass_tolerance": 0.92,
    "fragment_mass_tolerance": 0.92,
    "organism_part": 0.89,
    "disease": 0.89,
    "sex": 0.77,
    "age": 0.75,
    "cell_type": 0.70,
    "developmental_stage": 0.68,
    "ancestry_category": 0.66,
    "material_type": 0.58,
    "fragmentation_method": 0.55,
    "collision_energy": 0.28,
    "cell_line": 0.28,
    "ms2_mass_analyzer": 0.23,
    "fractionation_method": 0.17,
    "enrichment_method": 0.09,
    "treatment": 0.09,
    "separation": 0.08,
    "depletion": 0.06,
    "spiked_compound": 0.03,
    "strain": 0.03,
    "compound": 0.03,
    "synthetic_peptide": 0.03,
    "missed_cleavages": 0.03,
    "sampling_time": 0.02,
    "specimen": 0.02,
    "reduction_reagent": 0.02,
    "alkylation_reagent": 0.02,
    "time": 0.02,
    "bait": 0.01,
    "cell_part": 0.01,
    "bmi": 0.01,
    "temperature": 0.01,
    "growth_rate": 0.01,
    "staining": 0.01,
    "tumor_size": 0.01,
    "pooled_sample": 0.01,
    "genetic_modification": 0.01,
    "factor_value_type": 0.12,
}

# Value frequency priors for key columns (most common value → P(value|column))
VALUE_PRIORS = {
    "organism": {"Homo sapiens": 0.72, "Mus musculus": 0.06, "Saccharomyces cerevisiae": 0.03},
    "cleavage_agent": {"Trypsin": 0.69, "Lys-C": 0.22, "Trypsin/P": 0.04},
    "material_type": {"tissue": 0.22, "cell": 0.20, "organism part": 0.10, "cell line": 0.06},
    "fragmentation_method": {"HCD": 0.45, "CID": 0.21, "ETD": 0.03},
    "ms2_mass_analyzer": {"orbitrap": 0.80, "ion trap": 0.08},
}


def _apply_bayesian_defaults(extraction: dict):
    """Use training priors to fill missing values for high-activity columns."""
    filled = []
    # For high-activity columns (>90%), if LLM didn't extract, use the most common value
    HIGH_ACTIVITY_DEFAULTS = {
        # field → (threshold, default_value) — only fill if P(active) > threshold
        "precursor_mass_tolerance": (0.90, "10 ppm"),
        "fragment_mass_tolerance": (0.90, "0.02 Da"),
    }
    for field, (threshold, default) in HIGH_ACTIVITY_DEFAULTS.items():
        prior = COLUMN_ACTIVITY_PRIOR.get(field, 0)
        if prior < threshold:
            continue
        value = extraction.get(field, "")
        if not value or str(value).lower() in ("not available", "not applicable", "unknown", "none", ""):
            extraction[field] = default
            filled.append(f"{field}={default}")
    if filled:
        print(f"    Bayesian defaults: {', '.join(filled)}")


def format_node(state: PXDState) -> dict:
    """Build final SDRF rows from judge output, with PRIDE overrides."""
    pxd = state["pxd"]
    raw_files = state["raw_files"]
    judge_output = state["judge_output"]
    sample_rows = state["sample_rows"]
    pride_meta = state.get("pride_meta", {})

    print(f"\n  [FORMAT] Building SDRF rows")

    if not judge_output:
        print("    No judge output -- returning empty")
        return {"sdrf_rows": []}

    # Apply PRIDE overrides for high-confidence structured fields
    if pride_meta:
        _apply_pride_overrides(judge_output, pride_meta)

    # Apply Bayesian defaults — fill missing high-activity columns
    _apply_bayesian_defaults(judge_output)

    rows = build_sdrf_rows(pxd, raw_files, judge_output, sample_rows)
    print(f"    Generated {len(rows)} rows")

    return {"sdrf_rows": rows}


def _apply_pride_overrides(extraction: dict, pride_meta: dict):
    """Override extraction fields with high-confidence PRIDE API data."""
    # Instrument: PRIDE has accession + name from submitter — always trust it
    pride_instrument_sdrf = pride_instrument_to_sdrf(pride_meta)
    if pride_instrument_sdrf:
        current = extraction.get("instrument", "")
        # Always prefer PRIDE instrument when it has an accession
        if pride_meta.get("instrument_ac") or not current or current.lower() in ("not available", "not applicable", "unknown"):
            extraction["instrument"] = pride_instrument_sdrf
            print(f"    PRIDE override: instrument -> {pride_instrument_sdrf}")

    # Organism: PRIDE has taxonomy-backed organism name — always trust it
    pride_org = pride_meta.get("organism", "")
    if pride_org:
        extraction["organism"] = pride_org
        print(f"    PRIDE override: organism -> {pride_org}")

    # Organism part: use if LLM didn't extract
    pride_part = pride_meta.get("organism_part", "")
    if pride_part and pride_part.lower() not in ("not available", "not applicable"):
        current = extraction.get("organism_part", "")
        if not current or current.lower() in ("not available", "not applicable", "unknown"):
            extraction["organism_part"] = pride_part
            print(f"    PRIDE override: organism_part -> {pride_part}")

    # Disease: use if LLM didn't extract
    pride_disease = pride_meta.get("disease", "")
    if pride_disease:
        current = extraction.get("disease", "")
        if not current or current.lower() in ("not available", "not applicable", "unknown"):
            extraction["disease"] = pride_disease
            print(f"    PRIDE override: disease -> {pride_disease}")


# ═══════════════════════════════════════════════════════════════════════════════
# CONDITIONAL ROUTING
# ═══════════════════════════════════════════════════════════════════════════════

def should_refine(state: PXDState) -> Literal["refine", "format"]:
    errors = state.get("validation_errors", [])
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", MAX_REFINEMENT_ITERATIONS)

    critical = [e for e in errors if e.startswith("CRITICAL") or e.startswith("ROW_COUNT")]

    if critical and iteration < max_iter:
        print(f"    -> REFINE ({len(critical)} critical errors, iteration {iteration}/{max_iter})")
        return "refine"
    else:
        if critical:
            print(f"    -> FORMAT (max iterations hit, {len(critical)} unresolved)")
        else:
            print(f"    -> FORMAT (passed)")
        return "format"


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_pxd_graph():
    """
    Build the LangGraph for one PXD.

        extract_r1 -> deliberate_r2 -> judge_r3 -> validate ──┬──► format -> END
                                        ▲                   │
                                        └── refine ◄────────┘
    """
    graph = StateGraph(PXDState)

    graph.add_node("extract_r1", extract_r1_node)
    graph.add_node("deliberate_r2", deliberate_r2_node)
    graph.add_node("judge_r3", judge_r3_node)
    graph.add_node("validate", validate_node)
    graph.add_node("refine", refine_node)
    graph.add_node("format", format_node)

    graph.set_entry_point("extract_r1")
    graph.add_edge("extract_r1", "deliberate_r2")
    graph.add_edge("deliberate_r2", "judge_r3")
    graph.add_edge("judge_r3", "validate")

    graph.add_conditional_edges(
        "validate",
        should_refine,
        {"refine": "refine", "format": "format"},
    )

    # Refine produces new R2-style reconciliations -> judge again
    graph.add_edge("refine", "judge_r3")

    graph.add_edge("format", END)

    return graph.compile()


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def merge_judge_outputs(outputs: List[dict]) -> dict:
    """Merge multiple judge outputs from ensemble runs.
    Strategy:
    - Agreement: both same value → keep
    - One has value, other has N/A → keep the value
    - Conflict: keep Run 1 (lower temperature)
    - Modifications: UNION across runs
    """
    if len(outputs) == 1:
        return outputs[0]
    if not outputs:
        return {}

    base = dict(outputs[0])  # Run 1 wins ties
    for field in EXTRACTION_FIELDS:
        vals = [o.get(field, "Not Applicable") for o in outputs]

        if field == "modifications":
            # UNION all modification lists
            all_mods = set()
            for v in vals:
                if isinstance(v, list):
                    all_mods.update(v)
                elif isinstance(v, str) and v.lower() not in ("not applicable", "not available"):
                    all_mods.add(v)
            base[field] = sorted(all_mods) if all_mods else ["Not Applicable"]
            continue

        if field == "factor_values":
            # UNION factor values
            all_fv = set()
            for v in vals:
                if isinstance(v, list):
                    all_fv.update(v)
                elif isinstance(v, str) and v.lower() not in ("not applicable", "not available"):
                    all_fv.add(v)
            base[field] = sorted(all_fv) if all_fv else "Not Applicable"
            continue

        # For scalar fields
        str_vals = [str(v).strip() if v else "Not Applicable" for v in vals]
        na_set = {"not applicable", "not available", "unknown", "none", ""}

        non_na = [v for v in str_vals if v.lower() not in na_set]

        if not non_na:
            base[field] = "Not Applicable"
        elif len(non_na) == 1:
            # One run found it, other didn't → keep it
            base[field] = non_na[0]
        elif all(v == non_na[0] for v in non_na):
            # Agreement
            base[field] = non_na[0]
        else:
            # Conflict → keep Run 1 (lower temp, more deterministic)
            base[field] = str_vals[0] if str_vals[0].lower() not in na_set else non_na[0]

    return base


ENSEMBLE_TEMPERATURES = [0.1, 0.3]


def run_pipeline(data_dir: str, sample_sub_path: str, output_path: str,
                 model_ids: Optional[list] = None, single_pxd: Optional[str] = None,
                 paper_dir: Optional[str] = None, ensemble_runs: int = 1):
    if model_ids is None:
        model_ids = list(MODELS.keys())

    print("Loading sample submission...")
    with open(sample_sub_path) as f:
        reader = csv.DictReader(f)
        sub_rows = list(reader)

    pxd_rows = defaultdict(list)
    for row in sub_rows:
        pxd_rows[row['PXD']].append(row)

    pxds = sorted(pxd_rows.keys())
    if single_pxd:
        pxds = [p for p in pxds if p == single_pxd]
        if not pxds:
            print(f"ERROR: PXD {single_pxd} not found")
            sys.exit(1)

    print(f"Processing {len(pxds)} PXDs: {pxds}")
    print(f"Models: {model_ids}")
    print(f"Judge: {JUDGE_MODEL_ID}")
    print(f"Max refinement iterations: {MAX_REFINEMENT_ITERATIONS}")
    print(f"Ensemble runs: {ensemble_runs}")

    pxd_graph = build_pxd_graph()

    all_rows = []
    if paper_dir:
        test_dir = paper_dir
    else:
        test_dir = os.path.join(data_dir, "TestPubText")
        if not os.path.isdir(test_dir):
            test_dir = data_dir

    for pxd_idx, pxd in enumerate(pxds):
        print(f"\n{'='*70}")
        print(f"  [{pxd_idx+1}/{len(pxds)}] {pxd} -- {len(pxd_rows[pxd])} expected rows")
        print(f"{'='*70}")

        json_path = os.path.join(test_dir, f"{pxd}_PubText.json")
        if not os.path.exists(json_path):
            print(f"  WARNING: No paper at {json_path}")
            for row in pxd_rows[pxd]:
                r = {c: "Not Applicable" for c in SUBMISSION_COLS}
                r['ID'] = row['ID']
                r['PXD'] = pxd
                r['Raw Data File'] = row['Raw Data File']
                all_rows.append(r)
            continue

        with open(json_path) as f:
            paper = json.load(f)

        raw_files = sorted(set(row['Raw Data File'] for row in pxd_rows[pxd]))

        # Fetch supplementary metadata from PRIDE API
        pride_meta = fetch_pride_metadata(pxd)

        # Ensemble: run the graph N times, merge judge outputs, format once
        judge_outputs = []
        for run_idx in range(ensemble_runs):
            temp = ENSEMBLE_TEMPERATURES[run_idx] if run_idx < len(ENSEMBLE_TEMPERATURES) else 0.1
            if ensemble_runs > 1:
                print(f"\n  --- Ensemble Run {run_idx+1}/{ensemble_runs} (temp={temp}) ---")

            initial_state: PXDState = {
                "pxd": pxd,
                "paper": paper,
                "raw_files": raw_files,
                "sample_rows": pxd_rows[pxd],
                "model_ids": model_ids,
                "pride_meta": pride_meta,
                "data_dir": data_dir,
                "temperature": temp,
                "r1_extractions": [],
                "r2_reconciliations": [],
                "judge_output": {},
                "validation_errors": [],
                "refinement_feedback": "",
                "iteration": 0,
                "max_iterations": MAX_REFINEMENT_ITERATIONS,
                "sdrf_rows": [],
            }

            try:
                final_state = pxd_graph.invoke(initial_state)
                jo = final_state.get("judge_output", {})
                if jo:
                    judge_outputs.append(jo)
            except Exception as e:
                print(f"  ERROR in graph run {run_idx+1}: {e}")

        if judge_outputs:
            # Merge ensemble outputs and format
            if len(judge_outputs) > 1:
                merged = merge_judge_outputs(judge_outputs)
                print(f"  [ENSEMBLE] Merged {len(judge_outputs)} runs")
            else:
                merged = judge_outputs[0]

            # Apply PRIDE overrides and Bayesian defaults on merged output
            if pride_meta:
                _apply_pride_overrides(merged, pride_meta)
            _apply_bayesian_defaults(merged)

            rows = build_sdrf_rows(pxd, raw_files, merged, pxd_rows[pxd])
            print(f"    Generated {len(rows)} rows")
        else:
            rows = []

        if rows:
            all_rows.extend(rows)
        else:
            print(f"  FALLBACK: placeholder rows for {pxd}")
            for row in pxd_rows[pxd]:
                r = {c: "Not Applicable" for c in SUBMISSION_COLS}
                r['ID'] = row['ID']
                r['PXD'] = pxd
                r['Raw Data File'] = row['Raw Data File']
                all_rows.append(r)

    all_rows.sort(key=lambda r: int(r.get('ID', 0)))

    print(f"\nWriting {len(all_rows)} rows to {output_path}...")
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_COLS)
        writer.writeheader()
        for row in all_rows:
            out = {col: row.get(col, "Not Applicable") for col in SUBMISSION_COLS}
            writer.writerow(out)

    print(f"\n{'='*70}")
    print(f"  Done! {len(all_rows)} rows -> {output_path}")
    print(f"  Cost estimate: ~{len(pxds) * 13 * 0.03 * ensemble_runs:.2f} USD")
    print(f"{'='*70}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# CONNECTION TEST
# ═══════════════════════════════════════════════════════════════════════════════

def test_connection():
    print("Testing Azure AI Foundry connections...")
    print(f"Endpoint: {CONFIG['endpoint']}\n")
    success = 0
    for model_id, cfg in {**MODELS, **TIEBREAKER_MODEL}.items():
        print(f"  {cfg['display']} ({model_id})...", end=" ", flush=True)
        try:
            response = call_model(model_id, "You are helpful.", "Say 'OK'.", max_tokens=10)
            if response:
                print(f"[OK] '{response.strip()[:60]}'")
                if model_id in _working_paths:
                    print(f"    path: {_working_paths[model_id]}")
                success += 1
            else:
                print("[FAIL] empty response")
        except Exception as e:
            print(f"[FAIL] {e}")
    print(f"\n{success}/{len(MODELS) + len(TIEBREAKER_MODEL)} connected")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SDRF 3-Round Deliberation Pipeline (LangGraph)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Architecture:
  R1: 3 models extract independently (3 calls)
  R2: 3 models see all R1 outputs, reconcile with confidence (3 calls)
  R3: Claude Opus judge with ontology grounding (1 call)
  Validate -> [Refine -> R3 Judge]* -> Format

  Total: 7 calls/PXD (no errors) to 13 calls/PXD (2 refinement rounds)
        """,
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--sample_sub", default=None)
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--single-pxd", default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--judge-model", default=None,
                        help=f"Model for R3 judge (default: {JUDGE_MODEL_ID})")
    parser.add_argument("--ensemble", type=int, default=1,
                        help="Number of ensemble runs (default: 1)")
    args = parser.parse_args()

    if args.endpoint:
        url = args.endpoint.rstrip('/')
        if '/api/projects/' in url:
            url = url.split('/api/projects/')[0]
        CONFIG["endpoint"] = url
        print(f"Endpoint: {CONFIG['endpoint']}")
    if args.api_key:
        CONFIG["api_key"] = args.api_key
    if args.max_iterations is not None:
        MAX_REFINEMENT_ITERATIONS = args.max_iterations
    if args.judge_model:
        JUDGE_MODEL_ID = args.judge_model
        print(f"Judge model: {JUDGE_MODEL_ID}")

    if args.test:
        test_connection()
        sys.exit(0)

    sample_sub = args.sample_sub or os.path.join(args.data_dir, "SampleSubmission.csv")
    if not os.path.exists(sample_sub):
        print(f"ERROR: SampleSubmission.csv not found at {sample_sub}")
        sys.exit(1)

    run_pipeline(args.data_dir, sample_sub, args.output, args.models, args.single_pxd,
                 ensemble_runs=args.ensemble)
