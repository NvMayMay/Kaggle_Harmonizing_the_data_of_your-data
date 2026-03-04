#!/usr/bin/env python3
"""Agentic SDRF inference pipeline.

Uses a fine-tuned model (or GPT-4.1 fallback) with MCP tools to extract
SDRF metadata from proteomics papers.

Usage:
  # With GPT-4.1 (Azure OpenAI) — no fine-tuned model needed
  python inference_pipeline.py --model gpt4 --pxd PXD000070

  # With fine-tuned model on RunPod
  python inference_pipeline.py --model runpod --pxd PXD000070

  # Full submission run (all test PXDs)
  python inference_pipeline.py --model gpt4 --submit
"""
import os
import sys
import json
import csv
import re
import time
import argparse
from typing import Dict, List, Optional

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from server import (
    pride_lookup, ms_ontology_lookup, unimod_lookup,
    paper_fetch, sdrf_format_reference,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

# Tool definitions for OpenAI-compatible function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "pride_lookup",
            "description": "Fetch authoritative metadata from PRIDE REST API for a proteomics dataset. Returns instruments, organisms, modifications, quantification, and protocols.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pxd": {"type": "string", "description": "ProteomeXchange accession (e.g., PXD000070)"}
                },
                "required": ["pxd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ms_ontology_lookup",
            "description": "Look up PSI-MS accession codes for instruments, fragmentation methods, analyzers, ionization types, and cleavage agents. Returns formatted SDRF string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {"type": "string", "description": "Term to search for (e.g., Q Exactive HF, HCD, Orbitrap)"},
                    "category": {"type": "string", "description": "Optional: instrument, fragmentation, analyzer, ionization, cleavage_agent", "default": ""},
                },
                "required": ["term"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unimod_lookup",
            "description": "Look up UNIMOD modification entries. Returns formatted SDRF modification string with accession code, targets, and type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "modification": {"type": "string", "description": "Modification name (e.g., Carbamidomethyl, phosphorylation, oxidation)"}
                },
                "required": ["modification"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "paper_fetch",
            "description": "Retrieve paper text for a PXD. Checks local files first, then EuropePMC.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pxd": {"type": "string", "description": "ProteomeXchange accession"}
                },
                "required": ["pxd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sdrf_format_reference",
            "description": "Look up canonical SDRF-formatted values from 374+ gold SDRFs. Use to validate format strings match competition expectations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "SDRF column name (e.g., Comment[Instrument])"},
                    "query": {"type": "string", "description": "Search term to filter values", "default": ""},
                },
                "required": ["column"],
            },
        },
    },
]

# Tool dispatch
TOOL_DISPATCH = {
    "pride_lookup": lambda args: pride_lookup(**args),
    "ms_ontology_lookup": lambda args: ms_ontology_lookup(**args),
    "unimod_lookup": lambda args: unimod_lookup(**args),
    "paper_fetch": lambda args: paper_fetch(**args),
    "sdrf_format_reference": lambda args: sdrf_format_reference(**args),
}

# System prompt (same as training)
SYSTEM_PROMPT = """You are an expert proteomics SDRF metadata extraction agent. Your task is to extract structured metadata from proteomics papers and format it according to SDRF (Sample and Data Relationship Format) standards.

You have access to these tools:
- pride_lookup: Fetch authoritative metadata (instruments, organisms, modifications) from PRIDE
- ms_ontology_lookup: Look up PSI-MS accession codes for instruments, fragmentation methods, analyzers
- unimod_lookup: Look up UNIMOD modification entries with formatted strings
- sdrf_format_reference: Look up canonical SDRF format values from the gold corpus

Your workflow:
1. First call pride_lookup to get authoritative instrument/organism data
2. Use ms_ontology_lookup to get exact accession codes for instruments and fragmentation
3. Use unimod_lookup to format modification strings
4. Read the paper to extract biology (organism part, cell type, disease, etc.)
5. Use sdrf_format_reference to validate your format strings
6. Output final JSON with all SDRF column values

Output format - a JSON object with these keys (use exact column names):
- organism, organism_part, cell_type, cell_line, disease, sex, age, strain
- material_type, developmental_stage, ancestry_category
- treatment, compound, genetic_modification, bait, specimen, depletion
- spiked_compound, cell_part, temperature, staining, sampling_time
- instrument, cleavage_agent, fragmentation_method, ms2_analyzer
- separation, fractionation_method, collision_energy
- precursor_mass_tolerance, fragment_mass_tolerance
- enrichment_method, acquisition_method, ionization_type
- number_of_missed_cleavages
- modifications (list of formatted UNIMOD strings)
- label, biological_replicate
- alkylation_reagent, reduction_reagent

Use "not available" when the paper doesn't mention a biology field but it's commonly filled.
Use "Not Applicable" when a field is genuinely not relevant to this experiment.
Never guess ontology accession codes — always use the lookup tools."""

# JSON key → submission column mapping
KEY_TO_SUBMISSION = {
    "organism": "Characteristics[Organism]",
    "organism_part": "Characteristics[OrganismPart]",
    "cell_type": "Characteristics[CellType]",
    "cell_line": "Characteristics[CellLine]",
    "disease": "Characteristics[Disease]",
    "sex": "Characteristics[Sex]",
    "age": "Characteristics[Age]",
    "strain": "Characteristics[Strain]",
    "material_type": "Characteristics[MaterialType]",
    "developmental_stage": "Characteristics[DevelopmentalStage]",
    "ancestry_category": "Characteristics[AncestryCategory]",
    "treatment": "Characteristics[Treatment]",
    "compound": "Characteristics[Compound]",
    "genetic_modification": "Characteristics[GeneticModification]",
    "bait": "Characteristics[Bait]",
    "specimen": "Characteristics[Specimen]",
    "depletion": "Characteristics[Depletion]",
    "spiked_compound": "Characteristics[SpikedCompound]",
    "cell_part": "Characteristics[CellPart]",
    "temperature": "Characteristics[Temperature]",
    "staining": "Characteristics[Staining]",
    "sampling_time": "Characteristics[SamplingTime]",
    "label": "Characteristics[Label]",
    "biological_replicate": "Characteristics[BiologicalReplicate]",
    "alkylation_reagent": "Characteristics[AlkylationReagent]",
    "reduction_reagent": "Characteristics[ReductionReagent]",
    "instrument": "Comment[Instrument]",
    "cleavage_agent": "Characteristics[CleavageAgent]",
    "fragmentation_method": "Comment[FragmentationMethod]",
    "ms2_analyzer": "Comment[MS2MassAnalyzer]",
    "separation": "Comment[Separation]",
    "fractionation_method": "Comment[FractionationMethod]",
    "collision_energy": "Comment[CollisionEnergy]",
    "precursor_mass_tolerance": "Comment[PrecursorMassTolerance]",
    "fragment_mass_tolerance": "Comment[FragmentMassTolerance]",
    "enrichment_method": "Comment[EnrichmentMethod]",
    "acquisition_method": "Comment[AcquisitionMethod]",
    "ionization_type": "Comment[IonizationType]",
    "number_of_missed_cleavages": "Comment[NumberOfMissedCleavages]",
    "fraction_identifier": "Comment[FractionIdentifier]",
}


# ============================================================================
# Model Clients
# ============================================================================

def create_client(model_type: str):
    """Create an OpenAI-compatible client."""
    if model_type == "gpt4":
        from openai import AzureOpenAI
        endpoint = os.environ.get(
            "AZURE_OPENAI_ENDPOINT",
            "https://data-synthesis-foundry-east-us-2.services.ai.azure.com"
        )
        api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2025-01-01-preview",
        ), "gpt-4.1"

    elif model_type == "runpod":
        from openai import OpenAI
        endpoint = os.environ.get(
            "RUNPOD_ENDPOINT",
            "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1"
        )
        api_key = os.environ.get("RUNPOD_API_KEY", "")
        return OpenAI(
            base_url=endpoint,
            api_key=api_key,
        ), "default"

    elif model_type == "local":
        from openai import OpenAI
        return OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="none",
        ), "default"

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def call_model_with_tools(client, model_name: str, messages: list, max_iterations: int = 8):
    """Run the agentic loop: model generates, tools execute, repeat."""
    for iteration in range(max_iterations):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=4096,
            )
        except Exception as e:
            print(f"    Model call failed (iteration {iteration}): {e}")
            time.sleep(3)
            continue

        choice = response.choices[0]

        # Check if model wants to call tools
        if choice.finish_reason == "tool_calls" or (choice.message.tool_calls and len(choice.message.tool_calls) > 0):
            # Add assistant message with tool calls
            messages.append(choice.message.model_dump())

            # Execute each tool call
            for tool_call in choice.message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                print(f"    Tool: {fn_name}({json.dumps(fn_args)[:60]})")

                if fn_name in TOOL_DISPATCH:
                    try:
                        result = TOOL_DISPATCH[fn_name](fn_args)
                    except Exception as e:
                        result = json.dumps({"error": str(e)})
                else:
                    result = json.dumps({"error": f"Unknown tool: {fn_name}"})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        else:
            # Model produced final output
            content = choice.message.content or ""
            return content

    return None


def parse_model_output(content: str) -> dict:
    """Parse the model's final JSON output."""
    if not content:
        return {}

    # Try to extract JSON from markdown code block
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try direct JSON parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    print(f"    WARNING: Could not parse model output")
    return {}


# ============================================================================
# Submission Builder
# ============================================================================

# Submission columns (from SampleSubmission.csv)
SUBMISSION_COLS = None  # Loaded dynamically


def load_submission_template():
    """Load the SampleSubmission.csv to get row structure."""
    global SUBMISSION_COLS
    template_path = os.path.join(DATA_DIR, "SampleSubmission.csv")
    import pandas as pd
    df = pd.read_csv(template_path)
    SUBMISSION_COLS = list(df.columns)
    return df


def build_submission_rows(template_df, pxd, model_output: dict) -> list:
    """Build submission rows for a PXD from model output.

    Uses the template to get row count, IDs, and raw file names.
    """
    pxd_rows = template_df[template_df["PXD"] == pxd]
    if pxd_rows.empty:
        print(f"    WARNING: No rows for {pxd} in template")
        return []

    rows = []
    for _, template_row in pxd_rows.iterrows():
        row = {}
        for col in SUBMISSION_COLS:
            row[col] = str(template_row.get(col, "Not Applicable"))

        # Apply model output
        for key, sub_col in KEY_TO_SUBMISSION.items():
            value = model_output.get(key)
            if value is None:
                continue

            if key == "modifications" and isinstance(value, list):
                # Multi-value: fill Characteristics[Modification], .1, .2, etc.
                mod_cols = [c for c in SUBMISSION_COLS if c.startswith("Characteristics[Modification]")]
                for i, mod_val in enumerate(value):
                    if i < len(mod_cols):
                        row[mod_cols[i]] = str(mod_val)
            elif sub_col in SUBMISSION_COLS:
                if isinstance(value, list):
                    value = value[0] if value else "Not Applicable"
                row[sub_col] = str(value)

        rows.append(row)

    return rows


# ============================================================================
# Protected defaults (same as pipeline_v15)
# ============================================================================

PROTECTED_DEFAULTS = {
    "Characteristics[CleavageAgent]": "AC=MS:1001251;NT=Trypsin",
    "Comment[Separation]": "AC=PRIDE:0000563;NT=Reversed-phase chromatography",
    "Characteristics[Sex]": "not available",
    "Characteristics[AncestryCategory]": "not available",
    "Characteristics[Age]": "not available",
    "Characteristics[DevelopmentalStage]": "not available",
    "Characteristics[AlkylationReagent]": "iodoacetamide",
    "Characteristics[ReductionReagent]": "DTT",
    "Comment[NumberOfMissedCleavages]": "2",
}


def apply_protected_defaults(rows: list):
    """Apply protected defaults — these NEVER get overridden by model output."""
    for row in rows:
        for col, default in PROTECTED_DEFAULTS.items():
            if col in row:
                current = row[col]
                # Only apply if current is empty, "Not Applicable", or suspicious
                if current in ("Not Applicable", "", "nan"):
                    row[col] = default


# ============================================================================
# Main Pipeline
# ============================================================================

def process_pxd(client, model_name: str, pxd: str, paper_text: str) -> dict:
    """Run the agentic extraction pipeline for one PXD."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract SDRF metadata for {pxd}.\n\nPaper text:\n{paper_text[:12000]}"},
    ]

    print(f"  Calling model with tools for {pxd}...")
    content = call_model_with_tools(client, model_name, messages)

    if content:
        result = parse_model_output(content)
        if result:
            print(f"  Extracted {len(result)} fields")
            return result

    print(f"  WARNING: Model returned no valid output for {pxd}")
    return {}


def main():
    parser = argparse.ArgumentParser(description="Agentic SDRF Inference Pipeline")
    parser.add_argument("--model", choices=["gpt4", "runpod", "local"], default="gpt4")
    parser.add_argument("--pxd", help="Process a single PXD")
    parser.add_argument("--submit", action="store_true", help="Process all test PXDs and generate submission")
    parser.add_argument("--output", default="submission_agentic.csv", help="Output CSV path")
    args = parser.parse_args()

    # Load template
    template_df = load_submission_template()
    test_pxds = sorted(template_df["PXD"].unique())
    print(f"Template: {len(template_df)} rows, {len(test_pxds)} PXDs")

    # Create model client
    client, model_name = create_client(args.model)
    print(f"Model: {args.model} ({model_name})")

    # Select PXDs to process
    if args.pxd:
        pxds = [args.pxd]
    elif args.submit:
        pxds = test_pxds
    else:
        pxds = test_pxds[:3]  # Default: first 3 for testing

    print(f"\nProcessing {len(pxds)} PXDs...")
    print("=" * 60)

    all_rows = []

    for pxd in pxds:
        print(f"\n--- {pxd} ---")

        # Fetch paper text
        paper_result = json.loads(paper_fetch(pxd))
        paper_text = paper_result.get("text", "")
        if not paper_text:
            print(f"  No paper found, using PRIDE protocol only")
            pride_result = json.loads(pride_lookup(pxd))
            paper_text = f"Sample protocol: {pride_result.get('sample_protocol', '')}\nData protocol: {pride_result.get('data_protocol', '')}"

        # Run agentic extraction
        model_output = process_pxd(client, model_name, pxd, paper_text)

        # Build submission rows
        rows = build_submission_rows(template_df, pxd, model_output)
        apply_protected_defaults(rows)

        print(f"  Built {len(rows)} submission rows")
        all_rows.extend(rows)

        time.sleep(2)  # Rate limit

    # Write submission CSV
    output_path = os.path.join(SCRIPT_DIR, "..", args.output)
    import pandas as pd
    sub_df = pd.DataFrame(all_rows)

    # Ensure columns match template
    for col in SUBMISSION_COLS:
        if col not in sub_df.columns:
            sub_df[col] = "Not Applicable"
    sub_df = sub_df[SUBMISSION_COLS]

    sub_df.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"Submission saved: {output_path}")
    print(f"  Rows: {len(sub_df)}")
    print(f"  PXDs: {sub_df['PXD'].nunique()}")

    # Summary
    for pxd in pxds:
        pxd_rows = sub_df[sub_df["PXD"] == pxd]
        print(f"  {pxd}: {len(pxd_rows)} rows")


if __name__ == "__main__":
    main()
