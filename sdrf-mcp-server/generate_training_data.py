#!/usr/bin/env python3
"""Generate tool-calling training data from gold SDRFs.

For each gold SDRF with an available paper:
1. Load the gold SDRF values
2. Load the paper text
3. Simulate the agentic tool-calling flow:
   - pride_lookup -> get instruments/organisms/mods
   - ms_ontology_lookup -> format instrument accession
   - unimod_lookup -> format modifications
   - sdrf_format_reference -> validate formats
4. Record the complete conversation as a training example

Output: JSONL file with ChatML tool-calling conversations.
"""
import os
import sys
import json
import glob
import re
import time
from collections import defaultdict

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server import pride_lookup, ms_ontology_lookup, unimod_lookup, sdrf_format_reference
from build_databases import (
    parse_training_sdrf, parse_bigbio_sdrf, extract_pxd_from_path,
    TRAINING_TO_SUBMISSION, BIGBIO_TO_SUBMISSION
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "training_data")

# System prompt for the fine-tuned model
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

# Mapping from gold SDRF column names to JSON output keys
COLUMN_TO_KEY = {
    "Characteristics[Organism]": "organism",
    "Characteristics[OrganismPart]": "organism_part",
    "Characteristics[CellType]": "cell_type",
    "Characteristics[CellLine]": "cell_line",
    "Characteristics[Disease]": "disease",
    "Characteristics[Sex]": "sex",
    "Characteristics[Age]": "age",
    "Characteristics[Strain]": "strain",
    "Characteristics[MaterialType]": "material_type",
    "Characteristics[DevelopmentalStage]": "developmental_stage",
    "Characteristics[AncestryCategory]": "ancestry_category",
    "Characteristics[Treatment]": "treatment",
    "Characteristics[Compound]": "compound",
    "Characteristics[GeneticModification]": "genetic_modification",
    "Characteristics[Bait]": "bait",
    "Characteristics[Specimen]": "specimen",
    "Characteristics[Depletion]": "depletion",
    "Characteristics[SpikedCompound]": "spiked_compound",
    "Characteristics[CellPart]": "cell_part",
    "Characteristics[Temperature]": "temperature",
    "Characteristics[Staining]": "staining",
    "Characteristics[SamplingTime]": "sampling_time",
    "Characteristics[Label]": "label",
    "Characteristics[BiologicalReplicate]": "biological_replicate",
    "Characteristics[AlkylationReagent]": "alkylation_reagent",
    "Characteristics[ReductionReagent]": "reduction_reagent",
    "Characteristics[Modification]": "modifications",
    "Characteristics[CleavageAgent]": "cleavage_agent",
    "Comment[Instrument]": "instrument",
    "Comment[FragmentationMethod]": "fragmentation_method",
    "Comment[MS2MassAnalyzer]": "ms2_analyzer",
    "Comment[Separation]": "separation",
    "Comment[FractionationMethod]": "fractionation_method",
    "Comment[CollisionEnergy]": "collision_energy",
    "Comment[PrecursorMassTolerance]": "precursor_mass_tolerance",
    "Comment[FragmentMassTolerance]": "fragment_mass_tolerance",
    "Comment[EnrichmentMethod]": "enrichment_method",
    "Comment[AcquisitionMethod]": "acquisition_method",
    "Comment[IonizationType]": "ionization_type",
    "Comment[NumberOfMissedCleavages]": "number_of_missed_cleavages",
    "Comment[FractionIdentifier]": "fraction_identifier",
}

KEY_TO_COLUMN = {v: k for k, v in COLUMN_TO_KEY.items()}


def load_paper_text(pxd):
    """Load paper text for a PXD from local files."""
    # Try individual file first
    for ext in [".json", ".txt"]:
        path = os.path.join(DATA_DIR, "TrainingPubText", f"{pxd}_PubText{ext}")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                if ext == ".json":
                    data = json.load(f)
                    if isinstance(data, dict):
                        if pxd in data:
                            entry = data[pxd]
                            parts = []
                            if "TITLE" in entry:
                                parts.append(f"Title: {entry['TITLE']}")
                            if "ABSTRACT" in entry:
                                parts.append(f"\nAbstract: {entry['ABSTRACT']}")
                            # Include all other sections
                            for k, v in entry.items():
                                if k not in ("TITLE", "ABSTRACT") and isinstance(v, str):
                                    parts.append(f"\n{k}: {v}")
                            return "\n".join(parts)
                        else:
                            return json.dumps(data)[:15000]
                    return str(data)[:15000]
                else:
                    return f.read()[:15000]

    # Try combined file
    combined_path = os.path.join(DATA_DIR, "TrainingPubText", "PubText.json")
    if os.path.exists(combined_path):
        with open(combined_path, "r", encoding="utf-8") as f:
            all_papers = json.load(f)
        if pxd in all_papers:
            entry = all_papers[pxd]
            parts = []
            if isinstance(entry, dict):
                for k, v in entry.items():
                    if isinstance(v, str):
                        parts.append(f"{k}: {v}")
                return "\n".join(parts)[:15000]
            return str(entry)[:15000]

    return None


def build_gold_output(gold_values):
    """Convert gold SDRF values to the JSON output format."""
    output = {}

    for col, values in gold_values.items():
        key = COLUMN_TO_KEY.get(col)
        if not key:
            continue

        # Deduplicate and sort by most common
        unique_values = list(set(values))

        if key == "modifications":
            # Modifications is a list
            output[key] = unique_values
        elif key in ("biological_replicate", "fraction_identifier"):
            # These are row-specific, use the most common value
            output[key] = unique_values[0] if unique_values else "1"
        else:
            # Single value — use the most common
            output[key] = unique_values[0] if unique_values else "Not Applicable"

    return output


def generate_tool_trace(pxd, gold_values, paper_text):
    """Generate a synthetic tool-calling conversation trace.

    Returns a list of ChatML messages simulating the agentic flow.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract SDRF metadata for {pxd}.\n\nPaper text:\n{paper_text[:12000]}"},
    ]

    gold_output = build_gold_output(gold_values)

    # Step 1: Always call pride_lookup
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_pride",
            "type": "function",
            "function": {
                "name": "pride_lookup",
                "arguments": json.dumps({"pxd": pxd}),
            }
        }]
    })

    # Get real PRIDE response
    try:
        pride_result = pride_lookup(pxd)
        pride_data = json.loads(pride_result)
    except Exception:
        pride_data = {"error": "PRIDE API unavailable"}
        pride_result = json.dumps(pride_data)

    messages.append({
        "role": "tool",
        "tool_call_id": "call_pride",
        "content": pride_result,
    })

    # Step 2: Look up instrument if gold has one
    instrument_value = gold_output.get("instrument", "")
    if instrument_value and "Not Applicable" not in instrument_value:
        # Extract instrument name from the formatted string
        inst_name = instrument_value
        m = re.search(r"NT=([^;]+)", instrument_value)
        if m:
            inst_name = m.group(1)

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_inst",
                "type": "function",
                "function": {
                    "name": "ms_ontology_lookup",
                    "arguments": json.dumps({"term": inst_name, "category": "instrument"}),
                }
            }]
        })

        try:
            inst_result = ms_ontology_lookup(inst_name, "instrument")
        except Exception:
            inst_result = json.dumps({"error": "lookup failed"})

        messages.append({
            "role": "tool",
            "tool_call_id": "call_inst",
            "content": inst_result,
        })

    # Step 3: Look up modifications
    mods = gold_output.get("modifications", [])
    if mods:
        for i, mod_str in enumerate(mods[:4]):  # Limit to 4 mods per trace
            # Extract modification name
            mod_name = mod_str
            m = re.search(r"NT=([^;]+)", mod_str)
            if m:
                mod_name = m.group(1)

            call_id = f"call_mod_{i}"
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "unimod_lookup",
                        "arguments": json.dumps({"modification": mod_name}),
                    }
                }]
            })

            try:
                mod_result = unimod_lookup(mod_name)
            except Exception:
                mod_result = json.dumps({"error": "lookup failed"})

            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": mod_result,
            })

    # Step 4: Optionally look up fragmentation method format
    frag = gold_output.get("fragmentation_method", "")
    if frag and frag not in ("Not Applicable", "not available"):
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_frag_ref",
                "type": "function",
                "function": {
                    "name": "sdrf_format_reference",
                    "arguments": json.dumps({
                        "column": "Comment[FragmentationMethod]",
                        "query": frag[:30],
                    }),
                }
            }]
        })

        try:
            frag_ref_result = sdrf_format_reference("Comment[FragmentationMethod]", frag[:30])
        except Exception:
            frag_ref_result = json.dumps({"error": "lookup failed"})

        messages.append({
            "role": "tool",
            "tool_call_id": "call_frag_ref",
            "content": frag_ref_result,
        })

    # Step 5: Final assistant response with gold output
    final_output = json.dumps(gold_output, indent=2)
    messages.append({
        "role": "assistant",
        "content": f"Based on the paper text and tool lookups, here is the SDRF metadata for {pxd}:\n\n```json\n{final_output}\n```",
    })

    return messages


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect all gold SDRFs with papers
    training_sdrfs = glob.glob(os.path.join(DATA_DIR, "TrainingSDRFs", "*.tsv"))

    print(f"Generating training data from {len(training_sdrfs)} training SDRFs")
    print("=" * 80)

    examples = []
    skipped = 0
    errors = 0

    for sdrf_path in training_sdrfs:
        pxd = extract_pxd_from_path(sdrf_path)

        # Load paper
        paper = load_paper_text(pxd)
        if not paper:
            skipped += 1
            continue

        # Parse gold SDRF
        try:
            _, gold_values = parse_training_sdrf(sdrf_path)
        except Exception as e:
            print(f"  Error parsing {pxd}: {e}")
            errors += 1
            continue

        # Generate tool-calling trace
        try:
            messages = generate_tool_trace(pxd, gold_values, paper)
            examples.append({
                "pxd": pxd,
                "messages": messages,
            })
            n_tools = sum(1 for m in messages if m["role"] == "tool")
            print(f"  {pxd}: {len(messages)} messages, {n_tools} tool calls")
        except Exception as e:
            print(f"  Error generating trace for {pxd}: {e}")
            errors += 1

        # Rate limit for PRIDE API
        time.sleep(0.5)

    # Save as JSONL
    output_path = os.path.join(OUTPUT_DIR, "sdrf_training_data.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n{'='*80}")
    print(f"Generated {len(examples)} training examples")
    print(f"Skipped {skipped} (no paper), {errors} errors")
    print(f"Saved to {output_path}")

    # Also save in the format needed for fine-tuning (messages only, no pxd wrapper)
    ft_path = os.path.join(OUTPUT_DIR, "sdrf_finetune.jsonl")
    with open(ft_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")
    print(f"Fine-tune format saved to {ft_path}")

    # Stats
    total_messages = sum(len(ex["messages"]) for ex in examples)
    total_tool_calls = sum(
        sum(1 for m in ex["messages"] if m["role"] == "tool")
        for ex in examples
    )
    print(f"\nStats:")
    print(f"  Total messages: {total_messages}")
    print(f"  Total tool calls: {total_tool_calls}")
    print(f"  Avg messages/example: {total_messages / len(examples):.1f}")
    print(f"  Avg tool calls/example: {total_tool_calls / len(examples):.1f}")


if __name__ == "__main__":
    main()
