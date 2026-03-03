#!/usr/bin/env python3
"""
Local evaluation script for SDRF metadata extraction pipeline.

Fast mode (default): Tests format/ontology layer against training SDRFs.
Full mode: Runs full pipeline on training PXDs, scores against ground truth.

Usage:
  python evaluate.py --data_dir data --mode fast
  python evaluate.py --data_dir data --mode full --num-pxds 5
  python evaluate.py --data_dir data --mode priors   # compute training priors
"""
import argparse, csv, glob, json, os, re, sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import pandas as pd

from scoring import score as run_scorer

# ═══════════════════════════════════════════════════════════════════════════════
# COLUMN NAME MAPPING: Training SDRF → Submission Format
# ═══════════════════════════════════════════════════════════════════════════════

TRAINING_TO_SUBMISSION = {
    # Characteristics columns
    "Organism": "Characteristics[Organism]",
    "OrganismPart": "Characteristics[OrganismPart]",
    "CellType": "Characteristics[CellType]",
    "CellLine": "Characteristics[CellLine]",
    "Disease": "Characteristics[Disease]",
    "Sex": "Characteristics[Sex]",
    "Age": "Characteristics[Age]",
    "Strain": "Characteristics[Strain]",
    "MaterialType": "Characteristics[MaterialType]",
    "Label": "Characteristics[Label]",
    "BiologicalReplicate": "Characteristics[BiologicalReplicate]",
    "Treatment": "Characteristics[Treatment]",
    "Compound": "Characteristics[Compound]",
    "Temperature": "Characteristics[Temperature]",
    "SpikedCompound": "Characteristics[SpikedCompound]",
    "Specimen": "Characteristics[Specimen]",
    "PooledSample": "Characteristics[PooledSample]",
    "Depletion": "Characteristics[Depletion]",
    "Bait": "Characteristics[Bait]",
    "AlkylationReagent": "Characteristics[AlkylationReagent]",
    "ReductionReagent": "Characteristics[ReductionReagent]",
    "SamplingTime": "Characteristics[SamplingTime]",
    "CellPart": "Characteristics[CellPart]",
    "GrowthRate": "Characteristics[GrowthRate]",
    "Staining": "Characteristics[Staining]",
    "SyntheticPeptide": "Characteristics[SyntheticPeptide]",
    "Time": "Characteristics[Time]",
    "BMI": "Characteristics[BMI]",
    "TumorSize": "Characteristics[TumorSize]",
    "DevelopmentalStage": "Characteristics[DevelopmentalStage]",
    "AncestryCategory": "Characteristics[AncestryCategory]",
    # Comment columns
    "Instrument": "Comment[Instrument]",
    "CleavageAgent": "Characteristics[CleavageAgent]",
    "FragmentationMethod": "Comment[FragmentationMethod]",
    "FractionIdentifier": "Comment[FractionIdentifier]",
    "FractionationMethod": "Comment[FractionationMethod]",
    "MS2MassAnalyzer": "Comment[MS2MassAnalyzer]",
    "Separation": "Comment[Separation]",
    "CollisionEnergy": "Comment[CollisionEnergy]",
    "PrecursorMassTolerance": "Comment[PrecursorMassTolerance]",
    "FragmentMassTolerance": "Comment[FragmentMassTolerance]",
    "EnrichmentMethod": "Comment[EnrichmentMethod]",
    "NumberOfMissedCleavages": "Comment[NumberOfMissedCleavages]",
    # Modification is special — handled separately (multiple columns with same name)
    "Modification": "Characteristics[Modification]",
    # Skip these (not scored)
    "SourceName": None,
    "AssayName": None,
    "TechnicalReplicate": None,
    "technology type": None,
    "comment[data file]": None,
    "comment[file uri]": None,
    "comment[file url]": None,
    "comment[proteomexchange accession number]": None,
    "comment[associated file uri]": None,
    "characteristics[individual]": None,
    "Experiment": None,
}

# Factor value columns from training data
FACTOR_VALUE_MAPPING = {
    "factor value[phenotype]": "FactorValue[Treatment]",
    "factor value[individual]": None,  # skip
    "factor value[pool]": None,
    "factor value[ disease response]": "FactorValue[Disease]",
    "factor value[HLA]": None,
    "factor value[chemical entity]": "FactorValue[Compound]",
    "factor value[induced by]": None,
    "factor value[isolation width]": None,
    "factor value[multiplicities of infection]": None,
    "factor value[overproduction]": None,
    "factor value[overproduction].1": None,
    "factor value[protocol]": None,
    "factor value[subtype]": None,
}


def convert_training_sdrf_to_submission(sdrf_path: str, pxd: str) -> pd.DataFrame:
    """Read a training SDRF TSV and convert to submission column format."""
    df = pd.read_csv(sdrf_path, sep='\t')

    # Build the renamed DataFrame
    new_cols = {}
    mod_cols = []  # track modification columns separately

    for col in df.columns:
        if col in TRAINING_TO_SUBMISSION:
            mapped = TRAINING_TO_SUBMISSION[col]
            if mapped is None:
                continue  # skip
            if col == "Modification":
                mod_cols.append(col)
                continue  # handle below
            if mapped in new_cols:
                continue  # skip duplicate
            new_cols[mapped] = df[col]
        elif col in FACTOR_VALUE_MAPPING:
            mapped = FACTOR_VALUE_MAPPING[col]
            if mapped and mapped not in new_cols:
                new_cols[mapped] = df[col]
        elif col.startswith("characteristics[") or col.startswith("comment[") or col.startswith("factor value["):
            # Pass through lowercase bracket columns — try to map
            pass  # these are uncommon special columns, skip for now
        # else: skip unknown columns

    # Handle Modification columns (training SDRFs have multiple with same name "Modification")
    # pandas reads them as Modification, Modification.1, etc.
    mod_col_names = [c for c in df.columns if c == "Modification" or re.match(r'^Modification\.\d+$', c)]
    for i, mc in enumerate(mod_col_names):
        suffix = "" if i == 0 else f".{i}"
        sub_col = f"Characteristics[Modification]{suffix}"
        new_cols[sub_col] = df[mc]

    result = pd.DataFrame(new_cols)
    result.insert(0, 'PXD', pxd)
    result.insert(0, 'ID', range(len(result)))

    # Convert numeric columns to clean strings (avoid float artifacts like "1.0" from int→float)
    for col in result.columns:
        if result[col].dtype in ('float64', 'int64'):
            # Convert non-NaN values to int strings, preserve NaN
            result[col] = result[col].apply(
                lambda x: str(int(x)) if pd.notna(x) else x
            )

    # Fill NaN with "Not Applicable"
    result = result.fillna("Not Applicable")

    return result


def load_all_training_pxds(data_dir: str) -> Dict[str, str]:
    """Return dict of PXD -> SDRF file path for all training PXDs."""
    sdrf_dir = os.path.join(data_dir, "TrainingSDRFs")
    pxds = {}
    for fp in sorted(glob.glob(os.path.join(sdrf_dir, "PXD*_cleaned.sdrf.tsv"))):
        pxd = os.path.basename(fp).split("_")[0]
        pxds[pxd] = fp
    return pxds


# ═══════════════════════════════════════════════════════════════════════════════
# FAST MODE: Test format layer only
# ═══════════════════════════════════════════════════════════════════════════════

def run_fast_evaluation(data_dir: str, num_pxds: int = 0):
    """Test format functions against training ground truth. Zero API calls."""
    from pipeline_merged import (
        format_instrument, format_cleavage, format_modification,
        format_fragmentation, format_label, format_acquisition,
        format_ms2_analyzer, format_fractionation, format_separation,
        format_ionization,
    )

    pxd_files = load_all_training_pxds(data_dir)
    if num_pxds > 0:
        pxd_files = dict(list(pxd_files.items())[:num_pxds])

    print(f"Fast evaluation: {len(pxd_files)} training PXDs")
    print(f"Testing format/ontology layer only (no API calls)\n")

    # Format function mapping: submission column -> format function
    FORMAT_FNS = {
        "Comment[Instrument]": format_instrument,
        "Characteristics[CleavageAgent]": format_cleavage,
        "Comment[FragmentationMethod]": format_fragmentation,
        "Characteristics[Label]": format_label,
        "Comment[AcquisitionMethod]": format_acquisition,
        "Comment[MS2MassAnalyzer]": format_ms2_analyzer,
        "Comment[FractionationMethod]": format_fractionation,
        "Comment[Separation]": format_separation,
        "Comment[IonizationType]": format_ionization,
    }

    all_solution_rows = []
    all_submission_rows = []

    for pxd, sdrf_path in pxd_files.items():
        # Convert training SDRF to submission format = ground truth
        solution_df = convert_training_sdrf_to_submission(sdrf_path, pxd)

        # Create formatted version (apply our format functions)
        submission_df = solution_df.copy()
        for col, fmt_fn in FORMAT_FNS.items():
            if col in submission_df.columns:
                submission_df[col] = submission_df[col].apply(
                    lambda v: fmt_fn(v) if v != "Not Applicable" else v
                )

        # Handle modifications specially
        for mc in [c for c in submission_df.columns if c.startswith("Characteristics[Modification]")]:
            submission_df[mc] = submission_df[mc].apply(
                lambda v: format_modification(v) if v not in ("Not Applicable", "") else v
            )

        all_solution_rows.append(solution_df)
        all_submission_rows.append(submission_df)

    # Combine all PXDs
    solution_combined = pd.concat(all_solution_rows, ignore_index=True)
    submission_combined = pd.concat(all_submission_rows, ignore_index=True)

    # Score
    eval_df, overall_f1 = run_scorer(solution_combined, submission_combined, 'ID')

    # Report
    print(f"{'='*70}")
    print(f"  FAST MODE RESULTS: Overall F1 = {overall_f1:.4f}")
    print(f"  ({len(eval_df)} evaluated (PXD, column) pairs)")
    print(f"{'='*70}\n")

    # Show columns where formatting HURTS (F1 < 1.0)
    hurt_cols = eval_df[eval_df['f1'] < 1.0].copy()
    if not hurt_cols.empty:
        print(f"Columns where format functions CHANGE the score (F1 < 1.0):")
        col_avg = hurt_cols.groupby('AnnotationType')['f1'].agg(['mean', 'count']).sort_values('mean')
        for col, row in col_avg.iterrows():
            print(f"  {col}: avg F1={row['mean']:.3f} ({int(row['count'])} PXDs)")
    else:
        print("All format functions preserve ground truth perfectly!")

    print()

    # Show per-column average F1
    print("Per-column average F1:")
    col_scores = eval_df.groupby('AnnotationType')['f1'].mean().sort_values()
    for col, f1 in col_scores.items():
        marker = " ***" if f1 < 0.9 else ""
        print(f"  {f1:.3f}  {col}{marker}")

    return eval_df, overall_f1


# ═══════════════════════════════════════════════════════════════════════════════
# PRIORS MODE: Compute training data statistics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_priors(data_dir: str):
    """Compute Bayesian priors from training data."""
    pxd_files = load_all_training_pxds(data_dir)
    n_pxds = len(pxd_files)
    print(f"Computing priors from {n_pxds} training PXDs\n")

    # Column activity: how many PXDs have a real value (not NA) for each column
    col_activity = Counter()  # column -> count of PXDs with real values
    col_values = defaultdict(Counter)  # column -> value -> count

    for pxd, sdrf_path in pxd_files.items():
        df = convert_training_sdrf_to_submission(sdrf_path, pxd)
        for col in df.columns:
            if col in ('ID', 'PXD'):
                continue
            unique_vals = df[col].dropna().unique()
            real_vals = [v for v in unique_vals if str(v).lower() not in ('not applicable', '')]
            if real_vals:
                col_activity[col] += 1
                for v in real_vals:
                    # Normalize: extract NT= value for comparison
                    vstr = str(v).strip()
                    if 'NT=' in vstr:
                        parts = [r for r in vstr.split(';') if 'NT=' in r]
                        vstr = parts[0].replace('NT=', '').strip() if parts else vstr
                    col_values[col][vstr] += 1

    # Report column activity rates
    print(f"{'='*70}")
    print(f"  COLUMN ACTIVITY RATES (P(column has real value))")
    print(f"{'='*70}\n")

    activity_rates = {}
    for col in sorted(col_activity.keys(), key=lambda c: -col_activity[c]):
        rate = col_activity[col] / n_pxds
        activity_rates[col] = rate
        marker = ""
        if rate < 0.10:
            marker = " [RARE - default to Not Applicable]"
        elif rate < 0.25:
            marker = " [uncommon]"
        print(f"  {rate:.2f}  ({col_activity[col]:3d}/{n_pxds})  {col}{marker}")

    # Report top values per key column
    print(f"\n{'='*70}")
    print(f"  VALUE FREQUENCY PRIORS")
    print(f"{'='*70}\n")

    key_cols = [
        'Characteristics[Organism]', 'Comment[Instrument]',
        'Characteristics[CleavageAgent]', 'Characteristics[MaterialType]',
        'Characteristics[Label]', 'Comment[FragmentationMethod]',
        'Comment[MS2MassAnalyzer]', 'Comment[Separation]',
        'Comment[FractionationMethod]',
    ]

    value_priors = {}
    for col in key_cols:
        if col in col_values:
            total = sum(col_values[col].values())
            print(f"{col}:")
            priors = {}
            for val, cnt in col_values[col].most_common(10):
                freq = cnt / total
                priors[val] = freq
                print(f"  {freq:.2f}  ({cnt:4d}x)  {val[:70]}")
            value_priors[col] = priors
            print()

    return activity_rates, value_priors


# ═══════════════════════════════════════════════════════════════════════════════
# FULL MODE: Run pipeline on training PXDs
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_evaluation(data_dir: str, num_pxds: int = 5, models: list = None):
    """Run full pipeline on training PXDs and score against ground truth."""
    from pipeline_merged import run_pipeline, SUBMISSION_COLS

    pxd_files = load_all_training_pxds(data_dir)
    pub_text_dir = os.path.join(data_dir, "TrainingPubText")

    # Filter to PXDs that have paper text available
    available = {}
    for pxd, sdrf_path in pxd_files.items():
        json_path = os.path.join(pub_text_dir, f"{pxd}_PubText.json")
        if os.path.exists(json_path):
            available[pxd] = sdrf_path

    # Select subset
    selected = dict(list(available.items())[:num_pxds])
    print(f"Full evaluation: {len(selected)} training PXDs (of {len(available)} available)")
    print(f"PXDs: {list(selected.keys())}\n")

    # Build a fake SampleSubmission from training SDRFs
    sample_rows = []
    all_solution_rows = []
    row_id = 0

    for pxd, sdrf_path in selected.items():
        # Read training SDRF for raw file names
        df = pd.read_csv(sdrf_path, sep='\t')
        data_file_col = None
        for c in df.columns:
            if 'data file' in c.lower():
                data_file_col = c
                break

        if data_file_col is None:
            print(f"  WARNING: {pxd} has no data file column, skipping")
            continue

        raw_files = df[data_file_col].dropna().unique()

        for rf in raw_files:
            sample_rows.append({
                'ID': str(row_id),
                'PXD': pxd,
                'Raw Data File': rf,
            })
            row_id += 1

        # Convert ground truth to submission format
        solution_df = convert_training_sdrf_to_submission(sdrf_path, pxd)
        all_solution_rows.append(solution_df)

    # Write temporary sample submission
    tmp_sample = os.path.join(data_dir, "_tmp_eval_sample.csv")
    with open(tmp_sample, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_COLS)
        writer.writeheader()
        for row in sample_rows:
            out = {c: "Not Applicable" for c in SUBMISSION_COLS}
            out.update(row)
            writer.writerow(out)

    # Run pipeline
    tmp_output = os.path.join(data_dir, "_tmp_eval_output.csv")
    print("Running pipeline...")
    run_pipeline(
        data_dir=data_dir,
        sample_sub_path=tmp_sample,
        output_path=tmp_output,
        model_ids=models,
        paper_dir=pub_text_dir,
    )

    # Score
    solution_combined = pd.concat(all_solution_rows, ignore_index=True)
    # CRITICAL: fill NaN introduced by concat (columns that don't exist in some PXDs)
    # Without this, load_sdrf sees empty unique sets instead of "Not Applicable"
    solution_combined = solution_combined.fillna("Not Applicable")
    submission_df = pd.read_csv(tmp_output)

    eval_df, overall_f1 = run_scorer(solution_combined, submission_df, 'ID')

    # Report
    print(f"\n{'='*70}")
    print(f"  FULL MODE RESULTS: Overall F1 = {overall_f1:.4f}")
    print(f"  ({len(eval_df)} evaluated (PXD, column) pairs)")
    print(f"{'='*70}\n")

    # Per-PXD scores
    print("Per-PXD scores:")
    pxd_scores = eval_df.groupby('pxd')['f1'].mean().sort_values()
    for pxd, f1 in pxd_scores.items():
        print(f"  {f1:.3f}  {pxd}")

    print()

    # Per-column (worst first)
    print("Per-column average F1 (worst first, showing F1 < 0.9):")
    col_scores = eval_df.groupby('AnnotationType')['f1'].mean().sort_values()
    for col, f1 in col_scores.items():
        if f1 < 0.9:
            print(f"  {f1:.3f}  {col}")

    # Cleanup
    for tmp in [tmp_sample, tmp_output]:
        if os.path.exists(tmp):
            os.remove(tmp)

    return eval_df, overall_f1


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local SDRF evaluation")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--mode", choices=["fast", "full", "priors"], default="fast")
    parser.add_argument("--num-pxds", type=int, default=0, help="Limit number of PXDs (0=all)")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--output", default=None, help="Save eval CSV")
    args = parser.parse_args()

    if args.mode == "fast":
        eval_df, f1 = run_fast_evaluation(args.data_dir, args.num_pxds)
    elif args.mode == "priors":
        compute_priors(args.data_dir)
        sys.exit(0)
    elif args.mode == "full":
        n = args.num_pxds or 5
        eval_df, f1 = run_full_evaluation(args.data_dir, n, args.models)

    if args.output and 'eval_df' in dir():
        eval_df.to_csv(args.output, index=False)
        print(f"\nSaved evaluation to {args.output}")
