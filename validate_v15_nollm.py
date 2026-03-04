#!/usr/bin/env python3
"""Validate v15 pipeline WITHOUT LLM calls — just defaults + PRIDE.
Baseline should match error_analysis defaults+bio+PRIDE (~0.37)."""
import sys, os, io, csv, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
from evaluate import convert_training_sdrf_to_submission, load_all_training_pxds
from scoring import score as run_scorer
from pipeline_v15 import (
    SUBMISSION_COLS, fetch_pride_metadata, build_rows, pride_to_instrument_sdrf, pride_to_label
)

DATA_DIR = "data"
NUM_PXDS = 10


def make_sample_rows(gold_df, sdrf_path):
    """Create sample rows with raw file names from original SDRF."""
    import pandas as pd
    orig = pd.read_csv(sdrf_path, sep='\t')
    # Get raw files from comment[data file] column
    raw_col = None
    for c in orig.columns:
        if 'data file' in c.lower():
            raw_col = c
            break

    rows = []
    for i, (_, row) in enumerate(gold_df.iterrows()):
        r = {col: "Not Applicable" for col in SUBMISSION_COLS}
        r['ID'] = str(row['ID'])
        r['PXD'] = row['PXD']
        if raw_col and i < len(orig):
            r['Raw Data File'] = str(orig.iloc[i][raw_col])
        else:
            r['Raw Data File'] = f"file_{i+1}.raw"
        rows.append(r)
    return rows


def run_validation():
    pxd_files = load_all_training_pxds(DATA_DIR)
    selected = list(pxd_files.items())[:NUM_PXDS]

    print(f"V15 No-LLM Validation: {len(selected)} training PXDs")
    print("=" * 80)

    all_golds = []
    all_subs = []

    for pxd, sdrf_path in selected:
        gold = convert_training_sdrf_to_submission(sdrf_path, pxd)
        all_golds.append(gold)

        sample_rows = make_sample_rows(gold, sdrf_path)
        pride = fetch_pride_metadata(pxd)
        print(f"  {pxd}: PRIDE organism={pride.get('organism','?')}, instrument={pride.get('instrument','?')}")

        # No LLM — just defaults + PRIDE
        rows = build_rows(pxd, sample_rows, pride, llm_result={})

        sub_df = pd.DataFrame(rows)
        for col in gold.columns:
            if col not in sub_df.columns:
                sub_df[col] = "Not Applicable"
        all_subs.append(sub_df)

        import time
        time.sleep(0.5)

    combined_gold = pd.concat(all_golds, ignore_index=True)
    combined_sub = pd.concat(all_subs, ignore_index=True)
    eval_df, f1 = run_scorer(combined_gold, combined_sub, 'ID')

    print(f"\n{'='*80}")
    print(f"  No-LLM F1 = {f1:.4f}")
    print(f"  Expected (error_analysis sim): ~0.372")

    valid = eval_df.dropna(subset=['f1'])
    col_f1 = valid.groupby('AnnotationType')['f1'].mean().sort_values(ascending=False)

    print(f"\n  {'Column':<45} {'F1':>8}")
    print(f"  {'-'*45} {'------':>8}")
    for col, f1_val in col_f1.items():
        print(f"  {col:<45} {f1_val:>8.3f}")

    n_perfect = len(valid[valid['f1'] >= 0.999])
    n_zero = len(valid[valid['f1'] < 0.001])
    print(f"\n  {len(valid)} scored pairs, {n_perfect} perfect, {n_zero} zeros")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_validation()
