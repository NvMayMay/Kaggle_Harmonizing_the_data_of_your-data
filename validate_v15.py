#!/usr/bin/env python3
"""Validate v15 pipeline against training gold SDRFs."""
import sys, os, io, csv, json, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
from evaluate import convert_training_sdrf_to_submission, load_all_training_pxds
from scoring import score as run_scorer
from pipeline_v15 import (
    SUBMISSION_COLS, fetch_pride_metadata, load_paper_text,
    build_llm_prompt, call_model, parse_json_response, build_rows,
    apply_llm_results, LLM_SYSTEM
)

DATA_DIR = "data"
PAPER_DIR = os.path.join(DATA_DIR, "TrainingPubText")
NUM_PXDS = 10


def make_sample_rows(gold_df, sdrf_path):
    """Create sample rows with raw file names from original SDRF."""
    import pandas as pd
    orig = pd.read_csv(sdrf_path, sep='\t')
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

    print(f"V15 Validation: {len(selected)} training PXDs")
    print("=" * 80)

    all_golds = []
    all_subs = []

    for pxd, sdrf_path in selected:
        print(f"\n{'='*60}")
        print(f"Processing {pxd}")

        # Build gold
        gold = convert_training_sdrf_to_submission(sdrf_path, pxd)
        all_golds.append(gold)

        # Create sample rows
        sample_rows = make_sample_rows(gold, sdrf_path)

        # PRIDE
        pride = fetch_pride_metadata(pxd)
        print(f"  PRIDE: organism={pride.get('organism','?')}, instrument={pride.get('instrument','?')}")

        # Paper
        paper = load_paper_text(PAPER_DIR, pxd)
        if paper:
            print(f"  Paper: {len(paper)} chars")
        else:
            print(f"  Paper: NOT FOUND")

        # LLM
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

        # Build rows
        rows = build_rows(pxd, sample_rows, pride, llm_result)
        print(f"  Built {len(rows)} rows (gold has {len(gold)} rows)")

        # Convert to DataFrame matching gold columns
        sub_df = pd.DataFrame(rows)
        # Ensure all gold columns exist in sub
        for col in gold.columns:
            if col not in sub_df.columns:
                sub_df[col] = "Not Applicable"

        all_subs.append(sub_df)

        # Quick per-PXD score
        try:
            eval_df, f1 = run_scorer(gold, sub_df, 'ID')
            print(f"  PXD F1 = {f1:.4f}")
        except Exception as e:
            print(f"  PXD scoring error: {e}")

        import time
        time.sleep(2)

    # Overall score
    print(f"\n{'='*80}")
    print(f"OVERALL RESULTS")
    print(f"{'='*80}")

    combined_gold = pd.concat(all_golds, ignore_index=True)
    combined_sub = pd.concat(all_subs, ignore_index=True)

    eval_df, f1 = run_scorer(combined_gold, combined_sub, 'ID')

    print(f"\n  Overall F1 = {f1:.4f}")

    # Per-column breakdown
    valid = eval_df.dropna(subset=['f1'])
    col_f1 = valid.groupby('AnnotationType')['f1'].mean().sort_values(ascending=False)

    print(f"\n  {'Column':<45} {'F1':>8} {'Count':>6}")
    print(f"  {'-'*45} {'------':>8} {'-----':>6}")
    for col, f1_val in col_f1.items():
        count = len(valid[valid['AnnotationType'] == col])
        marker = " ***" if f1_val < 0.1 else ""
        print(f"  {col:<45} {f1_val:>8.3f} {count:>6}{marker}")

    # Compare to baselines
    print(f"\n  COMPARISON:")
    print(f"  v6 Kaggle:               0.318")
    print(f"  defaults+bio+PRIDE sim:  0.372")
    print(f"  v15 actual:              {f1:.3f}")

    n_perfect = len(valid[valid['f1'] >= 0.999])
    n_zero = len(valid[valid['f1'] < 0.001])
    print(f"\n  {len(valid)} scored pairs, {n_perfect} perfect, {n_zero} zeros")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_validation()
