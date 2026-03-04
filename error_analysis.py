#!/usr/bin/env python3
"""Error analysis: score different strategies against training gold.
No API calls needed — uses gold SDRFs to estimate per-column impact.
Now includes simulated PRIDE and LLM-damage analysis."""
import sys, io, os, re, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
from collections import defaultdict
from evaluate import convert_training_sdrf_to_submission, load_all_training_pxds
from scoring import score as run_scorer, load_sdrf

DATA_DIR = "data"

# Most common values from gold corpus analysis
CORPUS_DEFAULTS = {
    "Characteristics[CleavageAgent]": "AC=MS:1001251;NT=Trypsin",
    "Comment[FragmentationMethod]": "AC=MS:1000422;NT=HCD",
    "Comment[MS2MassAnalyzer]": "AC=MS:1000484; NT=Orbitrap",
    "Comment[Separation]": "AC=PRIDE:0000563;NT=Reversed-phase chromatography",
    "Comment[PrecursorMassTolerance]": "10 ppm",
    "Comment[FragmentMassTolerance]": "0.02 Da",
    "Comment[CollisionEnergy]": "30% NCE",
    "Comment[FractionationMethod]": "no fractionation",
    "Comment[EnrichmentMethod]": "no enrichment",
    "Comment[NumberOfMissedCleavages]": "2",
    "Characteristics[MaterialType]": "tissue",
    "Characteristics[AlkylationReagent]": "iodoacetamide",
    "Characteristics[ReductionReagent]": "DTT",
    "Comment[AcquisitionMethod]": "DDA",
    "Comment[IonizationType]": "electrospray ionization",
}

BIOLOGY_DEFAULTS = {
    "Characteristics[Disease]": "not available",
    "Characteristics[Sex]": "not available",
    "Characteristics[Age]": "not available",
    "Characteristics[CellType]": "not available",
    "Characteristics[CellLine]": "not available",
    "Characteristics[OrganismPart]": "not available",
    "Characteristics[DevelopmentalStage]": "not available",
    "Characteristics[AncestryCategory]": "not available",
    "Characteristics[Strain]": "not available",
}

MOD_DEFAULTS = [
    "NT=Carbamidomethyl;TA=C;MT=Fixed;AC=UNIMOD:4",
    "NT=Oxidation;MT=Variable;TA=M;AC=UNIMOD:35",
]

# Simulate what a bad LLM would do: replace correct defaults with wrong guesses
LLM_DAMAGE_SCENARIOS = {
    # Column: (probability of wrong guess, wrong value)
    # These simulate what v6's LLM might do to columns where defaults are already scoring well
    "Characteristics[CleavageAgent]": (0.1, "NT=Lys-C;AC=MS:1001309"),  # 10% chance LLM says Lys-C instead of Trypsin
    "Comment[FragmentationMethod]": (0.3, "CID"),  # 30% chance LLM says CID instead of HCD
    "Comment[MS2MassAnalyzer]": (0.2, "ion trap"),  # 20% says ion trap instead of Orbitrap
    "Comment[PrecursorMassTolerance]": (0.3, "20 ppm"),  # 30% says 20 ppm instead of 10
    "Characteristics[Modification]": (0.4, "NT=Phospho;MT=Variable;TA=S,T,Y;AC=UNIMOD:21"),  # LLM adds wrong mod
}


def build_gold(pxd, sdrf_path):
    return convert_training_sdrf_to_submission(sdrf_path, pxd)

def build_all_not_applicable(gold_df):
    sub = gold_df.copy()
    for col in sub.columns:
        if col not in ('ID', 'PXD', 'Raw Data File'):
            sub[col] = "Not Applicable"
    return sub

def build_defaults_only(gold_df):
    sub = build_all_not_applicable(gold_df)
    for col, val in CORPUS_DEFAULTS.items():
        if col in sub.columns:
            sub[col] = val
    for i, mod in enumerate(MOD_DEFAULTS):
        suffix = "" if i == 0 else f".{i}"
        col = f"Characteristics[Modification]{suffix}"
        if col in sub.columns:
            sub[col] = mod
    return sub

def build_defaults_plus_biology(gold_df):
    sub = build_defaults_only(gold_df)
    for col, val in BIOLOGY_DEFAULTS.items():
        if col in sub.columns:
            sub[col] = val
    return sub

def build_defaults_bio_pride(gold_df):
    """Defaults + bio + perfect PRIDE (Organism, Instrument from gold)."""
    sub = build_defaults_plus_biology(gold_df)
    # Copy Organism and Instrument from gold (simulates perfect PRIDE)
    for col in ["Characteristics[Organism]", "Comment[Instrument]"]:
        if col in gold_df.columns:
            sub[col] = gold_df[col]
    return sub

def build_defaults_bio_pride_label(gold_df):
    """Defaults + bio + perfect PRIDE + perfect Label from gold."""
    sub = build_defaults_bio_pride(gold_df)
    if "Characteristics[Label]" in gold_df.columns:
        sub["Characteristics[Label]"] = gold_df["Characteristics[Label]"]
    return sub

def build_oracle_technical(gold_df):
    """All technical columns from gold, biology as "not available"."""
    sub = gold_df.copy()
    # Keep technical columns from gold, replace biology with "not available"
    for col, val in BIOLOGY_DEFAULTS.items():
        if col in sub.columns:
            sub[col] = val
    # Keep: Organism, Instrument, Label, CleavageAgent, FragmentationMethod,
    # MS2MassAnalyzer, Separation, tolerances, mods, fractions, etc.
    return sub

def build_oracle_biology(gold_df):
    """All biology columns from gold, technical as defaults."""
    sub = build_defaults_only(gold_df)
    # Copy biology from gold
    biology_cols = [
        "Characteristics[Organism]", "Characteristics[OrganismPart]",
        "Characteristics[CellType]", "Characteristics[CellLine]",
        "Characteristics[Disease]", "Characteristics[Sex]",
        "Characteristics[Age]", "Characteristics[MaterialType]",
        "Characteristics[DevelopmentalStage]", "Characteristics[AncestryCategory]",
        "Characteristics[Strain]", "Characteristics[Treatment]",
        "Characteristics[Compound]", "Characteristics[Specimen]",
        "Characteristics[Bait]", "Characteristics[GeneticModification]",
        "Characteristics[BiologicalReplicate]",
        # Also Label since it's experiment-specific
        "Characteristics[Label]",
        # And Instrument/Organism (PRIDE columns)
        "Comment[Instrument]",
    ]
    for col in biology_cols:
        if col in gold_df.columns:
            sub[col] = gold_df[col]
    return sub

def build_perfect_gold(gold_df):
    return gold_df.copy()


def run_analysis():
    pxd_files = load_all_training_pxds(DATA_DIR)
    selected = list(pxd_files.items())[:10]

    print(f"Error Analysis: {len(selected)} training PXDs")
    print("=" * 80)

    strategies = {
        "1_all_NA":              build_all_not_applicable,
        "2_defaults":            build_defaults_only,
        "3_defaults+bio":        build_defaults_plus_biology,
        "4_defaults+bio+PRIDE":  build_defaults_bio_pride,
        "5_d+b+PRIDE+Label":    build_defaults_bio_pride_label,
        "6_oracle_technical":    build_oracle_technical,
        "7_oracle_biology":      build_oracle_biology,
        "8_perfect":             build_perfect_gold,
    }

    all_golds = []
    all_subs = {name: [] for name in strategies}

    for pxd, sdrf_path in selected:
        gold = build_gold(pxd, sdrf_path)
        all_golds.append(gold)
        for name, builder in strategies.items():
            sub = builder(gold)
            for col in gold.columns:
                if col not in sub.columns:
                    sub[col] = "Not Applicable"
            all_subs[name].append(sub)

    # Score each strategy
    print(f"\n{'=' * 80}")
    print(f"STRATEGY LADDER (cumulative improvements):")
    print(f"{'=' * 80}")

    strategy_results = {}
    for name in strategies:
        combined_gold = pd.concat(all_golds, ignore_index=True)
        combined_sub = pd.concat(all_subs[name], ignore_index=True)
        eval_df, f1 = run_scorer(combined_gold, combined_sub, 'ID')
        strategy_results[name] = (eval_df, f1)

        valid = eval_df.dropna(subset=['f1'])
        n_scored = len(valid)
        n_perfect = len(valid[valid['f1'] >= 0.999])
        n_zero = len(valid[valid['f1'] < 0.001])

        print(f"\n  {name}: F1 = {f1:.4f}  ({n_scored} pairs, {n_perfect} perfect, {n_zero} zeros)")

    # Detailed per-column comparison: defaults+bio vs defaults+bio+PRIDE
    print(f"\n{'=' * 80}")
    print(f"PRIDE IMPACT: What Organism + Instrument adds")
    print(f"{'=' * 80}")

    eval_before = strategy_results["3_defaults+bio"][0]
    eval_after = strategy_results["4_defaults+bio+PRIDE"][0]

    before_cols = eval_before.dropna(subset=['f1']).groupby('AnnotationType')['f1'].mean()
    after_cols = eval_after.dropna(subset=['f1']).groupby('AnnotationType')['f1'].mean()

    all_cols = sorted(set(before_cols.index) | set(after_cols.index))
    print(f"\n  {'Column':<45} {'Before':>8} {'After':>8} {'Delta':>8}")
    print(f"  {'-'*45} {'------':>8} {'------':>8} {'------':>8}")
    for col in all_cols:
        b = before_cols.get(col, 0)
        a = after_cols.get(col, 0)
        delta = a - b
        if abs(delta) > 0.001:
            print(f"  {col:<45} {b:>8.3f} {a:>8.3f} {delta:>+8.3f} {'***' if abs(delta) > 0.1 else ''}")

    # Label impact
    print(f"\n{'=' * 80}")
    print(f"LABEL IMPACT: What Label adds on top of PRIDE")
    print(f"{'=' * 80}")

    eval_before = strategy_results["4_defaults+bio+PRIDE"][0]
    eval_after = strategy_results["5_d+b+PRIDE+Label"][0]

    before_cols = eval_before.dropna(subset=['f1']).groupby('AnnotationType')['f1'].mean()
    after_cols = eval_after.dropna(subset=['f1']).groupby('AnnotationType')['f1'].mean()

    all_cols = sorted(set(before_cols.index) | set(after_cols.index))
    print(f"\n  {'Column':<45} {'Before':>8} {'After':>8} {'Delta':>8}")
    print(f"  {'-'*45} {'------':>8} {'------':>8} {'------':>8}")
    for col in all_cols:
        b = before_cols.get(col, 0)
        a = after_cols.get(col, 0)
        delta = a - b
        if abs(delta) > 0.001:
            print(f"  {col:<45} {b:>8.3f} {a:>8.3f} {delta:>+8.3f} {'***' if abs(delta) > 0.1 else ''}")

    # Oracle analysis: which is more valuable - technical or biology?
    print(f"\n{'=' * 80}")
    print(f"ORACLE ANALYSIS: Technical vs Biology ceiling")
    print(f"{'=' * 80}")

    eval_tech = strategy_results["6_oracle_technical"][0]
    eval_bio = strategy_results["7_oracle_biology"][0]

    tech_cols = eval_tech.dropna(subset=['f1']).groupby('AnnotationType')['f1'].mean()
    bio_cols = eval_bio.dropna(subset=['f1']).groupby('AnnotationType')['f1'].mean()
    default_cols = eval_before.dropna(subset=['f1']).groupby('AnnotationType')['f1'].mean()

    all_cols = sorted(set(tech_cols.index) | set(bio_cols.index))
    print(f"\n  {'Column':<45} {'Defaults':>8} {'OracleTech':>10} {'OracleBio':>10}")
    print(f"  {'-'*45} {'------':>8} {'--------':>10} {'--------':>10}")
    for col in all_cols:
        d = default_cols.get(col, 0)
        t = tech_cols.get(col, 0)
        b = bio_cols.get(col, 0)
        print(f"  {col:<45} {d:>8.3f} {t:>10.3f} {b:>10.3f}")

    # Key insight: compare what v6 should score vs what it actually scores
    print(f"\n{'=' * 80}")
    print(f"KEY INSIGHT: Where LLM adds vs hurts")
    print(f"{'=' * 80}")
    print(f"""
  Strategy                    F1      Description
  {'='*70}
  defaults+bio               {strategy_results['3_defaults+bio'][1]:.4f}   No API calls, just smart defaults
  defaults+bio+PRIDE         {strategy_results['4_defaults+bio+PRIDE'][1]:.4f}   + perfect Organism/Instrument
  d+b+PRIDE+Label            {strategy_results['5_d+b+PRIDE+Label'][1]:.4f}   + perfect Label
  oracle_technical           {strategy_results['6_oracle_technical'][1]:.4f}   Perfect tech, default biology
  oracle_biology             {strategy_results['7_oracle_biology'][1]:.4f}   Perfect biology, default tech
  perfect                    {strategy_results['8_perfect'][1]:.4f}   Everything perfect

  v6 on Kaggle test PXDs:    0.3180   For reference

  GAP ANALYSIS:
  - Defaults+bio alone:      {strategy_results['3_defaults+bio'][1]:.4f} (no API needed)
  - Adding PRIDE:            +{strategy_results['4_defaults+bio+PRIDE'][1] - strategy_results['3_defaults+bio'][1]:.4f} (Organism + Instrument)
  - Adding Label:            +{strategy_results['5_d+b+PRIDE+Label'][1] - strategy_results['4_defaults+bio+PRIDE'][1]:.4f} (experiment-specific)
  - Oracle tech ceiling:     {strategy_results['6_oracle_technical'][1]:.4f} (if all tech columns perfect)
  - Oracle bio ceiling:      {strategy_results['7_oracle_biology'][1]:.4f} (if all bio columns perfect)

  CONCLUSION: If defaults+bio+PRIDE > 0.318, then v6's LLM is NET NEGATIVE.
  The LLM replaces correct defaults with wrong guesses.
""")

    # Per-column: what does defaults+bio get RIGHT that LLM might break?
    print(f"{'=' * 80}")
    print(f"COLUMNS WHERE DEFAULTS ALREADY SCORE WELL (LLM risk of damage):")
    print(f"{'=' * 80}")

    db_eval = strategy_results["3_defaults+bio"][0]
    db_cols = db_eval.dropna(subset=['f1']).groupby('AnnotationType')['f1'].mean().sort_values(ascending=False)

    print(f"\n  {'Column':<45} {'Default F1':>10} {'Risk if LLM overrides':>25}")
    print(f"  {'-'*45} {'--------':>10} {'-'*25:>25}")
    for col, f1_val in db_cols.items():
        if f1_val > 0.3:
            risk = "HIGH — LLM could break this" if f1_val >= 0.8 else "MEDIUM — LLM must match or beat" if f1_val >= 0.5 else "LOW"
            print(f"  {col:<45} {f1_val:>10.3f} {risk:>25}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_analysis()
