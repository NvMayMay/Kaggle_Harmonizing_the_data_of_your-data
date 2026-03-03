"""
Analyze all 103 training SDRFs: column-level statistics.
"""
import os
import sys
import io
import pandas as pd
from collections import Counter, defaultdict
import glob

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SDRF_DIR = r"C:\Users\NVMeh\Downloads\my_submission_1\data\TrainingSDRFs"
NA_VALUES = {"not available", "not applicable", ""}

PROBLEM_COLUMNS = [
    "DevelopmentalStage",
    "Disease",
    "MaterialType",
    "BiologicalReplicate",
    "CellType",
    "Sex",
    "Age",
    "Separation",
    "MS2MassAnalyzer",
    "FragmentationMethod",
    "OrganismPart",
    "Instrument",
]

# ---- collect data ----
files = sorted(glob.glob(os.path.join(SDRF_DIR, "PXD*_cleaned.sdrf.tsv")))
print(f"Found {len(files)} SDRF files\n")

# Per-column trackers
col_pxd_count = Counter()          # how many PXDs have this column at all
col_pxd_real = Counter()           # how many PXDs have >=1 real value
col_all_values = defaultdict(list) # all values (for top-5)
col_all_real_values = defaultdict(list)  # only real values

all_columns_seen = set()

for f in files:
    pxd = os.path.basename(f).split("_")[0]
    try:
        df = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)
    except Exception as e:
        print(f"  ERROR reading {pxd}: {e}")
        continue

    # Deduplicate column names (some SDRFs have repeated columns like Modification)
    # pandas auto-renames them with .1, .2, etc. — keep originals for counting
    for col in df.columns:
        # Strip the .1 .2 suffixes for counting purposes
        base_col = col.split(".")[0] if col.endswith(tuple(f".{i}" for i in range(1, 30))) else col
        # Actually let's keep exact column names to avoid confusion
        pass

    for col in set(df.columns):  # unique column names (after pandas dedup)
        all_columns_seen.add(col)
        col_pxd_count[col] += 1

        vals = df[col].tolist()
        col_all_values[col].extend(vals)

        real_vals = [v for v in vals if v.strip().lower() not in NA_VALUES]
        col_all_real_values[col].extend(real_vals)

        if len(real_vals) > 0:
            col_pxd_real[col] += 1

total_pxds = len(files)

# =====================================================
# REPORT 1: All columns overview
# =====================================================
print("=" * 100)
print("REPORT 1: ALL COLUMNS — Presence & Real-Value Coverage")
print("=" * 100)
print(f"{'Column':<45} {'Present':<12} {'Has Real':<12} {'% Real':<10} {'Top 5 Values'}")
print("-" * 100)

for col in sorted(all_columns_seen, key=lambda c: col_pxd_real.get(c, 0), reverse=True):
    present = col_pxd_count[col]
    has_real = col_pxd_real.get(col, 0)
    pct = has_real / total_pxds * 100

    # Top 5 real values
    real_vals = col_all_real_values[col]
    top5 = Counter(real_vals).most_common(5)
    top5_str = "; ".join(f"{v} ({c})" for v, c in top5)
    if len(top5_str) > 80:
        top5_str = top5_str[:77] + "..."

    print(f"{col:<45} {present:<12} {has_real:<12} {pct:<10.1f} {top5_str}")

# =====================================================
# REPORT 2: Problem columns deep dive
# =====================================================
print("\n\n" + "=" * 100)
print("REPORT 2: PROBLEM COLUMNS — Detailed Value Distributions")
print("=" * 100)

for pcol in PROBLEM_COLUMNS:
    print(f"\n{'─' * 80}")
    print(f"  COLUMN: {pcol}")
    print(f"{'─' * 80}")

    if pcol not in all_columns_seen:
        print(f"  ** NOT FOUND in any SDRF **")
        continue

    present = col_pxd_count[pcol]
    has_real = col_pxd_real.get(pcol, 0)
    pct_present = present / total_pxds * 100
    pct_real = has_real / total_pxds * 100

    all_vals = col_all_values[pcol]
    real_vals = col_all_real_values[pcol]

    print(f"  PXDs with column present:    {present}/{total_pxds} ({pct_present:.1f}%)")
    print(f"  PXDs with real values:        {has_real}/{total_pxds} ({pct_real:.1f}%)")
    print(f"  Total rows with this column: {len(all_vals)}")
    print(f"  Rows with real value:         {len(real_vals)} ({len(real_vals)/max(len(all_vals),1)*100:.1f}%)")

    # Full value distribution
    val_counts = Counter(all_vals).most_common(30)
    print(f"\n  Value distribution (top 30, including NA):")
    for val, cnt in val_counts:
        is_na = val.strip().lower() in NA_VALUES
        marker = " [NA]" if is_na else ""
        display_val = val if val else "(empty)"
        print(f"    {display_val:<60} {cnt:>6}{marker}")

# =====================================================
# REPORT 3: Summary table — % of PXDs with real values
# =====================================================
print("\n\n" + "=" * 100)
print("REPORT 3: PROBLEM COLUMNS — % of Training PXDs with REAL Values")
print("=" * 100)
print(f"\n{'Column':<30} {'PXDs w/ Real':<15} {'Total PXDs':<15} {'% Real':<10}")
print("-" * 70)
for pcol in PROBLEM_COLUMNS:
    has_real = col_pxd_real.get(pcol, 0)
    pct = has_real / total_pxds * 100
    print(f"{pcol:<30} {has_real:<15} {total_pxds:<15} {pct:.1f}%")

# =====================================================
# REPORT 4: Columns sorted by rarity (least coverage first)
# =====================================================
print("\n\n" + "=" * 100)
print("REPORT 4: ALL COLUMNS SORTED BY RARITY (least real-value coverage first)")
print("=" * 100)
print(f"\n{'Column':<45} {'% PXDs w/ Real':<15}")
print("-" * 60)
for col in sorted(all_columns_seen, key=lambda c: col_pxd_real.get(c, 0) / total_pxds):
    has_real = col_pxd_real.get(col, 0)
    pct = has_real / total_pxds * 100
    print(f"{col:<45} {pct:.1f}%")
