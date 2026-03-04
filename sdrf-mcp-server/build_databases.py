#!/usr/bin/env python3
"""Build SQLite ontology databases for the SDRF MCP server.

Downloads and parses:
1. PSI-MS OBO → psi_ms.db (instruments, fragmentation, analyzers, ionization)
2. UNIMOD XML → unimod.db (modifications with SDRF format strings)
3. 374 Gold SDRFs → gold_sdrf.db (indexed by column + value for format reference)
"""
import os
import re
import sys
import glob
import json
import sqlite3
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter

ONTOLOGY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ontology")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

PSI_MS_URL = "https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo"
UNIMOD_URL = "http://www.unimod.org/xml/unimod.xml"


# ============================================================================
# 1. PSI-MS OBO Parser
# ============================================================================

# Category parent accessions in PSI-MS ontology
CATEGORY_PARENTS = {
    "MS:1000031": "instrument",         # instrument model
    "MS:1000044": "fragmentation",      # dissociation method
    "MS:1000443": "analyzer",           # mass analyzer type
    "MS:1000008": "ionization",         # ionization type
    "MS:1001045": "cleavage_agent",     # cleavage agent name
}

# Additional specific terms we want to categorize
SPECIFIC_TERMS = {
    # Instruments (common ones that might not be direct children)
    "MS:1001742": "instrument",  # LTQ Orbitrap Velos
    "MS:1002523": "instrument",  # Q Exactive HF
    "MS:1002877": "instrument",  # Q Exactive HF-X
    "MS:1002634": "instrument",  # Q Exactive Plus
    "MS:1002732": "instrument",  # Orbitrap Fusion Lumos
    "MS:1003028": "instrument",  # Orbitrap Exploris 480
    "MS:1003356": "instrument",  # Orbitrap Astral
    "MS:1001911": "instrument",  # Q Exactive
    "MS:1002835": "instrument",  # Orbitrap Fusion
    "MS:1000449": "instrument",  # LTQ Orbitrap
    "MS:1000556": "instrument",  # LTQ Orbitrap XL
    "MS:1000557": "instrument",  # LTQ Orbitrap Discovery
    "MS:1000855": "instrument",  # LTQ Orbitrap Elite
    "MS:1001910": "instrument",  # LTQ Orbitrap Velos Pro
    "MS:1003029": "instrument",  # Orbitrap Eclipse
    "MS:1002416": "instrument",  # Orbitrap Fusion
    "MS:1000483": "instrument",  # Thermo Fisher instrument model (parent)
    "MS:1000494": "instrument",  # Thermo Fisher instrument model (parent)
    # Analyzers
    "MS:1000484": "analyzer",  # orbitrap
    "MS:1000264": "analyzer",  # ion trap
    "MS:1000084": "analyzer",  # time-of-flight
    "MS:1000079": "analyzer",  # fourier transform ion cyclotron resonance
    "MS:1000082": "analyzer",  # quadrupole
    "MS:1003379": "analyzer",  # Astral analyzer
    # Fragmentation
    "MS:1000422": "fragmentation",  # beam-type collision-induced dissociation (HCD)
    "MS:1000133": "fragmentation",  # collision-induced dissociation (CID)
    "MS:1000598": "fragmentation",  # electron transfer dissociation (ETD)
    "MS:1001880": "fragmentation",  # in-source collision-induced dissociation
    "MS:1002631": "fragmentation",  # Electron-Transfer/Higher-Energy Collision Dissociation (EThcD)
    # Ionization
    "MS:1000073": "ionization",  # electrospray ionization (ESI)
    "MS:1000075": "ionization",  # MALDI
    "MS:1000398": "ionization",  # nanoelectrospray
}


def parse_obo(obo_text):
    """Parse OBO format into list of term dicts."""
    terms = []
    current = None

    for line in obo_text.split("\n"):
        line = line.strip()
        if line == "[Term]":
            if current:
                terms.append(current)
            current = {"id": "", "name": "", "synonyms": [], "is_a": [], "is_obsolete": False}
        elif current is not None:
            if line.startswith("id: "):
                current["id"] = line[4:]
            elif line.startswith("name: "):
                current["name"] = line[6:]
            elif line.startswith("synonym: "):
                # Extract synonym text from quotes
                m = re.match(r'synonym: "([^"]*)"', line)
                if m:
                    current["synonyms"].append(m.group(1))
            elif line.startswith("is_a: "):
                parent = line[6:].split("!")[0].strip()
                current["is_a"].append(parent)
            elif line.startswith("is_obsolete: true"):
                current["is_obsolete"] = True

    if current:
        terms.append(current)

    return terms


def categorize_terms(terms):
    """Assign categories to terms based on is_a hierarchy."""
    # Build parent→children index
    term_map = {t["id"]: t for t in terms}

    # Build transitive is_a closure (BFS from category parents)
    category_members = {}
    for parent_id, category in CATEGORY_PARENTS.items():
        members = set()
        queue = [parent_id]
        while queue:
            pid = queue.pop(0)
            for t in terms:
                if pid in t["is_a"] and t["id"] not in members:
                    members.add(t["id"])
                    queue.append(t["id"])
        category_members[category] = members

    # Assign categories
    categorized = []
    for t in terms:
        if t["is_obsolete"]:
            continue
        if not t["id"].startswith("MS:"):
            continue

        cat = None
        # Check specific terms first
        if t["id"] in SPECIFIC_TERMS:
            cat = SPECIFIC_TERMS[t["id"]]
        else:
            # Check category membership
            for category, members in category_members.items():
                if t["id"] in members:
                    cat = category
                    break

        if cat:
            parent = t["is_a"][0] if t["is_a"] else ""
            categorized.append({
                "accession": t["id"],
                "name": t["name"],
                "category": cat,
                "parent": parent,
                "synonyms": json.dumps(t["synonyms"]),
            })

    return categorized


def build_psi_ms_db():
    """Download PSI-MS OBO and build SQLite database."""
    db_path = os.path.join(ONTOLOGY_DIR, "psi_ms.db")
    obo_path = os.path.join(ONTOLOGY_DIR, "psi-ms.obo")

    # Download if not cached
    if not os.path.exists(obo_path):
        print("Downloading PSI-MS OBO...")
        urllib.request.urlretrieve(PSI_MS_URL, obo_path)
        print(f"  Saved to {obo_path}")
    else:
        print(f"Using cached PSI-MS OBO: {obo_path}")

    with open(obo_path, "r", encoding="utf-8") as f:
        obo_text = f.read()

    print("Parsing OBO terms...")
    terms = parse_obo(obo_text)
    print(f"  {len(terms)} total terms parsed")

    categorized = categorize_terms(terms)

    # Count by category
    cat_counts = Counter(t["category"] for t in categorized)
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count} terms")

    # Build SQLite
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE terms (
            accession TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            parent TEXT,
            synonyms TEXT
        )
    """)
    c.execute("CREATE INDEX idx_name ON terms(name COLLATE NOCASE)")
    c.execute("CREATE INDEX idx_category ON terms(category)")

    c.executemany(
        "INSERT OR IGNORE INTO terms VALUES (?, ?, ?, ?, ?)",
        [(t["accession"], t["name"], t["category"], t["parent"], t["synonyms"])
         for t in categorized],
    )

    conn.commit()
    total = c.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
    conn.close()
    print(f"  Built {db_path}: {total} terms")
    return db_path


# ============================================================================
# 2. UNIMOD XML Parser
# ============================================================================

# Common modification types (Fixed vs Variable) based on proteomics conventions
FIXED_MODS = {"Carbamidomethyl", "Propionamide", "MMTS"}
LABEL_MODS = {
    "TMT6plex", "TMTpro", "TMT2plex", "TMT10plex", "TMT11plex", "TMT16plex",
    "iTRAQ4plex", "iTRAQ8plex",
    "Label:13C(6)", "Label:13C(6)15N(2)", "Label:13C(6)15N(4)",
    "Dimethyl", "Dimethyl:2H(4)", "Dimethyl:2H(4)13C(2)",
    "mTRAQ", "ICAT-C", "ICAT-D",
}


def parse_unimod_xml(xml_path):
    """Parse UNIMOD XML into modification records."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Handle XML namespace
    ns = {"umod": "http://www.unimod.org/xmlns/schema/unimod_2"}

    mods = []
    for mod_elem in root.findall(".//umod:mod", ns):
        title = mod_elem.get("title", "")
        full_name = mod_elem.get("full_name", "")
        record_id = mod_elem.get("record_id", "")

        if not title or not record_id:
            continue

        accession = f"UNIMOD:{record_id}"

        # Get monoisotopic mass from delta
        delta = mod_elem.find("umod:delta", ns)
        mono_mass = float(delta.get("mono_mass", "0")) if delta is not None else 0.0

        # Get target amino acids from specificities
        targets = set()
        for spec in mod_elem.findall("umod:specificity", ns):
            site = spec.get("site", "")
            if site and len(site) == 1 and site.isalpha():
                targets.add(site)
            elif site == "N-term":
                targets.add("N-term")
            elif site == "C-term":
                targets.add("C-term")
            elif site == "Protein N-term":
                targets.add("Protein N-term")

        target_str = ",".join(sorted(targets)) if targets else ""

        # Determine modification type
        if title in FIXED_MODS:
            mod_type = "Fixed"
        elif title in LABEL_MODS:
            mod_type = "Fixed"  # Labels are typically fixed
        else:
            mod_type = "Variable"

        # Build formatted SDRF string
        parts = [f"NT={title}", f"AC={accession}"]
        if mod_type:
            parts.append(f"MT={mod_type}")
        if target_str:
            parts.append(f"TA={target_str}")
        formatted = ";".join(parts)

        mods.append({
            "accession": accession,
            "name": title,
            "full_name": full_name,
            "targets": target_str,
            "mod_type": mod_type,
            "mono_mass": mono_mass,
            "formatted": formatted,
        })

    return mods


def build_unimod_db():
    """Download UNIMOD XML and build SQLite database."""
    db_path = os.path.join(ONTOLOGY_DIR, "unimod.db")
    xml_path = os.path.join(ONTOLOGY_DIR, "unimod.xml")

    # Download if not cached
    if not os.path.exists(xml_path):
        print("Downloading UNIMOD XML...")
        urllib.request.urlretrieve(UNIMOD_URL, xml_path)
        print(f"  Saved to {xml_path}")
    else:
        print(f"Using cached UNIMOD XML: {xml_path}")

    print("Parsing UNIMOD modifications...")
    mods = parse_unimod_xml(xml_path)
    print(f"  {len(mods)} modifications parsed")

    # Build SQLite
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE modifications (
            accession TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            full_name TEXT,
            targets TEXT,
            mod_type TEXT,
            mono_mass REAL,
            formatted TEXT
        )
    """)
    c.execute("CREATE INDEX idx_mod_name ON modifications(name COLLATE NOCASE)")
    c.execute("CREATE INDEX idx_mod_fullname ON modifications(full_name COLLATE NOCASE)")

    c.executemany(
        "INSERT OR IGNORE INTO modifications VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(m["accession"], m["name"], m["full_name"], m["targets"],
          m["mod_type"], m["mono_mass"], m["formatted"])
         for m in mods],
    )

    conn.commit()
    total = c.execute("SELECT COUNT(*) FROM modifications").fetchone()[0]
    conn.close()
    print(f"  Built {db_path}: {total} modifications")

    # Print some common ones
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    for name in ["Carbamidomethyl", "Oxidation", "Phospho", "Acetyl", "TMT6plex", "TMTpro"]:
        row = c.execute("SELECT formatted FROM modifications WHERE name = ?", (name,)).fetchone()
        if row:
            print(f"    {name}: {row[0]}")
    conn.close()

    return db_path


# ============================================================================
# 3. Gold SDRF Parser
# ============================================================================

# Bigbio column name → submission column name mapping
BIGBIO_TO_SUBMISSION = {
    "characteristics[organism]": "Characteristics[Organism]",
    "characteristics[organism part]": "Characteristics[OrganismPart]",
    "characteristics[cell type]": "Characteristics[CellType]",
    "characteristics[cell line]": "Characteristics[CellLine]",
    "characteristics[disease]": "Characteristics[Disease]",
    "characteristics[sex]": "Characteristics[Sex]",
    "characteristics[age]": "Characteristics[Age]",
    "characteristics[material type]": "Characteristics[MaterialType]",
    "characteristics[developmental stage]": "Characteristics[DevelopmentalStage]",
    "characteristics[ancestry category]": "Characteristics[AncestryCategory]",
    "characteristics[biological replicate]": "Characteristics[BiologicalReplicate]",
    "characteristics[strain]": "Characteristics[Strain]",
    "characteristics[individual]": None,  # skip
    "comment[instrument]": "Comment[Instrument]",
    "comment[cleavage agent details]": "Characteristics[CleavageAgent]",
    "comment[dissociation method]": "Comment[FragmentationMethod]",
    "comment[modification parameters]": "Characteristics[Modification]",
    "comment[label]": "Characteristics[Label]",
    "comment[ms2 analyzer type]": "Comment[MS2MassAnalyzer]",
    "comment[precursor mass tolerance]": "Comment[PrecursorMassTolerance]",
    "comment[fragment mass tolerance]": "Comment[FragmentMassTolerance]",
    "comment[fraction identifier]": "Comment[FractionIdentifier]",
    "comment[fractionation method]": "Comment[FractionationMethod]",
    "comment[separation]": "Comment[Separation]",
    "comment[collision energy]": "Comment[CollisionEnergy]",
    "comment[enrichment method]": "Comment[EnrichmentMethod]",
    "comment[number of missed cleavages]": "Comment[NumberOfMissedCleavages]",
    "comment[proteomics data acquisition method]": "Comment[AcquisitionMethod]",
    "comment[ionization type]": "Comment[IonizationType]",
    # Skip metadata columns
    "source name": None,
    "assay name": None,
    "technology type": None,
    "comment[data file]": None,
    "comment[file uri]": None,
    "comment[file url]": None,
    "comment[proteomexchange accession number]": None,
    "comment[associated file uri]": None,
    "comment[technical replicate]": None,
    "comment[sdrf version]": None,
    "comment[sdrf template]": None,
}

# Training SDRF column mapping (reuse from evaluate.py)
TRAINING_TO_SUBMISSION = {
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
    "GeneticModification": "Characteristics[GeneticModification]",
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
    "Modification": "Characteristics[Modification]",
    # Skip
    "SourceName": None,
    "AssayName": None,
    "TechnicalReplicate": None,
}

# Columns that can appear multiple times (e.g., modification parameters)
MULTI_VALUE_COLUMNS = {"Characteristics[Modification]"}


def extract_pxd_from_path(path):
    """Extract PXD accession from file path."""
    basename = os.path.basename(path)
    m = re.match(r"(PXD\d+)", basename)
    return m.group(1) if m else os.path.splitext(basename)[0]


def parse_training_sdrf(sdrf_path):
    """Parse a training SDRF and return (pxd, column_values dict)."""
    import pandas as pd
    pxd = extract_pxd_from_path(sdrf_path)
    df = pd.read_csv(sdrf_path, sep="\t", dtype=str)

    values = defaultdict(list)  # submission_col → list of values

    for orig_col in df.columns:
        # Handle .1, .2 suffixes for multi-value columns
        base_col = re.sub(r"\.\d+$", "", orig_col)

        sub_col = TRAINING_TO_SUBMISSION.get(base_col)
        if sub_col is None:
            # Try lowercase
            sub_col = TRAINING_TO_SUBMISSION.get(base_col.lower())
        if sub_col is None:
            continue

        for val in df[orig_col].dropna().unique():
            val = str(val).strip()
            if val and val.lower() not in ("nan", ""):
                values[sub_col].append(val)

    return pxd, dict(values)


def parse_bigbio_sdrf(sdrf_path):
    """Parse a bigbio SDRF and return (pxd, column_values dict)."""
    import pandas as pd
    pxd = extract_pxd_from_path(sdrf_path)
    df = pd.read_csv(sdrf_path, sep="\t", dtype=str)

    values = defaultdict(list)

    for orig_col in df.columns:
        col_lower = orig_col.lower().strip()
        # Handle .1, .2 suffixes
        base_col = re.sub(r"\.\d+$", "", col_lower)

        sub_col = BIGBIO_TO_SUBMISSION.get(base_col)
        if sub_col is None:
            # Try matching with common prefixes
            for pattern, target in BIGBIO_TO_SUBMISSION.items():
                if target and base_col.startswith(pattern.rstrip("]")) and base_col.endswith("]"):
                    sub_col = target
                    break
        if sub_col is None:
            continue

        for val in df[orig_col].dropna().unique():
            val = str(val).strip()
            if val and val.lower() not in ("nan", ""):
                values[sub_col].append(val)

    return pxd, dict(values)


def build_gold_sdrf_db():
    """Parse all gold SDRFs and build SQLite database."""
    db_path = os.path.join(ONTOLOGY_DIR, "gold_sdrf.db")

    # Collect all SDRFs
    training_sdrfs = glob.glob(os.path.join(DATA_DIR, "TrainingSDRFs", "*.tsv"))
    bigbio_sdrfs = glob.glob(os.path.join(DATA_DIR, "bigbio_sdrf", "annotated-projects", "*", "*.sdrf.tsv"))

    print(f"Parsing gold SDRFs: {len(training_sdrfs)} training + {len(bigbio_sdrfs)} bigbio")

    # Parse all SDRFs
    all_values = defaultdict(Counter)  # column → {value: count}
    pxd_values = {}  # pxd → {column: [values]}

    # Training SDRFs
    for sdrf_path in training_sdrfs:
        try:
            pxd, vals = parse_training_sdrf(sdrf_path)
            pxd_values[pxd] = vals
            for col, value_list in vals.items():
                for v in value_list:
                    all_values[col][v] += 1
        except Exception as e:
            print(f"  Error parsing {sdrf_path}: {e}")

    print(f"  Parsed {len(training_sdrfs)} training SDRFs")

    # Bigbio SDRFs
    bigbio_count = 0
    for sdrf_path in bigbio_sdrfs:
        try:
            pxd, vals = parse_bigbio_sdrf(sdrf_path)
            if pxd not in pxd_values:  # Don't overwrite training data
                pxd_values[pxd] = vals
            for col, value_list in vals.items():
                for v in value_list:
                    all_values[col][v] += 1
            bigbio_count += 1
        except Exception as e:
            print(f"  Error parsing {sdrf_path}: {e}")

    print(f"  Parsed {bigbio_count} bigbio SDRFs")

    # Build SQLite
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Table 1: All unique values per column with frequency
    c.execute("""
        CREATE TABLE column_values (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            column_name TEXT NOT NULL,
            value TEXT NOT NULL,
            frequency INTEGER DEFAULT 1,
            UNIQUE(column_name, value)
        )
    """)
    c.execute("CREATE INDEX idx_col_val ON column_values(column_name, value)")
    c.execute("CREATE INDEX idx_col ON column_values(column_name)")

    # Table 2: Per-PXD values for template matching
    c.execute("""
        CREATE TABLE pxd_values (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pxd TEXT NOT NULL,
            column_name TEXT NOT NULL,
            value TEXT NOT NULL
        )
    """)
    c.execute("CREATE INDEX idx_pxd ON pxd_values(pxd)")
    c.execute("CREATE INDEX idx_pxd_col ON pxd_values(pxd, column_name)")

    # Insert column_values
    for col, counter in all_values.items():
        for val, count in counter.items():
            c.execute(
                "INSERT OR IGNORE INTO column_values (column_name, value, frequency) VALUES (?, ?, ?)",
                (col, val, count),
            )

    # Insert pxd_values
    for pxd, vals in pxd_values.items():
        for col, value_list in vals.items():
            for val in set(value_list):  # deduplicate
                c.execute(
                    "INSERT INTO pxd_values (pxd, column_name, value) VALUES (?, ?, ?)",
                    (pxd, col, val),
                )

    conn.commit()

    # Print summary
    total_values = c.execute("SELECT COUNT(*) FROM column_values").fetchone()[0]
    total_pxd_rows = c.execute("SELECT COUNT(*) FROM pxd_values").fetchone()[0]
    n_pxds = c.execute("SELECT COUNT(DISTINCT pxd) FROM pxd_values").fetchone()[0]
    n_cols = c.execute("SELECT COUNT(DISTINCT column_name) FROM column_values").fetchone()[0]

    print(f"  Built {db_path}:")
    print(f"    {total_values} unique (column, value) pairs across {n_cols} columns")
    print(f"    {total_pxd_rows} PXD-specific value entries across {n_pxds} PXDs")

    # Per-column breakdown
    print(f"\n  Per-column unique values:")
    rows = c.execute("""
        SELECT column_name, COUNT(*) as n, SUM(frequency) as total_freq
        FROM column_values
        GROUP BY column_name
        ORDER BY total_freq DESC
    """).fetchall()
    for col, n, freq in rows:
        print(f"    {col:<45} {n:>5} unique, {freq:>6} total")

    # Show top values for key columns
    for col in ["Comment[Instrument]", "Characteristics[Modification]",
                 "Characteristics[Label]", "Comment[FragmentationMethod]"]:
        print(f"\n  Top values for {col}:")
        top = c.execute("""
            SELECT value, frequency FROM column_values
            WHERE column_name = ? ORDER BY frequency DESC LIMIT 5
        """, (col,)).fetchall()
        for val, freq in top:
            print(f"    {freq:>5}x  {val[:80]}")

    conn.close()
    return db_path


# ============================================================================
# Main
# ============================================================================

def main():
    os.makedirs(ONTOLOGY_DIR, exist_ok=True)

    print("=" * 80)
    print("Building SDRF Ontology Databases")
    print("=" * 80)

    print("\n--- 1. PSI-MS OBO ---")
    build_psi_ms_db()

    print("\n--- 2. UNIMOD XML ---")
    build_unimod_db()

    print("\n--- 3. Gold SDRF Index ---")
    build_gold_sdrf_db()

    print("\n" + "=" * 80)
    print("All databases built successfully!")
    print(f"Location: {ONTOLOGY_DIR}")


if __name__ == "__main__":
    main()
