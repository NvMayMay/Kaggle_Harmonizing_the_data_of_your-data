#!/usr/bin/env python3
"""SDRF MCP Server — 5 tools for agentic SDRF metadata extraction.

Tools:
  1. pride_lookup      — Fetch authoritative metadata from PRIDE REST API
  2. ms_ontology_lookup — Look up PSI-MS accession codes (instruments, fragmentation, etc.)
  3. unimod_lookup      — Look up UNIMOD modification entries with formatted strings
  4. paper_fetch        — Retrieve paper text for a PXD
  5. sdrf_format_reference — Look up canonical SDRF-formatted values from gold corpus

Run locally:
  python server.py                    # stdio transport (for Claude Desktop / SDK)
  python server.py --transport http   # HTTP transport (for remote access)
"""
import os
import re
import json
import sqlite3
import argparse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from typing import Optional

from mcp.server.fastmcp import FastMCP

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ONTOLOGY_DIR = os.path.join(SCRIPT_DIR, "ontology")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

# Create MCP server
mcp = FastMCP(
    "sdrf-tools",
    instructions=(
        "SDRF metadata extraction tools for proteomics. "
        "Use pride_lookup first to get authoritative instrument/organism data, "
        "then ms_ontology_lookup to format accession codes, "
        "unimod_lookup for modification strings, "
        "and sdrf_format_reference to validate against gold corpus."
    ),
)


# ============================================================================
# Database helpers
# ============================================================================

def get_db(name: str) -> sqlite3.Connection:
    """Get a SQLite connection to an ontology database."""
    db_path = os.path.join(ONTOLOGY_DIR, name)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}. Run build_databases.py first.")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def fuzzy_match(query: str, candidates: list, threshold: float = 0.6) -> list:
    """Fuzzy match query against candidate strings."""
    results = []
    query_lower = query.lower().strip()
    for candidate in candidates:
        sim = SequenceMatcher(None, query_lower, candidate.lower()).ratio()
        if sim >= threshold:
            results.append((candidate, sim))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ============================================================================
# Tool 1: PRIDE Lookup
# ============================================================================

@mcp.tool()
def pride_lookup(pxd: str) -> str:
    """Fetch authoritative metadata from PRIDE REST API for a proteomics dataset.

    Returns instruments (with MS accession codes), organisms, modifications,
    quantification methods, and sample/data protocols.

    Args:
        pxd: ProteomeXchange accession (e.g., "PXD000070")
    """
    url = f"https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{pxd}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return json.dumps({"error": f"PRIDE API returned {e.code} for {pxd}"})
    except Exception as e:
        return json.dumps({"error": f"PRIDE API error: {str(e)}"})

    result = {"pxd": pxd}

    # Instruments
    instruments = []
    for inst in data.get("instruments", []):
        entry = {"name": inst.get("name", ""), "accession": inst.get("accession", "")}
        instruments.append(entry)
    result["instruments"] = instruments

    # Organisms
    organisms = []
    for org in data.get("organisms", []):
        entry = {"name": org.get("name", "")}
        organisms.append(entry)
    result["organisms"] = organisms

    # Organism parts
    org_parts = []
    for part in data.get("organismParts", []):
        name = part.get("name", "")
        if name.lower() not in ("not available", "not applicable", ""):
            org_parts.append({"name": name})
    result["organism_parts"] = org_parts

    # Diseases
    diseases = []
    for disease in data.get("diseases", []):
        name = disease.get("name", "")
        if name.lower() not in ("not available", "not applicable", ""):
            diseases.append({"name": name})
    result["diseases"] = diseases

    # PTMs / modifications
    mods = []
    for ptm in data.get("identifiedPTMStrings", []):
        mods.append(ptm)  # Already a string like "Oxidation"
    result["modifications"] = mods

    # Quantification
    quant = data.get("quantificationMethods", [])
    result["quantification_methods"] = quant

    # Detect label type
    quant_lower = " ".join(str(q) for q in quant).lower()
    title_lower = data.get("title", "").lower()
    desc_lower = data.get("projectDescription", "").lower()
    all_text = quant_lower + " " + title_lower + " " + desc_lower

    if any(t in all_text for t in ["tmt", "tandem mass tag", "tmtpro"]):
        result["label_type"] = "TMT"
    elif any(t in all_text for t in ["silac", "stable isotope"]):
        result["label_type"] = "SILAC"
    elif any(t in all_text for t in ["itraq"]):
        result["label_type"] = "iTRAQ"
    elif any(t in all_text for t in ["dimethyl label"]):
        result["label_type"] = "Dimethyl"
    else:
        result["label_type"] = "label-free"

    # Protocols
    result["sample_protocol"] = data.get("sampleProcessingProtocol", "")[:2000]
    result["data_protocol"] = data.get("dataProcessingProtocol", "")[:2000]

    # Publications
    pubs = data.get("references", [])
    result["publications"] = [
        {"doi": p.get("doi", ""), "pmid": p.get("pubmedId", "")}
        for p in pubs[:3]
    ]

    return json.dumps(result, indent=2)


# ============================================================================
# Tool 2: MS Ontology Lookup
# ============================================================================

@mcp.tool()
def ms_ontology_lookup(term: str, category: str = "") -> str:
    """Look up PSI-MS controlled vocabulary terms — instruments, fragmentation methods,
    mass analyzers, ionization types, and cleavage agents.

    Returns the exact accession code and formatted SDRF string (e.g., "NT=Q Exactive HF;AC=MS:1002523").

    Args:
        term: The term to search for (e.g., "Q Exactive HF", "HCD", "Orbitrap")
        category: Optional filter — "instrument", "fragmentation", "analyzer", "ionization", "cleavage_agent"
    """
    conn = get_db("psi_ms.db")
    c = conn.cursor()

    results = []

    # 1. Exact name match
    if category:
        rows = c.execute(
            "SELECT * FROM terms WHERE name = ? COLLATE NOCASE AND category = ?",
            (term, category)
        ).fetchall()
    else:
        rows = c.execute(
            "SELECT * FROM terms WHERE name = ? COLLATE NOCASE", (term,)
        ).fetchall()

    for row in rows:
        results.append({
            "accession": row["accession"],
            "name": row["name"],
            "category": row["category"],
            "formatted": f"NT={row['name']};AC={row['accession']}",
            "match_type": "exact",
        })

    # 2. Synonym match
    if not results:
        if category:
            all_terms = c.execute(
                "SELECT * FROM terms WHERE category = ?", (category,)
            ).fetchall()
        else:
            all_terms = c.execute("SELECT * FROM terms").fetchall()

        for row in all_terms:
            synonyms = json.loads(row["synonyms"]) if row["synonyms"] else []
            for syn in synonyms:
                if syn.lower() == term.lower():
                    results.append({
                        "accession": row["accession"],
                        "name": row["name"],
                        "category": row["category"],
                        "formatted": f"NT={row['name']};AC={row['accession']}",
                        "match_type": "synonym",
                        "matched_synonym": syn,
                    })

    # 3. Substring / fuzzy match
    if not results:
        if category:
            all_terms = c.execute(
                "SELECT * FROM terms WHERE category = ?", (category,)
            ).fetchall()
        else:
            all_terms = c.execute("SELECT * FROM terms").fetchall()

        term_lower = term.lower()
        for row in all_terms:
            name = row["name"]
            # Substring match
            if term_lower in name.lower() or name.lower() in term_lower:
                sim = SequenceMatcher(None, term_lower, name.lower()).ratio()
                results.append({
                    "accession": row["accession"],
                    "name": name,
                    "category": row["category"],
                    "formatted": f"NT={name};AC={row['accession']}",
                    "match_type": "fuzzy",
                    "similarity": round(sim, 3),
                })

        # If still nothing, do broad fuzzy
        if not results:
            names = [(row["accession"], row["name"], row["category"]) for row in all_terms]
            matches = fuzzy_match(term, [n[1] for n in names], threshold=0.5)
            for matched_name, sim in matches[:5]:
                for acc, name, cat in names:
                    if name == matched_name:
                        results.append({
                            "accession": acc,
                            "name": name,
                            "category": cat,
                            "formatted": f"NT={name};AC={acc}",
                            "match_type": "fuzzy",
                            "similarity": round(sim, 3),
                        })
                        break

        # Sort by similarity
        results.sort(key=lambda x: x.get("similarity", 1.0), reverse=True)
        results = results[:5]

    conn.close()

    if not results:
        return json.dumps({"error": f"No matches found for '{term}'" + (f" in category '{category}'" if category else "")})

    # Cross-reference with gold SDRF to find the format actually used in practice
    # Map category to gold SDRF column
    cat_to_gold_col = {
        "instrument": "Comment[Instrument]",
        "fragmentation": "Comment[FragmentationMethod]",
        "analyzer": "Comment[MS2MassAnalyzer]",
        "ionization": "Comment[IonizationType]",
        "cleavage_agent": "Characteristics[CleavageAgent]",
    }

    for r in results:
        acc = r["accession"]
        gold_col = cat_to_gold_col.get(r["category"], "")
        if gold_col:
            try:
                gconn = get_db("gold_sdrf.db")
                gc = gconn.cursor()
                # Search gold for this accession code
                gold_rows = gc.execute("""
                    SELECT value, frequency FROM column_values
                    WHERE column_name = ? AND value LIKE ?
                    ORDER BY frequency DESC LIMIT 3
                """, (gold_col, f"%{acc}%")).fetchall()
                gconn.close()

                if gold_rows:
                    r["gold_format"] = gold_rows[0]["value"]
                    r["gold_frequency"] = gold_rows[0]["frequency"]
                    # Also add the synonym/common-name formatted string
                    # The gold format is the one we should actually use
                    r["recommended_format"] = gold_rows[0]["value"]
            except Exception:
                pass

    return json.dumps({"matches": results}, indent=2)


# ============================================================================
# Tool 3: UNIMOD Lookup
# ============================================================================

# Common SDRF target conventions (more restrictive than UNIMOD's full list)
COMMON_TARGETS = {
    "Carbamidomethyl": "C",
    "Oxidation": "M",
    "Phospho": "S,T,Y",
    "Acetyl": "Protein N-term",
    "Deamidated": "N,Q",
    "GlyGly": "K",
    "Methyl": "K,R",
    "Dimethyl": "K,R",
    "TMT6plex": "K,N-term",
    "TMTpro": "K,N-term",
    "iTRAQ4plex": "K,N-term",
    "iTRAQ8plex": "K,N-term",
    "Label:13C(6)": "K,R",
    "Label:13C(6)15N(2)": "K",
    "Label:13C(6)15N(4)": "R",
    "Propionamide": "C",
    "Pyro-carbamidomethyl": "C",
    "MMTS": "C",
}


@mcp.tool()
def unimod_lookup(modification: str) -> str:
    """Look up a protein modification in the UNIMOD database.

    Returns the exact SDRF-formatted modification string with accession code,
    target amino acids, and modification type.

    Args:
        modification: Modification name to search for (e.g., "phosphorylation", "Carbamidomethyl", "oxidation")
    """
    conn = get_db("unimod.db")
    c = conn.cursor()

    results = []

    # 1. Exact name match
    rows = c.execute(
        "SELECT * FROM modifications WHERE name = ? COLLATE NOCASE",
        (modification,)
    ).fetchall()

    for row in rows:
        name = row["name"]
        # Use common targets if available, otherwise UNIMOD's full list
        targets = COMMON_TARGETS.get(name, row["targets"])
        mod_type = row["mod_type"]

        # Build SDRF-formatted string matching gold conventions
        parts = [f"NT={name}", f"AC={row['accession']}"]
        if targets:
            parts.append(f"TA={targets}")
        parts.append(f"MT={mod_type}")
        formatted = ";".join(parts)

        results.append({
            "accession": row["accession"],
            "name": name,
            "full_name": row["full_name"],
            "targets": targets,
            "mod_type": mod_type,
            "mono_mass": row["mono_mass"],
            "formatted": formatted,
            "match_type": "exact",
        })

    # 2. Full name match
    if not results:
        rows = c.execute(
            "SELECT * FROM modifications WHERE full_name LIKE ? COLLATE NOCASE",
            (f"%{modification}%",)
        ).fetchall()

        for row in rows[:5]:
            name = row["name"]
            targets = COMMON_TARGETS.get(name, row["targets"])
            mod_type = row["mod_type"]
            parts = [f"NT={name}", f"AC={row['accession']}"]
            if targets:
                parts.append(f"TA={targets}")
            parts.append(f"MT={mod_type}")
            formatted = ";".join(parts)

            results.append({
                "accession": row["accession"],
                "name": name,
                "full_name": row["full_name"],
                "targets": targets,
                "mod_type": mod_type,
                "formatted": formatted,
                "match_type": "full_name",
            })

    # 3. Fuzzy match
    if not results:
        all_mods = c.execute("SELECT * FROM modifications").fetchall()
        names = [row["name"] for row in all_mods]
        matches = fuzzy_match(modification, names, threshold=0.5)

        for matched_name, sim in matches[:5]:
            row = c.execute(
                "SELECT * FROM modifications WHERE name = ?", (matched_name,)
            ).fetchone()
            if row:
                name = row["name"]
                targets = COMMON_TARGETS.get(name, row["targets"])
                mod_type = row["mod_type"]
                parts = [f"NT={name}", f"AC={row['accession']}"]
                if targets:
                    parts.append(f"TA={targets}")
                parts.append(f"MT={mod_type}")
                formatted = ";".join(parts)

                results.append({
                    "accession": row["accession"],
                    "name": name,
                    "targets": targets,
                    "mod_type": mod_type,
                    "formatted": formatted,
                    "match_type": "fuzzy",
                    "similarity": round(sim, 3),
                })

    conn.close()

    # Also check gold SDRF for the exact format used in practice
    gold_matches = _gold_modification_lookup(modification)
    if gold_matches:
        for r in results:
            r["gold_formats"] = gold_matches

    if not results:
        return json.dumps({"error": f"No UNIMOD match found for '{modification}'"})

    return json.dumps({"matches": results}, indent=2)


def _gold_modification_lookup(modification: str) -> list:
    """Check gold SDRF database for how this modification appears in practice."""
    try:
        conn = get_db("gold_sdrf.db")
        c = conn.cursor()
        rows = c.execute("""
            SELECT value, frequency FROM column_values
            WHERE column_name = 'Characteristics[Modification]'
            AND value LIKE ? COLLATE NOCASE
            ORDER BY frequency DESC LIMIT 5
        """, (f"%{modification}%",)).fetchall()
        conn.close()
        return [{"value": row["value"], "frequency": row["frequency"]} for row in rows]
    except Exception:
        return []


# ============================================================================
# Tool 4: Paper Fetch
# ============================================================================

@mcp.tool()
def paper_fetch(pxd: str) -> str:
    """Retrieve paper text for a proteomics dataset.

    First checks local files, then tries to fetch from EuropePMC via the PRIDE API.

    Args:
        pxd: ProteomeXchange accession (e.g., "PXD000070")
    """
    # 1. Check local TestPubText
    local_paths = [
        os.path.join(DATA_DIR, "TestPubText", f"{pxd}_PubText.json"),
        os.path.join(DATA_DIR, "TrainingPubText", f"{pxd}_PubText.json"),
    ]
    for local_path in local_paths:
        if os.path.exists(local_path):
            with open(local_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            text = data if isinstance(data, str) else data.get("text", json.dumps(data))
            return json.dumps({
                "source": "local",
                "pxd": pxd,
                "text": text[:15000],  # Limit to 15K chars
                "length": len(text),
            })

    # 2. Try PRIDE API for publication DOI/PMID
    try:
        pride_url = f"https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{pxd}"
        req = urllib.request.Request(pride_url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            pride_data = json.loads(resp.read().decode("utf-8"))

        refs = pride_data.get("references", [])
        for ref in refs:
            pmid = ref.get("pubmedId", "")
            if pmid:
                # Try EuropePMC for full text
                pmc_search_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:{pmid}&format=json"
                req2 = urllib.request.Request(pmc_search_url)
                with urllib.request.urlopen(req2, timeout=15) as resp2:
                    search_data = json.loads(resp2.read().decode("utf-8"))

                results = search_data.get("resultList", {}).get("result", [])
                for result in results:
                    pmcid = result.get("pmcid", "")
                    if pmcid:
                        # Fetch full text XML
                        ft_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
                        req3 = urllib.request.Request(ft_url)
                        with urllib.request.urlopen(req3, timeout=30) as resp3:
                            xml_text = resp3.read().decode("utf-8")

                        # Parse JATS XML → extract methods section
                        text = _extract_methods_from_jats(xml_text)
                        return json.dumps({
                            "source": "europepmc",
                            "pxd": pxd,
                            "pmid": pmid,
                            "pmcid": pmcid,
                            "text": text[:15000],
                            "length": len(text),
                        })

                # If no PMC full text, return abstract from search results
                if results:
                    abstract = results[0].get("abstractText", "")
                    title = results[0].get("title", "")
                    return json.dumps({
                        "source": "abstract_only",
                        "pxd": pxd,
                        "pmid": pmid,
                        "title": title,
                        "text": f"Title: {title}\n\nAbstract: {abstract}",
                        "length": len(abstract),
                    })

    except Exception as e:
        return json.dumps({"error": f"Paper fetch failed: {str(e)}", "pxd": pxd})

    return json.dumps({"error": f"No paper found for {pxd}", "pxd": pxd})


def _extract_methods_from_jats(xml_text: str) -> str:
    """Extract methods/materials section from JATS XML, plus title and abstract."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        # Fall back to regex extraction
        text = re.sub(r"<[^>]+>", " ", xml_text)
        return text[:15000]

    parts = []

    # Title
    title_elem = root.find(".//article-title")
    if title_elem is not None:
        parts.append("Title: " + "".join(title_elem.itertext()))

    # Abstract
    abstract_elem = root.find(".//abstract")
    if abstract_elem is not None:
        parts.append("\nAbstract: " + "".join(abstract_elem.itertext()))

    # Methods section (various JATS naming conventions)
    for section in root.findall(".//sec"):
        title = section.find("title")
        if title is not None:
            title_text = "".join(title.itertext()).lower()
            if any(kw in title_text for kw in [
                "method", "material", "experimental", "procedure",
                "sample preparation", "mass spectrometry", "proteom",
                "lc-ms", "data acquisition",
            ]):
                section_text = "".join(section.itertext())
                parts.append(f"\n{section_text}")

    if not parts:
        # Fallback: return all text
        text = "".join(root.itertext())
        return text[:15000]

    return "\n".join(parts)[:15000]


# ============================================================================
# Tool 5: SDRF Format Reference
# ============================================================================

@mcp.tool()
def sdrf_format_reference(column: str, query: str = "") -> str:
    """Look up canonical SDRF-formatted values from the gold corpus of 374+ annotated SDRFs.

    Use this to find the exact format string that matches the competition's scoring expectations.
    For example, query "HCD" in column "Comment[FragmentationMethod]" to get "AC=MS:1000422;NT=HCD".

    Args:
        column: The SDRF column name (e.g., "Comment[Instrument]", "Characteristics[Modification]")
        query: Optional search term to filter values (e.g., "HCD", "Q Exactive", "Trypsin")
    """
    conn = get_db("gold_sdrf.db")
    c = conn.cursor()

    if query:
        # Search for matching values
        query_lower = query.lower()

        # Exact match first
        exact = c.execute("""
            SELECT value, frequency FROM column_values
            WHERE column_name = ? AND value = ? COLLATE NOCASE
            ORDER BY frequency DESC
        """, (column, query)).fetchall()

        # Substring match
        substring = c.execute("""
            SELECT value, frequency FROM column_values
            WHERE column_name = ? AND LOWER(value) LIKE ?
            ORDER BY frequency DESC LIMIT 20
        """, (column, f"%{query_lower}%")).fetchall()

        # Deduplicate
        seen = set()
        exact_matches = []
        for row in exact:
            if row["value"] not in seen:
                exact_matches.append({"value": row["value"], "frequency": row["frequency"]})
                seen.add(row["value"])

        fuzzy_matches = []
        for row in substring:
            if row["value"] not in seen:
                fuzzy_matches.append({"value": row["value"], "frequency": row["frequency"]})
                seen.add(row["value"])

        # If still no results, try fuzzy matching against all values
        if not exact_matches and not fuzzy_matches:
            all_vals = c.execute("""
                SELECT value, frequency FROM column_values
                WHERE column_name = ? ORDER BY frequency DESC
            """, (column,)).fetchall()

            candidates = [row["value"] for row in all_vals]
            fuzz = fuzzy_match(query, candidates, threshold=0.5)
            for matched_val, sim in fuzz[:10]:
                freq = next((r["frequency"] for r in all_vals if r["value"] == matched_val), 0)
                fuzzy_matches.append({
                    "value": matched_val,
                    "frequency": freq,
                    "similarity": round(sim, 3),
                })

        # Recommend the most frequent match
        all_matches = exact_matches + fuzzy_matches
        recommendation = all_matches[0]["value"] if all_matches else None

        result = {
            "column": column,
            "query": query,
            "exact_matches": exact_matches,
            "fuzzy_matches": fuzzy_matches[:10],
            "recommendation": recommendation,
        }

    else:
        # Return all values for this column, sorted by frequency
        rows = c.execute("""
            SELECT value, frequency FROM column_values
            WHERE column_name = ?
            ORDER BY frequency DESC LIMIT 50
        """, (column,)).fetchall()

        total = c.execute("""
            SELECT SUM(frequency) FROM column_values WHERE column_name = ?
        """, (column,)).fetchone()[0] or 0

        result = {
            "column": column,
            "total_occurrences": total,
            "unique_values": len(rows),
            "values": [{"value": row["value"], "frequency": row["frequency"]} for row in rows],
        }

    conn.close()
    return json.dumps(result, indent=2)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SDRF MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    if args.transport == "http":
        mcp.run(transport="streamable-http", host="0.0.0.0", port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
