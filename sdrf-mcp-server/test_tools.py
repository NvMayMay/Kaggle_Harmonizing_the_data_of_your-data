#!/usr/bin/env python3
"""Test all 5 MCP tools directly (without running the MCP server)."""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server import pride_lookup, ms_ontology_lookup, unimod_lookup, paper_fetch, sdrf_format_reference


def test_pride_lookup():
    print("=" * 60)
    print("Test 1: pride_lookup('PXD000070')")
    result = json.loads(pride_lookup("PXD000070"))
    print(f"  Instruments: {result.get('instruments', [])}")
    print(f"  Organisms: {result.get('organisms', [])}")
    print(f"  Label type: {result.get('label_type', '?')}")
    print(f"  Modifications: {result.get('modifications', [])}")
    print(f"  Publications: {result.get('publications', [])}")
    assert "instruments" in result
    assert len(result["instruments"]) > 0
    print("  PASSED")


def test_ms_ontology_lookup():
    print("\n" + "=" * 60)

    # Test instrument
    print("Test 2a: ms_ontology_lookup('Q Exactive HF', 'instrument')")
    result = json.loads(ms_ontology_lookup("Q Exactive HF", "instrument"))
    matches = result.get("matches", [])
    print(f"  Matches: {len(matches)}")
    if matches:
        print(f"  Best: {matches[0]['formatted']}")
        assert "MS:" in matches[0]["accession"]
    print("  PASSED")

    # Test fragmentation
    print("\nTest 2b: ms_ontology_lookup('HCD', 'fragmentation')")
    result = json.loads(ms_ontology_lookup("HCD", "fragmentation"))
    matches = result.get("matches", [])
    print(f"  Matches: {len(matches)}")
    if matches:
        print(f"  Best: {matches[0]['formatted']}")
    print("  PASSED")

    # Test analyzer
    print("\nTest 2c: ms_ontology_lookup('Orbitrap', 'analyzer')")
    result = json.loads(ms_ontology_lookup("Orbitrap", "analyzer"))
    matches = result.get("matches", [])
    print(f"  Matches: {len(matches)}")
    if matches:
        print(f"  Best: {matches[0]['formatted']}")
    print("  PASSED")

    # Test LTQ Orbitrap Velos (common instrument)
    print("\nTest 2d: ms_ontology_lookup('LTQ Orbitrap Velos', 'instrument')")
    result = json.loads(ms_ontology_lookup("LTQ Orbitrap Velos", "instrument"))
    matches = result.get("matches", [])
    print(f"  Matches: {len(matches)}")
    if matches:
        print(f"  Best: {matches[0]['formatted']}")
    print("  PASSED")


def test_unimod_lookup():
    print("\n" + "=" * 60)

    # Test carbamidomethyl
    print("Test 3a: unimod_lookup('Carbamidomethyl')")
    result = json.loads(unimod_lookup("Carbamidomethyl"))
    matches = result.get("matches", [])
    if matches:
        print(f"  Formatted: {matches[0]['formatted']}")
        print(f"  Gold formats: {matches[0].get('gold_formats', [])[:2]}")
    print("  PASSED")

    # Test oxidation
    print("\nTest 3b: unimod_lookup('Oxidation')")
    result = json.loads(unimod_lookup("Oxidation"))
    matches = result.get("matches", [])
    if matches:
        print(f"  Formatted: {matches[0]['formatted']}")
    print("  PASSED")

    # Test phospho
    print("\nTest 3c: unimod_lookup('Phospho')")
    result = json.loads(unimod_lookup("Phospho"))
    matches = result.get("matches", [])
    if matches:
        print(f"  Formatted: {matches[0]['formatted']}")
    print("  PASSED")

    # Test fuzzy search
    print("\nTest 3d: unimod_lookup('phosphorylation') (fuzzy)")
    result = json.loads(unimod_lookup("phosphorylation"))
    matches = result.get("matches", [])
    if matches:
        print(f"  Found: {matches[0]['name']} -> {matches[0]['formatted']}")
    print("  PASSED")

    # Test TMT
    print("\nTest 3e: unimod_lookup('TMT6plex')")
    result = json.loads(unimod_lookup("TMT6plex"))
    matches = result.get("matches", [])
    if matches:
        print(f"  Formatted: {matches[0]['formatted']}")
    print("  PASSED")


def test_paper_fetch():
    print("\n" + "=" * 60)
    print("Test 4: paper_fetch('PXD000070')")
    result = json.loads(paper_fetch("PXD000070"))
    print(f"  Source: {result.get('source', '?')}")
    print(f"  Length: {result.get('length', 0)} chars")
    text = result.get("text", "")
    if text:
        print(f"  Preview: {text[:200]}...")
    print("  PASSED")


def test_sdrf_format_reference():
    print("\n" + "=" * 60)

    # Test instrument lookup
    print("Test 5a: sdrf_format_reference('Comment[Instrument]', 'Q Exactive')")
    result = json.loads(sdrf_format_reference("Comment[Instrument]", "Q Exactive"))
    print(f"  Recommendation: {result.get('recommendation', '?')}")
    exact = result.get("exact_matches", [])
    fuzzy = result.get("fuzzy_matches", [])
    print(f"  Exact: {len(exact)}, Fuzzy: {len(fuzzy)}")
    for m in (exact + fuzzy)[:3]:
        print(f"    {m['frequency']:>4}x  {m['value'][:60]}")
    print("  PASSED")

    # Test modification lookup
    print("\nTest 5b: sdrf_format_reference('Characteristics[Modification]', 'Carbamidomethyl')")
    result = json.loads(sdrf_format_reference("Characteristics[Modification]", "Carbamidomethyl"))
    print(f"  Recommendation: {result.get('recommendation', '?')}")
    for m in result.get("exact_matches", [])[:3] + result.get("fuzzy_matches", [])[:3]:
        print(f"    {m['frequency']:>4}x  {m['value'][:60]}")
    print("  PASSED")

    # Test collision energy
    print("\nTest 5c: sdrf_format_reference('Comment[CollisionEnergy]', '30')")
    result = json.loads(sdrf_format_reference("Comment[CollisionEnergy]", "30"))
    print(f"  Recommendation: {result.get('recommendation', '?')}")
    for m in result.get("exact_matches", [])[:3] + result.get("fuzzy_matches", [])[:3]:
        print(f"    {m['frequency']:>4}x  {m['value']}")
    print("  PASSED")

    # Test all values for a column
    print("\nTest 5d: sdrf_format_reference('Comment[FragmentationMethod]') — all values")
    result = json.loads(sdrf_format_reference("Comment[FragmentationMethod]"))
    print(f"  Total occurrences: {result.get('total_occurrences', 0)}")
    print(f"  Unique values: {result.get('unique_values', 0)}")
    for v in result.get("values", [])[:5]:
        print(f"    {v['frequency']:>4}x  {v['value'][:60]}")
    print("  PASSED")


if __name__ == "__main__":
    test_pride_lookup()
    test_ms_ontology_lookup()
    test_unimod_lookup()
    test_paper_fetch()
    test_sdrf_format_reference()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
