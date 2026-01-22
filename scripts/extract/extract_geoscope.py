# scripts/extract/extract_geoscope.py
from typing import List

GEOSCOPE_LIST = [
    "Single Location",
    "Multiple Locations",
    "Global / Public Database Scale",
    "Not Specified",
]

def keyword_geoscope_candidates(text: str) -> List[str]:
    t = (text or "").lower()

    if any(k in t for k in [
        "public database", "genbank", "gbif", "global dataset",
        "worldwide", "global scale", "meta-analysis", "meta analysis"
    ]):
        return ["Global / Public Database Scale"]

    if any(k in t for k in [
        "multiple sites", "multiple populations", "across populations",
        "across regions", "across continents", "range-wide", "multisite", "multi-site"
    ]):
        return ["Multiple Locations"]

    if any(k in t for k in ["island", "archipelago", "river", "lake", "forest", "desert"]):
        # 지명이 있어도 100% single이라고 단정 못해서 Not Specified도 같이
        return ["Single Location", "Not Specified"]

    return ["Not Specified"]
