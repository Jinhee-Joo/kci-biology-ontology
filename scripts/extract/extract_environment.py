# scripts/extract/extract_environment.py
from typing import List

ENVIRONMENT_TYPE_LIST = [
    "Marine",
    "Freshwater",
    "Terrestrial",
    "Island",
    "High-altitude",
    "Polar",
    "Tropical",
    "Temperate",
    "Fragmented Habitat",
    "Not Specified",
]

def keyword_environment_candidates(text: str) -> List[str]:
    t = (text or "").lower()
    cands: List[str] = []

    if any(k in t for k in ["marine", "ocean", "sea", "coastal", "reef", "intertidal"]):
        cands.append("Marine")
    if any(k in t for k in ["freshwater", "river", "lake", "stream"]):
        cands.append("Freshwater")
    if any(k in t for k in ["terrestrial", "forest", "grassland", "desert", "savanna"]):
        cands.append("Terrestrial")

    if any(k in t for k in ["island", "archipelago"]):
        cands.append("Island")
    if any(k in t for k in ["high altitude", "high-altitude", "alpine", "mountain", "montane"]):
        cands.append("High-altitude")
    if any(k in t for k in ["polar", "arctic", "antarctic"]):
        cands.append("Polar")

    if any(k in t for k in ["tropical", "tropics"]):
        cands.append("Tropical")
    if any(k in t for k in ["temperate"]):
        cands.append("Temperate")

    if any(k in t for k in ["fragmented habitat", "habitat fragmentation", "fragmentation"]):
        cands.append("Fragmented Habitat")

    if not cands:
        cands.append("Not Specified")

    # 정식 라벨만
    cands = [c for c in cands if c in ENVIRONMENT_TYPE_LIST]
    # dedupe
    return sorted(set(cands))
