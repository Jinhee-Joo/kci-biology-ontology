# extract_context.py
import re
from typing import List

CONTEXT_KEYWORDS = [
    ("Island Systems", [r"\bisland(s)?\b", r"\barchipelago\b"]),
    ("Mainland Systems", [r"\bmainland\b", r"\bcontinental\b"]),
    ("Marine Environment", [r"\bmarine\b", r"\bocean\b", r"\bsea\b", r"\breef\b"]),
    ("Freshwater Environment", [r"\bfreshwater\b", r"\briver\b", r"\blake\b", r"\bstream\b"]),
    ("Tropical Region", [r"\btropical\b"]),
    ("Temperate Region", [r"\btemperate\b"]),
    ("Polar Region", [r"\barctic\b", r"\bantarctic\b", r"\bpolar\b"]),
    ("High-altitude Environment", [r"\bhigh altitude\b", r"\balpine\b", r"\bmontane\b"]),
    ("Fragmented Habitat", [r"\bfragment(ed|ation)\b", r"\bhabitat fragmentation\b"]),
]

def keyword_context_candidates(text: str) -> List[str]:
    if not text:
        return []
    t = text.lower()
    hits: List[str] = []
    for label, patterns in CONTEXT_KEYWORDS:
        for p in patterns:
            if re.search(p, t):
                hits.append(label)
                break
    out = []
    seen = set()
    for x in hits:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out
