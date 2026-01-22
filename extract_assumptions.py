# extract_assumptions.py
import re
from typing import List

ASSUMPTION_KEYWORDS = [
    ("Molecular Clock", [r"\bmolecular clock\b", r"\bclock model\b"]),
    ("Strict Clock", [r"\bstrict clock\b"]),
    ("Relaxed Clock", [r"\brelaxed clock\b"]),
    ("Neutral Evolution Assumption", [r"\bneutral evolution\b", r"\bneutral model\b"]),
    ("Selection Model", [r"\bselection model\b", r"\bpositive selection\b", r"\bpurifying selection\b"]),
    ("Panmictic Population", [r"\bpanmictic\b", r"\brandom mating\b"]),
    ("No Gene Flow", [r"\bno gene flow\b", r"\bwithout gene flow\b", r"\bno migration\b"]),
    ("Gene Flow Allowed", [r"\bgene flow\b", r"\bmigration\b", r"\bintrogression\b"]),
    ("Incomplete Lineage Sorting Considered", [r"\bincomplete lineage sorting\b", r"\bils\b", r"\bmultispecies coalescent\b", r"\bmsc\b"]),
    ("No Recombination", [r"\bno recombination\b", r"\bnon-?recombining\b"]),
    ("Constant Population Size", [r"\bconstant population size\b", r"\bconstant Ne\b"]),
    ("Bottleneck Model", [r"\bbottleneck\b"]),
    ("Expansion Model", [r"\bpopulation expansion\b", r"\brapid expansion\b"]),
]

# v2에서는 "Molecular Clock" 아래 Strict/Relaxed를 같이 넣고 싶으면
# Gemini가 둘 다 찍어도 되게 허용하면 됨. (여기선 일단 후보만 뽑아줌)
def keyword_assumption_candidates(text: str) -> List[str]:
    if not text:
        return []
    t = text.lower()
    hits: List[str] = []
    for label, patterns in ASSUMPTION_KEYWORDS:
        for p in patterns:
            if re.search(p, t):
                hits.append(label)
                break

    # 정리 규칙: Strict/Relaxed가 있으면 Molecular Clock도 같이 넣어주기
    if ("Strict Clock" in hits or "Relaxed Clock" in hits) and "Molecular Clock" not in hits:
        hits.insert(0, "Molecular Clock")

    # No Gene Flow vs Gene Flow Allowed 충돌 시: "No Gene Flow" 우선
    if "No Gene Flow" in hits and "Gene Flow Allowed" in hits:
        hits = [x for x in hits if x != "Gene Flow Allowed"]

    out = []
    seen = set()
    for x in hits:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out
