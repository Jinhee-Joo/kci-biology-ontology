# scripts/extract/extract_contribution.py
from typing import List

CONTRIBUTION_TYPE_LIST = [
    "Empirical Study",
    "Method / Model Paper",
    "Resource / Dataset Paper",
    "Review / Survey",
    "Protocol / Technical Note",
]

def keyword_contribution_candidates(title: str, abstract: str) -> List[str]:
    t = f"{title}\n{abstract}".lower()
    cands: List[str] = []

    if any(k in t for k in ["review", "survey", "meta-analysis", "meta analysis", "systematic review"]):
        cands.append("Review / Survey")

    if any(k in t for k in ["dataset", "database", "resource", "data release", "atlas", "catalog",
                            "genome assembly", "reference genome"]):
        cands.append("Resource / Dataset Paper")

    if any(k in t for k in ["protocol", "pipeline", "workflow", "technical note", "implementation", "tutorial"]):
        cands.append("Protocol / Technical Note")

    if any(k in t for k in ["we propose", "we introduce", "new method", "novel method", "algorithm", "framework", "model"]):
        cands.append("Method / Model Paper")

    if not cands:
        cands.append("Empirical Study")
    else:
        # 관계 늘리기 + 현실 반영: 대다수는 empirical 성격도 같이 가짐
        cands.append("Empirical Study")

    cands = [c for c in cands if c in CONTRIBUTION_TYPE_LIST]
    return sorted(set(cands))
