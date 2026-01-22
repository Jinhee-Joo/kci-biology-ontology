# extract_methods.py
import re
from typing import List, Tuple

# annotate_batch_resume.py의 FULL_METHOD_LIST 라벨과 "완전 동일"해야 함
METHOD_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("Maximum Likelihood", [
        r"\bmaximum likelihood\b", r"\bmax(?:imum)?-likelihood\b",
        r"\blog-likelihood\b", r"\bml\b(?!\w)",  # 'ml'은 오탐 가능해서 보수적으로
        r"\blikelihood\b",
    ]),
    ("Bayesian Inference", [
        r"\bbayesian\b", r"\bposterior\b", r"\bprior\b",
        r"\bmcmc\b", r"\bgibbs\b", r"\bmetropolis\b",
        r"\bmarkov chain\b",
    ]),
    ("Coalescent-based Model", [
        r"\bcoalescent\b",
        r"\bmultispecies coalescent\b", r"\bmsc\b(?!\w)",
        r"\bisolation with migration\b", r"\bim model\b",
    ]),
    ("Birth–Death Model", [
        r"\bbirth[-– ]death\b",
        r"\byule\b",
        r"\bspeciation[-– ]extinction\b",
    ]),
    ("dN/dS Analysis", [
        r"\bdn/ds\b", r"\bdnds\b",
        r"\bka/ks\b", r"\bkaks\b",
        r"\bomega\b", r"ω",
        r"\bpositive selection\b", r"\bpurifying selection\b",
    ]),
    ("Approximate Bayesian Computation (ABC)", [
        r"\bapproximate bayesian computation\b",
        r"\babc\b(?!\w)",
        r"\bsummary statistics\b",
    ]),
    ("Phylogenetic Comparative Methods (PCM)", [
        r"\bphylogenetic comparative\b",
        r"\bpcm\b(?!\w)",
        r"\bpgls\b", r"\bpic\b(?!\w)",  # phylogenetic independent contrasts
        r"\bpagel\b",
        r"\bancestral state reconstruction\b",
        r"\bornstein[-– ]uhlenbeck\b|\bou\b(?!\w)",
        r"\bbrownian motion\b",
    ]),
    ("Hidden Markov Models", [
        r"\bhidden markov\b", r"\bhmm\b(?!\w)",
        r"\bviterbi\b",
    ]),
    ("Simulation-based Inference", [
        r"\bsimulation\b", r"\bsimulate\b",
        r"\bforward[-– ]time\b", r"\bforward simulation\b",
        r"\bcoalescent simulation\b",
        r"\bindividual[-– ]based\b",
        r"\bagent[-– ]based\b",
        r"\bapproximate\b.*\bsimulation\b",
    ]),
    ("Network Phylogenetics", [
        r"\bphylogenetic network\b",
        r"\bnetwork phylogenetic\b",
        r"\breticulate\b",
        r"\bhybridization network\b",
        r"\bspecies network\b",
    ]),
]

def keyword_method_candidates(text: str) -> List[str]:
    """
    title+abstract 같은 텍스트에서 키워드로 Method 후보를 뽑는다.
    반환: 중복 제거된 method label 리스트 (라벨명은 FULL_METHOD_LIST와 동일)
    """
    if not text:
        return []

    t = text.lower()
    hits: List[str] = []

    for method, patterns in METHOD_KEYWORDS:
        for p in patterns:
            if re.search(p, t):
                hits.append(method)
                break

    # 중복 제거 + 순서 유지
    out: List[str] = []
    seen = set()
    for m in hits:
        if m not in seen:
            out.append(m)
            seen.add(m)

    return out
