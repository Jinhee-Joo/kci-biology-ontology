# extract_taxon.py
import re
from typing import List, Tuple

# "Model Organisms"는 Taxon과 별개로 잡아서 (Taxon + ModelOrganism) 구조로 두는 걸 추천
MODEL_ORGANISM_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("Drosophila", [r"\bdrosophila\b"]),
    ("Arabidopsis", [r"\barabidopsis\b"]),
    ("Mouse", [r"\bmouse\b", r"\bmus musculus\b"]),
    ("Yeast", [r"\byeast\b", r"\bsaccharomyces\b"]),
]

TAXON_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("Mammalia", [r"\bmammal(s)?\b", r"\brodent(s)?\b", r"\bprimate(s)?\b", r"\bbat(s)?\b"]),
    ("Aves", [r"\bbird(s)?\b", r"\bavian\b"]),
    ("Reptilia", [r"\breptile(s)?\b", r"\blizard(s)?\b", r"\bsnake(s)?\b", r"\bturtle(s)?\b"]),
    ("Teleost Fish", [r"\bteleost(s)?\b", r"\bfish(es)?\b", r"\bcichlid(s)?\b"]),
    ("Insects", [r"\binsect(s)?\b", r"\bcoleoptera\b", r"\bdiptera\b", r"\blepidoptera\b"]),
    ("Invertebrates", [r"\binvertebrate(s)?\b", r"\bmollusc(s)?\b", r"\bnematode(s)?\b", r"\barthropod(s)?\b"]),
    ("Angiosperms", [r"\bangiosperm(s)?\b", r"\bflowering plant(s)?\b"]),
    ("Plants", [r"\bplant(s)?\b", r"\bchloroplast\b", r"\bcpDNA\b"]),
    ("Vertebrates", [r"\bvertebrate(s)?\b"]),
]

def keyword_taxon_candidates(text: str) -> List[str]:
    if not text:
        return []
    t = text.lower()
    hits: List[str] = []
    for label, patterns in TAXON_KEYWORDS:
        for p in patterns:
            if re.search(p, t):
                hits.append(label)
                break
    # dedupe keep order
    out = []
    seen = set()
    for x in hits:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def keyword_model_organisms(text: str) -> List[str]:
    if not text:
        return []
    t = text.lower()
    hits: List[str] = []
    for label, patterns in MODEL_ORGANISM_KEYWORDS:
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
