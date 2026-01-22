# extract_datatype.py
import re
from typing import List

DATATYPE_KEYWORDS = [
    ("Whole Genome Sequence", [
        r"\bwhole genome\b", r"\bwgs\b", r"\bgenome-?wide\b", r"\bre-?sequenc",
        r"\bwhole-?genome\b",
    ]),
    ("Reduced Representation (RADseq, GBS)", [
        r"\brad-?seq\b", r"\bddrad\b", r"\bgbs\b", r"\brestriction site\b",
        r"\breduced representation\b",
    ]),
    ("SNP Genotype Data", [
        r"\bsnp(s)?\b", r"\bgenotyp", r"\bgenotyping\b", r"\bgenotype\b",
        r"\barray\b", r"\bimputation\b",
    ]),
    ("mtDNA", [
        r"\bmtDNA\b", r"\bmitochond", r"\bcytochrome\b", r"\bcox1\b|\bcoi\b",
        r"\bcontrol region\b",
    ]),
    ("Nuclear Gene Sequences", [
        r"\bnuclear gene\b", r"\bmulti-?locus\b", r"\blocus\b", r"\bgene sequences?\b",
        r"\bexon\b|\bintron\b",
    ]),
    ("Morphometric Data", [
        r"\bmorphometr", r"\bgeometric morph", r"\bmorphology\b", r"\bshape\b",
    ]),
    ("Behavioral Data", [
        r"\bbehavio(u)?r\b", r"\bmating\b", r"\bforaging\b", r"\bsong\b",
        r"\bcourtship\b",
    ]),
    ("Ecological / Environmental Variables", [
        r"\benvironment(al)?\b", r"\bclimate\b", r"\btemperature\b", r"\bprecipitation\b",
        r"\bniche model\b", r"\bsdm\b|\bspecies distribution model\b",
    ]),
    ("Fossil Record", [
        r"\bfossil(s)?\b", r"\bpaleo\b", r"\bmiocene\b", r"\bpliocene\b",
        r"\bpleistocene\b",
    ]),
    ("Occurrence / Distribution Records", [
        r"\boccurrence\b", r"\bdistribution records?\b", r"\brange\b",
        r"\bgbif\b", r"\bmuseum records?\b",
    ]),
    ("Transcriptome (RNA-seq)", [
        r"\brna-?seq\b", r"\btranscriptome\b", r"\bexpression\b", r"\breads\b",
    ]),
]

def keyword_datatype_candidates(text: str) -> List[str]:
    if not text:
        return []
    t = text.lower()
    hits: List[str] = []
    for label, patterns in DATATYPE_KEYWORDS:
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
