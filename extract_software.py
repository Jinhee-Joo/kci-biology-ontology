import re

SOFTWARE_PATTERNS = {
    "IQ-TREE": r"\biq[-\s]?tree\b",
    "RAxML": r"\braxml(?:-ng)?\b",
    "BEAST": r"\bbeast(?:\s?2)?\b",
    "MrBayes": r"\bmrbayes\b",
    "RevBayes": r"\brevbayes\b",
    "STRUCTURE": r"\bstructure\b",
    "ADMIXTURE": r"\badmixture\b",
    "fastsimcoal": r"\bfastsimcoal(?:2)?\b",
    "dadi": r"\bdadi\b",
    "HyPhy": r"\bh(y)?phy\b|\bhyphy\b",
}

def extract_software(text: str):
    if not text:
        return []
    t = text.lower()
    found = []
    for name, pat in SOFTWARE_PATTERNS.items():
        if re.search(pat, t, flags=re.IGNORECASE):
            found.append(name)
    return sorted(set(found))

# Software → Method 매핑 (힌트용)
SOFTWARE_TO_METHOD = {
    "IQ-TREE": ["Maximum Likelihood"],
    "RAxML": ["Maximum Likelihood"],
    "BEAST": ["Bayesian Inference"],
    "MrBayes": ["Bayesian Inference"],
    "RevBayes": ["Bayesian Inference"],
    "STRUCTURE": ["Population Structure Analysis"],
    "ADMIXTURE": ["Population Structure Analysis"],
    "fastsimcoal": ["Coalescent-based Model", "Simulation-based Inference"],
    "dadi": ["Coalescent-based Model"],
    "HyPhy": ["dN/dS Analysis", "Selection Detection"]
}

