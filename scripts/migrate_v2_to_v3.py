import json

IN_PATH  = "../data/raw/annotations/paper_annotations_all_v2.jsonl"
OUT_PATH = "../data/raw/annotations/paper_annotations_all_v3.jsonl"

# v2 -> v3 ResearchTask 라벨 매핑 (필요한 만큼 계속 추가)
RT_MAP = {
    "Gene Flow Analysis": "Gene Flow / Introgression Analysis",
    "Introgression Analysis": "Gene Flow / Introgression Analysis",
    "Positive Selection Detection": "Selection / Positive Selection Detection",
    "Selection Detection": "Selection / Positive Selection Detection",
    "Trait Evolution": "Trait Evolution / Ancestral State Reconstruction",
    "Ancestral State Reconstruction": "Trait Evolution / Ancestral State Reconstruction",
    "Comparative Methods": "Phylogenetic Comparative Methods",
    "Phylogenetic Comparative Analysis": "Phylogenetic Comparative Methods",
    "Macroevolution": "Macroevolutionary Pattern Analysis",
    "Biogeographic Analysis": "Biogeography",
    "Adaptation": "Adaptation Inference",
    "Demographic Inference": "Demographic History Inference",
    "Population Structure": "Population Structure Analysis",
    "Species Delimitation": "Species Delimitation",
    "Phylogeny Inference": "Phylogeny Inference",
    "Divergence Time Estimation": "Divergence Time Estimation",
    # 필요하면 여기에 계속 추가
}

# v3 스키마 키(빈 값 기본 세팅)
V3_KEYS = [
    "ResearchTask",
    "Method",
    "Software",
    "DataType",
    "Taxon",
    "GeoScope",
    "EnvironmentType",
    "ContributionType",
]

def normalize_researchtask_label(label: str) -> str:
    label = (label or "").strip()
    return RT_MAP.get(label, label)  # 매핑 없으면 그대로(나중에 수동 점검)

def ensure_v3_annotation(ann: dict) -> dict:
    ann = ann or {}
    # v2에 ResearchTask/Method/Software만 있어도 v3 키를 모두 채워둠
    for k in V3_KEYS:
        ann.setdefault(k, [])
    # ResearchTask 표준화
    new_rts = []
    for item in ann.get("ResearchTask", []):
        if not isinstance(item, dict):
            continue
        item = dict(item)
        item["label"] = normalize_researchtask_label(item.get("label"))
        new_rts.append(item)
    ann["ResearchTask"] = new_rts
    return ann

with open(IN_PATH, "r", encoding="utf-8") as f_in, open(OUT_PATH, "w", encoding="utf-8") as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)

        obj["_annotation"] = ensure_v3_annotation(obj.get("_annotation"))
        f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("DONE ->", OUT_PATH)
