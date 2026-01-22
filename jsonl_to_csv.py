import json
import csv
from pathlib import Path

INPUT = "openalex_evolbio_tiered_500.jsonl"
OUTDIR = Path("csv")
OUTDIR.mkdir(exist_ok=True)


# 컨테이너
works = {}
concepts = {}
domains = {}
tiers = {}

edge_work_concept = []
edge_work_domain = []
edge_work_tier = []


# JSONL 읽기
with open(INPUT, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)

        # ---- Work ----
        work_id = obj["id"].split("/")[-1]
        works[work_id] = {
            "work_id": work_id,
            "title": obj["title"],
            "year": obj["year"],
            "evolbio_score": obj["evolbio_score"],
        }

        # ---- Tier ----
        tier = obj["tier"]
        tiers[tier] = {"tier_id": tier, "tier_name": tier}
        edge_work_tier.append({
            "work_id": work_id,
            "tier_id": tier,
            "relation": "hasTier"
        })

        # ---- Domain ----
        domain = obj["domain"]
        domains[domain] = {"domain_id": domain, "domain_name": domain}
        edge_work_domain.append({
            "work_id": work_id,
            "domain_id": domain,
            "relation": "inDomain"
        })

        # ---- Concepts ----
        for c in obj.get("topics_concepts", []):
            cid = c.lower().replace(" ", "_")
            concepts[cid] = {
                "concept_id": cid,
                "concept_name": c
            }
            edge_work_concept.append({
                "work_id": work_id,
                "concept_id": cid,
                "relation": "hasConcept"
            })


# CSV 저장 함수
def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# Node CSV
write_csv(
    OUTDIR / "Node_Work.csv",
    works.values(),
    ["work_id", "title", "year", "evolbio_score"]
)

write_csv(
    OUTDIR / "Node_Concept.csv",
    concepts.values(),
    ["concept_id", "concept_name"]
)

write_csv(
    OUTDIR / "Node_Domain.csv",
    domains.values(),
    ["domain_id", "domain_name"]
)

write_csv(
    OUTDIR / "Node_Tier.csv",
    tiers.values(),
    ["tier_id", "tier_name"]
)

# Edge CSV
write_csv(
    OUTDIR / "Edge_Work_Concept.csv",
    edge_work_concept,
    ["work_id", "concept_id", "relation"]
)

write_csv(
    OUTDIR / "Edge_Work_Domain.csv",
    edge_work_domain,
    ["work_id", "domain_id", "relation"]
)

write_csv(
    OUTDIR / "Edge_Work_Tier.csv",
    edge_work_tier,
    ["work_id", "tier_id", "relation"]
)

print("CSV 변환 완료! (csv/ 폴더 확인)")
