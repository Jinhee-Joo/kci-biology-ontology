# import_step2_annotations.py
import json
import re
from neo4j import GraphDatabase

# Neo4j Desktop 값
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "ontology12!"

# scripts/import 폴더에서 실행 기준: v3 jsonl 위치
JSONL_PATH = "data/raw/annotations/paper_annotations_all_v3.jsonl"

# (annotation_key, node_label, rel_type)
MAP = [
    ("ResearchTask",      "ResearchTask",      "HAS_RESEARCH_TASK"),
    ("Method",            "Method",            "USES_METHOD"),
    ("Software",          "Software",          "USES_SOFTWARE"),
    ("Taxon",             "Taxon",             "STUDIES_TAXON"),
    ("DataType",          "DataType",          "USES_DATATYPE"),
    ("GeoScope",          "GeoScope",          "HAS_GEOSCOPE"),
    ("EnvironmentType",   "EnvironmentType",   "HAS_ENVIRONMENT"),
    ("ContributionType",  "ContributionType",  "HAS_CONTRIBUTION"),
]

# === 라벨링 기준(피드백 반영용) ===
# 낮은 확률도 라벨링: 0.2~0.35 권장
THRESHOLDS = {
    "ResearchTask": 0.25,
    "Method": 0.20,
    "Software": 0.15,
    "Taxon": 0.20,
    "DataType": 0.20,
    "GeoScope": 0.00,          # 초록만으로 애매하면 낮게
    "EnvironmentType": 0.00,   # 초록만으로 애매하면 낮게
    "ContributionType": 0.20,
}

# 한 논문당 후보 몇 개까지 붙일지 (taxon / contribution은 많이 붙여야 피드백 통과)
TOP_K = {
    "ResearchTask": 3,
    "Method": 3,
    "Software": 5,
    "Taxon": 5,
    "DataType": 3,
    "GeoScope": 2,
    "EnvironmentType": 3,
    "ContributionType": 5,
}

BATCH = 50


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def openalex_work_id_from_url(url: str):
    # "https://openalex.org/W123" -> "W123"
    if not url:
        return None
    m = re.search(r"/(W\d+)$", url.strip())
    return m.group(1) if m else None


def normalize_items(raw):
    """
    v3 방어:
    - ["Biogeography", ...]  -> [{"name": "Biogeography"}...]
    - [{"label": "...", "confidence":..., "evidenceText":...}, ...] -> 표준화
    - null/빈문자열 제거
    """
    if not raw:
        return []

    out = []

    # 문자열 하나
    if isinstance(raw, str):
        raw = [raw]

    # 문자열 리스트
    if isinstance(raw, list) and raw and all(isinstance(x, str) or x is None for x in raw):
        for s in raw:
            if s and str(s).strip():
                out.append({"name": str(s).strip(), "confidence": None, "evidenceText": None})
        return out

    # dict 리스트 (v2/v3 스타일)
    if isinstance(raw, list):
        for it in raw:
            if not isinstance(it, dict):
                continue
            name = it.get("name") or it.get("label")
            if not name or not str(name).strip():
                continue
            out.append({
                "name": str(name).strip(),
                "confidence": it.get("confidence"),
                "evidenceText": it.get("evidenceText") or it.get("evidence") or ""
            })
        return out

    # 그 외는 무시
    return []


def filter_rank_limit(items, ann_key: str):
    """
    confidence 기준으로 정렬하고 threshold/top-k 적용 + name 기준 dedupe
    """
    if not items:
        return []

    # name dedupe (같은 라벨 여러 번 나오면 최고 conf만)
    best = {}
    for it in items:
        name = it.get("name")
        if not name:
            continue

        conf = it.get("confidence")
        try:
            conf = float(conf) if conf is not None else None
        except Exception:
            conf = None

        ev = it.get("evidenceText") or ""
        prev = best.get(name)
        if (prev is None) or ((conf or -1) > (prev.get("confidence") or -1)):
            best[name] = {"name": name, "confidence": conf, "evidenceText": ev}

    items2 = list(best.values())
    items2.sort(key=lambda x: (x["confidence"] is not None, x["confidence"] or -1), reverse=True)

    thr = THRESHOLDS.get(ann_key, 0.0)
    k = TOP_K.get(ann_key, 3)

    out = []
    for it in items2:
        conf = it["confidence"]
        conf_val = float(conf) if conf is not None else 0.0
        if conf_val >= thr:
            out.append({"name": it["name"], "confidence": conf_val, "evidenceText": it.get("evidenceText") or ""})
        if len(out) >= k:
            break

    return out


def ensure_constraints(tx):
    # ✅ Paper는 openalexId(url)를 유니크로 (핵심)
    tx.run("CREATE CONSTRAINT paper_openalexId IF NOT EXISTS FOR (p:Paper) REQUIRE p.openalexId IS UNIQUE")

    # 분류 노드 name 유니크
    labels = {node_label for _, node_label, _ in MAP}
    for lab in labels:
        tx.run(f"CREATE CONSTRAINT {lab.lower()}_name IF NOT EXISTS FOR (n:{lab}) REQUIRE n.name IS UNIQUE")


def upsert_paper(tx, row, paper_id_url, refs):
    wid = openalex_work_id_from_url(paper_id_url)
    tx.run(
        """
        MERGE (p:Paper {openalexId: $pid_url})
        ON CREATE SET
            p.id = $pid_url,
            p.title = $title,
            p.year = $year,
            p.tier = $tier,
            p.referencedWorks = $refs
        ON MATCH SET
            p.title = coalesce(p.title, $title),
            p.year  = coalesce(p.year,  $year),
            p.tier  = coalesce(p.tier,  $tier),
            p.referencedWorks = coalesce(p.referencedWorks, $refs)
        SET p.openalexWorkId = coalesce(p.openalexWorkId, $wid)
        """,
        pid_url=paper_id_url,
        wid=wid,
        title=row.get("title", "") or "",
        year=row.get("year", None),
        tier=row.get("tier", None),
        refs=refs,
    )


def upsert_annotations(tx, paper_id_url, node_label, rel_type, items):
    """
    Paper는 이미 MERGE 되어있다고 가정하고, 관계만 upsert.
    - 더 높은 confidence가 들어왔을 때만 r.confidence/evidenceText 업데이트
    """
    cypher = f"""
    MATCH (p:Paper {{openalexId: $pid_url}})
    WITH p
    UNWIND $items AS it
      WITH p, it,
           it.name AS name,
           coalesce(it.confidence, 0.0) AS conf,
           coalesce(it.evidenceText, "") AS ev
      MERGE (x:{node_label} {{name: name}})
      MERGE (p)-[r:{rel_type}]->(x)

      ON CREATE SET
        r.confidence = conf,
        r.evidenceText = ev
      ON MATCH SET
        r.confidence = CASE WHEN conf > coalesce(r.confidence, 0.0) THEN conf ELSE coalesce(r.confidence, 0.0) END,
        r.evidenceText = CASE
            WHEN conf > coalesce(r.confidence, 0.0) AND ev <> "" THEN ev
            WHEN r.evidenceText IS NULL OR r.evidenceText = "" THEN ev
            ELSE r.evidenceText
        END
    """
    tx.run(cypher, pid_url=paper_id_url, items=items)


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    data = load_jsonl(JSONL_PATH)

    total_rows_scanned = 0
    total_rel_items = 0

    with driver:
        # constraints
        with driver.session() as session:
            session.execute_write(ensure_constraints)

        for batch in chunks(data, BATCH):
            with driver.session() as session:
                def work(tx):
                    nonlocal total_rows_scanned, total_rel_items

                    for row in batch:
                        paper_id = row.get("id") or row.get("openalexId")
                        if not paper_id:
                            continue

                        ann = row.get("_annotation", {}) or {}
                        if not isinstance(ann, dict):
                            continue

                        refs = (
                            row.get("referencedWorks")
                            or row.get("referenced_works")
                            or row.get("referencedWorks_ids")
                            or []
                        )
                        if not isinstance(refs, list):
                            refs = []

                        # 1) Paper upsert (한 번만)
                        upsert_paper(tx, row, paper_id, refs)

                        # 2) 분류 관계들 upsert (threshold/top-k 적용)
                        for ann_key, node_label, rel_type in MAP:
                            raw = ann.get(ann_key)
                            items = normalize_items(raw)
                            items = filter_rank_limit(items, ann_key)

                            if not items:
                                continue

                            upsert_annotations(tx, paper_id, node_label, rel_type, items)
                            total_rel_items += len(items)

                        total_rows_scanned += 1

                session.execute_write(work)

            print(f"[batch] rows_scanned={total_rows_scanned}, rel-items={total_rel_items}")

    print("DONE.")
    print("Total rows scanned:", total_rows_scanned)
    print("Total rel-items:", total_rel_items)


if __name__ == "__main__":
    main()
