# import_step3_refs.py  (Neo4j 5 호환 + 성능/정확도 개선 버전)
import re
from typing import Optional, List, Dict, Any
from neo4j import GraphDatabase

# ✅ Neo4j Desktop 값
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "ontology12!"

# batch 크기(너무 크면 한 번에 메모리/트랜잭션 부담)
BATCH = 500

def openalex_work_id_from_url(url: str) -> Optional[str]:
    # "https://openalex.org/W123" -> "W123"
    if not url:
        return None
    m = re.search(r"/(W\d+)$", str(url).strip())
    return m.group(1) if m else None

def normalize_ref_to_url(x: Any) -> Optional[str]:
    """
    referencedWorks 요소가
    - "https://openalex.org/W123"
    - "W123"
    - "openalex.org/W123"
    등 형태가 섞여 있어도 URL로 통일
    """
    if not x:
        return None
    s = str(x).strip()
    if not s:
        return None

    if s.startswith("http"):
        wid = openalex_work_id_from_url(s)
        return f"https://openalex.org/{wid}" if wid else s

    # "openalex.org/W123" 같은 경우
    if "openalex.org/" in s:
        m = re.search(r"(W\d+)", s)
        return f"https://openalex.org/{m.group(1)}" if m else None

    # "W123"만 있는 경우
    if s.startswith("W") and s[1:].isdigit():
        return f"https://openalex.org/{s}"

    # 혹시 "W123"가 중간에 있으면 추출
    m = re.search(r"(W\d+)", s)
    if m:
        return f"https://openalex.org/{m.group(1)}"

    return None

def ensure_schema(tx):
    # Paper lookup을 빠르게 하기 위해 openalexId 유니크 제약
    tx.run("CREATE CONSTRAINT paper_openalexId IF NOT EXISTS FOR (p:Paper) REQUIRE p.openalexId IS UNIQUE")
    # referencedWorks 조회/매칭은 openalexId로 이미 충분 (추가 인덱스 불필요)
    # CITES는 타입만 쓰므로 별도 제약 필요 없음

def connect_cites_batch(tx, rows: List[Dict[str, Any]]):
    """
    rows: [{"src": <paper_openalexId_url>, "refs": [<ref_url>, ...]}, ...]

    개선점:
    1) q를 (q:Paper {openalexId: ref})로 바로 MATCH (풀스캔 방지)
    2) MERGE (p)-[:CITES]->(q)로 연결
    """
    cypher = """
    UNWIND $rows AS row
      MATCH (p:Paper {openalexId: row.src})
      UNWIND row.refs AS ref
        WITH p, ref
        MATCH (q:Paper {openalexId: ref})
        MERGE (p)-[:CITES]->(q)
    """
    tx.run(cypher, rows=rows)

def fetch_sources_page(session, skip: int, limit: int) -> List[Dict[str, Any]]:
    """
    Neo4j 5에서는 size((p)--()) 패턴 금지 같은 게 있지만
    size(listProperty)는 OK. referencedWorks는 리스트 프로퍼티라 그대로 써도 됨.
    """
    return session.run(
        """
        MATCH (p:Paper)
        WHERE p.referencedWorks IS NOT NULL AND size(p.referencedWorks) > 0
        RETURN p.openalexId AS src, p.referencedWorks AS refs
        SKIP $skip LIMIT $limit
        """,
        skip=skip, limit=limit
    ).data()

def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    with driver:
        with driver.session() as session:
            session.execute_write(ensure_schema)

        total_src = 0
        total_refs_seen = 0
        total_edges_made = 0
        total_batches = 0

        skip = 0
        while True:
            with driver.session() as session:
                result = fetch_sources_page(session, skip=skip, limit=BATCH)

                if not result:
                    break

                rows = []
                for r in result:
                    src = r.get("src")
                    refs_raw = r.get("refs") or []
                    if not src or not isinstance(refs_raw, list):
                        continue

                    # 정규화 + dedupe
                    refs_norm_set = set()
                    for x in refs_raw:
                        ref_url = normalize_ref_to_url(x)
                        if ref_url:
                            refs_norm_set.add(ref_url)

                    if not refs_norm_set:
                        continue

                    refs_norm = sorted(refs_norm_set)
                    rows.append({"src": src, "refs": refs_norm})

                    total_src += 1
                    total_refs_seen += len(refs_norm)

                if rows:
                    def work(tx):
                        connect_cites_batch(tx, rows)

                    session.execute_write(work)
                    total_batches += 1
                    # edges는 중복 MERGE라 정확 count는 아니지만 대략치(참고용)
                    total_edges_made += sum(len(x["refs"]) for x in rows)

                print(f"[batch] sources={total_src}, refs_seen={total_refs_seen}, "
                      f"edges_attempted={total_edges_made}, batches={total_batches}")

                skip += BATCH

        print("DONE.")
        print("Total source papers processed:", total_src)
        print("Total referencedWorks entries seen:", total_refs_seen)
        print("Total CITES edges attempted (upper bound):", total_edges_made)

if __name__ == "__main__":
    main()
