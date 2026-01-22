import time
import json
import random
import re
import requests
from typing import Any, Dict, List, Optional, Tuple

BASE = "https://api.openalex.org"
MAILTO = "wnwlsgml0112@naver.com"

SEED = 42

# Tier thresholds
TIER1_MIN = 0.55
TIER2_MIN = 0.45  # Tier2: [0.45, 0.60)

TIER1_N = 250
TIER2_N = 250

# 후보 풀을 넉넉히 모아놓고 샘플링하는 방식
POOL_TARGET = 2000  # 최소 이 정도는 모아두면 quota 샘플링이 쉬움
PER_PAGE = 200

REQUIRE_ABSTRACT = True

# 도메인 쏠림 방지 상한선
DOMAIN_CAP_TIER1 = 88   # 250의 35% 정도
DOMAIN_CAP_TIER2 = 75   # 250의 30% 정도


# Utilities
def request_json(url: str, params: Dict[str, Any], timeout: int = 60, max_retries: int = 5) -> Dict[str, Any]:
    """간단한 재시도(backoff) 포함 GET JSON"""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            wait = 0.5 * (2 ** attempt)
            print(f"[warn] request failed (attempt {attempt+1}/{max_retries}): {e} -> sleep {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"Request failed after {max_retries} retries: {url}")


def pick_concept_id(query: str = "evolutionary biology") -> str:
    """Concept 검색 후 가장 적절해 보이는 concept id(C....)를 선택"""
    url = f"{BASE}/concepts"
    params = {"search": query, "per-page": 10, "mailto": MAILTO}
    data = request_json(url, params)

    results = data.get("results", [])
    if not results:
        raise RuntimeError("No concepts found for query:", query)

    for c in results:
        if (c.get("display_name") or "").strip().lower() == query.strip().lower():
            return c["id"].split("/")[-1]

    return results[0]["id"].split("/")[-1]


def invert_abstract(abstract_inverted_index: Optional[Dict[str, List[int]]]) -> Optional[str]:
    """abstract_inverted_index -> 원문 복원"""
    if not abstract_inverted_index:
        return None

    pos_to_word: Dict[int, str] = {}
    for word, positions in abstract_inverted_index.items():
        for p in positions:
            pos_to_word[p] = word

    if not pos_to_word:
        return None

    max_pos = max(pos_to_word.keys())
    words = [pos_to_word.get(i, "") for i in range(max_pos + 1)]
    text = " ".join([w for w in words if w])
    return text.strip() if text.strip() else None


def get_concept_score_for_work(work: Dict[str, Any], concept_id: str) -> float:
    """
    works 결과의 concepts 배열에는 보통
    {id: ".../Cxxxx", display_name: "...", score: 0.xxx} 형태가 들어있음.
    여기서 concept_id에 해당하는 score를 뽑아 evolbio_score로 사용.
    """
    concepts = work.get("concepts") or []
    target_url_suffix = f"/{concept_id}"
    for c in concepts:
        cid = c.get("id") or ""
        if cid.endswith(target_url_suffix):
            s = c.get("score")
            if isinstance(s, (int, float)):
                return float(s)
    return 0.0


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def classify_domain(w_simplified: Dict[str, Any]) -> str:
    """
    매우 러프한 도메인 분류(온톨로지 구축용 다양성 확보 목적)
    - 정확도보다 "쏠림 방지"가 목표라서 간단 키워드 매칭만.
    """
    title = normalize_text(w_simplified.get("title") or "")
    abstract = normalize_text(w_simplified.get("abstract") or "")
    kw = " ".join([normalize_text(x) for x in (w_simplified.get("keywords") or []) if isinstance(x, str)])
    topics = " ".join([normalize_text(x) for x in (w_simplified.get("topics_concepts") or []) if isinstance(x, str)])

    blob = " ".join([title, abstract, kw, topics])

    # 우선순위는 필요하면 조정
    if any(k in blob for k in ["sars-cov-2", "covid", "coronavirus", "influenza", "virus", "viral", "virome", "virology"]):
        return "virus"
    if any(k in blob for k in ["cancer", "tumor", "tumour", "oncology", "carcinoma"]):
        return "cancer"
    if any(k in blob for k in ["microbiome", "metagenom", "metagenome", "rhizosphere", "endosphere", "bacteria community"]):
        return "microbiome"
    if any(k in blob for k in ["plant", "arabidopsis", "zea mays", "wheat", "barley", "phyto", "botany", "crop"]):
        return "plant"
    if any(k in blob for k in ["human", "population", "haplotype", "genome diversity", "hgdp", "demography"]):
        return "human"
    if any(k in blob for k in ["software", "package", "tool", "pipeline", "workflow", "algorithm", "method", "iq-tree", "mega", "phytools"]):
        return "tool-method"

    return "other"


def simplify_work(w: Dict[str, Any], concept_id: str) -> Optional[Dict[str, Any]]:
    title = w.get("display_name")
    abstract = invert_abstract(w.get("abstract_inverted_index"))
    year = w.get("publication_year")
    language = w.get("language")

    if REQUIRE_ABSTRACT and not abstract:
        return None

    authorships = w.get("authorships") or []
    authors = []
    for a in authorships:
        author_obj = a.get("author") or {}
        name = author_obj.get("display_name")
        if name:
            authors.append(name)

    concepts = w.get("concepts") or []
    topic_keywords = [c.get("display_name") for c in concepts if c.get("display_name")]

    keywords = w.get("keywords") or []
    kw_names = []
    for k in keywords:
        if isinstance(k, dict) and k.get("display_name"):
            kw_names.append(k["display_name"])
        elif isinstance(k, str):
            kw_names.append(k)

    referenced_works = w.get("referenced_works") or []

    evolbio_score = get_concept_score_for_work(w, concept_id)

    row = {
        "id": w.get("id"),
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "keywords": kw_names,
        "topics_concepts": topic_keywords,
        "year": year,
        "language": language,
        "referenced_works": referenced_works,
        "evolbio_score": evolbio_score,
    }

    # 도메인 분류 추가
    row["domain"] = classify_domain(row)
    return row


def crawl_pool(concept_id: str, pool_target: int = POOL_TARGET) -> List[Dict[str, Any]]:
    """
    넓게 긁어 후보 풀 확보 → 로컬에서 Tier/Quota 샘플링
    """
    url = f"{BASE}/works"
    filter_str = f"from_publication_date:2020-01-01,language:en,concepts.id:{concept_id}"

    cursor = "*"
    collected: List[Dict[str, Any]] = []

    print(f"Filter: year>=2020, language=en, includes concept={concept_id}")
    print(f"Post-filter: require_abstract={REQUIRE_ABSTRACT}, will build pool_target={pool_target}")

    while len(collected) < pool_target:
        params = {
            "filter": filter_str,
            "per-page": PER_PAGE,
            "cursor": cursor,
            "mailto": MAILTO,
        }
        data = request_json(url, params)
        results = data.get("results", [])
        if not results:
            break

        for w in results:
            row = simplify_work(w, concept_id)
            if row is None:
                continue
            collected.append(row)
            if len(collected) >= pool_target:
                break

        cursor = (data.get("meta") or {}).get("next_cursor")
        print(f"pool_collected={len(collected)} next_cursor={'YES' if cursor else 'NO'}")
        if not cursor:
            break
        time.sleep(0.2)

    return collected


def sample_with_domain_caps(
    items: List[Dict[str, Any]],
    target_n: int,
    domain_cap: int,
    rng: random.Random
) -> List[Dict[str, Any]]:
    """
    items를 랜덤 셔플 후, domain별 cap 넘지 않게 target_n개 선택
    """
    shuffled = items[:]
    rng.shuffle(shuffled)

    picked: List[Dict[str, Any]] = []
    domain_counts: Dict[str, int] = {}

    for it in shuffled:
        if len(picked) >= target_n:
            break
        d = it.get("domain", "other")
        if domain_counts.get(d, 0) >= domain_cap:
            continue
        domain_counts[d] = domain_counts.get(d, 0) + 1
        picked.append(it)

    return picked


def build_tiered_dataset(pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Tier1/Tier2 분리 + 도메인 cap 적용해 250/250 뽑기
    """
    rng = random.Random(SEED)

    tier1_candidates = [x for x in pool if (x.get("evolbio_score", 0.0) >= TIER1_MIN)]
    tier2_candidates = [x for x in pool if (TIER2_MIN <= x.get("evolbio_score", 0.0) < TIER1_MIN)]

    print(f"Tier1 candidates: {len(tier1_candidates)} (score>={TIER1_MIN})")
    print(f"Tier2 candidates: {len(tier2_candidates)} ({TIER2_MIN}<=score<{TIER1_MIN})")

    if len(tier1_candidates) < TIER1_N:
        print("[warn] Tier1 후보가 부족합니다. POOL_TARGET을 늘리거나 TIER1_MIN을 낮추세요.")
    if len(tier2_candidates) < TIER2_N:
        print("[warn] Tier2 후보가 부족합니다. POOL_TARGET을 늘리거나 TIER2_MIN을 낮추세요.")

    tier1 = sample_with_domain_caps(tier1_candidates, TIER1_N, DOMAIN_CAP_TIER1, rng)
    tier2 = sample_with_domain_caps(tier2_candidates, TIER2_N, DOMAIN_CAP_TIER2, rng)

    # 혹시 cap 때문에 target_n을 못 채우면 cap 무시하고 추가로 채움
    def top_up(picked: List[Dict[str, Any]], candidates: List[Dict[str, Any]], target_n: int):
        if len(picked) >= target_n:
            return picked
        picked_ids = set(x["id"] for x in picked if x.get("id"))
        rest = [c for c in candidates if c.get("id") and c["id"] not in picked_ids]
        rng.shuffle(rest)
        for c in rest:
            if len(picked) >= target_n:
                break
            picked.append(c)
        return picked

    tier1 = top_up(tier1, tier1_candidates, TIER1_N)
    tier2 = top_up(tier2, tier2_candidates, TIER2_N)

    # tier 라벨링
    for x in tier1:
        x["tier"] = "tier1"
    for x in tier2:
        x["tier"] = "tier2"

    dataset = tier1 + tier2
    rng.shuffle(dataset)

    # 간단 통계 출력
    def stats(items: List[Dict[str, Any]], name: str):
        dc: Dict[str, int] = {}
        for it in items:
            d = it.get("domain", "other")
            dc[d] = dc.get(d, 0) + 1
        print(f"\n[{name}] size={len(items)} domain_counts={dc}")

    stats(tier1, "Tier1")
    stats(tier2, "Tier2")
    stats(dataset, "Total")

    return dataset


def main():
    concept_id = pick_concept_id("evolutionary biology")
    print("Chosen concept_id:", concept_id)

    pool = crawl_pool(concept_id, pool_target=POOL_TARGET)
    print(f"Pool built: {len(pool)}")

    dataset = build_tiered_dataset(pool)

    out_path = "openalex_evolbio_tiered_500.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(dataset)} works to {out_path}")


if __name__ == "__main__":
    main()
