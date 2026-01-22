import json
from collections import Counter, defaultdict

path = "openalex_evolbio_tiered_500.jsonl"

# 허용 값들
ALLOWED_TIERS = {"tier1", "tier2"}
REQUIRED_FIELDS = ["id", "title", "year", "language", "domain", "tier", "evolbio_score"]

bad = 0
tier_counts = Counter()
domain_by_tier = defaultdict(Counter)

def fail(i, msg, line=None):
    global bad
    bad += 1
    print("검증 실패")
    print("라인 번호:", i)
    print("이유:", msg)
    if line:
        print("내용(앞 200자):", line[:200])

with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue

        # 1) JSON 파싱
        try:
            obj = json.loads(line)
        except Exception as e:
            fail(i, f"JSON 파싱 오류: {e}", line)
            break

        # 2) 필수 필드 존재
        missing = [k for k in REQUIRED_FIELDS if k not in obj]
        if missing:
            fail(i, f"필수 필드 누락: {missing}", line)
            break

        # 3) tier 값 검증
        tier = obj.get("tier")
        if tier not in ALLOWED_TIERS:
            fail(i, f"tier 값이 이상함: {tier} (허용: {sorted(ALLOWED_TIERS)})", line)
            break

        # 4) evolbio_score 검증 (0~1 숫자)
        score = obj.get("evolbio_score")
        if not isinstance(score, (int, float)):
            fail(i, f"evolbio_score 타입이 숫자가 아님: {type(score)}", line)
            break
        if not (0.0 <= float(score) <= 1.0):
            fail(i, f"evolbio_score 범위 이상: {score} (0~1이어야 함)", line)
            break

        # 5) year 타입/범위 간단 체크
        year = obj.get("year")
        if not isinstance(year, int) or not (1800 <= year <= 2100):
            fail(i, f"year 값 이상: {year}", line)
            break

        # 6) 간단 통계
        tier_counts[tier] += 1
        domain_by_tier[tier][obj.get("domain")] += 1

if bad == 0:
    print("JSON + tier 검증 모두 통과 (정상)")
    print("tier_counts:", dict(tier_counts))
    for t in sorted(tier_counts.keys()):
        print(f"[{t}] domain_counts:", dict(domain_by_tier[t]))
