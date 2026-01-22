import json

path = "openalex_evolbio_tiered_500.jsonl"
bad = 0

with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
        except Exception as e:
            bad += 1
            print("JSON 오류 발생")
            print("라인 번호:", i)
            print("에러:", e)
            print("내용(앞 200자):", line[:200])
            break

if bad == 0:
    print("JSON 파싱 오류 없음 (정상)")
