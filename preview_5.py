import json

FILE = "openalex_evolbio_tiered_500.jsonl" 

def main():
    with open(FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            obj = json.loads(line)

            print("=" * 80)
            print(f"[{i+1}] id:", obj.get("id"))
            print("title:", obj.get("title"))
            print("year:", obj.get("year"))
            print("language:", obj.get("language"))

            abstract = obj.get("abstract") or ""
            print("abstract_len:", len(abstract))
            print("abstract_preview:", abstract[:200].replace("\n", " "))

            keywords = obj.get("keywords") or []
            print("keywords_count:", len(keywords))
            print("keywords_preview:", keywords[:10])

            topics = obj.get("topics_concepts") or []
            print("topics_concepts_count:", len(topics))
            print("topics_concepts_preview:", topics[:10])

            ref = obj.get("referenced_works") or []
            print("referenced_works_count:", len(ref))

            # 어떤 키들이 들어있는지 전체 키 목록도 확인
            print("all_keys:", sorted(list(obj.keys())))

if __name__ == "__main__":
    main()
