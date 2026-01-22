import csv

path = "paper_annotations_all_v2.csv"

with open(path, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    rows = []
    for i, row in enumerate(reader, start=1):
        rows.append(row)

print("Parsed rows:", len(rows))
print("Columns:", reader.fieldnames)

# 빈 라벨 비율
def empty_ratio(col):
    empty = sum(1 for r in rows if not (r.get(col) or "").strip())
    return empty, empty / max(1, len(rows))

for c in ["ResearchTask_labels", "Method_labels", "Software_labels"]:
    e, ratio = empty_ratio(c)
    print(f" - {c}: empty {e} ({ratio:.1%})")

# ResearchTask 라벨 분포(상위 20)
from collections import Counter
task_counter = Counter()
for r in rows:
    s = (r.get("ResearchTask_labels") or "").strip()
    if s:
        for t in s.split("|"):
            task_counter[t.strip()] += 1

print("\nTop ResearchTask labels:")
for k, v in task_counter.most_common(20):
    print(v, k)
