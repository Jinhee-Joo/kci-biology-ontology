import json
import re
import csv
import time
from typing import Any, Dict, List, Optional, Set

from google import genai
from google.genai.errors import ClientError

from extract_software import extract_software, SOFTWARE_TO_METHOD
from extract_methods import keyword_method_candidates  # ✅ 추가

# ====== 설정 ======
JSONL_IN = "openalex_evolbio_tiered_500.jsonl"   # 입력 파일
JSONL_OUT = "paper_annotations_all.jsonl"        # 누적 저장 (append)
CSV_OUT = "paper_annotations_all.csv"            # 누적 저장 (append)

# 모델은 자동선택
MODEL_ID = None  # type: ignore

BATCH_SIZE = 6               # 무료티어 안정적으로: 6~10 추천 (길면 줄여)
MAX_PAPERS = 500             # 최대 몇 편까지 처리할지 (500이면 500줄)

# retry/backoff
MAX_RETRIES = 8              # 같은 배치를 최대 몇 번 재시도할지
DEFAULT_WAIT_SEC = 30        # retryDelay 파싱 실패 시 기본 대기

FULL_METHOD_LIST = [
    "Maximum Likelihood",
    "Bayesian Inference",
    "Coalescent-based Model",
    "Birth–Death Model",
    "dN/dS Analysis",
    "Approximate Bayesian Computation (ABC)",
    "Phylogenetic Comparative Methods (PCM)",
    "Hidden Markov Models",
    "Simulation-based Inference",
    "Network Phylogenetics",
]

RESEARCH_TASK_LIST = [
    "Phylogeny Inference",
    "Divergence Time Estimation",
    "Species Delimitation",
    "Selection Detection",
    "Population Structure Analysis",
    "Demographic History Inference",
    "Trait Evolution Analysis",
    "Comparative Methods",
    "Biogeography",
    "Adaptation Inference",
    "Hybridization / Introgression Detection",
    "Gene Flow Analysis",
]

SOFTWARE_LIST = [
    "IQ-TREE", "RAxML", "BEAST", "MrBayes", "RevBayes",
    "STRUCTURE", "ADMIXTURE", "fastsimcoal", "dadi", "HyPhy"
]


# ====== 모델 선택 (강건 버전) ======
def pick_model_id(client: genai.Client) -> str:
    """
    ListModels가 제공하는 'supported_generation_methods'가 비어있거나
    SDK/버전 차이로 필드명이 달라도 동작하도록,
    실제로 generate_content를 "짧게" 호출해보고 되는 모델을 선택한다.

    우선순위:
    1) list() 결과 중 flash -> pro
    2) 안 되면 하드코딩 후보(환경마다 다를 수 있음) 순회
    """
    # 1) 일단 list()로 모델 이름을 모은다
    listed: List[str] = []
    try:
        for m in client.models.list():
            name = getattr(m, "name", "") or ""
            if name:
                listed.append(name)
    except Exception as e:
        print(f"⚠️ models.list() failed: {e}")

    # 보기 좋게 일부 출력 (디버깅에 도움)
    if listed:
        print("Models from ListModels (first 20):")
        for n in listed[:20]:
            print(" -", n)

    # 2) preference 정렬: gemini + flash 우선, 그다음 pro, 그다음 나머지
    def score(name: str) -> int:
        s = name.lower()
        sc = 0
        if "gemini" in s:
            sc += 100
        if "flash" in s:
            sc += 50
        if "pro" in s:
            sc += 20
        return -sc  # 오름차순 정렬용(큰 점수 우선)

    candidates = sorted(list(set(listed)), key=score)

    # 3) list 후보를 실제 호출로 검증
    for name in candidates:
        try:
            _ = client.models.generate_content(model=name, contents="ping")
            return name
        except ClientError as e:
            # NOT_FOUND / not supported 계열이면 다음 후보로
            msg = str(e)
            if "NOT_FOUND" in msg or "not found" in msg.lower() or "not supported" in msg.lower():
                continue
            # 다른 에러(권한/키/쿼터 등)는 바로 알려주는 게 낫다
            raise
        except Exception:
            # 네트워크/일시 오류 가능: 다음 후보로
            continue

    # 4) list 결과가 이상할 때를 위한 하드코딩 fallback
    fallback = [
        "models/gemini-1.5-flash",
        "models/gemini-1.5-pro",
        "models/gemini-1.0-pro",
        "models/gemini-pro",
    ]
    for name in fallback:
        try:
            _ = client.models.generate_content(model=name, contents="ping")
            return name
        except Exception:
            continue

    raise RuntimeError(
        "No working model found for generate_content.\n"
        "- If you are using Gemini Developer API: ensure GOOGLE_API_KEY is set.\n"
        "- If using Vertex: ensure ADC/project/region is set.\n"
        "- Also check models.list() output above."
    )


# ====== 유틸 ======
def load_done_ids(jsonl_out_path: str) -> Set[str]:
    """이미 처리한 paper_id 목록을 읽어서 resume 가능하게 함."""
    done = set()
    try:
        with open(jsonl_out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    pid = obj.get("id") or obj.get("paper_id")
                    if pid:
                        done.add(pid)
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return done


def safe_parse_json(text: str) -> Optional[Any]:
    """Gemini 응답에서 JSON 배열/오브젝트를 찾아 파싱."""
    if not text:
        return None
    t = text.strip()
    m = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", t)
    if not m:
        return None
    blob = m.group(1)
    try:
        return json.loads(blob)
    except Exception:
        return None


def extract_retry_delay_seconds(err: ClientError) -> int:
    """
    429 에러 메시지에 'Please retry in XXs' 또는 retryDelay가 포함되는 경우가 많음.
    파싱되면 그 초만큼 대기. 안 되면 DEFAULT_WAIT_SEC.
    """
    msg = ""
    try:
        msg = str(err)
    except Exception:
        return DEFAULT_WAIT_SEC

    m = re.search(r"retry in\s+(\d+(?:\.\d+)?)s", msg, re.IGNORECASE)
    if m:
        return max(1, int(float(m.group(1))) + 1)

    m2 = re.search(r"retryDelay'\s*:\s*'(\d+)s'", msg)
    if m2:
        return max(1, int(m2.group(1)) + 1)

    return DEFAULT_WAIT_SEC


def build_method_candidates(detected_softwares: List[str]) -> List[str]:
    """
    software → method 후보 (단, Method 라벨 목록과 교집합만)
    STRUCTURE 같은 건 method가 아니라서 자연스럽게 후보가 비게 됨(정상)
    """
    cands: List[str] = []
    for sw in detected_softwares:
        cands.extend(SOFTWARE_TO_METHOD.get(sw, []))
    cands = [c for c in cands if c in FULL_METHOD_LIST]
    return sorted(set(cands))


def build_batch_prompt(batch_payload: List[Dict[str, Any]]) -> str:
    payload_json = json.dumps(batch_payload, ensure_ascii=False)

    return f"""
You are an assistant that classifies evolutionary biology papers.

Input is a JSON array. For EACH item:
- Use ONLY labels from the lists below.
- Do NOT invent labels.
- Evidence must be a short quote from the abstract.

Constraints per paper:
- If method_candidates is non-empty: choose Method ONLY from method_candidates.
- Software output must be a subset of software_detected.
- If software_detected is empty: Software MUST be [].
- If unsure: return empty list for that field.

Return JSON ONLY with this schema (array length must match input length):
[
  {{
    "paper_id": "...",
    "ResearchTask": [{{"label":"...","confidence":0.0,"evidence":"..."}}],
    "Method":       [{{"label":"...","confidence":0.0,"evidence":"..."}}],
    "Software":     [{{"label":"...","confidence":0.0,"evidence":"..."}}]
  }}
]

Allowed labels:

ResearchTask:
{chr(10).join("- " + x for x in RESEARCH_TASK_LIST)}

Method:
{chr(10).join("- " + x for x in FULL_METHOD_LIST)}

Software:
{chr(10).join("- " + x for x in SOFTWARE_LIST)}

Papers(JSON):
{payload_json}
""".strip()


def call_gemini_with_retry(client: genai.Client, prompt: str, model_id: str) -> str:
    """429 등 발생 시 자동 대기 + 재시도."""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(model=model_id, contents=prompt)
            return resp.text or ""
        except ClientError as e:
            last_err = e
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_sec = extract_retry_delay_seconds(e)
                print(f"⚠️ 429 quota/rate hit. Waiting {wait_sec}s then retry (attempt {attempt}/{MAX_RETRIES})...")
                time.sleep(wait_sec)
                continue
            raise
        except Exception as e:
            last_err = e
            wait_sec = min(60, 2 ** attempt)
            print(f"⚠️ transient error. Waiting {wait_sec}s then retry (attempt {attempt}/{MAX_RETRIES})...")
            time.sleep(wait_sec)

    raise RuntimeError(f"Failed after retries. Last error: {last_err}")


def ensure_csv_header(csv_path: str) -> None:
    try:
        with open(csv_path, "r", encoding="utf-8") as _:
            return
    except FileNotFoundError:
        pass

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id", "title", "year", "tier",
                "software_detected",
                "method_candidates",
                "ResearchTask_labels",
                "Method_labels",
                "Software_labels",
            ],
        )
        writer.writeheader()


def labels_only(arr: Any) -> str:
    if not isinstance(arr, list):
        return ""
    out = []
    for x in arr:
        if isinstance(x, dict) and x.get("label"):
            out.append(x["label"])
    return "|".join(out)


def main():
    global MODEL_ID

    done_ids = load_done_ids(JSONL_OUT)
    print(f"Already done: {len(done_ids)} papers (resume enabled)")

    papers: List[Dict[str, Any]] = []
    with open(JSONL_IN, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= MAX_PAPERS:
                break
            papers.append(json.loads(line))

    todo = [p for p in papers if (p.get("id") not in done_ids)]
    print(f"To process now: {len(todo)} papers")

    if not todo:
        print("Nothing to do. All processed.")
        return

    out_jsonl = open(JSONL_OUT, "a", encoding="utf-8")
    ensure_csv_header(CSV_OUT)

    out_csv = open(CSV_OUT, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(
        out_csv,
        fieldnames=[
            "id", "title", "year", "tier",
            "software_detected",
            "method_candidates",
            "ResearchTask_labels",
            "Method_labels",
            "Software_labels",
        ],
    )

    client = genai.Client()

    # ✅ 여기서 실제로 되는 모델을 찾아서 고정
    MODEL_ID = pick_model_id(client)
    print("✅ Using MODEL_ID:", MODEL_ID)

    total = len(todo)
    processed = 0

    for start in range(0, total, BATCH_SIZE):
        chunk = todo[start:start + BATCH_SIZE]

        payload = []
        local_detected = {}

        for p in chunk:
            pid = p.get("id", "")
            title = p.get("title", "") or ""
            abstract = p.get("abstract", "") or ""
            topics = p.get("topics_concepts", []) or []

            text_for_methods = f"{title}\n{abstract}"

            softwares = extract_software(text_for_methods)

            # ✅ 1) software -> method 후보
            method_cands_sw = build_method_candidates(softwares)

            # ✅ 2) keyword -> method 후보 (FULL_METHOD_LIST와 교집합)
            method_cands_kw = [
                m for m in keyword_method_candidates(text_for_methods)
                if m in FULL_METHOD_LIST
            ]

            # ✅ 3) 합치고(순서 유지) 중복 제거
            method_cands = list(dict.fromkeys(method_cands_sw + method_cands_kw))

            local_detected[pid] = (softwares, method_cands, method_cands_kw)

            payload.append({
                "paper_id": pid,
                "title": title,
                "abstract": abstract,
                "topics_concepts": topics[:12],
                "software_detected": softwares,
                "method_candidates": method_cands,
            })

        prompt = build_batch_prompt(payload)

        raw = call_gemini_with_retry(client, prompt, MODEL_ID)
        parsed = safe_parse_json(raw)

        if not isinstance(parsed, list):
            raise RuntimeError(
                "Gemini output is not a JSON list. Raw (first 500 chars):\n"
                + (raw[:500] if raw else "")
            )

        by_id = {x.get("paper_id"): x for x in parsed if isinstance(x, dict)}

        for p in chunk:
            pid = p.get("id", "")
            softwares, method_cands, method_cands_kw = local_detected.get(pid, ([], [], []))
            ann = by_id.get(pid, {"paper_id": pid, "ResearchTask": [], "Method": [], "Software": []})

            record = {
                **p,
                "_detected": {
                    "software": softwares,
                    "method_candidates": method_cands,
                    "method_candidates_kw": method_cands_kw,  # ✅ (선택) 키워드 기반 후보도 저장
                    "model": MODEL_ID,
                },
                "_annotation": {
                    "ResearchTask": ann.get("ResearchTask", []),
                    "Method": ann.get("Method", []),
                    "Software": ann.get("Software", []),
                }
            }

            out_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

            writer.writerow({
                "id": pid,
                "title": p.get("title", ""),
                "year": p.get("year", ""),
                "tier": p.get("tier", ""),
                "software_detected": "|".join(softwares),
                "method_candidates": "|".join(method_cands),
                "ResearchTask_labels": labels_only(record["_annotation"]["ResearchTask"]),
                "Method_labels": labels_only(record["_annotation"]["Method"]),
                "Software_labels": labels_only(record["_annotation"]["Software"]),
            })

            processed += 1

        out_jsonl.flush()
        out_csv.flush()

        end = min(start + BATCH_SIZE, total)
        print(
            f"[{processed}/{total}] batch {start+1}-{end} saved "
            f"(software hits in batch: {sum(1 for p in chunk if local_detected.get(p.get('id',''), ([], [], []))[0])})"
        )

    out_jsonl.close()
    out_csv.close()
    try:
        client.close()
    except Exception:
        pass

    print("DONE. Saved:", JSONL_OUT, "and", CSV_OUT)


if __name__ == "__main__":
    main()
