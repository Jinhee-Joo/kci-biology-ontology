# annotate_batch_resume_v3.py
import json
import re
import csv
import time
from typing import Any, Dict, List, Optional, Set

from google import genai
from google.genai.errors import ClientError

from extract_software import extract_software, SOFTWARE_TO_METHOD
from extract_methods import keyword_method_candidates

from extract_datatype import keyword_datatype_candidates
from extract_taxon import keyword_taxon_candidates, keyword_model_organisms
from extract_context import keyword_context_candidates

# (선택) Assumption까지 쓸 거면 import
try:
    from extract_assumptions import keyword_assumption_candidates
    USE_ASSUMPTION = True
except Exception:
    USE_ASSUMPTION = False


# ====== 설정 ======
JSONL_IN  = "openalex_evolbio_tiered_500.jsonl"
JSONL_OUT = "paper_annotations_all_v3.jsonl"
CSV_OUT   = "paper_annotations_all_v3.csv"

MODEL_ID = "models/gemini-2.0-flash"

BATCH_SIZE  = 6
MAX_PAPERS  = 500

MAX_RETRIES       = 8
DEFAULT_WAIT_SEC  = 30


# ====== 라벨 리스트 (온톨로지에서 쓰는 “정식 라벨”만) ======
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

DATATYPE_LIST = [
    "Whole Genome Sequence",
    "Reduced Representation (RADseq, GBS)",
    "SNP Genotype Data",
    "mtDNA",
    "Nuclear Gene Sequences",
    "Morphometric Data",
    "Behavioral Data",
    "Ecological / Environmental Variables",
    "Fossil Record",
    "Occurrence / Distribution Records",
    "Transcriptome (RNA-seq)",
]

# Taxon은 계층이 가능하지만, MVP는 “상위 그룹 + 모델생물”로 고정 추천
TAXON_LIST = [
    "Vertebrates",
    "Invertebrates",
    "Mammalia",
    "Aves",
    "Reptilia",
    "Teleost Fish",
    "Insects",
    "Plants",
    "Angiosperms",
    "Model Organisms",
    "Drosophila",
    "Arabidopsis",
    "Mouse",
    "Yeast",
]

CONTEXT_LIST = [
    "Island Systems",
    "Mainland Systems",
    "Marine Environment",
    "Freshwater Environment",
    "Tropical Region",
    "Temperate Region",
    "Polar Region",
    "High-altitude Environment",
    "Fragmented Habitat",
]

ASSUMPTION_LIST = [
    "Molecular Clock",
    "Strict Clock",
    "Relaxed Clock",
    "Neutral Evolution Assumption",
    "Selection Model",
    "Panmictic Population",
    "No Gene Flow",
    "Gene Flow Allowed",
    "Incomplete Lineage Sorting Considered",
    "No Recombination",
    "Constant Population Size",
    "Bottleneck Model",
    "Expansion Model",
]


# ====== 모델 선택 (간단 안정 버전) ======
def pick_model_id(client: genai.Client) -> str:
    # 텍스트 라벨링은 flash/pro 텍스트 모델이 안정적
    preferred = [
        "models/gemini-2.0-flash",
        "models/gemini-1.5-flash",
        "models/gemini-1.5-pro",
    ]
    for name in preferred:
        try:
            _ = client.models.generate_content(model=name, contents="ping")
            return name
        except Exception:
            continue

    # 그래도 안 되면 list()에서 되는 것 아무거나
    try:
        for m in client.models.list():
            name = getattr(m, "name", "") or ""
            if not name:
                continue
            try:
                _ = client.models.generate_content(model=name, contents="ping")
                return name
            except Exception:
                continue
    except Exception:
        pass

    raise RuntimeError("No working model found. Check GOOGLE_API_KEY / quota / model access.")


# ====== 유틸 ======
def load_done_ids(jsonl_out_path: str) -> Set[str]:
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
    if not text:
        return None

    s = text.strip()

    # 1) ```json ... ``` 코드펜스 제거
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    # 2) 첫 '[' ~ 마지막 ']' 또는 첫 '{' ~ 마지막 '}' 만 뽑기
    l_arr, r_arr = s.find("["), s.rfind("]")
    l_obj, r_obj = s.find("{"), s.rfind("}")

    if l_arr != -1 and r_arr != -1 and r_arr > l_arr:
        blob = s[l_arr:r_arr + 1]
    elif l_obj != -1 and r_obj != -1 and r_obj > l_obj:
        blob = s[l_obj:r_obj + 1]
    else:
        return None

    # 3) trailing comma 제거: ", ]" / ", }" 패턴 정리
    blob = re.sub(r",\s*(\]|\})", r"\1", blob)

    # 4) 파싱
    try:
        return json.loads(blob)
    except Exception:
        return None



def extract_retry_delay_seconds(err: ClientError) -> int:
    msg = str(err) if err else ""
    m = re.search(r"retry in\s+(\d+(?:\.\d+)?)s", msg, re.IGNORECASE)
    if m:
        return max(1, int(float(m.group(1))) + 1)
    m2 = re.search(r"retryDelay'\s*:\s*'(\d+)s'", msg)
    if m2:
        return max(1, int(m2.group(1)) + 1)
    return DEFAULT_WAIT_SEC


def call_gemini_with_retry(client: genai.Client, prompt: str, model_id: str) -> str:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # 1) 먼저 JSON mode 시도
            try:
                resp = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config={"response_mime_type": "application/json"},
                )
                return resp.text or ""
            except ClientError as e:
                # 2) gemma처럼 JSON mode 미지원이면 일반 호출로 fallback
                if "JSON mode is not enabled" in str(e) or "response_mime_type" in str(e):
                    resp = client.models.generate_content(model=model_id, contents=prompt)
                    return resp.text or ""
                raise

        except ClientError as e:
            last_err = e
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_sec = extract_retry_delay_seconds(e)
                print(f"⚠️ 429 hit. Waiting {wait_sec}s (attempt {attempt}/{MAX_RETRIES})...")
                time.sleep(wait_sec)
                continue
            raise
        except Exception as e:
            last_err = e
            wait_sec = min(60, 2 ** attempt)
            print(f"⚠️ transient error. Waiting {wait_sec}s (attempt {attempt}/{MAX_RETRIES})...")
            time.sleep(wait_sec)

    raise RuntimeError(f"Failed after retries. Last error: {last_err}")



def labels_only(arr: Any) -> str:
    if not isinstance(arr, list):
        return ""
    out = []
    for x in arr:
        if isinstance(x, dict) and x.get("label"):
            out.append(x["label"])
    return "|".join(out)


def build_method_candidates_from_software(detected_softwares: List[str]) -> List[str]:
    cands: List[str] = []
    for sw in detected_softwares:
        cands.extend(SOFTWARE_TO_METHOD.get(sw, []))
    cands = [c for c in cands if c in FULL_METHOD_LIST]
    return sorted(set(cands))


def build_batch_prompt(batch_payload: List[Dict[str, Any]]) -> str:
    payload_json = json.dumps(batch_payload, ensure_ascii=False)

    # Assumption 스키마는 선택
    assumption_schema = ""
    assumption_allowed = ""
    if USE_ASSUMPTION:
        assumption_schema = '\n    "Assumption":   [{"label":"...","confidence":0.0,"evidence":"..."}],'
        assumption_allowed = "\nAssumption:\n" + "\n".join("- " + x for x in ASSUMPTION_LIST) + "\n"

    return f"""
You are an assistant that labels evolutionary biology papers.

Input is a JSON array. For EACH item:
- Use ONLY labels from the allowed lists below.
- Do NOT invent labels.
- Evidence must be a short quote from the abstract.

Hard constraints per paper:
- Software must be a subset of software_detected. If software_detected is empty => Software MUST be [].
- If method_candidates is non-empty: Method labels must be chosen ONLY from method_candidates.
- If method_candidates is empty: Method can still be [], or choose from keyword_method_candidates (provided).
- Taxon must be chosen ONLY from taxon_candidates (provided). If none => [].
- DataType must be chosen ONLY from datatype_candidates (provided). If none => [].
- Context (Region/Environment) must be chosen ONLY from context_candidates (provided). If none => [].
- If unsure, return [] for that field.

Return JSON ONLY with this schema (array length must match input length):
[
  {{
    "paper_id": "...",
    "ResearchTask": [{{"label":"...","confidence":0.0,"evidence":"..."}}],
    "Method":       [{{"label":"...","confidence":0.0,"evidence":"..."}}],
    "Software":     [{{"label":"...","confidence":0.0,"evidence":"..."}}],
    "Taxon":        [{{"label":"...","confidence":0.0,"evidence":"..."}}],
    "DataType":     [{{"label":"...","confidence":0.0,"evidence":"..."}}],
    "Context":      [{{"label":"...","confidence":0.0,"evidence":"..."}}],{assumption_schema}
  }}
]

Allowed labels:

ResearchTask:
{chr(10).join("- " + x for x in RESEARCH_TASK_LIST)}

Method:
{chr(10).join("- " + x for x in FULL_METHOD_LIST)}

Software:
{chr(10).join("- " + x for x in SOFTWARE_LIST)}

Taxon:
{chr(10).join("- " + x for x in TAXON_LIST)}

DataType:
{chr(10).join("- " + x for x in DATATYPE_LIST)}

Context:
{chr(10).join("- " + x for x in CONTEXT_LIST)}
{assumption_allowed}

Papers(JSON):
{payload_json}
""".strip()


def ensure_csv_header(csv_path: str) -> None:
    try:
        with open(csv_path, "r", encoding="utf-8") as _:
            return
    except FileNotFoundError:
        pass

    base_fields = [
        "id", "title", "year", "tier",
        "software_detected",
        "method_candidates_sw",
        "method_candidates_kw",
        "datatype_candidates",
        "taxon_candidates",
        "context_candidates",
        "ResearchTask_labels",
        "Method_labels",
        "Software_labels",
        "Taxon_labels",
        "DataType_labels",
        "Context_labels",
    ]
    if USE_ASSUMPTION:
        base_fields.append("Assumption_labels")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields)
        writer.writeheader()


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
    fieldnames = [
        "id", "title", "year", "tier",
        "software_detected",
        "method_candidates_sw",
        "method_candidates_kw",
        "datatype_candidates",
        "taxon_candidates",
        "context_candidates",
        "ResearchTask_labels",
        "Method_labels",
        "Software_labels",
        "Taxon_labels",
        "DataType_labels",
        "Context_labels",
    ]
    if USE_ASSUMPTION:
        fieldnames.append("Assumption_labels")

    writer = csv.DictWriter(out_csv, fieldnames=fieldnames)

    client = genai.Client()
    print("✅ Using MODEL_ID:", MODEL_ID)

    total = len(todo)
    processed = 0

    for start in range(0, total, BATCH_SIZE):
        chunk = todo[start:start + BATCH_SIZE]

        payload: List[Dict[str, Any]] = []
        local_detected: Dict[str, Any] = {}

        for p in chunk:
            pid = p.get("id", "")
            title = p.get("title", "") or ""
            abstract = p.get("abstract", "") or ""
            topics = p.get("topics_concepts", []) or []

            text = f"{title}\n{abstract}"

            softwares = extract_software(text)
            method_cands_sw = build_method_candidates_from_software(softwares)
            method_cands_kw = [m for m in keyword_method_candidates(text) if m in FULL_METHOD_LIST]

            datatype_cands = [d for d in keyword_datatype_candidates(text) if d in DATATYPE_LIST]

            taxon_cands = [t for t in keyword_taxon_candidates(text) if t in TAXON_LIST]
            # 모델생물 키워드가 잡히면 Model Organisms + 해당 생물 라벨도 후보에 포함
            mos = keyword_model_organisms(text)  # e.g., ["Drosophila"]
            for mo in mos:
                if mo in TAXON_LIST and mo not in taxon_cands:
                    taxon_cands.append(mo)
                if "Model Organisms" in TAXON_LIST and "Model Organisms" not in taxon_cands:
                    taxon_cands.append("Model Organisms")

            context_cands = [c for c in keyword_context_candidates(text) if c in CONTEXT_LIST]

            assumption_cands = []
            if USE_ASSUMPTION:
                assumption_cands = [a for a in keyword_assumption_candidates(text) if a in ASSUMPTION_LIST]

            local_detected[pid] = {
                "software": softwares,
                "method_candidates_sw": method_cands_sw,
                "method_candidates_kw": method_cands_kw,
                "datatype_candidates": sorted(set(datatype_cands)),
                "taxon_candidates": sorted(set(taxon_cands)),
                "context_candidates": sorted(set(context_cands)),
                "assumption_candidates": sorted(set(assumption_cands)),
            }

            payload.append({
                "paper_id": pid,
                "title": title,
                "abstract": abstract,
                "topics_concepts": topics[:12],
                "software_detected": softwares,
                "method_candidates": sorted(set(method_cands_sw + method_cands_kw)),
                "keyword_method_candidates": method_cands_kw,
                "datatype_candidates": sorted(set(datatype_cands)),
                "taxon_candidates": sorted(set(taxon_cands)),
                "context_candidates": sorted(set(context_cands)),
                "assumption_candidates": sorted(set(assumption_cands)) if USE_ASSUMPTION else [],
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
            det = local_detected.get(pid, {})
            ann = by_id.get(pid, {})

            record = {
                **p,
                "_detected": {
                    **det,
                    "model": MODEL_ID,
                },
                "_annotation": {
                    "ResearchTask": ann.get("ResearchTask", []) if isinstance(ann.get("ResearchTask", []), list) else [],
                    "Method":       ann.get("Method", []) if isinstance(ann.get("Method", []), list) else [],
                    "Software":     ann.get("Software", []) if isinstance(ann.get("Software", []), list) else [],
                    "Taxon":        ann.get("Taxon", []) if isinstance(ann.get("Taxon", []), list) else [],
                    "DataType":     ann.get("DataType", []) if isinstance(ann.get("DataType", []), list) else [],
                    "Context":      ann.get("Context", []) if isinstance(ann.get("Context", []), list) else [],
                }
            }

            if USE_ASSUMPTION:
                record["_annotation"]["Assumption"] = ann.get("Assumption", []) if isinstance(ann.get("Assumption", []), list) else []

            out_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

            row = {
                "id": pid,
                "title": p.get("title", ""),
                "year": p.get("year", ""),
                "tier": p.get("tier", ""),
                "software_detected": "|".join(det.get("software", [])),
                "method_candidates_sw": "|".join(det.get("method_candidates_sw", [])),
                "method_candidates_kw": "|".join(det.get("method_candidates_kw", [])),
                "datatype_candidates": "|".join(det.get("datatype_candidates", [])),
                "taxon_candidates": "|".join(det.get("taxon_candidates", [])),
                "context_candidates": "|".join(det.get("context_candidates", [])),
                "ResearchTask_labels": labels_only(record["_annotation"]["ResearchTask"]),
                "Method_labels": labels_only(record["_annotation"]["Method"]),
                "Software_labels": labels_only(record["_annotation"]["Software"]),
                "Taxon_labels": labels_only(record["_annotation"]["Taxon"]),
                "DataType_labels": labels_only(record["_annotation"]["DataType"]),
                "Context_labels": labels_only(record["_annotation"]["Context"]),
            }
            if USE_ASSUMPTION:
                row["Assumption_labels"] = labels_only(record["_annotation"].get("Assumption", []))

            writer.writerow(row)
            processed += 1

        out_jsonl.flush()
        out_csv.flush()

        end = min(start + BATCH_SIZE, total)
        print(f"[{processed}/{total}] batch {start+1}-{end} saved")

    out_jsonl.close()
    out_csv.close()
    try:
        client.close()
    except Exception:
        pass

    print("DONE. Saved:", JSONL_OUT, "and", CSV_OUT)


if __name__ == "__main__":
    main()
