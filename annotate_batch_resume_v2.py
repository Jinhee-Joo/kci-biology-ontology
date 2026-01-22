# annotate_batch_resume_v2.py
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
from extract_assumptions import keyword_assumption_candidates
from extract_context import keyword_context_candidates

# ====== 설정 ======
JSONL_IN = "openalex_evolbio_tiered_500.jsonl"     # 입력 파일
JSONL_OUT = "paper_annotations_all_v2.jsonl"       # 누적 저장 (append)
CSV_OUT = "paper_annotations_all_v2.csv"           # 누적 저장 (append)

MODEL_ID = None  # type: ignore

BATCH_SIZE = 6
MAX_PAPERS = 500

MAX_RETRIES = 8
DEFAULT_WAIT_SEC = 30

# ====== 라벨 리스트 ======
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
]

MODEL_ORGANISM_LIST = ["Drosophila", "Arabidopsis", "Mouse", "Yeast"]

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

# ====== 모델 선택 ======
def pick_model_id(client: genai.Client) -> str:
    listed: List[str] = []
    try:
        for m in client.models.list():
            name = getattr(m, "name", "") or ""
            if name:
                listed.append(name)
    except Exception as e:
        print(f"⚠️ models.list() failed: {e}")

    if listed:
        print("Models from ListModels (first 20):")
        for n in listed[:20]:
            print(" -", n)

    def score(name: str) -> int:
        s = name.lower()
        sc = 0
        if "gemini" in s:
            sc += 100
        if "flash" in s:
            sc += 50
        if "pro" in s:
            sc += 20
        return -sc

    candidates = sorted(list(set(listed)), key=score)

    for name in candidates:
        try:
            _ = client.models.generate_content(model=name, contents="ping")
            return name
        except ClientError as e:
            msg = str(e)
            if "NOT_FOUND" in msg or "not found" in msg.lower() or "not supported" in msg.lower():
                continue
            raise
        except Exception:
            continue

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

    raise RuntimeError("No working model found for generate_content.")


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


def call_gemini_with_retry(client: genai.Client, prompt: str, model_id: str) -> str:
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
                "datatype_candidates",
                "taxon_candidates",
                "model_organisms",
                "assumption_candidates",
                "context_candidates",
                "ResearchTask_labels",
                "Method_labels",
                "Software_labels",
                "DataType_labels",
                "Taxon_labels",
                "ModelOrganism_labels",
                "Assumption_labels",
                "Context_labels",
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


# ====== 후보 생성 ======
def build_method_candidates(softwares: List[str], text: str) -> List[str]:
    # software → method
    cands: List[str] = []
    for sw in softwares:
        cands.extend(SOFTWARE_TO_METHOD.get(sw, []))

    # keyword → method
    cands.extend(keyword_method_candidates(text))

    # Method 라벨 목록에 있는 것만
    cands = [c for c in cands if c in FULL_METHOD_LIST]
    return sorted(set(cands))


def build_batch_prompt(batch_payload: List[Dict[str, Any]]) -> str:
    payload_json = json.dumps(batch_payload, ensure_ascii=False)

    return f"""
You are an assistant that classifies evolutionary biology papers.

Input is a JSON array. For EACH item:
- Use ONLY labels from the allowed lists below.
- Do NOT invent labels.
- Evidence must be a short quote from the abstract.

Constraints per paper:
- If method_candidates is non-empty: choose Method ONLY from method_candidates.
- Software output must be a subset of software_detected. If software_detected is empty: Software MUST be [].
- If datatype_candidates is non-empty: choose DataType ONLY from datatype_candidates.
- If taxon_candidates is non-empty: choose Taxon ONLY from taxon_candidates.
- If model_organisms is non-empty: choose ModelOrganism ONLY from model_organisms.
- If assumption_candidates is non-empty: choose Assumption ONLY from assumption_candidates.
- If context_candidates is non-empty: choose Context ONLY from context_candidates.
- If unsure: return empty list for that field.

Return JSON ONLY with this schema (array length must match input length):
[
  {{
    "paper_id": "...",
    "ResearchTask":   [{{"label":"...","confidence":0.0,"evidence":"..."}}],
    "Method":         [{{"label":"...","confidence":0.0,"evidence":"..."}}],
    "Software":       [{{"label":"...","confidence":0.0,"evidence":"..."}}],
    "DataType":       [{{"label":"...","confidence":0.0,"evidence":"..."}}],
    "Taxon":          [{{"label":"...","confidence":0.0,"evidence":"..."}}],
    "ModelOrganism":  [{{"label":"...","confidence":0.0,"evidence":"..."}}],
    "Assumption":     [{{"label":"...","confidence":0.0,"evidence":"..."}}],
    "Context":        [{{"label":"...","confidence":0.0,"evidence":"..."}}]
  }}
]

Allowed labels:

ResearchTask:
{chr(10).join("- " + x for x in RESEARCH_TASK_LIST)}

Method:
{chr(10).join("- " + x for x in FULL_METHOD_LIST)}

Software:
{chr(10).join("- " + x for x in SOFTWARE_LIST)}

DataType:
{chr(10).join("- " + x for x in DATATYPE_LIST)}

Taxon:
{chr(10).join("- " + x for x in TAXON_LIST)}

ModelOrganism:
{chr(10).join("- " + x for x in MODEL_ORGANISM_LIST)}

Assumption:
{chr(10).join("- " + x for x in ASSUMPTION_LIST)}

Context:
{chr(10).join("- " + x for x in CONTEXT_LIST)}

Papers(JSON):
{payload_json}
""".strip()


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
            "datatype_candidates",
            "taxon_candidates",
            "model_organisms",
            "assumption_candidates",
            "context_candidates",
            "ResearchTask_labels",
            "Method_labels",
            "Software_labels",
            "DataType_labels",
            "Taxon_labels",
            "ModelOrganism_labels",
            "Assumption_labels",
            "Context_labels",
        ],
    )

    client = genai.Client()
    MODEL_ID = pick_model_id(client)
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

            text_for_rules = f"{title}\n{abstract}\n" + " ".join(topics[:20])  # 룰 기반 recall 올리기

            softwares = extract_software(f"{title}\n{abstract}")
            method_cands = build_method_candidates(softwares, text_for_rules)

            datatype_cands = [x for x in keyword_datatype_candidates(text_for_rules) if x in DATATYPE_LIST]
            taxon_cands = [x for x in keyword_taxon_candidates(text_for_rules) if x in TAXON_LIST]
            model_orgs = [x for x in keyword_model_organisms(text_for_rules) if x in MODEL_ORGANISM_LIST]
            assumption_cands = [x for x in keyword_assumption_candidates(text_for_rules) if x in ASSUMPTION_LIST]
            context_cands = [x for x in keyword_context_candidates(text_for_rules) if x in CONTEXT_LIST]

            local_detected[pid] = {
                "software": softwares,
                "method_candidates": method_cands,
                "datatype_candidates": datatype_cands,
                "taxon_candidates": taxon_cands,
                "model_organisms": model_orgs,
                "assumption_candidates": assumption_cands,
                "context_candidates": context_cands,
            }

            payload.append({
                "paper_id": pid,
                "title": title,
                "abstract": abstract,
                "topics_concepts": topics[:12],
                "software_detected": softwares,
                "method_candidates": method_cands,
                "datatype_candidates": datatype_cands,
                "taxon_candidates": taxon_cands,
                "model_organisms": model_orgs,
                "assumption_candidates": assumption_cands,
                "context_candidates": context_cands,
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
            ann = by_id.get(pid, {
                "paper_id": pid,
                "ResearchTask": [], "Method": [], "Software": [],
                "DataType": [], "Taxon": [], "ModelOrganism": [],
                "Assumption": [], "Context": [],
            })

            record = {
                **p,
                "_detected": {
                    **det,
                    "model": MODEL_ID,
                },
                "_annotation": {
                    "ResearchTask": ann.get("ResearchTask", []),
                    "Method": ann.get("Method", []),
                    "Software": ann.get("Software", []),
                    "DataType": ann.get("DataType", []),
                    "Taxon": ann.get("Taxon", []),
                    "ModelOrganism": ann.get("ModelOrganism", []),
                    "Assumption": ann.get("Assumption", []),
                    "Context": ann.get("Context", []),
                }
            }

            out_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

            writer.writerow({
                "id": pid,
                "title": p.get("title", ""),
                "year": p.get("year", ""),
                "tier": p.get("tier", ""),
                "software_detected": "|".join(det.get("software", [])),
                "method_candidates": "|".join(det.get("method_candidates", [])),
                "datatype_candidates": "|".join(det.get("datatype_candidates", [])),
                "taxon_candidates": "|".join(det.get("taxon_candidates", [])),
                "model_organisms": "|".join(det.get("model_organisms", [])),
                "assumption_candidates": "|".join(det.get("assumption_candidates", [])),
                "context_candidates": "|".join(det.get("context_candidates", [])),
                "ResearchTask_labels": labels_only(record["_annotation"]["ResearchTask"]),
                "Method_labels": labels_only(record["_annotation"]["Method"]),
                "Software_labels": labels_only(record["_annotation"]["Software"]),
                "DataType_labels": labels_only(record["_annotation"]["DataType"]),
                "Taxon_labels": labels_only(record["_annotation"]["Taxon"]),
                "ModelOrganism_labels": labels_only(record["_annotation"]["ModelOrganism"]),
                "Assumption_labels": labels_only(record["_annotation"]["Assumption"]),
                "Context_labels": labels_only(record["_annotation"]["Context"]),
            })

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
