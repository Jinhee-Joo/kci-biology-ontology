# annotate_batch_resume_v3.py
import json
import re
import csv
import time
from typing import Any, Dict, List, Optional, Set

from google import genai
from google.genai.errors import ClientError

from scripts.extract.extract_software import extract_software, SOFTWARE_TO_METHOD
from scripts.extract.extract_methods import keyword_method_candidates
from scripts.extract.extract_datatype import keyword_datatype_candidates
from scripts.extract.extract_taxon import keyword_taxon_candidates, keyword_model_organisms
from scripts.extract.extract_geoscope import keyword_geoscope_candidates
from scripts.extract.extract_environment import keyword_environment_candidates
from scripts.extract.extract_contribution import keyword_contribution_candidates


# ====== 설정 ======
JSONL_IN  = "data/raw/openalex_evolbio_tiered_500.jsonl"
JSONL_OUT = "data/raw/annotations/paper_annotations_all_v3.jsonl"
CSV_OUT   = "data/raw/annotations/paper_annotations_all_v3.csv"

MODEL_ID = "models/gemini-2.0-flash"

BATCH_SIZE  = 3
MAX_PAPERS  = 500

MAX_RETRIES       = 8
DEFAULT_WAIT_SEC  = 30


# ====== 라벨 리스트 (온톨로지 “정식 라벨”만) ======
RESEARCH_TASK_LIST = [
    "Phylogeny Inference",
    "Divergence Time Estimation",
    "Species Delimitation",
    "Population Structure Analysis",
    "Demographic History Inference",
    "Selection / Positive Selection Detection",
    "Trait Evolution / Ancestral State Reconstruction",
    "Phylogenetic Comparative Methods",
    "Biogeography",
    "Adaptation Inference",
    "Gene Flow / Introgression Analysis",
    "Macroevolutionary Pattern Analysis",
    "Eco-evolutionary Dynamics",
    "Evolutionary Theory / Conceptual Analysis",
]

FULL_METHOD_LIST = [
    "Maximum Likelihood",
    "Bayesian Inference",
    "Coalescent-based Model",
    "Birth–Death Model",
    "dN/dS Analysis",
    "Approximate Bayesian Computation",
    "Phylogenetic Comparative Methods",
    "Hidden Markov Models",
    "Simulation-based Inference",
    "Network Phylogenetics",
]

SOFTWARE_LIST = [
    "IQ-TREE", "RAxML",
    "BEAST", "MrBayes", "RevBayes",
    "STRUCTURE", "ADMIXTURE",
    "fastsimcoal", "dadi",
    "HyPhy"
]

DATATYPE_LIST = [
    "Whole Genome Sequence",
    "Reduced Representation (RADseq / GBS)",
    "SNP Genotype Data",
    "mtDNA",
    "Nuclear Gene Sequences",
    "Transcriptome (RNA-seq)",
    "Morphometric / Phenotypic Trait Data",
    "Behavioral Data",
    "Experimental Evolution Data",
    "Ecological / Environmental Variables",
    "Fossil Record",
    "Occurrence / Distribution Records",
    "Time-series Population Data",
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
    "Model Organisms",
    "Drosophila",
    "Arabidopsis",
    "Mouse",
    "Yeast",
]

GEOSCOPE_LIST = [
    "Single Location",
    "Multiple Locations",
    "Global / Public Database Scale",
    "Not Specified",
]

ENVIRONMENT_TYPE_LIST = [
    "Marine",
    "Freshwater",
    "Terrestrial",
    "Island",
    "High-altitude",
    "Polar",
    "Tropical",
    "Temperate",
    "Fragmented Habitat",
    "Not Specified",
]

CONTRIBUTION_TYPE_LIST = [
    "Empirical Study",
    "Method / Model Paper",
    "Resource / Dataset Paper",
    "Review / Survey",
    "Protocol / Technical Note",
]


# ====== 후보 개수(관계 늘리기용) ======
# LLM 출력 자체가 너무 보수적이면 관계가 안 늘어서, 여기서부터 “복수 후보”를 강제
TOPK_TARGET = {
    "ResearchTask": 4,
    "Method": 4,
    "Software": 5,
    "Taxon": 5,
    "DataType": 4,
    "GeoScope": 2,
    "EnvironmentType": 3,
    "ContributionType": 4,
}


# ====== 모델 선택(유지) ======
def pick_model_id(client: genai.Client) -> str:
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
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    l_arr, r_arr = s.find("["), s.rfind("]")
    l_obj, r_obj = s.find("{"), s.rfind("}")

    if l_arr != -1 and r_arr != -1 and r_arr > l_arr:
        blob = s[l_arr:r_arr + 1]
    elif l_obj != -1 and r_obj != -1 and r_obj > l_obj:
        blob = s[l_obj:r_obj + 1]
    else:
        return None

    blob = re.sub(r",\s*(\]|\})", r"\1", blob)

    try:
        return json.loads(blob)
    except Exception:
        return None


def parse_json_list_with_retries(
    client: genai.Client,
    prompt: str,
    model_id: str,
    max_extra_tries: int = 2
) -> Optional[List[Any]]:
    raw = call_gemini_with_retry(client, prompt, model_id)
    parsed = safe_parse_json(raw)

    tries = 0
    while not isinstance(parsed, list) and tries < max_extra_tries:
        tries += 1
        print(f"⚠️ JSON parse failed. Retrying extra {tries}/{max_extra_tries} ...")
        time.sleep(2 * tries)
        raw = call_gemini_with_retry(client, prompt, model_id)
        parsed = safe_parse_json(raw)

    return parsed if isinstance(parsed, list) else None


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
            try:
                resp = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config={"response_mime_type": "application/json"},
                )
                return resp.text or ""
            except ClientError as e:
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
    """CSV용: label/name만 뽑기"""
    if not isinstance(arr, list):
        return ""
    out = []
    for x in arr:
        if isinstance(x, dict):
            v = x.get("label") or x.get("name")
            if v:
                out.append(str(v))
    return "|".join(out)


def build_method_candidates_from_software(detected_softwares: List[str]) -> List[str]:
    cands: List[str] = []
    for sw in detected_softwares:
        cands.extend(SOFTWARE_TO_METHOD.get(sw, []))
    cands = [c for c in cands if c in FULL_METHOD_LIST]
    return sorted(set(cands))


# ====== (추가) GeoScope / EnvironmentType / ContributionType 후보 생성 ======
def keyword_geoscope_candidates(text: str) -> List[str]:
    t = (text or "").lower()

    # Global/Public DB scale
    if any(k in t for k in ["public database", "genbank", "gbif", "biodiversity information facility",
                            "global dataset", "worldwide", "global scale", "meta-analysis", "meta analysis"]):
        return ["Global / Public Database Scale"]

    # multiple locations
    if any(k in t for k in ["across", "multiple sites", "multiple populations", "across populations",
                            "across regions", "across continents", "range-wide", "range wide", "multi-site", "multisite"]):
        return ["Multiple Locations"]

    # single location
    if any(k in t for k in ["in ", "at ", "from "]):  # 매우 약한 신호라 단독으로 확정하지 않음
        return ["Single Location", "Not Specified"]

    return ["Not Specified"]


def keyword_environment_candidates(text: str) -> List[str]:
    t = (text or "").lower()
    cands: List[str] = []

    # Marine / Freshwater / Terrestrial
    if any(k in t for k in ["marine", "ocean", "sea", "coastal", "reef", "intertidal"]):
        cands.append("Marine")
    if any(k in t for k in ["freshwater", "river", "lake", "stream"]):
        cands.append("Freshwater")
    if any(k in t for k in ["terrestrial", "forest", "grassland", "desert", "savanna", "landscape"]):
        cands.append("Terrestrial")

    # Island / High-altitude / Polar
    if any(k in t for k in ["island", "archipelago"]):
        cands.append("Island")
    if any(k in t for k in ["high altitude", "high-altitude", "alpine", "mountain", "montane"]):
        cands.append("High-altitude")
    if any(k in t for k in ["polar", "arctic", "antarctic"]):
        cands.append("Polar")

    # Tropical / Temperate
    if any(k in t for k in ["tropical", "tropics"]):
        cands.append("Tropical")
    if any(k in t for k in ["temperate"]):
        cands.append("Temperate")

    # Fragmented habitat
    if any(k in t for k in ["fragmented habitat", "habitat fragmentation", "fragmentation"]):
        cands.append("Fragmented Habitat")

    if not cands:
        cands.append("Not Specified")

    # 정식 라벨만 유지
    cands = [c for c in cands if c in ENVIRONMENT_TYPE_LIST]
    return sorted(set(cands))


def keyword_contribution_candidates(title: str, abstract: str) -> List[str]:
    t = f"{title}\n{abstract}".lower()
    cands: List[str] = []

    # Review
    if any(k in t for k in ["review", "survey", "meta-analysis", "meta analysis", "systematic review"]):
        cands.append("Review / Survey")

    # Resource/Dataset
    if any(k in t for k in ["dataset", "database", "resource", "genome assembly", "genome assemblies",
                            "reference genome", "data set", "data release", "catalog", "atlas"]):
        cands.append("Resource / Dataset Paper")

    # Protocol/Technical
    if any(k in t for k in ["protocol", "pipeline", "workflow", "tutorial", "technical note", "implementation"]):
        cands.append("Protocol / Technical Note")

    # Method/Model
    if any(k in t for k in ["we propose", "we introduce", "new method", "novel method", "model", "algorithm", "framework"]):
        cands.append("Method / Model Paper")

    # 기본값: empirical
    if not cands:
        cands.append("Empirical Study")
    else:
        # 리뷰/리소스/방법이 있어도 empirical도 같이 달 수 있게 (관계 늘리기)
        cands.append("Empirical Study")

    cands = [c for c in cands if c in CONTRIBUTION_TYPE_LIST]
    return sorted(set(cands))


# ====== 프롬프트 ======
def build_batch_prompt(batch_payload: List[Dict[str, Any]]) -> str:
    payload_json = json.dumps(batch_payload, ensure_ascii=False)

    return f"""
You are an assistant that labels evolutionary biology papers.

Input is a JSON array. For EACH item:
- Use ONLY labels from the allowed lists below.
- Do NOT invent labels.
- Evidence must be a SHORT quote from the abstract (exact phrase).
- Return MULTIPLE candidates per field to increase recall (up to the recommended max shown).
- If unsure, you may still output low-confidence candidates (>=0.20) rather than empty, AS LONG AS evidence supports it.
- confidence is a float in [0,1].

Hard constraints per paper:
- Software MUST be a subset of software_detected. If software_detected is empty => Software MUST be [].
- If method_candidates is non-empty: Method labels must be chosen ONLY from method_candidates.
- Taxon must be chosen ONLY from taxon_candidates (provided). If none => [].
- DataType must be chosen ONLY from datatype_candidates (provided). If none => [].
- GeoScope must be chosen ONLY from geoscope_candidates (provided). If none => ["Not Specified"].
- EnvironmentType must be chosen ONLY from environment_candidates (provided). If none => ["Not Specified"].
- ContributionType must be chosen ONLY from contribution_candidates (provided). If none => ["Empirical Study"].

Recommended max candidates per field:
- ResearchTask: up to {TOPK_TARGET["ResearchTask"]}
- Method: up to {TOPK_TARGET["Method"]}
- Software: up to {TOPK_TARGET["Software"]}
- Taxon: up to {TOPK_TARGET["Taxon"]}
- DataType: up to {TOPK_TARGET["DataType"]}
- GeoScope: up to {TOPK_TARGET["GeoScope"]}
- EnvironmentType: up to {TOPK_TARGET["EnvironmentType"]}
- ContributionType: up to {TOPK_TARGET["ContributionType"]}

Return JSON ONLY with this schema (array length must match input length):
[
  {{
    "paper_id": "...",
    "ResearchTask":     [{{"label":"...","confidence":0.0,"evidenceText":"..."}}],
    "Method":           [{{"label":"...","confidence":0.0,"evidenceText":"..."}}],
    "Software":         [{{"label":"...","confidence":0.0,"evidenceText":"..."}}],
    "Taxon":            [{{"label":"...","confidence":0.0,"evidenceText":"..."}}],
    "DataType":         [{{"label":"...","confidence":0.0,"evidenceText":"..."}}],
    "GeoScope":         [{{"label":"...","confidence":0.0,"evidenceText":"..."}}],
    "EnvironmentType":  [{{"label":"...","confidence":0.0,"evidenceText":"..."}}],
    "ContributionType": [{{"label":"...","confidence":0.0,"evidenceText":"..."}}]
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

GeoScope:
{chr(10).join("- " + x for x in GEOSCOPE_LIST)}

EnvironmentType:
{chr(10).join("- " + x for x in ENVIRONMENT_TYPE_LIST)}

ContributionType:
{chr(10).join("- " + x for x in CONTRIBUTION_TYPE_LIST)}

Papers(JSON):
{payload_json}

Return ONLY a valid JSON array. No markdown fences. No explanations. No trailing text.
The output MUST start with '[' and end with ']'.
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
        "geoscope_candidates",
        "environment_candidates",
        "contribution_candidates",
        "ResearchTask_labels",
        "Method_labels",
        "Software_labels",
        "Taxon_labels",
        "DataType_labels",
        "GeoScope_labels",
        "EnvironmentType_labels",
        "ContributionType_labels",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields)
        writer.writeheader()


def coerce_list(x: Any) -> List[Dict[str, Any]]:
    """LLM 출력 방어: list[dict]만 통과 + 키 통일(label/confidence/evidenceText)."""
    if not isinstance(x, list):
        return []
    out: List[Dict[str, Any]] = []
    for it in x:
        if not isinstance(it, dict):
            continue
        label = it.get("label") or it.get("name")
        if not label or not str(label).strip():
            continue
        conf = it.get("confidence", 0.0)
        try:
            conf = float(conf)
        except Exception:
            conf = 0.0
        ev = it.get("evidenceText") or it.get("evidence") or ""
        out.append({"label": str(label).strip(), "confidence": conf, "evidenceText": str(ev)})
    return out


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
        "geoscope_candidates",
        "environment_candidates",
        "contribution_candidates",
        "ResearchTask_labels",
        "Method_labels",
        "Software_labels",
        "Taxon_labels",
        "DataType_labels",
        "GeoScope_labels",
        "EnvironmentType_labels",
        "ContributionType_labels",
    ]
    writer = csv.DictWriter(out_csv, fieldnames=fieldnames)

    client = genai.Client()
    print("✅ Using MODEL_ID:", MODEL_ID)

    total = len(todo)
    processed = 0

    SKIP_LOG = "paper_annotations_skipped_v3.txt"

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
            method_candidates = sorted(set(method_cands_sw + method_cands_kw))

            datatype_cands = [d for d in keyword_datatype_candidates(text) if d in DATATYPE_LIST]

            taxon_cands = [t for t in keyword_taxon_candidates(text) if t in TAXON_LIST]
            mos = keyword_model_organisms(text)  # e.g., ["Drosophila"]
            for mo in mos:
                if mo in TAXON_LIST and mo not in taxon_cands:
                    taxon_cands.append(mo)
                if "Model Organisms" in TAXON_LIST and "Model Organisms" not in taxon_cands:
                    taxon_cands.append("Model Organisms")

            geoscope_cands = [g for g in keyword_geoscope_candidates(text) if g in GEOSCOPE_LIST]
            env_cands = [e for e in keyword_environment_candidates(text) if e in ENVIRONMENT_TYPE_LIST]
            contrib_cands = [c for c in keyword_contribution_candidates(title, abstract) if c in CONTRIBUTION_TYPE_LIST]

            local_detected[pid] = {
                "software": softwares,
                "method_candidates_sw": method_cands_sw,
                "method_candidates_kw": method_cands_kw,
                "datatype_candidates": sorted(set(datatype_cands)),
                "taxon_candidates": sorted(set(taxon_cands)),
                "geoscope_candidates": sorted(set(geoscope_cands)),
                "environment_candidates": sorted(set(env_cands)),
                "contribution_candidates": sorted(set(contrib_cands)),
            }

            payload.append({
                "paper_id": pid,
                "title": title,
                "abstract": abstract,
                "topics_concepts": topics[:12],
                "software_detected": softwares,
                "method_candidates": method_candidates,
                "keyword_method_candidates": method_cands_kw,
                "datatype_candidates": sorted(set(datatype_cands)),
                "taxon_candidates": sorted(set(taxon_cands)),
                "geoscope_candidates": sorted(set(geoscope_cands)) or ["Not Specified"],
                "environment_candidates": sorted(set(env_cands)) or ["Not Specified"],
                "contribution_candidates": sorted(set(contrib_cands)) or ["Empirical Study"],
            })

        prompt = build_batch_prompt(payload)
        parsed = parse_json_list_with_retries(client, prompt, MODEL_ID, max_extra_tries=2)

        if not isinstance(parsed, list):
            end = min(start + BATCH_SIZE, total)
            skip_ids = [p.get("id", "") for p in chunk]
            print("❌ [SKIP] Gemini output not parseable as JSON list. Skipping batch "
                  f"{start+1}-{end}. ids={skip_ids[:3]}{'...' if len(skip_ids)>3 else ''}")

            try:
                with open(SKIP_LOG, "a", encoding="utf-8") as sf:
                    for sid in skip_ids:
                        if sid:
                            sf.write(sid + "\n")
            except Exception:
                pass
            continue

        by_id = {x.get("paper_id"): x for x in parsed if isinstance(x, dict)}

        for p in chunk:
            pid = p.get("id", "")
            det = local_detected.get(pid, {})
            ann = by_id.get(pid, {}) or {}

            record = {
                **p,
                "_detected": {
                    **det,
                    "model": MODEL_ID,
                },
                "_annotation": {
                    "ResearchTask":     coerce_list(ann.get("ResearchTask")),
                    "Method":           coerce_list(ann.get("Method")),
                    "Software":         coerce_list(ann.get("Software")),
                    "Taxon":            coerce_list(ann.get("Taxon")),
                    "DataType":         coerce_list(ann.get("DataType")),
                    "GeoScope":         coerce_list(ann.get("GeoScope")),
                    "EnvironmentType":  coerce_list(ann.get("EnvironmentType")),
                    "ContributionType": coerce_list(ann.get("ContributionType")),
                }
            }

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
                "geoscope_candidates": "|".join(det.get("geoscope_candidates", [])),
                "environment_candidates": "|".join(det.get("environment_candidates", [])),
                "contribution_candidates": "|".join(det.get("contribution_candidates", [])),
                "ResearchTask_labels": labels_only(record["_annotation"]["ResearchTask"]),
                "Method_labels": labels_only(record["_annotation"]["Method"]),
                "Software_labels": labels_only(record["_annotation"]["Software"]),
                "Taxon_labels": labels_only(record["_annotation"]["Taxon"]),
                "DataType_labels": labels_only(record["_annotation"]["DataType"]),
                "GeoScope_labels": labels_only(record["_annotation"]["GeoScope"]),
                "EnvironmentType_labels": labels_only(record["_annotation"]["EnvironmentType"]),
                "ContributionType_labels": labels_only(record["_annotation"]["ContributionType"]),
            }

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
