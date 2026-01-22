import json
import re
import csv
from typing import Any, Dict, List, Optional

from google import genai

from extract_software import extract_software, SOFTWARE_TO_METHOD

JSONL_IN = "openalex_evolbio_tiered_500.jsonl"   # 너 파일명
JSONL_OUT = "paper_annotations_30.jsonl"
CSV_OUT = "paper_annotations_30.csv"

MODEL_ID = "gemini-3-flash-preview"  # 공식 quickstart 예시 모델 :contentReference[oaicite:3]{index=3}

# ---- 유틸: Gemini 응답에서 JSON만 안전하게 파싱 ----
def safe_parse_json(text: str) -> Optional[dict]:
    """
    Gemini가 JSON만 준다고 해도 가끔 앞뒤 텍스트가 붙을 수 있어서,
    첫 번째 { ... } 덩어리를 찾아 파싱한다.
    """
    if not text:
        return None
    text = text.strip()

    # 이미 JSON처럼 시작하면 바로 시도
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass

    # 가장 바깥 JSON 오브젝트 추출(대충)
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        return None


def build_method_candidates(detected_softwares: List[str], full_method_list: List[str]) -> List[str]:
    # 소프트웨어 → 매핑된 후보들 모으기
    cands: List[str] = []
    for sw in detected_softwares:
        cands.extend(SOFTWARE_TO_METHOD.get(sw, []))

    # ⚠️ 너 매핑에 Task가 섞일 수 있어서(STRUCTURE -> Population Structure Analysis 같은),
    # Method 후보만 남기기 위해 full_method_list와 교집합 필터
    cands = [c for c in cands if c in full_method_list]

    # 중복 제거
    return sorted(set(cands))


def build_prompt(paper: Dict[str, Any], software_detected: List[str], method_candidates: List[str]) -> str:
    title = paper.get("title", "")
    abstract = paper.get("abstract", "") or ""
    topics = paper.get("topics_concepts", []) or []
    topics_preview = ", ".join(topics[:12])

    software_block = "\n".join([f"- {s}" for s in software_detected]) if software_detected else "- None"
    if method_candidates:
        method_block = "\n".join([f"- {m}" for m in method_candidates])
        method_constraint = "If MethodCandidates are provided, you MUST choose Method labels ONLY from MethodCandidates."
    else:
        method_block = "- (use full Method list)"
        method_constraint = "MethodCandidates are not provided, so you may choose Method labels from the full Method list."

    prompt = f"""
You are an assistant that classifies evolutionary biology papers.

Your task:
- Read the paper information.
- Classify the paper using ONLY the labels provided below.
- DO NOT invent new labels.
- If no label applies, return an empty list [].
- For each selected label, provide:
  - label
  - confidence (0.0 to 1.0)
  - evidence (a short quote from the abstract)

IMPORTANT CONSTRAINTS:
- {method_constraint}
- Software labels must match SoftwareDetected (if SoftwareDetected is None, Software must be []).

The output MUST follow this exact JSON schema:

{{
  "ResearchTask": [],
  "Method": [],
  "Software": []
}}

Each list must contain objects with:
- label
- confidence
- evidence

Return JSON only. No explanation.

===== Classification Labels =====

ResearchTask:
- Phylogeny Inference
- Divergence Time Estimation
- Species Delimitation
- Selection Detection
- Population Structure Analysis
- Demographic History Inference
- Trait Evolution Analysis
- Comparative Methods
- Biogeography
- Adaptation Inference
- Hybridization / Introgression Detection
- Gene Flow Analysis

Method:
- Maximum Likelihood
- Bayesian Inference
- Coalescent-based Model
- Birth–Death Model
- dN/dS Analysis
- Approximate Bayesian Computation (ABC)
- Phylogenetic Comparative Methods (PCM)
- Hidden Markov Models
- Simulation-based Inference
- Network Phylogenetics

Software:
- IQ-TREE
- RAxML
- BEAST
- MrBayes
- RevBayes
- STRUCTURE
- ADMIXTURE
- fastsimcoal
- dadi
- HyPhy

===== Detected Information =====

SoftwareDetected:
{software_block}

MethodCandidates:
{method_block}

===== Paper =====
Title:
{title}

Abstract:
{abstract}

Topics_Concepts:
{topics_preview}
""".strip()

    return prompt


def main():
    # full method list (프롬프트의 Method 라벨 목록과 동일하게 유지)
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

    # Gemini client (환경변수 GEMINI_API_KEY를 자동 사용) :contentReference[oaicite:4]{index=4}
    client = genai.Client()

    # 입력 30편 읽기
    papers: List[Dict[str, Any]] = []
    with open(JSONL_IN, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 30:
                break
            papers.append(json.loads(line))

    # 출력 파일 준비
    out_jsonl = open(JSONL_OUT, "w", encoding="utf-8")
    out_csv = open(CSV_OUT, "w", encoding="utf-8", newline="")
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
    writer.writeheader()

    for idx, paper in enumerate(papers, start=1):
        title = paper.get("title", "")
        abstract = paper.get("abstract", "") or ""
        text_for_sw = f"{title}\n{abstract}"

        # 1) software 정규식 추출
        software_detected = extract_software(text_for_sw)

        # 2) software → method 후보 좁히기
        method_candidates = build_method_candidates(software_detected, FULL_METHOD_LIST)

        # 3) 프롬프트 만들기
        prompt = build_prompt(paper, software_detected, method_candidates)

        # 4) Gemini 호출
        resp = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
        )
        parsed = safe_parse_json(resp.text)

        if parsed is None:
            # 실패하면 최소 정보라도 저장
            parsed = {"ResearchTask": [], "Method": [], "Software": []}

        # 5) 저장 (원본 paper + annotation)
        record = {
            **paper,
            "_detected": {
                "software": software_detected,
                "method_candidates": method_candidates,
                "model": MODEL_ID,
            },
            "_annotation": parsed,
        }
        out_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_jsonl.flush()

        # 6) CSV 요약 저장
        def labels_only(arr):
            if not isinstance(arr, list):
                return ""
            return "|".join([x.get("label", "") for x in arr if isinstance(x, dict)])

        writer.writerow({
            "id": paper.get("id", ""),
            "title": title,
            "year": paper.get("year", ""),
            "tier": paper.get("tier", ""),
            "software_detected": "|".join(software_detected),
            "method_candidates": "|".join(method_candidates),
            "ResearchTask_labels": labels_only(parsed.get("ResearchTask", [])),
            "Method_labels": labels_only(parsed.get("Method", [])),
            "Software_labels": labels_only(parsed.get("Software", [])),
        })
        out_csv.flush()

        print(f"[{idx}/30] done - software={software_detected} method_cands={method_candidates}")

    out_jsonl.close()
    out_csv.close()
    try:
        client.close()
    except Exception:
        pass

    print("Saved:", JSONL_OUT, "and", CSV_OUT)


if __name__ == "__main__":
    main()
