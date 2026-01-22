Dataset Size Note (Why 453 papers, not exactly 500?)

This dataset intentionally prioritizes ontology quality over raw size.
The final count (453 papers) reflects a conservative, structure-first sampling strategy rather than a fixed-number crawl.

본 데이터셋은 OpenAlex로부터 수집한 논문을 온톨로지 구축 목적에 맞게 선별(tiering)한 결과이며, 최종 논문 수는 453편이다.
이는 오류나 누락이 아니라 의도적인 품질 중심 필터링의 결과이다.

🔹 Tiered Sampling Strategy
본 프로젝트에서는 논문을 다음과 같은 2단계(Tier) 기준으로 선별했다.

Tier 1 (Core papers)
evolutionary biology와의 개념적 연관성 점수(evolbio_score) 가 높은 논문
핵심 개념 구조를 안정적으로 형성하기 위한 정확도 우선 샘플

Tier 2 (Extended papers)
점수는 다소 낮지만 관련 주제 확장을 위한 논문
온톨로지의 개념 다양성과 확장성 확보 목적

초기 목표는
Tier 1: 250편
Tier 2: 250편
총 500편 이었으나,

🔹 Why fewer than 500?
OpenAlex 데이터의 실제 분포상 높은 개념 연관성(evolbio_score)을 만족하는 Tier 1 후보 수가 제한적이었고 무작정 기준을 완화할 경우,
온톨로지의 핵심 개념이 흐려지고 “의미 기반 검색 / 설명 가능성”이라는 프로젝트 목표가 약화될 위험이 있었다.
따라서 본 프로젝트에서는 데이터 수를 인위적으로 맞추기보다 온톨로지 품질을 우선하는 보수적 선택을 채택했다.

그 결과:
Tier 1: 203편
Tier 2: 250편
Total: 453 papers

🔹 Why this is acceptable (and desirable)
온톨로지는 데이터 양보다 개념 구조의 명확성이 핵심입니다. 본 데이터셋은 핵심 개념(Tier 1)을 중심으로 안정적인 구조를 형성하고 Tier 2를 통해 도메인 확장과 탐색 가능성을 확보합니다.
향후 기준 완화 또는 도메인 확장을 통해 자연스럽게 확장 가능한 구조입니다.

🔹 Validation
수집된 모든 논문은 다음 검증을 통과했습니다.
JSON 파싱 무결성 검사
tier 값 및 score 범위 검증
연도(year) 범위 검증
domain 분포 확인

검증 스크립트:
validate_json.py
validate_tier_json.py


