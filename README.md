 KCI Biology Ontology Project

본 프로젝트는 국내 생물학(의학) 분야 논문을 온톨로지 관점에서 재구성하기 위한 기초 데이터 수집 단계이다.  
기존 키워드 기반 논문 검색의 한계를 보완하기 위해 논문을 단순 텍스트가 아닌 개념/방법/관계의 집합으로 다루는 것을 목표로 한다.


1. 프로젝트 개요
- 주제: 생물학(의학) 분야 논문 검색
- 문제의식: 논문 검색은 단순 문자열 검색이 아니라 연구 문제(Task), 방법(Method), 대상(Target)등의 개념 구조 검색 문제
- 접근 방식:
  - KCI에서 공식 제공하는 OAI-PMH(Open Archives Initiative Protocol for Metadata Harvesting)를 이용해 국내 학술지 논문 메타데이터를 수집, 이후 온톨로지 기반 검색 및 분석으로 확장


2. 데이터 수집
- 데이터 출처: KCI 공식 OAI-PMH 엔드포인트 사용 (API Key 불필요)
- 수집 범위:
  - 학술 논문 (ARTI set)
  - 논문 수: 100편
  - 수집 항목:
    - 제목 (title)
    - 초록 (abstract)
    - 저자 (authors)
    - 키워드/주제 (keywords)
    - 발행연도 (year)
    - 언어 (language)
- 수집 방식:
  - ListRecords 요청을 통한 메타데이터 수집
  - resumptionToken을 활용한 페이지네이션 처리


3. 프로젝트 구조
kci-biology-ontology/
├─ fetch_kci.py # KCI OAI-PMH에서 논문 메타데이터 수집 스크립트
├─ data/
│ └─ papers.csv # 수집된 논문 100편 CSV 파일
└─ README.md


4. 결과 예시
CSV 파일에는 다음과 같은 컬럼이 포함된다:
title	
abstract
authors	
keywords
year
language	


5. 향후 계획
- 논문을 Task / Method / Target / Topic 등의 개념으로 분류
- 개념 간 관계(is-a, alternative-to 등) 정의
- 키워드 검색과 온톨로지 기반 검색 결과 비교
- “A지만 B는 아닌 논문”과 같은 의도 기반 검색 구현


6. 참고
본 프로젝트는 연구/교육 목적의 메타데이터 수집을 목표로 하며 원문(PDF)은 수집하지 않는다.
데이터는 KCI에서 공식 제공하는 OAI-PMH 프로토콜을 사용하였다
