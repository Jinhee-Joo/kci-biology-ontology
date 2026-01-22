from extract_software import extract_software

# 테스트용 논문 1편 
title = "Contrasting signatures of genomic divergence during sympatric speciation"

abstract = """
We inferred phylogenetic relationships using BEAST v2.6 under a Bayesian framework.
Molecular clock models were applied to estimate divergence times.
"""


# 여기서 extract_software를 실제로 호출
text = title + " " + abstract
softwares = extract_software(text)

# 결과 출력
print("Detected softwares:", softwares)
