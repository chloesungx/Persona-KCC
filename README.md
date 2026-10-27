# Persona: KCC experiment code

KCC 논문 「사회 시뮬레이션을 위한 LLM 기반 페르소나 에이전트의 외부 정보 실천 방식 비교」의 실험 코드 정리본

## 포함 범위

- A: 탐색·스캐닝 — 에이전트가 검색어를 만들고 관련 정보를 능동적으로 선택
- B: 모니터링 — 공통 정보 풀을 평가하고 중요도가 높은 정보를 선택
- C: 프록시 — 사회관계망의 게시물 중 페르소나와 가까운 정보를 선택
- D: 외부 정보 없음 — 페르소나 정보만 사용하는 기준 조건
- 논문 정량 실험용 ANES 100명 표본, VCF0842 설문 매핑, 기사 풀, Proxy 네트워크
- LLM provider와 프롬프트, RMSE 등 평가 함수


## 설치와 실행

```bash
python -m venv .venv
pip install -r requirements.txt
```

`.env.example`을 `.env`로 복사하고 `ANTHROPIC_API_KEY`를 입력한 뒤 조건별로 실행

```bash
python run_experiment.py --condition A --num-personas 100 --seed 42
python run_experiment.py --condition B --num-personas 100 --seed 42
python run_experiment.py --condition C --num-personas 100 --seed 42
python run_experiment.py --condition D --num-personas 100 --seed 42
```

기본 모델은 논문에서 사용한 `claude-sonnet-4-6`이며 결과는 `data/output/`에 저장됨

## 데이터

| 경로 | 용도 |
| --- | --- |
| `data/anes_2020_2024_100.csv` | 100명 평가 표본과 정답 |
| `data/survey/survey_anes_VCF0842.json` | 페르소나 설문 문항 매핑 |
| `data/external/data_VCF0842_sample_balance.json` | 환경 규제 관련 외부 정보 풀 |
| `data/network/proxy_network20.json` | Proxy 조건의 프로필과 관계망 |

ANES 변수의 의미와 원자료 이용 조건은 [American National Election Studies](https://electionstudies.org/) 문서를 따름
