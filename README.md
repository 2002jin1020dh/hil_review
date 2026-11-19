# hil-review: VLM 보조 라벨링에서 사람 검토를 얼마나 줄일 수 있을까?

**연구 질문:** 로봇이 과제에 성공했는지를 판단하는 **보상 분류기(reward classifier)** 를 학습시키려면 "성공/실패" 라벨이 필요합니다.
이 라벨을 **로컬 VLM(이미지를 이해하는 AI 모델)** 이 먼저 달고 사람은 **일부만 검토**하면, 사람의 라벨링 시간을 얼마나 줄일 수 있을까요?
그리고 그렇게 해도 실제 로봇의 온라인 강화학습(HIL-SERL) 성공률이 유지될까요?
- 로봇: SO-101 (LeRobot 0.6.1 이상)
- 과제: 물체를 집어서 용기에 넣기 (pick-and-place)

> **현재 상태: 프로그램은 동작합니다. 다만 합성(가짜) 데이터로만 확인했습니다.**
> 실제 로봇, GPU, VLM, 사람이 직접 단 라벨은 아직 한 번도 쓰지 않았습니다.
> `results/synthetic/` 아래 결과는 "프로그램이 제대로 연결됐는지" 확인한 것일 뿐, 실제 성능에 대한 **증거가 아닙니다**.
> 자세한 내용은 `PROJECT_STATUS.md`(진행 상황)와 `LIMITATIONS.md`(한계)를 보세요.

## 비교하는 세 가지 조건
| 조건 | 사람이 검토하는 양 | 최종 라벨 |
|---|---|---|
| A | 전체의 100% | 사람 라벨 |
| B | 25% (무작위로 고른 에피소드) | 검토한 것은 사람 라벨, 나머지는 VLM 라벨 |

25%는 기본적으로 **에피소드 수** 기준이고, 후속 실험에서는 **사람 시간** 기준(`budget_unit: seconds`)으로도 비교할 수 있습니다.

| C | 0% (VLM만 사용) | VLM 라벨 |

## 무엇을 할 수 있나
| 단계 | 명령 | 어디서 | 여기서 검증됐나? |
|---|---|---|---|
| 합성 데이터로 실험 전체 실행 | `hilreview run --config configs/synthetic_base.yaml` | 어디서나 | 예 |
| 검토 도구 연습 | `hilreview make-demo-data` 후 `hilreview review ...` | 맥/노트북 | 예 (자동 입력으로) |
| 에피소드 단위 데이터 나누기를 고정 | `hilreview split` | 어디서나 | 예 |
| VLM 라벨 제안 받기 (로컬 서버) | `hilreview vlm-label` | 노트북 | 가짜 서버로만 |
| 사람 검토 / 블라인드 평가 라벨링 | `hilreview review --purpose pool_review / audit` | 맥/노트북 | 자동 입력으로만 |
| A/B/C 조건 계산, 사람 시간, 라벨 품질, 라벨 파일 내보내기 | `hilreview run --config configs/files_template.yaml --export-labels` | 어디서나 | 연습 데이터로만 |
| LeRobot 데이터 변환 / 보상 데이터셋 / 점수 계산 | `hilreview export-keyframes / build-reward-dataset / score` | 노트북 | **아니오 (미검증)** |
| 블라인드 실행 순서표 만들기 | `hilreview schedule` | 어디서나 | 예 |
| 실제 로봇 실행 중 에피소드 결과 기록 (글자 2개) | `hilreview log-episodes` | 노트북 | 자동 입력으로만 |
| 로컬 VLM 서버 점검 (연결, 응답, 확신 점수, 속도) | `hilreview vlm-check` | 노트북 | 가짜 서버로만 |
| 실제 로봇 결과 요약 | `hilreview online-summary` | 어디서나 | 가짜 기록으로만 |

**로봇을 움직이는 명령은 하나도 없습니다.** HIL-SERL 학습 자체는 LeRobot의 `lerobot.rl.learner`, `lerobot.rl.actor`로 사용자가 직접 실행합니다.

## 노트북에서 시작하기 (GPU와 다른 환경은 건드리지 않음)
처음 설치할 때는 `hil-review.bundle`과 `setup_on_laptop.sh`를 받은 뒤 다음을 실행합니다.
```bash
bash ~/Downloads/setup_on_laptop.sh ~/Downloads/hil-review.bundle          # ~/research/hil-review 폴더를 만들고 테스트 실행
bash ~/Downloads/setup_on_laptop.sh ~/Downloads/hil-review.bundle --push   # 위 작업 + GitHub(gawbi)에 비공개로 업로드
```
이미 설치된 상태에서 새 버전을 받을 때는 **먼저 업데이트하고, 그다음 스크립트를 실행**합니다. 파일 이름의 버전 부분은 받은 파일에 맞게 바꾸세요.
```bash
cd ~/research/hil-review
git pull --ff-only ~/Downloads/hil-review-<버전>.bundle main && git push    # 노트북 + GitHub 업데이트
bash scripts/setup_on_laptop.sh ~/Downloads/hil-review-<버전>.bundle       # 실행 환경 + 테스트
```
테스트만 직접 돌려 보려면:
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,review]"
pytest -q                                          # 테스트 130개 통과 (클라우드 Python 3.10, 3.11, 3.12에서 확인)
hilreview run --config configs/synthetic_base.yaml # 결과: results/synthetic/synthetic_base/report.md
```

## 먼저 읽을 문서 (모두 한국어)
- `START_HERE_KO.md`: 설치 3단계
- `docs/용어집.md`: 어려운 용어 풀이 (먼저 보시는 것을 추천합니다)
- `PROJECT_STATUS.md`: 지금 어디까지 했는지, 다음에 할 일
- `docs/LAPTOP_CHECKS.md`: 노트북에서 할 일 (단계별)
- `docs/PROTOCOL.md`: 실험 계획서 v0.2, 2026-10-20 사용자 승인 (조건, 순서, 지표, 통계, 예비 실험)
- `docs/PROTOCOL_결정_초안.md`: 계획서 빈칸에 대해 승인된 값과 그때 함께 바뀐 것 (기록용)
- `docs/LABEL_DEFINITION.md`: "성공/실패/애매함/무효"의 정확한 뜻
- `docs/RESOURCE_ESTIMATES.md`: 필요한 시간·표본 수 **추정치**. 요점: 비용 대부분은 라벨링이 아니라 로봇 실험입니다.
- `docs/VLM_SERVER_GUIDE.md`: 12GB GPU + 빠듯한 RAM에서 Qwen3-VL을 띄우는 방법 (미검증)
- `DECISIONS.md`(설계 결정과 이유), `LIMITATIONS.md`(한계와 미검증 사항), `docs/RELATED_WORK.md`(관련 연구)
- `docs/COMMIT_MAP.md`: 2026-10-20 작성자 변경(→ gawbi) 전후의 커밋 번호 대응표

## 폴더 구조
```
src/hilreview/
  data/        데이터 형식(에피소드, 키프레임, VLM 제안, 검토 기록), 파일 읽기/쓰기, 데이터 나누기, 합성 데이터 생성
  labeling/    VLM 질문문/응답 해석, 가짜 VLM, 로컬 VLM 서버 연결, 조건별 최종 라벨 만들기
  review/      검토 대상 고르기 규칙, 가짜 검토자, 사람용 검토 도구, 연습 데이터
  evaluation/  지표 계산(신뢰구간, 부트스트랩, 일치도), 가짜 성공 종료 재현, 사람 시간, 실제 로봇 결과
  models/      합성 실험용 대체 분류기(로지스틱 회귀), 임계값 고르기
  experiments/ 설정 파일, 실험 실행기, 리포트, 블라인드 실행 순서표
  robot/       가짜 로봇 환경, 하드웨어 승인 장치, LeRobot 연결 코드(미검증)
configs/       synthetic_base.yaml, synthetic_strategies.yaml, files_template.yaml (실제 데이터용 틀)
scripts/       setup_on_laptop.sh(설치), check_env_laptop.sh / check_env_mac.sh(읽기 전용 점검), sample_size.py(표본 수 계산), power_seeds.py(시드 수준 검정력 추정)
tests/         테스트 130개
results/       합성 데이터 결과만 있음 (각각 출처 기록 provenance.json과 "합성 데이터" 경고 포함)
```

## 언어 안내
문서는 한국어입니다. 코드, 명령어, 파일 이름은 실행에 필요하므로 영어로 둡니다.
예전 영어 문서는 git 기록(커밋 `72b872c`)에 남아 있습니다.
