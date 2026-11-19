# 관련 연구와 신규성 확인 (범위 제한적, 2026-10-17, 2026-10-20 보충)

방법: 웹 검색 몇 번과 LeRobot 문서·0.6.1 소스 읽기만 했습니다. 체계적인 문헌 조사가 **아닙니다**. 따라서 신규성은 **확인되지 않았습니다**.
논문을 쓰기 전에 Google Scholar, arXiv, CoRL/RSS/ICRA 발표 논문을 제대로 찾아보세요.

## 바로 관련된 시스템
- **HIL-SERL**: Luo, Xu, Wu, Levine. *Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning*. arXiv:2410.21845 (Science Robotics 2025에도 게재). 텔레오퍼레이션으로 모은 성공·실패 장면으로 이진 보상 분류기를 학습합니다. 프로젝트 페이지: https://hil-serl.github.io/
- **LeRobot HIL-SERL** (0.6.1 문서: https://huggingface.co/docs/lerobot/en/hilserl). 보상 분류기(`next.reward`, ResNet10 기반, `success_threshold`, `terminate_on_success`)와, 녹화 중 키 입력으로 성공을 기록하는 방식이 들어 있습니다.
- **LeRobot 보상 모델 API (0.6)**: 둘 다 추론 전용입니다.
  - TOPReward: 추가 학습 없음(zero-shot), Qwen3-VL-8B 기반, "True" 토큰의 로그 확률을 씀
  - ROBOMETER: Qwen3-VL-4B 기반, 진행도와 성공 판단 헤드가 있음
  - 문서: https://huggingface.co/docs/lerobot/en/topreward, https://huggingface.co/docs/lerobot/en/robometer

## VLM을 보상 모델이나 성공 판별기로 쓰는 연구
- **TOPReward** (arXiv:2602.19313): 토큰 확률로 추가 학습 없이 진행도를 추정합니다. SO-100/101 과제가 포함된 ManiRewardBench에서 오픈소스 Qwen3-VL-8B의 성공 판별 ROC-AUC가 0.654라고 보고했습니다. 추가 학습 없는 오픈 VLM의 성공 판별은 아직 완벽과 거리가 멀고, 그래서 사람 검토가 필요하다는 근거가 됩니다.
- **Robometer** (arXiv:2603.02115, RSS 2026): 다른 보상 모델(RoboReward)이 엉뚱한 과제에도 최대 보상을 주어 "조기 리셋으로 이어지는" 문제를 지적합니다. 이것이 바로 이 프로젝트가 재는 가짜 성공 종료입니다.
- **RoboReward** (arXiv:2601.00675): VLM 보상 모델과 벤치마크입니다. 사람이 주는 보상으로 한 강화학습과의 격차를 줄였다고 보고합니다.
- **VLMeg** (arXiv:2610.15401): VLM은 보상 값을 직접 맞히는 것보다 성공 판별에 더 적합하다고 주장합니다.
  - 2026-10-20 추가 확인: 실제 Franka 로봇 과제 3개에서 **HIL-SERL**을 강화학습 뼈대로 쓰고, VLM 성공 보상이 사람이 키보드로 주던 이진 보상을 대신해 라벨링 부담을 크게 줄인다고 보고합니다(v1 2026-10-11).
  - **현재 찾은 것 중 이 프로젝트의 C 조건(VLM만)과 가장 가까운 연구**입니다. 차이: VLMeg는 VLM을 학습 중에 직접 판정기로 씁니다. 이 프로젝트는 VLM 라벨로 작은 분류기를 학습하고, 사람 검토 예산(100/25/0%)과 사람 시간 전체를 비교합니다.
- **Large Reward Models** (arXiv:2603.16065): Qwen3-VL-8B를 보상 모델로 미세조정해 실제 로봇 pick-and-place의 강화학습 미세조정에 씁니다.
- RoboReward는 실제 로봇 강화학습에서 사람이 준 보상과의 격차를 줄였다고 보고합니다(위 항목).
- **SuccessVQA** (Du et al. 2023), **VLM-RMs** (Rocamonde et al. 2023, arXiv:2310.12921)

## VLM 라벨 + 선택적 사람 수정 (로봇 분야 밖)
- *Human-Corrected Labels Learning* (arXiv:2511.09063): VLM 라벨을 사람이 선택적으로 고칩니다. 이미지 분류 연구입니다.
- *Leveraging VLMs as Weak Annotators in Active Learning* (arXiv:2605.00480): 세밀한 이미지 분류 연구입니다.
- 일반적인 능동학습과 불확실성 기반 샘플링, 선호도 기반 강화학습의 질문 선택(예: PEBBLE)

## 이 프로젝트가 더할 수 있는 점 (주장이 아니라 가설)
1. HIL-SERL 보상 분류기용 VLM 제안 라벨에 사람 검토 **예산을 어떻게 배분할지**를 다룹니다. 비용은 **사람 시간(초)** 으로 재고, 라벨 정확도만이 아니라 **실제 로봇** 성능으로 평가합니다.
2. **강화학습에 특화된 선택 기준**인 "가짜 성공 우선"을 씁니다. 거짓 양성 프레임 하나가 에피소드를 거짓 보상으로 끝내 버리기 때문에 오류의 비용이 비대칭이라는 점에서 나온 기준입니다.
3. **모든** 사람 시간(검토, audit, 온라인 개입)을 셉니다. 실행은 블라인드이고 블록 무작위 순서를 씁니다.

검색에서 이 세 가지를 함께 다룬 연구는 나오지 않았지만, 이것만으로는 약한 근거입니다. 제대로 된 문헌 조사로 1~3을 다시 확인해야 합니다.

## 논문 가능성에 대한 어시스턴트의 판단 (2026-10-20, 의견이며 보장 아님)
- "VLM이 사람의 보상 라벨을 대신할 수 있나" 자체는 2026년에 활발한 주제라 새롭지 않습니다(VLMeg, RoboReward, Robometer, TOPReward, LRM).
- 차별점이 될 수 있는 것은 다음 세 가지입니다.
  - 사람 검토를 **얼마나, 어디에** 쓸지
  - 검토와 개입을 합친 **사람 시간 전체**의 실측
  - 가짜 성공 중심 분석
- 지금 규모(과제 1개, 로봇 1대, 시드 3개, 새 방법 없음)로는 주요 학회 본 트랙은 어렵다고 봅니다.
- 현실적인 목표는 국내 학회, 해외 학회 워크숍, arXiv 기술 보고서와 코드·데이터 공개입니다.
- 가치를 높일 수 있는 것 (기본 실험이 성공한 뒤):
  1. "어디를 검토할까" 후속 실험에서 같은 사람 시간 기준으로 뚜렷한 차이가 나오는 것
  2. VLM을 직접 온라인 판정기로 쓰는 방식과 비교
  3. 두 번째 과제 추가
