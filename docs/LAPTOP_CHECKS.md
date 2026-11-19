# 노트북·맥에서 할 일 (클라우드는 사용자 컴퓨터에 접근할 수 없습니다)

어시스턴트는 사용자 컴퓨터에서 아무것도 실행하지 않았습니다. 따로 적혀 있지 않은 명령은 모두 **읽기 전용**입니다.
설치를 하거나 GPU를 쓰는 단계에는 **[사용자 결정 필요]** 표시를 해 두었습니다.

## 지금까지 완료된 것
- 0단계: 코드 설치 + GitHub 업로드. 노트북 `~/research/hil-review`, GitHub 비공개 저장소 `gawbi/hil-review` (커밋 `4a9072d`)
- 1단계: 기본 점검 (`check_env_laptop.sh`). 결과는 `LIMITATIONS.md` 3a절에 기록했습니다.
- 1-2단계: `lerobot` 환경 확인. Python 3.12.14, torch 2.11.0+cu130, `sm_120` 지원 확인, LeRobot 0.6.2(GitHub 개발 버전) → `LIMITATIONS.md` 3a절
- 2단계: 노트북에서 테스트 109개 통과 (커밋 `cb5306b`)
- LeRobot 커밋 확인 (2026-10-20): `8c920c42`, 직접 고친 것 없음 → 이 커밋으로 고정 (`DECISIONS.md` D-33)

## 0단계 (참고용, 처음 설치할 때): 명령 한 줄
결과물에서 `hil-review.bundle`과 `setup_on_laptop.sh`를 받은 뒤 **노트북에서** 실행합니다.
```bash
bash ~/Downloads/setup_on_laptop.sh ~/Downloads/hil-review.bundle            # 폴더 + 실행 환경 + 테스트
bash ~/Downloads/setup_on_laptop.sh ~/Downloads/hil-review.bundle --push     # 위 작업 + GitHub 비공개 업로드
```
- `~/research/hil-review` 폴더를 git 기록과 함께 만들고, 그 안에 `.venv`(독립된 파이썬 환경)를 만듭니다.
- 다시 실행해도 안전합니다. 이전에 이 스크립트로 만든 폴더면 이어서 쓰고 새 bundle로 업데이트합니다. 다른 폴더는 건드리지 않습니다.
- 다른 파이썬 환경과 GPU는 건드리지 않습니다. GitHub 업로드는 `--push`를 줄 때만 하며, `gh auth login`이 한 번 필요합니다.
- `ensurepip is not available` 오류가 나면 `sudo apt install python3-venv`를 한 번 실행한 뒤 같은 명령을 다시 실행하세요. `uv`가 설치되어 있으면 스크립트가 자동으로 uv를 씁니다. 사용자 노트북에는 uv가 있으므로 sudo가 필요 없습니다.
- 어시스턴트가 대신 업로드할 수 없었던 이유: 클라우드에는 GitHub 로그인이 없고, 토큰을 채팅에 붙이면 유출 위험이 있습니다.

새 버전을 받을 때는 버전 이름이 붙은 bundle 파일을 받아 다음을 실행합니다.
```bash
cd ~/research/hil-review
git pull --ff-only ~/Downloads/hil-review-<버전>.bundle main && git push
```

## 1단계: 기본 점검 (완료, 다시 해도 안전함)
```bash
bash ~/research/hil-review/scripts/check_env_laptop.sh 2>&1 | tee /tmp/hilreview_env_check.txt   # 우분투 GPU 노트북
bash scripts/check_env_mac.sh 2>&1 | tee /tmp/hilreview_mac_check.txt                            # 맥북
```
GPU는 `nvidia-smi`로 정보만 읽고, **GPU 메모리를 쓰지 않습니다**(CUDA 컨텍스트를 만들지 않음). 장치 파일은 목록만 보고 열지 않습니다.

## 1-2단계: `lerobot` 환경과 메모리 확인 (지금 할 일, 읽기 전용)
기본 점검은 시스템 파이썬만 봤기 때문에, LeRobot이 설치된 conda 환경 `lerobot`은 따로 봐야 합니다.
```bash
conda run -n lerobot env PYTORCH_NVML_BASED_CUDA_CHECK=1 python -c "import sys, torch, importlib.metadata as m; print('python', sys.version.split()[0]); print('torch', torch.__version__, 'cuda', torch.version.cuda); print('arch', torch.cuda.get_arch_list()); print('gpu ok', torch.cuda.is_available()); print('lerobot', m.version('lerobot'))"
ps aux --sort=-%mem | head -6
```
결과를 이렇게 읽습니다.

| 확인 항목 | 왜 중요한가 | 좋은 결과 |
|---|---|---|
| `arch` 목록 | RTX 50 시리즈(Blackwell)는 `sm_120` 지원이 필요 | 목록에 `'sm_120'`이 있음. 없으면 **바로 재설치하지 말고** 결과를 먼저 알려 주세요. |
| `cuda` | Blackwell은 CUDA 12.8 이상으로 빌드된 torch가 필요 | `12.8` 이상 |
| `python` | LeRobot 0.6.x는 Python 3.12 이상 필요 | `3.12` 이상 |
| `lerobot` | 연결 코드는 0.6.1 기준으로 작성됨 | `0.6.1`. 다르면 버전을 알려 주세요. |
| `ps` 결과 | RAM 여유가 3.3GiB뿐이었음 | 메모리를 많이 쓰는 프로그램 확인 (VLM 실행 시점을 정하는 데 필요) |

## 2단계: 최신 버전으로 업데이트 → 실행 환경 만들기 + 테스트 (CPU만 사용, 약 10초)
**순서가 중요합니다. 먼저 업데이트하고, 그다음 스크립트를 실행하세요.** 노트북에 있는 옛 스크립트(`4a9072d`)는 실행 도중 자기 자신이 업데이트되면 오작동할 수 있습니다. 새 스크립트에는 이를 막는 장치가 있습니다.
```bash
cd ~/research/hil-review
git pull --ff-only ~/Downloads/hil-review-<버전>.bundle main && git push
bash scripts/setup_on_laptop.sh ~/Downloads/hil-review-<버전>.bundle
```
마지막에 `109 passed`가 나오면 성공입니다. 노트북에 uv가 있으므로 sudo는 필요 없습니다. 실패하면 화면을 붙여 주세요. 클라우드에서 통과했다고 해서 사용자 노트북에서도 된다는 보장은 없습니다.

## 3단계: 검토 도구 연습 (맥이나 노트북, 로봇·GPU 불필요)
```bash
cd ~/research/hil-review && . .venv/bin/activate
hilreview make-demo-data --out /tmp/demo
hilreview review --episodes /tmp/demo/episodes.jsonl --keyframes /tmp/demo/keyframes.jsonl \
  --split-file /tmp/demo/splits.json --purpose pool_review --vlm /tmp/demo/vlm_suggestions.jsonl \
  --out /tmp/demo/pool_reviews.jsonl --reviewer-id <내이름> --viewer xdg-open   # 맥은 --viewer open
```
연습 데이터는 **합성(가짜)** 입니다. 여기서 잰 시간은 연습용이고 연구 데이터가 아닙니다.

## 4단계: 로컬 VLM 서버 (예비 실험 P1) **[사용자 결정 필요: 설치 + GPU 사용]**
**자세한 방법은 `docs/VLM_SERVER_GUIDE.md`** 에 있습니다. 서버를 띄운 뒤 `hilreview vlm-check`로 연결, 응답, 확신 점수, 속도를 한 번에 확인하세요.
GPU가 비어 있거나 다른 작업과 나눠 써도 될 때만 합니다. RAM이 빠듯하므로 메모리를 많이 쓰는 다른 프로그램은 끄고 하세요.
OpenAI 호환 서버(예: vLLM, 또는 비전 모듈을 포함한 llama.cpp)로 Qwen3-VL **Instruct** 모델을 띄우고, **logprobs**(확신 점수)를 돌려주게 설정합니다. 그다음:
```bash
curl -s http://127.0.0.1:8000/v1/models                    # 읽기 전용: 서버가 떠 있는지, 어떤 모델인지
hilreview vlm-label --episodes /tmp/demo/episodes.jsonl --keyframes /tmp/demo/keyframes.jsonl \
  --out /tmp/demo/vlm_real.jsonl --task "put the red cube into the blue container" --limit 20
```
기록할 것: 모델 이름과 양자화 방식, 실행 중 VRAM 사용량(`nvidia-smi`), 키프레임당 걸린 시간(출력 파일의 `latency_s`), `p_success`가 채워지는지(logprobs 지원 여부).
연결 코드는 내 컴퓨터나 집 네트워크가 아닌 주소로는 이미지를 보내지 않습니다(`--allow-remote-host`를 줄 때만 예외).

## 5단계: LeRobot 연결 코드 (미검증 코드) **[사용자 결정 필요: LeRobot 환경 사용]**
작은 데이터셋 사본(에피소드 2개)으로 먼저 시험한 뒤에 믿으세요.
1. `hilreview export-keyframes`: PNG 몇 장을 열어 카메라가 맞는지, 방향이 맞는지, 잘림 영역이 맞는지 확인합니다.
2. `hilreview build-reward-dataset`: 결과를 `LeRobotDataset`으로 열어 `next.reward`가 라벨 파일과 같은지 확인합니다.
3. 같은 프레임에 `hilreview score --input-mode runtime`과 `--input-mode train`을 실행해 **점수를 비교**합니다. 많이 다르면 학습 때와 HIL-SERL 실행 때 이미지 정규화가 다르다는 뜻입니다(코드를 읽어서 찾은 의심 사항, `PROJECT_STATUS.md` 참고). 강화학습을 돌리기 전에 꼭 알려 주세요.

## 6단계: 로봇 (예비 실험 P2/P3) **[매번 사용자의 명시적 허락 필요]**
로봇 실행 중에는 다른 터미널에서 `hilreview log-episodes --run-id <실행 ID> --phase train --out online_episodes.jsonl --runs-out runs.jsonl --fps 10`을 켜 두고, 에피소드가 끝날 때마다 글자 2개를 입력합니다(예: `cy` 맞게 끝남, `cn` 가짜 성공 종료).
연습: `--data-kind synthetic`을 붙이면 연습 기록으로 저장됩니다.
로봇 프로그램은 사용자만 시작하고 멈춥니다. 움직이기 전에 확인할 것:
- 작업 공간이 비어 있음
- 비상 정지 버튼에 손이 닿음
- `lerobot-find-joint-limits`로 구한 끝단(EE) 이동 범위가 설정됨
- 로봇 팔이 닿는 범위에 사람이 없음

## 나중에 어시스턴트를 컴퓨터에 연결하고 싶다면 (선택)
지금은 접근 권한이 없습니다. 가장 안전한 방법은 지금처럼 사용자가 명령을 실행하고 결과를 붙여 주는 것입니다.
나중에 도구를 연결한다면(예: 제한된 SSH 계정), 다음 조건을 지키세요.
- sudo 권한이 없는 별도 사용자로 만들기
- 다른 프로젝트나 GPU 작업에 접근할 수 없게 하기
- 로봇 제어 권한은 주지 않기
