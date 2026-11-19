# VLM 서버 실행 가이드 (12GB GPU + 빠듯한 RAM용)

> **상태: 미검증.** 공식 문서와 Hugging Face 파일 정보를 바탕으로 썼지만, 클라우드에는 GPU가 없어서 **실제로 실행해 보지 않았습니다**.
> 아래 명령은 사용자 노트북에서 처음 실행하는 것입니다. 설치와 GPU 사용은 **[사용자 결정 필요]** 입니다.
> 서버가 제대로 동작하는지는 마지막의 `hilreview vlm-check`로 직접 확인하세요.

## 0. 사용자 노트북 사실 (점검 결과, 2026-10-17)
- GPU: RTX 5070 Ti Laptop **12GB**, Blackwell(`sm_120`), 드라이버 580. 점검 당시 약 10GB 비어 있었음.
- RAM: 14GiB 중 여유 **3.3GiB**. Chrome 탭 여러 개와 VS Code가 각각 약 0.5GB, 다른 강화학습 실험이 약 2.1GB와 CPU 코어 하나를 쓰고 있었음.
- uv가 있어서 sudo 없이 별도 파이썬 환경을 만들 수 있음.

## 1. 어떤 모델로 시작하나 (Hugging Face 파일 크기, 2026-10-17 확인)
| 모델 | 파일 크기 | 판단 |
|---|---|---|
| `Qwen/Qwen3-VL-4B-Instruct-FP8` | 6.02 GB | **여기서 시작 (추천)** |
| `Qwen/Qwen3-VL-4B-Instruct` (BF16) | 8.88 GB | 다른 작업과 같이 쓰기엔 빠듯함 |
| `Qwen/Qwen3-VL-2B-Instruct` (BF16) | 4.26 GB | FP8이 안 될 때의 대안 |
| `Qwen/Qwen3-VL-8B-Instruct-FP8` | 10.59 GB | 12GB GPU에는 너무 큼 |
| GGUF 4B Q8_0 + 비전 모듈 Q8_0 | 4.28 + 0.45 GB | llama.cpp용 (방법 C) |
| GGUF 8B Q4_K_M + 비전 모듈 Q8_0 | 5.03 + 0.75 GB | 4B 정확도가 부족할 때 나중에 (방법 C) |

**Instruct 모델**을 쓰세요(Thinking 모델 아님). 이 프로젝트는 yes/no 한 단어 답만 읽습니다.

## 2. 시작 전 확인 (읽기 전용)
```bash
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
free -h
```
- GPU 여유가 **9GB 이상**인지 봅니다.
- RAM 여유가 적으면 Chrome과 VS Code를 닫습니다. 서버를 띄우는 동안 RAM이 많이 듭니다.
- 다른 강화학습 실험(crrl)과 GPU를 같이 쓸지는 사용자가 정합니다. 그 실험이 끝난 뒤에 하는 것이 가장 안전합니다.

## 3. 방법 A: vLLM (추천) **[사용자 결정 필요: 설치 + GPU 사용]**
Qwen3-VL은 vLLM 0.11.0 이상에서 지원됩니다. vLLM은 자기 torch 버전을 함께 설치하므로, LeRobot 환경과 **반드시 분리**해서 설치하세요. 같이 설치하면 LeRobot이 깨질 수 있습니다.
```bash
uv venv ~/venvs/vlm --python 3.12
uv pip install --python ~/venvs/vlm/bin/python -U vllm
```
서버 실행 (메모리를 아끼는 설정):
```bash
OMP_NUM_THREADS=1 ~/venvs/vlm/bin/vllm serve Qwen/Qwen3-VL-4B-Instruct-FP8 \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 4096 --max-num-seqs 2 \
  --gpu-memory-utilization 0.75 \
  --limit-mm-per-prompt.video 0 --limit-mm-per-prompt.image 2 \
  --mm-processor-cache-gb 1 \
  --enforce-eager
```
| 옵션 | 뜻 |
|---|---|
| `--host 127.0.0.1` | 이 노트북 안에서만 접속 가능 (외부 노출 안 함) |
| `--max-model-len 4096`, `--max-num-seqs 2` | 입력 길이와 동시 요청 수를 줄여 GPU 메모리 절약 (우리는 짧은 질문 + 이미지 2장) |
| `--gpu-memory-utilization 0.75` | GPU 전체(12,227MiB)의 75%인 약 **9,170MiB(≈9.0GiB)** 까지만 사용. vLLM은 이 상한까지 미리 잡으므로 실제 사용량도 거의 이 값. 다른 작업 몫을 남김 |
| `--limit-mm-per-prompt.video 0 / .image 2` | 동영상은 받지 않고 이미지는 최대 2장(카메라 2대). 동영상용 메모리 예약을 없앰 |
| `--mm-processor-cache-gb 1` | 이미지 전처리 캐시를 기본 4GiB에서 1GiB로 줄임 (**RAM 절약**) |
| `--enforce-eager` | CUDA graph를 꺼서 GPU 메모리 절약 (속도는 조금 느려짐) |
| `OMP_NUM_THREADS=1` | 전처리 CPU 스레드를 줄여 다른 작업과 CPU 경쟁을 줄임 |

- 처음 실행할 때 모델 약 6GB를 내려받습니다(`~/.cache/huggingface`, 디스크 여유 충분).
- 끄기: 서버 터미널에서 **Ctrl+C**. 라벨링이 끝나면 꼭 끄세요. GPU 메모리가 반환됩니다. 오래 켜 두면 GPU 메모리가 계속 늘어난다는 보고도 있습니다.

문제가 생기면:
| 증상 | 할 일 |
|---|---|
| `CUDA out of memory` | `--gpu-memory-utilization 0.7`, `--max-model-len 2048`으로 줄이고, `nvidia-smi`로 다른 작업 확인 |
| RAM 부족, 스왑으로 매우 느려짐 | Chrome·VS Code 닫기, `--mm-processor-cache-gb`를 더 작게 |
| FP8 관련 오류 (이 GPU에서 FP8 커널이 안 됨) | 먼저 `Qwen/Qwen3-VL-2B-Instruct`로 동작 확인 |
| 설치는 됐는데 GPU를 못 쓰거나 Blackwell 관련 오류 | vLLM 문서의 CUDA 13 빌드 안내 확인, 또는 방법 B/C |

## 4. 방법 B: Ollama (설치가 가장 쉬움, 확신 점수 지원은 불확실) **[사용자 결정 필요: sudo 설치]**
- 설치 스크립트가 시스템 서비스를 설치합니다(sudo 필요). 설치 후 `ollama pull qwen3-vl:4b`를 실행합니다.
- 주소: `http://127.0.0.1:11434/v1`, 모델 이름: `qwen3-vl:4b`
- **주의: 확신 점수(logprobs) 지원에 대한 자료가 서로 다릅니다.** "0.12.11부터 OpenAI 호환 API에서도 지원"이라는 글이 있고, "OpenAI 호환 API가 logprobs 요청을 조용히 버린다"는 최근 이슈도 있습니다. 반드시 6절의 `vlm-check`로 확인하세요. 확신 점수가 없으면 A/B/C 조건과 무작위 정책만 쓸 수 있습니다.

## 5. 방법 C: llama.cpp (RAM 효율적, 설치 난이도 높음)
- Qwen 공식 GGUF 페이지 방식입니다: `llama-server -m <모델>.gguf --mmproj <비전모듈>.gguf`, 접속은 `/v1/chat/completions`.
  ```bash
  llama-server -m Qwen3VL-4B-Instruct-Q8_0.gguf --mmproj mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf \
    --host 127.0.0.1 --port 8080 -ngl 99 --ctx-size 4096
  ```
- GPU로 돌리려면 CUDA로 직접 빌드하거나 Docker 이미지가 필요해서 초보자에게는 어렵습니다.
- 알려진 위험:
  - Qwen3-VL을 llama-server에서 연속으로 요청하면 두 번째 요청이 깨진다는 버그 보고가 있습니다. `vlm-check`가 요청을 3번 연속 보내므로 이 문제를 잡아냅니다.
  - OpenAI 호환 채팅 API에서 logprobs를 주는지 문서상 분명하지 않습니다. 이것도 `vlm-check`로 확인합니다.

## 6. 서버 확인 (모든 방법 공통, 읽기 전용에 가까운 요청 3번)
```bash
cd ~/research/hil-review && . .venv/bin/activate
hilreview vlm-check --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-VL-4B-Instruct-FP8
# Ollama라면:   --base-url http://127.0.0.1:11434/v1 --model qwen3-vl:4b
# llama.cpp라면: --base-url http://127.0.0.1:8080/v1 --model <서버가 알려주는 이름>
```
| 출력 | 의미 |
|---|---|
| `[1/3] 서버 응답 OK` | 서버가 떠 있음. 모델 이름 목록도 보여 줌 |
| `[2/3] 응답 해석: 3/3 성공` | yes/no 답을 제대로 읽었음 (3번 연속 요청도 안정적) |
| `[3/3] 확신 점수(logprobs): 있음` | 모든 검토 정책 사용 가능. "없음"이면 무작위/A/B/C만 가능 |

그림은 만화 같은 연습용이라 여기서 맞히는지는 실제 성능과 **무관**합니다. 실제 정확도는 예비 실험 P2의 실제 데이터로 잽니다.

## 7. 예비 실험 P1에서 적어 둘 것 (`results/real/P1_vlm_server.md`에 기록 추천)
| 항목 | 값 |
|---|---|
| 서버 종류와 버전 (vLLM/Ollama/llama.cpp) | |
| 모델과 양자화 (예: Qwen3-VL-4B-Instruct-FP8) | |
| 서버 실행 중 GPU 메모리 (`nvidia-smi`) | |
| 서버 실행 중 RAM 여유 (`free -h`) | |
| 키프레임당 시간 (`vlm-check` 출력의 초) | |
| 확신 점수(logprobs) 있음/없음 | |
| 3번 연속 요청 안정성 | |
| 같이 돌고 있던 다른 작업 | |

## 8. 안전·자원 규칙
- 서버는 `127.0.0.1`에만 엽니다. 클라우드 VLM은 쓰지 않습니다. `--allow-remote-host`는 쓰지 마세요.
- 다른 GPU 작업과 겹치면 시작 전에 GPU 여유를 확인합니다.
- 라벨링이 끝나면 서버를 끕니다(Ctrl+C).

## 출처 (직접 확인용)
- vLLM Qwen3-VL 사용 안내: https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen3-VL.html
- vLLM 메모리 절약 안내: https://docs.vllm.ai/en/latest/configuration/conserving_memory/
- Qwen 공식 GGUF (llama.cpp 명령 예시): https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF
- llama.cpp 서버 문서: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- llama.cpp Qwen3-VL 두 번째 요청 버그 보고: https://github.com/ggml-org/llama.cpp/issues/17200
- vLLM Qwen3-VL GPU 메모리 증가 보고: https://github.com/vllm-project/vllm/issues/28230
- Ollama OpenAI 호환 문서: https://docs.ollama.com/api/openai-compatibility
- Ollama logprobs 관련 이슈: https://github.com/ollama/ollama/issues/16117
