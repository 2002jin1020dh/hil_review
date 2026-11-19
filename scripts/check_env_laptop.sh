#!/usr/bin/env bash
# 우분투 GPU 노트북용 읽기 전용 점검. 다른 GPU 작업이 돌고 있어도 안전합니다:
#   * GPU 메모리를 쓰거나 CUDA 컨텍스트를 만들지 않습니다 (torch는 NVML로만 확인),
#   * 시리얼 포트나 카메라를 열지 않습니다 (장치 파일 목록만 봄),
#   * 아무것도 설치·수정·시작·중지하지 않고, 파일도 쓰지 않습니다.
# 사용법:  bash scripts/check_env_laptop.sh 2>&1 | tee /tmp/hilreview_env_check.txt
# 출력 결과를 붙여 주세요. 비밀 정보는 없지만 컴퓨터 이름과 경로가 들어 있으니 먼저 한번 보세요.
set -u
section() { printf '\n===== %s =====\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

section "운영체제 / CPU / RAM / 디스크"
(lsb_release -ds 2>/dev/null || grep PRETTY_NAME /etc/os-release) ; uname -r
echo "CPU cores: $(nproc)"; free -h | sed -n 1,3p
df -h "$HOME" | tail -1
du -sh "$HOME/.cache/huggingface" 2>/dev/null || echo "~/.cache/huggingface 아직 없음"

section "NVIDIA GPU (읽기 전용 조회: 지금 돌고 있는 작업이 쓰는 양도 보여 줌)"
if have nvidia-smi; then
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version,compute_cap,utilization.gpu --format=csv
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
else
  echo "nvidia-smi 없음"
fi

section "파이썬 / 패키지 관리자"
for t in python3 python3.12 conda uv pip; do have "$t" && echo "$t: $($t --version 2>&1 | head -1)"; done
have conda && conda env list 2>/dev/null

section "현재 파이썬의 torch / LeRobot (CUDA 컨텍스트를 만들지 않음)"
PYTORCH_NVML_BASED_CUDA_CHECK=1 python3 - <<'EOF' 2>&1
import importlib.util, sys
print("python", sys.version.split()[0], sys.executable)
for mod in ("torch", "lerobot", "transformers", "numpy", "PIL", "yaml", "vllm"):
    print(f"{mod:12s}", "installed" if importlib.util.find_spec(mod) else "NOT installed")
if importlib.util.find_spec("torch"):
    import torch
    print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
    print("arch list:", torch.cuda.get_arch_list())  # Blackwell RTX 50xx needs 'sm_120' here
    print("cuda available (NVML check):", torch.cuda.is_available())
if importlib.util.find_spec("lerobot"):
    from importlib.metadata import version
    print("lerobot", version("lerobot"))
    for sub in ("lerobot.rl.gym_manipulator", "lerobot.rewards.classifier.modeling_classifier"):
        print(f"  {sub}:", "present" if importlib.util.find_spec(sub) else "MISSING")
EOF

section "로봇 시리얼 포트와 카메라 (목록만 봄, 아무것도 열지 않음)"
ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || echo "ttyACM/ttyUSB 장치 없음 (로봇이 연결되지 않음)"
ls -l /dev/video* 2>/dev/null || echo "/dev/video* 장치 없음"
groups | tr ' ' '\n' | grep -E '^(dialout|video)$' || echo "dialout/video 그룹에 속해 있지 않음 (나중에 필요할 수 있음)"

section "로컬 VLM 서버 (읽기 전용 조회, 내 컴퓨터만)"
if have curl; then
  for port in 8000 8080 11434; do
    printf "port %s: " "$port"; curl -s -m 2 "http://127.0.0.1:$port/v1/models" | head -c 300 || true; echo
  done
fi
echo; echo "완료. 아무것도 바꾸지 않았습니다."
