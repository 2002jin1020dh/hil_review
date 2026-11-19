#!/usr/bin/env bash
# 맥북용 읽기 전용 점검 (계획, 검토 세션, 분석용). 아무것도 바꾸지 않습니다.
# 사용법:  bash scripts/check_env_mac.sh 2>&1 | tee /tmp/hilreview_mac_check.txt
set -u
section() { printf '\n===== %s =====\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }
section "macOS / 하드웨어"
sw_vers 2>/dev/null
system_profiler SPHardwareDataType 2>/dev/null | grep -E "Model Name|Chip|Memory" || true
df -h "$HOME" | tail -1
section "파이썬 / 도구"
for t in python3 python3.12 uv conda git; do have "$t" && echo "$t: $($t --version 2>&1 | head -1)"; done
python3 - <<'EOF' 2>&1
import importlib.util
for mod in ("numpy", "yaml", "PIL", "pytest", "torch", "lerobot"):
    print(f"{mod:8s}", "installed" if importlib.util.find_spec(mod) else "NOT installed")
EOF
echo; echo "완료. 아무것도 바꾸지 않았습니다."
