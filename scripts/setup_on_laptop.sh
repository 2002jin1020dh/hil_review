#!/usr/bin/env bash
# 노트북에서 명령 한 줄로 설치하기 (우분투 또는 macOS). 사용자가 직접 실행합니다. 어시스턴트는 사용자 컴퓨터에 접근할 수 없습니다.
#
#   bash setup_on_laptop.sh                                        # ~/Downloads/hil-review.bundle 사용
#   bash setup_on_laptop.sh ~/Downloads/hil-review.bundle --push   # 위 작업 + GitHub 비공개 저장소 gawbi/hil-review에 업로드
#
# 하는 일 (이것 외에는 아무것도 하지 않음):
#   1. bundle에서 ~/research/hil-review를 만듭니다 (git 기록 포함). 이전에 이 스크립트로 만든 폴더면
#      그대로 쓰고 새 bundle로 업데이트합니다. 지우거나 덮어쓰는 것은 없습니다.
#   2. 그 폴더 안에 독립된 파이썬 환경(.venv)을 만듭니다. LeRobot/conda 환경과 GPU는 건드리지 않습니다.
#   3. 그 .venv 안에만 작은 패키지 4개(numpy, pyyaml, pytest, pillow)를 설치하고 CPU로 테스트를 돌립니다.
#   4. --push를 줄 때만 GitHub 비공개 저장소를 만들고 업로드합니다 (GitHub CLI 로그인 필요: `gh auth login`).
# 환경 변수로 바꿀 수 있는 것: HILREVIEW_DEST(설치 폴더), GH_REPO(기본 gawbi/hil-review), PYTHON.
set -euo pipefail

# Self-update guard: step 1 may `git pull` a new version of THIS file while bash is still reading it, which can make
# bash execute garbage. So always run from a private temporary copy of the script.
if [ -z "${HILREVIEW_SETUP_COPY:-}" ]; then
  _copy="$(mktemp "${TMPDIR:-/tmp}/hilreview_setup.XXXXXX")"
  cp "$0" "$_copy"
  HILREVIEW_SETUP_COPY="$_copy" exec bash "$_copy" "$@"
fi
trap 'rm -f "$HILREVIEW_SETUP_COPY"' EXIT

BUNDLE="${1:-$HOME/Downloads/hil-review.bundle}"
PUSH="${2:-}"
DEST="${HILREVIEW_DEST:-$HOME/research/hil-review}"
GH_REPO="${GH_REPO:-gawbi/hil-review}"
PY="${PYTHON:-$(command -v python3 || true)}"
say() { printf '\n==> %s\n' "$*"; }
is_our_checkout() { [ -d "$1/.git" ] && grep -qs 'name = "hilreview"' "$1/pyproject.toml"; }

[ -f "$BUNDLE" ] || { echo "bundle 파일이 없습니다: $BUNDLE  (먼저 hil-review.bundle을 내려받으세요)"; exit 1; }
command -v git >/dev/null || { echo "git이 필요합니다 (우분투: sudo apt install git)"; exit 1; }
[ -n "$PY" ] || { echo "python3가 필요합니다"; exit 1; }
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || { echo "Python 3.10 이상이 필요합니다"; exit 1; }

if [ -e "$DEST" ] && ! is_our_checkout "$DEST"; then
  echo "$DEST 폴더가 이미 있는데 hil-review 프로젝트가 아닙니다. 건드리지 않습니다. HILREVIEW_DEST=/다른/경로 로 지정하세요"; exit 1
fi

if is_our_checkout "$DEST"; then
  say "1/4 기존 프로젝트 폴더를 그대로 사용: $DEST (bundle로 업데이트)"
  cd "$DEST"
  [ "$(git branch --show-current)" = "main" ] || git branch -M main
  BR="$(git ls-remote --heads "$BUNDLE" | awk '{print $2}' | sed 's#refs/heads/##' | grep -E '^(main|master)$' | head -n1 || true)"
  if ! git pull -q --ff-only "$BUNDLE" "${BR:-main}"; then
    echo "업데이트하지 못했습니다 (폴더에서 직접 고친 내용이 있나요? 또는 저장소 기록이 다시 쓰였나요? 예: 2026-10-20 작성자 변경 → PROJECT_STATUS.md의 '작성자 변경 후 노트북·GitHub 맞추기'). 아무것도 바꾸지 않았습니다."; exit 1
  fi
else
  say "1/4 프로젝트 복원: $DEST"
  mkdir -p "$(dirname "$DEST")"
  git clone -q "$BUNDLE" "$DEST"
  cd "$DEST"
  git remote remove origin             # it pointed at the bundle file
  [ "$(git branch --show-current)" = "main" ] || git branch -M main
fi
git log --oneline -1                    # (no '| head': under pipefail a SIGPIPE would abort the script)
[ -f START_HERE_KO.md ] || echo "참고: 이 bundle은 옛날 버전입니다. hil-review.bundle을 다시 받아 이 명령을 한 번 더 실행하면 업데이트됩니다."

say "2/4 독립된 파이썬 환경 만들기: $DEST/.venv"
INSTALLER=""
if [ -x .venv/bin/python ] && .venv/bin/python -c 'import sys, pytest, numpy, yaml; assert sys.prefix != sys.base_prefix' 2>/dev/null; then
  echo "기존 .venv를 그대로 씁니다"
  # a .venv made by `uv venv` has no pip: then install with uv (or rebuild if uv is gone)
  if .venv/bin/python -m pip --version >/dev/null 2>&1; then INSTALLER=pip
  elif command -v uv >/dev/null; then INSTALLER=uv
  else echo "기존 .venv에 pip가 없어서 새로 만듭니다"
  fi
fi
if [ -z "$INSTALLER" ]; then
  rm -rf .venv                          # only the (broken/partial) environment this script creates inside the project
  if "$PY" -m venv .venv 2>/tmp/hilreview_venv_err.txt; then
    INSTALLER=pip
  elif command -v uv >/dev/null; then
    rm -rf .venv
    echo "python -m venv를 쓸 수 없어 uv로 대신 만듭니다 (시스템 패키지 설치 불필요)"
    uv venv -q --python "$("$PY" -c 'import sys; print(sys.executable)')" .venv
    INSTALLER=uv
  else
    rm -rf .venv
    PKG="$("$PY" -c 'import sys; print(f"python3.{sys.version_info.minor}-venv")')"
    echo "이 파이썬은 아직 가상환경을 만들 수 없습니다 (우분투는 이 부분을 따로 설치해야 합니다)."
    echo "한 번만 설치하세요 (비밀번호 필요):"
    echo "    sudo apt install $PKG"
    echo "그다음 **같은 명령**을 다시 실행하세요 (아무것도 지울 필요 없음)."
    exit 1
  fi
fi
if [ "$INSTALLER" = uv ]; then
  uv pip install -q --python .venv/bin/python -e ".[dev,review]"
else
  .venv/bin/python -m pip install -q --upgrade pip
  .venv/bin/python -m pip install -q -e ".[dev,review]"
fi

say "3/4 테스트 실행 (CPU만 사용, 약 10초)"
.venv/bin/python -m pytest -q

say "4/4 GitHub"
if [ "$PUSH" = "--push" ]; then
  if git remote get-url origin >/dev/null 2>&1; then
    git push -q -u origin main && echo "업로드 완료: $(git remote get-url origin)"
  elif command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
    gh repo create "$GH_REPO" --private --source=. --remote=origin --push
    echo "업로드 완료: https://github.com/$GH_REPO  (비공개)"
  else
    echo "GitHub CLI가 없거나 로그인되어 있지 않습니다. 두 가지 방법이 있습니다:"
    echo "  A) GitHub CLI 설치(우분투: sudo apt install gh) 후 'gh auth login'을 한 번 실행하고, 이 스크립트를 다시 실행"
    echo "  B) github.com에서 '${GH_REPO#*/}'라는 이름의 빈 비공개 저장소를 만든 뒤(README 없이):"
    echo "       cd $DEST && git remote add origin https://github.com/$GH_REPO.git && git push -u origin main"
  fi
else
  echo "건너뜀. (GitHub에 올리려면 --push를 붙여 실행하세요.)"
fi

say "완료. 프로젝트 폴더: $DEST"
echo "다음 단계 (읽기 전용, GPU 작업에 영향 없음):"
echo "  bash $DEST/scripts/check_env_laptop.sh 2>&1 | tee /tmp/hilreview_env_check.txt"
