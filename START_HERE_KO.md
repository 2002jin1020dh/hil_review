# 시작하기 (3단계)

## 1. 파일 2개 받기
세션 결과물에서 아래 두 파일을 받아 노트북의 `다운로드(Downloads)` 폴더에 저장하세요.
- `hil-review.bundle`: 프로젝트 전체와 변경 이력
- `setup_on_laptop.sh`: 설치 스크립트

## 2. 명령어 하나 실행하기 (노트북 터미널)
```bash
bash ~/Downloads/setup_on_laptop.sh ~/Downloads/hil-review.bundle
```
- `~/research/hil-review` 폴더가 새로 생기고, 테스트까지 자동으로 실행됩니다.
- 마지막에 **`130 passed`** 가 보이면 성공입니다 (버전에 따라 숫자가 다를 수 있음).
- 다른 파이썬 환경(LeRobot 등)과 GPU는 건드리지 않습니다.
- `ensurepip is not available`라는 오류가 나오면, 아래 명령을 한 번만 실행하고 **같은 명령을 다시** 실행하세요. 폴더를 지울 필요는 없습니다.
  ```bash
  sudo apt install python3-venv
  ```

**GitHub(gawbi)에도 올리려면** 위 명령 끝에 `--push`를 붙이세요. 비공개(private) 저장소로 올라갑니다.
먼저 GitHub CLI 설치와 로그인을 한 번 해 두셔야 합니다.
```bash
sudo apt install gh
gh auth login
```
`gh auth login`을 실행하면 브라우저로 로그인하게 됩니다. 그다음 같은 명령을 다시 실행하세요.

## 3. 점검 결과를 저에게 보내기
```bash
bash ~/research/hil-review/scripts/check_env_laptop.sh 2>&1 | tee /tmp/hilreview_env_check.txt
```
화면에 나온 내용을 복사해서 붙여 주세요. 읽기만 하는 점검이라, 지금 돌아가는 GPU 작업에 영향을 주지 않습니다.

---
**참고:** 지금까지의 결과는 전부 **가짜(합성) 데이터**로 "프로그램이 제대로 동작하는지"만 확인한 것입니다.
실제 로봇 실험 결과는 아직 없습니다. 자세한 내용은 `PROJECT_STATUS.md`에 있습니다.
