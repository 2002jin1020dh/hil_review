# 커밋 번호 대응표 (작성자 변경, 2026-10-20)

2026-10-20에 사용자 요청으로 **모든 커밋의 작성자(author)와 커미터(committer)를 `gawbi`로 바꿨습니다.** 이메일은 사용자의 GitHub 인증(verified) 이메일이고, 문서에는 적지 않습니다. 그래서 커밋 번호(해시)가 모두 바뀌었습니다(`DECISIONS.md` D-34).
- 파일 내용, 날짜, 커밋 메시지는 그대로입니다. 36개 커밋 모두 파일 트리 해시가 바뀌기 전과 같은 것을 클라우드에서 확인했습니다.
- 문서와 결과 파일(`results/*/provenance.json`, `report.md`)에 적힌 커밋 번호는 **바뀌기 전 번호** 그대로 두었습니다. 당시 기록을 그대로 남기기 위해서입니다. 이 표로 새 번호를 찾으세요.
- 바뀌기 전 작성자 표시는 `Claude` 29개, `hil-review assistant` 5개, `gawbi` 2개(마지막 두 개)였습니다. 모두 AI 어시스턴트가 클라우드에서 만든 커밋이고, 사용자가 노트북에서 받아 GitHub에 올렸습니다.

| 옛 번호 | 새 번호 | 날짜 | 내용 |
|---|---|---|---|
| `80335ac` | `435ae80` | 2026-10-17 | Core pipeline: schema, splits, policies, VLM adapters, synthetic data, runner, report, tes |
| `b784bdd` | `f346473` | 2026-10-17 | Online metrics, blinded schedule, review tool, LeRobot adapters (unverified), CLI, demo da |
| `294ad0f` | `2fc3ecc` | 2026-10-17 | fake_success_priority v2 (sum of 1-p); git dirty flag ignores results/; keep v1 negative r |
| `b9b729d` | `5eb57a0` | 2026-10-17 | Code review fixes: paired bootstrap, CI reporting, review-placement checks, confidence gua |
| `8b784cd` | `3f69118` | 2026-10-17 | Regenerate synthetic results with CI/paired reporting |
| `046ed79` | `ab28407` | 2026-10-17 | Docs (protocol, labels, laptop checks, estimates, related work, README), files template, f |
| `8678cdd` | `4198bdd` | 2026-10-17 | Status docs updated (milestones M0-M5, open issues), DECISIONS D-17..D-23, LIMITATIONS sec |
| `47c5a22` | `e984e91` | 2026-10-17 | PROJECT_STATUS: correct commit references |
| `d8afb0d` | `6505932` | 2026-10-17 | Add one-command laptop setup script; branch main; pytest prints summary |
| `1c24c76` | `4e35c0e` | 2026-10-17 | setup_on_laptop.sh: avoid SIGPIPE abort under pipefail |
| `901066c` | `b89ca6b` | 2026-10-17 | Docs: one-command laptop setup; Python 3.10 verified; status M6 |
| `f3f2877` | `764a19b` | 2026-10-17 | Add Korean 3-step start guide |
| `142db3b` | `dc908e8` | 2026-10-17 | setup_on_laptop.sh: resume without deleting, fast-forward from newer bundle, master->main, |
| `b999931` | `0325865` | 2026-10-17 | setup_on_laptop.sh: reuse .venv only if it is a real venv with the packages |
| `4a9072d` | `d1a6004` | 2026-10-17 | Guides: venv package hint, re-run semantics, gh login steps |
| `759fbca` | `51def44` | 2026-10-17 | Status: M7 code on laptop + private GitHub repo (user-pushed, hash-matched) |
| `72b872c` | `cf81ab9` | 2026-10-17 | Record user-run laptop check (RTX 5070 Ti Laptop 12GB, cc 12.0, RAM-limited); next: inspec |
| `3caea2c` | `817f29a` | 2026-10-17 | Korean: all human-facing docs, glossary, report (report_lang ko/en), review tool, CLI and |
| `04853e7` | `2c5b1e6` | 2026-10-17 | setup_on_laptop.sh: run from a temporary copy so a self-updating git pull cannot corrupt t |
| `b507b2a` | `60ce16a` | 2026-10-17 | Docs: update-then-setup order, 109 tests |
| `a0c1e61` | `bdf38ad` | 2026-10-17 | Regenerate synthetic reports in Korean; note for the v1 negative-result archive |
| `cb5306b` | `2b19f4f` | 2026-10-17 | Report: Korean evidence-status line |
| `29ceb9c` | `e35a7b5` | 2026-10-17 | Episode-outcome logger (log-episodes) + equal-time review budgets (budget_unit: seconds), |
| `54f79f2` | `7921ab5` | 2026-10-17 | M10: log-episodes, equal-time budgets, vlm-check, VLM server guide (unverified), LeRobot 0 |
| `e55a163` | `0382295` | 2026-10-17 | setup_on_laptop.sh: reuse a pip-less (uv-made) .venv by installing with uv |
| `013c78d` | `8886cdc` | 2026-10-17 | Status: M10 commits and setup-script fix |
| `e9e4222` | `660453f` | 2026-10-18 | Protocol decisions draft (needs approval); record LeRobot editable install; P1 deferred fo |
| `44c2224` | `6893672` | 2026-10-19 | Docs: GPU usage plan per stage; gym_hil as optional practice; no Isaac Sim |
| `e6cd9c9` | `b991954` | 2026-10-19 | Concrete GPU/RAM estimates per stage; replay-buffer RAM bottleneck (LeRobot code-based cal |
| `0e5c778` | `94626d3` | 2026-10-20 | Protocol v0.2 approved: gamepad (D-28), per-condition demo reward relabel (D-29), Welch se |
| `4badcd9` | `8e1ce14` | 2026-10-20 | D-28: record LeRobot version analysis (leader mode only <=v0.3.3; keyboard needs X11) |
| `6c22beb` | `eac28d1` | 2026-10-20 | Related work: VLMeg uses HIL-SERL on real robot (closest to condition C); paper-potential |
| `92bd511` | `c32942f` | 2026-10-20 | Publication plan (workshop + arXiv + GitHub), pre-release checklist; generalize other-expe |
| `fc7ac70` | `58f04c1` | 2026-10-20 | Pre-release checklist: publish a fresh snapshot repo (history contains old details) |
| `045798c` | `c260e05` | 2026-10-20 | Set project author to gawbi (pyproject authors; future commits authored as gawbi) |
| `224932b` | `7a7f07f` | 2026-10-20 | Pre-release checklist: author gawbi for public snapshot; history not rewritten (reasons); |

이 표 다음의 커밋은 처음부터 작성자가 `gawbi`이고, 번호도 바뀌지 않습니다.
