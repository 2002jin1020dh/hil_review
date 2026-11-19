"""CLI integration test of the full file-based workflow on SYNTHETIC demo data:
make-demo-data -> vlm-label (stub server) -> review (pool + audit) -> run --export-labels -> schedule."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import yaml

from hilreview.cli import main
from hilreview.data.io import read_jsonl
from hilreview.data.schema import ReviewRecord, VLMSuggestion


class _YesNo(BaseHTTPRequestHandler):
    n = 0

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers["Content-Length"]))
        type(self).n += 1
        ans = "yes" if type(self).n % 3 == 0 else "no"
        data = json.dumps({"choices": [{"message": {"content": ans}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture()
def stub_url():
    srv = HTTPServer(("127.0.0.1", 0), _YesNo)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_full_file_workflow(tmp_path, monkeypatch, stub_url, capsys):
    pytest.importorskip("PIL")
    demo = tmp_path / "demo"
    assert main(["make-demo-data", "--out", str(demo), "--n-episodes", "12", "--seed", "1"]) == 0
    eps, kfs, split = demo / "episodes.jsonl", demo / "keyframes.jsonl", demo / "splits.json"

    # VLM labeling against a local stub server: pilot limit, then resume without duplicates
    vlm_out = tmp_path / "vlm_real_server.jsonl"
    args = ["vlm-label", "--episodes", str(eps), "--keyframes", str(kfs), "--out", str(vlm_out),
            "--task", "put the red cube into the blue container", "--base-url", stub_url, "--timeout", "5"]
    assert main(args + ["--limit", "5"]) == 0
    assert len(read_jsonl(vlm_out)) == 5
    assert main(args) == 0
    sugg = read_jsonl(vlm_out, VLMSuggestion.from_dict)
    n_kf = len(read_jsonl(kfs))
    assert len(sugg) == n_kf and len({s.sample_id for s in sugg}) == n_kf
    assert all(s.data_kind.value == "synthetic" and s.latency_source == "measured" for s in sugg)

    # human review sessions (scripted input): pool = assisted, audit = blind
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    pool_out, audit_out = tmp_path / "pool_reviews.jsonl", tmp_path / "audit_reviews.jsonl"
    common = ["--episodes", str(eps), "--keyframes", str(kfs), "--split-file", str(split), "--reviewer-id", "tester",
              "--filmstrip-dir", str(tmp_path / "strips")]
    assert main(["review", *common, "--purpose", "pool_review", "--vlm", str(vlm_out), "--out", str(pool_out)]) == 0
    assert main(["review", *common, "--purpose", "audit", "--out", str(audit_out)]) == 0
    assert "합성(가짜) 에피소드" in capsys.readouterr().out
    pool = read_jsonl(pool_out, ReviewRecord.from_dict)
    audit = read_jsonl(audit_out, ReviewRecord.from_dict)
    sp = json.loads(split.read_text())["episode_split"]
    assert {r.episode_id for r in pool} == {e for e, s in sp.items() if s in ("train", "val")}
    assert {r.episode_id for r in audit} == {e for e, s in sp.items() if s == "audit"}
    assert all(r.timing_source == "measured" and r.data_kind.value == "synthetic" for r in pool + audit)
    assert all(not r.shown_vlm for r in audit) and all(r.shown_vlm for r in pool)
    assert list((tmp_path / "strips").glob("*.png"))

    # experiment over the files, with label export
    cfg = {"experiment_name": "demo_files",
           "data": {"kind": "files", "files": {"episodes": str(eps), "keyframes": str(kfs),
                                                "vlm_suggestions": str(vlm_out), "pool_reviews": str(pool_out),
                                                "audit_reviews": str(audit_out), "split_file": str(split)}},
           "conditions": [{"name": "A", "policy": "full", "fraction": 1.0},
                          {"name": "B", "policy": "random", "fraction": 0.25},
                          {"name": "C", "policy": "none", "fraction": 0.0}],
           "replicates": {"n": 2, "base_seed": 3}}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "res"
    assert main(["run", "--config", str(cfg_path), "--out", str(out), "--export-labels"]) == 0
    report = (out / "report.md").read_text()
    assert "경고: 합성(가짜) 데이터" in report
    assert "없음(MISSING) (분류기 점수가 제공되지 않음)" in report
    assert "사람 검토 시간 출처: **실측**" in report  # timing source comes from the records (measured)
    for c in ("A", "B", "C"):
        for r in (0, 1):
            rows = read_jsonl(out / "labels" / f"{c}_rep{r}.jsonl")
            assert rows and all(x["condition"] == c for x in rows)
    a_rows = read_jsonl(out / "labels" / "A_rep0.jsonl")
    assert all(x["source"] in ("human", "auto_invalid") for x in a_rows)
    assert all(x["source"] in ("vlm", "none", "auto_invalid") for x in read_jsonl(out / "labels" / "C_rep0.jsonl"))

    # blinded schedule refuses to overwrite the sealed key
    s_args = ["schedule", "--conditions", "A,B,C", "--replicates", "2", "--seed", "5",
              "--out-schedule", str(tmp_path / "sched.json"), "--out-key", str(tmp_path / "key.json")]
    assert main(s_args) == 0
    assert main(s_args) == 2

    # freezing a split refuses to overwrite an existing one
    assert main(["split", "--episodes", str(eps), "--out", str(split)]) == 2
