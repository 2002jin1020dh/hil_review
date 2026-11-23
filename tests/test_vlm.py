import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hilreview.data.schema import DataKind, Keyframe, Label, SyntheticKeyframeTruth, make_sample_id
from hilreview.labeling.vlm import (
    MockVLMLabeler,
    MockVLMProfile,
    OpenAICompatConfig,
    OpenAICompatVLMLabeler,
    p_success_from_logprobs,
    parse_answer_text,
)


@pytest.mark.parametrize("text,label", [
    ("yes", Label.SUCCESS), ("Yes.", Label.SUCCESS), ("  no", Label.FAILURE), ("No, the cube is outside", Label.FAILURE),
    ("unsure", Label.AMBIGUOUS), ("<think>hmm</think> yes", Label.SUCCESS), ("maybe", None), ("", None), (None, None),
    ("42", None),
])
def test_parse_answer_text(text, label):
    assert parse_answer_text(text) is label


def test_p_success_from_logprobs():
    import math
    content = [
        {"token": "\n", "logprob": -0.01, "top_logprobs": [{"token": "\n", "logprob": -0.01}]},
        {"token": "yes", "logprob": math.log(0.6),
         "top_logprobs": [{"token": "yes", "logprob": math.log(0.6)}, {"token": " No", "logprob": math.log(0.2)},
                          {"token": "Yes", "logprob": math.log(0.1)}, {"token": "unsure", "logprob": math.log(0.05)}]},
    ]
    assert p_success_from_logprobs(content) == pytest.approx(0.7 / 0.9)
    assert p_success_from_logprobs([{"token": "unsure", "logprob": -0.1, "top_logprobs": [
        {"token": "unsure", "logprob": -0.1}]}]) is None
    assert p_success_from_logprobs(None) is None


def _truth(n=200):
    t, kfs = {}, []
    for i in range(n):
        sid = make_sample_id("e", i)
        kfs.append(Keyframe(sid, "e", i, i / 10))
        t[sid] = SyntheticKeyframeTruth(sid, "S" if i % 2 else "F", False, False, False)
    return t, kfs


def test_mock_vlm_error_rate_and_order_independence():
    t, kfs = _truth(2000)
    prof = MockVLMProfile(base_error=0.1, unsure_rate=0.0, request_error_rate=0.0)
    m = MockVLMLabeler(t, prof, seed=1)
    out = m.suggest(kfs)
    err = sum((s.label is Label.SUCCESS) != (t[s.sample_id].state == "S") for s in out) / len(out)
    assert 0.07 < err < 0.13
    rev = {s.sample_id: s for s in m.suggest(list(reversed(kfs)))}
    assert all(rev[s.sample_id] == s for s in out)
    assert all(s.data_kind is DataKind.SYNTHETIC and s.latency_source == "simulated" for s in out)


# ---- local stub of an OpenAI-compatible server (no model, no network beyond localhost) ----

class _Handler(BaseHTTPRequestHandler):
    responses: list = []
    received: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).received.append({"body": body, "auth": self.headers.get("Authorization")})
        status, payload = type(self).responses.pop(0)
        data = json.dumps(payload).encode() if not isinstance(payload, bytes) else payload
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture()
def server():
    _Handler.responses, _Handler.received = [], []
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield srv
    srv.shutdown()


def _kf(tmp_path):
    img = tmp_path / "front.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    return Keyframe(make_sample_id("ep1", 10), "ep1", 10, 1.0, {"front": str(img)})


def test_http_client_happy_path_with_logprobs(server, tmp_path, monkeypatch):
    import math
    _Handler.responses = [(200, {"choices": [{"message": {"content": "yes"}, "logprobs": {"content": [
        {"token": "yes", "logprob": math.log(0.8), "top_logprobs": [
            {"token": "yes", "logprob": math.log(0.8)}, {"token": "no", "logprob": math.log(0.2)}]}]}}]})]
    monkeypatch.setenv("HILREVIEW_TEST_KEY", "secret-value")
    cfg = OpenAICompatConfig(base_url=f"http://127.0.0.1:{server.server_port}/v1", model="local-qwen",
                             api_key_env="HILREVIEW_TEST_KEY", timeout_s=5)
    lab = OpenAICompatVLMLabeler(cfg, task="put the cube in the bowl")
    s = lab.suggest_one(_kf(tmp_path))
    assert s.status == "ok" and s.label is Label.SUCCESS and s.p_success == pytest.approx(0.8)
    assert s.confidence_source == "logprob" and s.latency_source == "measured" and s.data_kind is DataKind.REAL
    req = _Handler.received[0]
    assert req["auth"] == "Bearer secret-value"
    assert "secret-value" not in json.dumps(s.to_dict())  # key never stored
    content = req["body"]["messages"][1]["content"]
    assert content[0]["type"] == "text" and "put the cube in the bowl" in content[0]["text"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert req["body"]["logprobs"] is True and req["body"]["temperature"] == 0.0


def test_http_client_errors_are_recorded_not_raised(server, tmp_path):
    _Handler.responses = [(500, {"error": "boom"}), (200, {"choices": [{"message": {"content": "perhaps"}}]}),
                          (200, {"unexpected": True}), (200, {"choices": [{"message": {"content": "no"}}]})]
    cfg = OpenAICompatConfig(base_url=f"http://127.0.0.1:{server.server_port}/v1", timeout_s=5)
    lab = OpenAICompatVLMLabeler(cfg, task="t")
    kf = _kf(tmp_path)
    s1, s2, s3, s4 = (lab.suggest_one(kf) for _ in range(4))
    assert s1.status == "request_error" and s1.label is None
    assert s2.status == "parse_error" and s2.label is None
    assert s3.status == "parse_error"
    assert s4.status == "ok" and s4.label is Label.FAILURE and s4.p_success is None and s4.confidence_source == "none"


def test_http_client_unreachable_server(tmp_path):
    cfg = OpenAICompatConfig(base_url="http://127.0.0.1:9/v1", timeout_s=2)
    s = OpenAICompatVLMLabeler(cfg, task="t").suggest_one(_kf(tmp_path))
    assert s.status == "request_error"


def test_missing_camera_frame_is_request_error(tmp_path):
    cfg = OpenAICompatConfig(base_url="http://127.0.0.1:9/v1", camera_order=("front", "wrist"), timeout_s=2)
    s = OpenAICompatVLMLabeler(cfg, task="t").suggest_one(_kf(tmp_path))
    assert s.status == "request_error" and "wrist" in (s.raw_response or "")


class _ModelsAndChat(BaseHTTPRequestHandler):
    with_logprobs = True

    def _send(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        self._send({"data": [{"id": "Qwen/Qwen3-VL-4B-Instruct-FP8"}]})

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers["Content-Length"]))
        choice = {"message": {"content": "no"}}
        if type(self).with_logprobs:
            choice["logprobs"] = {"content": [{"token": "no", "logprob": -0.1, "top_logprobs": [
                {"token": "no", "logprob": -0.1}, {"token": "yes", "logprob": -2.4}]}]}
        self._send({"choices": [choice]})

    def log_message(self, *a):
        pass


@pytest.mark.parametrize("with_lp", [True, False])
def test_vlm_check_command(capsys, with_lp):
    pytest.importorskip("PIL")
    from hilreview.cli import main

    _ModelsAndChat.with_logprobs = with_lp
    srv = HTTPServer(("127.0.0.1", 0), _ModelsAndChat)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        rc = main(["vlm-check", "--base-url", f"http://127.0.0.1:{srv.server_port}/v1",
                   "--model", "Qwen/Qwen3-VL-4B-Instruct-FP8", "--timeout", "5"])
    finally:
        srv.shutdown()
    out = capsys.readouterr().out
    assert rc == 0 and "[1/3] 서버 응답 OK" in out and "3/3 성공" in out
    assert ("확신 점수(logprobs): 있음" in out) is with_lp
    assert ("정책을 쓸 수 없습니다" in out) is (not with_lp)


def test_vlm_check_unreachable_and_remote(capsys):
    from hilreview.cli import main

    assert main(["vlm-check", "--base-url", "http://127.0.0.1:9/v1", "--model", "m", "--timeout", "2"]) == 1
    assert main(["vlm-check", "--base-url", "https://api.example.com/v1", "--model", "m"]) == 2
    out = capsys.readouterr().out
    assert "연결하지 못했습니다" in out and "중단" in out


def test_disable_thinking_request_and_version_tag(tmp_path):
    from PIL import Image

    from hilreview.data.schema import Keyframe
    from hilreview.labeling.vlm import PROMPT_VERSION, OpenAICompatConfig, OpenAICompatVLMLabeler

    img = tmp_path / "f.png"
    Image.new("RGB", (4, 4)).save(img)
    kf = Keyframe("e/f000000", "e", 0, 0.0, {"front": str(img)})
    plain = OpenAICompatVLMLabeler(OpenAICompatConfig(), task="t")
    assert "chat_template_kwargs" not in plain.build_request(kf) and plain.prompt_version == PROMPT_VERSION
    nt = OpenAICompatVLMLabeler(OpenAICompatConfig(disable_thinking=True), task="t")
    assert nt.build_request(kf)["chat_template_kwargs"] == {"enable_thinking": False}
    assert nt.prompt_version == PROMPT_VERSION + "+nothink"
