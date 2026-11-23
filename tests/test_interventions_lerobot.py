"""D-26/D-36 end-to-end with REAL LeRobot (8c920c42) learner logging + offline wandb.

Scripted per-step intervention flags -> the actor's per-episode message (same arithmetic as rl/actor.py)
-> lerobot.rl.learner.process_interaction_message -> lerobot WandBLogger.log_dict (offline wandb)
-> hilreview reads the .wandb file back and must recover the exact per-episode counts.

  ~/miniforge3/envs/lerobot/bin/python -m pytest -q tests/test_interventions_lerobot.py
"""

import json
import os

import pytest

pytest.importorskip("lerobot")
wandb = pytest.importorskip("wandb")

from lerobot.common.wandb_utils import WandBLogger  # noqa: E402
from lerobot.rl.learner import process_interaction_message  # noqa: E402
from lerobot.transport.utils import python_object_to_bytes  # noqa: E402

from hilreview.data.io import read_jsonl, write_jsonl  # noqa: E402
from hilreview.data.schema import DataKind  # noqa: E402
from hilreview.evaluation.interventions import episodes_from_history, read_wandb_history  # noqa: E402
from hilreview.evaluation.online import OnlineEpisodeRecord  # noqa: E402

# per episode: list of is_intervention flags, one per actor step
EPISODES = [
    [False] * 30 + [True] * 7 + [False] * 13,  # 50 steps, 7 intervention
    [False] * 80,  # 80 steps, none
    [True] * 3 + [False] * 10 + [True] * 20 + [False] * 7,  # 40 steps, 23 intervention
    [True] * 9,  # 9 steps, all intervention
]


def _actor_messages(episodes):
    """Mirror rl/actor.py: counters per episode; message sent when the episode ends."""
    msgs, interaction_step = [], -1
    for flags in episodes:
        total = inter = 0
        for f in flags:
            interaction_step += 1
            total += 1
            inter += int(f)
        msgs.append({"Episodic reward": 0.0, "Interaction step": interaction_step,
                     "Episode intervention": int(inter > 0), "Intervention rate": inter / total if total else 0.0,
                     "Policy frequency [Hz]": 10.0})
    return msgs


@pytest.fixture(scope="module")
def wandb_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("wb")
    os.environ["WANDB_SILENT"] = "true"
    logger = object.__new__(WandBLogger)  # skip cfg plumbing; log_dict is the code under test
    logger._wandb_custom_step_key = None
    logger._wandb = wandb
    run = wandb.init(project="hilreview-test", mode="offline", dir=str(d))
    for m in _actor_messages(EPISODES):
        process_interaction_message(python_object_to_bytes(m), interaction_step_shift=0, wandb_logger=logger)
        logger.log_dict({"Optimization step": 1, "loss_critic": 0.5}, mode="train",
                        custom_step_key="Optimization step")  # learner also logs other rows
    run_dir = run.dir
    run.finish()
    return d, run_dir


def test_counts_round_trip_exactly(wandb_run):
    d, _ = wandb_run
    run_dirs = list((d / "wandb").glob("offline-run-*"))
    assert len(run_dirs) == 1
    eps = episodes_from_history(read_wandb_history(run_dirs[0]))
    assert [e.total_steps for e in eps] == [len(f) for f in EPISODES]
    assert [e.intervention_steps for e in eps] == [sum(f) for f in EPISODES]
    assert [e.intervention_flag for e in eps] == [any(f) for f in EPISODES]


def test_cli_extract_and_attach(wandb_run, tmp_path, capsys):
    from hilreview.cli import main

    d, _ = wandb_run
    run_dir = next((d / "wandb").glob("offline-run-*"))
    recs = [OnlineEpisodeRecord("run7", "train", i, "timeout", None, DataKind.SYNTHETIC) for i in range(4)]
    write_jsonl(tmp_path / "eps.jsonl", recs)
    assert main(["extract-interventions", "--wandb-run", str(run_dir), "--fps", "10", "--out",
                 str(tmp_path / "actor.jsonl"), "--episodes", str(tmp_path / "eps.jsonl"), "--run-id", "run7",
                 "--episodes-out", str(tmp_path / "eps_filled.jsonl")]) == 0
    filled = read_jsonl(tmp_path / "eps_filled.jsonl", OnlineEpisodeRecord.from_dict)
    assert [r.intervention_steps for r in filled] == [7, 0, 23, 9]
    assert [r.total_steps for r in filled] == [50, 80, 40, 9]
    assert "개입 스텝 39개 = 3.9초" in capsys.readouterr().out
    assert json.loads((tmp_path / "actor.jsonl").read_text().splitlines()[0])["total_steps"] == 50
