"""D-26 (D-36): per-episode intervention steps from the LeRobot actor, via the learner's OFFLINE wandb log.

Where the numbers come from (LeRobot 8c920c42, read in the source and checked in tests):
* rl/actor.py counts, per episode, ``episode_total_steps`` and ``episode_intervention_steps`` (steps whose
  transition info has TeleopEvents.IS_INTERVENTION, i.e. the gamepad RB was held) and, when the episode ends,
  sends {"Episodic reward", "Interaction step" (global step index of the episode's LAST step),
  "Episode intervention" (0/1), "Intervention rate" (= intervention_steps / total_steps)} to the learner.
* rl/learner.py ``process_interaction_message`` adds the resume shift to "Interaction step" and logs the dict
  with WandBLogger.log_dict(mode="train") -> keys "train/<name>". Nothing else records these numbers, so the
  HIL-SERL config must enable wandb in OFFLINE mode (nothing is uploaded; files stay in <output_dir>/wandb).
Reconstruction:
  total_steps(k) = step(k) - step(k-1)   (first episode: step(0) + 1, assuming the run started at step 0)
  intervention_steps(k) = round(rate(k) * total_steps(k))   (checked to be an integer within 1e-6)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from hilreview.evaluation.online import OnlineEpisodeRecord

K_STEP = "train/Interaction step"
K_RATE = "train/Intervention rate"
K_FLAG = "train/Episode intervention"
K_REWARD = "train/Episodic reward"


@dataclass(frozen=True)
class ActorEpisode:
    seq: int  # 0-based order of the episode in the run
    end_interaction_step: int
    total_steps: int
    intervention_steps: int
    intervention_flag: bool
    episodic_reward: float | None
    timestamp: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def episodes_from_history(rows: Iterable[Mapping], first_step_offset: int = 0) -> list[ActorEpisode]:
    """wandb history rows -> per-episode actor counts. Rows without the actor keys (losses etc.) are ignored.

    ``first_step_offset``: global step index at which this run's first episode started (0 for a fresh run).
    """
    eps_rows = [r for r in rows if K_STEP in r and K_RATE in r]
    eps_rows.sort(key=lambda r: r[K_STEP])
    out: list[ActorEpisode] = []
    prev_end = first_step_offset - 1
    for i, r in enumerate(eps_rows):
        end = int(r[K_STEP])
        total = end - prev_end
        if total <= 0:
            raise ValueError(f"episode {i}: interaction step {end} does not increase (previous {prev_end})")
        rate = float(r[K_RATE])
        steps_f = rate * total
        steps = round(steps_f)
        if abs(steps_f - steps) > 1e-6 or not 0 <= steps <= total:
            raise ValueError(f"episode {i}: rate {rate} x total {total} = {steps_f} is not an integer step count "
                             "(missing episode rows or wrong first_step_offset?)")
        flag = bool(r.get(K_FLAG, steps > 0))
        if flag != (steps > 0):
            raise ValueError(f"episode {i}: 'Episode intervention'={flag} disagrees with {steps} intervention steps")
        out.append(ActorEpisode(i, end, total, steps, flag, r.get(K_REWARD), r.get("_timestamp")))
        prev_end = end
    return out


def read_wandb_history(path: str | Path) -> list[dict]:  # pragma: no cover - needs wandb (lerobot env)
    """Read history rows from an offline wandb run (.wandb file, or the offline-run-* directory containing one)."""
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore

    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("run-*.wandb")) or sorted(p.glob("*/run-*.wandb"))
        if len(files) != 1:
            raise ValueError(f"expected exactly one run-*.wandb under {p}, found {[str(f) for f in files]}")
        p = files[0]
    ds = datastore.DataStore()
    ds.open_for_scan(str(p))
    rows = []
    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") == "history":
            rows.append({(it.key or "/".join(it.nested_key)): json.loads(it.value_json) for it in rec.history.item})
    return rows


def attach_to_records(
    records: Sequence[OnlineEpisodeRecord], actor_eps: Sequence[ActorEpisode], run_id: str, fps: float
) -> tuple[list[OnlineEpisodeRecord], dict]:
    """Fill intervention_steps/total_steps/fps of this run's TRAIN episodes (in logged order) from the actor counts.

    The human log and the actor must contain the same number of train episodes; otherwise nothing is guessed.
    Returns (all records, with this run's train records updated) and a check report.
    """
    if fps <= 0:
        raise ValueError("fps must be > 0")
    mine = sorted((r for r in records if r.run_id == run_id and r.phase == "train"), key=lambda r: r.episode_index)
    if len(mine) != len(actor_eps):
        raise ValueError(f"run {run_id}: human log has {len(mine)} train episodes, actor has {len(actor_eps)}. "
                         "Fix the log (or the wandb run) first; episodes are matched by order.")
    updated = {}
    disagree = []
    for rec, a in zip(mine, actor_eps):
        if rec.human_intervened is not None and rec.human_intervened != a.intervention_flag:
            disagree.append(rec.episode_index)
        updated[id(rec)] = replace(rec, intervention_steps=a.intervention_steps, total_steps=a.total_steps, fps=fps)
    out = [updated.get(id(r), r) for r in records]
    report = {
        "run_id": run_id,
        "n_train_episodes": len(mine),
        "intervention_steps_total": sum(a.intervention_steps for a in actor_eps),
        "intervention_time_s": sum(a.intervention_steps for a in actor_eps) / fps,
        "episodes_with_intervention": sum(a.intervention_flag for a in actor_eps),
        "human_vs_actor_flag_disagreements": disagree,
    }
    return out, report
