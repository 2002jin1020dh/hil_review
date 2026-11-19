# Experiment report: synthetic_strategies

> **WARNING: SYNTHETIC DATA.** The episodes below are synthetic (generated, not recorded on a robot).
> Simulator parameters (mock VLM error rates, simulated reviewer accuracy/speed) are **invented**.
> These results only check that the pipeline is wired correctly. They are **not evidence** about real
> robots, real VLMs (e.g. Qwen3-VL), real human review time, or HIL-SERL success rates.

## Provenance

| field | value |
|---|---|
| data kind | **synthetic** (SYNTHETIC: pipeline validation only, not evidence about real systems) |
| config hash | `362c2b1c106318e3` |
| git commit | `b784bdd6976c570ca1fd8f55b22035440fd7b783` (dirty=True) |
| created (UTC) | 2026-10-17T16:10:44Z |
| replicates | 5 (base seed 1000) |
| python / numpy | 3.11.16 / 2.4.6 |

## Data and splits

Episode-level split (group_by = `episode_id`, stratified by `['session_id', 'collection_mode']`, seed 20261017): train 70, val 18, audit 29, excluded (ineligible) 3. The same audit episodes are used for every condition and replicate.

## Condition-independent audit facts

| quantity | value | source |
|---|---|---|
| VLM zero-shot accuracy on audit keyframes | 0.915 (FPR 0.079, FNR 0.118, n=898, 21 VLM-excluded, 33 audit-excluded) | mock VLM (SIMULATED) |
| Blind audit repeat agreement (Cohen's kappa, 4 labels) | 0.862 (observed 0.965, 289 keyframes, 9 episodes) | audit timing: simulated |
| Audit label counts | {'success': 136, 'failure': 783, 'ambiguous': 33, 'invalid': 2, 'none': 0} | reviewers ['sim-reviewer-1'] |
| Audit labels vs synthetic ground truth (accuracy) | 0.993 | synthetic only: real data has no oracle |

## Per-condition results (mean ± SD over replicates)

Primary threshold: `{'policy': 'fixed', 'value': 0.5}` (pre-registered; never tuned on audit). Termination replay: debounce_k=1, tolerance=0 keyframes.

### Labeling cost (human review time source: **simulated**)

| condition | episodes reviewed (k / N) | keyframes reviewed | human review time [min] | corrections | VLM calls |
|---|---|---|---|---|---|
| A_full_review | 88.0 ± 0.0 / 88 ± 0 | 2822.0 ± 0.0 | 42.9 ± 0.0 | 399.0 ± 0.0 | 2822 ± 0 |
| B_review25_random | 22.0 ± 0.0 / 88 ± 0 | 714.8 ± 6.7 | 11.0 ± 4.7 | 99.6 ± 5.1 | 2822 ± 0 |
| B_review25_lowconf | 22.0 ± 0.0 / 88 ± 0 | 779.0 ± 0.0 | 14.1 ± 0.0 | 135.0 ± 0.0 | 2822 ± 0 |
| B_review25_fakesuccess | 22.0 ± 0.0 / 88 ± 0 | 689.0 ± 0.0 | 7.3 ± 0.0 | 88.0 ± 0.0 | 2822 ± 0 |
| C_vlm_only | 0.0 ± 0.0 / 88 ± 0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 2822 ± 0 |

### Training-label quality on the labeling pool

Reference = labels from the 100% human review (itself imperfect). Ambiguous/invalid/no-label keyframes are excluded from training and counted.

| condition | accuracy vs full review | FPR (fake success) | FNR | excluded keyframes | accuracy vs synthetic truth |
|---|---|---|---|---|---|
| A_full_review (reference) | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 120.0 ± 0.0 | 0.983 ± 0.000 |
| B_review25_random | 0.937 ± 0.002 | 0.058 ± 0.003 | 0.088 ± 0.005 | 112.0 ± 4.1 | 0.923 ± 0.005 |
| B_review25_lowconf | 0.942 ± 0.000 | 0.054 ± 0.000 | 0.079 ± 0.000 | 97.0 ± 0.0 | 0.930 ± 0.000 |
| B_review25_fakesuccess | 0.932 ± 0.000 | 0.063 ± 0.000 | 0.092 ± 0.000 | 117.0 ± 0.0 | 0.916 ± 0.000 |
| C_vlm_only | 0.915 ± 0.000 | 0.080 ± 0.000 | 0.117 ± 0.000 | 108.0 ± 0.0 | 0.903 ± 0.000 |

### Reward classifier on the blind audit set

Classifier source: **logistic stand-in on synthetic features (SIMULATED)**.

| condition | keyframe accuracy | precision | FPR | FNR | false-success termination rate | missed-success rate |
|---|---|---|---|---|---|---|
| A_full_review | 0.923 ± 0.000 | 0.802 ± 0.003 | 0.027 ± 0.001 | 0.363 ± 0.004 | 0.392 ± 0.023 | 0.091 ± 0.000 |
| B_review25_random | 0.918 ± 0.002 | 0.807 ± 0.011 | 0.025 ± 0.002 | 0.412 ± 0.016 | 0.381 ± 0.062 | 0.156 ± 0.041 |
| B_review25_lowconf | 0.914 ± 0.000 | 0.782 ± 0.000 | 0.028 ± 0.000 | 0.419 ± 0.000 | 0.440 ± 0.000 | 0.167 ± 0.000 |
| B_review25_fakesuccess | 0.917 ± 0.001 | 0.807 ± 0.005 | 0.024 ± 0.001 | 0.426 ± 0.000 | 0.375 ± 0.000 | 0.182 ± 0.000 |
| C_vlm_only | 0.915 ± 0.001 | 0.793 ± 0.007 | 0.026 ± 0.001 | 0.425 ± 0.003 | 0.453 ± 0.012 | 0.159 ± 0.007 |

Per-replicate episode-cluster bootstrap 95% intervals are in `per_replicate.json` (`audit.keyframe.ci95_episode_bootstrap`). The false-success termination rate is an offline **proxy** from recorded audit episodes, not an online measurement.

## Metric provenance

| metric family | source |
|---|---|
| Data | synthetic |
| VLM suggestions | models ['mock-vlm'], latency ['simulated'], confidence ['mock', 'none'] |
| Pool review timing | simulated (reviewers ['sim-reviewer-1']) |
| Audit labeling timing | simulated |
| Reward classifier | logistic stand-in on synthetic features (SIMULATED) |
| Real-robot metrics | **MISSING** (see below) |

## Missing metrics (not estimated, not imputed)

- **Real-robot task success rate (human-judged)**: MISSING, requires blinded real-robot evaluation episodes.
- **Online false-success termination rate**: MISSING, requires human-judged outcomes of classifier-terminated episodes.
- **Human intervention count / time during online RL**: MISSING, requires LeRobot actor logs from real runs.
- **Training time, wall-clock time, run failure rate**: MISSING, requires real HIL-SERL runs.
- **Labels reviewed per training hour**: MISSING, requires real training wall-clock time.
- **Between-seed variance of robot success**: MISSING, requires >= 2 real seeds per condition.

## Interpretation guardrails

- 5 replicates per condition. Differences are descriptive; no significance test was run.
- Better label or classifier accuracy does not imply better robot success; only blinded real-robot evaluation can show that.
- Differences between conditions here follow from the simulator's invented error and timing parameters. Changing those parameters changes the conclusions.
