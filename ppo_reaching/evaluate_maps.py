"""Evaluate the reaching policy on generated maps in MuJoCo."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from stable_baselines3 import PPO
import torch

from map_generator import DEFAULT_CONFIG, ROOT, make_scene_env
from train_ppo import write_json
from trajectory import rollout, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/single_obstacle_20s_seed61133.zip")
    parser.add_argument("--joint-limits", type=Path, default=ROOT / "configs/joint_limits.json")
    parser.add_argument("--render", action="store_true", help="Render the reference rollout in MP4 and GIF")
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = json.loads(args.config.read_text(encoding="utf-8")) | {"allow_obstacle_contact": False}
    scenes = json.loads(args.maps.read_text(encoding="utf-8"))
    if not scenes:
        parser.error("--maps must contain at least one scene")
    model = PPO.load(args.checkpoint, device="cpu")
    model.predict(np.zeros(model.observation_space.shape, dtype=np.float32), deterministic=True)
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for case, scene in scenes.items():
        if not case.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Invalid map identifier {case!r}")
        env = make_scene_env(config, scene)
        timings = []

        def policy(_, obs):
            start = perf_counter()
            action, _ = model.predict(obs, deterministic=True)
            timings.append(perf_counter() - start)
            return action

        try:
            trace = rollout(env, config["evaluation_seed"] + 1000, policy,
                            extra_sample=lambda state: {"obstacle_clearance_m": np.min(state.link_clearance)})
            result = validate(env, trace, args.joint_limits)
            rows.append(dict(case=case, tier=scene.get("tier", "custom"),
                             minimum_goal_error_m=float(trace["distance_m"].min()),
                             inference_ms=1000. * sum(timings),
                             median_step_ms=1000. * float(np.median(timings)), **result))
            trace["ablation_group"] = np.array(case)
            np.savez_compressed(args.out / f"{case}.npz", inference_s=timings, **trace)
            if case == "reference" and args.render:
                from render_rollout import record

                record(env, trace, args.out / "reference", config["video"])
        finally:
            env.close()
    with (args.out / "evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for tier in dict.fromkeys(r["tier"] for r in rows):
        cases = [row for row in rows if row["tier"] == tier]
        summary[tier] = dict(count=len(cases), successes=sum(row["validated_success"] for row in cases))
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
