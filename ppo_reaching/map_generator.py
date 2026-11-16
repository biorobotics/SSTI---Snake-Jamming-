"""Generate reproducible, held-out single-obstacle reaching maps."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from clearance_ablation.study import ClearanceEnv, write_json


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs/single_obstacle.json"
TIERS = {
    "mild": dict(q_jitter=.025, along_jitter=.035, lateral_jitter_m=.006, size_jitter=.10),
    "moderate": dict(q_jitter=.065, along_jitter=.10, lateral_jitter_m=.015, size_jitter=.22),
}


def make_scene_env(config, scene):
    env = ClearanceEnv(config | dict(initial_q_rad=scene["initial_q"],
                                     target_witness_rad=scene["target_witness"]), "B_single")
    env.task_boxes = tuple(tuple(box) for box in scene["boxes"])
    env.reset(seed=config["evaluation_seed"])
    np.testing.assert_allclose(env.goal, scene["goal"], rtol=0., atol=1e-10)
    return env


def scene_from_env(env, **metadata):
    return dict(initial_q=env.base_q.tolist(), goal=env.fixed_goal.tolist(),
                target_witness=env.witness.tolist(), boxes=env.task_boxes, **metadata)


def blocked_direct_path(env):
    samples = max(1, math.ceil(np.max(np.abs(env.witness - env.base_q)) / .006))
    for q in np.linspace(env.base_q, env.witness, samples + 1):
        env.set_state(q)
        if env.geometric_failure():
            return True
    return False


def generate_maps(config, count, seed):
    rng = np.random.default_rng(seed)
    reference = ClearanceEnv(config, "B_single")
    reference.reset(seed=config["evaluation_seed"])
    bounds = reference.bounds.copy()
    scenes = {"reference": scene_from_env(reference, tier="reference", proposal=0)}
    reference.close()
    for tier, limits in TIERS.items():
        accepted = 0
        for attempt in range(1, 5001):
            witness = np.asarray(config["target_witness_rad"]) + rng.uniform(-1., 1., 6) * limits["q_jitter"]
            if np.any((witness < bounds[:, 0]) | (witness > bounds[:, 1])):
                continue
            obstacle = dict(config["obstacle"])
            obstacle["fraction_along_tip_goal"] += rng.uniform(-1., 1.) * limits["along_jitter"]
            obstacle["dimensions_m"] = (np.array(obstacle["dimensions_m"]) *
                                         (1. + rng.uniform(-1., 1., 2) * limits["size_jitter"])).tolist()
            offset = float(rng.uniform(-1., 1.) * limits["lateral_jitter_m"])
            env = ClearanceEnv(config | dict(target_witness_rad=witness.tolist(), obstacle=obstacle), "B_single")
            try:
                env.set_state(env.base_q)
                delta = env.fixed_goal - env.marker
                direction = delta / np.linalg.norm(delta)
                box = env.task_boxes[0]
                center = np.array(box[:2]) + offset * np.array([-direction[1], direction[0]])
                env.task_boxes = ((*center, *box[2:]),)
                env.use_boxes(env.task_boxes)
                env.set_state(env.base_q)
                if env.geometric_failure():
                    continue
                env.set_state(env.witness)
                if env.geometric_failure() or not blocked_direct_path(env):
                    continue
                accepted += 1
                scenes[f"{tier}-{accepted:02d}"] = scene_from_env(
                    env, tier=tier, proposal=attempt, lateral_offset_m=offset,
                    obstacle_fraction=obstacle["fraction_along_tip_goal"])
            finally:
                env.close()
            if accepted == count:
                break
        if accepted != count:
            raise RuntimeError(f"Only {accepted} valid {tier} variants found")
    return scenes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12, help="Number of cases per perturbation tier")
    parser.add_argument("--seed", type=int, default=20261115)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be at least one")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    scenes = generate_maps(config, args.count, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, scenes)
    print(f"Saved {len(scenes)} maps to {args.out}")


if __name__ == "__main__":
    main()
