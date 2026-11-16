"""Repeat the archived single-obstacle training protocol."""

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
import torch

from map_generator import DEFAULT_CONFIG, ROOT
from train_ppo import evaluate, train_phase, write_csv, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--initial", type=Path, default=ROOT / "checkpoints/unobstructed_random_initial.zip")
    parser.add_argument("--joint-limits", type=Path, default=ROOT / "configs/joint_limits.json")
    parser.add_argument("--steps", type=int, help="Override the single-obstacle transition budget")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    options = config["ppo"]
    steps = args.steps if args.steps is not None else options["obstacle_steps"]
    if steps < 1 or steps % (options["workers"] * options["n_steps"]):
        parser.error("--steps must be a positive multiple of workers × n_steps")
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "config.json", config)
    directory = args.out / "B_single"
    checkpoint = train_phase(config, "B_single", directory, args.joint_limits,
                             steps, args.initial, False, False)
    model = PPO.load(checkpoint, device="cpu")
    rows, _ = evaluate(model, config, "B_single", config["evaluation_seed"]+1000, 1, args.joint_limits, save=directory)
    write_csv(directory / "evaluation.csv", rows)
    write_json(args.out / "status.json", dict(state="complete", steps=steps, **rows[0]))
    print(json.dumps(rows[0], indent=2))


if __name__ == "__main__":
    main()
