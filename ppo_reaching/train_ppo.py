"""Train the reaching curriculum and distill successful MuJoCo executions."""

import argparse
import csv
from functools import partial
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from reaching_env import ReachingEnv, read_config
from trajectory import REWARD_NAMES, fit_spline, replay_spline, rollout, validate
from plots import fitting_plot, training_plot, trajectory_plots


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_env(config, phase, env_factory=ReachingEnv):
    return Monitor(env_factory(config, phase))


def evaluate(model, config, phase, seed, count, urdf, save=None, *, env_factory=ReachingEnv):
    env = env_factory(config, phase)
    rows, representative = [], None
    for i in range(count):
        trace = rollout(env, seed + i, lambda _, obs: model.predict(obs, deterministic=True)[0])
        summary = validate(env, trace, urdf)
        summary["seed"] = seed + i
        summary["episode_return"] = float(trace["reward"].sum())
        rows.append(summary)
        if save:
            suffix = "success" if summary["validated_success"] else "failed"
            np.savez_compressed(save / f"episode_{i:02d}_{suffix}.npz", **trace)
        rank = (not summary["validated_success"], summary["final_goal_error_m"])
        if representative is None or rank < representative[0]:
            representative = (rank, trace, summary)
    env.close()
    return rows, representative


class Progress(BaseCallback):
    def __init__(self, config, phase, directory, urdf, interval, *, env_factory=ReachingEnv):
        super().__init__()
        self.config, self.phase, self.directory, self.urdf = config, phase, directory, urdf
        self.interval = interval
        self.env_factory = env_factory
        self.next_evaluation = interval
        self.best = None
        self.evaluation_path = directory / "periodic_evaluation.csv"

    def _on_training_start(self):
        self.next_evaluation = (self.model.num_timesteps // self.interval + 1) * self.interval
        self.episode_sums = np.zeros((self.training_env.num_envs, len(REWARD_NAMES)))
        self.completed = []
        if (self.directory / "best_score.json").exists():
            self.best = tuple(json.loads((self.directory / "best_score.json").read_text()))

    def _on_step(self):
        for i, info in enumerate(self.locals["infos"]):
            self.episode_sums[i] += [info["reward_components"][key] for key in REWARD_NAMES]
            if "episode" in info:
                np.testing.assert_allclose(self.episode_sums[i].sum(), info["episode"]["r"], rtol=0., atol=1e-6)
                row = dict(steps=self.num_timesteps, status=info["status"], success=info["is_success"],
                           collision=info["collision"], episode_steps=info["episode"]["l"],
                           joint_limit=info["termination_flags"].get("joint_limit", False),
                           final_goal_error_m=info["distance_m"], episode_return=info["episode"]["r"],
                           terminal_tip_speed_m_s=info.get("tip_speed_m_s"))
                row.update(zip(REWARD_NAMES, self.episode_sums[i]))
                self.completed.append(row)
                self.episode_sums[i] = 0.
        return True

    def _on_rollout_end(self):
        if self.completed:
            write_csv(self.directory / "training_episodes.csv", self.completed, append=True)
            self.completed.clear()
        if self.num_timesteps < self.next_evaluation:
            return
        self.next_evaluation += self.interval
        rows, _ = evaluate(self.model, self.config, self.phase, self.config["evaluation_seed"],
                           6 if self.phase == "A_random" else 1, self.urdf, env_factory=self.env_factory)
        row = dict(steps=self.num_timesteps,
                   success_rate=float(np.mean([r["validated_success"] for r in rows])),
                   mean_final_error_m=float(np.mean([r["final_goal_error_m"] for r in rows])),
                   mean_terminal_tip_speed_m_s=float(np.mean([r["terminal_tip_speed_m_s"] for r in rows])),
                   mean_return=float(np.mean([r["episode_return"] for r in rows])),
                   collision_rate=float(np.mean([r["collision_count"] > 0 for r in rows])),
                   joint_limit_rate=float(np.mean([r["joint_limit_violations"] > 0 for r in rows])))
        write_csv(self.evaluation_path, [row], append=True)
        rank = (row["success_rate"], -row["mean_final_error_m"])
        if self.best is None or rank > self.best:
            self.best = rank
            self.model.save(self.directory / "best")
            write_json(self.directory / "best_score.json", rank)
        self.model.save(self.directory / f"checkpoint_{self.num_timesteps:08d}")
        write_json(self.directory.parent / "status.json", dict(state="training", phase=self.phase, **row))
        print(self.phase, row, flush=True)


def write_csv(path, rows, append=False):
    header = not append or not path.exists()
    with path.open("a" if append else "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        if header:
            writer.writeheader()
        writer.writerows(rows)


def train_phase(config, phase, directory, urdf, budget, previous, resume, quick, *, env_factory=ReachingEnv):
    directory.mkdir(exist_ok=True)
    options = config["ppo"]
    workers = 2 if quick else options["workers"]
    vec = SubprocVecEnv([partial(make_env, config, phase, env_factory)] * workers, start_method="spawn")
    try:
        stage_seed = config["seed"] + ("A_fixed", "A_random", "B_single", "C_multi").index(phase)
        vec.seed(stage_seed)
        checkpoints = sorted(directory.glob("checkpoint_*.zip"))
        checkpoint = checkpoints[-1] if resume and checkpoints else previous
        params = {k: options[k] for k in ("n_steps", "batch_size", "n_epochs", "learning_rate", "gamma", "gae_lambda")}
        if quick:
            params.update(n_steps=16, batch_size=32, n_epochs=1)
        if checkpoint:
            model = PPO.load(checkpoint, env=vec, device="cpu", seed=stage_seed, **params)
        else:
            model = PPO("MlpPolicy", vec, seed=stage_seed, device="cpu",
                        policy_kwargs=dict(net_arch=[128, 128]), **params)
        same_phase = bool(resume and checkpoints)
        remaining = max(0, budget - (model.num_timesteps if same_phase else 0))
        model.set_logger(configure(str(directory / ("resume_logs" if same_phase else "logs")), ["csv"]))
        if remaining:
            callback = Progress(config, phase, directory, urdf, 32 if quick else options["evaluation_interval"],
                                env_factory=env_factory)
            model.learn(total_timesteps=remaining, reset_num_timesteps=not same_phase, callback=callback, log_interval=1)
            model.logger.record("time/total_timesteps", model.num_timesteps)
            model.logger.dump(step=model.num_timesteps)
        model.save(directory / "last")
        if not (directory / "best.zip").exists():
            model.save(directory / "best")
    finally:
        vec.close()
    return directory / "best.zip"


def finish_phase(config, phase, directory, urdf, checkpoint, quick):
    from render_rollout import record
    from reaching_env import joint_limits

    model = PPO.load(checkpoint, device="cpu")
    count = config["ppo"]["random_eval_episodes"] if phase == "A_random" and not quick else 1
    seed = config["evaluation_seed"] + 1000
    rows, (_, original, metrics) = evaluate(model, config, phase, seed, count, urdf, save=directory)
    write_csv(directory / "evaluation.csv", rows)
    env = ReachingEnv(config, phase)
    env.reset(seed=metrics["seed"])
    prefix = "ppo_obstacle" if phase in ("B_single", "C_multi") else "ppo_reaching"
    record(env, original, directory / (prefix if metrics["validated_success"] else prefix + "_failed"), config["video"])
    spline = fitted = None
    summary = dict(phase=phase, evaluation_episodes=count,
                   validated_successes=sum(r["validated_success"] for r in rows), ppo=metrics)
    if metrics["validated_success"] and not quick:
        bounds, vmax, _ = joint_limits(env, urdf)
        spline = fit_spline(original, bounds, vmax, config["fit"])
        residual = spline(original["time_s"]) - original["q"]
        fitting_plot(original, spline, directory / "joint_fitting.png")
        np.savez_compressed(directory / "spline.npz", knots=spline.t, coefficients=spline.c, degree=spline.k,
                            joint_names=original["joint_names"])
        fitted = replay_spline(env, metrics["seed"], spline, config["fit"])
        np.savez_compressed(directory / "spline_replay.npz", **fitted)
        validated = validate(env, fitted, urdf)
        internal = np.unique(spline.t)[1:-1]
        left, right = np.nextafter(internal, -np.inf), np.nextafter(internal, np.inf)
        continuity = {f"derivative_{order}_max_jump": float(np.max(
            np.abs(spline(left, nu=order) - spline(right, nu=order)), initial=0.)) for order in (1, 2)}
        replay_reference = spline(np.minimum(fitted["time_s"], spline.t[-1]))
        summary.update(fitted=validated, spline_rmse_rad=float(np.sqrt(np.mean(residual**2))),
                       spline_max_error_rad=float(np.max(np.abs(residual))), spline_interior_continuity="C2",
                       spline_continuity=continuity, replay_tracking_rmse_rad=float(np.sqrt(
                           np.mean((fitted["q"] - replay_reference)**2))))
        stem = "fitted_reaching" if validated["validated_success"] else "fitted_reaching_failed"
        record(env, fitted, directory / stem, config["video"])
    if not summary["validated_successes"] and not quick:
        if phase in ("B_single", "C_multi"):
            from corridor_feasibility import feasible_path
            env.reset(seed=metrics["seed"])
            fixture = dict(q_start_rad=env.base_q, target_witness_q_rad=env.witness, goal_LT_m=env.goal)
            path, attempts = feasible_path(env, fixture, np.random.default_rng(config["evaluation_seed"]),
                                           config["obstacle"]["connectivity_attempts"])
            summary["geometry_probe"] = dict(path_found=path is not None, attempts=attempts)
            if path is not None:
                np.save(directory / "geometry_probe.npy", path)
        elif phase == "A_fixed":
            probe = rollout(env, metrics["seed"], lambda current, _: current.encode_position(current.witness))
            np.savez_compressed(directory / "fixed_setpoint_probe.npz", **probe)
            summary["fixed_setpoint_probe"] = validate(env, probe, urdf)
    write_json(directory / "comparison.json", summary)
    trajectory_plots(original, fitted, spline, directory / "trajectories.png", config["goal_tolerance_m"])
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--evaluate-checkpoint", type=Path)
    parser.add_argument("--phase", choices=("A_fixed", "A_random", "B_single", "C_multi"), default="A_fixed")
    args = parser.parse_args()
    config = read_config(args.config)
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "config.json", config)
    if args.evaluate_checkpoint:
        summary = finish_phase(config, args.phase, args.out, args.urdf, args.evaluate_checkpoint, False)
        print(summary, flush=True)
        return
    check_env(ReachingEnv(config, "A_fixed"), warn=True)
    for phase in ("A_fixed", "A_random", "B_single"):
        env = ReachingEnv(config, phase)
        obs, _ = env.reset(seed=config["seed"])
        if not env.observation_space.contains(obs):
            raise RuntimeError("Observation failed smoke test")
        before, time = env.dq.copy(), env.d.time
        _, reward, _, _, info = env.step(env.encode_position(env.q + .02))
        np.testing.assert_allclose(info["acceleration_rad_s2"], (env.dq - before) / (env.d.time - time))
        np.testing.assert_allclose(reward, sum(info["reward_components"].values()))
    print("MuJoCo, torque, observation and reward smoke checks passed", flush=True)
    previous = None
    summaries = []
    phases = [("A_fixed", "fixed_steps")] if args.quick else [
        ("A_fixed", "fixed_steps"), ("A_random", "random_steps"), ("B_single", "obstacle_steps")]
    for phase, key in phases:
        directory = args.out / phase
        if args.resume and (directory / "comparison.json").exists():
            summary = json.loads((directory / "comparison.json").read_text())
            previous = directory / "best.zip"
        else:
            write_json(args.out / "status.json", dict(state="starting", phase=phase))
            previous = train_phase(config, phase, directory, args.urdf, 32 if args.quick else config["ppo"][key],
                                   previous, args.resume, args.quick)
            write_json(args.out / "status.json", dict(state="evaluating", phase=phase))
            summary = finish_phase(config, phase, directory, args.urdf, previous, args.quick)
        summaries.append(summary)
        write_json(args.out / "summary.json", summaries)
        training_plot(args.out)
        print(summary, flush=True)
        if phase == "A_fixed" and not summary["validated_successes"]:
            break
    write_json(args.out / "status.json", dict(state="complete", phases=[row["phase"] for row in summaries]))


if __name__ == "__main__":
    main()
