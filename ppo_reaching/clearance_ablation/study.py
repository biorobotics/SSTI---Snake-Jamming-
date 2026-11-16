"""Paired clearance-reward experiment on the existing reaching environment."""

import argparse
from copy import deepcopy
from functools import partial
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reaching_env import ReachingEnv, read_config
from projected_progress.reward import REWARD_NAMES as PROJECTED_COMPONENTS
from train_ppo import write_csv, write_json
from trajectory import REWARD_NAMES, rollout, validate


GROUPS = {
    "A1": ("B_single", 0., "A1_baseline_single"),
    "A2": ("B_single", .1, "A2_clearance_single"),
    "B1": ("C_multi", 0., "B1_baseline_double"),
    "B2": ("C_multi", .1, "B2_clearance_double"),
}
SEEDS = (42, 123, 2026)
BUDGET = 262144
EVAL_SEEDS = range(85000, 85030)
THRESHOLD = .005
CLEARANCE_REWARD = "reward_obstacle_clearance"
COMPONENTS = (*REWARD_NAMES, CLEARANCE_REWARD)


def clearance_penalty(distance, threshold, weight):
    return -weight * max(0., 1. - distance / threshold)


def slow_arrival_reward(speed_m_s, config):
    if not config.get("stepped_arrival_bonus", False):
        return config["arrival_bonus"] if speed_m_s < config["arrival_speed_m_s"] else 0.
    thresholds = (.01, .03, .06, .10, .15)
    lower_bounds = (0., *thresholds)
    lower_bound = lower_bounds[np.searchsorted(thresholds, speed_m_s, side="right")]
    return config["arrival_bonus"] * (thresholds[-1] - lower_bound) / thresholds[-1]


class ClearanceEnv(ReachingEnv):
    @property
    def reward_names(self):
        if "projected_progress_reward" in self.spec:
            return PROJECTED_COMPONENTS
        if "time_penalty" in self.spec["reward"]:
            return ("progress", "success", "action", "smoothness", "collision", "time", CLEARANCE_REWARD)
        if "approach_speed_reward" in self.spec:
            return (*COMPONENTS, "slow_arrival_bonus")
        return COMPONENTS

    @property
    def obstacle_clearance(self):
        return float(np.min(self.link_clearance))

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        options = self.spec["obstacle_clearance"]
        distance = self.obstacle_clearance
        info["obstacle_clearance_m"] = distance
        if "projected_progress_reward" in self.spec:
            return obs, reward, terminated, truncated, info
        penalty = (clearance_penalty(distance, options["threshold_m"], options["weight"])
                   if self.geometry.obstacle_ids else 0.)
        info["reward_components"][CLEARANCE_REWARD] = penalty
        reward += penalty
        speed_config = self.spec.get("approach_speed_reward")
        if speed_config is not None:
            components = info["reward_components"]
            speed = self.tip_speed()
            gate = max(0., 1. - info["distance_m"] / speed_config["radius_m"])**2
            cost = -speed_config["weight"] * gate * speed**2 if info["status"] != "numerical_failure" else 0.
            bonus = (slow_arrival_reward(speed, speed_config)
                     if terminated and info["is_success"] and not info["collision"] else 0.)
            components["terminal_speed"] += cost
            components["slow_arrival_bonus"] = bonus
            reward += cost + bonus
        return obs, reward, terminated, truncated, info


def config_for(group, seed, threshold=THRESHOLD):
    phase, weight, _ = GROUPS[group]
    path = ROOT / "terminal-stop-no-speed-penalty" / phase / phase / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["reward"] == dict(progress=24., success=8., action=.001, smoothness=.0005,
                                    collision=8., other_failure=8.)
    assert config["max_steps"] == 1500 and config["ppo"]["gamma"] == .99
    assert config["goal_tolerance_m"] == .02 and config["torque_limit_nm"] == 2.
    assert config["terminal_stop"]["ee_weight"] == config["terminal_stop"]["joint_weight"] == 0.
    config.update(seed=seed, evaluation_seed=EVAL_SEEDS.start,
                  obstacle_clearance=dict(threshold_m=threshold, weight=weight))
    return config


def stl_ablation_config(group, seed):
    config = config_for(group, seed, .02)
    config.update(mesh_only_collision=True, allow_obstacle_contact=False, torque_limit_nm=4.)
    config.pop("terminal_stop")
    config["obstacle_clearance"]["weight"] = 1. if group == "A2" else 0.
    return config


def use_archived_corridor(config, fixture_path, *, walls_only=False):
    config["corridor_fixture"] = json.loads(fixture_path.read_text(encoding="utf-8"))
    config["mesh_only_collision"] = False
    if walls_only:
        config["corridor_walls_only"] = True


def make_env(config, phase):
    return Monitor(ClearanceEnv(config, phase))


def capture(env, model, seed):
    return rollout(env, seed, lambda _, obs: model.predict(obs, deterministic=True)[0],
                   reward_names=env.reward_names,
                   extra_sample=lambda state: {"obstacle_clearance_m": state.obstacle_clearance})


def metrics(env, trace, urdf):
    row = validate(env, trace, urdf)
    tip = trace["tip"]
    clearance = trace["obstacle_clearance_m"]
    window = int(np.searchsorted(trace["time_s"], trace["time_s"][-1] - 2., side="right") - 1)
    stagnated = (row["status"] == "timeout" and window >= 0
                 and np.max(np.linalg.norm(tip[window:] - tip[window], axis=1)) < .01
                 and trace["distance_m"][window] - trace["distance_m"][-1] < .002)
    return dict(**row, eval_seed=int(trace["seed"]), episode_steps=len(tip)-1,
                episode_return=float(trace["reward"].sum()),
                minimum_clearance_m=float(clearance.min()),
                near_obstacle_ratio=float(np.mean(clearance[1:] < THRESHOLD)),
                within_reward_threshold_ratio=float(np.mean(
                    clearance[1:] < env.spec["obstacle_clearance"]["threshold_m"])),
                timeout=row["status"] == "timeout", stagnated=bool(stagnated),
                torque_failure=row["status"] == "torque_limit" or
                               row["maximum_estimated_torque_nm"] > env.spec["torque_limit_nm"] + 1e-9,
                joint_limit_failure=bool(row["joint_limit_violations"]),
                tip_path_length_m=float(np.linalg.norm(np.diff(tip, axis=0), axis=1).sum()),
                final_joint_velocity_rad_s=trace["dq"][-1].tolist())


class Progress(BaseCallback):
    def __init__(self, config, phase, directory, urdf, *, eval_config=None, save_first_success=False):
        super().__init__()
        self.eval_env = ClearanceEnv(config if eval_config is None else eval_config, phase)
        self.directory, self.urdf = directory, urdf
        self.interval = config["ppo"]["evaluation_interval"]
        self.threshold = config["obstacle_clearance"]["threshold_m"]
        self.reward_names = self.eval_env.reward_names
        self.save_first_success = save_first_success
        self.best_arrival_speed = np.inf
        self.rows = []

    def _on_training_start(self):
        self.sums = np.zeros((self.training_env.num_envs, len(self.reward_names)))
        self.episode_near = np.zeros(self.training_env.num_envs, dtype=int)
        self.rollout_near = self.rollout_inside_threshold = self.rollout_steps = 0
        if "obstacle_contact_cost_per_s" in self.eval_env.spec["reward"]:
            self.episode_contact_s = np.zeros(self.training_env.num_envs)
            self.episode_contact = np.zeros(self.training_env.num_envs, dtype=bool)

    def _on_step(self):
        for i, info in enumerate(self.locals["infos"]):
            self.sums[i] += [info["reward_components"][name] for name in self.reward_names]
            near = info["obstacle_clearance_m"] < THRESHOLD
            self.episode_near[i] += near
            self.rollout_near += near
            self.rollout_inside_threshold += info["obstacle_clearance_m"] < self.threshold
            self.rollout_steps += 1
            duration_logged = hasattr(self, "episode_contact_s")
            if duration_logged:
                self.episode_contact_s[i] += info.get("obstacle_contact_s", 0.)
                self.episode_contact[i] |= info["collision"]
            if "episode" in info:
                episode = info["episode"]
                np.testing.assert_allclose(self.sums[i].sum(), episode["r"], atol=1e-6, rtol=0.)
                self.rows.append(dict(steps=self.num_timesteps, status=info["status"], is_success=info["is_success"],
                                      is_collision=info["collision"], episode_steps=episode["l"],
                                      episode_return=episode["r"], near_ratio=self.episode_near[i]/episode["l"],
                                      **dict(zip(self.reward_names, self.sums[i]))))
                if duration_logged:
                    self.rows[-1].update(episode_obstacle_contact_s=self.episode_contact_s[i],
                                         any_collision=bool(self.episode_contact[i]))
                    self.episode_contact_s[i] = 0.
                    self.episode_contact[i] = False
                self.sums[i] = 0.
                self.episode_near[i] = 0
        return True

    def _on_rollout_end(self):
        if self.rows:
            write_csv(self.directory / "training_episodes.csv", self.rows, append=True)
            self.rows.clear()
        write_csv(self.directory / "training_exposure.csv", [dict(steps=self.num_timesteps,
                  near_steps=self.rollout_near, within_threshold_steps=self.rollout_inside_threshold,
                  control_steps=self.rollout_steps)], append=True)
        self.rollout_near = self.rollout_inside_threshold = self.rollout_steps = 0
        if self.num_timesteps % self.interval:
            return
        trace = capture(self.eval_env, self.model, EVAL_SEEDS.start)
        row = metrics(self.eval_env, trace, self.urdf)
        if "approach_speed_reward" in self.eval_env.spec and row["validated_success"]:
            speed = row["terminal_tip_speed_m_s"]
            if speed < self.best_arrival_speed:
                stem = self.directory / "slowest_strict_success"
                self.model.save(stem)
                np.savez_compressed(stem.with_suffix(".npz"),
                                    ablation_group=np.array(f"Strict success | {self.num_timesteps:,} steps"), **trace)
                write_json(stem.with_suffix(".json"), dict(steps=self.num_timesteps, **row))
                self.best_arrival_speed = speed
        if self.save_first_success and row["validated_success"]:
            stem = self.directory / "first_strict_success"
            self.model.save(stem)
            np.savez_compressed(stem.with_suffix(".npz"),
                                ablation_group=np.array(f"Strict success | {self.num_timesteps:,} training steps"), **trace)
            write_json(stem.with_suffix(".json"), dict(steps=self.num_timesteps, **row))
            self.save_first_success = False
        write_csv(self.directory / "periodic_evaluation.csv", [dict(steps=self.num_timesteps, **row)], append=True)
        if self.num_timesteps % (4*self.interval) == 0:
            self.model.save(self.directory / f"checkpoint_{self.num_timesteps:08d}_before_update")
        write_json(self.directory / "status.json", dict(state="training", steps=self.num_timesteps,
                   success=row["validated_success"], collision=row["collision_count"] > 0))
        print(self.directory.parent.name, self.directory.name, self.num_timesteps,
              row["status"], round(row["final_goal_error_m"], 4), flush=True)


def run(args):
    config = config_for(args.group, args.seed, args.threshold_m)
    phase, _, label = GROUPS[args.group]
    if args.stl_reward_ablation:
        config = stl_ablation_config(args.group, args.seed)
        label = "Clearance enabled (w=1)" if args.group == "A2" else "Clearance disabled (w=0)"
    if args.corridor_fixture is not None:
        use_archived_corridor(config, args.corridor_fixture, walls_only=args.corridor_walls_only)
    if args.torque_limit_nm is not None:
        config["torque_limit_nm"] = args.torque_limit_nm
    if args.approach_speed_weight is not None:
        config["approach_speed_reward"] = dict(weight=args.approach_speed_weight, radius_m=args.approach_speed_radius_m,
                                               arrival_speed_m_s=.01, arrival_bonus=args.slow_arrival_bonus)
        if args.stepped_arrival_bonus:
            config["approach_speed_reward"]["stepped_arrival_bonus"] = True
    if args.obstacle_contact_cost_per_s is not None:
        config["allow_obstacle_contact"] = True
        config["reward"]["obstacle_contact_cost_per_s"] = args.obstacle_contact_cost_per_s
        label = "Strict evaluation | Clearance off"
    if args.regularization_scale != 1.:
        for term in ("action", "smoothness"):
            config["reward"][term] *= args.regularization_scale
        label += f" | Regularization x{args.regularization_scale:g}"
    eval_config = dict(config, allow_obstacle_contact=False) if args.obstacle_contact_cost_per_s is not None else config
    if args.threshold_m != THRESHOLD:
        label += f"  |  d_max = {1000*args.threshold_m:g} mm"
    directory = args.out / args.group / f"seed_{args.seed}"
    if (directory / "complete.json").exists():
        return
    directory.mkdir(parents=True, exist_ok=False)
    options = config["ppo"]
    initial_sha256 = hashlib.sha256(args.initial.read_bytes()).hexdigest()
    if args.threshold_m != THRESHOLD and not args.stl_reward_ablation:
        previous_path = Path(__file__).resolve().parent / args.group / f"seed_{args.seed}" / "config.json"
        previous_config = json.loads(previous_path.read_text(encoding="utf-8"))
        previous_config["obstacle_clearance"]["threshold_m"] = args.threshold_m
        assert all(previous_config[key] == value for key, value in config.items())
        assert previous_config["initialization_sha256"] == initial_sha256
    if args.projected_reward:
        config["projected_progress_reward"] = read_config(args.projected_reward)
        config["projected_progress_reward"]["progress_mode"] = args.progress_mode
        config["obstacle_clearance"]["weight"] = 0.
        progress_label = "Clipped projection" if args.progress_mode == "clipped" else "Euclidean progress"
        label = f"{phase}  |  {progress_label}"
    write_json(directory / "config.json", dict(**config, ablation_group=args.group, phase=phase,
               train_budget=args.budget, evaluation_episodes=args.episodes, initialization=str(args.initial),
               initialization_sha256=initial_sha256))
    if args.obstacle_contact_cost_per_s is not None:
        write_json(directory / "eval_config.json", eval_config)
    write_json(directory / "status.json", dict(state="starting", steps=0))
    vec = SubprocVecEnv([partial(make_env, config, phase)] * options["workers"], start_method="spawn")
    try:
        params = {key: options[key] for key in ("n_steps", "batch_size", "n_epochs", "learning_rate", "gamma", "gae_lambda")}
        model = PPO.load(args.initial, env=vec, device="cpu", seed=args.seed, **params)
        vec.seed(args.seed)
        model.set_logger(configure(str(directory / "logs"), ["csv"]))
        assert args.budget % (options["workers"] * options["n_steps"]) == 0
        model.learn(args.budget, callback=Progress(config, phase, directory, args.urdf,
                                                  eval_config=eval_config,
                                                  save_first_success=args.save_first_success), log_interval=1)
        assert model.num_timesteps == args.budget
        model.save(directory / "final")
    finally:
        vec.close()
    write_json(directory / "status.json", dict(state="evaluating", steps=args.budget))
    env = ClearanceEnv(eval_config, phase)
    rows = []
    for seed in range(EVAL_SEEDS.start, EVAL_SEEDS.start + args.episodes):
        trace = capture(env, model, seed)
        row = metrics(env, trace, args.urdf)
        fingerprint = hashlib.sha256(b"".join(np.ascontiguousarray(trace[key]).tobytes() for key in
                          ("q", "dq", "tip", "time_s", "obstacle_clearance_m", "goal", "boxes", "status"))).hexdigest()
        rows.append(dict(group=args.group, train_seed=args.seed, **row, trace_sha256=fingerprint))
        if seed == EVAL_SEEDS.start:
            np.savez_compressed(directory / "rollout.npz", ablation_group=np.array(label), **trace)
    write_csv(directory / "evaluation.csv", rows)
    result = dict(state="complete", steps=args.budget, deterministic_repeats=len(rows),
                  unique_traces=len({row["trace_sha256"] for row in rows}),
                  success_rate=float(np.mean([row["validated_success"] for row in rows])))
    write_json(directory / "status.json", result)
    write_json(directory / "complete.json", result)


def smoke(out, threshold=THRESHOLD):
    from render_rollout import record

    np.testing.assert_allclose([clearance_penalty(d, threshold, .1) for d in (2*threshold, threshold/2, 0.)],
                               [0., -.05, -.1])
    for group in ("A2", "B2"):
        phase = GROUPS[group][0]
        config = config_for(group, 42, threshold)
        plain, treatment = ReachingEnv(config, phase), ClearanceEnv(config, phase)
        for env in (plain, treatment):
            obs, _ = env.reset(seed=42)
            assert obs.shape == (97,) and env.observation_space.contains(obs)
            assert set(env.geometry.modules) == set(range(12))
            assert len(env.geometry.obstacle_ids) == (1 if group == "A2" else 2)
            assert not (env.geometry.obstacle_ids & set(env.geometry.robot))
            np.testing.assert_array_equal(env.m.jnt_qposadr[env.m.actuator_trnid[:, 0]], env.q_indices)
        for i in range(8):
            cmd = np.linspace(-.02, .02, 6) * (i + 1) / 8
            original = plain.step(plain.encode_position(cmd))
            obs, reward, term, trunc, info = treatment.step(treatment.encode_position(cmd))
            np.testing.assert_array_equal(obs, original[0])
            np.testing.assert_allclose(info["desired_q_rad"], cmd, atol=1e-12)
            np.testing.assert_allclose(obs[76:88], treatment.link_clearance / .12, atol=5e-8)
            assert (term, trunc) == original[2:4]
            assert np.isfinite(reward) and np.isfinite(treatment.link_clearance).all()
            assert 0. <= treatment.obstacle_clearance <= .12
            np.testing.assert_allclose(reward, original[1] + info["reward_components"][CLEARANCE_REWARD])
    config = config_for("A2", 42, threshold)
    free = ClearanceEnv(config, "A_fixed")
    free.reset(seed=42)
    _, _, _, _, info = free.step(free.encode_position(free.q))
    assert not free.geometry.obstacle_ids and info["reward_components"][CLEARANCE_REWARD] == 0.
    # Keep the fixed mount and place a forbidden box through an actual robot collision geom.
    collision_config = deepcopy(config)
    collision_config["max_steps"] = 10
    contact = ClearanceEnv(collision_config, "A_fixed")
    from collision import PLANE
    center = PLANE @ free.d.geom_xpos[free.geometry.robot[-1]]
    # Reset the original clear scene, then inject the box for the step under test.
    contact.reset(seed=42)
    contact.use_boxes(((*center, .06, .06, 0.),))
    contact.set_state(contact.base_q)
    _, reward, term, trunc, info = contact.step(contact.encode_position(contact.q))
    assert np.isfinite(reward) and term and not trunc and info["collision"]
    np.testing.assert_allclose(info["reward_components"][CLEARANCE_REWARD], -.1)
    preview_config = deepcopy(config)
    preview_config["max_steps"] = 5
    preview = ClearanceEnv(preview_config, "B_single")
    trace = rollout(preview, 42, lambda env, _: env.encode_position(env.q), reward_names=COMPONENTS,
                    extra_sample=lambda env: {"obstacle_clearance_m": env.obstacle_clearance})
    record(preview, dict(ablation_group=np.array("Environment smoke test"), **trace), out / "smoke", config["video"])
    print("Smoke passed: observation, actuators, clearance, collision, fixed mount, and renderer", flush=True)


def smoke_projected(settings):
    from collision import PLANE

    for phase, group in (("B_single", "A1"), ("C_multi", "B1")):
        config = config_for(group, 42)
        config["projected_progress_reward"] = settings
        env = ClearanceEnv(config, phase)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (97,)
        start = env.marker.copy()
        length = np.linalg.norm(env.goal - start)
        direction = (env.goal - start) / length
        marker_before = start
        total_progress = 0.
        for _ in range(10):
            _, reward, term, trunc, info = env.step(env.encode_position(env.q + .01))
            assert not term and not trunc
            marker_after = env.marker
            expected = np.diff(np.clip((np.array([marker_before, marker_after]) - start) @ direction, 0., length))[0]
            if settings["progress_mode"] == "euclidean":
                expected = np.linalg.norm(marker_before - env.goal) - np.linalg.norm(marker_after - env.goal)
            progress = info["reward_components"]["progress"]
            np.testing.assert_allclose(progress, settings["progress_weight"] * expected, atol=1e-12)
            total_progress += progress
            marker_before = marker_after
        net_progress = (length - np.linalg.norm(env.marker - env.goal) if settings["progress_mode"] == "euclidean"
                        else np.clip((env.marker - start) @ direction, 0., length))
        np.testing.assert_allclose(total_progress, settings["progress_weight"] * net_progress, atol=1e-12)
        _, reward, term, trunc, info = env.step(env.encode_position(env.q + .01))
        components = info["reward_components"]
        assert tuple(components) == PROJECTED_COMPONENTS and not term and not trunc
        np.testing.assert_allclose(reward, sum(components.values()))
        np.testing.assert_allclose(components["torque_regularization"],
                                   -settings["torque_weight"]*np.mean((env.estimated_torque/2.)**2))
        np.testing.assert_allclose(components["joint_velocity_regularization"],
                                   -settings["joint_velocity_weight"]*np.mean((env.dq/6.)**2))
        center = PLANE @ env.d.geom_xpos[env.geometry.robot[-1]]
        env.use_boxes(((*center, .06, .06, 0.),))
        env.set_state(env.base_q)
        _, reward, term, trunc, info = env.step(env.encode_position(env.q))
        assert term and not trunc and info["collision"]
        assert info["reward_components"]["failure"] == -settings["failure_penalty"]
        _, reward, term, trunc, info = env.step(np.full(6, np.nan))
        assert term and not trunc and info["status"] == "numerical_failure"
        assert reward == -settings["failure_penalty"]
    print("Projected reward passed: simulated before/after positions, episode sum, regularizers and failures", flush=True)


def smoke_stl_ablation():
    configs = [stl_ablation_config(group, 42) for group in ("A1", "A2")]
    expected = configs[0] | {"obstacle_clearance": {"threshold_m": .02, "weight": 1.}}
    assert expected == configs[1]
    left, right = [ClearanceEnv(config, "B_single") for config in configs]
    obs, _ = left.reset(seed=42)
    np.testing.assert_array_equal(obs, right.reset(seed=42)[0])
    assert obs.shape == (97,) and len(left.geometry.robot) == 12
    for _ in range(100):
        a, b = [env.step(env.encode_position(env.witness)) for env in (left, right)]
        np.testing.assert_array_equal(a[0], b[0])
        assert a[2:4] == b[2:4]
        info = b[4]
        terms_a, terms_b = [step[4]["reward_components"] for step in (a, b)]
        assert all(terms_a[key] == terms_b[key] for key in REWARD_NAMES)
        difference = clearance_penalty(info["obstacle_clearance_m"], .02, 1.)
        np.testing.assert_allclose(b[1] - a[1], difference, atol=1e-12)
        if b[2] or b[3]:
            assert b[2] and not b[3] and info["status"] == "collision" and terms_b["collision"] == -8.
            print("STL paired smoke passed: common observations and collision stop, clearance reward is the only difference",
                  flush=True)
            return
    raise RuntimeError("Direct-to-goal smoke rollout did not exercise obstacle contact")


def smoke_contact_duration(rate):
    config = stl_ablation_config("A1", 42)
    config["reward"]["obstacle_contact_cost_per_s"] = rate
    strict = ClearanceEnv(config, "B_single")
    training = ClearanceEnv(dict(config, allow_obstacle_contact=True), "B_single")
    obs, _ = strict.reset(seed=42)
    np.testing.assert_array_equal(obs, training.reset(seed=42)[0])
    assert obs.shape == (97,) and len(training.geometry.robot) == 12
    for _ in range(100):
        _, _, terminated, truncated, info = strict.step(strict.encode_position(strict.witness))
        if terminated or truncated:
            assert terminated and not truncated and info["status"] == "collision"
            assert info["reward_components"]["collision"] == -8.
            break
    else:
        raise RuntimeError("Strict evaluation did not terminate at obstacle contact")

    sustained = False
    for _ in range(100):
        before = training.d.time
        _, reward, terminated, truncated, info = training.step(training.encode_position(training.witness))
        duration = info["obstacle_contact_s"]
        assert 0. <= duration <= training.d.time - before + 1e-10
        terms = info["reward_components"]
        assert terms[CLEARANCE_REWARD] == 0.
        np.testing.assert_allclose(terms["collision"], -rate * duration -
                                   8. * info["termination_flags"]["self_collision"])
        np.testing.assert_allclose(reward, sum(terms.values()))
        if terminated or truncated:
            assert info["status"] != "collision"
            break
        if duration > 0. and info["obstacle_collision_events"] == 0:
            sustained = True
            break
    assert sustained, "Sustained obstacle contact did not continue in the test rollout"
    print("Smoke passed: 97 observations, zero clearance reward, sustained contact cost, strict evaluation", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--joint-limits", "--urdf", dest="urdf", type=Path,
                        default=ROOT / "configs/joint_limits.json",
                        help="Validation limits as JSON or a ReU URDF")
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--episodes", type=int, default=len(EVAL_SEEDS))
    parser.add_argument("--threshold-m", type=float, default=THRESHOLD)
    parser.add_argument("--projected-reward", type=Path)
    parser.add_argument("--progress-mode", choices=("clipped", "euclidean"), default="clipped")
    parser.add_argument("--stl-reward-ablation", action="store_true")
    parser.add_argument("--obstacle-contact-cost-per-s", type=float)
    parser.add_argument("--regularization-scale", type=float, default=1.,
                        help="Multiply the position-command and command-change weights")
    parser.add_argument("--save-first-success", action="store_true")
    parser.add_argument("--torque-limit-nm", type=float, help="Override actuator and RNEA torque limits")
    parser.add_argument("--approach-speed-weight", type=float)
    parser.add_argument("--approach-speed-radius-m", type=float, default=.06)
    parser.add_argument("--slow-arrival-bonus", type=float, default=0.)
    parser.add_argument("--stepped-arrival-bonus", action="store_true")
    parser.add_argument("--corridor-fixture", type=Path, help="Use an archived eight-obstacle corridor and its collision geometry")
    parser.add_argument("--corridor-walls-only", action="store_true", help="Remove the six background obstacles")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.corridor_walls_only and args.corridor_fixture is None:
        parser.error("--corridor-walls-only requires --corridor-fixture")
    if not 0. < args.threshold_m <= .12:
        parser.error("--threshold-m must be positive and at most the 0.12 m distance cap")
    if args.torque_limit_nm is not None and (
        not np.isfinite(args.torque_limit_nm) or args.torque_limit_nm <= 0. or args.smoke
    ):
        parser.error("--torque-limit-nm requires a positive finite limit and a training run")
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stl_reward_ablation and (args.projected_reward or args.group not in (None, "A1", "A2", "B1")
                                     or args.threshold_m != THRESHOLD):
        parser.error("STL reward ablation uses its fixed threshold and the A1/A2/B1 groups")
    rate = args.obstacle_contact_cost_per_s
    if not np.isfinite(args.approach_speed_radius_m) or args.approach_speed_radius_m <= 0.:
        parser.error("--approach-speed-radius-m must be finite and positive")
    if args.approach_speed_weight is not None and (
        not np.isfinite(args.approach_speed_weight) or args.approach_speed_weight <= 0.
        or not args.stl_reward_ablation or args.smoke or args.projected_reward
    ):
        parser.error("Approach speed shaping requires a positive finite weight and an STL training run")
    if not np.isfinite(args.slow_arrival_bonus) or args.slow_arrival_bonus < 0. or (
        args.slow_arrival_bonus and args.approach_speed_weight is None
    ):
        parser.error("Slow arrival bonus must be nonnegative and requires approach speed shaping")
    if args.stepped_arrival_bonus and not args.slow_arrival_bonus:
        parser.error("Stepped arrival bonus requires a positive --slow-arrival-bonus")
    if rate is not None and (not np.isfinite(rate) or rate <= 0. or not args.stl_reward_ablation
                             or args.group not in ("A1", "B1")):
        parser.error("Contact cost requires a positive rate and an STL no-clearance group")
    if not np.isfinite(args.regularization_scale) or args.regularization_scale <= 0.:
        parser.error("--regularization-scale must be finite and positive")
    if args.regularization_scale != 1. and (args.projected_reward or rate is None or args.smoke or args.group != "A1"):
        parser.error("Regularization scaling applies to the STL A1 contact-duration training run")
    if args.smoke and rate is not None:
        smoke_contact_duration(rate)
    elif args.smoke and args.stl_reward_ablation:
        smoke_stl_ablation()
    elif args.smoke and args.projected_reward:
        smoke_projected(dict(read_config(args.projected_reward), progress_mode=args.progress_mode))
    elif args.smoke:
        smoke(args.out, args.threshold_m)
    else:
        if any(value is None for value in (args.group, args.seed, args.initial, args.urdf)):
            parser.error("--group, --seed, --initial, and --urdf are required for training")
        run(args)
