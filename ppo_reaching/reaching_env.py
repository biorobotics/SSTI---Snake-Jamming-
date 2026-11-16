"""PPO reaching tasks on the existing MuJoCo position servo and 6-DOF model."""

import json
import os
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import gymnasium as gym
import mujoco as mj
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
BASE = Path(os.environ.get("SNAKE_PPO_BASE", HERE / "robot_env"))
sys.path.insert(1, str(BASE))
from snake_env import ACTIVE, PLANE, SnakeEnv, load_config
from observations import observation as base_observation
from projected_progress.reward import ProjectedReward


def read_config(path=HERE / "config.yaml"):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


class ReachingEnv(SnakeEnv):
    def __init__(self, config, phase):
        inherited = load_config(BASE / "config.yaml")
        inherited["environment"].update(goal_tolerance_m=config["goal_tolerance_m"],
                                        max_steps=config["max_steps"],
                                        actuator_limit_nm=config["torque_limit_nm"],
                                        rnea_limit_nm=config["torque_limit_nm"],
                                        allow_obstacle_contact=config.get("allow_obstacle_contact", False),
                                        mesh_only_collision=config.get("mesh_only_collision", False))
        super().__init__(inherited)
        self.spec, self.phase = config, phase
        corridor = config.get("corridor_fixture")
        if corridor is not None:
            if tuple(corridor["planned_joints"]) != tuple(ACTIVE):
                raise ValueError("Archived corridor joint order differs from the MuJoCo environment")
            self.base_q = np.asarray(corridor["q_start_rad"], dtype=float)
            self.witness = np.asarray(corridor["target_witness_q_rad"], dtype=float)
            self.fixed_goal = np.asarray(corridor["goal_LT_m"], dtype=float)
            self.fixture = corridor
            walls = [(*wall[:2], corridor["wall_length_m"], corridor["wall_thickness_m"], wall[2])
                     for wall in corridor["walls_LT_theta_rad"]]
            self.task_boxes = tuple(walls) if config.get("corridor_walls_only") else (*self.base_boxes, *walls)
            self.observation_space = gym.spaces.Box(-1., 1., (self.observation_space.shape[0] + 9,), np.float32)
            self.previous = self.base_q.copy()
            return
        self.base_q = np.array(config["initial_q_rad"], dtype=float)
        self.witness = np.array(config["target_witness_rad"], dtype=float)
        self.use_boxes(())
        self.set_state(self.base_q)
        start = self.marker.copy()
        self.set_state(self.witness)
        self.fixed_goal = self.marker.copy()
        self.task_boxes = ()
        if phase in {"B_single", "C_multi"}:
            obstacle = config["obstacle"]
            center = start + obstacle["fraction_along_tip_goal"] * (self.fixed_goal - start)
            self.task_boxes = ((*center, *obstacle["dimensions_m"], obstacle["yaw_rad"]),)
            if phase == "C_multi":
                self.task_boxes += tuple(tuple(box) for box in config["extra_boxes"])
        if phase != "A_random":
            self.fixture = {"target_witness_q_rad": self.witness}
        self.observation_space = gym.spaces.Box(-1., 1., (self.observation_space.shape[0] + 9,), np.float32)
        self.previous = self.base_q.copy()

    def task(self):
        if "corridor_fixture" in self.spec:
            return self.task_boxes, self.base_q, self.fixed_goal, self.fixture["gap_cm"]
        if self.phase == "A_random":
            radius = self.spec["random_start_radius_rad"]
            return (), self.base_q + self.np_random.uniform(-radius, radius, 6), None, None
        return self.task_boxes, self.base_q, self.fixed_goal, None

    def sample_goal(self, start_marker):
        radius = self.spec["random_target_radius_rad"]
        for _ in range(80):
            self.set_state(self.witness + self.np_random.uniform(-radius, radius, 6))
            goal = self.marker
            if .06 < np.linalg.norm(goal - start_marker) < .5 and self.geometric_failure() is None:
                return goal
        return None

    def augment(self, observation):
        return np.clip(np.r_[observation, self.d.site_xpos[self.marker_id],
                             self.previous / self.joint_scale], -1., 1.).astype(np.float32)

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self.previous = self.q.copy()
        self.settled_steps = 0
        reward_config = self.spec.get("projected_progress_reward")
        self.projected_reward = (ProjectedReward(self.marker, self.goal, reward_config,
                                                 self.spec["torque_limit_nm"], self.cfg["velocity_scale_rad_s"])
                                 if reward_config is not None else None)
        return self.augment(observation), info

    def tip_speed(self):
        velocity = np.empty(6)
        mj.mj_objectVelocity(self.m, self.d, mj.mjtObj.mjOBJ_SITE, self.marker_id, velocity, 0)
        return float(np.linalg.norm(PLANE @ velocity[3:]))

    def step(self, action):
        marker_before = self.marker
        before = np.linalg.norm(marker_before - self.goal)
        previous_q, previous_velocity, previous_time = self.q.copy(), self.dq.copy(), self.d.time
        if not np.all(np.isfinite(action)):
            return self.numerical_failure({})
        requested = self.bounds[:, 0] + (.5 * (np.clip(action, -1., 1.) + 1.)) * np.ptp(self.bounds, axis=1)
        # Map absolute policy setpoints onto the existing bounded position increment.
        cap = self.spec["servo_error_cap_rad"]
        delta = np.clip(requested - self.q, -cap, cap) / self.cfg["action_scale_rad"]
        warnings = sum(self.d.warning[i].number for i in range(int(mj.mjtWarning.mjWARN_BADQPOS),
                                                                int(mj.mjtWarning.mjWARN_BADCTRL) + 1))
        observation, _, terminated, truncated, info = super().step(delta)
        current_warnings = sum(self.d.warning[i].number for i in range(int(mj.mjtWarning.mjWARN_BADQPOS),
                                                                        int(mj.mjtWarning.mjWARN_BADCTRL) + 1))
        elapsed = self.d.time - previous_time
        if current_warnings > warnings or not np.all(np.isfinite(observation)) or elapsed <= 0.:
            self.set_state(previous_q)
            self.d.qvel[self.v_indices] = previous_velocity
            self.d.time = previous_time
            mj.mj_forward(self.m, self.d)
            return self.numerical_failure(info["termination_flags"])
        acceleration = (self.dq - previous_velocity) / elapsed
        actual_acceleration = self.d.qacc.copy()
        try:
            self.d.qacc[self.v_indices] = acceleration
            mj.mj_rne(self.m, self.d, 1, self.rne_force)
        finally:
            self.d.qacc[:] = actual_acceleration
        self.estimated_torque = (self.rne_force - self.d.qfrc_passive)[self.v_indices].copy()
        if not np.all(np.isfinite(self.estimated_torque)):
            return self.numerical_failure(info["termination_flags"])
        if np.any(np.abs(self.estimated_torque) > self.spec["torque_limit_nm"] + 1e-9):
            terminated, truncated = True, False
            info["termination_flags"]["torque_limit"] = True
            info.update(is_success=False, torque_violation=True)
            if info["status"] in {"running", "success", "timeout"}:
                info["status"] = "torque_limit"
        speed_penalty = 0.
        stop = self.spec.get("terminal_stop")
        if stop:
            speed = self.tip_speed()
            peak_joint_speed = float(np.max(np.abs(self.dq)))
            gate = max(0., 1. - info["distance_m"] / stop["penalty_radius_m"])**2
            speed_penalty = -gate * (stop["ee_weight"] * (speed / stop["ee_scale_m_s"])**2
                                    + stop["joint_weight"] * np.mean((self.dq / stop["joint_scale_rad_s"])**2))
            slow = (speed <= stop["ee_speed_tolerance_m_s"]
                    and peak_joint_speed <= stop["joint_speed_tolerance_rad_s"])
            close = info["termination_flags"]["success"]
            self.settled_steps = self.settled_steps + 1 if info["is_success"] and slow else 0
            info.update(goal_position_met=close, settled_steps=self.settled_steps, tip_speed_m_s=speed,
                        max_joint_speed_rad_s=peak_joint_speed)
            if info["is_success"] and self.settled_steps < stop["settle_steps"]:
                terminated = False
                truncated = info["termination_flags"]["timeout"]
                info["termination_flags"]["success"] = False
                info.update(status="timeout" if truncated else "running", is_success=False)
        weights = self.spec["reward"]
        failure = terminated and not info["is_success"] and not info["collision"]
        progress = (weights["distance_base"] - weights.get("distance_weight", 1.) * info["distance_m"]
                    if "distance_base" in weights
                    else weights["progress"] * (before - info["distance_m"]))
        collision_events = (info["obstacle_collision_events"] + int(info["termination_flags"]["self_collision"])
                            if "obstacle_collision_events" in info else int(info["collision"]))
        collision_reward = -weights["collision"] * collision_events
        if "obstacle_contact_cost_per_s" in weights and "obstacle_contact_s" in info:
            self_collision = info["termination_flags"]["self_collision"]
            collision_reward = (-weights["obstacle_contact_cost_per_s"] * info["obstacle_contact_s"]
                                - weights["collision"] * int(self_collision))
            failure = terminated and not info["is_success"] and not self_collision
        components = dict(progress=progress,
                          success=weights["success"] * info["is_success"],
                          action=-weights["action"] * float(requested @ requested),
                          smoothness=-weights["smoothness"] * float(np.sum((requested - self.previous)**2)),
                          collision=collision_reward,
                          safety=-weights["other_failure"] * failure, terminal_speed=speed_penalty)
        if "time_penalty" in weights:
            components["time"] = -weights["time_penalty"]
        if self.projected_reward is not None:
            components = self.projected_reward.components(
                marker_before, self.marker, self.estimated_torque, self.dq,
                failed=terminated and not info["is_success"], numerical=False)
        self.previous = requested
        info.update(reward_components=components, desired_q_rad=requested.copy(), acceleration_rad_s2=acceleration)
        return self.augment(observation), float(sum(components.values())), terminated, truncated, info

    def numerical_failure(self, flags):
        self.actual_torque = self.d.actuator_force.copy()
        self.estimated_torque = np.zeros(6)
        self.command = self.d.ctrl.copy()
        self.link_clearance = self.geometry.link_clearance(self.cfg["clearance_scale_m"])
        info = self.info("numerical_failure")
        components = dict(progress=0., success=0., action=0., smoothness=0., collision=0.,
                          safety=-self.spec["reward"]["other_failure"], terminal_speed=0.)
        if "time_penalty" in self.spec["reward"]:
            components["time"] = -self.spec["reward"]["time_penalty"]
        if self.projected_reward is not None:
            marker = self.marker
            components = self.projected_reward.components(
                marker, marker, self.estimated_torque, self.dq, failed=True, numerical=True)
        info.update(termination_flags=dict(flags, numerical_failure=True), reward_components=components,
                    desired_q_rad=self.previous.copy(), acceleration_rad_s2=np.zeros(6))
        return self.augment(base_observation(self, self.link_clearance)), float(sum(components.values())), True, False, info

    def rnea_demand(self, command):
        # The experiment checks inverse dynamics after observing the velocity change.
        return np.zeros(6)

    def encode_position(self, q):
        return np.clip(2. * (q - self.bounds[:, 0]) / np.ptp(self.bounds, axis=1) - 1., -1., 1.)


def joint_limits(env, urdf_path):
    if urdf_path.suffix == ".json":
        limits = json.loads(urdf_path.read_text(encoding="utf-8"))
        velocity = [limits[name]["velocity_rad_s"] for name in ACTIVE]
        effort = [limits[name]["effort_nm"] for name in ACTIVE]
        velocity = np.array([np.inf if value is None else value for value in velocity])
        effort = np.array([np.inf if value is None else value for value in effort])
        return env.bounds.copy(), velocity, np.minimum(effort, env.actuator_limits)
    joints = {j.get("name"): j.find("limit") for j in ET.parse(urdf_path).getroot().findall("joint")
              if j.get("type") == "revolute"}
    velocity = np.full(len(ACTIVE), np.inf)
    effort = np.full(len(ACTIVE), np.inf)
    for i, name in enumerate(ACTIVE):
        if name.startswith("joint_"):
            limit = joints[f"reu_joint_out2in_{name.split('_')[-1]}"]
            velocity[i], effort[i] = float(limit.get("velocity")), float(limit.get("effort"))
    return env.bounds.copy(), velocity, np.minimum(effort, env.actuator_limits)
