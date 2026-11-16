"""Projected reaching progress and physical joint regularization."""

import numpy as np


REWARD_NAMES = ("progress", "torque_regularization", "joint_velocity_regularization", "failure")


class ProjectedReward:
    def __init__(self, start, goal, config, torque_limit, velocity_scale):
        self.start = start.copy()
        self.goal = goal.copy()
        displacement = goal - start
        self.length = float(np.linalg.norm(displacement))
        self.direction = displacement / self.length if self.length > 0. else np.zeros_like(displacement)
        self.config = config
        self.torque_limit = torque_limit
        self.velocity_scale = velocity_scale

    def clipped_projection(self, marker):
        return float(np.clip((marker - self.start) @ self.direction, 0., self.length))

    def progress(self, before, after):
        if self.config.get("progress_mode", "clipped") == "euclidean":
            delta = np.linalg.norm(before - self.goal) - np.linalg.norm(after - self.goal)
        else:
            delta = self.clipped_projection(after) - self.clipped_projection(before)
        return self.config["progress_weight"] * delta

    def components(self, before, after, torque, dq, *, failed, numerical):
        if numerical:
            return dict(progress=0., torque_regularization=0., joint_velocity_regularization=0.,
                        failure=-self.config["failure_penalty"])
        return dict(progress=self.progress(before, after),
                    torque_regularization=-self.config["torque_weight"] * float(np.mean((torque/self.torque_limit)**2)),
                    joint_velocity_regularization=-self.config["joint_velocity_weight"] * float(np.mean((dq/self.velocity_scale)**2)),
                    failure=-self.config["failure_penalty"] if failed else 0.)
