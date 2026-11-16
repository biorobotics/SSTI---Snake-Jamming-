import numpy as np


# Eight slots accommodate the original corridor benchmark. The training
# curriculum uses at most three and leaves the remaining slots masked.
BOX_SLOTS = 8
OBS_SIZE = 6 + 6 + 2 + 6 + BOX_SLOTS * 7 + 12


def observation(env, clearance=None):
    obstacle_features = np.zeros((BOX_SLOTS, 7))
    for index, (longitudinal, transverse, length, width, angle) in enumerate(env.boxes):
        obstacle_features[index] = [2. * longitudinal - 1., 2. * transverse, length, width,
                                    np.sin(angle), np.cos(angle), 1.]
    if clearance is None:
        clearance = env.geometry.link_clearance(env.cfg["clearance_scale_m"])
    state = np.r_[env.q / env.joint_scale, env.dq / env.cfg["velocity_scale_rad_s"],
                  env.goal - env.marker, env.actual_torque / env.actuator_limits,
                  obstacle_features.ravel(), clearance / env.cfg["clearance_scale_m"]]
    return np.clip(state, -1., 1.).astype(np.float32)
