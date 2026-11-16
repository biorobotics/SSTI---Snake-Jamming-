"""Capture real rollouts, fit a bounded cubic B-spline, and validate replay."""

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import Bounds, LinearConstraint, minimize

from reaching_env import ACTIVE, joint_limits


REWARD_NAMES = ("progress", "success", "action", "smoothness", "collision", "safety", "terminal_speed")


def rollout(env, seed, policy, *, reward_names=REWARD_NAMES, extra_sample=None):
    obs, _ = env.reset(seed=seed)
    samples = []

    def append(action, reward, info):
        samples.append(dict(time_s=float(env.d.time), q=env.q.copy(), dq=env.dq.copy(),
            ddq=info.get("acceleration_rad_s2", np.zeros(6)),
            requested_q=env.previous.copy(), command_q=env.command.copy(),
            torque=env.estimated_torque.copy(), actuator_torque=env.actual_torque.copy(),
            tip=env.marker.copy(), tip_world=env.d.site_xpos[env.marker_id].copy(),
            anchor=env.d.xanchor[env.m.joint(ACTIVE[0]).id].copy(),
            tip_speed_m_s=env.tip_speed(), settled_steps=info.get("settled_steps", 0),
            distance_m=np.linalg.norm(env.marker - env.goal), action=action.copy(), reward=reward,
            reward_components=[info.get("reward_components", {}).get(name, 0.) for name in reward_names],
            collision=info.get("collision", False),
            contact_point=info.get("collision_point_LT_m") or [np.nan, np.nan],
            joint_limit=info.get("termination_flags", {}).get("joint_limit", False)))
        if extra_sample is not None:
            samples[-1].update(extra_sample(env))

    append(env.encode_position(env.q), 0., {})
    terminated = truncated = False
    while not (terminated or truncated):
        action = np.asarray(policy(env, obs))
        obs, reward, terminated, truncated, info = env.step(action)
        append(action, reward, info)
    trace = {name: np.array([row[name] for row in samples]) for name in samples[0]}
    trace.update(status=np.array(info["status"]), success=np.array(info["is_success"]),
                 reward_names=np.array(reward_names), joint_names=np.array(ACTIVE),
                 goal=env.goal.copy(), boxes=np.array(env.boxes).reshape(-1, 5), seed=np.array(seed))
    return trace


def fit_spline(trace, bounds, vmax, settings):
    if not bool(trace["success"]):
        raise ValueError("Only a successful trajectory can be fitted")
    t, q = trace["time_s"], trace["q"]
    if len(t) < 4 or np.any(np.diff(t) <= 0):
        raise ValueError("Spline needs at least four strictly increasing samples")
    degree = 3
    count = min(settings["coefficients"], len(t))
    internal = np.linspace(t[0], t[-1], count - 2)[1:-1]
    knots = np.r_[np.repeat(t[0], 4), internal, np.repeat(t[-1], 4)]
    basis = BSpline(knots, np.eye(count), degree)
    design = basis(t)
    grid = np.linspace(t[0], t[-1], 301)
    curvature = basis(grid, nu=2)
    quadrature = np.full(len(grid), (grid[-1] - grid[0]) / (len(grid) - 1))
    quadrature[[0, -1]] *= .5
    system = design.T @ design / len(t) + settings["smoothing"] * ((curvature.T * quadrature) @ curvature)
    rhs = design.T @ q / len(t)
    derivative = (degree / (knots[4:count + 3] - knots[1:count]))[:, None] * np.diff(np.eye(count), axis=0)
    coefficients = []
    for i in range(q.shape[1]):
        lower, upper = np.full(count, bounds[i, 0]), np.full(count, bounds[i, 1])
        start_next = q[0, i] + trace["dq"][0, i] * (knots[4] - knots[1]) / degree
        lower[[0, 1, -1]] = upper[[0, 1, -1]] = [q[0, i], start_next, q[-1, i]]
        constraint = [LinearConstraint(derivative, -vmax[i], vmax[i])] if np.isfinite(vmax[i]) else []
        result = minimize(lambda x: .5 * x @ system @ x - rhs[:, i] @ x,
                          np.clip(np.interp(np.linspace(t[0], t[-1], count), t, q[:, i]), lower, upper),
                          jac=lambda x: system @ x - rhs[:, i], method="SLSQP",
                          bounds=Bounds(lower, upper), constraints=constraint,
                          options=dict(ftol=1e-12, maxiter=400))
        if not result.success:
            raise RuntimeError(f"Spline fitting failed for {ACTIVE[i]}: {result.message}")
        coefficients.append(result.x)
    spline = BSpline(knots, np.asarray(coefficients).T, degree, extrapolate=False)
    np.testing.assert_allclose(spline(t[[0, -1]]), q[[0, -1]], atol=1e-8, rtol=0.)
    np.testing.assert_allclose(spline(t[0], nu=1), trace["dq"][0], atol=1e-8, rtol=0.)
    return spline


def replay_spline(env, seed, spline, settings):
    period = env.cfg["timestep"] * env.cfg["substeps"]
    horizon = float(spline.t[-1])
    env.cfg["max_steps"] = int(np.ceil((horizon + settings["terminal_hold_s"]) / period))

    def track(current, _):
        t = min(current.d.time + period, horizon)
        q = spline(t)
        dq = spline(t, nu=1) if current.d.time < horizon else np.zeros(6)
        damping = current.m.dof_damping[current.v_indices] + current.cfg["position_kv"]
        target = (current.q + settings["tracking_gain"] * (q - current.q)
                  + damping * dq / current.cfg["position_kp"])
        return current.encode_position(target)

    return rollout(env, seed, track)


def validate(env, trace, urdf_path):
    bounds, vmax, tmax = joint_limits(env, urdf_path)
    peak_speed = np.max(np.abs(trace["dq"]), axis=0)
    peak_torque = np.max(np.abs(trace["torque"]), axis=0)
    peak_actual = np.max(np.abs(trace["actuator_torque"]), axis=0)
    joint = np.any((trace["q"] < bounds[:, 0] - 1e-9) | (trace["q"] > bounds[:, 1] + 1e-9), axis=1)
    joint |= trace["joint_limit"]
    finite = all(np.all(np.isfinite(trace[key])) for key in ("q", "dq", "ddq", "torque", "actuator_torque", "time_s"))
    anchor_drift = float(np.max(np.linalg.norm(trace["anchor"] - trace["anchor"][0], axis=1)))
    collisions = int(np.count_nonzero(trace["collision"]))
    success = bool(trace["success"])
    goal_error = float(trace["distance_m"][-1])
    safe = (finite and not collisions and not np.any(joint) and anchor_drift < 1e-8
            and np.all(peak_speed <= vmax + 1e-9) and np.all(np.maximum(peak_torque, peak_actual) <= tmax + 1e-9)
            and str(trace["status"]) in {"success", "timeout"})
    terminal_speed = (float(trace["tip_speed_m_s"][-1]) if "tip_speed_m_s" in trace else None)
    stop = env.spec.get("terminal_stop")
    settled = None
    if stop:
        n = stop["settle_steps"]
        settled = bool(len(trace["q"]) > n
                       and np.all(trace["distance_m"][-n:] < env.cfg["goal_tolerance_m"])
                       and np.all(trace["tip_speed_m_s"][-n:] <= stop["ee_speed_tolerance_m_s"])
                       and np.all(np.abs(trace["dq"][-n:]) <= stop["joint_speed_tolerance_rad_s"]))
        success = success and settled
    return dict(status=str(trace["status"]), reaching_success=success, validated_success=bool(success and safe),
                constraints_passed=bool(safe), final_goal_error_m=goal_error,
                terminal_stop_verified=settled,
                terminal_tip_speed_m_s=terminal_speed, terminal_max_joint_speed_rad_s=float(np.max(np.abs(trace["dq"][-1]))),
                execution_time_s=float(trace["time_s"][-1]), maximum_joint_velocity_rad_s=float(peak_speed.max()),
                maximum_actuator_torque_nm=float(peak_actual.max()), maximum_estimated_torque_nm=float(peak_torque.max()),
                collision_count=collisions, joint_limit_violations=int(np.count_nonzero(joint)),
                anchor_drift_m=anchor_drift, velocity_limits_rad_s=[float(v) if np.isfinite(v) else None for v in vmax],
                effort_limits_nm=tmax.tolist(), finite_state=bool(finite), sample_interval_s=env.cfg["timestep"] * env.cfg["substeps"])
