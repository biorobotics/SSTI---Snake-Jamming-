"""Portable version of the validated legacy Min-Max snapshot equations."""

from dataclasses import asdict, dataclass
import math

import cvxpy as cp
import numpy as np


@dataclass(frozen=True)
class Geometry:
    h_m: float
    left_arm_m: float
    right_arm_m: float

    def __post_init__(self):
        values = np.array([self.h_m, self.left_arm_m, self.right_arm_m], dtype=float)
        if not np.isfinite(values).all() or self.h_m <= 0 or np.any(values[1:] < 0):
            raise ValueError("Geometry requires finite h > 0 and nonnegative arms in metres")

    @classmethod
    def from_configuration(cls, q_rad, link_length_m=0.1021, radius_m=0.0167):
        """Use the existing geometry angle, supplied after feedback conversion."""
        if not all(math.isfinite(v) for v in (q_rad, link_length_m, radius_m)):
            raise ValueError("Geometry inputs must be finite")
        if not 0 <= q_rad <= math.pi or link_length_m <= 0 or radius_m < 0:
            raise ValueError("Require 0 <= q <= pi, length > 0 and radius >= 0")
        phi = (math.pi - q_rad) / 2
        h = link_length_m * math.cos(phi) + 2 * radius_m
        arm = link_length_m * math.sin(phi)
        return cls(h, arm, arm)


@dataclass(frozen=True)
class InitialState:
    tau0_Nm: float
    mu: float
    geometry: Geometry
    fn_native_N: tuple
    ft_native_N: tuple
    solver_status: str
    max_force_residual_N: float
    max_moment_residual_Nm: float
    max_contact_violation_N: float

    def to_dict(self):
        result = asdict(self)
        result["status"] = "FEASIBLE"
        result["joint_reaction_native_N"] = [self.fn_native_N[2], self.ft_native_N[2]]
        result["ft_common_N"] = (np.asarray(self.ft_native_N) * [1, -1, 1]).tolist()
        result["response_fn0_N"] = list(self.fn_native_N)
        result["response_ft0_N"] = [0.0, 0.0, 0.0]
        return result


def solve_initial(geometry, measured_torque_Nm, mu=0.5, audit_tolerance=1e-6):
    """Solve once at the supplied no-wrench initialization snapshot.

    The complete native six-force optimum is preserved. Fn-only transfer is
    performed later by ContactTracker. Infeasibility is explicit.
    """
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be a Geometry instance")
    if not all(math.isfinite(v) for v in (measured_torque_Nm, mu, audit_tolerance)):
        raise ValueError("Torque, mu and audit tolerance must be finite")
    if mu < 0 or audit_tolerance <= 0:
        raise ValueError("Require mu >= 0 and audit tolerance > 0")
    tau = abs(float(measured_torque_Nm))
    fn = cp.Variable(3)
    ft = cp.Variable(3)
    rx = cp.Variable()
    ry = cp.Variable()
    z = cp.Variable(nonneg=True)
    constraints = [
        fn[0] - fn[1] + rx == 0,
        ft[0] - ft[1] + ry == 0,
        fn[2] - rx == 0,
        ft[2] - ry == 0,
        geometry.h_m * fn[0] - geometry.left_arm_m * ft[0] == tau,
        geometry.h_m * fn[2] + geometry.right_arm_m * ft[2] == tau,
        fn >= 0,
        ft <= mu * fn,
        ft >= -mu * fn,
    ]
    for i in range(3):
        constraints.append(cp.square(fn[i]) + cp.square(ft[i]) <= z)
    problem = cp.Problem(cp.Minimize(z), constraints)
    try:
        problem.solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        problem.solve(solver=cp.SCS, eps=1e-8, max_iters=30000)
    if problem.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
        raise RuntimeError("INFEASIBLE")
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Initial solver failed with status {problem.status}")
    n = np.asarray(fn.value, dtype=float)
    t = np.asarray(ft.value, dtype=float)
    force = np.array([n[0]-n[1]+rx.value, t[0]-t[1]+ry.value,
                      n[2]-rx.value, t[2]-ry.value], dtype=float)
    moment = np.array([geometry.h_m*n[0]-geometry.left_arm_m*t[0]-tau,
                       geometry.h_m*n[2]+geometry.right_arm_m*t[2]-tau])
    contact_violation = max(0.0, float(np.max(-n)), float(np.max(np.abs(t)-mu*n)))
    if not np.isfinite(np.r_[n, t, force, moment]).all():
        raise RuntimeError("Initial solver returned nonfinite values")
    if max(np.max(np.abs(force)), np.max(np.abs(moment)), contact_violation) > audit_tolerance:
        raise RuntimeError("Initial solution failed the independent physical audit")
    return InitialState(tau, float(mu), geometry, tuple(n), tuple(t), str(problem.status),
                        float(np.max(np.abs(force))), float(np.max(np.abs(moment))),
                        contact_violation)
