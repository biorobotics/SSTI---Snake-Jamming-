"""Fixed-base planar rod dynamics reused from the measured-state NE workflow.

Zero in-plane gravity, no extra link contacts, joint friction or rotor inertia.
The returned wrench is robot-on-base, about the fixed proximal J1 origin.
"""

import numpy as np


JOINTS_BASE_TO_TIP = ("RUC-016", "RUC-001", "RUB-017", "RUC-009", "RUC-012", "RUC-011")
NAMED_AXIS_SIGNS = (-1, 1, -1, 1, -1, 1)


def _states(*values):
    arrays = [np.atleast_2d(np.asarray(value, dtype=float)) for value in values]
    shape = arrays[0].shape
    if len(shape) != 2 or not shape[0] or not shape[1]:
        raise ValueError("Joint arrays must be nonempty [samples, joints]")
    if any(array.shape != shape or not np.isfinite(array).all() for array in arrays):
        raise ValueError("Joint arrays must have equal shapes and finite values")
    return arrays


def _links(n, lengths_m, masses_kg):
    lengths = np.full(n, 0.11) if lengths_m is None else np.asarray(lengths_m, dtype=float)
    masses = np.full(n, 0.5) if masses_kg is None else np.asarray(masses_kg, dtype=float)
    for value in (lengths, masses):
        if value.shape != (n,) or not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError("Link lengths and masses must be finite positive vectors")
    return lengths, masses


def newton_euler(q, dq, ddq, lengths_m=None, masses_kg=None):
    """Return inverse joint torques and robot-on-base [Fx, Fy, Mz]."""
    q, dq, ddq = _states(q, dq, ddq)
    samples, n = q.shape
    lengths, masses = _links(n, lengths_m, masses_kg)
    theta = np.cumsum(q, axis=1)
    omega = np.cumsum(dq, axis=1)
    alpha = np.cumsum(ddq, axis=1)
    unit = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    tangent = np.stack([-np.sin(theta), np.cos(theta)], axis=-1)
    segments = lengths[None, :, None] * unit
    segment_acceleration = lengths[None, :, None] * (
        -omega[:, :, None]**2 * unit + alpha[:, :, None] * tangent)
    joints = np.concatenate([np.zeros((samples, 1, 2)), np.cumsum(segments, axis=1)], axis=1)
    joint_acceleration = np.concatenate([
        np.zeros((samples, 1, 2)), np.cumsum(segment_acceleration, axis=1)], axis=1)
    com = joints[:, :-1] + segments/2
    com_acceleration = joint_acceleration[:, :-1] + segment_acceleration/2
    inertia = masses * lengths**2 / 12
    inertial_force = masses[None, :, None] * com_acceleration

    def cross(a, b):
        return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]

    force_on_base = -inertial_force.sum(axis=1)
    moment_on_base = -(inertia[None, :]*alpha + cross(com, inertial_force)).sum(axis=1)
    tau = np.empty((samples, n))
    for j in range(n):
        tau[:, j] = (inertia[None, j:]*alpha[:, j:] + cross(
            com[:, j:]-joints[:, j:j+1], inertial_force[:, j:])).sum(axis=1)
    return tau, np.column_stack([force_on_base, moment_on_base])


def mass_matrix(q, lengths_m=None, masses_kg=None):
    """Lagrange mass matrix from COM and angular Jacobians."""
    q, = _states(q)
    samples, n = q.shape
    lengths, masses = _links(n, lengths_m, masses_kg)
    theta = np.cumsum(q, axis=1)
    tangents = np.stack([-np.sin(theta), np.cos(theta)], axis=-1)
    matrix = np.zeros((samples, n, n))
    for i in range(n):
        jacobian = np.zeros((samples, 2, n))
        for j in range(i+1):
            jacobian[:, :, j] = (
                (lengths[None, j:i, None]*tangents[:, j:i]).sum(axis=1)
                + lengths[i]/2*tangents[:, i])
        angular = np.zeros(n)
        angular[:i+1] = 1
        matrix += masses[i]*np.einsum("bki,bkj->bij", jacobian, jacobian)
        matrix += masses[i]*lengths[i]**2/12*np.outer(angular, angular)[None]
    return matrix


def forward_dynamics(q, dq, torque_Nm, lengths_m=None, masses_kg=None,
                     sign_mask=None, yaw_rad=0.0):
    """Solve instantaneous ddq without integrating trajectories or using history.

    q=0 for link 1 points along +X. All arrays use physical base-to-tip order.
    Apply the same configured named-axis signs to q, dq and feedback torque.
    The yaw rotation expresses the base wrench in the selected anchor frame.
    """
    q, dq, torque = _states(q, dq, torque_Nm)
    n = q.shape[1]
    mask = np.ones(n) if sign_mask is None else np.asarray(sign_mask, dtype=float)
    if mask.shape != (n,) or not np.isin(mask, [-1, 1]).all() or not np.isfinite(yaw_rad):
        raise ValueError("Use one +/-1 sign per joint and a finite yaw")
    q, dq, torque = q*mask, dq*mask, torque*mask
    matrix = mass_matrix(q, lengths_m, masses_kg)
    bias, _ = newton_euler(q, dq, np.zeros_like(q), lengths_m, masses_kg)
    ddq = np.linalg.solve(matrix, (torque-bias)[..., None])[..., 0]
    rebuilt, wrench = newton_euler(q, dq, ddq, lengths_m, masses_kg)
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    wrench[:, :2] = wrench[:, :2] @ np.array([[c, s], [-s, c]])
    return {
        "ddq_model_rad_s2": ddq, "robot_on_base_wrench": wrench,
        "torque_model_Nm": torque, "torque_reconstructed_Nm": rebuilt,
        "dynamics_residual_Nm": rebuilt-torque,
    }
