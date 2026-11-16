"""Snapshot response from a frozen Fn-only reference, with latched warnings."""

from dataclasses import dataclass
import math

import numpy as np

from .statics import InitialState


@dataclass(frozen=True)
class Redistribution:
    mode: str = "contact1_transfer_weighted"
    site: str = "ATI136"
    force_tolerance_N: float = 0.0

    def __post_init__(self):
        allowed = {
            "contact1_transfer_weighted": ("ATI136",),
            "legacy_site_linear": ("Rpre1", "Rnext2"),
        }
        if self.mode not in allowed or self.site not in allowed[self.mode]:
            raise ValueError("Select contact1_transfer_weighted/ATI136 or legacy_site_linear/Rpre1|Rnext2")
        if not math.isfinite(self.force_tolerance_N) or self.force_tolerance_N < 0:
            raise ValueError("Force tolerance must be finite and nonnegative")


class ContactTracker:
    """Each sample starts from Fn0. Warnings latch until a new tracker is made.

    Inputs are signed anchor-frame reaction increments relative to the
    no-external-wrench baseline. The optional moment is reported as unmapped.
    """

    def __init__(self, initial, redistribution=None):
        if not isinstance(initial, InitialState):
            raise TypeError("A validated InitialState is required")
        self.initial = initial
        self.redistribution = redistribution or Redistribution()
        self._last_time = None
        self._first_alarm = None

    def update(self, time_s, Fh_N, Fv_N, M_Nm=0.0):
        if not all(math.isfinite(v) for v in (time_s, Fh_N, Fv_N, M_Nm)):
            raise ValueError("Time and reaction components must be finite")
        if self._last_time is not None and time_s <= self._last_time:
            raise ValueError("Tracking samples must have strictly increasing times")
        n = np.array(self.initial.fn_native_N, dtype=float)
        if self.redistribution.mode == "contact1_transfer_weighted":
            n += [-Fv_N, Fv_N, Fv_N]
            wall_sum = n[0] + n[2]
            if wall_sum <= 0:
                raise ValueError("Weighted sharing requires Fn1 + Fn3 > 0")
            weights = np.array([0.5*n[0]/wall_sum, 0.5, 0.5*n[2]/wall_sum])
        else:
            index = 0 if self.redistribution.site == "Rpre1" else 2
            n[index] -= Fv_N
            weights = np.array([0.25, 0.5, 0.25])
        # Common shear signs already incorporate the contact-2 frame reversal.
        t = -Fh_N * weights
        margin = self.initial.mu * n - np.abs(t)
        tol = self.redistribution.force_tolerance_N
        normal_loss = n <= tol
        cone_out = (n > tol) & (margin < -tol)
        status = []
        utilization = []
        for i in range(3):
            if normal_loss[i]:
                status.append("NORMAL_CONTACT_LOSS")
                utilization.append(None)
            elif cone_out[i]:
                status.append("OUT_OF_CONE")
                utilization.append(float(abs(t[i])/(self.initial.mu*n[i])) if self.initial.mu > 0 else None)
            else:
                status.append("WITHIN_CONE")
                utilization.append(float(abs(t[i])/(self.initial.mu*n[i])) if self.initial.mu > 0 else None)
        instantaneous = bool(normal_loss.any() or cone_out.any())
        first_alarm = self._first_alarm
        if instantaneous and first_alarm is None:
            first_alarm = float(time_s)
        force_residual = [float(t.sum()+Fh_N), float(n[0]-n[1]+n[2]+Fv_N)]
        result = {
            "time_s": float(time_s), "Fh_N": float(Fh_N), "Fv_N": float(Fv_N),
            "M_Nm": float(M_Nm), "Fn_N": n.tolist(), "Ft_common_N": t.tolist(),
            "margin_N": margin.tolist(), "utilization": utilization, "contact_status": status,
            "warning_instantaneous": instantaneous, "warning_latched": first_alarm is not None,
            "first_alarm_s": first_alarm, "normal_loss": normal_loss.tolist(),
            "cone_out": cone_out.tolist(), "global_force_residual_N": force_residual,
            "unmapped_applied_moment_Nm": float(M_Nm),
            "full_two_body_moment_equilibrium_verified": False,
            "post_alarm_algebraic_demand": first_alarm is not None,
        }
        # Commit temporal state only after the complete snapshot succeeds.
        self._last_time = float(time_s)
        self._first_alarm = first_alarm
        return result
