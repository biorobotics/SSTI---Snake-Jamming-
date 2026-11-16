"""CSV entry points for force tracking and optional measured-state dynamics."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dynamics import JOINTS_BASE_TO_TIP, forward_dynamics
from .statics import Geometry, solve_initial
from .tracking import ContactTracker, Redistribution


CHANNELS = ["Fn1", "Fn2", "Fn3", "Ft1", "Ft2", "Ft3"]


def replay(data, initial, redistribution):
    """Preserve every input sample and its optional measured-force columns."""
    required = ["time_s", "Fh_N", "Fv_N"]
    if data.empty or not set(required).issubset(data.columns):
        raise ValueError("Input needs nonempty time_s, Fh_N and Fv_N columns")
    columns = required + (["M_Nm"] if "M_Nm" in data else [])
    if not np.isfinite(data[columns].to_numpy(float)).all():
        raise ValueError("Time and reactions must be finite")
    if np.any(np.diff(data.time_s.to_numpy(float)) <= 0):
        raise ValueError("Input times must be strictly increasing")
    tracker = ContactTracker(initial, redistribution)
    outputs = []
    for _, row in data.iterrows():
        outputs.append(tracker.update(float(row.time_s), float(row.Fh_N),
                                      float(row.Fv_N), float(row.get("M_Nm", 0.0))))
    result = data.copy().reset_index(drop=True)
    for index, channel in enumerate(CHANNELS):
        field, contact = ("Fn_N", index) if index < 3 else ("Ft_common_N", index-3)
        result[f"pred_{channel}_N"] = [item[field][contact] for item in outputs]
    for contact in range(3):
        for key, field in [("margin", "margin_N"), ("utilization", "utilization"),
                           ("status", "contact_status")]:
            result[f"{key}_{contact+1}"] = [item[field][contact] for item in outputs]
    for name in ("warning_instantaneous", "warning_latched", "first_alarm_s",
                 "unmapped_applied_moment_Nm", "full_two_body_moment_equilibrium_verified"):
        result[name] = [item[name] for item in outputs]
    residual = np.array([item["global_force_residual_N"] for item in outputs])
    result["force_residual_h_N"] = residual[:, 0]
    result["force_residual_v_N"] = residual[:, 1]
    return result, outputs


def add_joint_reactions(data, configuration):
    dynamics = dict(configuration.get("dynamics", {}))
    names = dynamics.pop("joint_order", list(JOINTS_BASE_TO_TIP))
    if tuple(names) != JOINTS_BASE_TO_TIP:
        raise ValueError("Joint input must use the documented physical base-to-tip order")
    q = data[[f"q{i}" for i in range(1, 7)]].to_numpy(float)
    dq = data[[f"dq{i}" for i in range(1, 7)]].to_numpy(float)
    tau = data[[f"tau{i}" for i in range(1, 7)]].to_numpy(float)
    calculated = forward_dynamics(q, dq, tau, **dynamics)
    result = data.copy()
    wrench = calculated["robot_on_base_wrench"]
    result["Fh_N"], result["Fv_N"], result["M_Nm"] = wrench.T
    for i in range(6):
        result[f"ddq_model_{i+1}_rad_s2"] = calculated["ddq_model_rad_s2"][:, i]
        result[f"torque_residual_{i+1}_Nm"] = calculated["dynamics_residual_Nm"][:, i]
    return result


def plot_timeline(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey="row")
    for axis, channel in zip(axes.flat, CHANNELS):
        values = data[f"pred_{channel}_N"]
        axis.plot(data.time_s, values.where(~data.warning_latched), color="#D55E00", lw=2)
        axis.plot(data.time_s, values.where(data.warning_latched), color="#D55E00", lw=1.5, ls=":")
        measured = f"measured_{channel}_N"
        if measured in data:
            axis.scatter(data.time_s, data[measured], color="#222222", s=5,
                         alpha=0.5, edgecolors="none")
        alarm = data.first_alarm_s.dropna()
        if len(alarm):
            axis.axvline(alarm.iloc[0], color="#A52332", ls="--", lw=1)
        axis.set_title(channel)
        axis.grid(alpha=0.15)
    axes[0, 0].set_ylabel("Normal force [N]")
    axes[1, 0].set_ylabel("Shear force [N]")
    for axis in axes[1]:
        axis.set_xlabel("Time [s]")
    handles = [Line2D([], [], color="#D55E00", lw=2, label="Model"),
               Line2D([], [], color="#D55E00", ls=":", label="Post-alarm demand"),
               Line2D([], [], color="#A52332", ls="--", label="First alarm")]
    if any(f"measured_{channel}_N" in data for channel in CHANNELS):
        handles.append(Line2D([], [], color="#222222", marker="o", ls="none", label="Data"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.14, top=0.94, hspace=0.28, wspace=0.12)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Offline Min-Max and contact tracking")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--joint-input", action="store_true")
    args = parser.parse_args()
    configuration = json.loads(args.config.read_text(encoding="utf-8"))
    setting = configuration["initial"]
    if "geometry" in setting:
        geometry = Geometry(**setting["geometry"])
    else:
        geometry = Geometry.from_configuration(
            setting["q_rad"], setting.get("link_length_m", 0.1021), setting.get("radius_m", 0.0167))
    initial = solve_initial(geometry, setting["measured_torque_Nm"], configuration.get("mu", 0.5))
    redistribution = Redistribution(**configuration["redistribution"])
    data = pd.read_csv(args.input, float_precision="round_trip")
    if args.joint_input:
        if redistribution.mode != "legacy_site_linear" or redistribution.site != "Rnext2":
            raise ValueError("Fixed-base joint replay uses legacy_site_linear at Rnext2")
        data = add_joint_reactions(data, configuration)
    result, states = replay(data, initial, redistribution)
    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output/"tracking.csv", index=False)
    (args.output/"initial_state.json").write_text(
        json.dumps(initial.to_dict(), indent=2), encoding="utf-8")
    summary = {
        "samples": len(result), "first_alarm_s": states[-1]["first_alarm_s"],
        "redistribution": asdict(redistribution),
        "maximum_global_force_residual_N": float(np.max(np.abs(
            result[["force_residual_h_N", "force_residual_v_N"]].to_numpy()))),
        "maximum_abs_unmapped_moment_Nm": float(result.unmapped_applied_moment_Nm.abs().max()),
        "full_two_body_moment_equilibrium_verified": False,
        "configuration": configuration,
    }
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    plot_timeline(result, args.output/"six_channel_timeline.png")
    print(json.dumps({"samples": len(result), "first_alarm_s": states[-1]["first_alarm_s"],
                      "output": str(args.output)}, indent=2))
