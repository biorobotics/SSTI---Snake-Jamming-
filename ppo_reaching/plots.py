import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from reaching_env import ACTIVE


def trajectory_plots(original, fitted, spline, path, tolerance=.02):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    traces = [("PPO", original)] if fitted is None else [("PPO", original), ("Spline replay", fitted)]
    for label, trace in traces:
        time = trace["time_s"]
        axes[0, 0].plot(time, 100 * trace["distance_m"], label=label)
        axes[0, 1].plot(*trace["tip"].T, label=label)
        for i in range(len(ACTIVE)):
            style = dict(ls="-" if label == "PPO" else "--", color=f"C{i}")
            axes[1, 0].plot(time, np.rad2deg(trace["q"][:, i]), **style)
            axes[1, 1].plot(time, np.rad2deg(trace["dq"][:, i]), **style)
    if spline is not None:
        t = np.linspace(0, spline.t[-1], 301)
        axes[1, 0].plot(t, np.rad2deg(spline(t)), ":", color="#555555", alpha=.45)
        axes[1, 1].plot(t, np.rad2deg(spline(t, nu=1)), ":", color="#555555", alpha=.45)
    axes[0, 0].axhline(100 * tolerance, color="#666666", linestyle=":")
    axes[0, 0].set(xlabel="Time (s)", ylabel="M1-to-goal distance (cm)")
    axes[0, 1].scatter(*original["goal"], marker="*", s=125, color="#8d417a", label="Goal")
    extent = [original["goal"], *[point for _, trace in traces for point in trace["tip"]]]
    for x, y, length, width, angle in original["boxes"]:
        u = np.array([np.cos(angle), np.sin(angle)]) * length / 2
        v = np.array([-np.sin(angle), np.cos(angle)]) * width / 2
        vertices = [np.array([x, y]) + i*u + j*v for i, j in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
        axes[0, 1].add_patch(Polygon(vertices, color="#8c99a7"))
        extent.extend(vertices)
    low, high = np.min(extent, axis=0), np.max(extent, axis=0)
    center = (low + high) / 2
    half_span = max(.03, .64 * np.max(high - low))
    axes[0, 1].set(xlim=center[0] + np.array([-half_span, half_span]),
                  ylim=center[1] + np.array([-half_span, half_span]))
    axes[0, 1].set(xlabel="L (m)", ylabel="T (m)", aspect="equal")
    axes[1, 0].set(xlabel="Time (s)", ylabel="Joint position (deg)",
                   title="Solid PPO · dashed fitted replay · dotted spline reference")
    axes[1, 1].set(xlabel="Time (s)", ylabel="Joint velocity (deg/s)")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    for ax in axes[0]:
        ax.legend()
    fig.savefig(path, dpi=145)
    plt.close(fig)


def fitting_plot(original, spline, path):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
    t = original["time_s"]
    for i, (ax, name) in enumerate(zip(axes.flat, ACTIVE)):
        ax.plot(t, np.rad2deg(original["q"][:, i]), label="PPO")
        ax.plot(t, np.rad2deg(spline(t)[:, i]), "--", label="Spline")
        ax.set(title=name, xlabel="Time (s)", ylabel="Position (deg)")
        ax.grid(alpha=.2)
    axes.flat[0].legend()
    fig.savefig(path, dpi=145)
    plt.close(fig)


def training_plot(root):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for path in sorted(root.glob("*/periodic_evaluation.csv")):
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for ax, key, title in zip(axes, ("mean_return", "success_rate", "mean_final_error_m"),
                                 ("Mean return", "Validated success rate", "Mean final error (m)")):
            ax.plot([int(r["steps"]) for r in rows], [float(r[key]) for r in rows], ".-", label=path.parent.name)
            ax.set(xlabel="Stage environment steps", title=title)
            ax.grid(alpha=.2)
    if axes[0].lines:
        axes[0].legend()
    fig.savefig(root / "training_curves.png", dpi=145)
    plt.close(fig)
