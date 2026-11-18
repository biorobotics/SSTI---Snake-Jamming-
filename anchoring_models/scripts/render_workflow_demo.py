"""Animate the current initialization, normal update and shear overwrite law."""

from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from PIL import Image


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "anchoring_models" / "outputs" / "workflow_demo"
sys.path.insert(0, str(REPO))
from anchoring_models import ContactTracker, Geometry, Redistribution, solve_initial


NORMAL = "#0072B2"
SHEAR = "#D55E00"
REACTION = "#009E73"
ALARM = "#B2182B"
INK = "#182535"
GHOST = "#A2A7AB"
CODE = [
    "INITIALIZE",
    "F0 <- MinMax(q0, tau0, mu)",
    "store Fn0 and Ft0",
    "",
    "FOR EACH SNAPSHOT",
    "read Fh, Fv",
    "Fn <- Fn0 + normal_change(Fv)",
    "Ft <- shear_share(Fh, Fn)",
    "margin <- mu*Fn - abs(Ft)",
    "if any(Fn <= 0 or margin < 0)",
    "    latch_warning()",
]

EQUATIONS = {
    "solve": r"$\min\ \max_i(F_{n i}^{2}+F_{t i}^{2})$",
    "store": r"$\mathbf{F}_{n0}=\mathbf{F}_n^*,\quad\mathbf{F}_{t0}=\mathbf{F}_t^*$",
    "read": r"$\mathbf{f}_{\mathrm{reach}}=(F_h,F_v)$",
    "normal": r"$\mathbf{F}_n=\mathbf{F}_{n0}+[-F_v,F_v,F_v]^{\mathsf{T}}$",
    "shear": r"$\mathbf{F}_t\leftarrow-\frac{F_h}{2}[w_1,1,w_3]^{\mathsf{T}}$",
    "margin": r"$m_i=\mu F_{n i}-|F_{t i}|$",
    "check": r"$m_i=\mu F_{n i}-|F_{t i}|,\quad F_{n i}\leq0\ \mathrm{or}\ m_i<0$",
}


def make_frames(initial):
    n0 = np.asarray(initial.fn_native_N)
    t0 = np.asarray(initial.ft_native_N)*[1, -1, 1]
    frames = []

    def append(case, stage, line, n, t, h=0., v=0., duration=150, state=None):
        frames.append({
            "case": case, "stage": stage, "line": line,
            "Fn": np.asarray(n).tolist(), "Ft": np.asarray(t).tolist(),
            "Fh": float(h), "Fv": float(v), "duration_ms": duration,
            "state": state,
        })

    append("Initialization", "solve", 1, n0, t0, duration=1500)
    append("Initialization", "store", 2, n0, t0, duration=1800)
    amplitude = float(n0[0])
    for case, hmax, vmax in [
        ("Horizontal reaction", 2.3*amplitude, 0.),
        ("Normal unloading", 0., 1.12*amplitude),
    ]:
        append(case, "store", 2, n0, t0, duration=1200)
        previous_n, previous_t = n0.copy(), t0.copy()
        tracker = ContactTracker(initial, Redistribution("contact1_transfer_weighted", "ATI136"))
        for index, fraction in enumerate(np.linspace(0, 1, 23)):
            h, v = hmax*fraction, vmax*fraction
            state = tracker.update(index*.12, float(h), float(v))
            n, t = np.asarray(state["Fn_N"]), np.asarray(state["Ft_common_N"])
            append(case, "read", 5, previous_n, previous_t, h, v)
            append(case, "normal", 6, n, previous_t, h, v)
            append(case, "shear", 7, n, t, h, v,
                   duration=1100 if index == 0 else 150)
            append(case, "margin", 8, n, t, h, v, duration=120)
            alarm = state["warning_instantaneous"]
            append(case, "check", 10 if alarm else 9, n, t, h, v,
                   duration=2600 if alarm else 150, state=state)
            previous_n, previous_t = n, t
            if alarm:
                break
    return frames


def build(initial, frames):
    fig = plt.figure(figsize=(16, 9), dpi=95, facecolor="white")
    fig.suptitle("Min-Max initialization → force response → contact warning",
                 x=.5, y=.974, fontsize=21, color=INK)
    code_ax = fig.add_axes([.025, .15, .315, .735], xlim=(0, 1), ylim=(0, 1))
    code_ax.axis("off")
    code_ax.add_patch(FancyBboxPatch((.0, .0), .995, .99, boxstyle="round,pad=0.005",
                                    facecolor="#F2F5F7", edgecolor="none"))
    code_ax.text(.055, .945, "Algorithm", fontsize=17, weight="bold", va="center", color=INK)
    positions = np.linspace(.84, .105, len(CODE))
    highlight = FancyBboxPatch((.035, .1), .93, .057, boxstyle="round,pad=.006",
                               facecolor="#FFE4A6", edgecolor="none")
    code_ax.add_patch(highlight)
    code_text = []
    for y, line in zip(positions, CODE):
        item = code_ax.text(.065, y, line, family="DejaVu Sans Mono", fontsize=11.7,
                            va="center", color=INK)
        code_text.append(item)

    state_text = fig.text(.035, .107, "", fontsize=14, color=INK, weight="bold")
    case_text = fig.text(.365, .907, "", fontsize=16, color=INK, weight="bold")
    equation_text = fig.text(.66, .47, "", fontsize=18, ha="center", va="center", color=INK)
    fbd = fig.add_axes([.365, .49, .61, .385])
    g = initial.geometry
    points = np.array([[g.left_arm_m, g.h_m], [0., 0.], [-g.right_arm_m, g.h_m]])
    extent = max(g.left_arm_m, g.right_arm_m)
    fbd.set(xlim=(-extent-.045, extent+.14), ylim=(-.056, g.h_m+.085))
    fbd.set_aspect("equal")
    fbd.axis("off")
    fbd.plot([points[0, 0], 0], [points[0, 1], 0], color="#567086", lw=10, solid_capstyle="round")
    fbd.plot([0, points[2, 0]], [0, points[2, 1]], color="#81909D", lw=10, solid_capstyle="round")
    fbd.scatter([0], [0], s=105, facecolors="white", edgecolors=INK, zorder=7)
    for wall in [0, g.h_m]:
        fbd.plot([-extent-.035, extent+.125], [wall, wall], color="#C3C9CE", lw=1.7, zorder=-2)
    labels = ["ATI136", "ATI157", "ATI158"]
    for index, point in enumerate(points):
        offset = .008 if index != 1 else -.012
        fbd.text(point[0], point[1]+offset, labels[index], fontsize=11,
                 ha="center", va="bottom" if index != 1 else "top", color=INK)
    contacts = fbd.scatter(points[:, 0], points[:, 1], s=35, color=INK, zorder=9)
    n0 = np.asarray(initial.fn_native_N)
    t0 = np.asarray(initial.ft_native_N)*[1, -1, 1]
    scale = .0016
    normal_arrows, shear_arrows = [], []
    for index, point in enumerate(points):
        ghost = FancyArrowPatch(tuple(point), tuple(point+[scale*t0[index], 0]),
                                arrowstyle="-|>", mutation_scale=12, lw=1.5,
                                color=GHOST, linestyle="--", zorder=2)
        fbd.add_patch(ghost)
        for collection, color in [(normal_arrows, NORMAL), (shear_arrows, SHEAR)]:
            arrow = FancyArrowPatch(tuple(point), tuple(point), arrowstyle="-|>",
                                    mutation_scale=15, lw=2.7, color=color, zorder=8)
            fbd.add_patch(arrow)
            collection.append(arrow)
    load_origin = points[0]+[.024, .031]
    leader, = fbd.plot([points[0, 0], load_origin[0]], [points[0, 1], load_origin[1]],
                       color=REACTION, lw=1, ls=":")
    external = FancyArrowPatch(tuple(load_origin), tuple(load_origin), arrowstyle="-|>",
                               mutation_scale=19, lw=3.1, color=REACTION, zorder=10)
    fbd.add_patch(external)
    load_label = fbd.text(load_origin[0], load_origin[1]+.014, "Reaction", color=REACTION,
                         ha="center", fontsize=11)

    max_n = max(np.max(frame["Fn"]) for frame in frames)
    low_n = min(-.07*max_n, min(np.min(frame["Fn"]) for frame in frames)*1.3)
    high_n = max_n*1.13
    y_limit = initial.mu*high_n*1.12
    cone_dots = []
    for i in range(3):
        axis = fig.add_axes([.37+i*.205, .14, .176, .265])
        axis.fill([0, high_n, high_n], [0, initial.mu*high_n, -initial.mu*high_n],
                  color="#E2F0E6", zorder=-2)
        axis.plot([0, high_n], [0, initial.mu*high_n], color="#5C8268", lw=1.2)
        axis.plot([0, high_n], [0, -initial.mu*high_n], color="#5C8268", lw=1.2)
        axis.axvspan(low_n, 0, color="#EEEEEE", zorder=-3)
        axis.axhline(0, color="#D7DCDF", lw=.6)
        axis.plot([n0[i]], [t0[i]], marker="o", ms=7, mfc="none", mec=GHOST, mew=1.5)
        dot, = axis.plot([], [], marker="o", ms=8, mfc=REACTION, mec="white", mew=1, ls="none")
        axis.set(xlim=(low_n, high_n), ylim=(-y_limit, y_limit), title=labels[i], xlabel="Fn [N]")
        if i == 0:
            axis.set_ylabel("Ft [N]")
        axis.tick_params(labelsize=9)
        cone_dots.append(dot)

    handles = [Line2D([], [], color=NORMAL, lw=2.6, label="Normal"),
               Line2D([], [], color=SHEAR, lw=2.6, label="Shear"),
               Line2D([], [], color=REACTION, lw=2.6, label="Reaction"),
               Line2D([], [], color=GHOST, lw=1.5, ls="--", label="Initial shear")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .042),
               ncol=4, frameon=False, fontsize=11)
    fig.text(.5, .018, "μ = 0.5 · synthetic reactions at ATI136 · fixed schematic geometry · redistribution moment balance under validation",
             ha="center", fontsize=10, color="#5A636B")

    def update(index):
        frame = frames[index]
        stage = frame["stage"]
        n, t = np.asarray(frame["Fn"]), np.asarray(frame["Ft"])
        highlight.set_y(positions[frame["line"]]-.028)
        for i, item in enumerate(code_text):
            item.set_fontweight("bold" if i == frame["line"] else "normal")
        stage_labels = {"solve": "Full Min-Max optimum", "store": "Store the complete initial state",
                        "read": "Read reaction", "normal": "Update normal forces",
                        "shear": "Overwrite shear forces", "margin": "Compute friction margins",
                        "check": "Check contact state"}
        case_text.set_text(frame["case"]+"  ·  "+stage_labels[stage])
        equation_text.set_text(EQUATIONS[stage])
        state = frame["state"]
        warning = state is not None and state["warning_instantaneous"]
        if warning:
            message = "WARN · contact loss" if any(state["normal_loss"]) else "WARN · friction limit"
        elif stage in ("solve", "store"):
            message = "Initial Fn and Ft retained"
        elif stage == "check":
            message = "Within assumed contacts"
        else:
            message = "Updating snapshot"
        state_text.set_text(message)
        state_text.set_color(ALARM if warning else INK)
        colors = []
        for i in range(3):
            normal_arrows[i].set_visible(n[i] >= 0)
            shear_arrows[i].set_visible(n[i] >= 0 and abs(t[i]) > 1e-8)
            normal_direction = 1 if i == 1 else -1
            normal_arrows[i].set_positions(tuple(points[i]), tuple(points[i]+[0, scale*n[i]*normal_direction]))
            shear_arrows[i].set_positions(tuple(points[i]), tuple(points[i]+[scale*t[i], 0]))
            failed = state is not None and state["contact_status"][i] != "WITHIN_CONE"
            color = ALARM if failed else REACTION
            cone_dots[i].set_data([n[i]], [t[i]])
            cone_dots[i].set_markerfacecolor(color if stage == "check" else SHEAR)
            colors.append(color if failed else INK)
        contacts.set_facecolors(colors)
        load = np.array([frame["Fh"], -frame["Fv"]])
        visible = np.linalg.norm(load) > .01
        external.set_positions(tuple(load_origin), tuple(load_origin+scale*load))
        external.set_visible(visible)
        leader.set_visible(visible)
        load_label.set_visible(visible)

    return fig, update


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()
    initial = solve_initial(Geometry.from_configuration(.65), 2.5, .5)
    frames = make_frames(initial)
    fig, update = build(initial, frames)
    selected = {
        "initial": 1,
        "shear_overwrite": next(i for i, f in enumerate(frames) if f["stage"] == "shear"),
        "friction_warning": next(i for i, f in enumerate(frames) if f["case"] == "Horizontal reaction" and f["state"] and f["state"]["warning_instantaneous"]),
        "contact_loss": len(frames)-1,
    }
    for name, index in selected.items():
        update(index)
        fig.savefig(OUT/f"preview_{name}.png", dpi=105)
    rows = []
    for i, frame in enumerate(frames):
        row = {key: value for key, value in frame.items() if key not in ("Fn", "Ft", "state")}
        row["frame"] = i
        for j in range(3):
            row[f"Fn{j+1}_N"] = frame["Fn"][j]
            row[f"Ft{j+1}_N"] = frame["Ft"][j]
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/"ANIMATION_STATES.csv", index=False)
    audit = {
        "initial": asdict(initial), "frame_count": len(frames),
        "duration_s": sum(f["duration_ms"] for f in frames)/1000,
        "input_type": "synthetic reaction illustration", "application_site": "ATI136",
        "initial_shear_preserved_common_N": (np.asarray(initial.ft_native_N)*[1, -1, 1]).tolist(),
        "shear_during_response": "overwrite from current horizontal reaction and normal-force weights",
        "normal_during_response": "initial normal plus current vertical-load increment",
        "initial_shear_zero_assumed": False,
        "full_response_moment_equilibrium_verified": False,
        "force_arrows": "separate components on snake, shared linear scale, no numeric force labels",
        "load_glyph": "offset for readability, dotted leader marks ATI136 application site",
        "source_sha256": {name: hashlib.sha256((REPO/"anchoring_models"/name).read_bytes()).hexdigest()
                          for name in ("statics.py", "tracking.py")},
    }
    assert np.max(np.abs(initial.ft_native_N)) > 1
    first_shear = frames[selected["shear_overwrite"]]
    np.testing.assert_array_equal(first_shear["Ft"], [0, 0, 0])
    for frame in frames:
        if frame["state"]:
            np.testing.assert_allclose(frame["state"]["global_force_residual_N"], 0, atol=1e-6)
    (OUT/"MODEL_AND_ANIMATION.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if args.preview_only:
        plt.close(fig)
        print(json.dumps({"frames": len(frames), "duration_s": audit["duration_s"]}))
        return
    update(selected["contact_loss"])
    fig.canvas.draw()
    palette = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()).convert(
        "P", palette=Image.Palette.ADAPTIVE, colors=224)
    images = []
    for i in range(len(frames)):
        update(i)
        fig.canvas.draw()
        rgb = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        images.append(rgb.quantize(palette=palette, dither=Image.Dither.NONE))
        if i % 30 == 0:
            print(f"Frame {i+1}/{len(frames)}", flush=True)
    path = OUT/"anchor_pseudocode_free_body.gif"
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=[f["duration_ms"] for f in frames], loop=0, disposal=2, optimize=False)
    with Image.open(path) as saved:
        assert saved.n_frames == len(frames)
        duration = 0
        for i in range(saved.n_frames):
            saved.seek(i)
            saved.load()
            duration += saved.info["duration"]
        assert duration == sum(f["duration_ms"] for f in frames)
    plt.close(fig)
    print(f"Saved {path.name}, {path.stat().st_size/1e6:.2f} MB", flush=True)


if __name__ == "__main__":
    main()
