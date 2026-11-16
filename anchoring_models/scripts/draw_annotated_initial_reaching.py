"""Add complete English symbol definitions below the compact model panel."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from draw_compact_initial_reaching import draw_formula_panel


OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"

LEFT = [
    (r"$i=1,2,3$", "Contact 1/2/3 = ATI136/157/158.\n1 and 3 share a wall. 2 is opposite."),
    (r"$t$", "Time during reaching [s]."),
    (r"$F_{n i}$", "Normal contact force [N].\nPositive in compression."),
    (r"$F_{t i}$", "Signed tangential contact force [N]."),
    (r"$\mathbf{F}_n,\mathbf{F}_t$", "Three-component force vectors [N],\nordered by contacts 1, 2, 3."),
    (r"$h,\ell_L,\ell_R$", "Geometry-derived moment arms [m].\nL and R denote left and right."),
    (r"$\tau_0$", "Initial measured joint-torque\nmagnitude [N m]."),
    (r"$\mu$", "Coulomb friction coefficient\n(dimensionless, configurable)."),
    (r"$W_{\mathrm{ext},0}$", "Initial external planar wrench.\nZero applied force and moment."),
    (r"$0,\ *$", "Subscript 0 = initial reference.\nSuperscript * = optimizer output."),
]

RIGHT = [
    (r"$\mathbf{F}_{n0}$", "Frozen optimized normal-force\nreference [N]."),
    (r"$\mathbf{F}_{t0}$", "Initial shear set to zero after\noptimization [N]."),
    (r"$\mathbf{f}_{\mathrm{reach}}$", "Reaction applied at an anchor site\n(Fh, Fv) [N]."),
    (r"$F_h,F_v$", "Horizontal and vertical reaction\ncomponents [N]."),
    (r"$\Delta\mathbf{F}_n$", "Normal-force increment from the\nsite/direction response rule [N]."),
    (r"$w_1,w_3$", "Unitless within-wall weights.\nEach wall receives Fh/2."),
    (r"$m_i$", "Friction margin [N].\nNegative values exceed the cone."),
    (r"$|\cdot|,\ {}^{\mathsf{T}}$", "Absolute value and vector transpose."),
    (r"$\exists i,\ \mathrm{or}$", "At least one contact satisfies\neither warning condition."),
    (r"$\min,\max,\mathrm{s.t.}$", "Minimize, largest value, subject to.\nSquared-force objective units [N²]."),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(8.8, 14), facecolor="white")
    formula = fig.add_axes([0, 0.55, 1, 6.1/14], xlim=(0, 1), ylim=(0, 1))
    formula.set_axis_off()
    texts = draw_formula_panel(formula)

    legend = fig.add_axes([0, 0.07, 1, 0.45], xlim=(0, 1), ylim=(0, 1))
    legend.set_axis_off()
    texts.append(legend.text(0.5, 0.977, "Symbols and units", ha="center", va="center",
                             fontsize=16, fontweight="bold"))
    legend.plot([0.03, 0.97], [0.936, 0.936], color="#BBBBBB", lw=0.65)
    groups = []
    for entries, symbol_x, definition_x, right_edge in [
        (LEFT, 0.025, 0.18, 0.493),
        (RIGHT, 0.515, 0.70, 0.985),
    ]:
        previous = None
        for row, (symbol, definition) in enumerate(entries):
            y = 0.876 - row * 0.087
            symbol_text = legend.text(symbol_x, y, symbol, va="center", fontsize=13.5)
            definition_text = legend.text(definition_x, y, definition, va="center",
                                          fontsize=11.8, linespacing=1.15)
            texts.extend([symbol_text, definition_text])
            groups.append((symbol_text, definition_text, right_edge, previous))
            previous = (symbol_text, definition_text)

    texts.append(fig.text(
        0.5, 0.055,
        r"$\mathbf{F}_t^{\mathrm{common}}="
        r"[F_{t1}^{\mathrm{local}},-F_{t2}^{\mathrm{local}},F_{t3}^{\mathrm{local}}]^{\mathsf{T}}$",
        ha="center", va="center", fontsize=16,
    ))
    texts.append(fig.text(
        0.5, 0.029,
        "Initial shear uses local signs. Reaching shear uses common signs.",
        ha="center", va="center", fontsize=11.8,
    ))
    texts.append(fig.text(
        0.5, 0.011,
        "Fn-only transfer and load sharing are provisional. Reaching-time moment consistency awaits validation.",
        ha="center", va="center", fontsize=10.5, color="#555555",
    ))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in texts:
        box = item.get_window_extent(renderer)
        assert box.x0 >= 8 and box.x1 <= fig.bbox.width - 8, item.get_text()
        assert box.y0 >= 5 and box.y1 <= fig.bbox.height - 5, item.get_text()
    for symbol, definition, right_edge, previous in groups:
        symbol_box = symbol.get_window_extent(renderer)
        definition_box = definition.get_window_extent(renderer)
        assert symbol_box.x1 + 6 < definition_box.x0, symbol.get_text()
        assert definition_box.x1 <= right_edge * fig.bbox.width, definition.get_text()
        if previous is not None:
            upper_bottom = min(item.get_window_extent(renderer).y0 for item in previous)
            assert max(symbol_box.y1, definition_box.y1) + 3 < upper_bottom
    fig.savefig(OUT / "compact_initial_reaching_annotated.png", dpi=240, facecolor="white")
    svg_path = OUT / "compact_initial_reaching_annotated.svg"
    fig.savefig(svg_path, facecolor="white")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(
        encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)
    print("Saved annotated model PNG and SVG with complete symbol definitions.")


if __name__ == "__main__":
    main()
