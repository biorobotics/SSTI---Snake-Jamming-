"""Compact formula panel, matching the supplied reference aspect ratio."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"


def draw_formula_panel(axis):
    # Joint reactions are eliminated from the four force equations in the
    # original two-body model. The two moment equations remain unchanged.
    rows = [
        (0.958, "Initial state · Min-Max", 16, "bold"),
        (0.875,
         r"$\min_{\mathbf{F}_n,\,\mathbf{F}_t}\ \max_{i=1,2,3}"
         r"\left(F_{n i}^{2}+F_{t i}^{2}\right),\qquad W_{\mathrm{ext},0}=0$",
         22, "normal"),
        (0.781,
         r"$\mathrm{s.t.}\quad F_{n1}-F_{n2}+F_{n3}=0,\qquad F_{t1}-F_{t2}+F_{t3}=0$",
         21, "normal"),
        (0.707,
         r"$hF_{n1}-\ell_LF_{t1}=\tau_0,\qquad hF_{n3}+\ell_RF_{t3}=\tau_0$",
         21, "normal"),
        (0.632,
         r"$F_{n i}\geq0,\qquad |F_{t i}|\leq\mu F_{n i},\qquad i=1,2,3$",
         21, "normal"),
        (0.557,
         r"$\mathbf{F}_{n0}=\mathbf{F}_n^{*},\qquad \mathbf{F}_{t0}=\mathbf{F}_t^{*}$",
         22, "normal"),
        (0.472, "During reaching · prescribed load sharing · common shear frame", 15, "bold"),
        (0.397,
         r"$\mathbf{F}_n(t)=\mathbf{F}_{n0}+\Delta\mathbf{F}_n"
         r"\!\left(\mathbf{f}_{\mathrm{reach}}(t)\right)$",
         22, "normal"),
        (0.317,
         r"$w_1(t)=\frac{F_{n1}(t)}{F_{n1}(t)+F_{n3}(t)},\qquad w_3(t)=1-w_1(t)$",
         22, "normal"),
        (0.231,
         r"$\mathbf{F}_t(t)\leftarrow-\frac{F_h(t)}{2}\left[w_1(t),\ 1,\ w_3(t)\right]^{\mathsf{T}}$",
         23, "normal"),
        (0.144,
         r"$m_i(t)=\mu F_{n i}(t)-|F_{t i}(t)|$",
         22, "normal"),
        (0.065,
         r"$\mathrm{Warn\ if}\quad\exists i\!:\ "
         r"F_{n i}(t)\leq0\quad\mathrm{or}\quad m_i(t)<0$",
         21, "normal"),
    ]
    axis.plot([0.045, 0.955], [0.514, 0.514], lw=0.65, color="#BBBBBB")
    texts = []
    for y, label, size, weight in rows:
        texts.append(axis.text(
            0.5, y, label, ha="center", va="center", color="#111111",
            fontsize=size, fontweight=weight,
        ))
    return texts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(8.8, 6.1), facecolor="white")
    axis = fig.add_axes([0, 0, 1, 1], xlim=(0, 1), ylim=(0, 1))
    axis.set_axis_off()
    texts = draw_formula_panel(axis)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    previous = None
    for item in texts:
        bounds = item.get_window_extent(renderer)
        assert bounds.x0 >= 10 and bounds.x1 <= fig.bbox.width - 10, item.get_text()
        assert bounds.y0 >= 10 and bounds.y1 <= fig.bbox.height - 10, item.get_text()
        if previous is not None:
            assert bounds.y1 + 3 < previous.y0, item.get_text()
        previous = bounds
    fig.savefig(OUT / "compact_initial_reaching.png", dpi=300, facecolor="white")
    svg_path = OUT / "compact_initial_reaching.svg"
    fig.savefig(svg_path, facecolor="white")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(
        encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)
    print("Saved compact 8.8 x 6.1 inch formula panel.")


if __name__ == "__main__":
    main()
