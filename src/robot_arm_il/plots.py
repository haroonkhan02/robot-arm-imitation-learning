"""Figures from the reports: `ra-plots` writes reports/figures/*.png."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .analysis import LABELS, OUTCOMES, load_reports, train_layouts  # noqa: E402
from .dataset import load_demoset  # noqa: E402

COLOURS = {
    "reached": "#b0b7c3",
    "grasped": "#8e9aaf",
    "lifted": "#d1495b",
    "at_goal": "#edae49",
    "released": "#f2d492",
    "settled": "#66a5ad",
    "success": "#2e7d5b",
}
OUTCOME_RUNS = [
    ("bc_clean", "MLP"),
    ("bc_h4", "MLP, history 4"),
    ("bc_h8", "MLP, history 8"),
    ("bc_h20", "MLP, history 20"),
    ("bc_big", "MLP, 3.2M params"),
    ("bc_k5", "MLP, chunk 5"),
    ("bc_k20", "MLP, chunk 20"),
    ("bc_k20_newest", "MLP, chunk 20, newest"),
    ("act_k1", "ACT, chunk 1"),
    ("act_k5", "ACT, chunk 5"),
    ("act_clean", "ACT, chunk 20"),
    ("act_clean_newest", "ACT, chunk 20, newest"),
]


def learning_curves(models: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for name, label, colour in (
        ("bc_clean", "MLP (next action)", "#d1495b"),
        ("bc_k20", "MLP chunk 20", "#edae49"),
        ("act_clean", "ACT chunk 20", "#2e7d5b"),
    ):
        path = models / f"{name}.json"
        if path.exists():
            hist = json.loads(path.read_text())["history"]
            ax.plot([h["epoch"] for h in hist], [h["val_l1"] for h in hist], label=label, c=colour)
    ax.set(xlabel="epoch", ylabel="validation L1 (normalised)", yscale="log")
    ax.set_title("Validation loss is not comparable across horizons", loc="left", fontsize=10)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=150)
    plt.close(fig)


def outcomes(reports: dict, out: Path) -> None:
    runs = [(k, v) for k, v in OUTCOME_RUNS if k in reports]
    fig, ax = plt.subplots(figsize=(8, 0.36 * len(runs) + 1.5))
    for i, (key, _) in enumerate(runs):
        rows = reports[key]["episodes_detail"]
        left = 0.0
        for o in OUTCOMES:
            n = sum(("success" if r["success"] else r["failed_at"]) == o for r in rows) / len(rows)
            ax.barh(i, n * 100, left=left, color=COLOURS[o], label=LABELS[o] if i == 0 else None)
            left += n * 100
    ax.set_yticks(range(len(runs)), [label for _, label in runs])
    ax.invert_yaxis()
    ax.set(xlabel="episodes, %", xlim=(0, 100))
    ax.spines[["top", "right"]].set_visible(False)
    handles, names = ax.get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "outcomes.png", dpi=150)
    plt.close(fig)


def demo_sweep(reports: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    sizes = [10, 20, 50]
    means = []
    for n in sizes:
        v = [
            reports[k]["success_rate"] * 100
            for k in (f"act_n{n}", f"act_n{n}_s1", f"act_n{n}_s2")
            if k in reports
        ]
        ax.scatter([n] * len(v), v, c="#2e7d5b", alpha=0.6, s=24)
        means.append(np.mean(v))
    ax.plot(sizes, means, c="#2e7d5b", label="ACT, mean of seeds")
    ax.set(xscale="log", xlabel="training demonstrations", ylabel="success, %", ylim=(0, 104))
    ax.set_xticks(sizes, sizes)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "demo_sweep.png", dpi=150)
    plt.close(fig)


def noise(analysis: dict, out: Path) -> None:
    levels = sorted(analysis["noise"], key=float)
    x = [0.0] + [float(v) for v in levels]
    zero = {"expert": 1.0, "act_clean": 1.0, "act_noisy": 1.0}
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for key, label, colour in (
        ("expert", "Scripted expert", "#555"),
        ("act_noisy", "ACT, noisy demonstrations", "#2e7d5b"),
        ("act_clean", "ACT, clean demonstrations", "#d1495b"),
    ):
        y = [zero[key]] + [analysis["noise"][v][key] for v in levels]
        ax.plot(x, np.array(y) * 100, marker="o", ms=4, label=label, c=colour)
    ax.set(xlabel="joint noise while executing, rad", ylabel="success, %", ylim=(0, 104))
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "noise.png", dpi=150)
    plt.close(fig)


def workspace(reports: dict, demos, out: Path) -> None:
    keys = [("act_n10", 10), ("act_n20", 20)]
    keys = [(k, n) for k, n in keys if k in reports]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.8), squeeze=False)
    for ax, (key, n) in zip(axes[0], keys, strict=True):
        rows = reports[key]["episodes_detail"]
        cube = np.array([r["cube_xy"] for r in rows])
        ok = np.array([r["success"] for r in rows])
        ax.scatter(*cube[~ok].T, s=12, c="#d1495b", label="failed")
        ax.scatter(*cube[ok].T, s=12, c="#2e7d5b", label="succeeded")
        ax.scatter(*train_layouts(demos, n, 0)[:, :2].T, s=70, marker="x", c="k", label="training")
        ax.set(title=f"{n} demonstrations", xlabel="cube x, m", ylabel="cube y, m")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "workspace.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports", default="reports")
    p.add_argument("--models", default="models")
    p.add_argument("--demos", default="data/demos_clean.npz")
    args = p.parse_args(argv)
    folder = Path(args.reports)
    out = folder / "figures"
    out.mkdir(parents=True, exist_ok=True)
    reports = load_reports(folder)
    learning_curves(Path(args.models), out)
    outcomes(reports, out)
    demo_sweep(reports, out)
    noise(json.loads((folder / "analysis.json").read_text()), out)
    workspace(reports, load_demoset(args.demos), out)
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
