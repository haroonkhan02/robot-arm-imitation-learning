"""Failure analysis of the rollout reports: `ra-analyse` writes reports/analysis.{md,json}."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .dataset import DemoSet, load_demoset, split_episodes
from .policy_eval import STAGES

OUTCOMES = (*STAGES, "settled", "success")
LABELS = {
    "reached": "never reached the cube",
    "grasped": "never closed on the cube",
    "lifted": "closed but never lifted",
    "at_goal": "lifted but never at the goal",
    "released": "at the goal but never released",
    "settled": "released but not resting on the goal",
    "success": "success",
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval of a success rate."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (0.0 if k == 0 else float(mid - half)), (1.0 if k == n else float(mid + half))


def outcome_counts(rows: list[dict]) -> dict[str, int]:
    c = Counter("success" if r["success"] else r["failed_at"] for r in rows)
    return {o: c.get(o, 0) for o in OUTCOMES}


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Probability that a positive scores higher than a negative (ties count half)."""
    pos, neg = scores[positive], scores[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def train_layouts(demos: DemoSet, n_train: int, seed: int) -> np.ndarray:
    """Cube and goal xy (E, 4) of the episodes `ra-train --n-train-episodes` selected."""
    ids = split_episodes(demos.n_episodes)["train"]
    if n_train:
        ids = np.sort(np.random.default_rng(seed).permutation(ids)[:n_train])
    starts = np.concatenate([[0], demos.episode_ends[:-1]])[ids]
    first = demos.observations[starts]
    return np.concatenate([first[:, 6:8], first[:, 11:13]], axis=1)


def coverage(rows: list[dict], layouts: np.ndarray) -> dict:
    """How far each test layout is from its nearest training layout, and whether that matters."""
    cube = np.array([r["cube_xy"] for r in rows])
    goal = np.array([r["goal_xy"] for r in rows])
    ok = np.array([r["success"] for r in rows])
    d_cube = np.linalg.norm(cube[:, None] - layouts[None, :, :2], axis=2).min(1)
    d_goal = np.linalg.norm(goal[:, None] - layouts[None, :, 2:], axis=2).min(1)
    return {
        "auc_cube": auc(d_cube, ~ok),
        "auc_goal": auc(d_goal, ~ok),
        "median_cube_cm": {
            "success": float(np.median(d_cube[ok]) * 100) if ok.any() else None,
            "failure": float(np.median(d_cube[~ok]) * 100) if (~ok).any() else None,
        },
        "median_goal_cm": {
            "success": float(np.median(d_goal[ok]) * 100) if ok.any() else None,
            "failure": float(np.median(d_goal[~ok]) * 100) if (~ok).any() else None,
        },
    }


def load_reports(folder: Path) -> dict[str, dict]:
    return {p.stem.removeprefix("eval_"): json.loads(p.read_text()) for p in folder.glob("eval_*")}


def row(name: str, rep: dict) -> dict:
    rows = rep["episodes_detail"]
    k, n = sum(r["success"] for r in rows), len(rows)
    lo, hi = wilson(k, n)
    return {
        "name": name,
        "episodes": n,
        "success": k / n,
        "ci95": [lo, hi],
        "lifted": float(np.mean([r["lifted"] for r in rows])),
        "outcomes": outcome_counts(rows),
    }


def seeds_of(reports: dict, base: str, suffix: str = "") -> list[float]:
    """Success rates of a run and its _s1, _s2 repeats; `suffix` selects e.g. "_newest"."""
    names = [base + suffix] + [f"{base}_s{s}{suffix}" for s in (1, 2)]
    return [reports[n]["success_rate"] for n in names if n in reports]


def table(rows: list[dict]) -> list[str]:
    head = "| Run | Success | 95 % CI | Lifted | " + " | ".join(OUTCOMES[:-1]) + " |"
    lines = [head, "|---|---|---|---|" + "---|" * (len(OUTCOMES) - 1)]
    for r in rows:
        lo, hi = r["ci95"]
        counts = " | ".join(str(r["outcomes"][o]) for o in OUTCOMES[:-1])
        rates = f"{r['success']:.1%} | {lo:.1%}–{hi:.1%} | {r['lifted']:.1%}"
        lines.append(f"| {r['name']} | {rates} | {counts} |")
    return lines


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports", default="reports")
    p.add_argument("--demos", default="data/demos_clean.npz")
    args = p.parse_args(argv)
    folder = Path(args.reports)
    reports = load_reports(folder)
    demos = load_demoset(args.demos)

    rows = {k: row(k, v) for k, v in sorted(reports.items())}
    seeds = {
        b + x: seeds_of(reports, b, x)
        for b, x in (
            ("bc_clean", ""),
            ("bc_k20", ""),
            ("bc_k20", "_newest"),
            ("act_clean", ""),
            ("act_clean", "_newest"),
            ("act_n10", ""),
            ("act_n20", ""),
            ("act_n50", ""),
        )
    }
    cover = {}
    for n in (10, 20, 50):
        for s in (0, 1, 2):
            name = f"act_n{n}" + (f"_s{s}" if s else "")
            if name in reports:
                layouts = train_layouts(demos, n, s)
                cover[name] = coverage(reports[name]["episodes_detail"], layouts)
    noise = {}
    for f in sorted(folder.glob("noise_expert_*.json")):
        level = f.stem.rsplit("_", 1)[1]
        noise[level] = {
            "expert": json.loads(f.read_text())["success_rate"],
            "act_clean": reports[f"noise_act_clean_{level}"]["success_rate"],
            "act_noisy": reports[f"noise_act_noisy_{level}"]["success_rate"],
        }
    out = {"runs": rows, "seeds": seeds, "coverage": cover, "noise": noise}
    (folder / "analysis.json").write_text(json.dumps(out, indent=1))

    lines = ["# Rollout analysis", "", "Generated by `ra-analyse` from `reports/eval_*.json`.", ""]
    lines += ["Columns after Lifted count the failed episodes by the first stage they missed:"]
    lines += [f"- {o}: {LABELS[o]}" for o in OUTCOMES[:-1]] + [""]
    lines += table(list(rows.values())) + ["", "## Repeat seeds", ""]
    lines += ["| Run | Success per seed | Mean |", "|---|---|---|"]
    for b, v in seeds.items():
        if v:
            lines.append(f"| {b} | {', '.join(f'{x:.1%}' for x in v)} | {np.mean(v):.1%} |")
    lines += ["", "## Distance to the nearest training layout", ""]
    lines += ["AUC 0.5 = distance says nothing about failure, 1.0 = failures are always farther."]
    lines += ["", "| Run | AUC cube | AUC goal | Cube cm, success / failure |", "|---|---|---|---|"]
    for k, c in cover.items():
        m = c["median_cube_cm"]
        fmt = "/".join("–" if v is None else f"{v:.1f}" for v in (m["success"], m["failure"]))
        lines.append(f"| {k} | {c['auc_cube']:.2f} | {c['auc_goal']:.2f} | {fmt} |")
    lines += ["", "## Execution noise", "", "| Noise, rad | Expert | ACT clean | ACT noisy |"]
    lines += ["|---|---|---|---|"]
    for k, v in noise.items():
        lines.append(f"| {k} | {v['expert']:.1%} | {v['act_clean']:.1%} | {v['act_noisy']:.1%} |")
    (folder / "analysis.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {folder / 'analysis.md'}")


if __name__ == "__main__":
    main()
