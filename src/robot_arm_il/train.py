"""Train a policy on demonstrations: `ra-train --policy act --demos data/demos_clean.npz`."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .dataset import (
    chunk_targets,
    load_demoset,
    normalisation_stats,
    split_episodes,
    stack_history,
)
from .policies import build_policy, pick_device


def make_tensors(demos, chunk: int, stats: dict, device: torch.device, history: int = 1):
    actions, mask = chunk_targets(demos, chunk)
    norm = (demos.observations - stats["obs"]["mean"]) / stats["obs"]["std"]
    obs = torch.as_tensor(stack_history(demos, norm, history))
    act = torch.as_tensor((actions - stats["act"]["mean"]) / stats["act"]["std"])
    return obs.to(device), act.float().to(device), torch.as_tensor(mask).to(device)


@torch.no_grad()
def evaluate(model, tensors, batch: int = 2048) -> float:
    model.eval()
    obs, act, mask = tensors
    total, count = 0.0, 0.0
    for i in range(0, len(obs), batch):
        pred = model(obs[i : i + batch])
        m = mask[i : i + batch][:, : pred.shape[1]]
        err = (pred - act[i : i + batch, : pred.shape[1]]).abs().mean(-1)
        total += float((err * m).sum())
        count += float(m.sum())
    return total / count


def save_checkpoint(path, model, kind, kwargs, stats, extra):
    torch.save(
        {
            "kind": kind,
            "kwargs": kwargs,
            "state_dict": model.state_dict(),
            "stats": {k: {a: v.tolist() for a, v in s.items()} for k, s in stats.items()},
            **extra,
        },
        path,
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", choices=["bc", "act"], required=True)
    p.add_argument("--demos", default="data/demos_clean.npz")
    p.add_argument("--n-train-episodes", type=int, default=0, help="0 = all training episodes")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--chunk", type=int, help="actions per prediction (default: ACT 20, MLP 1)")
    p.add_argument("--history", type=int, default=1, help="MLP only: observations per input")
    p.add_argument("--hidden", type=int, default=256, help="MLP only: layer width")
    p.add_argument("--layers", type=int, default=3, help="MLP only: hidden layers")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="checkpoint path, e.g. models/act.pt")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device()
    demos = load_demoset(args.demos)
    split = split_episodes(demos.n_episodes)
    train_ids = split["train"]
    if args.n_train_episodes:
        train_ids = np.sort(rng.permutation(train_ids)[: args.n_train_episodes])
    train, val = demos.subset(train_ids), demos.subset(split["val"])
    stats = {
        "obs": normalisation_stats(train.observations),
        "act": normalisation_stats(train.actions),
    }
    if args.policy == "act" and args.history != 1:
        p.error("--history is only available for the MLP")
    chunk = args.chunk or (1 if args.policy == "bc" else 20)
    kwargs: dict = {"chunk": chunk}
    if args.policy == "bc":
        kwargs |= {"history": args.history, "hidden": args.hidden, "layers": args.layers}
    model = build_policy(args.policy, **kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    tr = make_tensors(train, chunk, stats, device, args.history)
    va = make_tensors(val, chunk, stats, device, args.history)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = max(1, len(tr[0]) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch
    )
    print(
        f"{args.policy}: {n_params:,} params, {len(train_ids)} train episodes "
        f"({len(tr[0])} frames), device {device}",
        flush=True,
    )

    history, best, t0 = [], float("inf"), time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(tr[0]), device=device)
        run = 0.0
        for s in range(steps_per_epoch):
            idx = perm[s * args.batch_size : (s + 1) * args.batch_size]
            out = model.loss(tr[0][idx], tr[1][idx][:, :chunk], tr[2][idx][:, :chunk])
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            run += float(out["l1"])
        val_l1 = evaluate(model, va)
        history.append({"epoch": epoch + 1, "train_l1": run / steps_per_epoch, "val_l1": val_l1})
        if val_l1 < best:
            best = val_l1
            save_checkpoint(args.out, model, args.policy, kwargs, stats, {"epoch": epoch + 1})
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"epoch {epoch + 1:3d}  train L1 {history[-1]['train_l1']:.4f}  "
                f"val L1 {val_l1:.4f}  ({time.time() - t0:.0f}s)",
                flush=True,
            )
    log = {
        "policy": args.policy,
        "params": n_params,
        "train_episodes": int(len(train_ids)),
        "demos": args.demos,
        "best_val_l1": best,
        "seconds": round(time.time() - t0, 1),
        "args": vars(args),
        "history": history,
    }
    Path(args.out).with_suffix(".json").write_text(json.dumps(log, indent=1))
    print(json.dumps({k: v for k, v in log.items() if k not in ("history", "args")}))


if __name__ == "__main__":
    main()
