"""§5 training: fit `ProbeHead` on cached, frozen backbone activations.

Mirrors bdsm's `training/train_head.py` structure in PyTorch idioms: warmup
+ cosine LR, grad clipping, a hard stop on non-finite loss/grad (never
zero-and-continue), periodic holdout eval, and a full `run_config` written
alongside every checkpoint.
"""

import argparse
import json
import logging
import math
import os
import time

import numpy as np
import torch
from torch import nn

from .data import assert_no_overlap, load_cached
from .head import ProbeHead
from .metrics import average_precision

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("train_probe")

SEED = 42


def _lr_lambda(step: int, warmup_steps: int, total_steps: int, end_ratio: float) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return end_ratio + (1 - end_ratio) * cosine


def _check_finite(loss_val: float, grad_norm: float, step: int) -> None:
    if not (math.isfinite(loss_val) and math.isfinite(grad_norm)):
        raise RuntimeError(
            f"non-finite loss/grad at step {step}: loss={loss_val} grad_norm={grad_norm} — "
            "stopping for investigation (never zero-and-continue)"
        )


def evaluate(head: ProbeHead, holdout: dict) -> dict:
    head.eval()
    with torch.no_grad():
        scores = head.score(torch.from_numpy(holdout["x"])).numpy()
    return {
        "n_eval": len(scores),
        "pr_auc": average_precision(holdout["y"], scores),
        "pos_frac": float(holdout["y"].mean()) if len(holdout["y"]) else 0.0,
    }


def save_checkpoint(run_dir: str, step: int, head: ProbeHead, run_config: dict) -> str:
    step_dir = os.path.join(run_dir, f"step_{step:06d}")
    head.save(step_dir)
    with open(os.path.join(step_dir, "config.json"), "w") as f:
        json.dump(run_config, f, indent=1)
    with open(os.path.join(run_dir, "latest.txt"), "w") as f:
        f.write(step_dir)
    return step_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cls-dir", required=True, help="dir with train/ and holdout/ activation caches"
    )
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--peak-lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument(
        "--hidden-dims",
        default="",
        help="comma-separated hidden widths; empty = linear head",
    )
    args = ap.parse_args(argv)

    train = load_cached(os.path.join(args.cls_dir, "train"))
    holdout = load_cached(os.path.join(args.cls_dir, "holdout"))
    assert_no_overlap(train, holdout)
    emb_dim = train["x"].shape[1]
    log.info(f"train: {len(train['x'])} exchanges, holdout: {len(holdout['x'])}, dim {emb_dim}")

    hidden_dims = tuple(int(d) for d in args.hidden_dims.split(",") if d)
    torch.manual_seed(SEED)
    head = ProbeHead(emb_dim, hidden_dims)
    n_trainable = sum(p.numel() for p in head.parameters())
    log.info(f"trainable params: {n_trainable:,} (backbone frozen — cached activations)")

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.peak_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_lambda(step, args.warmup_steps, args.steps, end_ratio=0.1),
    )
    loss_fn = nn.BCEWithLogitsLoss()

    run_dir = os.path.join(args.checkpoint_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    run_config = {
        "run_name": args.run_name,
        "seed": SEED,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "peak_lr": args.peak_lr,
        "warmup_steps": args.warmup_steps,
        "hidden_dims": list(hidden_dims),
        "n_trainable_params": int(n_trainable),
        "n_train": len(train["x"]),
        "n_holdout": len(holdout["x"]),
        "cls_dir": args.cls_dir,
        "training_mode": "cached-activations (frozen backbone precomputed once)",
    }
    log.info(json.dumps(run_config, indent=1))

    rng = np.random.default_rng(SEED)
    n = len(train["x"])
    x_all = torch.from_numpy(train["x"])
    y_all = torch.from_numpy(train["y"])
    order = rng.permutation(n)
    cursor = 0

    t0 = time.time()
    for step in range(args.steps):
        if cursor + args.batch_size > n:
            order = rng.permutation(n)
            cursor = 0
        idx = order[cursor : cursor + args.batch_size]
        cursor += args.batch_size

        head.train()
        optimizer.zero_grad()
        logits = head.logit(x_all[idx])
        loss = loss_fn(logits, y_all[idx])
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_val, gn = float(loss.item()), float(grad_norm)
        _check_finite(loss_val, gn, step)
        if step % 50 == 0 or step == args.steps - 1:
            log.info(f"[{step:6d}/{args.steps}] loss={loss_val:.4f} grad={gn:.3f}")

        if step > 0 and step % args.eval_every == 0:
            m = evaluate(head, holdout)
            log.info(
                "eval@"
                f"{step}: "
                + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})
            )
            save_checkpoint(run_dir, step, head, run_config)

    m = evaluate(head, holdout)
    log.info(
        "final eval: "
        + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})
    )
    save_checkpoint(run_dir, args.steps, head, run_config)
    log.info(f"done in {time.time() - t0:.0f}s: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
