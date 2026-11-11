import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from .data import MelodyDataset, grouped_split
from .labels import UNKNOWN
from .model import CorpusAdversary, MelodyVAE, VAEConfig, vae_loss

IGNORE = -100


def beta_schedule(step: int, warmup: int, max_beta: float, cycles: int = 0):
    if cycles > 0:
        period = max(warmup // cycles, 1)
        return max_beta * min(1.0, step % period / (0.5 * period))
    return max_beta * min(1.0, step / max(warmup, 1))


def lambda_schedule(step: int, warmup: int, max_lambda: float):
    p = min(1.0, step / max(warmup, 1))
    return max_lambda * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


def adversary_targets(ds: MelodyDataset, field: str):
    if field not in ds.fields:
        raise SystemExit(f"adv_field {field!r} not in dataset; have {ds.fields}")
    names = ds.names(field)
    codes = np.asarray(ds.codes(field), dtype=np.int64)
    if UNKNOWN in names:
        codes = np.where(codes == names.index(UNKNOWN), IGNORE, codes)
    counts = np.bincount(codes[codes >= 0], minlength=len(names)).astype(np.float64)
    weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1) / len(names), 0.0)
    return (torch.from_numpy(codes), torch.from_numpy(weights).float(), names)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run(cfg: dict):
    device = cfg.get("device") or pick_device()
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    ds = MelodyDataset(cfg["data"], augment_transpose=cfg.get("augment_transpose", 3))
    val_frac, split_seed = (cfg.get("val_frac", 0.05), cfg.get("split_seed", 0))
    if ds.source_file is None:
        raise SystemExit("dataset has no source_file; re-run preprocess.py to get a leak-free split")
    tr_idx, va_idx = grouped_split(ds.source_file, val_frac, split_seed)
    train_ds, val_ds = (Subset(ds, tr_idx), Subset(ds.eval_view(), va_idx))
    print(
        f"split by source file: {len(tr_idx)} train / {len(va_idx)} val windows from {len(np.unique(ds.source_file))} files",
        flush=True,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=cfg.get("num_workers", 2),
    )
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"])
    model = MelodyVAE(VAEConfig(**cfg.get("model", {}))).to(device)
    params = list(model.parameters())
    max_lambda = float(cfg.get("adv_lambda", 0.0))
    adv, adv_codes, adv_weights, adv_names = (None, None, None, [])
    if max_lambda > 0:
        adv_codes, adv_weights, adv_names = adversary_targets(ds, cfg.get("adv_field", "corpus"))
        adv_codes, adv_weights = (adv_codes.to(device), adv_weights.to(device))
        adv = CorpusAdversary(model.cfg.latent_dim, len(adv_names), cfg.get("adv_hidden", 256)).to(device)
        params += list(adv.parameters())
        print(
            f"corpus adversary on {cfg.get('adv_field', 'corpus')}: {adv_names}, lambda -> {max_lambda} over {cfg.get('adv_warmup_steps', cfg['kl_warmup_steps'])} steps",
            flush=True,
        )
    # TODO try plain adam again
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("wd", 0.01))
    total_steps = cfg["epochs"] * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / max(total_steps, 1)))
    )
    adv_warmup = cfg.get("adv_warmup_steps", cfg["kl_warmup_steps"])
    step, best, lam, adv_ce = (0, float("inf"), 0.0, 0.0)
    log_path = out / "log.jsonl"
    for epoch in range(cfg["epochs"]):
    # print('epoch', epoch)
        model.train()
        if adv is not None:
            adv.train()
        t0 = time.time()
        for x, idx in train_dl:
            x = x.to(device)
            beta = beta_schedule(step, cfg["kl_warmup_steps"], cfg["max_beta"], cfg.get("kl_cycles", 0))
            logits, mu, logvar, z = model(x)
            stats = vae_loss(logits, x, mu, logvar, beta, cfg.get("free_bits", 0.05))
            loss = stats["loss"]
            if adv is not None:
                lam = lambda_schedule(step, adv_warmup, max_lambda)
                y = adv_codes[idx.to(device)]
                ce = F.cross_entropy(adv(z, lam), y, weight=adv_weights, ignore_index=IGNORE)
                loss = loss + ce
                adv_ce = ce.item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.get("clip", 1.0))
            opt.step()
            sched.step()
            step += 1
        val = evaluate(model, val_dl, device, adv, adv_codes, len(adv_names))
        record = {
            "epoch": epoch,
            "beta": round(beta, 4),
            "train_recon": round(stats["recon"].item(), 3),
            "train_kl": round(stats["kl"].item(), 3),
            "active_dims": int(stats["active_dims"]),
            "secs": round(time.time() - t0, 1),
            **val,
        }
        if adv is not None:
            record["lambda"] = round(lam, 4)
            record["train_adv_ce"] = round(adv_ce, 4)
            record["adv_chance"] = round(1.0 / len(adv_names), 4)
        print(json.dumps(record), flush=True)
        with log_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        blob = {"model": model.state_dict(), "cfg": cfg}
        if adv is not None:
            blob["adv"] = adv.state_dict()
            blob["adv_names"] = adv_names
        torch.save(blob, out / "last.pt")
        if val["val_recon"] < best:
            best = val["val_recon"]
            torch.save(blob, out / "best.pt")
    return out / "best.pt"


@torch.no_grad()
def evaluate(model, loader, device: str, adv=None, adv_codes=None, n_classes: int = 0):
    model.eval()
    if adv is not None:
        adv.eval()
    tot_recon, tot_kl, correct, count, batches = (0.0, 0.0, 0, 0, 0)
    hit = torch.zeros(max(n_classes, 1), device=device)
    seen = torch.zeros(max(n_classes, 1), device=device)
    for x, idx in loader:
        x = x.to(device)
        logits, mu, logvar, _ = model(x)
        stats = vae_loss(logits, x, mu, logvar, beta=1.0)
        tot_recon += stats["recon"].item()
        tot_kl += stats["kl"].item()
        correct += (logits.argmax(-1) == x).sum().item()
        count += x.numel()
        batches += 1
        if adv is not None:
            y = adv_codes[idx.to(device)]
            keep = y >= 0
            if keep.any():
                pred = adv(mu[keep], 0.0).argmax(-1)
                yk = y[keep]
                hit.index_add_(0, yk, (pred == yk).float())
                seen.index_add_(0, yk, torch.ones_like(yk, dtype=torch.float))
    out = {
        "val_recon": round(tot_recon / max(batches, 1), 3),
        "val_kl": round(tot_kl / max(batches, 1), 3),
        "val_token_acc": round(correct / max(count, 1), 4),
    }
    if adv is not None:
        present = seen > 0
        out["val_adv_bacc"] = round((hit[present] / seen[present]).mean().item(), 4)
    return out
