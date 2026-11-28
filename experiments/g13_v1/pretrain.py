"""Stage 1: learn a compact terrain descriptor by reconstructing masked WF height scans."""

import argparse
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rsl_rl"))

import torch
from torch import nn

from experiments.g13_v1.terrain_scans import HEIGHT_SCALE, TERRAIN_NAMES, make_scan_dataset
from rsl_rl.modules.frozen_terrain_encoder import (
    HEIGHT_CELLS, HEIGHT_COLS, HEIGHT_ROWS, TERRAIN_LATENT_DIM, TerrainEncoder, terrain_encoder_checkpoint,
)


def corrupt(scans, rng, allow_clean):
    """Hide a contiguous patch and scattered points. Retain some deployment-style clean examples."""
    n = len(scans)
    y = torch.arange(HEIGHT_ROWS)
    x = torch.arange(HEIGHT_COLS)
    # Center of each occluded rectangle stays within the scan.
    patch_y = (y - torch.randint(HEIGHT_ROWS, (n, 1), generator=rng)).abs() <= 2
    patch_x = (x - torch.randint(HEIGHT_COLS, (n, 1), generator=rng)).abs() <= 3
    block = (patch_y[:, :, None] & patch_x[:, None, :]).reshape_as(scans)
    mask = block | (torch.rand(scans.shape, generator=rng) < 0.20)
    noise = (2 * torch.rand(scans.shape, generator=rng) - 1) * (0.02 * HEIGHT_SCALE)
    if allow_clean:
        augmented = torch.rand(n, 1, generator=rng) >= 0.15
        mask = mask & augmented
        noise = noise * augmented
    values = (scans + noise).clamp(-HEIGHT_SCALE, HEIGHT_SCALE).masked_fill(mask, 0.0)
    return values, mask


def reconstruction_loss(prediction, target, mask):
    """Learn both scan values and local changes across risers and slopes."""
    error = (prediction - target).square()
    hidden = (error * mask).sum() / mask.sum().clamp_min(1)
    grid_error = (prediction - target).view(-1, HEIGHT_ROWS, HEIGHT_COLS)
    dx = grid_error[:, :, 1:] - grid_error[:, :, :-1]
    dy = grid_error[:, 1:, :] - grid_error[:, :-1, :]
    return hidden + 0.5 * error.mean() + 0.25 * (dx.square().mean() + dy.square().mean())


@torch.no_grad()
def evaluate(encoder, decoder, data, batch_size, device):
    clean, terrain_ids, corrupted, masks = data
    encoder.eval()
    decoder.eval()
    hidden_errors, clean_errors = [], []
    for target, values, mask in zip(clean.split(batch_size), corrupted.split(batch_size), masks.split(batch_size)):
        filled = decoder(encoder(values.to(device), mask.to(device))).cpu()
        decoded_clean = decoder(encoder(target.to(device))).cpu()
        hidden_errors.append(((filled - target).abs() * mask).sum(1))
        clean_errors.append((decoded_clean - target).abs().mean(1))
    hidden_error = torch.cat(hidden_errors)
    clean_error = torch.cat(clean_errors)
    # Compare the identical hidden cells to filling them with that scan's visible mean.
    visible_mean = corrupted.sum(1, keepdim=True) / (~masks).sum(1, keepdim=True).clamp_min(1)
    baseline = ((visible_mean - clean).abs() * masks).sum(1)
    counts = masks.sum(1)
    cm = 100 / HEIGHT_SCALE

    def summarize(selected):
        cells = counts[selected].sum()
        return {"masked_mae_cm": (cm * hidden_error[selected].sum() / cells).item(),
                "mean_fill_mae_cm": (cm * baseline[selected].sum() / cells).item(),
                "clean_mae_cm": (cm * clean_error[selected].mean()).item()}

    return dict(summarize(slice(None)), by_terrain={
        name: summarize(terrain_ids == i) for i, name in enumerate(TERRAIN_NAMES)
    })


def save_checkpoint(path, encoder, args, epoch, metrics):
    checkpoint = terrain_encoder_checkpoint(encoder)
    checkpoint["pretraining"] = {
        "objective": "masked denoising and finite-difference reconstruction",
        "generator": "G13 WF trimesh observation raster",
        "sampling": "balanced active families, stratified levels, 65% edge-biased centers",
        "root_clearance_m": [0.35, 0.70],
        "train_tile_seed": args.seed, "validation_tile_seed": args.seed + 1,
        "training": {k: v for k, v in vars(args).items() if k != "output"},
        "best_epoch": epoch, "selection_metric": "masked_mae_cm", "validation": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), prefix=path.name + ".", delete=False) as tmp:
        try:
            torch.save(checkpoint, tmp)
            tmp.close()
            os.replace(tmp.name, path)
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)


def pretrain(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train = torch.from_numpy(make_scan_dataset(args.samples, args.seed)[0])
    val, ids = (torch.from_numpy(a) for a in make_scan_dataset(args.val_samples, args.seed + 1))
    validation = (val, ids, *corrupt(val, torch.Generator().manual_seed(args.seed + 2), False))
    rng = torch.Generator().manual_seed(args.seed + 3)
    encoder = TerrainEncoder().to(device)
    decoder = nn.Sequential(nn.Linear(TERRAIN_LATENT_DIM, 128), nn.ELU(),
                            nn.Linear(128, 256), nn.ELU(), nn.Linear(256, HEIGHT_CELLS)).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    best, best_epoch = None, 0
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        decoder.train()
        for indices in torch.randperm(len(train), generator=rng).split(args.batch_size):
            target = train[indices]
            values, mask = corrupt(target, rng, True)
            mask = mask.to(device)
            reconstructed = decoder(encoder(values.to(device), mask))
            loss = reconstruction_loss(reconstructed, target.to(device), mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        metrics = evaluate(encoder, decoder, validation, args.batch_size, device)
        score = metrics["masked_mae_cm"]
        if not math.isfinite(score):
            raise RuntimeError("Nonfinite validation error at epoch {}".format(epoch))
        if best is None or score < best["masked_mae_cm"]:
            best, best_epoch = metrics, epoch
            save_checkpoint(args.output, encoder, args, epoch, metrics)
        print("epoch {:3d}/{:d}  val masked MAE {:.3f} cm  clean MAE {:.3f} cm".format(
            epoch, args.epochs, score, metrics["clean_mae_cm"]), flush=True)
    print("Best epoch {} saved to {}".format(best_epoch, args.output))
    print("{:16s} {:>12s} {:>14s} {:>12s}".format("terrain", "masked_cm", "mean_fill_cm", "clean_cm"))
    for name, result in list(best["by_terrain"].items()) + [("all", best)]:
        print("{:16s} {:12.3f} {:14.3f} {:12.3f}".format(name, result["masked_mae_cm"],
                                                          result["mean_fill_mae_cm"], result["clean_mae_cm"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "logs/g13_v1/terrain_encoder.pt",
                        help="Best encoder weights and validation metadata")
    parser.add_argument("--samples", type=int, default=8192, help="Training scans")
    parser.add_argument("--val-samples", type=int, default=1536, help="Scans from separate WF tiles and poses")
    parser.add_argument("--epochs", type=int, default=64, help="Training passes")
    parser.add_argument("--batch-size", type=int, default=128, help="Scans per update")
    parser.add_argument("--lr", type=float, default=0.002, help="Adam learning rate")
    parser.add_argument("--threads", type=int, default=2, help="Torch CPU threads")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--seed", type=int, default=10, help="Seed for map generation, masking and initialization")
    args = parser.parse_args()
    if min(args.samples, args.val_samples) < len(TERRAIN_NAMES):
        parser.error("Each split needs at least {} scans".format(len(TERRAIN_NAMES)))
    if min(args.batch_size, args.epochs, args.threads) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("Specify positive batch size, epochs, threads and finite positive learning rate")
    if not 0 <= args.seed < 2**32 - 3:
        parser.error("Seed must be between 0 and 2**32 - 4")
    pretrain(args)


if __name__ == "__main__":
    main()
