"""Train day and week images as a regression on 5-day excess.

residual: train the day tower first, freeze it, then fit a weekly residual
on week-end rows only. The kept week epoch is the one whose validation
top-group daily excess is at least the day tower's, with the higher Sharpe.
concat and day keep the old rule: highest validation Spearman.
"""
from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cnn import build_model

REMOTE = Path("/storage/server/144server/cq/ChenSiwei/k_pictures_gray_body")
WEEK_REMOTE = Path("/storage/server/144server/cq/ChenSiwei/k_pictures_gray_fuse_week")
OUTPUT_REMOTE = Path("/storage/server/144server/cq/ChenSiwei/k_pictures_gray_fuse_residual")
WINDOW_KEY = "I20"
TARGET_COL = "future_ret"


def normalize_window_key(window_key) -> str:
    """Accept I20 / I60 or the bar count 20 / 60."""
    if isinstance(window_key, str):
        key = window_key.strip().upper()
        if key in {"I20", "20"}:
            return "I20"
        if key in {"I60", "60"}:
            return "I60"
    else:
        try:
            n = int(window_key)
        except (TypeError, ValueError):
            n = None
        if n == 20:
            return "I20"
        if n == 60:
            return "I60"
    raise ValueError(f"窗口只能是 I20、I60、20 或 60，收到 {window_key!r}")


def image_hw(cfg, window_key: str) -> tuple[int, int, int]:
    key = normalize_window_key(window_key)
    window = int(getattr(cfg.image.windows, key))
    height = int(cfg.image.height)
    width = window * int(cfg.image.px_per_day)
    channels = int(getattr(cfg.image, "channels", 1))
    return channels, height, width


def resolve_window_dir(paths: dict, cfg, window_key: str = WINDOW_KEY) -> Path:
    window_key = normalize_window_key(window_key)
    pointer = Path(paths["images"]) / f"{window_key}_LOCATION.txt"
    if pointer.exists():
        p = Path(pointer.read_text(encoding="utf-8").strip())
        if (p / "images.npy").exists():
            return p
    remote = Path(getattr(cfg.paths, "images_remote", REMOTE)) / window_key
    if (remote / "images.npy").exists():
        return remote
    local = Path(paths["images"]) / window_key
    if (local / "images.npy").exists():
        return local
    raise FileNotFoundError(
        f"找不到 {window_key}/images.npy。"
        f"应在 {remote}。这一版要重新生成灰度图，不要读 RGB 那份 images.npy。"
    )


def _writable_dir(preferred: Path, fallback: Path) -> Path:
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".write_probe"
        probe.write_bytes(b"ok")
        probe.unlink()
        return preferred
    except Exception as e:
        print(f"{preferred} 不可写 ({e}), fallback {fallback}")
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def resolve_model_dir(paths: dict, cfg, window_key: str = WINDOW_KEY) -> Path:
    window_key = normalize_window_key(window_key)
    remote_root = Path(getattr(cfg.paths, "output_remote", OUTPUT_REMOTE))
    return _writable_dir(remote_root / "models" / window_key, Path(paths["models"]) / window_key)


def resolve_results_dir(paths: dict, cfg) -> Path:
    remote_root = Path(getattr(cfg.paths, "output_remote", OUTPUT_REMOTE))
    return _writable_dir(remote_root / "results", Path(paths["results"]))


def _seed_ckpts(model_dir: Path) -> list[Path]:
    """Final ensemble files. Skips ``seed_k_day.pt`` and in-progress copies."""
    return sorted(
        p for p in model_dir.glob("seed_*.pt")
        if ".running." not in p.name and not p.stem.endswith("_day")
    )


class IndexedImageDataset(Dataset):
    """Rows of a memmap npy, shape (3, H, W). Scalar mean/std. Target is float excess."""

    def __init__(self, images: np.ndarray, indices: np.ndarray, targets: np.ndarray,
                 mean: float = 0.0, std: float = 1.0):
        self.images = images
        self.indices = np.asarray(indices, dtype=np.int64)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.mean = float(mean)
        self.std = float(std) if std > 0 else 1.0

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        x = self.images[int(self.indices[i])].astype(np.float32)
        x = (x - self.mean) / self.std
        y = float(self.targets[i])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


class PairedImageDataset(Dataset):
    """One daily image and the weekly image of the last completed week."""

    def __init__(self, day_images, week_images, frame_idx, day_rows, week_rows,
                 targets, day_mean, day_std, week_mean, week_std, read_week: bool = True):
        self.day_images = day_images
        self.week_images = week_images
        self.frame_idx = np.asarray(frame_idx, dtype=np.int64)
        self.day_rows = np.asarray(day_rows, dtype=np.int64)
        self.week_rows = np.asarray(week_rows, dtype=np.int64)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.day_mean = float(day_mean)
        self.day_std = float(day_std) if day_std > 0 else 1.0
        self.week_mean = float(week_mean)
        self.week_std = float(week_std) if week_std > 0 else 1.0
        self.read_week = bool(read_week)

    def __len__(self) -> int:
        return len(self.frame_idx)

    def __getitem__(self, i: int):
        j = int(self.frame_idx[i])
        day = self.day_images[int(self.day_rows[j])].astype(np.float32)
        day = torch.from_numpy((day - self.day_mean) / self.day_std)
        if self.read_week:
            week = self.week_images[int(self.week_rows[j])].astype(np.float32)
            week = torch.from_numpy((week - self.week_mean) / self.week_std)
        else:
            week = torch.zeros_like(day)
        y = torch.tensor(float(self.targets[i]), dtype=torch.float32)
        return day, week, y


def fit_pixel_stats_indexed(images: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    take = min(20_000, len(indices))
    sub = rng.choice(indices, take, replace=False)
    sample = np.asarray(images[sub], dtype=np.float32)
    return float(sample.mean()), float(sample.std())


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _spearman(pred: np.ndarray, y: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if pred.size < 3:
        return float("nan")
    pr = np.argsort(np.argsort(pred)).astype(np.float64)
    yr = np.argsort(np.argsort(y)).astype(np.float64)
    pr -= pr.mean()
    yr -= yr.mean()
    denom = float(np.sqrt((pr * pr).sum() * (yr * yr).sum()))
    if denom == 0.0:
        return float("nan")
    return float((pr * yr).sum() / denom)


def _winsor_bounds(y_train: np.ndarray, cfg) -> tuple[float, float]:
    qs = getattr(cfg.cnn, "winsor_quantiles", None)
    if qs is None:
        q_lo, q_hi = 0.01, 0.99
    else:
        q_lo, q_hi = float(qs[0]), float(qs[1])
    lo = float(np.quantile(y_train, q_lo))
    hi = float(np.quantile(y_train, q_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
        lo = float(np.min(y_train))
        hi = float(np.max(y_train))
    return lo, hi


def _run_epoch(model, loader, device, criterion, y_mean: float, y_std: float,
               y_lo: float, y_hi: float, optimiser=None, desc: str = "",
               collect: bool = False):
    """Loss is on the winsorized then z-scored excess. Logged mse uses that same clip."""
    train_mode = optimiser is not None
    model.train(train_mode)
    total_se = 0.0
    correct = 0
    seen = 0
    preds: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    it = tqdm(loader, desc=desc or ("train" if train_mode else "valid"),
              leave=False, mininterval=2.0)
    for batch in it:
        if len(batch) == 3:
            day, week, yb = batch
            day = day.to(device, non_blocking=True)
            week = week.to(device, non_blocking=True)
            n_here = day.size(0)
            pred_in = (day, week)
        else:
            xb, yb = batch
            xb = xb.to(device, non_blocking=True)
            n_here = xb.size(0)
            pred_in = (xb,)
        y_raw = yb.to(device, non_blocking=True).float().view(-1)
        y_clip = torch.clamp(y_raw, y_lo, y_hi)
        y_z = (y_clip - y_mean) / y_std
        if train_mode:
            optimiser.zero_grad(set_to_none=True)
        pred_z = model(*pred_in).view(-1)
        loss = criterion(pred_z, y_z)
        if train_mode:
            loss.backward()
            optimiser.step()
        pred_raw = pred_z.detach() * y_std + y_mean
        total_se += torch.sum((pred_raw - y_clip) ** 2).item()
        correct += ((pred_raw > 0) == (y_raw > 0)).sum().item()
        seen += n_here
        if collect:
            preds.append(pred_raw.detach().cpu().numpy())
            ys.append(y_raw.detach().cpu().numpy())
    n = max(seen, 1)
    mse = total_se / n
    direction = correct / n
    if collect:
        return mse, direction, np.concatenate(preds), np.concatenate(ys)
    return mse, direction


def _save_ckpt(out_path: Path, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out_path)


def _finite_mask(labels_df: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    y = labels_df[TARGET_COL].to_numpy(np.float64)
    keep = np.isfinite(y[idx])
    return idx[keep]


def train_one_seed(images, labels_df, cfg, window_key, seed, out_path: Path,
                   num_workers: int = 0, batch_size: int | None = None,
                   pin_memory: bool = False, week_images=None,
                   kind: str | None = None) -> dict:
    rng = np.random.default_rng(seed)
    device = _pick_device()
    window_key = normalize_window_key(window_key)
    if TARGET_COL not in labels_df.columns:
        raise RuntimeError(f"labels.parquet 没有 {TARGET_COL} 列")

    split = labels_df["split"].astype(str)
    train_idx = _finite_mask(labels_df, np.flatnonzero(split == "train"))
    valid_idx = _finite_mask(labels_df, np.flatnonzero(split == "valid"))
    if len(train_idx) == 0 or len(valid_idx) == 0:
        raise RuntimeError("labels 需要 split 列为 train / valid，且 future_ret 为有限值")

    y_all = labels_df[TARGET_COL].to_numpy(np.float32)
    rng.shuffle(train_idx)
    y_train = y_all[train_idx]
    y_lo, y_hi = _winsor_bounds(y_train, cfg)
    y_w = np.clip(y_train, y_lo, y_hi)
    y_mean = float(y_w.mean())
    y_std = float(y_w.std())
    if not np.isfinite(y_std) or y_std <= 0:
        y_std = 1.0

    print(f"[seed {seed}] device={device}  regression target={TARGET_COL}  "
          f"train={len(train_idx)}  valid={len(valid_idx)}", flush=True)
    print(f"[seed {seed}] winsor [{y_lo:.6f}, {y_hi:.6f}]  "
          f"target mean={y_mean:.6f} std={y_std:.6f}  "
          f"（损失用截尾后再标准化；选模型用原始超额的 Spearman）", flush=True)

    from pair_lib import fuse_mode
    mode = fuse_mode(cfg) if kind is None else str(kind)
    if mode == "residual":
        raise RuntimeError("残差模型用 train_residual_seed，不要走 train_one_seed。")
    if mode not in {"day", "concat"}:
        raise RuntimeError(f"train_one_seed 只训 day 或 concat，实际是 {mode}")
    inputs = ["day", "week"] if mode == "concat" else ["day"]
    if "day_row" not in labels_df.columns or "week_row" not in labels_df.columns:
        raise RuntimeError("labels 需要 day_row 和 week_row。先跑配对，不要直接用日 K 的 labels。")
    day_rows = labels_df["day_row"].to_numpy(np.int64)
    week_rows = labels_df["week_row"].to_numpy(np.int64)
    read_week = mode == "concat"
    if read_week and week_images is None:
        raise RuntimeError("fuse.inputs 含 week，但没有周 K 图")
    print(f"[seed {seed}] fitting pixel stats ...  inputs={inputs}", flush=True)
    mean, std = fit_pixel_stats_indexed(images, np.unique(day_rows[train_idx]))
    if read_week:
        w_mean, w_std = fit_pixel_stats_indexed(week_images, np.unique(week_rows[train_idx]))
    else:
        w_mean, w_std = 0.0, 1.0
    print(f"[seed {seed}] day mean={mean:.4f} std={std:.4f}  "
          f"week mean={w_mean:.4f} std={w_std:.4f}", flush=True)

    def _ds(idx):
        return PairedImageDataset(
            images, week_images, idx, day_rows, week_rows, y_all[idx],
            mean, std, w_mean, w_std, read_week=read_week)

    ds_train = _ds(train_idx)
    ds_valid = _ds(valid_idx)
    if batch_size is None:
        batch = 32 if window_key == "I60" else 64
    else:
        batch = int(batch_size)
    pin = bool(pin_memory) and device.type == "cuda"
    n_train_batches = int(np.ceil(len(ds_train) / max(batch, 1)))
    print(f"[seed {seed}] batch={batch}  train_batches≈{n_train_batches}  "
          f"pin_memory={pin}", flush=True)
    dl_train = DataLoader(ds_train, batch_size=batch, shuffle=True,
                          num_workers=num_workers, pin_memory=pin)
    dl_valid = DataLoader(ds_valid, batch_size=batch, shuffle=False,
                          num_workers=num_workers, pin_memory=pin)

    torch.manual_seed(seed)
    print(f"[seed {seed}] building model ...  mode={mode}", flush=True)
    if not hasattr(cfg, "cnn"):
        root = getattr(getattr(cfg, "project", None), "root", ".")
        raise AttributeError(f"{root}/config.yaml 没有 cnn 段。")
    model = build_model(window_key, cfg, kind=mode).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=float(cfg.cnn.optimizer.lr))
    criterion = nn.MSELoss()

    max_epochs = int(cfg.cnn.max_epochs)
    best_ic = -float("inf")
    best_val = float("nan")
    best_val_dir = float("nan")
    best_epoch = 0
    best_state = None
    hist = []
    t0 = time.time()
    running_path = out_path.with_name(out_path.stem + ".running.pt")
    print(f"[seed {seed}] fixed {max_epochs} epochs, no early stop, "
          f"keep highest valid Spearman IC", flush=True)
    for epoch in range(1, max_epochs + 1):
        print(f"[seed {seed}] start epoch {epoch}/{max_epochs}", flush=True)
        tr_mse, tr_dir = _run_epoch(
            model, dl_train, device, criterion, y_mean, y_std, y_lo, y_hi, optimiser,
            desc=f"seed{seed} ep{epoch} train")
        va_mse, va_dir, va_pred, va_y = _run_epoch(
            model, dl_valid, device, criterion, y_mean, y_std, y_lo, y_hi, None,
            desc=f"seed{seed} ep{epoch} valid", collect=True)
        va_ic = _spearman(va_pred, va_y)
        hist.append({
            "epoch": epoch,
            "train_mse": tr_mse, "train_dir": tr_dir,
            "valid_mse": va_mse, "valid_dir": va_dir, "valid_ic": va_ic,
        })
        improved = np.isfinite(va_ic) and va_ic > best_ic
        if improved:
            best_ic = va_ic
            best_val = va_mse
            best_val_dir = va_dir
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        mark = "  best" if improved else ""
        print(f"[seed {seed}] ep {epoch:02d}/{max_epochs}  "
              f"train mse={tr_mse:.6f} dir={tr_dir:.3f}  "
              f"valid mse={va_mse:.6f} dir={va_dir:.3f} ic={va_ic:.4f}  "
              f"best_epoch={best_epoch}{mark}", flush=True)
        payload = {
            "state_dict": best_state,
            "mean": mean, "std": std,
            "week_mean": w_mean, "week_std": w_std,
            "fuse_inputs": inputs,
            "fuse_mode": mode,
            "y_mean": y_mean, "y_std": y_std,
            "winsor_lo": y_lo, "winsor_hi": y_hi,
            "seed": seed, "window": window_key,
            "task": "regression", "target": TARGET_COL,
            "history": hist, "best_epoch": best_epoch,
            "best_val_mse": best_val, "best_val_dir": best_val_dir,
            "best_val_ic": best_ic,
            "elapsed_s": time.time() - t0,
            "incomplete": True,
        }
        _save_ckpt(running_path, payload)
        print(f"[seed {seed}] wrote {running_path.name} after epoch {epoch}", flush=True)

    elapsed = time.time() - t0
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = max_epochs
    _save_ckpt(out_path, {
        "state_dict": best_state,
        "mean": mean, "std": std,
        "week_mean": w_mean, "week_std": w_std,
        "fuse_inputs": inputs,
        "fuse_mode": mode,
        "y_mean": y_mean, "y_std": y_std,
        "winsor_lo": y_lo, "winsor_hi": y_hi,
        "seed": seed, "window": window_key,
        "task": "regression", "target": TARGET_COL,
        "history": hist, "best_epoch": best_epoch,
        "best_val_mse": best_val, "best_val_dir": best_val_dir,
        "best_val_ic": best_ic,
        "epochs": len(hist), "elapsed_s": elapsed,
    })
    if running_path.exists():
        running_path.unlink()
    print(f"[seed {seed}] saved {out_path}  best_epoch={best_epoch}/{len(hist)}  "
          f"val_ic={best_ic:.4f} val_mse={best_val:.6f} val_dir={best_val_dir:.4f}  {elapsed:.1f}s",
          flush=True)
    return {
        "seed": seed, "best_epoch": best_epoch, "best_val_mse": best_val,
        "best_val_dir": best_val_dir, "best_val_ic": best_ic,
        "epochs": len(hist), "elapsed_s": elapsed, "path": str(out_path),
    }


def _image_dir_ok(path: Path) -> bool:
    return (path / "images.npy").is_file() and (path / "labels.parquet").is_file()


def resolve_day_dir(cfg, window_key: str) -> Path:
    window_key = normalize_window_key(window_key)
    path = Path(getattr(cfg.paths, "images_remote", REMOTE)) / window_key
    if not _image_dir_ok(path):
        raise FileNotFoundError(
            f"日 K 不在 {path}。这一支直接读 k_pictures_gray_body，不重新画日 K。"
        )
    return path


def resolve_week_dir(paths: dict, cfg, window_key: str) -> Path:
    window_key = normalize_window_key(window_key)
    root = Path(getattr(cfg.paths, "week_images_remote", WEEK_REMOTE))
    if root.name in {"k_pictures_gray_body", "k_pictures_gray_week",
                     "k_pictures_rgb_I20", "k_pictures_gray_reg"}:
        raise RuntimeError(
            f"week_images_remote={root} 指到了已有实验。周 K 要写到 k_pictures_gray_fuse_week。"
        )
    remote = root / window_key
    if _image_dir_ok(remote):
        return remote
    pointer = Path(paths["week_images"]) / f"{window_key}_LOCATION.txt"
    if pointer.exists():
        pointed = Path(pointer.read_text(encoding="utf-8").strip())
        if _image_dir_ok(pointed):
            return pointed
    local = Path(paths["week_images"]) / window_key
    if _image_dir_ok(local):
        return local
    raise FileNotFoundError(
        f"找不到周 K 图。先跑 03，写到 {remote}。不要读 k_pictures_gray_week。"
    )


def _paired_label_file(paths: dict, cfg, window_key: str) -> Path:
    window_key = normalize_window_key(window_key)
    remote = Path(getattr(cfg.paths, "output_remote", OUTPUT_REMOTE)) / window_key
    return _writable_dir(remote, Path(paths["results"]) / window_key) / "paired_labels.parquet"


def load_paired_frame(cfg, paths, window_key: str):
    """Daily images, weekly images, and the joined labels."""
    from pair_lib import align_day_week

    window_key = normalize_window_key(window_key)
    day_dir = resolve_day_dir(cfg, window_key)
    week_dir = resolve_week_dir(paths, cfg, window_key)
    print("day images =", day_dir)
    print("week images =", week_dir)
    day_images = np.load(day_dir / "images.npy", mmap_mode="r")
    week_images = np.load(week_dir / "images.npy", mmap_mode="r")
    day_labels = pd.read_parquet(day_dir / "labels.parquet")
    week_labels = pd.read_parquet(week_dir / "labels.parquet")
    expect = image_hw(cfg, window_key)
    for name, arr in (("day", day_images), ("week", week_images)):
        if arr.ndim != 4 or tuple(arr.shape[1:]) != expect:
            raise ValueError(
                f"期望 {name} images (N,{expect[0]},{expect[1]},{expect[2]})，实际 {arr.shape}"
            )
    if len(day_labels) != len(day_images) or len(week_labels) != len(week_images):
        raise RuntimeError(
            f"labels 和 images 行数不一致 day {len(day_labels)}/{len(day_images)} "
            f"week {len(week_labels)}/{len(week_images)}"
        )
    paired = align_day_week(day_labels, week_labels, cfg)
    out = _paired_label_file(paths, cfg, window_key)
    paired.to_parquet(out, index=False)
    print("paired labels =", out)
    return day_images, week_images, paired


def _torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _keep_week_epoch(mu: float, sh: float, best_mu: float, best_sh: float, day_mu: float) -> bool:
    """Keep a week epoch only when top-group excess stays at least the day tower's.

    Among those epochs, the higher Sharpe wins. Equal Sharpe then prefers the
    higher daily excess.
    """
    if not (np.isfinite(mu) and np.isfinite(sh) and np.isfinite(day_mu)):
        return False
    if mu + 1e-12 < float(day_mu):
        return False
    if not np.isfinite(best_sh):
        return True
    if sh > float(best_sh) + 1e-12:
        return True
    if abs(sh - float(best_sh)) <= 1e-12 and mu > float(best_mu) + 1e-12:
        return True
    return False


def _top_group_mu_sharpe(codes, dates, pred, cfg, market) -> tuple[float, float]:
    """Validation top decile, same 5 overlapping sleeves as the decile chart."""
    from backtest_lib import assign_deciles, sharpe

    n_deciles = int(getattr(cfg.backtest, "deciles", 10))
    frame = pd.DataFrame({
        "code": pd.Series(codes).astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6),
        "date": pd.to_datetime(dates),
        "pred": np.asarray(pred, dtype=np.float64),
    })
    frame = assign_deciles(frame, n_deciles=n_deciles, score_col="pred")
    top = frame.loc[frame["decile"] == n_deciles].copy()
    if top.empty:
        return float("nan"), float("nan")
    top["bucket"] = 0
    baskets = formation_baskets_group(top, "bucket", 0)
    ret_wide, mkt, cal = market
    horizon, n_sleeves = _horizon_sleeves(cfg)
    daily = overlapping_sleeve_daily(
        baskets, ret_wide, mkt, cal,
        horizon=horizon, n_sleeves=n_sleeves, min_sleeves=n_sleeves,
    )
    if daily.empty:
        return float("nan"), float("nan")
    r = daily["daily_excess"].dropna()
    if len(r) < 2:
        return float("nan"), float("nan")
    days = int(getattr(cfg.backtest, "trading_days_per_year", 252))
    return float(r.mean()), float(sharpe(r, days))


def _residual_epoch(model, loader, device, criterion, y_mean, y_std, y_lo, y_hi,
                    optimiser=None, desc: str = "", collect: bool = False):
    """Week tower trains. Day tower stays eval and out of the graph."""
    train_mode = optimiser is not None
    model.day.eval()
    model.week.train(train_mode)
    total_se = 0.0
    correct = 0
    seen = 0
    preds: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    it = tqdm(loader, desc=desc or ("train" if train_mode else "valid"),
              leave=False, mininterval=2.0)
    for batch in it:
        day, week, yb = batch[:3]
        day = day.to(device, non_blocking=True)
        week = week.to(device, non_blocking=True)
        n_here = day.size(0)
        y_raw = yb.to(device, non_blocking=True).float().view(-1)
        y_clip = torch.clamp(y_raw, y_lo, y_hi)
        y_z = (y_clip - y_mean) / y_std
        if train_mode:
            optimiser.zero_grad(set_to_none=True)
        pred_z = model(day, week).view(-1)
        loss = criterion(pred_z, y_z)
        if train_mode:
            loss.backward()
            optimiser.step()
        pred_raw = pred_z.detach() * y_std + y_mean
        total_se += torch.sum((pred_raw - y_clip) ** 2).item()
        correct += ((pred_raw > 0) == (y_raw > 0)).sum().item()
        seen += n_here
        if collect:
            preds.append(pred_raw.detach().cpu().numpy())
            ys.append(y_raw.detach().cpu().numpy())
    n = max(seen, 1)
    mse = total_se / n
    direction = correct / n
    if collect:
        return mse, direction, np.concatenate(preds), np.concatenate(ys)
    return mse, direction


def train_residual_seed(images, week_images, labels_df, cfg, paths, window_key,
                        seed, day_ckpt: Path, out_path: Path,
                        num_workers: int = 0, batch_size: int | None = None,
                        pin_memory: bool = False) -> dict:
    """Freeze the day tower. Fit the week residual on week-end rows only."""
    if not day_ckpt.is_file():
        raise FileNotFoundError(f"没有日塔权重 {day_ckpt}")
    day_ck = _torch_load(day_ckpt, map_location="cpu")
    day_mode = day_ck.get("fuse_mode")
    day_inputs = [str(x) for x in day_ck.get("fuse_inputs", [])]
    if day_mode not in (None, "day") or day_inputs not in ([], ["day"]):
        raise RuntimeError(
            f"{day_ckpt} 不是日塔权重（fuse_mode={day_mode!r} fuse_inputs={day_inputs}）"
        )
    for key in ("state_dict", "mean", "std", "y_mean", "y_std", "winsor_lo", "winsor_hi"):
        if key not in day_ck:
            raise RuntimeError(f"{day_ckpt} 缺少 {key}")

    device = _pick_device()
    window_key = normalize_window_key(window_key)
    if "anchor" not in labels_df.columns:
        raise RuntimeError("配对表没有 anchor。周塔只在 date==anchor 的行上更新。")
    split = labels_df["split"].astype(str)
    train_idx = _finite_mask(labels_df, np.flatnonzero(split == "train"))
    valid_idx = _finite_mask(labels_df, np.flatnonzero(split == "valid"))
    is_anchor = labels_df["date"].eq(labels_df["anchor"]).to_numpy()
    train_anchor = train_idx[is_anchor[train_idx]]
    if len(train_anchor) == 0 or len(valid_idx) == 0:
        raise RuntimeError(
            f"残差阶段没有样本 train_anchor={len(train_anchor)} valid={len(valid_idx)}"
        )

    y_mean = float(day_ck["y_mean"])
    y_std = float(day_ck["y_std"]) if float(day_ck["y_std"]) > 0 else 1.0
    y_lo = float(day_ck["winsor_lo"])
    y_hi = float(day_ck["winsor_hi"])
    day_mean = float(day_ck["mean"])
    day_std = float(day_ck["std"]) if float(day_ck["std"]) > 0 else 1.0
    y_all = labels_df[TARGET_COL].to_numpy(np.float32)
    day_rows = labels_df["day_row"].to_numpy(np.int64)
    week_rows = labels_df["week_row"].to_numpy(np.int64)
    print(f"[seed {seed}] residual  device={device}  "
          f"train_anchor={len(train_anchor)}  valid={len(valid_idx)}  "
          f"day_ckpt={day_ckpt.name}", flush=True)
    print(f"[seed {seed}] week tower uses day z-score  "
          f"mean={y_mean:.6f} std={y_std:.6f}", flush=True)
    print(f"[seed {seed}] fitting week pixel stats ...", flush=True)
    w_mean, w_std = fit_pixel_stats_indexed(week_images, np.unique(week_rows[train_idx]))
    print(f"[seed {seed}] week mean={w_mean:.4f} std={w_std:.4f}", flush=True)

    def _ds(idx):
        return PairedImageDataset(
            images, week_images, idx, day_rows, week_rows, y_all[idx],
            day_mean, day_std, w_mean, w_std, read_week=True)

    if batch_size is None:
        batch = 32 if window_key == "I60" else 64
    else:
        batch = int(batch_size)
    pin = bool(pin_memory) and device.type == "cuda"
    dl_train = DataLoader(_ds(train_anchor), batch_size=batch, shuffle=True,
                          num_workers=num_workers, pin_memory=pin)
    dl_valid = DataLoader(_ds(valid_idx), batch_size=batch, shuffle=False,
                          num_workers=num_workers, pin_memory=pin)
    valid_codes = labels_df["code"].to_numpy()[valid_idx]
    valid_dates = labels_df["date"].to_numpy()[valid_idx]

    torch.manual_seed(seed)
    print(f"[seed {seed}] building residual model ...", flush=True)
    model = build_model(window_key, cfg, kind="residual").to(device)
    model.day.load_state_dict(day_ck["state_dict"])
    for p in model.day.parameters():
        p.requires_grad_(False)
    model.day.eval()
    optimiser = torch.optim.Adam(model.week.parameters(), lr=float(cfg.cnn.optimizer.lr))
    criterion = nn.MSELoss()

    print(f"[seed {seed}] validation top-group baseline from the frozen day tower ...", flush=True)
    market = load_backtest_market(cfg, paths)
    _, _, base_pred, base_y = _residual_epoch(
        model, dl_valid, device, criterion, y_mean, y_std, y_lo, y_hi, None,
        desc=f"seed{seed} day-base", collect=True)
    day_mu, day_sh = _top_group_mu_sharpe(valid_codes, valid_dates, base_pred, cfg, market)
    day_ic = _spearman(base_pred, base_y)
    if not (np.isfinite(day_mu) and np.isfinite(day_sh)):
        raise RuntimeError(
            "验证集上算不出日塔最高一组的日均超额。检查行情文件和验证集日期。"
        )
    print(f"[seed {seed}] day baseline  top_mu={day_mu:.6f}  top_sharpe={day_sh:.4f}  "
          f"ic={day_ic:.4f}", flush=True)

    max_epochs = int(cfg.cnn.max_epochs)
    best_mu = day_mu
    best_sh = day_sh
    best_ic = day_ic
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    hist = []
    t0 = time.time()
    running_path = out_path.with_name(out_path.stem + ".running.pt")
    print(f"[seed {seed}] fixed {max_epochs} epochs, week head starts at 0, "
          f"keep a later epoch only if top-group excess >= day and Sharpe is higher",
          flush=True)

    def _payload(incomplete: bool) -> dict:
        body = {
            "state_dict": best_state,
            "mean": day_mean, "std": day_std,
            "week_mean": w_mean, "week_std": w_std,
            "fuse_inputs": ["day", "week"],
            "fuse_mode": "residual",
            "day_ckpt": day_ckpt.name,
            "y_mean": y_mean, "y_std": y_std,
            "winsor_lo": y_lo, "winsor_hi": y_hi,
            "seed": seed, "window": window_key,
            "task": "regression", "target": TARGET_COL,
            "history": hist, "best_epoch": best_epoch,
            "best_top_mu": best_mu, "best_top_sharpe": best_sh,
            "day_top_mu": day_mu, "day_top_sharpe": day_sh,
            "best_val_ic": best_ic,
            "select": "top_group_sharpe",
            "elapsed_s": time.time() - t0,
        }
        if incomplete:
            body["incomplete"] = True
        else:
            body["epochs"] = len(hist)
        return body

    for epoch in range(1, max_epochs + 1):
        print(f"[seed {seed}] start residual epoch {epoch}/{max_epochs}", flush=True)
        tr_mse, tr_dir = _residual_epoch(
            model, dl_train, device, criterion, y_mean, y_std, y_lo, y_hi, optimiser,
            desc=f"seed{seed} ep{epoch} week")
        va_mse, va_dir, va_pred, va_y = _residual_epoch(
            model, dl_valid, device, criterion, y_mean, y_std, y_lo, y_hi, None,
            desc=f"seed{seed} ep{epoch} valid", collect=True)
        va_ic = _spearman(va_pred, va_y)
        top_mu, top_sh = _top_group_mu_sharpe(valid_codes, valid_dates, va_pred, cfg, market)
        hist.append({
            "epoch": epoch,
            "train_mse": tr_mse, "train_dir": tr_dir,
            "valid_mse": va_mse, "valid_dir": va_dir, "valid_ic": va_ic,
            "top_mu": top_mu, "top_sharpe": top_sh,
        })
        improved = _keep_week_epoch(top_mu, top_sh, best_mu, best_sh, day_mu)
        if improved:
            best_mu = top_mu
            best_sh = top_sh
            best_ic = va_ic
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        mark = "  best" if improved else ""
        print(f"[seed {seed}] ep {epoch:02d}/{max_epochs}  "
              f"train mse={tr_mse:.6f} dir={tr_dir:.3f}  "
              f"valid mse={va_mse:.6f} dir={va_dir:.3f} ic={va_ic:.4f}  "
              f"top_mu={top_mu:.6f} top_sharpe={top_sh:.4f}  "
              f"best_epoch={best_epoch}{mark}", flush=True)
        _save_ckpt(running_path, _payload(True))
        print(f"[seed {seed}] wrote {running_path.name} after epoch {epoch}", flush=True)

    if best_epoch == 0:
        print(f"[seed {seed}] 周塔没有超过日塔，留下全 0 的周线性层。预测等于日塔。", flush=True)
    _save_ckpt(out_path, _payload(False))
    if running_path.exists():
        running_path.unlink()
    elapsed = time.time() - t0
    print(f"[seed {seed}] saved {out_path}  best_epoch={best_epoch}/{max_epochs}  "
          f"top_mu={best_mu:.6f} top_sharpe={best_sh:.4f}  "
          f"day_mu={day_mu:.6f} day_sharpe={day_sh:.4f}  {elapsed:.1f}s",
          flush=True)
    return {
        "seed": seed, "best_epoch": best_epoch,
        "best_top_mu": best_mu, "best_top_sharpe": best_sh,
        "day_top_mu": day_mu, "day_top_sharpe": day_sh,
        "best_val_ic": best_ic, "epochs": len(hist),
        "elapsed_s": elapsed, "path": str(out_path), "day_ckpt": str(day_ckpt),
    }


def train_window(cfg, paths, window_key: str = WINDOW_KEY, n_seeds: int = 5,
                 num_workers: int = 0, skip_existing: bool = True,
                 batch_size: int | None = None,
                 pin_memory: bool = False) -> list[dict]:
    window_key = normalize_window_key(window_key)
    task = str(getattr(cfg.cnn, "task", ""))
    if task != "regression":
        raise RuntimeError(f"cnn.task 应为 regression，实际是 {task!r}")
    from pair_lib import fuse_mode
    mode = fuse_mode(cfg)
    model_dir = resolve_model_dir(paths, cfg, window_key)
    print("models =", model_dir)
    print("mode =", mode)
    if mode == "residual":
        print(f"epochs = {int(cfg.cnn.max_epochs)}  （先训日塔，按 Spearman 留；"
              f"再训周塔残差，验证集最高一组的日均超额不低于日塔时，留夏普更高的一轮）")
    else:
        print(f"epochs = {int(cfg.cnn.max_epochs)}  （固定轮数，不早停，按验证集 Spearman IC 留最好一轮）")
    images, week_images, labels = load_paired_frame(cfg, paths, window_key)

    results = []
    for k in range(n_seeds):
        seed = int(cfg.project.random_seed) + k
        ckpt = model_dir / f"seed_{k}.pt"
        if mode == "residual":
            day_ckpt = model_dir / f"seed_{k}_day.pt"
            if skip_existing and day_ckpt.exists():
                print(f"skip day seed {k} (exists {day_ckpt})")
            else:
                results.append(train_one_seed(
                    images, labels, cfg, window_key, seed, day_ckpt,
                    num_workers=num_workers, batch_size=batch_size,
                    pin_memory=pin_memory, week_images=week_images, kind="day"))
            if skip_existing and ckpt.exists():
                print(f"skip residual seed {k} (exists {ckpt})")
                continue
            results.append(train_residual_seed(
                images, week_images, labels, cfg, paths, window_key, seed,
                day_ckpt, ckpt, num_workers=num_workers, batch_size=batch_size,
                pin_memory=pin_memory))
            continue
        if skip_existing and ckpt.exists():
            print(f"skip seed {k} (exists {ckpt})")
            continue
        results.append(train_one_seed(
            images, labels, cfg, window_key, seed, ckpt,
            num_workers=num_workers, batch_size=batch_size,
            pin_memory=pin_memory, week_images=week_images, kind=mode))
    log_dir = Path(paths["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f"train_{window_key}_summary.json").open("w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def _amp_ctx(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    try:
        return torch.amp.autocast("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast()


def _infer_pred(model, loader, device, y_mean: float, y_std: float) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="infer", leave=False):
            if len(batch) == 3:
                day, week, _ = batch
                tensors = (day.to(device, non_blocking=True), week.to(device, non_blocking=True))
            else:
                xb, _ = batch
                tensors = (xb.to(device, non_blocking=True),)
            with _amp_ctx(device):
                z = model(*tensors).view(-1)
                pred = z * y_std + y_mean
            preds.append(pred.detach().float().cpu().numpy())
    return np.concatenate(preds)


def predict_window(cfg, paths, window_key: str = WINDOW_KEY,
                   num_workers: int = 0) -> pd.DataFrame:
    from pair_lib import fuse_inputs, fuse_mode

    window_key = normalize_window_key(window_key)
    model_dir = resolve_model_dir(paths, cfg, window_key)
    results_dir = resolve_results_dir(paths, cfg)
    images, week_images, labels = load_paired_frame(cfg, paths, window_key)
    inputs = fuse_inputs(cfg)
    mode = fuse_mode(cfg)
    test_idx = np.flatnonzero(labels["split"].astype(str) == "test")
    if len(test_idx) == 0:
        raise RuntimeError("没有 test 样本")
    y = labels[TARGET_COL].to_numpy(np.float32)
    finite = np.isfinite(y[test_idx])
    if not finite.all():
        test_idx = test_idx[finite]
    test_labels = labels.iloc[test_idx].reset_index(drop=True)
    print(f"{window_key} test images={len(test_idx)}  inputs={inputs}  mode={mode}")

    ckpts = _seed_ckpts(model_dir)
    if not ckpts:
        raise FileNotFoundError(f"没有权重 {model_dir}/seed_*.pt")

    device = _pick_device()
    batch = int(cfg.cnn.batch_size) * 4
    ensemble = []
    for ck_path in ckpts:
        try:
            ck = torch.load(ck_path, map_location=device, weights_only=False)
        except TypeError:
            ck = torch.load(ck_path, map_location=device)
        if ck.get("task") not in (None, "regression"):
            raise RuntimeError(f"{ck_path} 不是回归权重（task={ck.get('task')!r}）")
        if "y_mean" not in ck or "y_std" not in ck:
            raise RuntimeError(f"{ck_path} 缺少 y_mean/y_std，不是这一版回归权重")
        ck_inputs = [str(x) for x in ck.get("fuse_inputs", [])]
        ck_mode = ck.get("fuse_mode")
        if not ck_mode:
            ck_mode = "day" if ck_inputs == ["day"] else "concat"
        if ck_mode != mode or ck_inputs != inputs:
            raise RuntimeError(
                f"{ck_path} 的 fuse_mode={ck_mode} fuse_inputs={ck_inputs}，"
                f"config 是 mode={mode} inputs={inputs}。不要混用对照权重。"
            )
        model = build_model(window_key, cfg, kind=ck_mode).to(device)
        model.load_state_dict(ck["state_dict"])
        ds = PairedImageDataset(
            images, week_images, test_idx,
            labels["day_row"].to_numpy(np.int64),
            labels["week_row"].to_numpy(np.int64),
            y[test_idx], ck["mean"], ck["std"],
            ck.get("week_mean", 0.0), ck.get("week_std", 1.0),
            read_week=(ck_mode != "day"),
        )
        dl = DataLoader(ds, batch_size=batch, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))
        print("infer", ck_path.name, "seed", ck.get("seed"),
              "best_epoch", ck.get("best_epoch"))
        ensemble.append(_infer_pred(model, dl, device, float(ck["y_mean"]), float(ck["y_std"])))

    avg = np.mean(np.stack(ensemble, axis=0), axis=0)
    out = pd.DataFrame({
        "code": test_labels["code"].values,
        "date": test_labels["date"].values,
        "pred": avg,
        "future_ret": test_labels[TARGET_COL].values,
        "label": test_labels["label"].values if "label" in test_labels.columns else np.nan,
    })
    for col in ("stock_ret", "mkt_ret"):
        if col in test_labels.columns:
            out[col] = test_labels[col].values
    out_path = results_dir / f"pred_{window_key}.parquet"
    out.to_parquet(out_path, index=False)
    yt = out["future_ret"].to_numpy(np.float64)
    mse = float(np.mean((avg - yt) ** 2))
    dir_acc = float(np.mean((avg > 0) == (yt > 0)))
    if np.std(avg) > 0 and np.std(yt) > 0:
        ic = float(np.corrcoef(avg, yt)[0, 1])
    else:
        ic = float("nan")
    print(f"wrote {out_path}  n={len(out)}  seeds={len(ensemble)}")
    print(f"OOS mse={mse:.6f}  dir_acc={dir_acc:.4f}  ic={ic:.4f}")
    return out


def prepend_zero_cum(daily_ret: pd.Series) -> pd.Series:
    """Cumulative path that starts at 0 on the first date."""
    r = daily_ret.sort_index().fillna(0.0)
    cum = r.cumsum()
    if cum.empty:
        return cum
    t0 = pd.Timestamp(cum.index.min())
    return pd.concat([pd.Series([0.0], index=pd.DatetimeIndex([t0])), cum])


def _horizon_sleeves(cfg) -> tuple[int, int]:
    h = int(getattr(cfg.image, "return_horizon_days", 5))
    n = int(getattr(cfg.backtest, "n_sleeves", h))
    return max(1, h), max(1, n)


def load_pingquan_688(path: str | Path) -> pd.Series:
    """Daily 688 equal-weight return. CSV is in percent."""
    df = pd.read_csv(path)
    if "date" not in df.columns or "688" not in df.columns:
        raise ValueError(f"pingquan.csv 需要 date 和 688 列，实际={list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"])
    r = pd.to_numeric(df["688"], errors="coerce") / 100.0
    s = pd.Series(r.to_numpy(float), index=pd.DatetimeIndex(df["date"]))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    print(f"pingquan 688  n={len(s)}  {s.index.min().date()}→{s.index.max().date()}  "
          f"daily mean={s.mean():.5f}")
    return s


def resolve_processed_dir(cfg, paths: dict) -> Path:
    root = Path(cfg.project.root).resolve()
    cands = [
        Path(paths["processed"]),
        Path("/home/shixi05/ChenSiwei/0914/data/processed"),
        root.parent / "no_rolling" / "data" / "processed",
        root.parent / "0914" / "data" / "processed",
    ]
    parent = root.parent
    if parent.is_dir():
        for sib in sorted(parent.iterdir()):
            if sib.is_dir():
                cands.append(sib / "data" / "processed")
    seen: set[str] = set()
    uniq: list[Path] = []
    for c in cands:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    for c in uniq:
        if (c / "prices.parquet").exists():
            if c != Path(paths["processed"]):
                print(f"prices 改用 {c}")
            return c
    looked = "\n".join(f"  {c}" for c in uniq)
    raise FileNotFoundError(f"找不到 prices.parquet，已查找:\n{looked}")


def load_backtest_market(cfg, paths) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """Daily stock ret (wide), 688 pingquan, trading calendar."""
    proc = resolve_processed_dir(cfg, paths)
    prices = pd.read_parquet(proc / "prices.parquet", columns=["code", "date", "ret"])
    prices["date"] = pd.to_datetime(prices["date"])
    prices["code"] = prices["code"].astype(str).str.zfill(6)
    ret_wide = prices.pivot_table(
        index="date", columns="code", values="ret", aggfunc="mean")
    ret_wide.index = pd.to_datetime(ret_wide.index)
    pingquan_path = str(getattr(
        cfg.paths, "pingquan_csv",
        "/storage/server/227server/marketdata/pingquan/daily/pingquan.csv"))
    mkt = load_pingquan_688(pingquan_path)
    cal_path = proc / "trading_calendar.parquet"
    if cal_path.exists():
        cal = pd.DatetimeIndex(
            pd.to_datetime(pd.read_parquet(cal_path)["trade_date"]).sort_values())
    else:
        cal = pd.DatetimeIndex(ret_wide.index.sort_values())
    return ret_wide, mkt, cal


def _pred_path(paths: dict, cfg, window_key: str) -> Path:
    results_dir = resolve_results_dir(paths, cfg)
    pred_path = results_dir / f"pred_{window_key}.parquet"
    if pred_path.exists():
        return pred_path
    local = Path(paths["results"]) / f"pred_{window_key}.parquet"
    if local.exists():
        return local
    raise FileNotFoundError(pred_path)


def _load_test_pred(paths: dict, cfg, window_key: str) -> pd.DataFrame:
    pred = pd.read_parquet(_pred_path(paths, cfg, window_key))
    pred["date"] = pd.to_datetime(pred["date"])
    if "pred" not in pred.columns:
        raise RuntimeError("预测文件没有 pred 列。请用这一版的 predict_window 重新预测。")
    pred = pred.dropna(subset=["pred"])
    pred = pred[(pred["date"] >= pd.Timestamp(cfg.data.test_start))
                & (pred["date"] <= pd.Timestamp(cfg.data.test_end))]
    return pred


def formation_baskets_topfrac(
    pred: pd.DataFrame, top_frac: float = 0.10, score_col: str = "pred",
) -> tuple[dict, pd.DataFrame]:
    """Each formation date: highest predicted excess, top `top_frac`, equal weight at t close."""
    pred = pred.dropna(subset=[score_col]).copy()
    pred["date"] = pd.to_datetime(pred["date"])
    pred["code"] = pred["code"].astype(str).str.zfill(6)
    baskets: dict[pd.Timestamp, list[str]] = {}
    rows = []
    for dt, g in pred.groupby("date", sort=True):
        n = max(1, int(np.ceil(len(g) * float(top_frac))))
        take = g.nlargest(n, score_col)
        codes = take["code"].tolist()
        baskets[pd.Timestamp(dt)] = codes
        w = 1.0 / len(codes)
        for rec in take.itertuples(index=False):
            rows.append({
                "code": rec.code,
                "date": pd.Timestamp(dt),
                "pred": getattr(rec, score_col, np.nan),
                "future_ret": getattr(rec, "future_ret", np.nan),
                "n_hold": int(len(codes)),
                "weight": w,
            })
    holdings = pd.DataFrame(rows)
    return baskets, holdings


def formation_baskets_group(
    pred: pd.DataFrame, group_col: str, group_val,
) -> dict[pd.Timestamp, list[str]]:
    sub = pred.loc[pred[group_col] == group_val, ["code", "date"]].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    sub["code"] = sub["code"].astype(str).str.zfill(6)
    out: dict[pd.Timestamp, list[str]] = {}
    for dt, g in sub.groupby("date", sort=True):
        out[pd.Timestamp(dt)] = g["code"].tolist()
    return out


def _one_way_turnover(w_old: dict[str, float], w_new: dict[str, float]) -> float:
    keys = set(w_old) | set(w_new)
    return 0.5 * sum(abs(w_new.get(k, 0.0) - w_old.get(k, 0.0)) for k in keys)


def overlapping_sleeve_daily(
    baskets: dict[pd.Timestamp, list[str]],
    ret_wide: pd.DataFrame,
    mkt: pd.Series,
    cal: pd.DatetimeIndex,
    horizon: int = 5,
    n_sleeves: int | None = None,
    min_sleeves: int | None = None,
    cost_bps_per_side: float = 0.0,
) -> pd.DataFrame:
    """5 overlapping sleeves, daily MTM, sleeve NAVs compound independently.

    Buy at formation close t, sell at t+horizon close. Sleeve k formed on
    calendar index i uses proceeds of the batch that expires the same close.
    Portfolio excess on day d = NAV-weighted mean of (sleeve_abs - 688).
    Days with fewer than ``min_sleeves`` active sleeves are dropped.
    Intra-sleeve weights start equal and drift with prices (no daily rebalance).
    ``cost_bps_per_side`` is charged on one-way turnover x2 at each sleeve
    rebalance (sell old / buy new at the same close).
    """
    if n_sleeves is None:
        n_sleeves = int(horizon)
    if min_sleeves is None:
        min_sleeves = int(n_sleeves)
    cal = pd.DatetimeIndex(pd.to_datetime(cal)).sort_values()
    pos = {pd.Timestamp(d): i for i, d in enumerate(cal)}
    cols = {str(c).zfill(6): c for c in ret_wide.columns}
    mkt = mkt.copy()
    mkt.index = pd.to_datetime(mkt.index)
    mkt = mkt[~mkt.index.duplicated(keep="last")].sort_index()
    bps = float(cost_bps_per_side) / 1e4

    events: dict[pd.Timestamp, list[tuple[int, float]]] = {}
    costs: dict[pd.Timestamp, dict[int, float]] = {}
    last_w: dict[int, dict[str, float]] = {}
    for t, codes in sorted(baskets.items(), key=lambda x: pd.Timestamp(x[0])):
        t = pd.Timestamp(t)
        if t not in pos:
            continue
        i = pos[t]
        if i + horizon >= len(cal):
            continue
        sid = i % int(n_sleeves)
        earn_idx = pd.DatetimeIndex(cal[i + 1: i + 1 + horizon])
        names = [str(c).zfill(6) for c in codes]
        names = [c for c in names if c in cols]
        n = len(names)
        if n == 0:
            for d in earn_idx:
                events.setdefault(pd.Timestamp(d), []).append((sid, 0.0))
            continue
        new_w = {c: 1.0 / n for c in names}
        if bps > 0.0:
            if sid in last_w:
                to = _one_way_turnover(last_w[sid], new_w)
                costs.setdefault(t, {})[sid] = 2.0 * to * bps
            else:
                to = _one_way_turnover({}, new_w)
                costs.setdefault(pd.Timestamp(earn_idx[0]), {})[sid] = 2.0 * to * bps
        use_cols = [cols[c] for c in names]
        sub = ret_wide.reindex(index=earn_idx, columns=use_cols).to_numpy(dtype=float)
        sub = np.where(np.isfinite(sub), sub, 0.0)
        w = np.full(n, 1.0 / n, dtype=float)
        for h in range(horizon):
            r_s = float(np.dot(w, sub[h]))
            d = pd.Timestamp(earn_idx[h])
            events.setdefault(d, []).append((sid, r_s))
            growth = w * (1.0 + sub[h])
            tot = float(growth.sum())
            if tot > 0.0:
                w = growth / tot
        last_w[sid] = {names[j]: float(w[j]) for j in range(n)}

    nav = np.ones(int(n_sleeves), dtype=float)
    rows = []
    for d in sorted(events):
        by_sid: dict[int, float] = {}
        for sid, r_s in events[d]:
            if sid not in by_sid:
                by_sid[sid] = r_s
        cost_d = costs.get(d, {})
        by_sid_net = {s: r - float(cost_d.get(s, 0.0)) for s, r in by_sid.items()}
        m = mkt.reindex([d]).iloc[0] if d in mkt.index else np.nan
        if not np.isfinite(m):
            m = 0.0
        active = list(by_sid_net)
        if len(active) >= int(min_sleeves):
            nav_sum = float(sum(nav[s] for s in active))
            if nav_sum <= 0.0:
                nav_sum = float(len(active))
            port_abs = float(sum(nav[s] / nav_sum * by_sid_net[s] for s in active))
            port_cost = float(sum(nav[s] / nav_sum * float(cost_d.get(s, 0.0)) for s in active))
            rows.append({
                "date": d,
                "daily_excess": port_abs - float(m),
                "daily_abs": port_abs,
                "mkt": float(m),
                "n_sleeves": int(len(active)),
                "daily_cost": port_cost,
            })
        for sid, r_s in by_sid_net.items():
            nav[sid] *= (1.0 + r_s)

    if not rows:
        return pd.DataFrame(
            columns=["daily_excess", "daily_abs", "mkt", "n_sleeves", "daily_cost"])
    return pd.DataFrame(rows).set_index("date").sort_index()


def overlapping_group_excess(
    pred: pd.DataFrame, group_col: str,
    ret_wide: pd.DataFrame, mkt: pd.Series, cal: pd.DatetimeIndex,
    horizon: int = 5, n_sleeves: int | None = None,
) -> pd.DataFrame:
    """One overlapping-sleeve daily excess series per group value."""
    series = {}
    for gval in sorted(pd.unique(pred[group_col].dropna())):
        baskets = formation_baskets_group(pred, group_col, gval)
        daily = overlapping_sleeve_daily(
            baskets, ret_wide, mkt, cal,
            horizon=horizon, n_sleeves=n_sleeves)
        if daily.empty:
            continue
        series[gval] = daily["daily_excess"]
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index()


def _save_fig_local_remote(fig, fig_path: Path, results_dir: Path) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    try:
        remote_fig = results_dir / "figures"
        remote_fig.mkdir(parents=True, exist_ok=True)
        fig.savefig(remote_fig / fig_path.name, dpi=150, bbox_inches="tight")
    except Exception:
        pass


def plot_top10_excess(cfg, paths, window_key: str = WINDOW_KEY,
                      top_frac: float | None = None):
    """Top 10% by predicted excess, 5 overlapping sleeves, daily MTM excess vs 688."""
    import matplotlib.pyplot as plt

    window_key = normalize_window_key(window_key)
    if top_frac is None:
        top_frac = float(getattr(cfg.backtest, "top_frac", 0.10))
    horizon, n_sleeves = _horizon_sleeves(cfg)
    results_dir = resolve_results_dir(paths, cfg)
    pred = _load_test_pred(paths, cfg, window_key)

    baskets, holdings = formation_baskets_topfrac(pred, top_frac=top_frac, score_col="pred")
    ret_wide, mkt, cal = load_backtest_market(cfg, paths)
    daily = overlapping_sleeve_daily(
        baskets, ret_wide, mkt, cal,
        horizon=horizon, n_sleeves=n_sleeves, min_sleeves=n_sleeves)
    if daily.empty:
        raise RuntimeError(f"{window_key}: 没有满仓后的逐日超额（检查 pred / prices / 日历）")

    r = daily["daily_excess"]
    cum = r.fillna(0.0).cumsum()
    n_by_date = holdings.groupby("date")["n_hold"].first() if len(holdings) else pd.Series(dtype=float)
    out = pd.DataFrame({
        "date": r.index,
        "daily_excess": r.values,
        "daily_abs": daily["daily_abs"].values,
        "mkt": daily["mkt"].values,
        "cum_excess": cum.values,
        "n_sleeves": daily["n_sleeves"].values,
        "n_names": n_by_date.reindex(r.index).to_numpy(),
    })
    csv_path = results_dir / f"top10_excess_{window_key}.csv"
    out.to_csv(csv_path, index=False)
    hold_path = results_dir / f"holdings_{window_key}_top10.csv"
    holdings.to_csv(hold_path, index=False)

    days_per_year = int(getattr(cfg.backtest, "trading_days_per_year", 252))
    mu = float(r.mean())
    sd = float(r.std(ddof=1)) if len(r) > 1 else float("nan")
    sharpe = mu / sd * np.sqrt(days_per_year) if sd and np.isfinite(sd) and sd > 0 else float("nan")
    ann = mu * days_per_year

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    drawn = prepend_zero_cum(r)
    ax.plot(drawn.index, drawn.values * 100, lw=1.6, color="#4C78A8")
    ax.axhline(0.0, color="black", lw=0.7)
    ax.set_ylabel("Cumulative excess vs 688 pingquan (%)")
    ax.set_title(
        f"{window_key} 日K+周K  每天调仓  {n_sleeves}个组合  持有{int(getattr(cfg.image, 'return_horizon_days', 5))}日  "
        f"前{top_frac:.0%}  ann={ann:.2%}  sharpe={sharpe:.2f}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig_dir = Path(paths["figures"])
    fig_path = fig_dir / f"top10_excess_{window_key}.png"
    _save_fig_local_remote(fig, fig_path, results_dir)

    print(f"days={len(r)}  mean daily excess={mu:.5f}  "
          f"ann={ann:.2%}  sharpe={sharpe:.3f}  "
          f"cum={float(cum.iloc[-1]):.3%}  "
          f"n/day≈{float(n_by_date.mean()) if len(n_by_date) else float('nan'):.1f}  "
          f"start={r.index.min().date()}")
    print("wrote", csv_path)
    print("wrote", hold_path)
    print("wrote", fig_path)
    plt.show()
    return out


def plot_top10_gross_net(cfg, paths, window_key: str = WINDOW_KEY,
                         cost_bps_per_side: float = 5.0,
                         top_frac: float | None = None):
    """Top10% excess: gross vs net of per-side cost (default 万五)."""
    import matplotlib.pyplot as plt

    window_key = normalize_window_key(window_key)
    if top_frac is None:
        top_frac = float(getattr(cfg.backtest, "top_frac", 0.10))
    horizon, n_sleeves = _horizon_sleeves(cfg)
    results_dir = resolve_results_dir(paths, cfg)
    pred = _load_test_pred(paths, cfg, window_key)

    baskets, _ = formation_baskets_topfrac(pred, top_frac=top_frac, score_col="pred")
    ret_wide, mkt, cal = load_backtest_market(cfg, paths)
    kwargs = dict(
        baskets=baskets, ret_wide=ret_wide, mkt=mkt, cal=cal,
        horizon=horizon, n_sleeves=n_sleeves, min_sleeves=n_sleeves)
    gross = overlapping_sleeve_daily(**kwargs, cost_bps_per_side=0.0)
    net = overlapping_sleeve_daily(**kwargs, cost_bps_per_side=cost_bps_per_side)
    if gross.empty or net.empty:
        raise RuntimeError(f"{window_key}: 没有满仓后的逐日超额")

    days_per_year = int(getattr(cfg.backtest, "trading_days_per_year", 252))

    def _stats(r: pd.Series):
        mu = float(r.mean())
        sd = float(r.std(ddof=1)) if len(r) > 1 else float("nan")
        sharpe = (mu / sd * np.sqrt(days_per_year)
                  if sd and np.isfinite(sd) and sd > 0 else float("nan"))
        return mu, mu * days_per_year, sharpe

    net_lab = "net 万五/边" if cost_bps_per_side == 5 else f"net {cost_bps_per_side:.0f}bps/边"
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for series, label, color, lw in [
        (gross["daily_excess"], "gross", "#4C78A8", 1.8),
        (net["daily_excess"], net_lab, "#E45756", 1.8),
    ]:
        mu, ann, sh = _stats(series)
        drawn = prepend_zero_cum(series)
        ax.plot(drawn.index, drawn.values * 100, lw=lw, color=color,
                label=f"{label}  ann={ann:.2%}  sharpe={sh:.2f}")
    ax.axhline(0.0, color="black", lw=0.7)
    ax.set_ylabel("Cumulative excess vs 688 pingquan (%)")
    ax.set_title(
        f"{window_key} 日K+周K  每天调仓  {n_sleeves}个组合  gross vs net")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig_path = Path(paths["figures"]) / f"top10_gross_net_{window_key}.png"
    _save_fig_local_remote(fig, fig_path, results_dir)
    g_mu, g_ann, g_sh = _stats(gross["daily_excess"])
    n_mu, n_ann, n_sh = _stats(net["daily_excess"])
    cost_ann = float(net["daily_cost"].mean()) * days_per_year
    print(f"{window_key}  gross ann={g_ann:.2%} sharpe={g_sh:.3f}  "
          f"net ann={n_ann:.2%} sharpe={n_sh:.3f}  "
          f"cost drag={cost_ann:.2%}  "
          f"mean daily TO-cost={float(net['daily_cost'].mean()):.5f}")
    print("wrote", fig_path)
    plt.show()
    return gross, net


def assign_random_groups(pred: pd.DataFrame, n_groups: int = 10,
                         seed: int = 20260417) -> pd.DataFrame:
    out = pred.copy()
    rng = np.random.default_rng(seed)

    def _split(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        if n < n_groups:
            return pd.Series(np.nan, index=g.index)
        order = rng.permutation(n)
        bins = np.floor(order * n_groups / n).astype(int) + 1
        bins = np.clip(bins, 1, n_groups)
        return pd.Series(bins, index=g.index)

    out["rand_group"] = out.groupby("date", group_keys=False).apply(_split)
    return out


def plot_random_groups(cfg, paths, window_key: str = WINDOW_KEY, n_groups: int = 10):
    """Random groups with the same overlapping-sleeve daily MTM as the main plot."""
    import matplotlib.pyplot as plt

    window_key = normalize_window_key(window_key)
    results_dir = resolve_results_dir(paths, cfg)
    pred = _load_test_pred(paths, cfg, window_key)
    pred = assign_random_groups(pred, n_groups=n_groups,
                                seed=int(cfg.project.random_seed))
    horizon, n_sleeves = _horizon_sleeves(cfg)
    ret_wide, mkt, cal = load_backtest_market(cfg, paths)
    rets = overlapping_group_excess(
        pred, "rand_group", ret_wide, mkt, cal,
        horizon=horizon, n_sleeves=n_sleeves)
    rets.columns = [f"R{int(c)}" for c in rets.columns]
    cols = [f"R{i}" for i in range(1, n_groups + 1) if f"R{i}" in rets.columns]

    baskets_all: dict[pd.Timestamp, list[str]] = {}
    for dt, g in pred.groupby("date", sort=True):
        baskets_all[pd.Timestamp(dt)] = (
            g["code"].astype(str).str.zfill(6).tolist())
    uni = overlapping_sleeve_daily(
        baskets_all, ret_wide, mkt, cal,
        horizon=horizon, n_sleeves=n_sleeves)
    ew_xs = uni["daily_excess"] if not uni.empty else pd.Series(dtype=float)

    cmap = plt.cm.RdYlBu_r
    colors = cmap(np.linspace(0.05, 0.95, max(len(cols), 1)))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for col, color in zip(cols, colors):
        drawn = prepend_zero_cum(rets[col])
        ax.plot(drawn.index, drawn.values * 100, label=col, color=color, lw=1.2)
    if not ew_xs.empty:
        drawn = prepend_zero_cum(ew_xs)
        ax.plot(drawn.index, drawn.values * 100, color="black", lw=2.3,
                label="所选平权-688", zorder=5)
    ax.axhline(0.0, color="black", lw=0.7)
    ax.set_title(f"{window_key} 日K+周K  每天调仓  {n_sleeves}个组合")
    ax.set_ylabel("Cumulative excess vs pingquan (%)")
    ax.legend(frameon=False, ncol=5)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig_path = Path(paths["figures"]) / f"random_groups_{window_key}.png"
    _save_fig_local_remote(fig, fig_path, results_dir)
    print("wrote", fig_path)
    plt.show()
    return rets
