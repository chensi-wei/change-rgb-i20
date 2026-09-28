"""Image helpers. This branch's generate_images draws completed weekly bars."""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from draw_gray import macd_window, render_gray

_WP: dict = {}
REMOTE_IMAGES = Path("/storage/server/144server/cq/ChenSiwei/k_pictures_gray_body")


def daily_rebalance_dates(cal: pd.DatetimeIndex, start: str, end: str
                          ) -> pd.DatetimeIndex:
    """Every trading day in [start, end] is a formation / 调仓日."""
    td = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    if len(td) == 0:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(sorted(pd.unique(td)))


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


def _init_worker(params: dict) -> None:
    global _WP
    _WP = params


def _window_prices(o_all, h_all, l_all, c_all, ret_all, vol_all,
                   start_ix: int, end_ix: int, window: int):
    """Anchor on the first close and rebuild OHLC on the return path."""
    c_raw = c_all[start_ix:end_ix + 1]
    anchor = c_raw[0]
    v = vol_all[start_ix:end_ix + 1]
    if not (np.isfinite(anchor) and anchor > 0):
        nan = np.full(window, np.nan)
        return nan, nan, nan, nan, v
    ret_w = ret_all[start_ix:end_ix + 1].copy()
    ret_w[0] = 0.0
    ret_w = np.where(np.isfinite(ret_w), ret_w, 0.0)
    c_path = np.cumprod(1.0 + ret_w)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = c_path / c_raw
    o = o_all[start_ix:end_ix + 1] * scale
    h = h_all[start_ix:end_ix + 1] * scale
    l = l_all[start_ix:end_ix + 1] * scale
    return o, h, l, c_path, v


def _gen_one_stock(payload: tuple):
    p = _WP
    window = p["window"]
    horizon = p["horizon"]
    ipo_buffer = p["ipo_buffer_days"]
    excl_reb = p["excl_reb"]
    excl_entry = p["excl_entry"]
    count_only = p["count_only"]
    has_limit = p["has_limit"]
    excl_limit_window = p["excl_limit_window"]
    excl_st_window = p["excl_st_window"]
    has_st = p["has_st"]
    excl_susp_window = p["excl_susp_window"]
    has_open_cap = p["has_open_cap"]
    require_volume = p["require_volume"]
    draw_kw = p["draw_kw"]
    macd_warmup = int(p.get("macd_warmup", 105))

    (code, dates, o_all, h_all, l_all, c_all, ret_all, vol_all,
     lu, ld, st, olu, market_pos, reb_ix, mkt_all) = payload
    n = len(c_all)

    imgs: list[np.ndarray] = []
    meta: list[tuple] = []
    for end_ix in reb_ix:
        start_ix = end_ix - window + 1
        fwd_end = end_ix + horizon
        if start_ix < 0 or fwd_end >= n:
            continue
        if start_ix < ipo_buffer:
            continue
        if has_limit and excl_reb and (lu[end_ix] or ld[end_ix]):
            continue
        if excl_entry:
            if has_open_cap and olu[end_ix + 1]:
                continue
            if (not has_open_cap) and has_limit and lu[end_ix + 1]:
                continue
        if has_limit and excl_limit_window and (
                lu[start_ix:fwd_end + 1].any() or ld[start_ix:fwd_end + 1].any()):
            continue
        if has_st and excl_st_window and st[start_ix:fwd_end + 1].any():
            continue
        if excl_susp_window and market_pos is not None:
            mspan = int(market_pos[fwd_end] - market_pos[start_ix] + 1)
            expected = window + horizon
            if mspan > expected:
                continue
        if require_volume and not np.all(np.isfinite(vol_all[start_ix:end_ix + 1])):
            continue

        fwd_rets = ret_all[end_ix + 1:fwd_end + 1]
        if not np.all(np.isfinite(fwd_rets)):
            continue
        fwd_mkt = mkt_all[end_ix + 1:fwd_end + 1]
        if not np.all(np.isfinite(fwd_mkt)):
            continue
        stock_ret = float(np.prod(1.0 + fwd_rets) - 1.0)
        mkt_ret = float(np.prod(1.0 + fwd_mkt) - 1.0)
        excess = stock_ret - mkt_ret
        label = 1 if excess > 0 else 0
        meta.append((code, dates[end_ix], excess, label, stock_ret, mkt_ret))

        if not count_only:
            o, h, l, c_path, v = _window_prices(
                o_all, h_all, l_all, c_all, ret_all, vol_all,
                start_ix, end_ix, window)
            dif, dea, hist = macd_window(
                ret_all, start_ix, end_ix,
                fast=int(draw_kw["macd_fast"]),
                slow=int(draw_kw["macd_slow"]),
                signal=int(draw_kw["macd_signal"]),
                warmup=macd_warmup,
            )
            imgs.append(render_gray(o, h, l, c_path, v, dif, dea, hist, **draw_kw))

    if not imgs:
        return None, meta
    stacked = np.stack(imgs, axis=0)
    stacked = np.ascontiguousarray(stacked[:, None, :, :])
    return stacked, meta


def _gen_one_stock_shard(task: tuple):
    idx, payload, shard_dir = task
    imgs_arr, meta = _gen_one_stock(payload)
    if imgs_arr is None or len(imgs_arr) == 0:
        return idx, None, 0, meta
    path = str(Path(shard_dir) / f"{idx:08d}.npy")
    np.save(path, imgs_arr)
    return idx, path, int(len(imgs_arr)), meta


def _concat_image_shards(shard_rows: list, out_path: Path,
                         height: int, width: int, jobs: int = 8,
                         channels: int = 1
                         ) -> tuple[int, ...]:
    """Merge per-stock shards into (N, 1, H, W) uint8."""
    import shutil
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    ordered = sorted(shard_rows, key=lambda r: r[0])
    paths_n = [(p, n) for _, p, n, _ in ordered if p and n > 0]
    n_img = int(sum(n for _, n in paths_n))
    if n_img == 0:
        raise RuntimeError("no image shards to concat")
    shape = (n_img, int(channels), int(height), int(width))
    offs, acc = [], 0
    for _, n in paths_n:
        offs.append(acc)
        acc += n

    local_tmp = Path(tempfile.gettempdir()) / f"{out_path.stem}_concat_{os.getpid()}.npy"
    if local_tmp.exists():
        local_tmp.unlink()
    print(f"concat {len(paths_n)} shards -> local {local_tmp}  then copy to {out_path}")
    out = np.lib.format.open_memmap(
        local_tmp, mode="w+", dtype=np.uint8, shape=shape)

    def _copy_one(i: int) -> int:
        p, n = paths_n[i]
        arr = np.load(p)
        off = offs[i]
        expect = (n, int(channels), int(height), int(width))
        if arr.shape != expect:
            raise ValueError(f"shard shape {arr.shape} != {expect} {p}")
        out[off:off + n] = arr
        return i

    jobs = max(1, int(jobs))
    if jobs == 1:
        for i in tqdm(range(len(paths_n)), desc="concat shards"):
            _copy_one(i)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            list(tqdm(ex.map(_copy_one, range(len(paths_n))),
                      total=len(paths_n), desc=f"concat shards x{jobs}"))
    out.flush()
    del out

    out_path.parent.mkdir(parents=True, exist_ok=True)
    remote_tmp = Path(str(out_path) + ".tmp")
    if remote_tmp.exists():
        remote_tmp.unlink()
    print("copying concatenated npy ...")
    shutil.copyfile(local_tmp, remote_tmp)
    if out_path.exists():
        out_path.unlink()
    remote_tmp.replace(out_path)
    local_tmp.unlink(missing_ok=True)
    return shape


def resolve_image_root(cfg, paths: dict, *, use_remote, max_codes, debug_codes
                       ) -> tuple[Path, bool]:
    remote = paths.get("images_remote")
    if remote is None or str(remote) in ("", ".", "None"):
        remote = Path(getattr(cfg.paths, "images_remote", REMOTE_IMAGES))
    remote = Path(remote)
    is_full = (max_codes is None) and not debug_codes
    want = (use_remote is True) or (use_remote == "auto" and is_full)
    if want:
        try:
            remote.mkdir(parents=True, exist_ok=True)
            probe = remote / ".write_probe"
            probe.write_bytes(b"ok")
            probe.unlink()
            print(f"image root (remote) = {remote}")
            return remote, True
        except Exception as e:
            print(f"remote {remote} 不可写 ({type(e).__name__}: {e})")
            print(f"fallback local {paths['images']}")
    print(f"image root (local) = {paths['images']}")
    return Path(paths["images"]), False


def _processed_candidates(cfg, paths: dict) -> list[Path]:
    """prices.parquet lives with the no_rolling panel, not in this folder."""
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
    out: list[Path] = []
    for c in cands:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def resolve_processed_dir(cfg, paths: dict) -> Path:
    cands = _processed_candidates(cfg, paths)
    for c in cands:
        if (c / "prices.parquet").exists():
            if c != Path(paths["processed"]):
                print(f"prices 改用 {c}")
            return c
    looked = "\n".join(f"  {c}" for c in cands)
    raise FileNotFoundError(f"找不到 prices.parquet，已查找:\n{looked}")


def _first_date_after_n_trading_days(cal: pd.DatetimeIndex, last: pd.Timestamp,
                                     n: int) -> pd.Timestamp | None:
    cal = pd.DatetimeIndex(pd.to_datetime(cal)).sort_values()
    last = pd.Timestamp(last)
    i = int(cal.searchsorted(last))
    if i < len(cal) and cal[i] == last:
        j = i + int(n)
    else:
        j = i + int(n)
    if j >= len(cal):
        return None
    return pd.Timestamp(cal[j])


def assign_nonoverlapping_tv_splits(
    meta_df: pd.DataFrame,
    cal: pd.DatetimeIndex,
    window: int,
    train_start, train_end, test_start, test_end,
    ratio: float = 0.7,
) -> pd.DataFrame:
    """Time-split so train / valid / test image windows do not share bars.

    Last train rebalance = ratio quantile of unique dates in the TV span.
    Valid starts after ``window`` trading days; test starts after another
    ``window`` days past the last valid rebalance.
    """
    out = meta_df.copy()
    train_start = pd.Timestamp(train_start)
    train_end = pd.Timestamp(train_end)
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)
    out["split"] = "ignore"

    tv_m = (out["date"] >= train_start) & (out["date"] <= train_end)
    tv_dates = np.sort(out.loc[tv_m, "date"].unique())
    if len(tv_dates) == 0:
        return out

    cut_i = max(0, int(round(len(tv_dates) * float(ratio))) - 1)
    last_train = pd.Timestamp(tv_dates[cut_i])
    valid_from = _first_date_after_n_trading_days(cal, last_train, window)
    out.loc[tv_m & (out["date"] <= last_train), "split"] = "train"
    if valid_from is not None:
        out.loc[tv_m & (out["date"] >= valid_from), "split"] = "valid"

    valid_dates = out.loc[out["split"] == "valid", "date"]
    anchor = pd.Timestamp(valid_dates.max()) if len(valid_dates) else last_train
    test_from = _first_date_after_n_trading_days(cal, anchor, window)
    if test_from is None:
        test_from_print = None
    else:
        if test_from < test_start:
            test_from = test_start
        out.loc[(out["date"] >= test_from) & (out["date"] <= test_end), "split"] = "test"
        test_from_print = test_from.date()
    print(f"split: last_train={last_train.date()}  "
          f"valid_from={None if valid_from is None else valid_from.date()}  "
          f"test_from={test_from_print}  embargo={window} trading days")
    return out


def assert_no_split_kline_overlap(labels: pd.DataFrame, cal: pd.DatetimeIndex,
                                  window: int) -> bool:
    lab = labels.copy()
    lab["date"] = pd.to_datetime(lab["date"])
    ok_all = True
    for a, b in (("train", "valid"), ("valid", "test"), ("train", "test")):
        n_bad = 0
        for _, g in lab.groupby("code", sort=False):
            da = g.loc[g["split"] == a, "date"]
            db = g.loc[g["split"] == b, "date"]
            if da.empty or db.empty:
                continue
            need = _first_date_after_n_trading_days(cal, da.max(), window)
            if need is None or pd.Timestamp(db.min()) < pd.Timestamp(need):
                n_bad += 1
        ok = n_bad == 0
        ok_all = ok_all and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {a}/{b} k-line embargo  "
              f"bad_codes={n_bad}  window={window}")
    return ok_all


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


def _image_cfg(cfg, window_key="I20") -> dict:
    im = cfg.image
    window_key = normalize_window_key(window_key)
    height = int(im.height)
    price_rows = int(im.price_rows)
    volume_rows = int(im.volume_rows)
    macd_rows = int(im.macd_rows)
    if price_rows + volume_rows + macd_rows != height:
        raise ValueError("price_rows + volume_rows + macd_rows 必须等于 height")
    builtin = {"I20": 20, "I60": 60}
    windows = getattr(im, "windows", None)
    if windows is not None and hasattr(windows, window_key):
        window = int(getattr(windows, window_key))
    else:
        window = builtin[window_key]
    px = int(im.px_per_day)
    return {
        "window_key": str(window_key),
        "window": window,
        "height": height,
        "width": window * px,
        "draw_kw": dict(
            image_height=height,
            px_per_day=px,
            price_rows=price_rows,
            volume_rows=volume_rows,
            macd_rows=macd_rows,
            macd_fast=int(im.macd_fast),
            macd_slow=int(im.macd_slow),
            macd_signal=int(im.macd_signal),
        ),
    }


def generate_images(
    cfg,
    paths: dict,
    window_key="I20",
    *,
    max_codes: int | None = 20,
    debug_codes: list[str] | None = None,
    count_only: bool = False,
    jobs: int = 12,
    require_volume: bool = True,
    use_remote="auto",
) -> dict:
    """One picture per completed week. Does not rewrite the daily images."""
    from week_pictures import generate_week_pictures
    return generate_week_pictures(
        cfg, paths, window_key,
        max_codes=max_codes,
        debug_codes=debug_codes,
        count_only=count_only,
        jobs=jobs,
        require_volume=require_volume,
        use_remote=use_remote,
    )


def verify_images(cfg, paths: dict, window_key: str = "I20",
                  img_dir: Path | None = None) -> None:
    spec = _image_cfg(cfg, window_key)
    window = spec["window"]
    window_key = spec["window_key"]
    if img_dir is None:
        pointer = Path(paths.get("week_images", paths["images"])) / f"{window_key}_LOCATION.txt"
        if pointer.exists():
            img_dir = Path(pointer.read_text(encoding="utf-8").strip())
        else:
            img_dir = Path(paths.get("week_images", paths["images"])) / window_key
    img_dir = Path(img_dir)
    fig_dir = Path(paths["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    images = np.load(img_dir / "images.npy", mmap_mode="r")
    labels = pd.read_parquet(img_dir / "labels.parquet")
    labels["date"] = pd.to_datetime(labels["date"])

    height = spec["height"]
    width = spec["width"]
    price_rows = int(cfg.image.price_rows)
    volume_rows = int(cfg.image.volume_rows)
    px = int(cfg.image.px_per_day)

    def verdict(ok: bool, msg: str) -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {msg}")

    print("=" * 60)
    print(f"VERIFY {window_key}  images={images.shape} labels={len(labels)}")
    print("=" * 60)

    ok = (images.ndim == 4 and images.shape[1] == 1
          and images.shape[2] == height and images.shape[3] == width
          and images.dtype == np.uint8 and images.shape[0] == len(labels))
    verdict(ok, f"shape={images.shape} dtype={images.dtype} "
            f"expected (N,1,{height},{width})")

    rng = np.random.default_rng(0)
    n_sample = min(256, len(images))
    sample = np.asarray(images[rng.choice(len(images), n_sample, replace=False)])[:, 0]
    price = sample[:, :price_rows, :]
    vol = sample[:, price_rows:price_rows + volume_rows, :]
    macd = sample[:, price_rows + volume_rows:, :]

    from draw_gray import DEA_GRAY, DIF_GRAY, DOWN, HIST_NEG, HIST_POS, UP, VOL_DOWN, VOL_UP, WICK
    price_known = np.isin(price, [0, int(UP), int(DOWN), int(WICK)])
    verdict(bool(price_known.all()), "price panel is background / body / wick")
    verdict(bool((price == int(UP)).any()), "price has up body")
    verdict(bool((price == int(DOWN)).any()), "price has down body")
    left = price[:, :, 0::px]
    right = price[:, :, 2::px]
    verdict(bool(((left == int(UP)) | (left == int(DOWN))).any()
                 and ((right == int(UP)) | (right == int(DOWN))).any()),
            "body spans left and right columns")

    vol_off = np.ones(width, dtype=bool)
    vol_off[1::px] = False
    verdict(bool((vol[:, :, vol_off] == 0).all()), "volume only on middle column")
    vol_mid = vol[:, :, 1::px]
    verdict(bool(np.isin(vol_mid, [0, int(VOL_UP), int(VOL_DOWN)]).all()),
            "volume gray is up or down, not the body gray")
    verdict(bool((vol_mid != 0).any()), "volume middle column is drawn")

    macd_left = macd[:, :, 0::px]
    macd_mid = macd[:, :, 1::px]
    macd_right = macd[:, :, 2::px]
    verdict(bool(np.isin(macd_left, [0, int(DIF_GRAY)]).all()), "DIF only on left columns")
    verdict(bool(np.isin(macd_right, [0, int(DEA_GRAY)]).all()), "DEA only on right columns")
    verdict(bool(np.isin(macd_mid, [0, int(HIST_POS), int(HIST_NEG)]).all()),
            "histogram only on middle column")
    verdict(bool((macd_left == int(DIF_GRAY)).any() and (macd_right == int(DEA_GRAY)).any()),
            "MACD panel has DIF and DEA")

    if "split" in labels.columns:
        train = labels[labels["split"] == "train"]
        valid = labels[labels["split"] == "valid"]
        test = labels[labels["split"] == "test"]
        cal_path = resolve_processed_dir(cfg, paths) / "trading_calendar.parquet"
        if cal_path.exists() and len(train):
            cal = pd.DatetimeIndex(
                pd.to_datetime(pd.read_parquet(cal_path)["trade_date"]).sort_values())
            assert_no_split_kline_overlap(labels, cal, window)
        if len(train):
            print(f"[INFO] train {train['date'].min().date()}→{train['date'].max().date()}  "
                  f"n={len(train)}")
        if len(valid):
            print(f"[INFO] valid {valid['date'].min().date()}→{valid['date'].max().date()}  "
                  f"n={len(valid)}")
        if len(test):
            print(f"[INFO] test  {test['date'].min().date()}→{test['date'].max().date()}  "
                  f"n={len(test)}")
        tv = labels[labels["split"].isin(["train", "valid"])]
        if len(tv) and "future_ret" in tv.columns:
            print(f"[INFO] train+valid label1={float(tv['label'].mean()):.3f}  "
                  f"excess mean={float(tv['future_ret'].mean()):.5f}")
    else:
        print(f"[INFO] week pictures n={len(labels)}  "
              f"{labels['date'].min().date()}→{labels['date'].max().date()}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib missing, skip montage")
        return
    n_show = min(25, len(images))
    pick = rng.choice(len(images), n_show, replace=False)
    fig, axes = plt.subplots(5, 5, figsize=(12, 16), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i >= n_show:
            ax.axis("off")
            continue
        k = int(pick[i])
        ax.imshow(images[k, 0], cmap="gray", vmin=0, vmax=255,
                  interpolation="nearest", aspect="equal")
        meta = labels.iloc[k]
        title = f"{meta['code']} {pd.Timestamp(meta['date']).date()}"
        if "future_ret" in labels.columns:
            title += f"\ny={int(meta['label'])} ret={meta['future_ret']:+.2%}"
        else:
            title += "\nweek-end"
        ax.set_title(title, fontsize=7)
        ax.axis("off")
    fig_path = fig_dir / f"verify_{window_key}.png"
    fig.suptitle(f"{window_key} completed week {height}x{width}", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] montage → {fig_path}")
    print("=" * 60)
