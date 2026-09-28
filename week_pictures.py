"""One grayscale image per completed market week.

A later Tuesday reuses that picture. It is not stored again. Limit-up and ST
stay on the daily label; they do not decide whether the weekly picture exists.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from draw_gray import macd_window, render_gray
from images_lib import (
    _concat_image_shards,
    _image_cfg,
    resolve_processed_dir,
)

_WP: dict = {}
_PROTECTED = {
    "k_pictures_gray_body",
    "k_pictures_gray_week",
    "k_pictures_rgb_I20",
    "k_pictures_gray_reg",
}


def _refuse(path: Path) -> None:
    if path.name in _PROTECTED:
        raise RuntimeError(
            f"拒绝写入 {path}。日 K 在 k_pictures_gray_body，"
            "周末调仓的周 K 在 k_pictures_gray_week，这一支不要覆盖它们。"
        )


def market_weeks(cal: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """ISO weeks on the exchange calendar. The end is that week's last session."""
    cal = pd.DatetimeIndex(pd.to_datetime(cal)).sort_values().unique()
    if len(cal) == 0:
        empty = np.array([], dtype="datetime64[ns]")
        return empty, empty
    iso = pd.Series(cal).dt.isocalendar()
    key = iso["year"].to_numpy(np.int64) * 100 + iso["week"].to_numpy(np.int64)
    change = np.flatnonzero(np.diff(key)) + 1
    bounds = np.concatenate([[0], change, [len(cal)]])
    starts = np.asarray(cal[bounds[:-1]], dtype="datetime64[ns]")
    ends = np.asarray(cal[bounds[1:] - 1], dtype="datetime64[ns]")
    return starts, ends


def _init_worker(params: dict) -> None:
    global _WP
    _WP = params


def _week_spans(dates: np.ndarray, week_starts: np.ndarray, week_ends: np.ndarray
                ) -> dict[int, tuple[int, int]]:
    n_weeks = len(week_ends)
    if n_weeks == 0 or len(dates) == 0:
        return {}
    wix = np.searchsorted(week_ends, dates, side="left")
    clipped = np.clip(wix, 0, n_weeks - 1)
    inside = (
        (wix < n_weeks)
        & (dates >= week_starts[clipped])
        & (dates <= week_ends[clipped])
    )
    rows = np.flatnonzero(inside)
    if len(rows) == 0:
        return {}
    wix = wix[rows]
    cuts = np.flatnonzero(np.diff(wix)) + 1
    bounds = np.concatenate([[0], cuts, [len(rows)]])
    spans: dict[int, tuple[int, int]] = {}
    for a, b in zip(bounds[:-1], bounds[1:]):
        spans[int(wix[a])] = (int(rows[a]), int(rows[b - 1]) + 1)
    return spans


def _gen_one_stock_week(payload: tuple):
    """One picture per week the stock actually traded through the week-end."""
    p = _WP
    window = int(p["window"])
    ipo_buffer = int(p["ipo_buffer_days"])
    count_only = p["count_only"]
    require_volume = p["require_volume"]
    draw_kw = p["draw_kw"]
    macd_warmup = int(p["macd_warmup"])
    week_starts = p["week_starts"]
    week_ends = p["week_ends"]

    code, dates, o_all, h_all, l_all, c_all, vol_all, reb_ix = payload
    dates = np.asarray(dates, dtype="datetime64[ns]")
    spans_all = _week_spans(dates, week_starts, week_ends)
    if not spans_all:
        return None, []
    earliest = min(spans_all)

    imgs: list[np.ndarray] = []
    meta: list[tuple] = []
    for end_ix in reb_ix:
        t = dates[int(end_ix)]
        wi = int(np.searchsorted(week_ends, t, side="left"))
        if wi >= len(week_ends) or week_ends[wi] != t:
            continue
        first_wi = wi - window + 1
        if first_wi < 0:
            continue
        if ipo_buffer > 0 and earliest > first_wi - ipo_buffer:
            continue
        spans = []
        missing = False
        for k in range(first_wi, wi + 1):
            sp = spans_all.get(k)
            if sp is None:
                missing = True
                break
            spans.append(sp)
        if missing:
            continue
        if int(spans[-1][1]) - 1 != int(end_ix):
            continue
        if require_volume and any(
                not np.all(np.isfinite(vol_all[a:b])) for a, b in spans):
            continue

        o = np.empty(window, dtype=np.float64)
        h = np.empty(window, dtype=np.float64)
        l = np.empty(window, dtype=np.float64)
        c = np.empty(window, dtype=np.float64)
        v = np.empty(window, dtype=np.float64)
        for i, (a, b) in enumerate(spans):
            o[i] = o_all[a]
            h[i] = np.nanmax(h_all[a:b])
            l[i] = np.nanmin(l_all[a:b])
            c[i] = c_all[b - 1]
            v[i] = float(np.sum(vol_all[a:b]))
        if not np.all(np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)):
            continue

        meta.append((code, pd.Timestamp(t)))
        if count_only:
            continue

        warm: list[float] = []
        k = first_wi - 1
        while k >= 0 and len(warm) < macd_warmup:
            sp = spans_all.get(k)
            if sp is None:
                break
            warm.append(float(c_all[sp[1] - 1]))
            k -= 1
        warm.reverse()
        closes_ext = np.concatenate([np.asarray(warm, dtype=np.float64), c])
        rets = np.zeros(len(closes_ext), dtype=np.float64)
        prev = closes_ext[:-1]
        nxt = closes_ext[1:]
        good = np.isfinite(prev) & (prev != 0.0) & np.isfinite(nxt)
        rets[1:] = np.where(good, nxt / prev - 1.0, 0.0)
        start_i = len(closes_ext) - window
        dif, dea, hist = macd_window(
            rets, start_i, len(closes_ext) - 1,
            fast=int(draw_kw["macd_fast"]),
            slow=int(draw_kw["macd_slow"]),
            signal=int(draw_kw["macd_signal"]),
            warmup=start_i,
        )
        ret_w = np.zeros(window, dtype=np.float64)
        prev_c = c[:-1]
        good_c = np.isfinite(prev_c) & (prev_c != 0.0)
        ret_w[1:] = np.where(good_c, c[1:] / prev_c - 1.0, 0.0)
        c_path = np.cumprod(1.0 + ret_w)
        scale = c_path / c
        imgs.append(render_gray(
            o * scale, h * scale, l * scale, c_path, v, dif, dea, hist, **draw_kw))

    if not imgs:
        return None, meta
    stacked = np.stack(imgs, axis=0)
    stacked = np.ascontiguousarray(stacked[:, None, :, :])
    return stacked, meta


def _gen_shard(task: tuple):
    idx, payload, shard_dir = task
    imgs_arr, meta = _gen_one_stock_week(payload)
    if imgs_arr is None or len(imgs_arr) == 0:
        return idx, None, 0, meta
    path = str(Path(shard_dir) / f"{idx:08d}.npy")
    np.save(path, imgs_arr)
    return idx, path, int(len(imgs_arr)), meta


def _week_root(cfg, paths: dict, *, max_codes, debug_codes, use_remote) -> tuple[Path, bool]:
    remote = Path(str(getattr(cfg.paths, "week_images_remote")))
    _refuse(remote)
    local = Path(paths["week_images"])
    is_full = (max_codes is None) and not debug_codes
    want = (use_remote is True) or (use_remote == "auto" and is_full)
    if want:
        try:
            remote.mkdir(parents=True, exist_ok=True)
            probe = remote / ".write_probe"
            probe.write_bytes(b"ok")
            probe.unlink()
            print(f"week image root (remote) = {remote}")
            return remote, True
        except Exception as e:
            print(f"remote {remote} 不可写 ({type(e).__name__}: {e})")
            print(f"fallback local {local}")
    local.mkdir(parents=True, exist_ok=True)
    print(f"week image root (local) = {local}")
    return local, False


def generate_week_pictures(
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
    """Write week images. Never writes k_pictures_gray_body or k_pictures_gray_week."""
    spec = _image_cfg(cfg, window_key)
    window_key = spec["window_key"]
    window = spec["window"]
    height = spec["height"]
    width = spec["width"]

    proc_dir = resolve_processed_dir(cfg, paths)
    prices = pd.read_parquet(proc_dir / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    prices["code"] = prices["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    uni_path = proc_dir / "universe.parquet"
    if uni_path.exists():
        u = pd.read_parquet(uni_path)
        eligible = {str(c).zfill(6) for c in u.loc[u["eligible"], "code"].astype(str)}
        prices = prices[prices["code"].isin(eligible)]
    if debug_codes:
        want = {str(c).zfill(6) for c in debug_codes}
        prices = prices[prices["code"].isin(want)]
    elif max_codes is not None:
        codes = sorted(prices["code"].unique())[: int(max_codes)]
        prices = prices[prices["code"].isin(codes)]

    cal_path = proc_dir / "trading_calendar.parquet"
    if not cal_path.exists():
        raise RuntimeError(f"缺少 {cal_path}")
    cal = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(cal_path)["trade_date"])).sort_values()
    week_starts, week_ends = market_weeks(cal)
    # The first sessions of the sample still need the previous completed week.
    start = pd.Timestamp(cfg.data.train_start) - pd.Timedelta(days=21)
    end = pd.Timestamp(cfg.data.test_end)
    ends = pd.DatetimeIndex(pd.to_datetime(week_ends))
    reb_dates = ends[(ends >= start) & (ends <= end)]
    reb_vals = reb_dates.to_numpy(dtype="datetime64[ns]")
    print(
        f"week pictures  n_week_ends={len(reb_dates)}  "
        f"{None if len(reb_dates) == 0 else pd.Timestamp(reb_dates.min()).date()}"
        f"→{None if len(reb_dates) == 0 else pd.Timestamp(reb_dates.max()).date()}  "
        f"window={window} weeks",
        flush=True,
    )

    img_root, used_remote = _week_root(
        cfg, paths, max_codes=max_codes, debug_codes=debug_codes, use_remote=use_remote)
    img_dir = img_root / window_key
    img_dir.mkdir(parents=True, exist_ok=True)

    uni = getattr(cfg, "universe", None)
    ipo_buffer = int(getattr(uni, "ipo_buffer_days", 0)) if uni else 0
    payloads = []
    for code, grp in prices.groupby("code", sort=False):
        grp = grp.sort_values("date")
        dates = grp["date"].to_numpy("datetime64[ns]")
        n = len(dates)
        pos = np.searchsorted(dates, reb_vals)
        inb = pos < n
        pos_clip = np.where(inb, pos, 0)
        match = inb & (dates[pos_clip] == reb_vals)
        reb_ix = pos[match].astype(np.int64)
        if reb_ix.size == 0:
            continue
        payloads.append((
            str(code).zfill(6), dates,
            grp["open"].to_numpy(float), grp["high"].to_numpy(float),
            grp["low"].to_numpy(float), grp["close"].to_numpy(float),
            grp["volume"].to_numpy(float), reb_ix,
        ))
    del prices

    print(
        "filters: 20 consecutive completed weeks, finite volume, "
        "ipo buffer. Limit and ST are applied later from the daily labels.",
        flush=True,
    )
    params = dict(
        window=window,
        ipo_buffer_days=ipo_buffer,
        count_only=count_only,
        require_volume=require_volume,
        draw_kw=spec["draw_kw"],
        macd_warmup=int(getattr(cfg.image, "macd_warmup", 105)),
        week_starts=np.asarray(week_starts, dtype="datetime64[ns]"),
        week_ends=np.asarray(week_ends, dtype="datetime64[ns]"),
    )
    jobs = max(1, int(jobs))
    all_meta: list[tuple] = []
    shard_rows: list[tuple] = []
    desc = f"{'count' if count_only else 'week pictures'} {window_key}"
    import tempfile
    shard_dir = None
    if not count_only:
        shard_dir = Path(tempfile.mkdtemp(prefix=f"{window_key}_fuse_week_"))
        print(f"shard dir (local) = {shard_dir}")
        if jobs > 1:
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    if count_only:
        _init_worker(params)
        if jobs == 1:
            for pl in tqdm(payloads, desc=desc):
                _, meta = _gen_one_stock_week(pl)
                all_meta.extend(meta)
        else:
            with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker,
                                     initargs=(params,)) as ex:
                for _, meta in tqdm(ex.map(_gen_one_stock_week, payloads, chunksize=4),
                                    total=len(payloads), desc=desc):
                    all_meta.extend(meta)
    else:
        tasks = [(i, pl, str(shard_dir)) for i, pl in enumerate(payloads)]
        if jobs == 1:
            _init_worker(params)
            for task in tqdm(tasks, desc=desc):
                shard_rows.append(_gen_shard(task))
                all_meta.extend(shard_rows[-1][3])
        else:
            with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker,
                                     initargs=(params,)) as ex:
                for row in tqdm(ex.map(_gen_shard, tasks, chunksize=1),
                                total=len(tasks), desc=desc):
                    shard_rows.append(row)
                    all_meta.extend(row[3])

    info = {
        "window_key": window_key,
        "n_payloads": len(payloads),
        "n_obs": len(all_meta),
        "img_dir": img_dir,
        "used_remote": used_remote,
        "shape": None,
        "bar": "completed_week",
    }
    if not all_meta:
        print("no week pictures")
        return info
    if count_only:
        print(f"count {len(all_meta)}")
        return info

    meta_df = pd.DataFrame(all_meta, columns=["code", "date"])
    meta_df["date"] = pd.to_datetime(meta_df["date"])
    meta_df.to_parquet(img_dir / "labels.parquet", index=False)
    (img_dir / "image_spec.json").write_text(json.dumps({
        "layout": "NCHW",
        "window": window,
        "height": height,
        "width": width,
        "channels": 1,
        "bar": "completed_week",
        "rule": "one image per completed ISO week; a weekday reuses the latest week_end <= t",
    }, indent=2), encoding="utf-8")
    pointer = Path(paths["week_images"]) / f"{window_key}_LOCATION.txt"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(str(img_dir.resolve()), encoding="utf-8")
    out_npy = img_dir / "images.npy"
    print(f"saved {img_dir / 'labels.parquet'}  n={len(meta_df)}")
    shape = _concat_image_shards(shard_rows, out_npy, height, width, jobs=max(jobs, 8))
    if shard_dir is not None:
        import shutil
        shutil.rmtree(shard_dir, ignore_errors=True)
    info["shape"] = shape
    print(f"saved {out_npy}  {shape}")
    return info
