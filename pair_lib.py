"""Pair each trading day with the last completed weekly image.

Tuesday uses the previous week's picture. The week's last session uses that
week. The picture is not redrawn for Monday through Thursday.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _codes(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def anchor_week_ends(dates, week_ends) -> pd.Series:
    """Latest week-end on or before each date. Earlier than every week-end is NaT."""
    ends = pd.DatetimeIndex(pd.to_datetime(week_ends)).sort_values().unique()
    when = pd.DatetimeIndex(pd.to_datetime(dates))
    out = np.full(len(when), np.datetime64("NaT"), dtype="datetime64[ns]")
    if len(ends) == 0 or len(when) == 0:
        return pd.Series(pd.to_datetime(out))
    pos = ends.searchsorted(when, side="right") - 1
    ok = pos >= 0
    out[np.flatnonzero(ok)] = ends.to_numpy(dtype="datetime64[ns]")[pos[ok]]
    return pd.Series(pd.to_datetime(out))


def align_day_week(day_labels: pd.DataFrame, week_labels: pd.DataFrame, cfg) -> pd.DataFrame:
    """Inner join on code and the last completed week. Split is recomputed here.

    The daily label supplies the 5-day excess. The weekly file only supplies
    the image row. Embargo is fuse.embargo_days (100), long enough for 20 weeks.
    """
    from images_lib import assert_no_split_kline_overlap, assign_nonoverlapping_tv_splits

    day = day_labels.copy()
    week = week_labels.copy()
    day["code"] = _codes(day["code"])
    week["code"] = _codes(week["code"])
    day["date"] = pd.to_datetime(day["date"])
    week["date"] = pd.to_datetime(week["date"])
    day["day_row"] = np.arange(len(day), dtype=np.int64)
    week["week_row"] = np.arange(len(week), dtype=np.int64)
    week = week.drop_duplicates(["code", "date"], keep="last")

    ends = pd.DatetimeIndex(week["date"].unique())
    day["anchor"] = anchor_week_ends(day["date"], ends)
    before = len(day)
    day = day.dropna(subset=["anchor"])
    key = week[["code", "date", "week_row"]].rename(columns={"date": "anchor"})
    paired = day.merge(key, on=["code", "anchor"], how="inner")
    if paired.empty:
        raise RuntimeError("日 K 和周 K 没有配上任何样本。确认两边的 code 都是 6 位。")

    cal = pd.DatetimeIndex(np.sort(pd.to_datetime(day_labels["date"]).unique()))
    embargo = int(getattr(getattr(cfg, "fuse", None), "embargo_days", 100))
    ratio = float(getattr(cfg.data, "train_valid_split", 0.7))
    paired = assign_nonoverlapping_tv_splits(
        paired, cal, embargo,
        cfg.data.train_start, cfg.data.train_end,
        cfg.data.test_start, cfg.data.test_end,
        ratio=ratio,
    )
    ok = assert_no_split_kline_overlap(paired, cal, embargo)
    if not ok:
        raise RuntimeError(f"切分间隔 {embargo} 个交易日不够，同一根周 K 跨进了两个集合")
    paired = paired.sort_values(["date", "code"]).reset_index(drop=True)

    same = paired["date"] == paired["anchor"]
    print(
        f"paired {len(paired)}  day rows {before}  week rows {len(week)}  "
        f"week-end days {int(same.sum())}  other days {int((~same).sum())}  "
        f"embargo={embargo}  split={paired['split'].value_counts().to_dict()}",
        flush=True,
    )
    return paired


def fuse_inputs(cfg) -> list[str]:
    raw = getattr(getattr(cfg, "fuse", None), "inputs", ["day", "week"])
    inputs = [str(x) for x in list(raw)]
    if inputs not in (["day", "week"], ["day"]):
        raise RuntimeError(
            f"fuse.inputs 只能是 [day, week] 或对照用的 [day]，实际是 {inputs}"
        )
    return inputs


def fuse_mode(cfg) -> str:
    """day：只训日塔。residual：冻结日塔，周塔学残差。concat：旧的特征拼接。"""
    inputs = fuse_inputs(cfg)
    if inputs == ["day"]:
        return "day"
    raw = getattr(getattr(cfg, "fuse", None), "mode", None)
    mode = "concat" if raw is None else str(raw)
    if mode not in {"residual", "concat"}:
        raise RuntimeError(
            f"fuse.mode 只能是 residual 或 concat，实际是 {mode!r}"
        )
    return mode
