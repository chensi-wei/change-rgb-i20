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


def _tv_cut(dates: pd.DatetimeIndex, cal: pd.DatetimeIndex, embargo: int, ratio: float):
    """Last train date and first valid date inside ``dates``. The gap is ``embargo`` sessions."""
    from images_lib import _first_date_after_n_trading_days

    dates = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values().unique()
    if len(dates) == 0:
        raise RuntimeError("这段日期里没有交易日，无法切分")
    cut_i = max(0, int(round(len(dates) * float(ratio))) - 1)
    last_train = pd.Timestamp(dates[cut_i])
    valid_from = _first_date_after_n_trading_days(cal, last_train, int(embargo))
    if valid_from is None:
        raise RuntimeError(f"从 {last_train.date()} 再隔 {embargo} 个交易日已经超出日历")
    return last_train, pd.Timestamp(valid_from)


def assign_day20_week100(paired: pd.DataFrame, cal: pd.DatetimeIndex, cfg) -> pd.DataFrame:
    """Day tower uses a 20-day embargo. Week tower uses 100, and starts later.

    Day validation counts as used. Week training starts 100 trading days after
    that. Week validation and the test set are each another 100 trading days later.
    """
    from images_lib import _first_date_after_n_trading_days, assert_no_split_kline_overlap

    fuse = getattr(cfg, "fuse", None)
    day_embargo = int(getattr(fuse, "day_embargo_days", 20))
    week_embargo = int(getattr(fuse, "embargo_days", 100))
    day_ratio = float(getattr(cfg.data, "train_valid_split", 0.7))
    week_ratio = float(getattr(fuse, "week_train_valid_split", day_ratio))
    train_start = pd.Timestamp(cfg.data.train_start)
    train_end = pd.Timestamp(cfg.data.train_end)
    test_start = pd.Timestamp(cfg.data.test_start)
    test_end = pd.Timestamp(cfg.data.test_end)
    day_span_end = pd.Timestamp(getattr(fuse, "day_span_end", "2017-12-31"))
    cal = pd.DatetimeIndex(pd.to_datetime(cal)).sort_values().unique()

    out = paired.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["day_split"] = "ignore"
    out["week_split"] = "ignore"
    out["split"] = "ignore"

    day_dates = pd.DatetimeIndex(
        out.loc[(out["date"] >= train_start) & (out["date"] <= day_span_end), "date"].unique()
    )
    day_last, day_valid_from = _tv_cut(day_dates, cal, day_embargo, day_ratio)
    day_m = (out["date"] >= train_start) & (out["date"] <= day_span_end)
    out.loc[day_m & (out["date"] <= day_last), "day_split"] = "train"
    out.loc[day_m & (out["date"] >= day_valid_from), "day_split"] = "valid"
    if (out["day_split"] == "valid").sum() == 0 or (out["day_split"] == "train").sum() == 0:
        raise RuntimeError(
            f"日塔在 {train_start.date()} 到 {day_span_end.date()} 里切不出训练和验证。"
            f"间隔是 {day_embargo} 个交易日。"
        )

    day_used_end = pd.Timestamp(out.loc[out["day_split"] == "valid", "date"].max())
    week_from = _first_date_after_n_trading_days(cal, day_used_end, week_embargo)
    if week_from is None or week_from > train_end:
        raise RuntimeError(
            f"日塔验证结束于 {day_used_end.date()}，再隔 {week_embargo} 个交易日"
            f"已经晚于 {train_end.date()}，周塔没有训练日。"
        )
    week_from = pd.Timestamp(week_from)
    week_dates = pd.DatetimeIndex(
        out.loc[(out["date"] >= week_from) & (out["date"] <= train_end), "date"].unique()
    )
    week_last, week_valid_from = _tv_cut(week_dates, cal, week_embargo, week_ratio)
    week_m = (out["date"] >= week_from) & (out["date"] <= train_end)
    out.loc[week_m & (out["date"] <= week_last), "week_split"] = "train"
    out.loc[week_m & (out["date"] >= week_valid_from), "week_split"] = "valid"
    if (out["week_split"] == "train").sum() == 0 or (out["week_split"] == "valid").sum() == 0:
        raise RuntimeError(
            f"周塔在 {week_from.date()} 到 {train_end.date()} 里切不出训练和验证。"
            f"间隔是 {week_embargo} 个交易日。把 fuse.day_span_end 提前，给周塔留出日子。"
        )

    week_used_end = pd.Timestamp(out.loc[out["week_split"] == "valid", "date"].max())
    test_from = _first_date_after_n_trading_days(cal, week_used_end, week_embargo)
    if test_from is None:
        raise RuntimeError(f"周塔验证结束于 {week_used_end.date()}，再隔 {week_embargo} 个交易日超出日历")
    test_from = max(pd.Timestamp(test_from), test_start)
    out.loc[(out["date"] >= test_from) & (out["date"] <= test_end), "split"] = "test"
    if (out["split"] == "test").sum() == 0:
        raise RuntimeError(f"测试集为空。test_from={test_from.date()} test_end={test_end.date()}")

    day_check = out.copy()
    day_check["split"] = out["day_split"]
    if not assert_no_split_kline_overlap(day_check, cal, day_embargo):
        raise RuntimeError(f"日塔切分间隔 {day_embargo} 个交易日不够")
    week_check = out.copy()
    week_check["split"] = out["week_split"]
    week_check.loc[out["split"] == "test", "split"] = "test"
    if not assert_no_split_kline_overlap(week_check, cal, week_embargo):
        raise RuntimeError(f"周塔切分间隔 {week_embargo} 个交易日不够")

    print(
        f"day embargo={day_embargo}  train_end={day_last.date()}  "
        f"valid_from={day_valid_from.date()}  valid_end={day_used_end.date()}  "
        f"day_split={out['day_split'].value_counts().to_dict()}",
        flush=True,
    )
    print(
        f"week embargo={week_embargo}  train_from={week_from.date()}  "
        f"train_end={week_last.date()}  valid_from={week_valid_from.date()}  "
        f"valid_end={week_used_end.date()}  test_from={test_from.date()}  "
        f"week_split={out['week_split'].value_counts().to_dict()}  "
        f"test={(out['split'] == 'test').sum()}",
        flush=True,
    )
    return out


def align_day_week(day_labels: pd.DataFrame, week_labels: pd.DataFrame, cfg) -> pd.DataFrame:
    """Inner join on code and the last completed week. Split is recomputed here.

    The daily label supplies the 5-day excess. The weekly file only supplies
    the image row. Residual mode gives the day tower a 20-day embargo and the
    week tower a later 100-day embargo. Concat keeps one 100-day split.
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
    if fuse_mode(cfg) == "residual":
        paired = assign_day20_week100(paired, cal, cfg)
    else:
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
    extra = ""
    if "day_split" in paired.columns:
        extra = (
            f"  day_split={paired['day_split'].value_counts().to_dict()}"
            f"  week_split={paired['week_split'].value_counts().to_dict()}"
        )
    print(
        f"paired {len(paired)}  day rows {before}  week rows {len(week)}  "
        f"week-end days {int(same.sum())}  other days {int((~same).sum())}  "
        f"split={paired['split'].value_counts().to_dict()}{extra}",
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
