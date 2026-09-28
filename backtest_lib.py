"""Decile dashboard. Paths use the same 5-sleeve daily MTM as no_rolling."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from train_lib import (
    load_backtest_market,
    normalize_window_key,
    overlapping_group_excess,
    prepend_zero_cum,
    resolve_results_dir,
)

GROUP_COLORS = [
    "#E53935", "#FB8C00", "#FDD835", "#7CB342", "#26A69A",
    "#42A5F5", "#5C6BC0", "#AB47BC", "#8D6E63", "#424242",
]


def assign_deciles(pred_df: pd.DataFrame, n_deciles: int = 10,
                   score_col: str = "pred",
                   date_col: str = "date") -> pd.DataFrame:
    out = pred_df.copy()

    def _qcut(group: pd.DataFrame) -> pd.Series:
        n = len(group)
        if n < 2:
            return pd.Series(np.nan, index=group.index)
        r = group[score_col].rank(method="first")
        if n >= n_deciles:
            q = np.ceil(r / n * n_deciles).clip(1, n_deciles).astype(int)
        else:
            q = (1 + np.round((r - 1) * (n_deciles - 1) / (n - 1))).astype(int)
        return pd.Series(q.to_numpy(), index=group.index, dtype=float)

    out["decile"] = out.groupby(date_col, group_keys=False).apply(_qcut)
    return out


def annualised_return(period_ret: pd.Series, periods_per_year: int = 252) -> float:
    return float(period_ret.mean() * periods_per_year)


def sharpe(period_ret: pd.Series, periods_per_year: int = 252) -> float:
    mu = period_ret.mean()
    sd = period_ret.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(mu / sd * np.sqrt(periods_per_year))


def plot_decile_dashboard(cfg, paths, window_key: str = "I20", n_deciles: int = 10):
    """One PNG: overlapping-sleeve daily MTM path + stats table. Ranked by predicted excess."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    window_key = normalize_window_key(window_key)
    results = resolve_results_dir(paths, cfg)
    pred_path = results / f"pred_{window_key}.parquet"
    if not pred_path.exists():
        local = Path(paths["results"]) / f"pred_{window_key}.parquet"
        if local.exists():
            pred_path = local
        else:
            raise FileNotFoundError(pred_path)
    pred = pd.read_parquet(pred_path)
    pred["date"] = pd.to_datetime(pred["date"])
    test_start = pd.Timestamp(cfg.data.test_start)
    test_end = pd.Timestamp(cfg.data.test_end)
    pred = pred[(pred["date"] >= test_start) & (pred["date"] <= test_end)]
    pred_d = assign_deciles(pred, n_deciles=n_deciles, score_col="pred")
    horizon = int(getattr(cfg.image, "return_horizon_days", 5))
    n_sleeves = int(getattr(cfg.backtest, "n_sleeves", horizon))
    min_sleeves = int(getattr(cfg.backtest, "min_sleeves", n_sleeves))
    ret_wide, mkt, cal = load_backtest_market(cfg, paths)
    rets = overlapping_group_excess(
        pred_d, "decile", ret_wide, mkt, cal,
        horizon=horizon, n_sleeves=n_sleeves, min_sleeves=min_sleeves)
    rets.columns = [f"D{int(c)}" for c in rets.columns]
    days_per_year = int(getattr(cfg.backtest, "trading_days_per_year", 252))

    fig = plt.figure(figsize=(18.0, 6.4))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.7, 1.0], left=0.055, right=0.99,
        top=0.86, bottom=0.12, wspace=0.06)
    ax = fig.add_subplot(gs[0])
    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.set_axis_off()

    table_rows = []
    legend_handles = []
    for g in range(n_deciles):
        decile = n_deciles - g
        col = f"D{decile}"
        color = GROUP_COLORS[g % len(GROUP_COLORS)]
        if col not in rets.columns:
            continue
        r = rets[col].dropna()
        drawn = prepend_zero_cum(r)
        ax.plot(drawn.index, drawn.to_numpy(float) * 100, color=color, lw=1.8)
        legend_handles.append(Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor=color, markeredgecolor=color, markersize=8,
            label=str(g)))
        table_rows.append([
            str(g),
            "全周期",
            f"{float(r.mean()):.5f}",
            f"{annualised_return(r, days_per_year):.2%}",
            f"{sharpe(r, days_per_year):.5f}",
            f"{float((r > 0).mean()):.5f}",
        ])

    ax.axhline(0.0, color="#222222", lw=0.8)
    ax.set_ylabel("累计超额 (%)")
    ax.set_title(f"{window_key} gray week", loc="left", fontsize=13, pad=10)
    ax.grid(True, axis="y", color="#EEEEEE", lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=n_deciles, frameon=False, handletextpad=0.2, columnspacing=0.85)

    headers = ["组号", "周期", "均值", "年化收益", "夏普率", "日胜率"]
    tbl = ax_tbl.table(
        cellText=table_rows, colLabels=headers,
        loc="center", cellLoc="center", bbox=[0.0, 0.08, 1.0, 0.84])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (ri, ci), cell in tbl.get_celld().items():
        cell.set_edgecolor("#E5E5E5")
        cell.set_linewidth(0.5)
        cell.set_height(0.075)
        if ri == 0:
            cell.set_facecolor("#F3F3F3")
        elif ci == 0:
            cell.get_text().set_color(GROUP_COLORS[int(table_rows[ri - 1][0])])

    fig_dir = Path(paths["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / f"{window_key}_decile.png"
    fig.savefig(fig_path, dpi=160, facecolor="white")
    print("wrote", fig_path.resolve())
    plt.show()
    return pd.DataFrame(table_rows, columns=headers)
