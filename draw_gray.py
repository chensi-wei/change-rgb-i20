"""Grayscale candlestick renderer. Width is n_bars * 3.

Body fills all 3 columns between open and close. Wick is the middle column
from low to high. Volume is the middle column. DIF is drawn on the left
column, DEA on the right column, and the MACD histogram on the middle column.
Background is black. Each role uses its own gray level.
"""
from __future__ import annotations

import numpy as np

UP = np.uint8(255)
DOWN = np.uint8(96)
WICK = np.uint8(176)
VOL_UP = np.uint8(220)
VOL_DOWN = np.uint8(48)
DIF_GRAY = np.uint8(200)
DEA_GRAY = np.uint8(140)
HIST_POS = np.uint8(255)
HIST_NEG = np.uint8(64)

IMAGE_HEIGHT = 100
PRICE_ROWS = 70
VOLUME_ROWS = 15
MACD_ROWS = 15
PX_PER_DAY = 3
WINDOW = 20


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    """``pandas.Series.ewm(span=span, adjust=False).mean()``."""
    alpha = 2.0 / (span + 1.0)
    out = np.empty(len(x), dtype=np.float64)
    out[0] = x[0]
    one_m = 1.0 - alpha
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + one_m * out[i - 1]
    return out


def macd_lines(close: np.ndarray, fast: int = 12, slow: int = 26,
               signal: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = np.asarray(close, dtype=np.float64)
    dif = _ema(c, fast) - _ema(c, slow)
    dea = _ema(dif, signal)
    hist = dif - dea
    return dif, dea, hist


def macd_window(ret_all: np.ndarray, start_ix: int, end_ix: int,
                fast: int = 12, slow: int = 26, signal: int = 9,
                warmup: int = 105) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD on the return path, warmed up with closes before the window.

    The anchor close (first bar of the window) is 1. Only the window slice
    is returned.
    """
    hist_start = max(0, int(start_ix) - int(warmup))
    rets = np.asarray(ret_all[hist_start:end_ix + 1], dtype=np.float64).copy()
    rets[0] = 0.0
    rets = np.where(np.isfinite(rets), rets, 0.0)
    level = np.cumprod(1.0 + rets)
    anchor_i = int(start_ix) - hist_start
    base = level[anchor_i]
    if not np.isfinite(base) or base == 0.0:
        close = level[anchor_i:]
        return macd_lines(close, fast, slow, signal)
    level = level / base
    dif, dea, hist = macd_lines(level, fast, slow, signal)
    return dif[anchor_i:], dea[anchor_i:], hist[anchor_i:]


def _price_to_row(prices: np.ndarray, price_min: float, price_max: float,
                  n_rows: int) -> np.ndarray:
    out = np.full(prices.shape, -1, dtype=np.int64)
    valid = np.isfinite(prices)
    if price_max > price_min and valid.any():
        scaled = (prices[valid] - price_min) / (price_max - price_min)
        scaled = np.clip(scaled, 0.0, 1.0)
        rows = np.round((1.0 - scaled) * (n_rows - 1)).astype(np.int64)
        out[valid] = rows
    elif valid.any():
        out[valid] = (n_rows - 1) // 2
    return out


def _signed_to_row(values: np.ndarray, n_rows: int) -> np.ndarray:
    v = np.clip(values, -1.0, 1.0)
    return np.round((1.0 - v) * 0.5 * (n_rows - 1)).astype(np.int64)


def _paint_span(img: np.ndarray, top: np.ndarray, bot: np.ndarray,
                cols: np.ndarray, gray: np.ndarray) -> None:
    if len(top) == 0:
        return
    span = int((bot - top).max()) + 1
    if span <= 0:
        return
    offsets = np.arange(span)
    rows = top[:, None] + offsets[None, :]
    mask = rows <= bot[:, None]
    counts = mask.sum(axis=1)
    if int(counts.sum()) == 0:
        return
    img[rows[mask], cols[np.repeat(np.arange(len(top)), counts)]] = gray[
        np.repeat(np.arange(len(top)), counts)]


def _draw_points(img: np.ndarray, row: np.ndarray, col: np.ndarray,
                 gray: int, row_lo: int, row_hi: int) -> None:
    """One pixel per bar on the given column. No segment across other columns."""
    valid = (row >= row_lo) & (row < row_hi)
    if int(valid.sum()) == 0:
        return
    img[row[valid], col[valid]] = np.uint8(gray)


def _draw_polyline(img: np.ndarray, row: np.ndarray, col: np.ndarray,
                   gray: int, row_lo: int, row_hi: int) -> None:
    valid = (row >= row_lo) & (row < row_hi)
    if int(valid.sum()) == 0:
        return
    img[row[valid], col[valid]] = np.uint8(gray)
    idx = np.where(valid)[0]
    width = img.shape[1]
    for a, b in zip(idx[:-1], idx[1:]):
        xs = np.arange(int(col[a]), int(col[b]) + 1)
        if len(xs) < 2:
            continue
        ys = np.round(
            row[a] + (row[b] - row[a]) * (xs - col[a]) / (col[b] - col[a])
        ).astype(np.int64)
        np.clip(ys, row_lo, row_hi - 1, out=ys)
        np.clip(xs, 0, width - 1, out=xs)
        img[ys, xs] = np.uint8(gray)


def render_gray(
    o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
    vol: np.ndarray,
    dif: np.ndarray | None = None,
    dea: np.ndarray | None = None,
    hist: np.ndarray | None = None,
    *,
    image_height: int = IMAGE_HEIGHT,
    px_per_day: int = PX_PER_DAY,
    price_rows: int = PRICE_ROWS,
    volume_rows: int = VOLUME_ROWS,
    macd_rows: int = MACD_ROWS,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> np.ndarray:
    """Return uint8 grayscale image, shape (H, W)."""
    if price_rows + volume_rows + macd_rows != image_height:
        raise ValueError(
            f"panel rows {price_rows}+{volume_rows}+{macd_rows} != {image_height}")
    o = np.asarray(o, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    vol = np.asarray(vol, dtype=np.float64)
    n = len(c)
    width = n * px_per_day
    img = np.zeros((image_height, width), dtype=np.uint8)

    finite = (np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c))
    stack = np.concatenate([o[finite], h[finite], l[finite], c[finite]])
    if stack.size == 0:
        return img
    price_min = float(stack.min())
    price_max = float(stack.max())
    o_r = _price_to_row(o, price_min, price_max, price_rows)
    h_r = _price_to_row(h, price_min, price_max, price_rows)
    l_r = _price_to_row(l, price_min, price_max, price_rows)
    c_r = _price_to_row(c, price_min, price_max, price_rows)

    up = finite & (c > o)
    body_gray = np.where(up, UP, DOWN).astype(np.uint8)
    vol_gray = np.where(up, VOL_UP, VOL_DOWN).astype(np.uint8)

    day = np.arange(n)
    col_open = day * px_per_day
    col_mid = col_open + 1
    col_close = col_open + 2

    ok = finite & (h_r >= 0) & (l_r >= 0)
    if ok.any():
        top = np.minimum(h_r[ok], l_r[ok])
        bot = np.maximum(h_r[ok], l_r[ok])
        _paint_span(img, top, bot, col_mid[ok], np.full(int(ok.sum()), WICK, dtype=np.uint8))

    ok_body = finite & (o_r >= 0) & (c_r >= 0)
    if ok_body.any():
        top = np.minimum(o_r[ok_body], c_r[ok_body])
        bot = np.maximum(o_r[ok_body], c_r[ok_body])
        cols = col_open[ok_body]
        for dc in range(px_per_day):
            _paint_span(img, top, bot, cols + dc, body_gray[ok_body])

    if volume_rows > 0:
        v = vol.copy()
        v[~np.isfinite(v)] = 0.0
        vmax = float(v.max()) if v.size else 0.0
        if vmax > 0 and ok_body.any():
            bar_px = np.clip(np.round(v / vmax * volume_rows).astype(np.int64),
                             0, volume_rows)
            use = ok_body & (bar_px > 0)
            if use.any():
                top = price_rows + (volume_rows - bar_px[use])
                bot = np.full(int(use.sum()), price_rows + volume_rows - 1, dtype=np.int64)
                _paint_span(img, top, bot, col_mid[use], vol_gray[use])

    if macd_rows > 0 and np.all(np.isfinite(c)):
        if dif is None or dea is None or hist is None:
            dif, dea, hist = macd_lines(c, macd_fast, macd_slow, macd_signal)
        dif = np.asarray(dif, dtype=np.float64)
        dea = np.asarray(dea, dtype=np.float64)
        hist = np.asarray(hist, dtype=np.float64)
        scale = max(float(np.nanmax(np.abs(dif))), float(np.nanmax(np.abs(dea))),
                    float(np.nanmax(np.abs(hist))), 1e-12)
        dif = dif / scale
        dea = dea / scale
        hist = hist / scale
        row0 = price_rows + volume_rows
        dif_r = row0 + _signed_to_row(dif, macd_rows)
        dea_r = row0 + _signed_to_row(dea, macd_rows)
        hist_r = row0 + _signed_to_row(hist, macd_rows)
        zero_r = int(row0 + _signed_to_row(np.array([0.0]), macd_rows)[0])
        _draw_points(img, dif_r, col_open, int(DIF_GRAY), row0, image_height)
        _draw_points(img, dea_r, col_close, int(DEA_GRAY), row0, image_height)
        hist_gray = np.where(hist >= 0, HIST_POS, HIST_NEG).astype(np.uint8)
        top = np.minimum(hist_r, zero_r)
        bot = np.maximum(hist_r, zero_r)
        _paint_span(img, top, bot, col_mid, hist_gray)
    return img
