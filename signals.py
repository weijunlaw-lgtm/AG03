"""
Vectorized signal computation and structural stop-loss geometries for MNQ EBP.

Implements:
- Bull EBP: (Low_t < Low_{t-1}) & (Close_t > Open_{t-1})
- Bear EBP: (High_t > High_{t-1}) & (Close_t < Open_{t-1})
- EMA Trend Filter: Close_t > EMA for Long, Close_t < EMA for Short
- Session Gate: can_enter_mask_t
- 50-Point Midpoint Rule:
  If High_t - Low_t > 50.0: SL = (High_t + Low_t) / 2.0
  Else: Long SL = Low_t, Short SL = High_t
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

from config import StrategyConfig


@dataclass
class StrategySignals:
    """Vectorized strategy signals and price anchors ready for execution kernel."""

    bull_ebp: np.ndarray  # bool[:]
    bear_ebp: np.ndarray  # bool[:]
    ema: np.ndarray  # float64[:]
    long_signal: np.ndarray  # bool[:]
    short_signal: np.ndarray  # bool[:]
    sl_long: np.ndarray  # float64[:]
    sl_short: np.ndarray  # float64[:]
    is_midpoint_rule: np.ndarray  # bool[:]
    range_pts: np.ndarray  # float64[:]


def compute_ema(series: pd.Series, period: int) -> np.ndarray:
    """Compute exponential moving average using standard recursive formula."""
    return series.ewm(span=period, adjust=False).mean().to_numpy(dtype=np.float64)


def compute_signals(
    df: pd.DataFrame, config: Optional[StrategyConfig] = None
) -> StrategySignals:
    """
    Compute vectorized signals and structural stop anchors evaluated at Close of Bar t.
    All calculations strictly use data available up to Bar t (Zero Lookahead Bias).
    """
    cfg = config or StrategyConfig()

    open_s = df["Open"]
    high_s = df["High"]
    low_s = df["Low"]
    close_s = df["Close"]

    n = len(df)
    if n < 2:
        # Trivial edge case: empty or 1-bar data
        empty_bool = np.zeros(n, dtype=np.bool_)
        empty_float = np.zeros(n, dtype=np.float64)
        return StrategySignals(
            bull_ebp=empty_bool,
            bear_ebp=empty_bool,
            ema=empty_float,
            long_signal=empty_bool,
            short_signal=empty_bool,
            sl_long=empty_float,
            sl_short=empty_float,
            is_midpoint_rule=empty_bool,
            range_pts=empty_float,
        )

    # 1. Previous bar geometries (shifted by 1 bar)
    prev_low = low_s.shift(1).to_numpy(dtype=np.float64)
    prev_high = high_s.shift(1).to_numpy(dtype=np.float64)
    prev_open = open_s.shift(1).to_numpy(dtype=np.float64)

    curr_low = low_s.to_numpy(dtype=np.float64)
    curr_high = high_s.to_numpy(dtype=np.float64)
    curr_close = close_s.to_numpy(dtype=np.float64)

    # 2. EBP Patterns
    # Bull EBP: sweep below prev low and close above prev open
    bull_ebp = (curr_low < prev_low) & (curr_close > prev_open)
    bull_ebp[0] = False  # No prev bar on bar 0

    # Bear EBP: sweep above prev high and close below prev open
    bear_ebp = (curr_high > prev_high) & (curr_close < prev_open)
    bear_ebp[0] = False

    # 3. EMA Filter
    ema = compute_ema(close_s, cfg.ema_period)

    # 4. Session Mask
    if "can_enter_mask" in df.columns:
        can_enter = df["can_enter_mask"].to_numpy(dtype=np.bool_)
    else:
        can_enter = np.ones(n, dtype=np.bool_)

    # 5. Long & Short Entry Signals
    long_signal = bull_ebp & (curr_close > ema) & can_enter
    short_signal = bear_ebp & (curr_close < ema) & can_enter

    # 6. Stop-Loss Structural Price & 50-Point Rule
    range_pts = curr_high - curr_low
    midpoint = (curr_high + curr_low) / 2.0
    is_midpoint_rule = range_pts > cfg.large_candle_threshold_pts

    sl_long = np.where(is_midpoint_rule, midpoint, curr_low)
    sl_short = np.where(is_midpoint_rule, midpoint, curr_high)

    return StrategySignals(
        bull_ebp=bull_ebp,
        bear_ebp=bear_ebp,
        ema=ema,
        long_signal=long_signal,
        short_signal=short_signal,
        sl_long=sl_long,
        sl_short=sl_short,
        is_midpoint_rule=is_midpoint_rule,
        range_pts=range_pts,
    )
