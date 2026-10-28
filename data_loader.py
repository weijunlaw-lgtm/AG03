"""
Data loading, validation, timezone normalization, and synthetic generation for MNQ 5m bars.
"""

from datetime import time
from typing import List, Optional
import numpy as np
import pandas as pd

from config import StrategyConfig


def normalize_dataframe(
    df: pd.DataFrame,
    timezone: str = "America/New_York",
    mt5_shift_hours: Optional[int] = None,
) -> pd.DataFrame:
    """
    Validate, clean, and normalize OHLCV dataframe.
    Ensures datetime index in America/New_York timezone and standard columns.
    """
    df = df.copy()

    # Column name mapping (case-insensitive and stripping angle brackets for MT5)
    col_map = {}
    date_col = None
    time_col = None

    for col in df.columns:
        c_clean = str(col).replace("<", "").replace(">", "").strip().lower()
        if c_clean in ("open", "o"):
            col_map[col] = "Open"
        elif c_clean in ("high", "h"):
            col_map[col] = "High"
        elif c_clean in ("low", "l"):
            col_map[col] = "Low"
        elif c_clean in ("close", "c"):
            col_map[col] = "Close"
        elif c_clean in ("volume", "vol", "v", "tickvol"):
            if "Volume" not in col_map.values():
                col_map[col] = "Volume"
        elif c_clean in ("datetime", "timestamp", "dt"):
            date_col = col
        elif c_clean == "date":
            date_col = col
        elif c_clean == "time":
            time_col = col

    df = df.rename(columns=col_map)
    # Deduplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()]
    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required price columns: {missing}")

    if "Volume" not in df.columns:
        df["Volume"] = 1.0

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if date_col is not None and time_col is not None and date_col in df.columns and time_col in df.columns:
            # Combine Date and Time columns (e.g. '2024-01-08' and '09:30:00' or '2020.01.02')
            df.index = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str))
            df = df.drop(columns=[date_col, time_col])
        elif date_col is not None and date_col in df.columns:
            df.index = pd.to_datetime(df[date_col])
            df = df.drop(columns=[date_col])
        elif time_col is not None and time_col in df.columns:
            df.index = pd.to_datetime(df[time_col])
            df = df.drop(columns=[time_col])
        else:
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError(f"Could not convert index to DatetimeIndex: {e}")

    # Optional MT5 server time shift (e.g., -7 hours to convert Vantage MT5 to America/New_York)
    if mt5_shift_hours is not None and mt5_shift_hours != 0:
        df.index = df.index + pd.Timedelta(hours=mt5_shift_hours)

    # Timezone normalization to America/New_York across DST transitions
    if df.index.tz is None:
        try:
            df.index = df.index.tz_localize(
                timezone, nonexistent="shift_forward", ambiguous="infer"
            )
        except Exception:
            df.index = df.index.tz_localize(
                timezone, nonexistent="shift_forward", ambiguous=False
            )
    else:
        df.index = df.index.tz_convert(timezone)

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.dropna(subset=required)

    # Convert prices to float64
    for col in required:
        df[col] = df[col].astype(np.float64)
    df["Volume"] = df["Volume"].astype(np.float64)

    return df


def compute_session_masks(
    index: pd.DatetimeIndex,
    entry_cutoff_est: str = "15:50",
    flatten_time_est: str = "16:05",
    session_start_est: Optional[str] = "18:00",
    allowed_windows_est: Optional[List[tuple[str, str]]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute vectorized boolean masks for trade entries and session flattens.

    - can_enter_mask: True if current bar timestamp allows new trade entry.
    - force_flatten_mask: True on bars where prop firm overnight risk requires flattening (e.g. 16:05 EST).
    """
    times = index.time

    cutoff_h, cutoff_m = map(int, entry_cutoff_est.split(":"))
    cutoff_t = time(cutoff_h, cutoff_m)

    flatten_h, flatten_m = map(int, flatten_time_est.split(":"))
    flatten_t = time(flatten_h, flatten_m)

    if session_start_est:
        start_h, start_m = map(int, session_start_est.split(":"))
        start_t = time(start_h, start_m)
    else:
        start_t = None

    parsed_windows = []
    if allowed_windows_est:
        for w_start, w_end in allowed_windows_est:
            sh, sm = map(int, w_start.split(":"))
            eh, em = map(int, w_end.split(":"))
            parsed_windows.append((time(sh, sm), time(eh, em)))

    n = len(index)
    can_enter = np.zeros(n, dtype=np.bool_)
    force_flatten = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        t = times[i]
        # Flatten mask: strictly on or after flatten cutoff and before session reopen
        if t == flatten_t:
            force_flatten[i] = True
        elif flatten_t <= t < time(17, 0):
            force_flatten[i] = True

        # Can enter mask:
        if parsed_windows:
            in_window = False
            for w_s, w_e in parsed_windows:
                if w_s <= t <= w_e:
                    in_window = True
                    break
            if in_window:
                can_enter[i] = True
        else:
            if start_t is not None:
                if t < cutoff_t or t >= start_t:
                    can_enter[i] = True
            else:
                if t < cutoff_t:
                    can_enter[i] = True

    return can_enter, force_flatten


def prepare_data(
    df: pd.DataFrame,
    config: Optional[StrategyConfig] = None,
    mt5_shift_hours: Optional[int] = None,
) -> pd.DataFrame:
    """Normalize dataframe and attach session masks."""
    cfg = config or StrategyConfig()
    df_clean = normalize_dataframe(
        df, timezone=cfg.timezone, mt5_shift_hours=mt5_shift_hours
    )
    can_enter, force_flatten = compute_session_masks(
        df_clean.index,
        entry_cutoff_est=cfg.entry_cutoff_est,
        flatten_time_est=cfg.flatten_time_est,
        session_start_est=cfg.session_start_est,
        allowed_windows_est=getattr(cfg, "allowed_windows_est", None),
    )
    df_clean["can_enter_mask"] = can_enter
    df_clean["force_flatten_mask"] = force_flatten
    return df_clean


def load_from_parquet(
    file_path: str, config: Optional[StrategyConfig] = None
) -> pd.DataFrame:
    """Load 5m OHLCV data from Parquet file."""
    df = pd.read_parquet(file_path)
    return prepare_data(df, config)


def load_from_csv(
    file_path: str,
    config: Optional[StrategyConfig] = None,
    mt5_shift_hours: Optional[int] = None,
) -> pd.DataFrame:
    """Load 5m OHLCV data from CSV or TSV file, auto-detecting delimiters and MT5 server time."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline()
        sep = "\t" if "\t" in first_line else ","
        df = pd.read_csv(file_path, sep=sep)
    except Exception:
        df = pd.read_csv(file_path, sep=None, engine="python")

    # If file contains MT5 angle bracket headers, default shift to -7 hours (Vantage MT5 -> NY)
    is_mt5 = any("<date>" in str(c).lower() or "<time>" in str(c).lower() for c in df.columns)
    shift = mt5_shift_hours if mt5_shift_hours is not None else (-7 if is_mt5 else None)

    return prepare_data(df, config, mt5_shift_hours=shift)


def generate_synthetic_mnq_data(
    days: int = 5,
    start_date: str = "2024-01-08",
    initial_price: float = 17500.0,
    seed: int = 42,
    config: Optional[StrategyConfig] = None,
) -> pd.DataFrame:
    """
    Generate realistic synthetic 5-minute intraday bars for MNQ futures.
    Simulates:
    - CME trading sessions (18:00 to 16:05 ET)
    - Realistic intraday volatility and drift
    - Engulfing sweeps to trigger Bull/Bear EBP patterns
    - Wide range bars (>50 pts) and normal bars (<=50 pts) to test the 50pt rule
    """
    np.random.seed(seed)
    cfg = config or StrategyConfig()

    start_dt = pd.to_datetime(start_date).tz_localize(cfg.timezone)

    # Build 5m timestamps across active CME intraday hours
    all_timestamps = []
    current_day = start_dt
    days_generated = 0

    while days_generated < days:
        if current_day.weekday() < 5:  # Mon - Fri
            # Session spans 18:00 prev day to 16:05 current day
            # To simulate intraday bars cleanly: 09:30 to 16:05 plus overnight
            day_str = current_day.strftime("%Y-%m-%d")
            # 5m bars from 09:00 to 16:05 (85 bars per day)
            day_bars = pd.date_range(
                f"{day_str} 09:00:00",
                f"{day_str} 16:05:00",
                freq="5min",
                tz=cfg.timezone,
            )
            all_timestamps.extend(day_bars)
            days_generated += 1
        current_day += pd.Timedelta(days=1)

    index = pd.DatetimeIndex(all_timestamps)
    n = len(index)

    # Random walk with mean reversion and momentum
    current_price = initial_price
    opens = np.empty(n, dtype=np.float64)
    highs = np.empty(n, dtype=np.float64)
    lows = np.empty(n, dtype=np.float64)
    closes = np.empty(n, dtype=np.float64)
    volumes = np.random.uniform(500, 5000, size=n)

    for i in range(n):
        open_p = current_price

        # Periodic injection of large range bars (>50 pts) to test midpoint rule
        if i % 23 == 0:
            bar_range = np.random.uniform(55.0, 95.0)
        else:
            bar_range = np.random.uniform(6.0, 35.0)

        # Periodic injection of intentional EBP sweep patterns
        if i > 0 and (i % 7 == 0):
            # Force Bull EBP: Low_t < Low_{t-1} and Close_t > Open_{t-1}
            prev_low = lows[i - 1]
            prev_open = opens[i - 1]
            low_p = prev_low - np.random.uniform(2.0, 8.0)
            close_p = max(prev_open + np.random.uniform(3.0, 15.0), low_p + 10.0)
            high_p = max(close_p, open_p) + np.random.uniform(1.0, 5.0)
        elif i > 0 and (i % 11 == 0):
            # Force Bear EBP: High_t > High_{t-1} and Close_t < Open_{t-1}
            prev_high = highs[i - 1]
            prev_open = opens[i - 1]
            high_p = prev_high + np.random.uniform(2.0, 8.0)
            close_p = min(prev_open - np.random.uniform(3.0, 15.0), high_p - 10.0)
            low_p = min(close_p, open_p) - np.random.uniform(1.0, 5.0)
        else:
            # Standard price bar
            direction = np.random.choice([-1.0, 1.0])
            close_p = open_p + direction * (bar_range * np.random.uniform(0.2, 0.8))
            high_p = max(open_p, close_p) + bar_range * np.random.uniform(0.1, 0.4)
            low_p = min(open_p, close_p) - bar_range * np.random.uniform(0.1, 0.4)

        # Round to 0.25 tick
        open_p = np.round(open_p * 4.0) / 4.0
        high_p = np.round(high_p * 4.0) / 4.0
        low_p = np.round(low_p * 4.0) / 4.0
        close_p = np.round(close_p * 4.0) / 4.0

        opens[i] = open_p
        highs[i] = high_p
        lows[i] = low_p
        closes[i] = close_p

        # Next open has slight gap / continuation
        current_price = close_p + np.random.choice([-0.5, -0.25, 0.0, 0.25, 0.5])

    df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        },
        index=index,
    )

    return prepare_data(df, cfg)
