"""
Orthogonal parameter matrix grid search runner for MNQ EBP strategy.

Sweeps:
- ema_period: [10, 15, 20]
- rr_ratio: [1.0, 1.5, 2.0, 2.5, 3.0]
- entry_cutoff_est: ['15:30', '15:55']
Outputs institutional leaderboard sorted by performance.
"""

from dataclasses import replace
from typing import Optional
import itertools
import numpy as np
import pandas as pd

from config import GridSearchConfig, StrategyConfig
from data_loader import compute_session_masks, generate_synthetic_mnq_data, prepare_data
from execution_kernel import run_numba_execution
from metrics_vbt import calculate_performance_metrics
from signals import compute_signals


def run_grid_search(
    df: pd.DataFrame, grid_config: Optional[GridSearchConfig] = None
) -> pd.DataFrame:
    """
    Execute orthogonal parameter sweep across all combinations of EMA, RR, and Entry Cutoff.
    Utilizes precomputed masks and signals to maximize execution speed.
    """
    g_cfg = grid_config or GridSearchConfig()
    base_cfg = g_cfg.base_config

    rows = []

    # Clean / prepare data once
    clean_df = df.copy()

    # Pre-extract numpy price arrays
    open_arr = clean_df["Open"].to_numpy()
    high_arr = clean_df["High"].to_numpy()
    low_arr = clean_df["Low"].to_numpy()
    close_arr = clean_df["Close"].to_numpy()

    # Group combinations to reuse cached signals & masks
    for cutoff in g_cfg.entry_cutoffs_est:
        can_enter, force_flatten = compute_session_masks(
            clean_df.index,
            entry_cutoff_est=cutoff,
            flatten_time_est=base_cfg.flatten_time_est,
            session_start_est=base_cfg.session_start_est,
        )
        clean_df["can_enter_mask"] = can_enter
        clean_df["force_flatten_mask"] = force_flatten

        for ema_p in g_cfg.ema_periods:
            current_cfg = replace(
                base_cfg, ema_period=ema_p, entry_cutoff_est=cutoff
            )
            # Compute signals for this (cutoff, ema_period)
            signals = compute_signals(clean_df, current_cfg)

            for rr in g_cfg.rr_ratios:
                run_cfg = replace(current_cfg, rr_ratio=rr)

                # Fast Numba kernel execution
                exec_res = run_numba_execution(
                    open_arr,
                    high_arr,
                    low_arr,
                    close_arr,
                    signals,
                    force_flatten,
                    run_cfg,
                )

                metrics = calculate_performance_metrics(clean_df, exec_res, run_cfg)

                row = {
                    "ema_period": ema_p,
                    "rr_ratio": rr,
                    "entry_cutoff_est": cutoff,
                    **metrics,
                }
                rows.append(row)

    leaderboard = pd.DataFrame(rows)

    # Sort leaderboard by Sharpe ratio (descending), then total net profit
    if not leaderboard.empty and "sharpe_ratio" in leaderboard.columns:
        leaderboard = leaderboard.sort_values(
            by=["sharpe_ratio", "total_net_profit"], ascending=[False, False]
        ).reset_index(drop=True)

    return leaderboard


if __name__ == "__main__":
    print("Generating synthetic 5m MNQ dataset for demonstration...")
    demo_df = generate_synthetic_mnq_data(days=15, seed=123)
    print(f"Loaded {len(demo_df)} bars across {len(np.unique(demo_df.index.date))} trading days.")

    print("\nRunning orthogonal grid search (30 parameter combinations)...")
    results = run_grid_search(demo_df)

    display_cols = [
        "ema_period",
        "rr_ratio",
        "entry_cutoff_est",
        "total_trades",
        "overall_win_rate_pct",
        "profit_factor",
        "sharpe_ratio",
        "total_net_profit",
        "max_intraday_drawdown",
        "max_eod_drawdown",
        "peak_concurrent_contracts",
        "skipped_by_contract_cap",
    ]

    print("\n=== TOP 5 LEADERBOARD CONFIGURATIONS ===")
    print(results[display_cols].head(5).to_string(index=False))
