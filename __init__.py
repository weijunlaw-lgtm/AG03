"""
AG03 Quantitative Backtesting & Research Engine.
High-performance Numba + VectorBT trading architecture.
"""

from config import StrategyConfig
from data_loader import (
    compute_session_masks,
    generate_synthetic_mnq_data,
    load_from_csv,
    load_from_parquet,
    normalize_dataframe,
    prepare_data,
)
from execution_kernel import ExecutionResult, StrategySignals, run_numba_execution
from metrics_vbt import (
    build_vbt_trades,
    calculate_performance_metrics,
    trades_to_dataframe,
)
from report_generator import generate_html_dashboard

__all__ = [
    "StrategyConfig",
    "StrategySignals",
    "ExecutionResult",
    "run_numba_execution",
    "calculate_performance_metrics",
    "build_vbt_trades",
    "trades_to_dataframe",
    "generate_synthetic_mnq_data",
    "load_from_parquet",
    "load_from_csv",
    "prepare_data",
    "normalize_dataframe",
    "compute_session_masks",
    "generate_html_dashboard",
]
