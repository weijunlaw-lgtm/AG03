"""
Unit tests for the AG03 Core Backtesting Engine infrastructure.
Verifies data loading, synthetic data generation, Numba execution kernel,
VectorBT trade extraction, and metrics calculation.
"""

import unittest
import numpy as np
import pandas as pd
from config import StrategyConfig
from data_loader import (
    generate_synthetic_mnq_data,
    prepare_data,
)
from execution_kernel import ExecutionResult, StrategySignals, run_numba_execution
from metrics_vbt import calculate_performance_metrics, trades_to_dataframe


class TestCoreEngineInfrastructure(unittest.TestCase):
    """Test suite ensuring all core backtesting infrastructure is operational."""

    def test_synthetic_data_generation_and_preparation(self):
        """Synthetic data generator generates valid OHLCV and prepare_data creates required columns."""
        df = generate_synthetic_mnq_data(days=3, seed=42)
        self.assertFalse(df.empty)
        self.assertIn("Open", df.columns)
        self.assertIn("High", df.columns)
        self.assertIn("Low", df.columns)
        self.assertIn("Close", df.columns)
        self.assertIn("Volume", df.columns)

        prep_df = prepare_data(df)
        self.assertIn("can_enter_mask", prep_df.columns)
        self.assertIn("force_flatten_mask", prep_df.columns)

    def test_numba_execution_kernel_and_metrics(self):
        """Execution kernel executes trades and metrics compute without errors."""
        n = 100
        open_p = np.full(n, 100.0)
        high_p = np.full(n, 105.0)
        low_p = np.full(n, 95.0)
        close_p = np.full(n, 102.0)

        # Trigger a long trade at bar 10 with SL distance = 10.0 pts (contracts = 300 / (10.375*2) = 14 <= 50)
        long_sig = np.zeros(n, dtype=bool)
        short_sig = np.zeros(n, dtype=bool)
        long_sig[10] = True

        sl_long = np.full(n, 90.0)
        sl_short = np.full(n, 110.0)
        force_flat = np.zeros(n, dtype=bool)

        sigs = StrategySignals(
            long_signal=long_sig,
            short_signal=short_sig,
            sl_long=sl_long,
            sl_short=sl_short,
        )

        cfg = StrategyConfig(rr_ratio=2.0, risk_per_trade_usd=300.0)
        res: ExecutionResult = run_numba_execution(open_p, high_p, low_p, close_p, sigs, force_flat, cfg)

        self.assertGreater(len(res.entry_bars), 0)
        self.assertEqual(res.entry_bars[0], 11)  # Zero lookahead: entered at bar t+1

        # Check metrics calculation
        dates = pd.date_range("2024-01-08 09:30:00", periods=n, freq="5min", tz="America/New_York")
        df = pd.DataFrame({
            "Open": open_p,
            "High": high_p,
            "Low": low_p,
            "Close": close_p,
        }, index=dates)

        trades_df = trades_to_dataframe(df, res)
        self.assertFalse(trades_df.empty)
        self.assertIn("net_pnl_usd", trades_df.columns)

        metrics = calculate_performance_metrics(df, res, cfg)
        self.assertIn("total_trades", metrics)
        self.assertIn("profit_factor", metrics)


if __name__ == "__main__":
    unittest.main()
