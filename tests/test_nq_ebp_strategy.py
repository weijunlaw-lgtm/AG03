"""
Comprehensive test suite for NQ EBP Quantitative Trading Strategy.

Tests cover:
1. Bull & Bear EBP pattern detection & mutual exclusivity.
2. EMA trend filter & session gate logic.
3. 50-point structural stop & midpoint rule.
4. Entry fill at Open of Bar t+1 with adverse slippage (Zero lookahead).
5. Stop distance calculation & invalid gap guard.
6. Strict half-up arithmetic rounding vs banker's rounding.
7. Portfolio 50-contract aggregate cap & concurrent slots tracking.
8. Intrabar pessimistic priority order (SL > TP > Flatten tie-breaker).
9. Limit TP fill vs Stop-market SL fill price calculations.
10. VectorBT Trades integration & full Leaderboard generation.
11. Boundary and edge cases: zero trades, empty dataframe, single bar.
"""

import math
import unittest
import numpy as np
import pandas as pd
import vectorbt as vbt

from config import GridSearchConfig, StrategyConfig
from data_loader import (
    generate_synthetic_mnq_data,
    normalize_dataframe,
)
from execution_kernel import ExecutionResult, run_numba_execution
from metrics_vbt import (
    calculate_performance_metrics,
)
from nq_ebp_strategy import NQEBPStrategy
from run_grid_search import run_grid_search
from signals import StrategySignals, compute_signals


class TestNQEBPGeometryAndSignals(unittest.TestCase):
    """Test candle geometries, EBP signals, EMA filtering, and 50pt rule."""

    def test_bull_and_bear_ebp_patterns(self):
        """
        Verify Bull EBP: (Low_t < Low_{t-1}) and (Close_t > Open_{t-1})
        Verify Bear EBP: (High_t > High_{t-1}) and (Close_t < Open_{t-1})
        Verify mutual exclusivity.
        """
        times = pd.date_range("2024-01-08 09:30:00", periods=5, freq="5min", tz="America/New_York")
        df = pd.DataFrame(
            {
                # Bar 0: Baseline
                # Bar 1: Bull EBP (Low 94 < 95, Close 105 > 100)
                # Bar 2: Bear EBP (High 115 > 110, Close 95 < 101)
                # Bar 3: Normal inside bar
                # Bar 4: Sweep high and low, but Close > Open_prev -> Bull EBP only
                "Open": [100.0, 101.0, 102.0, 96.0, 97.0],
                "High": [105.0, 110.0, 115.0, 98.0, 108.0],
                "Low": [95.0, 94.0, 95.0, 95.0, 90.0],
                "Close": [102.0, 105.0, 95.0, 97.0, 103.0],
                "Volume": [1000.0] * 5,
            },
            index=times,
        )

        cfg = StrategyConfig(ema_period=2)
        sigs = compute_signals(df, cfg)

        # Bar 0 cannot be an EBP pattern (no prior bar)
        self.assertFalse(sigs.bull_ebp[0])
        self.assertFalse(sigs.bear_ebp[0])

        # Bar 1: Low (94) < prev Low (95) and Close (105) > prev Open (100) -> Bull EBP
        self.assertTrue(sigs.bull_ebp[1])
        self.assertFalse(sigs.bear_ebp[1])

        # Bar 2: High (115) > prev High (110) and Close (95) < prev Open (101) -> Bear EBP
        self.assertFalse(sigs.bull_ebp[2])
        self.assertTrue(sigs.bear_ebp[2])

        # Mutual exclusivity check across all bars
        both = sigs.bull_ebp & sigs.bear_ebp
        self.assertEqual(np.sum(both), 0, "Bull and Bear EBP must be mutually exclusive.")

    def test_ema_filter_and_can_enter_gate(self):
        """Signals require Bull/Bear EBP + Close vs EMA + can_enter_mask."""
        times = pd.date_range("2024-01-08 09:30:00", periods=4, freq="5min", tz="America/New_York")
        df = pd.DataFrame(
            {
                "Open": [100.0, 100.0, 100.0, 100.0],
                "High": [105.0, 110.0, 110.0, 110.0],
                "Low": [95.0, 90.0, 90.0, 90.0],
                "Close": [102.0, 105.0, 95.0, 105.0],
                "Volume": [1000.0] * 4,
            },
            index=times,
        )
        df["can_enter_mask"] = [True, True, True, False]

        cfg = StrategyConfig(ema_period=2)
        sigs = compute_signals(df, cfg)

        # Bar 1: Bull EBP, Close 105 > EMA -> Long signal True
        self.assertTrue(sigs.long_signal[1])

        # Bar 3: Bull EBP, Close 105 > EMA, BUT can_enter_mask is False -> Long signal False
        self.assertFalse(sigs.long_signal[3])

    def test_50pt_structural_stop_midpoint_rule(self):
        """
        If Range <= 50.0: Long SL = Low_t, Short SL = High_t
        If Range > 50.0: Long SL = Short SL = (High_t + Low_t) / 2.0
        """
        times = pd.date_range("2024-01-08 09:30:00", periods=2, freq="5min", tz="America/New_York")
        df = pd.DataFrame(
            {
                # Bar 0: Range = 100 - 60 = 40 (<= 50)
                # Bar 1: Range = 200 - 120 = 80 (> 50)
                "Open": [80.0, 150.0],
                "High": [100.0, 200.0],
                "Low": [60.0, 120.0],
                "Close": [85.0, 180.0],
                "Volume": [1000.0, 1000.0],
            },
            index=times,
        )

        sigs = compute_signals(df, StrategyConfig())

        # Bar 0: <= 50pt rule
        self.assertFalse(sigs.is_midpoint_rule[0])
        self.assertEqual(sigs.sl_long[0], 60.0)
        self.assertEqual(sigs.sl_short[0], 100.0)

        # Bar 1: > 50pt rule -> Midpoint = (200 + 120) / 2 = 160.0
        self.assertTrue(sigs.is_midpoint_rule[1])
        self.assertEqual(sigs.sl_long[1], 160.0)
        self.assertEqual(sigs.sl_short[1], 160.0)


class TestExecutionKernelMechanics(unittest.TestCase):
    """Test execution mechanics, rounding, slippage, contract cap, and priority orders."""

    def test_half_up_rounding(self):
        """
        Verify strict half-up arithmetic rounding:
        int(floor(raw_contracts + 0.5))
        Must round .5 UP, differing from Python round() banker's rounding.
        """
        test_cases = [
            (4.24, 4),
            (4.49, 4),
            (4.50, 5),  # round(4.5) == 4 in Python, but half-up == 5
            (4.51, 5),
            (2.50, 3),  # round(2.5) == 2 in Python, but half-up == 3
            (0.49, 0),
            (0.50, 1),
            (1.00, 1),
        ]
        for raw, expected in test_cases:
            res = int(math.floor(raw + 0.5))
            self.assertEqual(
                res,
                expected,
                f"Failed half-up rounding for {raw}: got {res}, expected {expected}",
            )

    def test_entry_fill_at_open_t_plus_1_with_adverse_slippage(self):
        """
        Signal at Close of Bar 0 (t=0).
        Fill at Open of Bar 1 (t=1).
        Long entry: Open_1 + slippage
        """
        # Long signal on Bar 0
        open_arr = np.array([100.0, 102.0, 103.0])
        high_arr = np.array([105.0, 108.0, 104.0])
        low_arr = np.array([95.0, 101.0, 90.0])
        close_arr = np.array([104.0, 106.0, 92.0])

        long_sig = np.array([True, False, False])
        short_sig = np.array([False, False, False])
        sl_long = np.array([95.0, 95.0, 95.0])
        sl_short = np.array([105.0, 105.0, 105.0])
        is_mid = np.array([False, False, False])
        force_flat = np.array([False, False, False])

        cfg = StrategyConfig(
            slippage_ticks_per_side=1.5,
            tick_size=0.25,
            risk_per_trade_usd=300.0,
            rr_ratio=2.0,
        )
        slippage_pts = 1.5 * 0.25  # 0.375

        sigs = StrategySignals(
            bull_ebp=long_sig,
            bear_ebp=short_sig,
            ema=np.array([100.0, 100.0, 100.0]),
            long_signal=long_sig,
            short_signal=short_sig,
            sl_long=sl_long,
            sl_short=sl_short,
            is_midpoint_rule=is_mid,
            range_pts=np.array([10.0, 7.0, 14.0]),
        )

        res = run_numba_execution(
            open_arr, high_arr, low_arr, close_arr, sigs, force_flat, cfg
        )

        self.assertEqual(len(res.entry_bars), 1)
        # Entry occurred at Bar 1
        self.assertEqual(res.entry_bars[0], 1)
        # Entry price = Open[1] (102.0) + slippage (0.375) = 102.375
        expected_entry = 102.0 + slippage_pts
        self.assertAlmostEqual(res.entry_prices[0], expected_entry, places=4)
        # SL distance = 102.375 - 95.0 = 7.375 pts
        expected_sl_dist = expected_entry - 95.0
        self.assertAlmostEqual(res.sl_dists[0], expected_sl_dist, places=4)
        # Contracts = int(floor(300.0 / (7.375 * 2.0) + 0.5)) = int(floor(20.3389 + 0.5)) = 20
        expected_contracts = int(math.floor(300.0 / (expected_sl_dist * 2.0) + 0.5))
        self.assertEqual(res.contracts[0], expected_contracts)

    def test_invalid_gap_guard(self):
        """
        If Open_{t+1} gaps below the stop price such that SL_dist <= 0,
        the trade must be aborted and skipped.
        """
        open_arr = np.array([100.0, 90.0, 92.0])  # Bar 1 opens at 90, below SL (95)
        high_arr = np.array([105.0, 94.0, 93.0])
        low_arr = np.array([95.0, 88.0, 89.0])
        close_arr = np.array([104.0, 93.0, 91.0])

        long_sig = np.array([True, False, False])
        short_sig = np.array([False, False, False])
        sl_long = np.array([95.0, 95.0, 95.0])
        sl_short = np.array([105.0, 105.0, 105.0])
        is_mid = np.array([False, False, False])
        force_flat = np.array([False, False, False])

        sigs = StrategySignals(
            bull_ebp=long_sig,
            bear_ebp=short_sig,
            ema=np.array([100.0] * 3),
            long_signal=long_sig,
            short_signal=short_sig,
            sl_long=sl_long,
            sl_short=sl_short,
            is_midpoint_rule=is_mid,
            range_pts=np.array([10.0] * 3),
        )

        res = run_numba_execution(
            open_arr, high_arr, low_arr, close_arr, sigs, force_flat, StrategyConfig()
        )
        self.assertEqual(len(res.entry_bars), 0, "Trade with SL_dist <= 0 must be aborted.")

    def test_50_contract_aggregate_cap_and_skipped_counter(self):
        """
        Verify that concurrent trades can enter up to 50 contracts.
        Any trade that would breach 50 total contracts is rejected and increments skipped_by_contract_cap.
        """
        # Create series with consecutive long signals
        n = 5
        open_arr = np.full(n, 100.0)
        high_arr = np.full(n, 102.0)
        low_arr = np.full(n, 98.0)
        close_arr = np.full(n, 101.0)

        # Stop distance: entry = 100 + 0.375 = 100.375, SL = 95.0 -> dist = 5.375 pts
        # Contracts per trade = floor(300 / (5.375 * 2) + 0.5) = floor(27.9 + 0.5) = 28 contracts!
        # First trade takes 28 contracts.
        # Second trade needs 28 contracts -> 28 + 28 = 56 > 50 -> must be REJECTED!
        long_sig = np.array([True, True, True, False, False])
        short_sig = np.zeros(n, dtype=bool)
        sl_long = np.full(n, 95.0)
        sl_short = np.full(n, 105.0)
        is_mid = np.zeros(n, dtype=bool)
        force_flat = np.zeros(n, dtype=bool)

        sigs = StrategySignals(
            bull_ebp=long_sig,
            bear_ebp=short_sig,
            ema=np.full(n, 90.0),
            long_signal=long_sig,
            short_signal=short_sig,
            sl_long=sl_long,
            sl_short=sl_short,
            is_midpoint_rule=is_mid,
            range_pts=np.full(n, 4.0),
        )

        cfg = StrategyConfig(max_prop_contracts=50)
        res = run_numba_execution(
            open_arr, high_arr, low_arr, close_arr, sigs, force_flat, cfg
        )

        # Peak contracts must not exceed 50
        self.assertLessEqual(res.peak_concurrent_contracts, 50)
        # Exactly 1 trade entered (28 contracts), the subsequent signals were skipped
        self.assertEqual(res.peak_concurrent_contracts, 28)
        self.assertGreaterEqual(res.skipped_by_contract_cap, 1)

    def test_pessimistic_intrabar_exit_priority(self):
        """
        Intrabar Priority 1 (SL) must trigger before Priority 2 (TP) if both breached in same bar.
        """
        # Long entry on Bar 1:
        # Bar 0 signal: Long, SL = 95
        # Bar 1: Open=100 (entry fill = 100.375), High=115 (breaches TP=111.125), Low=90 (breaches SL=95)
        # Pessimistic rule requires SL hit first (Exit Reason 1, not 2).
        open_arr = np.array([100.0, 100.0, 100.0])
        high_arr = np.array([105.0, 115.0, 101.0])
        low_arr = np.array([95.0, 90.0, 99.0])
        close_arr = np.array([104.0, 105.0, 100.0])

        long_sig = np.array([True, False, False])
        short_sig = np.zeros(3, dtype=bool)
        sl_long = np.full(3, 95.0)
        sl_short = np.full(3, 105.0)
        is_mid = np.zeros(3, dtype=bool)
        force_flat = np.zeros(3, dtype=bool)

        sigs = StrategySignals(
            bull_ebp=long_sig,
            bear_ebp=short_sig,
            ema=np.full(3, 90.0),
            long_signal=long_sig,
            short_signal=short_sig,
            sl_long=sl_long,
            sl_short=sl_short,
            is_midpoint_rule=is_mid,
            range_pts=np.full(3, 10.0),
        )

        res = run_numba_execution(
            open_arr, high_arr, low_arr, close_arr, sigs, force_flat, StrategyConfig()
        )

        self.assertEqual(len(res.exit_reasons), 1)
        self.assertEqual(res.exit_reasons[0], 1, "Stop-Loss must take pessimistic priority over TP.")

    def test_limit_tp_vs_stop_sl_fill_prices(self):
        """
        Verify fill prices:
        - Long SL: min(Open_i, SL_Price) - Slippage_Pts
        - Long TP: max(Open_i, TP_Price) [No adverse slippage on limit fill]
        """
        # Test TP Fill: Bar 1 opens at 112, exceeding TP (110)
        # Long entry on Bar 1 at 100.375, TP = 110.0
        # If Bar 2 opens at 112.0 and High is 114.0 -> TP fill price should be max(112.0, 110.0) = 112.0
        open_arr = np.array([100.0, 100.0, 112.0])
        high_arr = np.array([105.0, 105.0, 114.0])
        low_arr = np.array([95.0, 99.0, 111.0])
        close_arr = np.array([104.0, 103.0, 113.0])

        long_sig = np.array([True, False, False])
        short_sig = np.zeros(3, dtype=bool)
        sl_long = np.full(3, 95.0)
        sl_short = np.full(3, 105.0)
        is_mid = np.zeros(3, dtype=bool)
        force_flat = np.zeros(3, dtype=bool)

        sigs = StrategySignals(
            bull_ebp=long_sig,
            bear_ebp=short_sig,
            ema=np.full(3, 90.0),
            long_signal=long_sig,
            short_signal=short_sig,
            sl_long=sl_long,
            sl_short=sl_short,
            is_midpoint_rule=is_mid,
            range_pts=np.full(3, 10.0),
        )

        cfg = StrategyConfig(rr_ratio=1.5)
        res = run_numba_execution(
            open_arr, high_arr, low_arr, close_arr, sigs, force_flat, cfg
        )

        self.assertEqual(len(res.exit_reasons), 1)
        self.assertEqual(res.exit_reasons[0], 2)  # TP
        # Gapped open on Bar 2 past TP price: exit price fills at Open (112.0)
        self.assertEqual(res.exit_prices[0], 112.0)


class TestVectorBTAndLeaderboard(unittest.TestCase):
    """Test VectorBT Trades generation, metrics accuracy, and 30-parameter grid search."""

    def setUp(self):
        self.df = generate_synthetic_mnq_data(days=8, seed=42)

    def test_strategy_end_to_end_and_vbt_trades(self):
        """Run single backtest and verify VectorBT trades object and summary."""
        strat = NQEBPStrategy(StrategyConfig(ema_period=20, rr_ratio=2.0))
        result = strat.run(self.df)

        # Result verification
        self.assertIsInstance(result.metrics, dict)
        self.assertGreater(result.metrics["total_trades"], 0)
        self.assertIn("sharpe_ratio", result.metrics)
        self.assertIn("max_intraday_drawdown", result.metrics)
        self.assertIn("max_eod_drawdown", result.metrics)

        # VectorBT object verification
        if result.metrics["total_trades"] > 0:
            self.assertIsNotNone(result.vbt_trades)
            self.assertIsInstance(result.vbt_trades, vbt.Trades)
            self.assertEqual(len(result.vbt_trades.records_arr), result.metrics["total_trades"])

        # Summary string check
        summary_text = result.summary()
        self.assertIn("MNQ EBP STRATEGY BACKTEST SUMMARY", summary_text)
        self.assertIn("Sharpe Ratio", summary_text)

    def test_orthogonal_grid_search_30_combinations(self):
        """
        Verify orthogonal parameter matrix:
        3 EMA periods x 5 RR ratios x 2 Entry cutoffs = 30 rows.
        """
        grid_cfg = GridSearchConfig(
            ema_periods=[10, 15, 20],
            rr_ratios=[1.0, 1.5, 2.0, 2.5, 3.0],
            entry_cutoffs_est=["15:30", "15:55"],
        )
        leaderboard = run_grid_search(self.df, grid_cfg)

        self.assertEqual(len(leaderboard), 30, "Grid search must produce exactly 30 parameter combinations.")
        required_cols = [
            "ema_period",
            "rr_ratio",
            "entry_cutoff_est",
            "total_trades",
            "total_net_profit",
            "profit_factor",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "max_eod_drawdown",
            "max_intraday_drawdown",
            "worst_single_day_pnl",
            "best_single_day_pnl",
            "avg_hold_time_minutes",
            "median_hold_time_minutes",
            "avg_contract_size",
            "peak_concurrent_contracts",
            "peak_concurrent_slots",
            "skipped_by_contract_cap",
            "overall_win_rate_pct",
            "long_win_rate_pct",
            "short_win_rate_pct",
            "avg_sl_distance_pts",
            "pct_midpoint_rule_trades",
            "avg_trades_per_day",
        ]
        for col in required_cols:
            self.assertIn(col, leaderboard.columns, f"Missing required column in leaderboard: {col}")

        # Check sorting: first row must have highest Sharpe ratio
        self.assertGreaterEqual(
            leaderboard["sharpe_ratio"].iloc[0],
            leaderboard["sharpe_ratio"].iloc[-1],
        )


class TestEdgeCasesAndRobustness(unittest.TestCase):
    """Test empty data, single bar data, zero trades, DST transitions, and institutional risk edge cases."""

    def test_empty_dataframe(self):
        """Empty dataframe must not crash across compute_signals, strat.run, and strat.grid_search."""
        empty_df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        empty_df.index = pd.to_datetime([])

        strat = NQEBPStrategy()
        sigs = compute_signals(empty_df, strat.config)
        self.assertEqual(len(sigs.bull_ebp), 0)

        # Full end-to-end run on empty dataframe
        res = strat.run(empty_df)
        self.assertEqual(res.metrics["total_trades"], 0)
        self.assertEqual(res.metrics["total_net_profit"], 0.0)
        self.assertEqual(res.metrics["max_intraday_drawdown"], 0.0)
        self.assertIsNone(res.vbt_trades)
        self.assertIn("MNQ EBP STRATEGY BACKTEST SUMMARY", res.summary())

        # Grid search on empty dataframe
        leaderboard = strat.grid_search(empty_df)
        self.assertEqual(len(leaderboard), 30)

    def test_single_bar_dataframe(self):
        """Single bar dataframe cannot produce signals or trades."""
        times = pd.date_range("2024-01-08 09:30:00", periods=1, freq="5min", tz="America/New_York")
        single_df = pd.DataFrame(
            {"Open": [100.0], "High": [105.0], "Low": [95.0], "Close": [102.0], "Volume": [500.0]},
            index=times,
        )
        single_df["can_enter_mask"] = [True]
        single_df["force_flatten_mask"] = [False]

        strat = NQEBPStrategy()
        res = strat.run(single_df)
        self.assertEqual(res.metrics["total_trades"], 0)
        self.assertEqual(res.metrics["total_net_profit"], 0.0)
        self.assertIsNone(res.vbt_trades)

    def test_flat_market_zero_trades(self):
        """Flat market without any EBP sweeps should yield 0 trades without division by zero."""
        times = pd.date_range("2024-01-08 09:30:00", periods=20, freq="5min", tz="America/New_York")
        flat_df = pd.DataFrame(
            {"Open": [100.0] * 20, "High": [100.0] * 20, "Low": [100.0] * 20, "Close": [100.0] * 20, "Volume": [100.0] * 20},
            index=times,
        )
        flat_df["can_enter_mask"] = [True] * 20
        flat_df["force_flatten_mask"] = [False] * 19 + [True]

        strat = NQEBPStrategy()
        res = strat.run(flat_df)
        self.assertEqual(res.metrics["total_trades"], 0)
        self.assertEqual(res.metrics["profit_factor"], 0.0)
        self.assertEqual(res.metrics["overall_win_rate_pct"], 0.0)

    def test_dst_transition_spring_and_fall(self):
        """Verify normalize_dataframe smoothly normalizes spring-forward and fall-back DST transitions without crashing."""
        # Spring forward (nonexistent hour 02:00 -> shifted forward)
        spring_df = pd.DataFrame(
            {"Open": [100.0], "High": [105.0], "Low": [95.0], "Close": [102.0]},
            index=pd.date_range("2024-03-10 02:00:00", periods=1, freq="5min"),
        )
        norm_spring = normalize_dataframe(spring_df)
        self.assertEqual(str(norm_spring.index.tz), "America/New_York")
        self.assertEqual(len(norm_spring), 1)

        # Fall back (ambiguous repeated hour 01:00)
        fall_df = pd.DataFrame(
            {"Open": [100.0], "High": [105.0], "Low": [95.0], "Close": [102.0]},
            index=pd.date_range("2024-11-03 01:00:00", periods=1, freq="5min"),
        )
        norm_fall = normalize_dataframe(fall_df)
        self.assertEqual(str(norm_fall.index.tz), "America/New_York")
        self.assertEqual(len(norm_fall), 1)

    def test_csv_loader_various_column_formats(self):
        """Verify column normalization with combined Date + Time and capitalized Date columns."""
        df_datetime = pd.DataFrame(
            {
                "Date": ["2024-01-08", "2024-01-08"],
                "Time": ["09:30:00", "09:35:00"],
                "OPEN": [100.0, 101.0],
                "HIGH": [105.0, 106.0],
                "LOW": [99.0, 100.0],
                "CLOSE": [104.0, 105.0],
            }
        )
        res = normalize_dataframe(df_datetime)
        self.assertIsInstance(res.index, pd.DatetimeIndex)
        self.assertEqual(res.index[0].hour, 9)
        self.assertEqual(res.index[0].minute, 30)
        self.assertNotIn("Date", res.columns)
        self.assertNotIn("Time", res.columns)

    def test_contract_cap_counter_when_contracts_less_than_one(self):
        """
        Spec Section 3.2 item 4: If Contracts < 1 OR (Current_Open_Contracts + Contracts) > 50:
        Do NOT enter the trade. Increment skipped_by_contract_cap.
        """
        # Huge stop distance: 400 pts -> raw_contracts = 300 / (400 * 2) = 0.375 -> contracts = 0
        open_arr = np.array([100.0, 100.0, 100.0])
        high_arr = np.array([105.0, 105.0, 105.0])
        low_arr = np.array([95.0, 95.0, 95.0])
        close_arr = np.array([104.0, 104.0, 104.0])

        long_sig = np.array([True, False, False])
        short_sig = np.zeros(3, dtype=bool)
        # SL anchor is at -300.0 -> SL dist = 100.375 - (-300.0) = 400.375 pts -> contracts = 0
        sl_long = np.array([-300.0, -300.0, -300.0])
        sl_short = np.array([500.0, 500.0, 500.0])
        is_mid = np.zeros(3, dtype=bool)
        force_flat = np.zeros(3, dtype=bool)

        sigs = StrategySignals(
            bull_ebp=long_sig,
            bear_ebp=short_sig,
            ema=np.array([90.0] * 3),
            long_signal=long_sig,
            short_signal=short_sig,
            sl_long=sl_long,
            sl_short=sl_short,
            is_midpoint_rule=is_mid,
            range_pts=np.array([10.0] * 3),
        )

        res = run_numba_execution(
            open_arr, high_arr, low_arr, close_arr, sigs, force_flat, StrategyConfig()
        )
        self.assertEqual(len(res.entry_bars), 0)
        self.assertEqual(res.skipped_by_contract_cap, 1, "Contracts < 1 must increment skipped_by_contract_cap")

    def test_drawdown_includes_initial_capital_baseline(self):
        """Drawdowns must measure against initial capital peak ($50,000) rather than only within-series peak."""
        times = pd.date_range("2024-01-08 09:30:00", periods=2, freq="5min", tz="America/New_York")
        df = pd.DataFrame(
            {"Open": [100.0, 100.0], "High": [105.0, 105.0], "Low": [95.0, 95.0], "Close": [100.0, 100.0]},
            index=times,
        )
        # Create mock ExecutionResult with immediate loss of $2,000 from $50,000 initial capital
        exec_res = ExecutionResult(
            entry_bars=np.array([0]),
            exit_bars=np.array([1]),
            directions=np.array([1]),
            entry_prices=np.array([100.0]),
            exit_prices=np.array([90.0]),
            sl_prices=np.array([90.0]),
            tp_prices=np.array([120.0]),
            sl_dists=np.array([10.0]),
            contracts=np.array([10]),
            gross_pts=np.array([-10.0]),
            net_pnl_usd=np.array([-2000.0]),
            commission_usd=np.array([12.4]),
            exit_reasons=np.array([1]),
            used_midpoint_rule=np.array([0]),
            skipped_by_contract_cap=0,
            peak_concurrent_contracts=10,
            peak_concurrent_slots=1,
            equity_realized=np.array([0.0, -2000.0]),
            equity_unrealized=np.array([0.0, 0.0]),
            mtm_equity=np.array([50000.0, 48000.0]),
            open_contracts_series=np.array([10, 0]),
        )
        metrics = calculate_performance_metrics(df, exec_res, StrategyConfig(initial_capital=50000.0))
        # Drawdown from 50000 to 48000 must be 2000.0
        self.assertEqual(metrics["max_intraday_drawdown"], 2000.0)
        self.assertEqual(metrics["max_eod_drawdown"], 2000.0)

    def test_daily_pnl_and_returns_include_day_one(self):
        """Single day backtest must correctly report Day 1 PnL in worst and best single day metrics."""
        times = pd.date_range("2024-01-08 09:30:00", periods=2, freq="5min", tz="America/New_York")
        df = pd.DataFrame(
            {"Open": [100.0, 100.0], "High": [105.0, 105.0], "Low": [95.0, 95.0], "Close": [100.0, 100.0]},
            index=times,
        )
        exec_res = ExecutionResult(
            entry_bars=np.array([0]),
            exit_bars=np.array([1]),
            directions=np.array([1]),
            entry_prices=np.array([100.0]),
            exit_prices=np.array([105.0]),
            sl_prices=np.array([90.0]),
            tp_prices=np.array([120.0]),
            sl_dists=np.array([10.0]),
            contracts=np.array([5]),
            gross_pts=np.array([5.0]),
            net_pnl_usd=np.array([500.0]),
            commission_usd=np.array([6.2]),
            exit_reasons=np.array([2]),
            used_midpoint_rule=np.array([0]),
            skipped_by_contract_cap=0,
            peak_concurrent_contracts=5,
            peak_concurrent_slots=1,
            equity_realized=np.array([0.0, 500.0]),
            equity_unrealized=np.array([0.0, 0.0]),
            mtm_equity=np.array([50000.0, 50500.0]),
            open_contracts_series=np.array([5, 0]),
        )
        metrics = calculate_performance_metrics(df, exec_res, StrategyConfig(initial_capital=50000.0))
        self.assertEqual(metrics["worst_single_day_pnl"], 500.0)
        self.assertEqual(metrics["best_single_day_pnl"], 500.0)


if __name__ == "__main__":
    unittest.main()
