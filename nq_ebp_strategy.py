"""
NQ EBP Strategy: Micro E-mini Nasdaq-100 Futures Engulfing / Sweep Bar Pattern Strategy.

A clean, modular, and vectorized backtesting engine in Python utilizing
VectorBT, Numba (@njit), pandas, and numpy.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
import argparse
import pandas as pd
import vectorbt as vbt

from config import GridSearchConfig, StrategyConfig
from data_loader import (
    generate_synthetic_mnq_data,
    load_from_csv,
    load_from_parquet,
    prepare_data,
)
from execution_kernel import ExecutionResult, run_numba_execution
from metrics_vbt import (
    build_vbt_trades,
    calculate_performance_metrics,
    trades_to_dataframe,
)
from run_grid_search import run_grid_search
from signals import StrategySignals, compute_signals


@dataclass
class StrategyResult:
    """Consolidated backtest result for an NQ EBP Strategy run."""

    config: StrategyConfig
    data: pd.DataFrame
    signals: StrategySignals
    execution: ExecutionResult
    metrics: Dict[str, Any]
    trades_df: pd.DataFrame
    vbt_trades: Optional[vbt.Trades]

    def summary(self) -> str:
        """Return formatted summary table of institutional & prop-firm metrics."""
        m = self.metrics
        lines = [
            "==================================================",
            "        MNQ EBP STRATEGY BACKTEST SUMMARY         ",
            "==================================================",
            f"Parameters: EMA={self.config.ema_period} | RR={self.config.rr_ratio} | Cutoff={self.config.entry_cutoff_est}",
            "--------------------------------------------------",
            f"Total Trades:               {m['total_trades']}",
            f"Overall Win Rate:           {m['overall_win_rate_pct']}%",
            f"Profit Factor:              {m['profit_factor']}",
            f"Total Net Profit:           ${m['total_net_profit']:,.2f}",
            f"Gross Profit:               ${m['gross_profit']:,.2f}",
            f"Gross Loss:                 ${m['gross_loss']:,.2f}",
            f"Total Commissions:          ${m['total_commissions']:,.2f}",
            "--------------------------------------------------",
            f"Sharpe Ratio (Annualized):  {m['sharpe_ratio']}",
            f"Sortino Ratio:              {m['sortino_ratio']}",
            f"Calmar Ratio:               {m['calmar_ratio']}",
            f"Max Intraday Drawdown:      ${m['max_intraday_drawdown']:,.2f}",
            f"Max EOD Drawdown:           ${m['max_eod_drawdown']:,.2f}",
            f"Worst Single Day PnL:       ${m['worst_single_day_pnl']:,.2f}",
            f"Best Single Day PnL:        ${m['best_single_day_pnl']:,.2f}",
            "--------------------------------------------------",
            f"Long Trades / Win Rate:     {m['total_long_trades']} ({m['long_win_rate_pct']}%)",
            f"Short Trades / Win Rate:    {m['total_short_trades']} ({m['short_win_rate_pct']}%)",
            f"Avg Hold Time (Mins):       {m['avg_hold_time_minutes']} (Median: {m['median_hold_time_minutes']})",
            f"Avg Contract Size:          {m['avg_contract_size']} (Long: {m['avg_long_contract_size']}, Short: {m['avg_short_contract_size']})",
            f"Peak Open Contracts:        {m['peak_concurrent_contracts']} / {self.config.max_prop_contracts}",
            f"Peak Open Trade Slots:      {m['peak_concurrent_slots']}",
            f"Skipped by Contract Cap:    {m['skipped_by_contract_cap']}",
            f"Avg SL Distance:            {m['avg_sl_distance_pts']} pts",
            f"Trades Using >50pt Rule:    {m['pct_midpoint_rule_trades']}%",
            f"Trades / Active Day:        {m['avg_trades_per_day']}",
            "==================================================",
        ]
        return "\n".join(lines)


class NQEBPStrategy:
    """
    Institutional Micro E-mini Nasdaq-100 Futures EBP Quantitative Strategy.
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()

    def load_data(
        self,
        file_path_or_df: Optional[Union[str, pd.DataFrame]] = None,
        days: int = 15,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Load market data from CSV/Parquet, or generate synthetic 5m data if None.
        """
        if file_path_or_df is None:
            return generate_synthetic_mnq_data(days=days, seed=seed, config=self.config)
        elif isinstance(file_path_or_df, pd.DataFrame):
            return prepare_data(file_path_or_df, self.config)
        elif isinstance(file_path_or_df, str):
            if file_path_or_df.endswith(".parquet"):
                return load_from_parquet(file_path_or_df, self.config)
            elif file_path_or_df.endswith(".csv"):
                return load_from_csv(file_path_or_df, self.config)
            else:
                raise ValueError(f"Unsupported file format: {file_path_or_df}")
        else:
            raise TypeError("Expected file path (str), DataFrame, or None.")

    def run(self, df: Optional[pd.DataFrame] = None) -> StrategyResult:
        """
        Execute full vectorized signals + Numba execution kernel + VectorBT metrics.
        """
        if df is None:
            df = self.load_data()

        prepared_df = prepare_data(df, self.config)
        signals = compute_signals(prepared_df, self.config)

        open_arr = prepared_df["Open"].to_numpy()
        high_arr = prepared_df["High"].to_numpy()
        low_arr = prepared_df["Low"].to_numpy()
        close_arr = prepared_df["Close"].to_numpy()
        force_flatten_arr = prepared_df["force_flatten_mask"].to_numpy()

        exec_res = run_numba_execution(
            open_arr,
            high_arr,
            low_arr,
            close_arr,
            signals,
            force_flatten_arr,
            self.config,
        )

        metrics = calculate_performance_metrics(prepared_df, exec_res, self.config)
        trades_df = trades_to_dataframe(prepared_df, exec_res)
        vbt_trades = build_vbt_trades(prepared_df, exec_res)

        return StrategyResult(
            config=self.config,
            data=prepared_df,
            signals=signals,
            execution=exec_res,
            metrics=metrics,
            trades_df=trades_df,
            vbt_trades=vbt_trades,
        )

    def grid_search(
        self,
        df: Optional[pd.DataFrame] = None,
        grid_config: Optional[GridSearchConfig] = None,
    ) -> pd.DataFrame:
        """Run multi-parameter grid search across the orthogonal parameter space."""
        if df is None:
            df = self.load_data()
        prepared_df = prepare_data(df, self.config)
        g_cfg = grid_config or GridSearchConfig(base_config=self.config)
        return run_grid_search(prepared_df, g_cfg)


def main():
    parser = argparse.ArgumentParser(description="MNQ EBP Strategy Engine")
    parser.add_argument("--file", type=str, default=None, help="Path to CSV or Parquet 5m OHLCV data")
    parser.add_argument("--days", type=int, default=15, help="Number of synthetic days if generating data")
    parser.add_argument("--ema", type=int, default=20, help="EMA period filter (default: 20)")
    parser.add_argument("--rr", type=float, default=2.0, help="Risk-to-reward ratio (default: 2.0)")
    parser.add_argument("--cutoff", type=str, default="15:50", help="Daily entry cutoff EST (default: 15:50)")
    parser.add_argument("--grid", action="store_true", help="Execute 30-combination grid search")
    parser.add_argument("--output", type=str, default=None, help="Save trades or grid results to CSV")

    args = parser.parse_args()

    cfg = StrategyConfig(
        ema_period=args.ema,
        rr_ratio=args.rr,
        entry_cutoff_est=args.cutoff,
    )
    strat = NQEBPStrategy(cfg)

    print("Loading data...")
    data = strat.load_data(args.file, days=args.days)
    print(f"Data ready: {len(data)} bars.")

    if args.grid:
        print("\nExecuting orthogonal grid search...")
        leaderboard = strat.grid_search(data)
        print("\n=== GRID SEARCH TOP 10 LEADERBOARD ===")
        cols = [
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
        ]
        print(leaderboard[cols].head(10).to_string(index=False))
        if args.output:
            leaderboard.to_csv(args.output, index=False)
            print(f"Saved leaderboard to {args.output}")
    else:
        print("\nExecuting single strategy backtest...")
        result = strat.run(data)
        print(result.summary())
        if not result.trades_df.empty:
            print("\nSample Trades (first 5):")
            print(result.trades_df.head(5).to_string(index=False))
            if args.output:
                result.trades_df.to_csv(args.output, index=False)
                print(f"Saved trades to {args.output}")


if __name__ == "__main__":
    main()
