"""
Run comprehensive institutional backtest on the full historical 2020-2026 dataset,
exporting trades.csv, backtest_results.html visual suite, and BACKTEST_SUMMARY.md.
"""

import time
import pandas as pd
from config import StrategyConfig
from data_loader import load_from_csv
from nq_ebp_strategy import NQEBPStrategy
from report_generator import generate_html_dashboard


def main():
    print("=" * 70)
    print("  LAUNCHING NQ EBP INSTITUTIONAL BACKTEST (2020 - 2026)")
    print("=" * 70)

    csv_path = "data/nas100_m5_2020_2026.csv"
    print(f"\n[1/4] Loading and normalizing 5m market dataset: {csv_path}")
    t0 = time.time()
    df = load_from_csv(csv_path)
    load_time = time.time() - t0
    print(f"      Loaded {len(df):,} bars ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}) in {load_time:.2f}s")

    cfg = StrategyConfig(
        ema_period=20,
        rr_ratio=2.0,
        entry_cutoff_est="15:50",
        flatten_time_est="16:05",
        risk_per_trade_usd=300.0,
        max_prop_contracts=50,
        slippage_ticks_per_side=1.5,
        commission_rt_per_contract=1.24,
    )
    strat = NQEBPStrategy(cfg)

    print("\n[2/4] Executing zero-lookahead signals and multi-slot Numba kernel...")
    t1 = time.time()
    result = strat.run(df)
    run_time = time.time() - t1
    print(f"      Execution finished in {run_time:.2f}s")

    print("\n[3/4] Exporting trades.csv and generating HTML dashboard...")
    trades_path = "trades.csv"
    result.trades_df.to_csv(trades_path, index=False)
    print(f"      Saved {len(result.trades_df):,} trades to: {trades_path}")

    # Generate MTM equity series
    pnl_by_exit = result.trades_df.groupby("exit_time")["net_pnl_usd"].sum()
    equity_series = 50000.0 + pnl_by_exit.cumsum()
    # Reindex to trade exit timestamps with starting baseline
    start_ts = df.index[0]
    equity_series = pd.concat([pd.Series([50000.0], index=[start_ts]), equity_series])

    dashboard_path = generate_html_dashboard(
        metrics=result.metrics,
        trades_df=result.trades_df,
        equity_series=equity_series,
        output_path="backtest_results.html",
        title="NQ EBP QUANTITATIVE SUITE — Micro E-mini Nasdaq-100 Futures",
    )
    print(f"      Generated visual dashboard: {dashboard_path}")

    print("\n[4/4] Writing BACKTEST_SUMMARY.md report...")
    m = result.metrics
    summary_md = f"""# NQ EBP Strategy Backtest Report (2020–2026)

## Executive Summary
- **Target Asset:** Micro E-mini Nasdaq-100 Futures (`MNQ`)
- **Resolution:** 5-Minute Intraday Bars (`5m`)
- **Data Horizon:** {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')} ({len(df):,} bars)
- **Initial Capital Baseline:** $50,000.00 USD
- **Risk Per Trade:** $300.00 USD
- **Portfolio Dynamic Cap:** 50 MNQ Contracts

---

## Performance Tearsheet

| Metric | Value |
| :--- | :--- |
| **Total Trades** | {m['total_trades']:,} |
| **Active Trading Days** | {m.get('active_days', 0):,} |
| **Trades Per Day** | {m['avg_trades_per_day']:.2f} |
| **Overall Win Rate** | {m['overall_win_rate_pct']:.2f}% |
| **Profit Factor** | {m['profit_factor']:.4f} |
| **Total Net Profit** | ${m['total_net_profit']:,.2f} |
| **Gross Profit** | ${m['gross_profit']:,.2f} |
| **Gross Loss** | ${m['gross_loss']:,.2f} |
| **Total Commissions Paid** | ${m['total_commissions']:,.2f} |
| **Annualized Sharpe Ratio** | {m['sharpe_ratio']:.4f} |
| **Sortino Ratio** | {m['sortino_ratio']:.4f} |
| **Calmar Ratio** | {m['calmar_ratio']:.4f} |
| **Max Intraday Drawdown** | ${m['max_intraday_drawdown']:,.2f} |
| **Max End-of-Day (EOD) Drawdown** | ${m['max_eod_drawdown']:,.2f} |
| **Worst Single Day PnL** | ${m['worst_single_day_pnl']:,.2f} |
| **Best Single Day PnL** | ${m['best_single_day_pnl']:,.2f} |

---

## Directional & Execution Analytics

| Parameter | Value |
| :--- | :--- |
| **Long Trades Count & Win Rate** | {m['total_long_trades']:,} ({m['long_win_rate_pct']:.2f}%) |
| **Short Trades Count & Win Rate** | {m['total_short_trades']:,} ({m['short_win_rate_pct']:.2f}%) |
| **Average Hold Time** | {m['avg_hold_time_minutes']:.1f} mins (Median: {m['median_hold_time_minutes']:.1f} mins) |
| **Average Contract Sizing** | {m['avg_contract_size']:.2f} contracts (Long: {m['avg_long_contract_size']:.2f}, Short: {m['avg_short_contract_size']:.2f}) |
| **Peak Concurrent Contracts** | {m['peak_concurrent_contracts']} / 50 contracts |
| **Peak Concurrent Open Slots** | {m['peak_concurrent_slots']} |
| **Trades Skipped by 50-Contract Cap** | {m['skipped_by_contract_cap']:,} |
| **Average Stop-Loss Distance** | {m['avg_sl_distance_pts']:.2f} points |
| **Trades Triggering 50pt Midpoint Rule** | {m['pct_midpoint_rule_trades']:.2f}% |

---

## Deliverables Generated
- **Interactive Dashboard:** [`backtest_results.html`](./backtest_results.html)
- **Trade Ledger CSV:** [`trades.csv`](./trades.csv)
"""
    with open("BACKTEST_SUMMARY.md", "w", encoding="utf-8") as f:
        f.write(summary_md)
    print("      Saved BACKTEST_SUMMARY.md")

    print("\n" + "=" * 70)
    print(result.summary())
    print("=" * 70)


if __name__ == "__main__":
    main()
