# NQ EBP Strategy Backtest Report (2020–2026)

## Executive Summary
- **Target Asset:** Micro E-mini Nasdaq-100 Futures (`MNQ`)
- **Resolution:** 5-Minute Intraday Bars (`5m`)
- **Data Horizon:** 2020-01-01 → 2026-10-19 (460,058 bars)
- **Initial Capital Baseline:** $50,000.00 USD
- **Risk Per Trade:** $300.00 USD
- **Portfolio Dynamic Cap:** 50 MNQ Contracts

---

## Performance Tearsheet

| Metric | Value |
| :--- | :--- |
| **Total Trades** | 956 |
| **Active Trading Days** | 2,118 |
| **Trades Per Day** | 0.45 |
| **Overall Win Rate** | 36.51% |
| **Profit Factor** | 1.0699 |
| **Total Net Profit** | $13,419.10 |
| **Gross Profit** | $205,477.58 |
| **Gross Loss** | $-192,058.48 |
| **Total Commissions Paid** | $10,208.92 |
| **Annualized Sharpe Ratio** | 0.3762 |
| **Sortino Ratio** | 0.2657 |
| **Calmar Ratio** | 0.1423 |
| **Max Intraday Drawdown** | $14,291.34 |
| **Max End-of-Day (EOD) Drawdown** | $13,865.99 |
| **Worst Single Day PnL** | $-1,567.52 |
| **Best Single Day PnL** | $1,914.62 |

---

## Directional & Execution Analytics

| Parameter | Value |
| :--- | :--- |
| **Long Trades Count & Win Rate** | 486 (36.21%) |
| **Short Trades Count & Win Rate** | 470 (36.81%) |
| **Average Hold Time** | 39.1 mins (Median: 20.0 mins) |
| **Average Contract Sizing** | 8.61 contracts (Long: 8.80, Short: 8.42) |
| **Peak Concurrent Contracts** | 29 / 50 contracts |
| **Peak Concurrent Open Slots** | 3 |
| **Trades Skipped by 50-Contract Cap** | 0 |
| **Average Stop-Loss Distance** | 19.27 points |
| **Trades Triggering 50pt Midpoint Rule** | 2.72% |

---

## Deliverables Generated
- **Interactive Dashboard:** [`backtest_results.html`](./backtest_results.html)
- **Trade Ledger CSV:** [`trades.csv`](./trades.csv)
