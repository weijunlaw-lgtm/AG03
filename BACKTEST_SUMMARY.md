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
| **Total Trades** | 19,322 |
| **Active Trading Days** | 2,118 |
| **Trades Per Day** | 9.12 |
| **Overall Win Rate** | 32.62% |
| **Profit Factor** | 0.8911 |
| **Total Net Profit** | $-445,468.61 |
| **Gross Profit** | $3,644,389.35 |
| **Gross Loss** | $-4,089,857.96 |
| **Total Commissions Paid** | $182,032.00 |
| **Annualized Sharpe Ratio** | 0.0732 |
| **Sortino Ratio** | 0.0586 |
| **Calmar Ratio** | -0.1432 |
| **Max Intraday Drawdown** | $459,113.84 |
| **Max End-of-Day (EOD) Drawdown** | $457,415.63 |
| **Worst Single Day PnL** | $-5,617.54 |
| **Best Single Day PnL** | $5,989.13 |

---

## Directional & Execution Analytics

| Parameter | Value |
| :--- | :--- |
| **Long Trades Count & Win Rate** | 9,678 (32.69%) |
| **Short Trades Count & Win Rate** | 9,644 (32.54%) |
| **Average Hold Time** | 31.2 mins (Median: 15.0 mins) |
| **Average Contract Sizing** | 7.60 contracts (Long: 7.73, Short: 7.46) |
| **Peak Concurrent Contracts** | 42 / 50 contracts |
| **Peak Concurrent Open Slots** | 6 |
| **Trades Skipped by 50-Contract Cap** | 0 |
| **Average Stop-Loss Distance** | 22.73 points |
| **Trades Triggering 50pt Midpoint Rule** | 11.80% |

---

## Deliverables Generated
- **Interactive Dashboard:** [`backtest_results.html`](./backtest_results.html)
- **Trade Ledger CSV:** [`trades.csv`](./trades.csv)
