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
| **Total Trades** | 39,659 |
| **Active Trading Days** | 2,118 |
| **Trades Per Day** | 18.72 |
| **Overall Win Rate** | 31.99% |
| **Profit Factor** | 0.8170 |
| **Total Net Profit** | $-1,634,448.49 |
| **Gross Profit** | $7,296,048.40 |
| **Gross Loss** | $-8,930,496.89 |
| **Total Commissions Paid** | $771,058.04 |
| **Annualized Sharpe Ratio** | 0.0548 |
| **Sortino Ratio** | 0.0371 |
| **Calmar Ratio** | -0.1462 |
| **Max Intraday Drawdown** | $1,646,538.85 |
| **Max End-of-Day (EOD) Drawdown** | $1,644,352.68 |
| **Worst Single Day PnL** | $-7,355.48 |
| **Best Single Day PnL** | $8,418.25 |

---

## Directional & Execution Analytics

| Parameter | Value |
| :--- | :--- |
| **Long Trades Count & Win Rate** | 21,119 (32.28%) |
| **Short Trades Count & Win Rate** | 18,540 (31.66%) |
| **Average Hold Time** | 23.0 mins (Median: 10.0 mins) |
| **Average Contract Sizing** | 15.68 contracts (Long: 16.23, Short: 15.05) |
| **Peak Concurrent Contracts** | 50 / 50 contracts |
| **Peak Concurrent Open Slots** | 6 |
| **Trades Skipped by 50-Contract Cap** | 5,195 |
| **Average Stop-Loss Distance** | 14.83 points |
| **Trades Triggering 50pt Midpoint Rule** | 6.45% |

---

## Deliverables Generated
- **Interactive Dashboard:** [`backtest_results.html`](./backtest_results.html)
- **Trade Ledger CSV:** [`trades.csv`](./trades.csv)
