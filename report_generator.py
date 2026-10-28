"""
Automated Report and Dashboard Generator for NQ EBP Strategy Backtests.
Produces interactive, standalone HTML dashboards featuring comprehensive
Win vs Loss analytics, MAE/MFE excursions, time-based plots, and robust offline styling.
"""

from typing import Any, Dict
import json
import numpy as np
import pandas as pd


def generate_html_dashboard(
    metrics: Dict[str, Any],
    trades_df: pd.DataFrame,
    equity_series: pd.Series,
    output_path: str = "backtest_results.html",
    title: str = "NQ EBP QUANTITATIVE SUITE — Micro E-mini Nasdaq-100 Futures",
) -> str:
    """
    Generate a self-contained, publication-grade dark-themed HTML report.
    Uses pure embedded CSS (zero external CSS framework dependency) to ensure
    immediate rendering in all environments (offline, airgapped, file://, CDN-blocked).
    Integrates an asynchronous multi-CDN fallback Plotly loader with graceful degradation:
    - Intra-Trade MAE vs MFE Scatter Plot (R-multiples)
    - Time Plot 1: Performance by Hour of Day (EST)
    - Time Plot 2: Performance by Day of Week
    - Time Plot 3: Cumulative Winning vs Losing Trades PnL Timeline
    - Hold Time Duration Analysis
    - Exit Reason Distribution (Pie & Breakdown Table)
    - Core MTM Equity Curve and Underwater Drawdown
    - Comprehensive Side-by-Side Win vs Loss Tearsheet
    """
    # Downsample equity series for smooth web charting
    if len(equity_series) > 10000:
        step = max(1, len(equity_series) // 5000)
        chart_equity = equity_series.iloc[::step].copy()
    else:
        chart_equity = equity_series.copy()

    # Drawdown curve
    peak_equity = chart_equity.cummax()
    drawdown_series = chart_equity - peak_equity

    eq_dates = [ts.strftime("%Y-%m-%d %H:%M") for ts in chart_equity.index]
    eq_vals = [round(float(v), 2) for v in chart_equity.values]
    dd_vals = [round(float(v), 2) for v in drawdown_series.values]

    # Partition Wins and Losses
    wins_df = trades_df[trades_df["net_pnl_usd"] > 0].copy() if not trades_df.empty else pd.DataFrame()
    loss_df = trades_df[trades_df["net_pnl_usd"] <= 0].copy() if not trades_df.empty else pd.DataFrame()

    n_wins = len(wins_df)
    n_losses = len(loss_df)
    total_trades = len(trades_df)

    # Detailed Winning vs Losing Statistics
    win_stats = {
        "count": n_wins,
        "pct": (n_wins / total_trades * 100.0) if total_trades > 0 else 0.0,
        "total_pnl": float(wins_df["net_pnl_usd"].sum()) if n_wins > 0 else 0.0,
        "avg_pnl": float(wins_df["net_pnl_usd"].mean()) if n_wins > 0 else 0.0,
        "median_pnl": float(wins_df["net_pnl_usd"].median()) if n_wins > 0 else 0.0,
        "max_pnl": float(wins_df["net_pnl_usd"].max()) if n_wins > 0 else 0.0,
        "avg_pts": float(wins_df["gross_pts"].mean()) if n_wins > 0 else 0.0,
        "avg_hold": float(wins_df["hold_time_minutes"].mean()) if n_wins > 0 else 0.0,
        "median_hold": float(wins_df["hold_time_minutes"].median()) if n_wins > 0 else 0.0,
        "avg_contracts": float(wins_df["contracts"].mean()) if n_wins > 0 else 0.0,
        "avg_mae_pts": float(wins_df["mae_pts"].mean()) if "mae_pts" in wins_df.columns and n_wins > 0 else 0.0,
        "avg_mfe_pts": float(wins_df["mfe_pts"].mean()) if "mfe_pts" in wins_df.columns and n_wins > 0 else 0.0,
        "avg_mae_r": float(wins_df["mae_r"].mean()) if "mae_r" in wins_df.columns and n_wins > 0 else 0.0,
        "avg_mfe_r": float(wins_df["mfe_r"].mean()) if "mfe_r" in wins_df.columns and n_wins > 0 else 0.0,
        "avg_sl_dist": float(wins_df["sl_distance_pts"].mean()) if n_wins > 0 else 0.0,
    }

    loss_stats = {
        "count": n_losses,
        "pct": (n_losses / total_trades * 100.0) if total_trades > 0 else 0.0,
        "total_pnl": float(loss_df["net_pnl_usd"].sum()) if n_losses > 0 else 0.0,
        "avg_pnl": float(loss_df["net_pnl_usd"].mean()) if n_losses > 0 else 0.0,
        "median_pnl": float(loss_df["net_pnl_usd"].median()) if n_losses > 0 else 0.0,
        "max_pnl": float(loss_df["net_pnl_usd"].min()) if n_losses > 0 else 0.0,
        "avg_pts": float(loss_df["gross_pts"].mean()) if n_losses > 0 else 0.0,
        "avg_hold": float(loss_df["hold_time_minutes"].mean()) if n_losses > 0 else 0.0,
        "median_hold": float(loss_df["hold_time_minutes"].median()) if n_losses > 0 else 0.0,
        "avg_contracts": float(loss_df["contracts"].mean()) if n_losses > 0 else 0.0,
        "avg_mae_pts": float(loss_df["mae_pts"].mean()) if "mae_pts" in loss_df.columns and n_losses > 0 else 0.0,
        "avg_mfe_pts": float(loss_df["mfe_pts"].mean()) if "mfe_pts" in loss_df.columns and n_losses > 0 else 0.0,
        "avg_mae_r": float(loss_df["mae_r"].mean()) if "mae_r" in loss_df.columns and n_losses > 0 else 0.0,
        "avg_mfe_r": float(loss_df["mfe_r"].mean()) if "mfe_r" in loss_df.columns and n_losses > 0 else 0.0,
        "avg_sl_dist": float(loss_df["sl_distance_pts"].mean()) if n_losses > 0 else 0.0,
    }

    win_loss_payoff = abs(win_stats["avg_pnl"] / loss_stats["avg_pnl"]) if loss_stats["avg_pnl"] != 0 else 0.0

    # Hourly Performance Analysis (00:00 to 23:00 EST)
    if not trades_df.empty:
        t_entry = pd.to_datetime(trades_df["entry_time"], utc=True).dt.tz_convert("America/New_York")
        trades_df_copy = trades_df.copy()
        trades_df_copy["entry_hour"] = t_entry.dt.hour
        hourly = trades_df_copy.groupby("entry_hour")["net_pnl_usd"].agg(
            trades="count",
            pnl="sum",
            win_rate=lambda s: float((s > 0).mean() * 100.0),
        ).reset_index()

        all_hours = pd.DataFrame({"entry_hour": range(24)})
        hourly = all_hours.merge(hourly, on="entry_hour", how="left").fillna(0)
        hours_labels = [f"{h:02d}:00" for h in hourly["entry_hour"]]
        hourly_pnl = [round(float(v), 2) for v in hourly["pnl"]]
        hourly_wr = [round(float(v), 1) for v in hourly["win_rate"]]
        hourly_trades = [int(v) for v in hourly["trades"]]

        # Day of Week Performance
        trades_df_copy["dow"] = t_entry.dt.day_name()
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        dow_grouped = trades_df_copy.groupby("dow")["net_pnl_usd"].agg(
            trades="count",
            pnl="sum",
            win_rate=lambda s: float((s > 0).mean() * 100.0),
        ).reindex(dow_order).fillna(0).reset_index()

        dow_labels = list(dow_grouped["dow"])
        dow_pnl = [round(float(v), 2) for v in dow_grouped["pnl"]]
        dow_wr = [round(float(v), 1) for v in dow_grouped["win_rate"]]
        dow_trades = [int(v) for v in dow_grouped["trades"]]

        # Cumulative PnL Timeline for Wins vs Losses
        trades_sorted = trades_df.sort_values("exit_time").copy()
        trades_sorted["win_pnl"] = np.where(trades_sorted["net_pnl_usd"] > 0, trades_sorted["net_pnl_usd"], 0.0)
        trades_sorted["loss_pnl"] = np.where(trades_sorted["net_pnl_usd"] <= 0, trades_sorted["net_pnl_usd"], 0.0)

        # Downsample timeline if large
        step_tl = max(1, len(trades_sorted) // 2500)
        tl_sample = trades_sorted.iloc[::step_tl]
        tl_dates = [str(t)[:16] for t in tl_sample["exit_time"]]
        tl_win_cum = [round(float(v), 2) for v in tl_sample["win_pnl"].cumsum()]
        tl_loss_cum = [round(float(v), 2) for v in tl_sample["loss_pnl"].cumsum()]

        # MAE vs MFE Scatter Data (Sample max 1500 for crisp WebGL / SVG rendering)
        win_sample = wins_df.sample(min(1500, len(wins_df)), random_state=42) if len(wins_df) > 0 else pd.DataFrame()
        loss_sample = loss_df.sample(min(1500, len(loss_df)), random_state=42) if len(loss_df) > 0 else pd.DataFrame()

        win_mae_r = [round(float(v), 3) for v in win_sample.get("mae_r", [])]
        win_mfe_r = [round(float(v), 3) for v in win_sample.get("mfe_r", [])]
        win_text = [
            f"ID: {row['trade_id']}<br>Date: {str(row['entry_time'])[:16]}<br>PnL: ${row['net_pnl_usd']:+,.2f}<br>MAE: {row.get('mae_r', 0):.2f}R | MFE: {row.get('mfe_r', 0):.2f}R<br>Hold: {row['hold_time_minutes']}m | Qty: {row['contracts']}"
            for _, row in win_sample.iterrows()
        ]

        loss_mae_r = [round(float(v), 3) for v in loss_sample.get("mae_r", [])]
        loss_mfe_r = [round(float(v), 3) for v in loss_sample.get("mfe_r", [])]
        loss_text = [
            f"ID: {row['trade_id']}<br>Date: {str(row['entry_time'])[:16]}<br>PnL: ${row['net_pnl_usd']:+,.2f}<br>MAE: {row.get('mae_r', 0):.2f}R | MFE: {row.get('mfe_r', 0):.2f}R<br>Hold: {row['hold_time_minutes']}m | Qty: {row['contracts']}"
            for _, row in loss_sample.iterrows()
        ]

        # Duration buckets
        bins = [-1, 5, 15, 30, 60, 120, 999999]
        bin_labels = ["< 5m", "5-15m", "15-30m", "30-60m", "1-2h", "> 2h"]
        trades_df_copy["dur_bin"] = pd.cut(trades_df_copy["hold_time_minutes"], bins=bins, labels=bin_labels)
        dur_grp = trades_df_copy.groupby("dur_bin", observed=False)["net_pnl_usd"].agg(
            trades="count",
            pnl="sum",
            win_rate=lambda s: float((s > 0).mean() * 100.0) if len(s) > 0 else 0.0,
        ).reset_index()
        dur_labels = list(dur_grp["dur_bin"])
        dur_counts = [int(v) for v in dur_grp["trades"]]
        dur_pnl = [round(float(v), 2) for v in dur_grp["pnl"]]
        dur_wr = [round(float(v), 1) for v in dur_grp["win_rate"]]

    else:
        hours_labels, hourly_pnl, hourly_wr, hourly_trades = [], [], [], []
        dow_labels, dow_pnl, dow_wr, dow_trades = [], [], [], []
        tl_dates, tl_win_cum, tl_loss_cum = [], [], []
        win_mae_r, win_mfe_r, win_text = [], [], []
        loss_mae_r, loss_mfe_r, loss_text = [], [], []
        dur_labels, dur_counts, dur_pnl, dur_wr = [], [], [], []

    # Exit reason breakdown
    exit_counts = trades_df["exit_reason"].value_counts().to_dict() if not trades_df.empty else {}
    exit_labels = list(exit_counts.keys())
    exit_values = [int(v) for v in exit_counts.values()]

    # Sample trades for preview
    sample_trades = trades_df.head(50).to_dict(orient="records") if not trades_df.empty else []

    # Date range string
    horizon_str = (
        f"{chart_equity.index[0].strftime('%Y-%m-%d')} → {chart_equity.index[-1].strftime('%Y-%m-%d')}"
        if len(chart_equity) > 0
        else "N/A"
    )

    # Pre-render static HTML tables so the report displays instantly even without JavaScript
    hourly_rows = "".join([
        f"<tr><td>{hl}</td><td>{trades:,}</td><td class='{('text-emerald' if pnl >= 0 else 'text-rose')}'>${pnl:+,.2f}</td><td>{wr:.1f}%</td></tr>"
        for hl, trades, pnl, wr in zip(hours_labels, hourly_trades, hourly_pnl, hourly_wr)
        if trades > 0
    ])

    dow_rows = "".join([
        f"<tr><td class='font-bold'>{dl}</td><td>{dt:,}</td><td class='{('text-emerald' if dp >= 0 else 'text-rose')}'>${dp:+,.2f}</td><td>{dwr:.1f}%</td></tr>"
        for dl, dt, dp, dwr in zip(dow_labels, dow_trades, dow_pnl, dow_wr)
    ])

    dur_rows = "".join([
        f"<tr><td class='font-bold'>{dl}</td><td>{dc:,}</td><td class='{('text-emerald' if dp >= 0 else 'text-rose')}'>${dp:+,.2f}</td><td>{dw:.1f}%</td></tr>"
        for dl, dc, dp, dw in zip(dur_labels, dur_counts, dur_pnl, dur_wr)
    ])

    exit_rows = "".join([
        f"<tr><td class='font-bold'>{el}</td><td>{ev:,}</td><td>{(ev / max(1, total_trades) * 100):.1f}%</td></tr>"
        for el, ev in zip(exit_labels, exit_values)
    ])

    trade_preview_rows = "".join([
        f"""<tr>
          <td class="text-muted">{t.get("trade_id", "")}</td>
          <td class="font-bold {('text-emerald' if t.get('direction') == 'LONG' else 'text-rose')}">{t.get("direction", "")}</td>
          <td>{str(t.get("entry_time", ""))[:19]}</td>
          <td>{t.get("hold_time_minutes", "")}m</td>
          <td>{t.get("contracts", "")}</td>
          <td>{t.get("entry_price", 0):.2f}</td>
          <td>{t.get("exit_price", 0):.2f}</td>
          <td class="text-amber">{t.get("mae_r", 0):.2f}R</td>
          <td class="text-cyan">{t.get("mfe_r", 0):.2f}R</td>
          <td class="{('text-emerald' if t.get('gross_pts', 0) > 0 else 'text-rose')}">{t.get("gross_pts", 0):+.2f}</td>
          <td class="font-bold {('text-emerald' if t.get('net_pnl_usd', 0) > 0 else 'text-rose')}">${t.get("net_pnl_usd", 0):+,.2f}</td>
          <td class="text-muted">{t.get("exit_reason", "")}</td>
        </tr>"""
        for t in sample_trades
    ])

    # Self-contained HTML with embedded styles (Zero external CSS dependency)
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    :root {{
      --bg-page: #0b1120;
      --bg-card: #0f172a;
      --bg-card-sub: #162036;
      --bg-table-head: #090e1a;
      --border-color: #1e293b;
      --border-highlight: #334155;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --accent-emerald: #10b981;
      --accent-rose: #f43f5e;
      --accent-cyan: #06b6d4;
      --accent-amber: #f59e0b;
      --accent-indigo: #818cf8;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background-color: var(--bg-page);
      color: var(--text-main);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.5;
      padding: 1.25rem;
    }}
    .max-container {{ max-width: 1380px; margin: 0 auto; display: flex; flex-direction: column; gap: 1.5rem; }}
    
    /* Typography & Utility */
    h1 {{ font-size: 1.75rem; font-weight: 800; letter-spacing: -0.025em; color: #fff; }}
    h2 {{ font-size: 1.125rem; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 0.5rem; }}
    p.subtitle {{ font-size: 0.875rem; color: var(--text-muted); margin-top: 0.25rem; }}
    .font-mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
    .font-bold {{ font-weight: 700; }}
    .text-emerald {{ color: var(--accent-emerald) !important; }}
    .text-rose {{ color: var(--accent-rose) !important; }}
    .text-cyan {{ color: var(--accent-cyan) !important; }}
    .text-amber {{ color: var(--accent-amber) !important; }}
    .text-indigo {{ color: var(--accent-indigo) !important; }}
    .text-muted {{ color: var(--text-muted) !important; }}
    .text-dim {{ color: var(--text-dim) !important; }}
    .text-right {{ text-align: right; }}

    /* Cards */
    .card {{
      background-color: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 0.875rem;
      padding: 1.25rem;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
    }}
    .card-banner {{
      background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
      border: 1px solid #3730a3;
      padding: 1.5rem;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
    }}

    /* Badges */
    .badge-group {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.5rem; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 0.25rem 0.65rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      border: 1px solid transparent;
    }}
    .badge-emerald {{ background-color: rgba(16, 185, 129, 0.15); color: #34d399; border-color: rgba(16, 185, 129, 0.3); }}
    .badge-cyan {{ background-color: rgba(6, 182, 212, 0.15); color: #38bdf8; border-color: rgba(6, 182, 212, 0.3); }}
    .badge-indigo {{ background-color: rgba(99, 102, 241, 0.15); color: #a5b4fc; border-color: rgba(99, 102, 241, 0.3); }}

    /* Grid Layouts */
    .grid-kpi {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 1rem;
    }}
    .grid-2col {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
    }}
    @media (max-width: 960px) {{
      .grid-2col {{ grid-template-columns: 1fr; }}
    }}

    /* Stat Cards */
    .stat-title {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); }}
    .stat-value {{ font-size: 1.625rem; font-weight: 800; margin-top: 0.25rem; }}
    .stat-meta {{ font-size: 0.75rem; color: var(--text-dim); margin-top: 0.25rem; }}

    /* Tables */
    .table-container {{
      width: 100%;
      overflow-x: auto;
      border-radius: 0.5rem;
      margin-top: 0.75rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.8125rem;
      text-align: left;
    }}
    th {{
      background-color: var(--bg-table-head);
      color: var(--text-muted);
      text-transform: uppercase;
      font-size: 0.7rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      padding: 0.65rem 0.85rem;
      border-bottom: 1px solid var(--border-color);
      white-space: nowrap;
    }}
    td {{
      padding: 0.6rem 0.85rem;
      border-bottom: 1px solid var(--border-color);
      white-space: nowrap;
    }}
    tr:hover td {{ background-color: rgba(30, 41, 59, 0.5); }}
    .table-ledger {{ max-height: 420px; overflow-y: auto; display: block; }}
    .table-ledger thead {{ position: sticky; top: 0; z-index: 10; }}

    /* Chart Containers */
    .chart-box {{
      width: 100%;
      position: relative;
      border-radius: 0.5rem;
      background: var(--bg-card);
      min-height: 350px;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .chart-placeholder {{
      color: var(--text-dim);
      font-size: 0.8125rem;
      text-align: center;
      padding: 1.5rem;
    }}
    .chart-fallback-notice {{
      background: rgba(15, 23, 42, 0.8);
      border: 1px dashed var(--border-highlight);
      border-radius: 0.5rem;
      padding: 1rem;
      margin-top: 0.5rem;
      font-size: 0.75rem;
      color: var(--text-muted);
    }}

    /* Buttons */
    .btn {{
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.45rem 0.9rem;
      border-radius: 0.5rem;
      font-size: 0.75rem;
      font-weight: 600;
      text-decoration: none;
      cursor: pointer;
      border: none;
      transition: background 0.15s ease;
    }}
    .btn-cyan {{ background-color: #0284c7; color: #fff; }}
    .btn-cyan:hover {{ background-color: #0369a1; }}
  </style>
</head>
<body>
  <div class="max-container">

    <!-- Executive Header Banner -->
    <div class="card card-banner">
      <div>
        <div class="badge-group">
          <span class="badge badge-emerald">Institutional Quant Suite</span>
          <span class="badge badge-cyan">Intra-Trade MAE & MFE Excursions</span>
          <span class="badge badge-indigo">Time & Session Profiling</span>
        </div>
        <h1>{title}</h1>
        <p class="subtitle">
          Zero Lookahead Bias • Multi-Slot Numba Execution Kernel • Dynamic 50-Contract Cap • Realistic Slippage & Commissions
        </p>
      </div>
      <div class="text-right">
        <div class="stat-title">Backtest Horizon</div>
        <div class="stat-value text-cyan" style="font-size: 1.25rem;">{horizon_str}</div>
        <div class="stat-meta">Micro E-mini Nasdaq-100 (MNQ 5-Minute)</div>
      </div>
    </div>

    <!-- KPI Metric Cards Grid -->
    <div class="grid-kpi">
      <div class="card">
        <div class="stat-title">Total Trades</div>
        <div class="stat-value">{total_trades:,}</div>
        <div class="stat-meta">{metrics.get('avg_trades_per_day', 0):.1f} trades / active day</div>
      </div>
      <div class="card">
        <div class="stat-title">Win Rate</div>
        <div class="stat-value text-amber">{metrics.get('overall_win_rate_pct', 0):.2f}%</div>
        <div class="stat-meta">{n_wins:,} Wins / {n_losses:,} Losses</div>
      </div>
      <div class="card">
        <div class="stat-title">Profit Factor</div>
        <div class="stat-value {'text-emerald' if metrics.get('profit_factor', 0) >= 1.0 else 'text-rose'}">{metrics.get('profit_factor', 0):.4f}</div>
        <div class="stat-meta">Payoff Ratio: {win_loss_payoff:.2f}x</div>
      </div>
      <div class="card">
        <div class="stat-title">Total Net Profit</div>
        <div class="stat-value {'text-emerald' if metrics.get('total_net_profit', 0) >= 0 else 'text-rose'}">${metrics.get('total_net_profit', 0):,.2f}</div>
        <div class="stat-meta">Friction: ${metrics.get('total_commissions', 0):,.2f}</div>
      </div>
      <div class="card">
        <div class="stat-title">Max Drawdown</div>
        <div class="stat-value text-rose">${metrics.get('max_intraday_drawdown', 0):,.2f}</div>
        <div class="stat-meta">EOD DD: ${metrics.get('max_eod_drawdown', 0):,.2f}</div>
      </div>
      <div class="card">
        <div class="stat-title">Contract Cap Skips</div>
        <div class="stat-value text-indigo">{metrics.get('skipped_by_contract_cap', 0):,}</div>
        <div class="stat-meta">Peak: {metrics.get('peak_concurrent_contracts', 0)} / 50 contracts</div>
      </div>
    </div>

    <!-- SIDE-BY-SIDE WINNING VS LOSING TEARSHEET -->
    <div class="card">
      <h2>
        <span style="display:inline-block; width:10px; height:10px; border-radius:50%; background-color:#38bdf8;"></span>
        Winning Trades vs. Losing Trades Comparative Audit
      </h2>
      <p class="subtitle" style="margin-bottom: 0.75rem;">
        Mathematical breakdown comparing winning trades against losing trades across sizing, hold durations, and excursion dynamics.
      </p>
      <div class="table-container">
        <table>
          <thead>
            <tr>
              <th>Performance Metric</th>
              <th class="text-emerald">Winning Trades ({win_stats['pct']:.1f}%)</th>
              <th class="text-rose">Losing Trades ({loss_stats['pct']:.1f}%)</th>
              <th class="text-cyan">Delta / Comparison</th>
            </tr>
          </thead>
          <tbody class="font-mono">
            <tr>
              <td style="font-family: inherit; font-weight: 600;">Total Trade Count</td>
              <td class="text-emerald font-bold">{win_stats['count']:,}</td>
              <td class="text-rose font-bold">{loss_stats['count']:,}</td>
              <td class="text-muted">{loss_stats['count'] / max(1, win_stats['count']):.2f}x more losses than wins</td>
            </tr>
            <tr>
              <td style="font-family: inherit; font-weight: 600;">Cumulative Net PnL</td>
              <td class="text-emerald font-bold">+${win_stats['total_pnl']:,.2f}</td>
              <td class="text-rose font-bold">${loss_stats['total_pnl']:,.2f}</td>
              <td class="font-bold {'text-emerald' if (win_stats['total_pnl'] + loss_stats['total_pnl']) >= 0 else 'text-rose'}">Net: ${win_stats['total_pnl'] + loss_stats['total_pnl']:,.2f}</td>
            </tr>
            <tr>
              <td style="font-family: inherit; font-weight: 600;">Average PnL per Trade</td>
              <td class="text-emerald">+${win_stats['avg_pnl']:,.2f}</td>
              <td class="text-rose">${loss_stats['avg_pnl']:,.2f}</td>
              <td class="text-emerald font-bold">Payoff Ratio: {win_loss_payoff:.2f}:1</td>
            </tr>
            <tr>
              <td style="font-family: inherit; font-weight: 600;">Median PnL per Trade</td>
              <td class="text-emerald">+${win_stats['median_pnl']:,.2f}</td>
              <td class="text-rose">${loss_stats['median_pnl']:,.2f}</td>
              <td class="text-muted">${win_stats['median_pnl'] - abs(loss_stats['median_pnl']):+,.2f}</td>
            </tr>
            <tr>
              <td style="font-family: inherit; font-weight: 600;">Average Gross Points</td>
              <td class="text-emerald">+{win_stats['avg_pts']:.2f} pts</td>
              <td class="text-rose">{loss_stats['avg_pts']:.2f} pts</td>
              <td class="text-muted">{win_stats['avg_pts'] / max(0.01, abs(loss_stats['avg_pts'])):.2f}x points ratio</td>
            </tr>
            <tr>
              <td style="font-family: inherit; font-weight: 600;">Largest Outlier Trade</td>
              <td class="text-emerald">+${win_stats['max_pnl']:,.2f}</td>
              <td class="text-rose">${loss_stats['max_pnl']:,.2f}</td>
              <td class="text-muted">-</td>
            </tr>
            <tr>
              <td style="font-family: inherit; font-weight: 600;">Average Duration (Hold Time)</td>
              <td>{win_stats['avg_hold']:.1f} mins (Median: {win_stats['median_hold']:.0f}m)</td>
              <td>{loss_stats['avg_hold']:.1f} mins (Median: {loss_stats['median_hold']:.0f}m)</td>
              <td class="text-cyan">{'Winners hold longer' if win_stats['avg_hold'] > loss_stats['avg_hold'] else 'Losers hold longer'}</td>
            </tr>
            <tr>
              <td style="font-family: inherit; font-weight: 600;">Average Contract Sizing</td>
              <td>{win_stats['avg_contracts']:.2f} contracts</td>
              <td>{loss_stats['avg_contracts']:.2f} contracts</td>
              <td class="text-muted">{win_stats['avg_contracts'] - loss_stats['avg_contracts']:+.2f} contracts</td>
            </tr>
            <tr>
              <td style="font-family: inherit; font-weight: 600;">Average Initial Stop Distance</td>
              <td>{win_stats['avg_sl_dist']:.2f} pts</td>
              <td>{loss_stats['avg_sl_dist']:.2f} pts</td>
              <td class="text-muted">{win_stats['avg_sl_dist'] - loss_stats['avg_sl_dist']:+.2f} pts</td>
            </tr>
            <tr>
              <td style="font-family: inherit; font-weight: 600;">Average MAE (Max Adverse Excursion)</td>
              <td class="text-amber font-bold">{win_stats['avg_mae_r']:.3f} R ({win_stats['avg_mae_pts']:.2f} pts)</td>
              <td class="text-rose font-bold">{loss_stats['avg_mae_r']:.3f} R ({loss_stats['avg_mae_pts']:.2f} pts)</td>
              <td class="text-muted">Winners suffer {win_stats['avg_mae_r']:.2f}R heat before TP</td>
            </tr>
            <tr>
              <td style="font-family: inherit; font-weight: 600;">Average MFE (Max Favorable Excursion)</td>
              <td class="text-emerald font-bold">{win_stats['avg_mfe_r']:.3f} R ({win_stats['avg_mfe_pts']:.2f} pts)</td>
              <td class="text-amber font-bold">{loss_stats['avg_mfe_r']:.3f} R ({loss_stats['avg_mfe_pts']:.2f} pts)</td>
              <td class="text-muted">Losers ran {loss_stats['avg_mfe_r']:.2f}R green before reversing</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- MAE vs MFE SCATTER PLOT -->
    <div class="card">
      <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:0.5rem; margin-bottom: 0.75rem;">
        <div>
          <h2>Intra-Trade MAE vs. MFE Excursion Scatter (in R-Multiples)</h2>
          <p class="subtitle">
            Reveals trade journey: Adverse heat taken (MAE) vs Favorable peak reached (MFE). Green = Winning Trades, Red = Losing Trades.
          </p>
        </div>
        <div style="display:flex; gap:1rem; font-size:0.75rem;" class="font-mono">
          <span style="display:flex; align-items:center; gap:0.35rem;"><span style="width:8px; height:8px; border-radius:50%; background:#10b981; display:inline-block;"></span> Wins ({n_wins:,})</span>
          <span style="display:flex; align-items:center; gap:0.35rem;"><span style="width:8px; height:8px; border-radius:50%; background:#f43f5e; display:inline-block;"></span> Losses ({n_losses:,})</span>
        </div>
      </div>
      <div id="maeMfeChart" class="chart-box" style="height: 480px;">
        <div class="chart-placeholder">Loading MAE vs MFE Interactive Scatter...</div>
      </div>
    </div>

    <!-- TIME PLOTS SECTION -->
    <div class="grid-2col">
      <!-- Hourly Performance (EST) -->
      <div class="card">
        <h2>Performance by Entry Hour (EST)</h2>
        <p class="subtitle">Intraday session profile and edge distribution across 24 market hours.</p>
        <div id="hourlyChart" class="chart-box" style="height: 350px;">
          <div class="chart-placeholder">Loading Hourly Profile Chart...</div>
        </div>
        <div class="table-container" style="max-height: 200px; overflow-y: auto;">
          <table>
            <thead><tr><th>Hour (EST)</th><th>Trades</th><th>Net PnL</th><th>Win Rate</th></tr></thead>
            <tbody class="font-mono">{hourly_rows}</tbody>
          </table>
        </div>
      </div>

      <!-- Day of Week Performance -->
      <div class="card">
        <h2>Performance by Day of Week</h2>
        <p class="subtitle">Weekly cyclicality and edge stability across trading sessions.</p>
        <div id="dowChart" class="chart-box" style="height: 350px;">
          <div class="chart-placeholder">Loading Day of Week Chart...</div>
        </div>
        <div class="table-container">
          <table>
            <thead><tr><th>Day of Week</th><th>Trades</th><th>Net PnL</th><th>Win Rate</th></tr></thead>
            <tbody class="font-mono">{dow_rows}</tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- CUMULATIVE WINS VS LOSSES TIMELINE & DURATION BREAKDOWN -->
    <div class="grid-2col">
      <!-- Cumulative PnL Timeline -->
      <div class="card">
        <h2>Cumulative Timeline: Winning vs. Losing Trades PnL</h2>
        <p class="subtitle">Visualizing the spread between gross winning power vs gross losing drag over time.</p>
        <div id="timelineChart" class="chart-box" style="height: 350px;">
          <div class="chart-placeholder">Loading Cumulative Timeline Chart...</div>
        </div>
      </div>

      <!-- Trade Duration Buckets -->
      <div class="card">
        <h2>Trade Duration Breakdown (Hold Time)</h2>
        <p class="subtitle">Trade frequency, cumulative PnL, and win rate across trade hold buckets.</p>
        <div id="durationChart" class="chart-box" style="height: 350px;">
          <div class="chart-placeholder">Loading Duration Breakdown Chart...</div>
        </div>
        <div class="table-container">
          <table>
            <thead><tr><th>Hold Time</th><th>Trade Count</th><th>Net PnL</th><th>Win Rate</th></tr></thead>
            <tbody class="font-mono">{dur_rows}</tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- CORE EQUITY & DRAWDOWN CHARTS -->
    <div class="grid-2col">
      <div class="card">
        <h2>Cumulative Mark-to-Market Equity Curve</h2>
        <p class="subtitle">Baseline: $50,000 Initial Capital across full backtest horizon.</p>
        <div id="equityChart" class="chart-box" style="height: 350px;">
          <div class="chart-placeholder">Loading Equity Curve...</div>
        </div>
      </div>
      <div class="card">
        <h2>Underwater Equity Drawdown ($)</h2>
        <p class="subtitle">Peak-to-trough mark-to-market drawdown trajectory.</p>
        <div id="drawdownChart" class="chart-box" style="height: 350px;">
          <div class="chart-placeholder">Loading Drawdown Curve...</div>
        </div>
      </div>
    </div>

    <!-- EXIT REASON BREAKDOWN -->
    <div class="card">
      <h2>Exit Reason Distribution & Execution Mechanics</h2>
      <p class="subtitle">Categorization of trade exits across Stop Loss, Take Profit, and EOD Session Flattens.</p>
      <div class="grid-2col" style="margin-top: 0.75rem;">
        <div id="exitReasonChart" class="chart-box" style="height: 320px;">
          <div class="chart-placeholder">Loading Exit Reason Breakdown Chart...</div>
        </div>
        <div class="table-container" style="display:flex; align-items:center;">
          <table>
            <thead><tr><th>Exit Reason</th><th>Trade Count</th><th>Share (%)</th></tr></thead>
            <tbody class="font-mono">{exit_rows}</tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Trade Ledger Preview -->
    <div class="card">
      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
        <div>
          <h2>Trade Ledger Preview (with MAE / MFE)</h2>
          <p class="subtitle">Showing first 50 trades. Full trade ledger exported to <code class="text-cyan">trades.csv</code></p>
        </div>
        <a href="trades.csv" download class="btn btn-cyan">
          Download Complete trades.csv ({total_trades:,} trades)
        </a>
      </div>
      <div class="table-container table-ledger">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Direction</th>
              <th>Entry Time (EST)</th>
              <th>Hold</th>
              <th>Qty</th>
              <th>Entry Price</th>
              <th>Exit Price</th>
              <th>MAE (R)</th>
              <th>MFE (R)</th>
              <th>Gross Pts</th>
              <th>Net PnL ($)</th>
              <th>Exit Reason</th>
            </tr>
          </thead>
          <tbody class="font-mono">{trade_preview_rows}</tbody>
        </table>
      </div>
    </div>

  </div>

  <!-- Asynchronous Multi-CDN Plotly Loader & Safe Execution Engine -->
  <script>
    (function() {{
      const cdnSources = [
        "https://cdn.plot.ly/plotly-2.35.2.min.js",
        "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js",
        "https://unpkg.com/plotly.js-dist-min@2.35.2/plotly.min.js"
      ];
      let sourceIdx = 0;

      function loadScript(idx) {{
        if (idx >= cdnSources.length) {{
          console.warn("All Plotly CDNs failed to load. Displaying offline static tables.");
          showOfflineFallbacks();
          return;
        }}
        const script = document.createElement("script");
        script.src = cdnSources[idx];
        script.async = true;
        script.onload = () => {{
          if (window.Plotly) renderCharts();
          else loadScript(idx + 1);
        }};
        script.onerror = () => loadScript(idx + 1);
        document.head.appendChild(script);
      }}

      function showOfflineFallbacks() {{
        const chartIds = [
          'maeMfeChart', 'hourlyChart', 'dowChart', 'timelineChart',
          'durationChart', 'equityChart', 'drawdownChart', 'exitReasonChart'
        ];
        chartIds.forEach(id => {{
          const el = document.getElementById(id);
          if (el) {{
            el.innerHTML = '<div class="chart-fallback-notice">' +
              '<p class="font-bold text-amber">📊 Chart Visualization Engine Offline</p>' +
              '<p style="margin-top:0.25rem;">Plotly CDN could not be loaded in this browser environment. ' +
              'All quantitative metrics, distributions, and tearsheets are fully available in the tables above and below.</p>' +
              '</div>';
          }}
        }});
      }}

      function safePlot(domId, data, layout, config) {{
        const el = document.getElementById(domId);
        if (!el || !window.Plotly) return;
        try {{
          Plotly.newPlot(domId, data, layout, config || {{ responsive: true }});
        }} catch (err) {{
          console.error("Plotly render error on " + domId, err);
          el.innerHTML = '<div class="chart-fallback-notice text-rose font-bold">Chart error: ' + err.message + '</div>';
        }}
      }}

      function renderCharts() {{
        const eqDates = {json.dumps(eq_dates)};
        const eqVals = {json.dumps(eq_vals)};

        // 1. MAE vs MFE Scatter Plot
        safePlot('maeMfeChart', [
          {{
            x: {json.dumps(loss_mae_r)},
            y: {json.dumps(loss_mfe_r)},
            mode: 'markers',
            name: 'Losing Trades',
            text: {json.dumps(loss_text)},
            hoverinfo: 'text',
            marker: {{ color: '#f43f5e', size: 6, opacity: 0.65 }}
          }},
          {{
            x: {json.dumps(win_mae_r)},
            y: {json.dumps(win_mfe_r)},
            mode: 'markers',
            name: 'Winning Trades',
            text: {json.dumps(win_text)},
            hoverinfo: 'text',
            marker: {{ color: '#10b981', size: 7, opacity: 0.8 }}
          }}
        ], {{
          template: 'plotly_dark',
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          margin: {{ l: 60, r: 30, t: 30, b: 50 }},
          xaxis: {{ title: 'Maximum Adverse Excursion (MAE in R-Multiples)', showgrid: true, gridcolor: '#1e293b' }},
          yaxis: {{ title: 'Maximum Favorable Excursion (MFE in R-Multiples)', showgrid: true, gridcolor: '#1e293b' }},
          shapes: [
            {{ type: 'line', x0: 1.0, x1: 1.0, y0: 0, y1: 3.5, line: {{ color: '#ef4444', dash: 'dash', width: 1.5 }} }},
            {{ type: 'line', x0: 0, x1: 2.0, y0: 2.0, y1: 2.0, line: {{ color: '#10b981', dash: 'dash', width: 1.5 }} }}
          ],
          annotations: [
            {{ x: 1.05, y: 0.2, xref: 'x', yref: 'y', text: '1.0R Initial SL', showarrow: false, font: {{ color: '#ef4444', size: 10 }} }},
            {{ x: 0.1, y: 2.08, xref: 'x', yref: 'y', text: '2.0R Target TP', showarrow: false, font: {{ color: '#10b981', size: 10 }} }}
          ]
        }});

        // 2. Hourly Performance Chart (EST)
        safePlot('hourlyChart', [
          {{
            x: {json.dumps(hours_labels)},
            y: {json.dumps(hourly_pnl)},
            type: 'bar',
            name: 'Net PnL ($)',
            marker: {{
              color: {json.dumps(hourly_pnl)}.map(v => v >= 0 ? '#10b981' : '#f43f5e')
            }}
          }},
          {{
            x: {json.dumps(hours_labels)},
            y: {json.dumps(hourly_wr)},
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Win Rate (%)',
            yaxis: 'y2',
            line: {{ color: '#38bdf8', width: 2 }}
          }}
        ], {{
          template: 'plotly_dark',
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          margin: {{ l: 60, r: 50, t: 20, b: 40 }},
          xaxis: {{ title: 'Entry Hour (EST)', showgrid: false }},
          yaxis: {{ title: 'Net PnL ($)', showgrid: true, gridcolor: '#1e293b' }},
          yaxis2: {{ title: 'Win Rate (%)', overlaying: 'y', side: 'right', range: [0, 60], showgrid: false }},
          legend: {{ orientation: 'h', y: -0.25 }}
        }});

        // 3. Day of Week Performance Chart
        safePlot('dowChart', [
          {{
            x: {json.dumps(dow_labels)},
            y: {json.dumps(dow_pnl)},
            type: 'bar',
            name: 'Net PnL ($)',
            marker: {{
              color: {json.dumps(dow_pnl)}.map(v => v >= 0 ? '#10b981' : '#f43f5e')
            }}
          }},
          {{
            x: {json.dumps(dow_labels)},
            y: {json.dumps(dow_wr)},
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Win Rate (%)',
            yaxis: 'y2',
            line: {{ color: '#fbbf24', width: 2 }}
          }}
        ], {{
          template: 'plotly_dark',
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          margin: {{ l: 60, r: 50, t: 20, b: 40 }},
          xaxis: {{ showgrid: false }},
          yaxis: {{ title: 'Net PnL ($)', showgrid: true, gridcolor: '#1e293b' }},
          yaxis2: {{ title: 'Win Rate (%)', overlaying: 'y', side: 'right', range: [0, 50], showgrid: false }},
          legend: {{ orientation: 'h', y: -0.25 }}
        }});

        // 4. Cumulative Wins vs Losses Timeline
        safePlot('timelineChart', [
          {{
            x: {json.dumps(tl_dates)},
            y: {json.dumps(tl_win_cum)},
            type: 'scatter',
            mode: 'lines',
            name: 'Cumulative Wins PnL (+)',
            line: {{ color: '#10b981', width: 2 }}
          }},
          {{
            x: {json.dumps(tl_dates)},
            y: {json.dumps(tl_loss_cum)},
            type: 'scatter',
            mode: 'lines',
            name: 'Cumulative Losses PnL (-)',
            line: {{ color: '#f43f5e', width: 2 }}
          }}
        ], {{
          template: 'plotly_dark',
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          margin: {{ l: 60, r: 20, t: 20, b: 40 }},
          xaxis: {{ showgrid: true, gridcolor: '#1e293b' }},
          yaxis: {{ title: 'Cumulative PnL ($)', showgrid: true, gridcolor: '#1e293b', tickprefix: '$' }},
          legend: {{ orientation: 'h', y: -0.25 }}
        }});

        // 5. Trade Duration Analysis
        safePlot('durationChart', [
          {{
            x: {json.dumps(dur_labels)},
            y: {json.dumps(dur_counts)},
            type: 'bar',
            name: 'Trade Count',
            marker: {{ color: '#6366f1' }}
          }},
          {{
            x: {json.dumps(dur_labels)},
            y: {json.dumps(dur_wr)},
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Win Rate (%)',
            yaxis: 'y2',
            line: {{ color: '#38bdf8', width: 2 }}
          }}
        ], {{
          template: 'plotly_dark',
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          margin: {{ l: 60, r: 50, t: 20, b: 40 }},
          xaxis: {{ title: 'Hold Time Duration', showgrid: false }},
          yaxis: {{ title: 'Trades Count', showgrid: true, gridcolor: '#1e293b' }},
          yaxis2: {{ title: 'Win Rate (%)', overlaying: 'y', side: 'right', range: [0, 50], showgrid: false }},
          legend: {{ orientation: 'h', y: -0.25 }}
        }});

        // 6. Underwater Drawdown Plot
        safePlot('drawdownChart', [{{
          x: eqDates,
          y: {json.dumps(dd_vals)},
          type: 'scatter',
          mode: 'lines',
          name: 'Underwater DD ($)',
          line: {{ color: '#f43f5e', width: 1.5 }},
          fill: 'tozeroy',
          fillcolor: 'rgba(244, 63, 94, 0.15)'
        }}], {{
          template: 'plotly_dark',
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          margin: {{ l: 60, r: 20, t: 10, b: 40 }},
          xaxis: {{ showgrid: true, gridcolor: '#1e293b' }},
          yaxis: {{ showgrid: true, gridcolor: '#1e293b', tickprefix: '$' }}
        }});

        // 7. Equity Curve Plot
        safePlot('equityChart', [{{
          x: eqDates,
          y: eqVals,
          type: 'scatter',
          mode: 'lines',
          name: 'Equity (USD)',
          line: {{ color: '#06b6d4', width: 2 }},
          fill: 'tozeroy',
          fillcolor: 'rgba(6, 182, 212, 0.05)'
        }}], {{
          template: 'plotly_dark',
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          margin: {{ l: 60, r: 20, t: 10, b: 40 }},
          xaxis: {{ showgrid: true, gridcolor: '#1e293b' }},
          yaxis: {{ showgrid: true, gridcolor: '#1e293b', tickprefix: '$' }}
        }});

        // 8. Exit Reason Pie Chart
        safePlot('exitReasonChart', [{{
          labels: {json.dumps(exit_labels)},
          values: {json.dumps(exit_values)},
          type: 'pie',
          hole: 0.5,
          marker: {{ colors: ['#10b981', '#ef4444', '#f59e0b', '#6366f1'] }}
        }}], {{
          template: 'plotly_dark',
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          margin: {{ l: 20, r: 20, t: 10, b: 20 }},
          legend: {{ orientation: 'h', y: -0.1 }}
        }});
      }}

      // Start asynchronous loader
      loadScript(0);
    }})();
  </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path
