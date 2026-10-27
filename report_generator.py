"""
Automated Report and Dashboard Generator for NQ EBP Strategy Backtests.
Produces interactive HTML dashboards (Tailwind + Plotly) and summary reports
featuring comprehensive Win vs Loss analytics, MAE/MFE excursions, and time-based plots.
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
    Integrates Tailwind CSS and Plotly.js for interactive visualizations:
    - Intra-Trade MAE vs MFE Scatter Plot (R-multiples)
    - Time Plot 1: Performance by Hour of Day (EST)
    - Time Plot 2: Performance by Day of Week
    - Time Plot 3: Cumulative Winning vs Losing Trades PnL Timeline
    - Hold Time Duration Analysis
    - Detailed Winning vs Losing Trades Comparative Tearsheet
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
        dur_wr = [round(float(v), 1) for v in dur_grp["win_rate"]]

    else:
        hours_labels, hourly_pnl, hourly_wr = [], [], []
        dow_labels, dow_pnl, dow_wr = [], [], []
        tl_dates, tl_win_cum, tl_loss_cum = [], [], []
        win_mae_r, win_mfe_r, win_text = [], [], []
        loss_mae_r, loss_mfe_r, loss_text = [], [], []
        dur_labels, dur_counts, dur_wr = [], [], []

    # Exit reason breakdown
    exit_counts = trades_df["exit_reason"].value_counts().to_dict() if not trades_df.empty else {}
    exit_labels = list(exit_counts.keys())
    exit_values = list(exit_counts.values())

    # Sample trades for preview
    sample_trades = trades_df.head(50).to_dict(orient="records") if not trades_df.empty else []

    # Generate Full HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ background-color: #0b1120; color: #f8fafc; font-family: system-ui, -apple-system, sans-serif; }}
    .card {{ background-color: #0f172a; border: 1px solid #1e293b; border-radius: 1rem; }}
    .card-header {{ border-bottom: 1px solid #1e293b; padding: 1rem 1.5rem; }}
  </style>
</head>
<body class="antialiased p-4 md:p-8">
  <div class="max-w-7xl mx-auto space-y-6">

    <!-- Header Banner -->
    <div class="card p-6 shadow-2xl relative overflow-hidden bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 border-indigo-900/40">
      <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <div class="flex items-center gap-2 mb-2">
            <span class="px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
              Winning vs. Losing Deep Analytics
            </span>
            <span class="px-2.5 py-1 rounded-full text-xs font-semibold bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
              MAE & MFE Excursions
            </span>
            <span class="px-2.5 py-1 rounded-full text-xs font-semibold bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
              Time & Hourly Profiling
            </span>
          </div>
          <h1 class="text-2xl md:text-3xl font-extrabold tracking-tight text-white">
            {title}
          </h1>
          <p class="text-sm text-slate-400 mt-1">
            Zero Lookahead Bias • Multi-Slot Numba Execution Kernel • Dynamic 50-Contract Cap • Realistic Friction
          </p>
        </div>
        <div class="text-right">
          <div class="text-xs uppercase tracking-wider text-slate-400">Total Backtest Horizon</div>
          <div class="text-lg font-bold text-cyan-400 mt-0.5">
            {chart_equity.index[0].strftime('%Y-%m-%d') if len(chart_equity) > 0 else 'N/A'} → {chart_equity.index[-1].strftime('%Y-%m-%d') if len(chart_equity) > 0 else 'N/A'}
          </div>
        </div>
      </div>
    </div>

    <!-- Stat Grid -->
    <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
      <div class="card p-4">
        <div class="text-xs text-slate-400 uppercase tracking-wider">Total Trades</div>
        <div class="text-2xl font-bold text-white mt-1">{total_trades:,}</div>
        <div class="text-xs text-slate-500 mt-1">{metrics.get('avg_trades_per_day', 0):.1f} trades / active day</div>
      </div>
      <div class="card p-4">
        <div class="text-xs text-slate-400 uppercase tracking-wider">Win Rate</div>
        <div class="text-2xl font-bold text-amber-400 mt-1">{metrics.get('overall_win_rate_pct', 0):.2f}%</div>
        <div class="text-xs text-slate-500 mt-1">{n_wins:,} Wins / {n_losses:,} Losses</div>
      </div>
      <div class="card p-4">
        <div class="text-xs text-slate-400 uppercase tracking-wider">Profit Factor</div>
        <div class="text-2xl font-bold {'text-emerald-400' if metrics.get('profit_factor', 0) >= 1.0 else 'text-rose-400'} mt-1">{metrics.get('profit_factor', 0):.4f}</div>
        <div class="text-xs text-slate-500 mt-1">Payoff Ratio: {win_loss_payoff:.2f}x</div>
      </div>
      <div class="card p-4">
        <div class="text-xs text-slate-400 uppercase tracking-wider">Total Net Profit</div>
        <div class="text-2xl font-bold {'text-emerald-400' if metrics.get('total_net_profit', 0) >= 0 else 'text-rose-400'} mt-1">${metrics.get('total_net_profit', 0):,.2f}</div>
        <div class="text-xs text-slate-500 mt-1">Comm: ${metrics.get('total_commissions', 0):,.2f}</div>
      </div>
      <div class="card p-4">
        <div class="text-xs text-slate-400 uppercase tracking-wider">Max Drawdown</div>
        <div class="text-2xl font-bold text-rose-400 mt-1">${metrics.get('max_intraday_drawdown', 0):,.2f}</div>
        <div class="text-xs text-slate-500 mt-1">EOD DD: ${metrics.get('max_eod_drawdown', 0):,.2f}</div>
      </div>
      <div class="card p-4">
        <div class="text-xs text-slate-400 uppercase tracking-wider">Contract Cap Skips</div>
        <div class="text-2xl font-bold text-indigo-400 mt-1">{metrics.get('skipped_by_contract_cap', 0):,}</div>
        <div class="text-xs text-slate-500 mt-1">Peak: {metrics.get('peak_concurrent_contracts', 0)} / 50 contracts</div>
      </div>
    </div>

    <!-- SIDE-BY-SIDE WINNING VS LOSING TEARSHEET -->
    <div class="card p-6">
      <h2 class="text-lg font-bold text-white mb-4 flex items-center gap-2">
        <span class="w-3 h-3 rounded-full bg-cyan-400"></span>
        Winning Trades vs. Losing Trades Comparative Audit
      </h2>
      <div class="overflow-x-auto">
        <table class="w-full text-sm text-left text-slate-300">
          <thead class="text-xs text-slate-400 uppercase bg-slate-950/80 border-b border-slate-800">
            <tr>
              <th class="py-3 px-4">Performance Metric</th>
              <th class="py-3 px-4 text-emerald-400 font-bold">Winning Trades ({win_stats['pct']:.1f}%)</th>
              <th class="py-3 px-4 text-rose-400 font-bold">Losing Trades ({loss_stats['pct']:.1f}%)</th>
              <th class="py-3 px-4 text-cyan-400 font-bold">Delta / Comparison</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-800 font-mono text-xs md:text-sm">
            <tr class="hover:bg-slate-800/30">
              <td class="py-2.5 px-4 font-sans font-medium text-slate-300">Total Trade Count</td>
              <td class="py-2.5 px-4 text-emerald-400 font-bold">{win_stats['count']:,}</td>
              <td class="py-2.5 px-4 text-rose-400 font-bold">{loss_stats['count']:,}</td>
              <td class="py-2.5 px-4 text-slate-400">{loss_stats['count'] / max(1, win_stats['count']):.2f}x more losses than wins</td>
            </tr>
            <tr class="hover:bg-slate-800/30">
              <td class="py-2.5 px-4 font-sans font-medium text-slate-300">Cumulative Net PnL</td>
              <td class="py-2.5 px-4 text-emerald-400 font-bold">+${win_stats['total_pnl']:,.2f}</td>
              <td class="py-2.5 px-4 text-rose-400 font-bold">${loss_stats['total_pnl']:,.2f}</td>
              <td class="py-2.5 px-4 text-rose-400 font-bold">Net: ${win_stats['total_pnl'] + loss_stats['total_pnl']:,.2f}</td>
            </tr>
            <tr class="hover:bg-slate-800/30">
              <td class="py-2.5 px-4 font-sans font-medium text-slate-300">Average PnL per Trade</td>
              <td class="py-2.5 px-4 text-emerald-400">+${win_stats['avg_pnl']:,.2f}</td>
              <td class="py-2.5 px-4 text-rose-400">${loss_stats['avg_pnl']:,.2f}</td>
              <td class="py-2.5 px-4 text-emerald-400 font-bold">Payoff Ratio: {win_loss_payoff:.2f}:1</td>
            </tr>
            <tr class="hover:bg-slate-800/30">
              <td class="py-2.5 px-4 font-sans font-medium text-slate-300">Median PnL per Trade</td>
              <td class="py-2.5 px-4 text-emerald-400">+${win_stats['median_pnl']:,.2f}</td>
              <td class="py-2.5 px-4 text-rose-400">${loss_stats['median_pnl']:,.2f}</td>
              <td class="py-2.5 px-4 text-slate-400">${win_stats['median_pnl'] - abs(loss_stats['median_pnl']):+,.2f}</td>
            </tr>
            <tr class="hover:bg-slate-800/30">
              <td class="py-2.5 px-4 font-sans font-medium text-slate-300">Average Gross Points</td>
              <td class="py-2.5 px-4 text-emerald-400">+{win_stats['avg_pts']:.2f} pts</td>
              <td class="py-2.5 px-4 text-rose-400">{loss_stats['avg_pts']:.2f} pts</td>
              <td class="py-2.5 px-4 text-slate-400">{win_stats['avg_pts'] / max(0.01, abs(loss_stats['avg_pts'])):.2f}x points ratio</td>
            </tr>
            <tr class="hover:bg-slate-800/30">
              <td class="py-2.5 px-4 font-sans font-medium text-slate-300">Largest Outlier Trade</td>
              <td class="py-2.5 px-4 text-emerald-400">+${win_stats['max_pnl']:,.2f}</td>
              <td class="py-2.5 px-4 text-rose-400">${loss_stats['max_pnl']:,.2f}</td>
              <td class="py-2.5 px-4 text-slate-400">-</td>
            </tr>
            <tr class="hover:bg-slate-800/30">
              <td class="py-2.5 px-4 font-sans font-medium text-slate-300">Average Duration (Hold Time)</td>
              <td class="py-2.5 px-4">{win_stats['avg_hold']:.1f} mins (Median: {win_stats['median_hold']:.0f}m)</td>
              <td class="py-2.5 px-4">{loss_stats['avg_hold']:.1f} mins (Median: {loss_stats['median_hold']:.0f}m)</td>
              <td class="py-2.5 px-4 text-cyan-400">{'Winners hold longer' if win_stats['avg_hold'] > loss_stats['avg_hold'] else 'Losers hold longer'}</td>
            </tr>
            <tr class="hover:bg-slate-800/30">
              <td class="py-2.5 px-4 font-sans font-medium text-slate-300">Average Contract Sizing</td>
              <td class="py-2.5 px-4">{win_stats['avg_contracts']:.2f} contracts</td>
              <td class="py-2.5 px-4">{loss_stats['avg_contracts']:.2f} contracts</td>
              <td class="py-2.5 px-4 text-slate-400">{win_stats['avg_contracts'] - loss_stats['avg_contracts']:+.2f} contracts</td>
            </tr>
            <tr class="hover:bg-slate-800/30">
              <td class="py-2.5 px-4 font-sans font-medium text-slate-300">Average Initial Stop Distance</td>
              <td class="py-2.5 px-4">{win_stats['avg_sl_dist']:.2f} pts</td>
              <td class="py-2.5 px-4">{loss_stats['avg_sl_dist']:.2f} pts</td>
              <td class="py-2.5 px-4 text-slate-400">{win_stats['avg_sl_dist'] - loss_stats['avg_sl_dist']:+.2f} pts</td>
            </tr>
            <tr class="hover:bg-slate-800/30">
              <td class="py-2.5 px-4 font-sans font-medium text-slate-300">Average MAE (Max Adverse Excursion)</td>
              <td class="py-2.5 px-4 text-amber-400 font-bold">{win_stats['avg_mae_r']:.3f} R ({win_stats['avg_mae_pts']:.2f} pts)</td>
              <td class="py-2.5 px-4 text-rose-400 font-bold">{loss_stats['avg_mae_r']:.3f} R ({loss_stats['avg_mae_pts']:.2f} pts)</td>
              <td class="py-2.5 px-4 text-slate-400">Winners suffer {win_stats['avg_mae_r']:.2f}R heat before TP</td>
            </tr>
            <tr class="hover:bg-slate-800/30">
              <td class="py-2.5 px-4 font-sans font-medium text-slate-300">Average MFE (Max Favorable Excursion)</td>
              <td class="py-2.5 px-4 text-emerald-400 font-bold">{win_stats['avg_mfe_r']:.3f} R ({win_stats['avg_mfe_pts']:.2f} pts)</td>
              <td class="py-2.5 px-4 text-amber-400 font-bold">{loss_stats['avg_mfe_r']:.3f} R ({loss_stats['avg_mfe_pts']:.2f} pts)</td>
              <td class="py-2.5 px-4 text-slate-400">Losers ran {loss_stats['avg_mfe_r']:.2f}R green before reversing</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- MAE vs MFE SCATTER PLOT -->
    <div class="card p-6">
      <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-2 mb-4">
        <div>
          <h2 class="text-base font-semibold text-white">Intra-Trade MAE vs. MFE Excursion Scatter (in R-Multiples)</h2>
          <p class="text-xs text-slate-400">
            Reveals trade journey: Adverse heat taken (MAE) vs Favorable peak reached (MFE). Green = Winning Trades, Red = Losing Trades.
          </p>
        </div>
        <div class="flex items-center gap-4 text-xs font-mono">
          <span class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full bg-emerald-500"></span> Wins ({n_wins:,})</span>
          <span class="flex items-center gap-1.5"><span class="w-2.5 h-2.5 rounded-full bg-rose-500"></span> Losses ({n_losses:,})</span>
        </div>
      </div>
      <div id="maeMfeChart" style="height: 480px;"></div>
    </div>

    <!-- TIME PLOTS SECTION -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Hourly Performance (EST) -->
      <div class="card p-5">
        <div class="flex justify-between items-center mb-3">
          <h2 class="text-base font-semibold text-white">Performance by Entry Hour (EST)</h2>
          <span class="text-xs text-slate-400">Session Profile</span>
        </div>
        <div id="hourlyChart" style="height: 350px;"></div>
      </div>

      <!-- Day of Week Performance -->
      <div class="card p-5">
        <div class="flex justify-between items-center mb-3">
          <h2 class="text-base font-semibold text-white">Performance by Day of Week</h2>
          <span class="text-xs text-slate-400">Mon - Fri Regularity</span>
        </div>
        <div id="dowChart" style="height: 350px;"></div>
      </div>
    </div>

    <!-- CUMULATIVE WINS VS LOSSES TIMELINE & DURATION BREAKDOWN -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Cumulative PnL Timeline -->
      <div class="card p-5">
        <div class="flex justify-between items-center mb-3">
          <h2 class="text-base font-semibold text-white">Cumulative Timeline: Winning vs. Losing Trades PnL</h2>
          <span class="text-xs text-slate-400">Compounding Spread</span>
        </div>
        <div id="timelineChart" style="height: 350px;"></div>
      </div>

      <!-- Trade Duration Buckets -->
      <div class="card p-5">
        <div class="flex justify-between items-center mb-3">
          <h2 class="text-base font-semibold text-white">Trade Duration Breakdown (Hold Time)</h2>
          <span class="text-xs text-slate-400">Win Rate vs PnL</span>
        </div>
        <div id="durationChart" style="height: 350px;"></div>
      </div>
    </div>

    <!-- CORE EQUITY & DRAWDOWN CHARTS -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div class="card p-5">
        <div class="flex justify-between items-center mb-4">
          <h2 class="text-base font-semibold text-white">Cumulative Mark-to-Market Equity Curve</h2>
          <span class="text-xs text-slate-400">Baseline: $50,000 Initial Capital</span>
        </div>
        <div id="equityChart" style="height: 350px;"></div>
      </div>
      <div class="card p-5">
        <h2 class="text-base font-semibold text-white mb-4">Underwater Equity Drawdown ($)</h2>
        <div id="drawdownChart" style="height: 350px;"></div>
      </div>
    </div>

    <!-- Trade Ledger Preview -->
    <div class="card p-5">
      <div class="flex justify-between items-center mb-4">
        <div>
          <h2 class="text-base font-semibold text-white">Recent Trade Ledger Preview (with MAE / MFE)</h2>
          <p class="text-xs text-slate-400">Showing first 50 trades. Full trade ledger exported to <code class="text-cyan-400">trades.csv</code></p>
        </div>
        <a href="trades.csv" download class="px-3 py-1.5 text-xs font-semibold bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg transition">
          Download Full trades.csv
        </a>
      </div>
      <div class="overflow-x-auto max-h-96">
        <table class="w-full text-xs text-left text-slate-300">
          <thead class="text-slate-400 uppercase bg-slate-950 sticky top-0 border-b border-slate-800">
            <tr>
              <th class="p-2">#</th>
              <th class="p-2">Direction</th>
              <th class="p-2">Entry Time (EST)</th>
              <th class="p-2">Hold (m)</th>
              <th class="p-2">Qty</th>
              <th class="p-2">Entry</th>
              <th class="p-2">Exit</th>
              <th class="p-2">MAE (R)</th>
              <th class="p-2">MFE (R)</th>
              <th class="p-2">Gross Pts</th>
              <th class="p-2">Net PnL ($)</th>
              <th class="p-2">Exit Reason</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-800/60 font-mono">
            {''.join([f'''<tr class="hover:bg-slate-800/40">
              <td class="p-2 text-slate-500">{t.get("trade_id", "")}</td>
              <td class="p-2 font-bold {'text-emerald-400' if t.get('direction') == 'LONG' else 'text-rose-400'}">{t.get("direction", "")}</td>
              <td class="p-2 text-slate-300">{str(t.get("entry_time", ""))[:19]}</td>
              <td class="p-2">{t.get("hold_time_minutes", "")}</td>
              <td class="p-2">{t.get("contracts", "")}</td>
              <td class="p-2">{t.get("entry_price", 0):.4f}</td>
              <td class="p-2">{t.get("exit_price", 0):.4f}</td>
              <td class="p-2 text-amber-400">{t.get("mae_r", 0):.2f}R</td>
              <td class="p-2 text-cyan-400">{t.get("mfe_r", 0):.2f}R</td>
              <td class="p-2 {'text-emerald-400' if t.get('gross_pts', 0) > 0 else 'text-rose-400'}">{t.get("gross_pts", 0):+.2f}</td>
              <td class="p-2 font-bold {'text-emerald-400' if t.get('net_pnl_usd', 0) > 0 else 'text-rose-400'}">${t.get("net_pnl_usd", 0):+,.2f}</td>
              <td class="p-2 text-slate-400">{t.get("exit_reason", "")}</td>
            </tr>''' for t in sample_trades])}
          </tbody>
        </table>
      </div>
    </div>

  </div>

  <script>
    const eqDates = {json.dumps(eq_dates)};
    const eqVals = {json.dumps(eq_vals)};

    // 1. MAE vs MFE Scatter Plot
    Plotly.newPlot('maeMfeChart', [
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
    }}, {{ responsive: true }});

    // 2. Hourly Performance Chart (EST)
    Plotly.newPlot('hourlyChart', [
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
      legend: {{ orientation: 'h', y: -0.2 }}
    }}, {{ responsive: true }});

    // 3. Day of Week Performance Chart
    Plotly.newPlot('dowChart', [
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
      legend: {{ orientation: 'h', y: -0.2 }}
    }}, {{ responsive: true }});

    // 4. Cumulative Wins vs Losses Timeline
    Plotly.newPlot('timelineChart', [
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
      legend: {{ orientation: 'h', y: -0.2 }}
    }}, {{ responsive: true }});

    // 5. Trade Duration Analysis
    Plotly.newPlot('durationChart', [
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
      legend: {{ orientation: 'h', y: -0.2 }}
    }}, {{ responsive: true }});

    // 6. Underwater Drawdown Plot
    Plotly.newPlot('drawdownChart', [{{
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
    }}, {{ responsive: true }});

    // 7. Equity Curve Plot
    Plotly.newPlot('equityChart', [{{
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
    }}, {{ responsive: true }});

    // 7. Exit Reason Pie
    Plotly.newPlot('exitReasonChart', [{{
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
    }}, {{ responsive: true }});
  </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path
