"""
Automated Report and Dashboard Generator for NQ EBP Strategy Backtests.
Produces interactive HTML dashboards (Tailwind + Plotly) and summary reports.
"""

from typing import Any, Dict
import json
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
    Integrates Tailwind CSS and Plotly.js for interactive visualizations.
    """
    # Downsample equity series for ultra-smooth web charting if trade count is massive
    if len(equity_series) > 10000:
        step = max(1, len(equity_series) // 5000)
        chart_equity = equity_series.iloc[::step].copy()
    else:
        chart_equity = equity_series.copy()

    # Calculate Drawdown curve from peak equity
    peak_equity = chart_equity.cummax()
    drawdown_series = chart_equity - peak_equity

    # Format equity dates & values for JSON
    eq_dates = [ts.strftime("%Y-%m-%d %H:%M") for ts in chart_equity.index]
    eq_vals = [round(float(v), 2) for v in chart_equity.values]
    dd_vals = [round(float(v), 2) for v in drawdown_series.values]

    # Exit reason breakdown
    exit_counts = trades_df["exit_reason"].value_counts().to_dict() if not trades_df.empty else {}
    exit_labels = list(exit_counts.keys())
    exit_values = list(exit_counts.values())

    # Sample top/recent trades for preview table
    sample_trades = trades_df.head(50).to_dict(orient="records") if not trades_df.empty else []

    # HTML Template
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
              Institutional Backtest Complete
            </span>
            <span class="px-2.5 py-1 rounded-full text-xs font-semibold bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
              CME MNQ Micro Futures
            </span>
            <span class="px-2.5 py-1 rounded-full text-xs font-semibold bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
              5m Resolution
            </span>
          </div>
          <h1 class="text-2xl md:text-3xl font-extrabold tracking-tight text-white">
            {title}
          </h1>
          <p class="text-sm text-slate-400 mt-1">
            Zero Lookahead Bias • Multi-Slot Numba Execution Kernel • Dynamic 50-Contract Cap • Realistic Slippage & Commissions
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
        <div class="text-2xl font-bold text-white mt-1">{metrics.get('total_trades', 0):,}</div>
        <div class="text-xs text-slate-500 mt-1">{metrics.get('avg_trades_per_day', 0):.1f} trades / active day</div>
      </div>
      <div class="card p-4">
        <div class="text-xs text-slate-400 uppercase tracking-wider">Win Rate</div>
        <div class="text-2xl font-bold text-amber-400 mt-1">{metrics.get('overall_win_rate_pct', 0):.2f}%</div>
        <div class="text-xs text-slate-500 mt-1">Long: {metrics.get('long_win_rate_pct', 0):.1f}% | Short: {metrics.get('short_win_rate_pct', 0):.1f}%</div>
      </div>
      <div class="card p-4">
        <div class="text-xs text-slate-400 uppercase tracking-wider">Profit Factor</div>
        <div class="text-2xl font-bold {'text-emerald-400' if metrics.get('profit_factor', 0) >= 1.0 else 'text-rose-400'} mt-1">{metrics.get('profit_factor', 0):.4f}</div>
        <div class="text-xs text-slate-500 mt-1">Sharpe: {metrics.get('sharpe_ratio', 0):.2f}</div>
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

    <!-- Charts Section -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Cumulative Equity Curve (2 cols) -->
      <div class="card p-5 lg:col-span-2">
        <div class="flex justify-between items-center mb-4">
          <h2 class="text-base font-semibold text-white">Cumulative Mark-to-Market Equity Curve</h2>
          <span class="text-xs text-slate-400">Baseline: $50,000 Initial Capital</span>
        </div>
        <div id="equityChart" style="height: 380px;"></div>
      </div>

      <!-- Exit Reasons Pie Chart (1 col) -->
      <div class="card p-5">
        <h2 class="text-base font-semibold text-white mb-4">Exit Reason Distribution</h2>
        <div id="exitReasonChart" style="height: 380px;"></div>
      </div>
    </div>

    <!-- Drawdown & Performance Breakdown -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div class="card p-5">
        <h2 class="text-base font-semibold text-white mb-4">Underwater Equity Drawdown ($)</h2>
        <div id="drawdownChart" style="height: 300px;"></div>
      </div>

      <!-- Strategy Parameters & Rules Table -->
      <div class="card p-5 space-y-4">
        <h2 class="text-base font-semibold text-white">Execution & Risk Control Architecture</h2>
        <div class="overflow-x-auto">
          <table class="w-full text-sm text-left text-slate-300">
            <tbody class="divide-y divide-slate-800">
              <tr>
                <td class="py-2.5 font-medium text-slate-400">Target Asset</td>
                <td class="py-2.5 text-right font-semibold text-white">Micro E-mini Nasdaq-100 Futures (MNQ)</td>
              </tr>
              <tr>
                <td class="py-2.5 font-medium text-slate-400">Position Sizing Risk Target</td>
                <td class="py-2.5 text-right font-semibold text-white">$300.00 / Trade (Half-Up Rounding)</td>
              </tr>
              <tr>
                <td class="py-2.5 font-medium text-slate-400">Structural Stop Distance</td>
                <td class="py-2.5 text-right font-semibold text-white">50-Pt Rule: {metrics.get('pct_midpoint_rule_trades', 0):.1f}% trades used 50% midpoint</td>
              </tr>
              <tr>
                <td class="py-2.5 font-medium text-slate-400">Average Stop-Loss Distance</td>
                <td class="py-2.5 text-right font-semibold text-white">{metrics.get('avg_sl_distance_pts', 0):.2f} points (${metrics.get('avg_sl_distance_pts', 0)*2.0:.2f})</td>
              </tr>
              <tr>
                <td class="py-2.5 font-medium text-slate-400">Average Trade Duration</td>
                <td class="py-2.5 text-right font-semibold text-white">{metrics.get('avg_hold_time_minutes', 0):.1f} mins (Median: {metrics.get('median_hold_time_minutes', 0):.1f} mins)</td>
              </tr>
              <tr>
                <td class="py-2.5 font-medium text-slate-400">Portfolio Dynamic Cap</td>
                <td class="py-2.5 text-right font-semibold text-white">50 MNQ Contracts (Peak slots: {metrics.get('peak_concurrent_slots', 0)})</td>
              </tr>
              <tr>
                <td class="py-2.5 font-medium text-slate-400">Friction Parameters</td>
                <td class="py-2.5 text-right font-semibold text-white">1.5 ticks adverse slippage, $1.24/contract RT commission</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Trade Ledger Preview -->
    <div class="card p-5">
      <div class="flex justify-between items-center mb-4">
        <div>
          <h2 class="text-base font-semibold text-white">Recent Trade Ledger Preview</h2>
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
              <th class="p-2">Exit Time (EST)</th>
              <th class="p-2">Hold (m)</th>
              <th class="p-2">Qty</th>
              <th class="p-2">Entry</th>
              <th class="p-2">Exit</th>
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
              <td class="p-2 text-slate-300">{str(t.get("exit_time", ""))[:19]}</td>
              <td class="p-2">{t.get("hold_time_minutes", "")}</td>
              <td class="p-2">{t.get("contracts", "")}</td>
              <td class="p-2">{t.get("entry_price", 0):.4f}</td>
              <td class="p-2">{t.get("exit_price", 0):.4f}</td>
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
    const ddVals = {json.dumps(dd_vals)};

    // Equity Curve Plot
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
      margin: {{ l: 50, r: 20, t: 10, b: 40 }},
      xaxis: {{ showgrid: true, gridcolor: '#1e293b' }},
      yaxis: {{ showgrid: true, gridcolor: '#1e293b', tickprefix: '$' }}
    }}, {{ responsive: true }});

    // Drawdown Plot
    Plotly.newPlot('drawdownChart', [{{
      x: eqDates,
      y: ddVals,
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
      margin: {{ l: 50, r: 20, t: 10, b: 40 }},
      xaxis: {{ showgrid: true, gridcolor: '#1e293b' }},
      yaxis: {{ showgrid: true, gridcolor: '#1e293b', tickprefix: '$' }}
    }}, {{ responsive: true }});

    // Exit Reason Pie
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
