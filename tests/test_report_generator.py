"""
Unit tests for report_generator.py.
Verifies HTML dashboard generation, containment integrity,
Plotly container completeness, and offline fallback compatibility.
"""

import os
import unittest
import pandas as pd
from report_generator import generate_html_dashboard


class TestReportGenerator(unittest.TestCase):
    """Test HTML dashboard generation and offline-readiness."""

    def setUp(self):
        self.output_test_html = "test_dashboard.html"

    def tearDown(self):
        if os.path.exists(self.output_test_html):
            os.remove(self.output_test_html)

    def test_generate_html_dashboard_with_sample_trades(self):
        """Verify report generator produces valid HTML with all required containers and pure CSS."""
        dates = pd.date_range("2024-01-08 09:30:00", periods=5, freq="30min", tz="America/New_York")
        trades_df = pd.DataFrame({
            "trade_id": [1, 2, 3, 4, 5],
            "direction": ["LONG", "SHORT", "LONG", "SHORT", "LONG"],
            "entry_time": dates,
            "exit_time": dates + pd.Timedelta(minutes=15),
            "hold_time_minutes": [15.0, 15.0, 15.0, 15.0, 15.0],
            "contracts": [2, 3, 1, 4, 2],
            "entry_price": [15000.0, 15050.0, 15020.0, 15100.0, 15080.0],
            "exit_price": [15020.0, 15070.0, 15060.0, 15080.0, 15050.0],
            "mae_pts": [2.0, 25.0, 1.5, 4.0, 35.0],
            "mfe_pts": [22.0, 5.0, 42.0, 24.0, 3.0],
            "mae_r": [0.2, 1.0, 0.15, 0.35, 1.0],
            "mfe_r": [2.0, 0.25, 2.0, 2.0, 0.1],
            "gross_pts": [20.0, -20.0, 40.0, 20.0, -30.0],
            "net_pnl_usd": [790.0, -1210.0, 1590.0, 1580.0, -1220.0],
            "sl_distance_pts": [10.0, 20.0, 15.0, 12.0, 30.0],
            "exit_reason": ["TAKE_PROFIT", "STOP_LOSS", "TAKE_PROFIT", "TAKE_PROFIT", "STOP_LOSS"],
        })
        equity_series = pd.Series([50000.0, 50790.0, 49580.0, 51170.0, 52750.0, 51530.0], index=pd.date_range("2024-01-08 09:30:00", periods=6, freq="15min", tz="America/New_York"))
        metrics = {
            "total_trades": 5,
            "overall_win_rate_pct": 60.0,
            "profit_factor": 1.62,
            "total_net_profit": 1530.0,
            "max_intraday_drawdown": 1220.0,
            "max_eod_drawdown": 1220.0,
            "total_commissions": 30.0,
            "avg_trades_per_day": 5.0,
            "skipped_by_contract_cap": 0,
            "peak_concurrent_contracts": 4,
        }

        path = generate_html_dashboard(
            metrics=metrics,
            trades_df=trades_df,
            equity_series=equity_series,
            output_path=self.output_test_html,
            title="TEST QUANT DASHBOARD",
        )
        self.assertTrue(os.path.exists(path))

        with open(path, "r", encoding="utf-8") as f:
            html = f.read()

        # 1. No render-blocking external Tailwind script
        self.assertNotIn("cdn.tailwindcss.com", html, "Must not rely on render-blocking external Tailwind CDN script")

        # 2. Pure CSS embedded
        self.assertIn("<style>", html)
        self.assertIn("--bg-page: #0b1120;", html)
        self.assertIn(".card", html)
        self.assertIn(".grid-kpi", html)
        self.assertIn("minmax(0, 1fr)", html)

        # 3. All chart containers exist in DOM and are isolated block elements
        required_chart_ids = [
            "equityChart",
            "drawdownChart",
            "maeMfeChart",
            "hourlyChart",
            "dowChart",
            "timelineChart",
            "durationChart",
            "exitReasonChart",
        ]
        for chart_id in required_chart_ids:
            self.assertIn(f'id="{chart_id}"', html, f"Missing chart DOM element: {chart_id}")

        # 4. Safe innerHTML clearing before plot
        self.assertIn("el.innerHTML = '';", html, "safePlot must clear container before rendering to prevent overlapping")

        # 5. Order: Equity curve appears before comparative audit and MAE/MFE scatter
        idx_eq = html.find('id="equityChart"')
        idx_mae = html.find('id="maeMfeChart"')
        self.assertLess(idx_eq, idx_mae, "Equity Curve must be positioned above MAE/MFE excursion scatter")

    def test_generate_html_dashboard_empty_trades(self):
        """Dashboard must generate cleanly without exceptions even if trades dataframe is empty."""
        empty_trades = pd.DataFrame(columns=[
            "trade_id", "direction", "entry_time", "exit_time", "hold_time_minutes",
            "contracts", "entry_price", "exit_price", "mae_pts", "mfe_pts",
            "mae_r", "mfe_r", "gross_pts", "net_pnl_usd", "sl_distance_pts", "exit_reason"
        ])
        empty_equity = pd.Series([50000.0], index=pd.date_range("2024-01-08 09:30:00", periods=1, tz="America/New_York"))
        metrics = {
            "total_trades": 0,
            "overall_win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "total_net_profit": 0.0,
            "max_intraday_drawdown": 0.0,
            "max_eod_drawdown": 0.0,
            "total_commissions": 0.0,
            "avg_trades_per_day": 0.0,
            "skipped_by_contract_cap": 0,
            "peak_concurrent_contracts": 0,
        }

        path = generate_html_dashboard(
            metrics=metrics,
            trades_df=empty_trades,
            equity_series=empty_equity,
            output_path=self.output_test_html,
        )
        self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
