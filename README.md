# ⚡ AG03 Quantitative Backtesting Engine

High-performance quantitative backtesting and research platform for intraday futures trading (e.g. Micro E-mini Nasdaq-100 Futures, `MNQ` / `NQ`).

Built with a **VectorBT + Numba (`@njit`) + pandas + numpy** hybrid architecture supporting:
- **Zero lookahead bias** (Signal evaluated at Close of bar $t$, executed at Open of bar $t+1$).
- **Realistic adverse execution modeling** (custom slippage on entry/stop/flatten, commissions, limit fills).
- **Multi-slot concurrent positions** bounded by aggregate portfolio contract caps.
- **Mark-to-Market (MTM) equity tracking** for institutional / prop-firm trailing drawdown rules.
- **Publication-grade HTML reporting** with standalone interactive Plotly visualizations.

---

## 🏗️ Architecture

```
AG03/
├── config.py                 # StrategyConfig dataclass for asset specs & trade parameters
├── data_loader.py            # CSV/Parquet loader, timezone normalizer, session masks & synthetic generator
├── execution_kernel.py       # High-speed Numba @njit multi-slot execution engine
├── metrics_vbt.py            # Institutional & prop-firm performance metrics suite
├── report_generator.py       # Interactive standalone HTML dashboard generator
├── data/
│   └── nas100_m5_2020_2026.csv # 7-year historical 5-minute dataset
└── tests/
    ├── test_engine.py        # Core backtest engine & execution tests
    └── test_report_generator.py # HTML report dashboard tests
```

---

## 🧪 Running Tests

```bash
python -m unittest discover tests
```
