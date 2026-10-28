"""
Configuration dataclasses for the MNQ EBP (Engulfing / Sweep Bar Pattern) Strategy.

Asset Specifications:
- Micro E-mini Nasdaq-100 Futures (MNQ)
- Tick Size: 0.25 points = $0.50 USD
- Point Value: 1.00 point = 4 ticks = $2.00 USD
- Bar Resolution: 5m (5-minute intraday bars)
- Timezone: America/New_York (EST/EDT)
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StrategyConfig:
    """Strategy parameters and institutional constants for MNQ EBP strategy."""

    # Swept strategy parameters
    ema_period: int = 20
    rr_ratio: float = 2.0
    entry_cutoff_est: str = "15:50"  # e.g., '15:30', '15:50', '15:55'
    allowed_windows_est: Optional[List[tuple[str, str]]] = None  # Specific windows e.g. [('02:00', '02:59'), ('22:00', '22:59')]

    # Fixed / Configurable institutional constants
    flatten_time_est: str = "16:05"  # Daily cutoff flatten before 16:10 EST
    session_start_est: Optional[str] = "18:00"  # CME Globex open; None for all bars
    risk_per_trade_usd: float = 300.0  # Fixed USD risk per trade
    max_prop_contracts: int = 50  # Dynamic portfolio contract cap
    min_sl_distance_pts: float = 0.0  # Minimum structural stop distance (pts); 0.0 disables filter
    large_candle_threshold_pts: float = 50.0  # 50pt rule threshold
    slippage_ticks_per_side: float = 1.5  # 1.5 ticks = 0.375 points
    tick_size: float = 0.25  # MNQ tick size (pts)
    point_value: float = 2.0  # $2.00 USD per MNQ point
    commission_rt_per_contract: float = 1.24  # $1.24 USD round-turn per contract
    initial_capital: float = 50000.0  # Standard $50k prop-firm account capital
    timezone: str = "America/New_York"  # CME / Prop firm reference timezone

    @property
    def slippage_pts(self) -> float:
        """Adverse slippage in price points (1.5 ticks * 0.25 = 0.375 pts)."""
        return self.slippage_ticks_per_side * self.tick_size


@dataclass
class GridSearchConfig:
    """Configuration for orthogonal parameter grid search."""

    ema_periods: List[int] = field(default_factory=lambda: [10, 15, 20])
    rr_ratios: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 2.5, 3.0])
    entry_cutoffs_est: List[str] = field(default_factory=lambda: ["15:30", "15:55"])
    base_config: StrategyConfig = field(default_factory=StrategyConfig)
