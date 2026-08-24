"""Transaction cost models (5 variants)."""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class EventType(Enum):
    MARKET = 'MARKET'
    SIGNAL = 'SIGNAL'
    ORDER  = 'ORDER'
    FILL   = 'FILL'

@dataclass
class MarketEvent:
    date: object
    type: EventType = EventType.MARKET

@dataclass
class SignalEvent:
    date: object
    long_targets: Dict = field(default_factory=dict)
    short_targets: Dict = field(default_factory=dict)
    type: EventType = EventType.SIGNAL

@dataclass
class OrderEvent:
    date: object
    orders: List = field(default_factory=list)
    type: EventType = EventType.ORDER

@dataclass
class FillEvent:
    date: object
    fills: List = field(default_factory=list)
    type: EventType = EventType.FILL


class ExecutionHandler:
    """Five transaction cost models: fixed, tiered, sqrt, linear, full."""
    def __init__(self, cost_model='sqrt', commission_bps=1.0, spread_bps=5.0,
                 impact_coeff=0.3, short_borrow_bps=30.0, sec_fee_bps=0.8, **kw):
        self.cost_model = cost_model
        self.commission_bps = commission_bps / 10000
        self.spread_bps = spread_bps / 10000
        self.impact_coeff = impact_coeff
        self.short_borrow_bps = short_borrow_bps / 10000 / 12
        self.sec_fee_bps = sec_fee_bps / 10000  # SEC fee on sells

    def calc_cost(self, turnover, avg_pos, portfolio, bar, new_long, new_short):
        volume = bar.get('volume', pd.Series())
        adv_dollar = bar.get('adv_dollar', pd.Series())
        prices = bar.get('prices', pd.Series())
        capital = portfolio.capital
        changed = set()
        for p in new_long:
            if p not in portfolio.long_positions: changed.add(p)
        for p in new_short:
            if p not in portfolio.short_positions: changed.add(p)
        for p in portfolio.long_positions:
            if p not in new_long: changed.add(p)
        for p in portfolio.short_positions:
            if p not in new_short: changed.add(p)

        if self.cost_model == 'fixed':
            return turnover * avg_pos * (self.commission_bps + self.spread_bps / 2)
        elif self.cost_model == 'tiered':
            tv = capital * avg_pos
            rate = 0.0020 if tv < 10000 else (0.0010 if tv < 100000 else 0.0005)
            return turnover * avg_pos * rate
        elif self.cost_model in ('sqrt', 'linear', 'full'):
            cost = 0
            for p in changed:
                # Use real ADV (dollar volume) if available
                adv_val = 1e9  # default: assume infinite liquidity
                if len(adv_dollar) > 0 and p in adv_dollar.index:
                    adv_val = max(adv_dollar.get(p, 1e9), 1e4)  # floor at $10K
                elif len(volume) > 0 and p in volume.index and p in prices.index:
                    adv_val = volume.get(p, 1e6) * prices.get(p, 100)
                trade_value = capital * avg_pos
                part = trade_value / (adv_val + 1e-6)  # participation rate
                fixed_c = (self.commission_bps + self.spread_bps / 2) * avg_pos
                if self.cost_model == 'linear':
                    impact = self.impact_coeff * min(part, 0.5) * avg_pos
                else:
                    # MI = k * sqrt(Q/ADV) * avg_position_size
                    impact = self.impact_coeff * np.sqrt(min(part, 1.0)) * avg_pos
                # SEC fee on sell-side
                sec = self.sec_fee_bps * avg_pos * 0.5
                cost += fixed_c + impact + sec
            # Short borrow cost (all models with short exposure)
            cost += sum(new_short.values()) * self.short_borrow_bps
            return cost
        # SEC fee on sell-side turnover (exits + short entries)
        sec_cost = turnover * avg_pos * self.sec_fee_bps * 0.5  # ~half of trades are sells
        return turnover * avg_pos * 0.001 + sec_cost

    def compute_holding_return(self, bar, portfolio):
        """Compute the return of existing positions for this period (excluding rebalance costs)."""
        rets_row = bar['returns']
        month_ret = 0.0
        for permno, weight in portfolio.long_positions.items():
            if pd.notna(rets_row.get(permno)):
                month_ret += weight * float(rets_row[permno])
        for permno, weight in portfolio.short_positions.items():
            if pd.notna(rets_row.get(permno)):
                month_ret -= weight * float(rets_row[permno])
        return month_ret

    def on_order(self, order_event, bar, portfolio):
        """Execute rebalance orders and return the rebalance cost (position returns are not computed here)."""
        tradable = bar['tradable']
        new_long = {}; new_short = {}
        turnover = 0

        for permno, weight, side in order_event.orders:
            if permno not in tradable:
                if side == 'LONG' and permno in portfolio.long_positions:
                    new_long[permno] = portfolio.long_positions[permno]
                elif side == 'SHORT' and permno in portfolio.short_positions:
                    new_short[permno] = portfolio.short_positions[permno]
                continue
            if side == 'LONG':
                new_long[permno] = weight
                if permno not in portfolio.long_positions: turnover += 1
            elif side == 'SHORT':
                new_short[permno] = weight
                if permno not in portfolio.short_positions: turnover += 1
            elif side in ('CLOSE_LONG', 'CLOSE_SHORT'):
                turnover += 1

        avg_pos = 1.30 / max(len(new_long) + len(new_short), 1)
        cost = self.calc_cost(turnover, avg_pos, portfolio, bar, new_long, new_short)

        portfolio.long_positions = new_long
        portfolio.short_positions = new_short
        portfolio.trade_count += turnover
        return FillEvent(date=order_event.date), cost
