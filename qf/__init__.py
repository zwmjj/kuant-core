"""
qf — 量化研究框架
==================
可插拔策略 + 事件驱动回测 + 风控 + 可视化

使用方法:
  from qf import Framework
  from strategies.momentum import MomentumStrategy

  fw = Framework()
  fw.run(MomentumStrategy())
"""
from qf.data import DataLoader, prepare_data
from qf.signals import (SignalGenerator, build_signal, build_factor_signal,
                         IMPLEMENTED_FACTORS, FUNDAMENTAL_FACTORS, ALL_FACTORS)
from qf.strategy import BaseStrategy
from qf.portfolio import CovarianceEstimator, PortfolioOptimizer
from qf.costs import ExecutionHandler
from qf.backtest import (Backtester, BacktestResult, EventDrivenBacktester,
                          DataHandler, Portfolio, run_event_driven)
from qf.attribution import AttributionAnalyzer
from qf.risk import RiskAnalyzer, GateCheck
from qf.stress import StressTest
from qf.framework import Framework
from qf.stat_toolkit import (WalkForwardCV, cross_sectional_zscore,
                             regime_conditional_ic, strategy_correlation_matrix,
                             marginal_risk_contribution, estimate_capacity,
                             sigmoid_regime_weight, orthogonalize_factors,
                             fundamental_law_decomposition, run_full_gsa_audit)
