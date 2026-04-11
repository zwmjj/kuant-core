"""Import smoke test — verifies the public API surface loads cleanly."""
import pytest


def test_core_imports():
    """Top-level qf imports should all resolve."""
    import qf  # noqa: F401
    from qf import (
        Framework,
        BacktestResult,
        DataLoader,
        SignalGenerator,
        BaseStrategy,
        CovarianceEstimator,
        PortfolioOptimizer,
        ExecutionHandler,
        RiskAnalyzer,
        GateCheck,
        AttributionAnalyzer,
        StressTest,
    )


def test_signals_module():
    from qf.signals import build_signal, build_factor_signal, ALL_FACTORS
    assert len(ALL_FACTORS) >= 10


def test_costs_module():
    from qf.costs import ExecutionHandler
    eh = ExecutionHandler(cost_model="sqrt")
    assert eh is not None


def test_risk_module():
    from qf.risk import GateCheck
    gc = GateCheck()
    gc.check("Sharpe > 1", 1.5, 1.0, ">", "Test")
    summary = gc.summary()
    assert summary["n_pass"] == 1
    assert summary["n_total"] == 1
