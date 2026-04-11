# kuant-core

Production-grade quantitative research framework. Event-driven backtester,
factor library, risk toolkit, walk-forward CV, multi-asset portfolio
construction. Extracted from the Kuant research platform as a
standalone pip-installable library.

> **Companion projects:**
> [`kuant-research`](https://github.com/zwmjj/kuant-research) — 14
> reproducible empirical studies built on this library.
> [`alt-data-research`](https://github.com/zwmjj/alt-data-research) —
> SEC NLP alt-data alpha factor research (t-stat 2.11, ICIR 0.80).

## Scope

- **Data loaders**: WRDS CRSP monthly, Compustat fundamentals, Alpaca
  historical, Kenneth French FF5, akshare A-share, optional Alpaca
  live feeds (no order flow — `live_executor.py` deliberately left
  out of this public package).
- **Factor library**: 28+ factors across momentum, value, quality,
  volatility, interaction, orthogonalization, A-share. See
  `qf/signals.py` for the full list.
- **Event-driven backtester**: `qf/backtest.py` — monthly rebalance,
  turnover-penalty signal blending, full cost model with 5 variants
  (fixed / tiered / sqrt-impact / linear / full).
- **Portfolio construction**: `qf/portfolio.py` — covariance
  estimators (sample, shrinkage, Ledoit-Wolf), min-variance and
  mean-variance optimizers with box / sector / turnover constraints.
- **Risk toolkit**: `qf/risk.py` — Sharpe / Sortino / Calmar, VaR /
  CVaR, max drawdown, tail risk, gate-check system for SOP-compliant
  deployment.
- **Walk-forward CV**: `qf/stat_toolkit.py` — standard walk-forward
  cross-validation with IS/OOS split, rolling Sharpe, fundamental-law
  decomposition, and Deflated Sharpe Ratio.
- **Attribution**: `qf/attribution.py` — Fama-French 5-factor
  regression with Newey-West standard errors.
- **Stress testing**: `qf/stress.py` — replay against 2008 GFC,
  2020 COVID, 2022 tech sell-off, plus cost sensitivity.
- **Visualization**: `qf/viz.py` — equity curves, drawdowns, IC
  time series, quintile sort plots, tearsheet generation.
- **Multi-market**: same engine runs US (CRSP + Compustat) and China
  A-share (baostock + akshare intraday) universes.

## Install

```bash
# core only (numpy + pandas + scipy + matplotlib + sklearn + statsmodels)
pip install kuant-core

# with US data plumbing (WRDS + pandas-datareader + yfinance)
pip install "kuant-core[us-data]"

# with China A-share data
pip install "kuant-core[cn-data]"

# with ML factor blending (LightGBM + XGBoost)
pip install "kuant-core[ml]"

# the works
pip install "kuant-core[full]"
```

## Quickstart

```python
from qf import Framework
from qf import build_factor_signal, run_event_driven

# Build a monthly-rebalanced US equity backtest in 4 lines
framework = Framework()
framework.run(
    signal_fn=lambda data: build_factor_signal('mom12', data),
    factor_name='12-month momentum',
)
```

Or drop to the lower-level API:

```python
from qf.data import prepare_data
from qf.signals import build_factor_signal
from qf.backtest import run_event_driven

data = prepare_data()                           # WRDS CRSP + Compustat
signal = build_factor_signal('gpa', data)       # gross profitability
result, metrics = run_event_driven(data, signal, turnover_penalty=0.25)
print(f"Sharpe={metrics['sharpe']:.3f}, CAGR={metrics['cagr']:.2%}")
```

## Design notes

1. **Event-driven, not vectorized.** The backtester iterates one period
   at a time so you can plug in arbitrary stateful rebalance logic
   (regime switching, drawdown control, vol targeting) without
   rewriting the engine. Vectorized backtests are faster but every
   real desk eventually needs the event-driven path — this library
   starts there.

2. **Turnover penalty is a first-class citizen.** Every backtest
   takes a `turnover_penalty` parameter that shrinks signal changes
   before ranking. This makes cost-robustness testing trivial — sweep
   the penalty from 0 to 1 and watch the Sharpe curve.

3. **Walk-forward CV is the default validation protocol.** Not
   in-sample Sharpe. `qf/stat_toolkit.py` ships `WalkForwardCV` with
   configurable train/test windows, and `qf/risk.GateCheck` runs the
   SOP-compliant gate-check system on every strategy before it's
   considered tradable.

4. **Multi-market, single-engine.** The same `prepare_data()` /
   `build_factor_signal` / `run_event_driven` trio runs unchanged on
   US and China A-share universes — swap the data loader and the
   rest of the pipeline is agnostic.

5. **Cost model sanity.** `qf/costs.py` implements 5 execution-cost
   variants (fixed bps, tiered by venue, sqrt-impact, linear-impact,
   and the full Kyle/Almgren-Chriss model). Studies that look like
   alpha under the fixed model often disappear under sqrt-impact —
   this library makes that comparison one line.

## What's NOT in this package

- **Live order execution** (`live_executor.py`, `alpaca_bridge.py`) —
  kept private in the `kuant` platform. This library is for research
  and backtesting; live trading needs its own operational discipline
  that doesn't belong in a public pypi package.
- **Per-strategy implementations** — see the companion
  [`kuant-strategies`](https://github.com/zwmjj/kuant-strategies)
  repo for 25+ strategies that depend on this library.
- **Web UI, FastAPI backend, scheduled cron** — see
  [`kuant-api`](https://github.com/zwmjj/kuant-api) and
  [`kuant-web`](https://github.com/zwmjj/kuant-web).

## Project layout

```
kuant-core/
├── qf/                             # the package
│   ├── __init__.py                 # public API facade
│   ├── data.py                     # WRDS / CRSP / Compustat / yfinance
│   ├── data_cn.py                  # A-share (baostock monthly)
│   ├── data_cn_intraday.py         # A-share intraday (akshare minute bars)
│   ├── data_alpaca.py              # Alpaca historical / live data feed (no orders)
│   ├── data_stream.py              # real-time streaming bars
│   ├── signals.py                  # 28+ factor signal library
│   ├── signals_advanced.py         # interaction, orthogonal, regime-adjusted
│   ├── signals_alpha.py            # alpha-specific composites
│   ├── signals_cn.py               # A-share factors
│   ├── signals_daily.py            # daily-frequency signal generators
│   ├── signals_macro.py            # macro regime signals
│   ├── signals_news.py             # news sentiment signals
│   ├── signals_options.py          # options-implied signals
│   ├── signals_screener.py         # cross-sectional screening
│   ├── signals_realtime.py         # streaming signal pipelines
│   ├── strategy.py                 # BaseStrategy abstract class
│   ├── strategy_crypto.py          # crypto-specific base
│   ├── multi_strategy.py           # strategy composition
│   ├── backtest.py                 # event-driven engine + BacktestResult
│   ├── costs.py                    # 5 cost models + ExecutionHandler
│   ├── portfolio.py                # covariance + portfolio optimization
│   ├── optimizer.py                # vol target / DD control / risk parity
│   ├── factor_analysis.py          # IC / ICIR / quantile analytics
│   ├── factor_importance.py        # SHAP-style importance for blends
│   ├── attribution.py              # FF5 regression with Newey-West
│   ├── risk.py                     # RiskAnalyzer + GateCheck
│   ├── risk_manager.py             # live-risk monitoring primitives
│   ├── anomaly_detector.py         # return-series anomaly detection
│   ├── regime_detector.py          # macro regime classification
│   ├── stress.py                   # crisis replay + cost sensitivity
│   ├── stat_toolkit.py             # walk-forward CV + GSA audit
│   ├── framework.py                # Framework facade + one-line runner
│   └── viz.py                      # tearsheet + plots
├── tests/                          # pytest suite
├── examples/                       # minimal runnable examples
├── pyproject.toml
├── README.md
└── LICENSE
```

## License

MIT. Data dependencies have their own terms:
- WRDS / CRSP / Compustat: academic license, NOT redistributed by
  this library — users need their own WRDS subscription
- Kenneth French data library: free for academic use
- akshare: free (scrapes public Sina/Tencent endpoints)
- yfinance, pandas-datareader: free, best-effort terms of service
