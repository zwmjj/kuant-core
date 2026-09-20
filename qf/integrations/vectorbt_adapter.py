"""vectorbt fast vectorized backtest adapter
vectorbt: https://github.com/polakvoj/vectorbt
Provides high-performance vectorized backtesting, portfolio optimization, signal
generation and performance analysis.
Results are returned as a dict so they can be JSON-serialized for the API layer.
"""
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Union

warnings.filterwarnings("ignore")

try:
    import vectorbt as vbt

    _HAS_VBT = True
except ImportError:
    _HAS_VBT = False


def _check_vbt():
    if not _HAS_VBT:
        raise ImportError("vectorbt 未安装。请运行: pip install vectorbt")


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _signal_to_entries_exits(
    signal: pd.DataFrame,
    top_n: int = 20,
    rebalance_freq: int = 21,
) -> tuple:
    """将 KQ 信号矩阵转为 vectorbt 的 entries/exits 布尔矩阵。

    Parameters
    ----------
    signal : pd.DataFrame
        信号矩阵 (date x stock)
    top_n : int
        每期持仓数量
    rebalance_freq : int
        再平衡频率 (交易日)

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (entries, exits) 布尔矩阵
    """
    entries = pd.DataFrame(False, index=signal.index, columns=signal.columns)
    exits = pd.DataFrame(False, index=signal.index, columns=signal.columns)

    prev_holdings = set()
    for i, date in enumerate(signal.index):
        if i % rebalance_freq != 0:
            continue
        row = signal.loc[date].dropna().nlargest(top_n)
        current = set(row.index)

        # New entries
        new_buys = current - prev_holdings
        for col in new_buys:
            entries.loc[date, col] = True

        # Exits
        sells = prev_holdings - current
        for col in sells:
            exits.loc[date, col] = True

        prev_holdings = current

    return entries, exits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_vbt_backtest(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    cash: float = 1_000_000.0,
    commission: float = 0.001,
    top_n: int = 20,
    rebalance_freq: int = 21,
    freq: str = "D",
) -> dict:
    """Run a fast vectorized backtest with vectorbt.

    Parameters
    ----------
    signal : pd.DataFrame
        KQ signal matrix (date x stock)
    prices : pd.DataFrame
        Price matrix (date x stock), daily
    cash : float
        Initial capital
    commission : float
        Commission rate
    top_n : int
        Number of positions to hold
    rebalance_freq : int
        Rebalance frequency (trading days)
    freq : str
        Data frequency, 'D' for daily, 'M' for monthly

    Returns
    -------
    dict
        Backtest results, including total_return, sharpe, max_drawdown, etc.
    """
    _check_vbt()

    # Align signal and prices
    common_cols = signal.columns.intersection(prices.columns)
    common_dates = signal.index.intersection(prices.index)
    if len(common_cols) == 0 or len(common_dates) == 0:
        raise ValueError("信号与价格无重叠股票或日期")

    prices_aligned = prices.loc[common_dates, common_cols].ffill()
    signal_aligned = signal.loc[common_dates, common_cols]

    entries, exits = _signal_to_entries_exits(
        signal_aligned, top_n=top_n, rebalance_freq=rebalance_freq
    )

    # Run vectorbt portfolio
    pf = vbt.Portfolio.from_signals(
        close=prices_aligned,
        entries=entries,
        exits=exits,
        init_cash=cash,
        fees=commission,
        freq=freq,
    )

    return _extract_vbt_results(pf)


def run_vbt_from_orders(
    prices: pd.DataFrame,
    size: pd.DataFrame,
    cash: float = 1_000_000.0,
    commission: float = 0.001,
    freq: str = "D",
) -> dict:
    """Run a backtest with vectorbt from target positions.

    Parameters
    ----------
    prices : pd.DataFrame
        Price matrix (date x stock)
    size : pd.DataFrame
        Target position weight matrix (date x stock), values in [0, 1]
    cash : float
        Initial capital
    commission : float
        Commission rate
    freq : str
        Data frequency

    Returns
    -------
    dict
        Backtest results
    """
    _check_vbt()

    common_cols = size.columns.intersection(prices.columns)
    common_dates = size.index.intersection(prices.index)
    prices_a = prices.loc[common_dates, common_cols].ffill()
    size_a = size.loc[common_dates, common_cols].fillna(0)

    pf = vbt.Portfolio.from_order_func(
        close=prices_a,
        order_func_nb=None,  # use target percent
        init_cash=cash,
        fees=commission,
        freq=freq,
    )

    return _extract_vbt_results(pf)


def run_vbt_indicator_backtest(
    prices: pd.DataFrame,
    fast_window: int = 10,
    slow_window: int = 50,
    cash: float = 1_000_000.0,
    commission: float = 0.001,
) -> dict:
    """Backtest vectorbt's built-in MA crossover signal.

    Parameters
    ----------
    prices : pd.DataFrame
        Price matrix (date x stock)
    fast_window : int
        Fast moving average window
    slow_window : int
        Slow moving average window
    cash : float
        Initial capital
    commission : float
        Commission rate

    Returns
    -------
    dict
        Backtest results
    """
    _check_vbt()

    fast_ma = vbt.MA.run(prices, window=fast_window)
    slow_ma = vbt.MA.run(prices, window=slow_window)

    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)

    pf = vbt.Portfolio.from_signals(
        close=prices,
        entries=entries,
        exits=exits,
        init_cash=cash,
        fees=commission,
    )

    return _extract_vbt_results(pf)


def _extract_vbt_results(pf) -> dict:
    """将 vectorbt Portfolio 结果转为 JSON-serializable dict。

    Parameters
    ----------
    pf : vbt.Portfolio
        vectorbt Portfolio 实例

    Returns
    -------
    dict
    """
    stats = pf.stats()

    result = {
        "total_return": float(pf.total_return()),
        "sharpe_ratio": float(stats.get("Sharpe Ratio", 0.0)),
        "max_drawdown": float(stats.get("Max Drawdown [%]", 0.0)) / 100.0,
        "annual_return": float(stats.get("Annualized Return [%]", 0.0)) / 100.0,
        "annual_volatility": float(stats.get("Annualized Volatility [%]", 0.0)) / 100.0,
        "calmar_ratio": float(stats.get("Calmar Ratio", 0.0)),
        "total_trades": int(stats.get("Total Trades", 0)),
        "win_rate": float(stats.get("Win Rate [%]", 0.0)) / 100.0,
        "profit_factor": float(stats.get("Profit Factor", 0.0)),
    }

    # Portfolio value series
    try:
        pv = pf.value()
        if isinstance(pv, pd.Series):
            result["portfolio_values"] = pv.tolist()
            result["portfolio_dates"] = [str(d) for d in pv.index]
    except Exception:
        pass

    return result


def get_vbt_portfolio(
    prices: pd.DataFrame,
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    cash: float = 1_000_000.0,
    commission: float = 0.001,
):
    """Return the raw vectorbt Portfolio object for further custom analysis.

    Parameters
    ----------
    prices : pd.DataFrame
    entries : pd.DataFrame
        Boolean matrix, True = buy
    exits : pd.DataFrame
        Boolean matrix, True = sell
    cash : float
    commission : float

    Returns
    -------
    vbt.Portfolio
    """
    _check_vbt()

    pf = vbt.Portfolio.from_signals(
        close=prices,
        entries=entries,
        exits=exits,
        init_cash=cash,
        fees=commission,
    )
    return pf
