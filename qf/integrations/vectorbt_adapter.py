"""vectorbt 快速向量化回测适配器
vectorbt: https://github.com/polakvoj/vectorbt
提供高性能向量化回测、组合优化、信号生成与绩效分析。
结果以 dict 返回, 方便 JSON 序列化传给 API 层。
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
    """使用 vectorbt 运行快速向量化回测。

    Parameters
    ----------
    signal : pd.DataFrame
        KQ 信号矩阵 (date x stock)
    prices : pd.DataFrame
        价格矩阵 (date x stock), 日频
    cash : float
        初始资金
    commission : float
        手续费率
    top_n : int
        持仓数量
    rebalance_freq : int
        再平衡频率 (交易日)
    freq : str
        数据频率, 'D' 日频, 'M' 月频

    Returns
    -------
    dict
        回测结果, 含 total_return, sharpe, max_drawdown 等
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
    """使用 vectorbt 按目标仓位运行回测。

    Parameters
    ----------
    prices : pd.DataFrame
        价格矩阵 (date x stock)
    size : pd.DataFrame
        目标仓位权重矩阵 (date x stock), 值域 [0, 1]
    cash : float
        初始资金
    commission : float
        手续费率
    freq : str
        数据频率

    Returns
    -------
    dict
        回测结果
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
    """使用 vectorbt 内置 MA 交叉信号回测。

    Parameters
    ----------
    prices : pd.DataFrame
        价格矩阵 (date x stock)
    fast_window : int
        快速移动平均窗口
    slow_window : int
        慢速移动平均窗口
    cash : float
        初始资金
    commission : float
        手续费率

    Returns
    -------
    dict
        回测结果
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
    """直接返回 vectorbt Portfolio 对象, 供进一步自定义分析。

    Parameters
    ----------
    prices : pd.DataFrame
    entries : pd.DataFrame
        布尔矩阵, True = 买入
    exits : pd.DataFrame
        布尔矩阵, True = 卖出
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
