"""Backtrader 策略适配器
Backtrader: https://github.com/mementum/backtrader
将 Kuant 信号转为 Backtrader 策略, 支持日频事件驱动回测。
"""
import warnings
import numpy as np
import pandas as pd
from typing import Optional

warnings.filterwarnings("ignore")

try:
    import backtrader as bt

    _HAS_BT = True
except ImportError:
    _HAS_BT = False


def _check_bt():
    if not _HAS_BT:
        raise ImportError("backtrader 未安装。请运行: pip install backtrader")


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

if _HAS_BT:

    class KQBacktraderStrategy(bt.Strategy):
        """Kuant 信号驱动的 Backtrader 策略。

        将预计算的 KQ 信号矩阵 (date x stock) 注入 Backtrader 框架,
        在每个 rebalance bar 上按信号排名分配仓位。

        Parameters (via params)
        -----------------------
        signal_dict : dict
            {stock_name: pd.Series(date -> signal_value)}
        top_n : int
            做多股票数量
        rebalance_freq : int
            再平衡频率 (bar 数, 如月频=21)
        """

        params = (
            ("signal_dict", {}),
            ("top_n", 20),
            ("rebalance_freq", 21),
        )

        def __init__(self):
            self.bar_count = 0
            self.order_dict = {}

        def next(self):
            self.bar_count += 1
            if self.bar_count % self.params.rebalance_freq != 0:
                return

            current_date = self.datas[0].datetime.date(0)

            # Collect signals for all data feeds
            signals = {}
            for data in self.datas:
                name = data._name
                if name in self.params.signal_dict:
                    sig_series = self.params.signal_dict[name]
                    # Find nearest date
                    ts = pd.Timestamp(current_date)
                    if ts in sig_series.index:
                        signals[name] = sig_series[ts]
                    else:
                        # Find closest prior date
                        prior = sig_series.index[sig_series.index <= ts]
                        if len(prior) > 0:
                            signals[name] = sig_series[prior[-1]]

            if not signals:
                return

            # Rank and select top N
            ranked = sorted(signals.items(), key=lambda x: x[1], reverse=True)
            top = [name for name, _ in ranked[: self.params.top_n]]

            # Equal weight
            target_pct = 1.0 / self.params.top_n if self.params.top_n > 0 else 0

            # Close positions not in top
            for data in self.datas:
                pos = self.getposition(data).size
                if data._name not in top and pos != 0:
                    self.close(data)

            # Open / adjust positions
            for data in self.datas:
                if data._name in top:
                    self.order_target_percent(data, target=target_pct)

else:
    # Fallback stub when backtrader is not installed
    class KQBacktraderStrategy:
        """Stub: backtrader 未安装。"""

        def __init__(self, *args, **kwargs):
            raise ImportError("backtrader 未安装。请运行: pip install backtrader")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _signal_to_dict(signal: pd.DataFrame) -> dict:
    """将 KQ 信号矩阵 (date x stock) 转为 {stock: Series}。"""
    result = {}
    for col in signal.columns:
        s = signal[col].dropna()
        if len(s) > 0:
            result[str(col)] = s
    return result


def run_bt_backtest(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    cash: float = 1_000_000.0,
    commission: float = 0.001,
    top_n: int = 20,
    rebalance_freq: int = 21,
) -> dict:
    """使用 Backtrader 运行回测。

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

    Returns
    -------
    dict
        回测结果, 含 final_value, total_return, trades 等
    """
    _check_bt()

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)

    # Add data feeds
    signal_dict = _signal_to_dict(signal)
    common_cols = [c for c in prices.columns if str(c) in signal_dict]

    if not common_cols:
        raise ValueError("信号与价格无重叠股票")

    for col in common_cols:
        col_str = str(col)
        df = prices[[col]].dropna().rename(columns={col: "close"})
        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]
        df["volume"] = 1000
        df["openinterest"] = 0

        data = bt.feeds.PandasData(
            dataname=df,
            datetime=None,
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            openinterest="openinterest",
        )
        data._name = col_str
        cerebro.adddata(data, name=col_str)

    # Add strategy
    cerebro.addstrategy(
        KQBacktraderStrategy,
        signal_dict=signal_dict,
        top_n=min(top_n, len(common_cols)),
        rebalance_freq=rebalance_freq,
    )

    # Run
    results = cerebro.run()
    return convert_bt_results(cerebro, cash)


def convert_bt_results(cerebro, initial_cash: float = 1_000_000.0) -> dict:
    """将 Backtrader cerebro 结果转为 JSON-serializable dict。

    Parameters
    ----------
    cerebro : bt.Cerebro
        已运行的 Cerebro 实例
    initial_cash : float
        初始资金

    Returns
    -------
    dict
    """
    _check_bt()

    final_value = cerebro.broker.getvalue()
    total_return = final_value / initial_cash - 1

    return {
        "initial_cash": initial_cash,
        "final_value": float(final_value),
        "total_return": float(total_return),
        "pnl": float(final_value - initial_cash),
    }
