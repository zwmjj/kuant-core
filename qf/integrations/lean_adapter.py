"""QuantConnect LEAN 引擎适配器
QuantConnect LEAN: https://github.com/QuantConnect/Lean
将 Kuant 因子/策略集成到 LEAN 引擎，支持回测与实盘。
"""
import json
import os
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 条件导入 LEAN (QuantConnect)
# ---------------------------------------------------------------------------

try:
    from AlgorithmImports import *  # LEAN 标准导入
    from QuantConnect import (
        Resolution,
        SecurityType,
        Market,
    )
    from QuantConnect.Algorithm import QCAlgorithm
    from QuantConnect.Algorithm.Framework.Alphas import InsightDirection
    from QuantConnect.Data import SubscriptionDataSource
    from QuantConnect.Orders import OrderDirection
    from QuantConnect.Brokerages import BrokerageName

    _HAS_LEAN = True
except ImportError:
    _HAS_LEAN = False


def _check_lean():
    """检查 LEAN 是否可用，不可用时抛出清晰提示。"""
    if not _HAS_LEAN:
        raise ImportError(
            "QuantConnect LEAN 未安装。\n"
            "  本地引擎: pip install quantconnect\n"
            "  或使用 QuantConnect Cloud: https://www.quantconnect.com/\n"
            "  LEAN CLI: pip install lean"
        )


# ---------------------------------------------------------------------------
# 符号映射工具
# ---------------------------------------------------------------------------


def map_symbol_kuant_to_lean(symbol: str) -> str:
    """Kuant ticker → LEAN Symbol 字符串映射。

    Kuant 常用大写美股 ticker (如 'AAPL')，LEAN 也用大写，
    但中国 A 股需特殊处理 (如 '600000' → '600000 XSHG')。
    """
    symbol = str(symbol).strip().upper()

    # A 股：纯数字 6 位
    if symbol.isdigit() and len(symbol) == 6:
        if symbol.startswith(("60", "68")):
            return f"{symbol} XSHG"  # 上海
        else:
            return f"{symbol} XSHE"  # 深圳

    # 美股 / 加密：直接返回
    return symbol


def map_symbol_lean_to_kuant(symbol: str) -> str:
    """LEAN Symbol 字符串 → Kuant ticker 映射。"""
    symbol = str(symbol).strip()
    # 去掉交易所后缀
    for suffix in (" XSHG", " XSHE", " XNYS", " XNAS"):
        if symbol.endswith(suffix):
            return symbol.replace(suffix, "").strip()
    return symbol.upper()


# ---------------------------------------------------------------------------
# 信号转换工具
# ---------------------------------------------------------------------------


def signal_to_lean_targets(
    signal_df: pd.DataFrame,
    date: Union[str, pd.Timestamp, datetime],
    long_n: int = 50,
    short_n: int = 50,
    long_pct: float = 1.0,
    short_pct: float = 0.0,
) -> Dict[str, float]:
    """从 Kuant 信号矩阵提取目标持仓权重。

    Parameters
    ----------
    signal_df : pd.DataFrame
        信号矩阵 (date x stock)，值越高越看多
    date : str / Timestamp / datetime
        当前日期
    long_n : int
        做多股票数量
    short_n : int
        做空股票数量
    long_pct : float
        多头总仓位占比 (1.0 = 100%)
    short_pct : float
        空头总仓位占比

    Returns
    -------
    dict
        {symbol: weight}, 正值做多, 负值做空
    """
    date = pd.Timestamp(date)

    # 找最近的可用日期
    if date in signal_df.index:
        sig_row = signal_df.loc[date].dropna()
    else:
        prior = signal_df.index[signal_df.index <= date]
        if len(prior) == 0:
            return {}
        sig_row = signal_df.loc[prior[-1]].dropna()

    if len(sig_row) == 0:
        return {}

    targets = {}

    # 多头
    if long_n > 0 and long_pct > 0:
        long_stocks = sig_row.nlargest(min(long_n, len(sig_row)))
        w = long_pct / len(long_stocks)
        for sym in long_stocks.index:
            targets[str(sym)] = w

    # 空头
    if short_n > 0 and short_pct > 0:
        short_stocks = sig_row.nsmallest(min(short_n, len(sig_row)))
        w = short_pct / len(short_stocks)
        for sym in short_stocks.index:
            # 避免多空冲突
            if str(sym) not in targets:
                targets[str(sym)] = -w

    return targets


def lean_results_to_kuant(lean_results: Dict[str, Any]) -> Dict[str, Any]:
    """LEAN 回测结果转 Kuant 标准指标 dict。

    Parameters
    ----------
    lean_results : dict
        LEAN 输出的回测统计（从 backtests/ JSON 或 API 获取）

    Returns
    -------
    dict
        Kuant 标准指标，兼容 BacktestResult.metrics() 输出格式
    """
    stats = lean_results.get("Statistics", lean_results)

    def _safe_pct(key: str, fallback: float = 0.0) -> float:
        """解析 LEAN 百分比字符串 (如 '12.34%') 为浮点数。"""
        val = stats.get(key, fallback)
        if isinstance(val, str):
            val = val.replace("%", "").replace(",", "").strip()
            try:
                return float(val) / 100.0
            except ValueError:
                return fallback
        return float(val) if val is not None else fallback

    def _safe_float(key: str, fallback: float = 0.0) -> float:
        val = stats.get(key, fallback)
        if isinstance(val, str):
            val = val.replace("$", "").replace(",", "").strip()
            try:
                return float(val)
            except ValueError:
                return fallback
        return float(val) if val is not None else fallback

    return {
        "total_return": _safe_pct("Total Net Profit", _safe_pct("total_return")),
        "cagr": _safe_pct("Compounding Annual Return", _safe_pct("cagr")),
        "sharpe": _safe_float("Sharpe Ratio", _safe_float("sharpe")),
        "sortino": _safe_float("Sortino Ratio", _safe_float("sortino")),
        "max_drawdown": _safe_pct("Drawdown", _safe_pct("max_drawdown")),
        "win_rate": _safe_pct("Win Rate", _safe_pct("win_rate")),
        "final_value": _safe_float("Net Profit"),
        "total_trades": int(_safe_float("Total Trades")),
        "alpha": _safe_pct("Alpha"),
        "beta": _safe_float("Beta"),
        "treynor_ratio": _safe_float("Treynor Ratio"),
        "information_ratio": _safe_float("Information Ratio"),
    }


# ---------------------------------------------------------------------------
# LEAN 算法基类
# ---------------------------------------------------------------------------

if _HAS_LEAN:

    class KuantAlgorithm(QCAlgorithm):
        """LEAN 算法基类，集成 Kuant 信号。

        子类需设置:
            self.signal_df: 信号矩阵 (date x stock DataFrame)
            self.universe_symbols: 股票池列表

        或传入 BaseStrategy 实例使用 KuantStrategyWrapper。
        """

        def Initialize(self):
            """初始化算法：设置 universe, resolution, rebalance schedule。"""
            # --- 默认参数 (子类可在 super().Initialize() 前覆盖) ---
            if not hasattr(self, "start_date"):
                self.start_date = datetime(2020, 1, 1)
            if not hasattr(self, "end_date"):
                self.end_date = datetime(2024, 12, 31)
            if not hasattr(self, "initial_cash"):
                self.initial_cash = 1_000_000
            if not hasattr(self, "resolution"):
                self.resolution = Resolution.Daily
            if not hasattr(self, "universe_symbols"):
                self.universe_symbols = []
            if not hasattr(self, "signal_df"):
                self.signal_df = pd.DataFrame()
            if not hasattr(self, "long_n"):
                self.long_n = 50
            if not hasattr(self, "short_n"):
                self.short_n = 0
            if not hasattr(self, "long_pct"):
                self.long_pct = 1.0
            if not hasattr(self, "short_pct"):
                self.short_pct = 0.0
            if not hasattr(self, "rebalance_days"):
                self.rebalance_days = 21  # 月频
            if not hasattr(self, "max_position_pct"):
                self.max_position_pct = 0.10  # 单票上限 10%
            if not hasattr(self, "max_sector_pct"):
                self.max_sector_pct = 0.30  # 行业上限 30%

            self.SetStartDate(self.start_date)
            self.SetEndDate(self.end_date)
            self.SetCash(self.initial_cash)

            # 添加股票
            self._lean_symbols = {}
            for sym in self.universe_symbols:
                lean_sym = self.AddEquity(
                    map_symbol_kuant_to_lean(sym), self.resolution
                ).Symbol
                self._lean_symbols[str(sym)] = lean_sym

            # 再平衡计数器
            self._bar_count = 0
            self._last_rebalance = None

        def OnData(self, data):
            """接收行情，触发信号计算和下单。"""
            self._bar_count += 1
            if self._bar_count % self.rebalance_days != 0:
                return

            self.Rebalance(data)

        def Rebalance(self, data=None):
            """定期再平衡，调用 Kuant 信号生成目标仓位。"""
            current_date = self.Time

            if self.signal_df is None or self.signal_df.empty:
                return

            targets = signal_to_lean_targets(
                self.signal_df,
                current_date,
                long_n=self.long_n,
                short_n=self.short_n,
                long_pct=self.long_pct,
                short_pct=self.short_pct,
            )

            if not targets:
                return

            # 风控
            targets = self._apply_risk_limits(targets)

            # 转为 LEAN 订单
            self._signal_to_orders(targets)

            self._last_rebalance = current_date

        def _signal_to_orders(self, targets: Dict[str, float]):
            """将目标持仓权重 dict 转为 LEAN SetHoldings 调用。

            Parameters
            ----------
            targets : dict
                {kuant_symbol: weight}，正值做多、负值做空
            """
            # 平掉不在目标中的持仓
            for kvp in self.Portfolio:
                holding = kvp.Value
                if holding.Invested:
                    kuant_sym = map_symbol_lean_to_kuant(str(holding.Symbol))
                    if kuant_sym not in targets:
                        self.Liquidate(holding.Symbol)

            # 设置目标仓位
            for kuant_sym, weight in targets.items():
                if kuant_sym in self._lean_symbols:
                    lean_sym = self._lean_symbols[kuant_sym]
                    self.SetHoldings(lean_sym, weight)

        def _apply_risk_limits(self, targets: Dict[str, float]) -> Dict[str, float]:
            """应用风控限制：单票上限、行业上限等。

            Parameters
            ----------
            targets : dict
                {symbol: weight}

            Returns
            -------
            dict
                调整后的 {symbol: weight}
            """
            adjusted = {}
            for sym, w in targets.items():
                # 单票上限
                capped = min(abs(w), self.max_position_pct) * np.sign(w)
                adjusted[sym] = capped

            # 确保总多头和空头权重不超标
            long_total = sum(w for w in adjusted.values() if w > 0)
            short_total = sum(abs(w) for w in adjusted.values() if w < 0)

            if long_total > self.long_pct and long_total > 0:
                scale = self.long_pct / long_total
                adjusted = {
                    s: w * scale if w > 0 else w for s, w in adjusted.items()
                }

            if short_total > self.short_pct and self.short_pct > 0 and short_total > 0:
                scale = self.short_pct / short_total
                adjusted = {
                    s: w * scale if w < 0 else w for s, w in adjusted.items()
                }

            return adjusted


    class KuantUniverseSelection:
        """自定义 Universe 选股，对接 Kuant 的 tradable_mask。

        在 LEAN 的 CoarseSelectionFunction 中使用，根据 Kuant 信号矩阵
        的列名（即有效信号的股票）来筛选 Universe。
        """

        def __init__(self, signal_df: pd.DataFrame, tradable_mask: Optional[pd.DataFrame] = None):
            """
            Parameters
            ----------
            signal_df : pd.DataFrame
                信号矩阵 (date x stock)
            tradable_mask : pd.DataFrame, optional
                可交易掩码 (date x stock, bool)，None 则使用信号非空判断
            """
            self.signal_df = signal_df
            self.tradable_mask = tradable_mask

        def get_universe(self, date: Union[str, pd.Timestamp, datetime]) -> List[str]:
            """获取指定日期的可交易股票池。"""
            date = pd.Timestamp(date)
            if self.tradable_mask is not None and date in self.tradable_mask.index:
                row = self.tradable_mask.loc[date]
                return [str(s) for s in row[row].index]
            elif date in self.signal_df.index:
                return [str(s) for s in self.signal_df.loc[date].dropna().index]
            return []

        def CoarseSelectionFunction(self, algorithm, coarse):
            """LEAN CoarseSelectionFunction 接口。"""
            universe = self.get_universe(algorithm.Time)
            return [
                c.Symbol
                for c in coarse
                if map_symbol_lean_to_kuant(str(c.Symbol)) in universe
            ]


    class KuantFeeModel:
        """自定义费率模型，对接 Kuant 的 ExecutionHandler 成本参数。

        将 Kuant 的 commission_bps, spread_bps 等映射到 LEAN 的 FeeModel 接口。
        """

        def __init__(
            self,
            commission_bps: float = 1.0,
            spread_bps: float = 5.0,
            short_borrow_bps: float = 30.0,
        ):
            """
            Parameters
            ----------
            commission_bps : float
                手续费 (基点)
            spread_bps : float
                买卖价差 (基点)
            short_borrow_bps : float
                融券年化费率 (基点)
            """
            self.commission_rate = commission_bps / 10000.0
            self.spread_rate = spread_bps / 10000.0
            self.short_borrow_rate = short_borrow_bps / 10000.0 / 252.0  # 日化

        def GetOrderFee(self, parameters):
            """LEAN FeeModel 接口：计算订单费用。"""
            order = parameters.Order
            security = parameters.Security
            price = security.Price
            quantity = abs(order.Quantity)
            notional = price * quantity

            fee = notional * self.commission_rate
            fee += notional * self.spread_rate / 2.0  # 半边价差

            if order.Direction == OrderDirection.Sell:
                # 融券借入成本 (按日)
                fee += notional * self.short_borrow_rate

            from QuantConnect.Orders.Fees import OrderFee
            from QuantConnect.Securities import CashAmount

            return OrderFee(CashAmount(fee, "USD"))

else:
    # LEAN 未安装时的 Stub 类

    class KuantAlgorithm:
        """Stub: QuantConnect LEAN 未安装。"""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "QuantConnect LEAN 未安装。请运行: pip install quantconnect"
            )

    class KuantUniverseSelection:
        """Stub: QuantConnect LEAN 未安装。"""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "QuantConnect LEAN 未安装。请运行: pip install quantconnect"
            )

    class KuantFeeModel:
        """Stub: QuantConnect LEAN 未安装。"""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "QuantConnect LEAN 未安装。请运行: pip install quantconnect"
            )


# ---------------------------------------------------------------------------
# 策略包装器
# ---------------------------------------------------------------------------


class KuantStrategyWrapper:
    """包装任意 Kuant BaseStrategy，转为 LEAN 可执行算法。

    在 LEAN 的 OnData 中调用 strategy.generate_signal()，
    将 Kuant 信号格式 (date x stock DataFrame) 转为 LEAN 订单。

    用法:
        from qf.strategy import BaseStrategy

        class MyStrat(BaseStrategy):
            name = "动量因子"
            def generate_signal(self, data):
                return signal_df

        wrapper = KuantStrategyWrapper(MyStrat(), data_dict)
        # 将 wrapper 传入 run_lean_backtest() 或 generate_lean_project()
    """

    def __init__(
        self,
        strategy,
        data: Optional[dict] = None,
        long_n: Optional[int] = None,
        short_n: Optional[int] = None,
        long_pct: Optional[float] = None,
        short_pct: Optional[float] = None,
        rebalance_days: int = 21,
    ):
        """
        Parameters
        ----------
        strategy : BaseStrategy
            Kuant 策略实例
        data : dict, optional
            Kuant 数据字典 (含 'prices', 'returns' 等)，用于 generate_signal
        long_n : int, optional
            做多数量，None 则取策略默认值
        short_n : int, optional
            做空数量
        long_pct : float, optional
            多头仓位比例
        short_pct : float, optional
            空头仓位比例
        rebalance_days : int
            再平衡周期 (交易日)
        """
        self.strategy = strategy
        self.data = data
        self.long_n = long_n if long_n is not None else strategy.long_n
        self.short_n = short_n if short_n is not None else strategy.short_n
        self.long_pct = long_pct if long_pct is not None else strategy.long_pct
        self.short_pct = short_pct if short_pct is not None else strategy.short_pct
        self.rebalance_days = rebalance_days

        # 预计算信号
        self._signal_df = None
        if data is not None:
            self._precompute_signal()

    def _precompute_signal(self):
        """预计算信号矩阵。"""
        try:
            self._signal_df = self.strategy.generate_signal(self.data)
        except Exception as e:
            warnings.warn(f"信号预计算失败: {e}")
            self._signal_df = pd.DataFrame()

    @property
    def signal_df(self) -> pd.DataFrame:
        """获取信号矩阵，懒加载。"""
        if self._signal_df is None and self.data is not None:
            self._precompute_signal()
        return self._signal_df if self._signal_df is not None else pd.DataFrame()

    def get_targets(self, date: Union[str, pd.Timestamp, datetime]) -> Dict[str, float]:
        """获取指定日期的目标持仓权重。

        Parameters
        ----------
        date : 日期

        Returns
        -------
        dict
            {symbol: weight}
        """
        return signal_to_lean_targets(
            self.signal_df,
            date,
            long_n=self.long_n,
            short_n=self.short_n,
            long_pct=self.long_pct,
            short_pct=self.short_pct,
        )

    def get_universe(self) -> List[str]:
        """从信号矩阵提取完整股票池。"""
        if self.signal_df is not None and not self.signal_df.empty:
            return [str(c) for c in self.signal_df.columns]
        return []

    def generate_lean_algorithm_code(self) -> str:
        """生成可在 LEAN 中直接运行的 Python 算法代码。

        Returns
        -------
        str
            完整的 LEAN 算法 Python 代码
        """
        strategy_name = self.strategy.name.replace(" ", "_")
        symbols_str = ", ".join(f'"{s}"' for s in self.get_universe()[:200])

        code = f'''"""由 Kuant 平台自动生成的 LEAN 算法
策略: {self.strategy.name}
{self.strategy.description}
"""
from AlgorithmImports import *
import numpy as np
import pandas as pd


class {strategy_name}Algorithm(QCAlgorithm):
    """Kuant 策略 '{self.strategy.name}' 的 LEAN 实现。"""

    def Initialize(self):
        self.SetStartDate({self.data.get("start_year", 2020)}, 1, 1)
        self.SetEndDate({self.data.get("end_year", 2024)}, 12, 31)
        self.SetCash(1000000)

        # 股票池
        symbols = [{symbols_str}]
        self.lean_symbols = {{}}
        for sym in symbols:
            equity = self.AddEquity(sym, Resolution.Daily)
            self.lean_symbols[sym] = equity.Symbol

        # 再平衡
        self.rebalance_days = {self.rebalance_days}
        self.bar_count = 0
        self.long_n = {self.long_n}
        self.short_n = {self.short_n}
        self.long_pct = {self.long_pct}
        self.short_pct = {self.short_pct}

    def OnData(self, data):
        self.bar_count += 1
        if self.bar_count % self.rebalance_days != 0:
            return
        self.Rebalance()

    def Rebalance(self):
        # 这里放入策略信号逻辑
        # 实际使用时，通过 ObjectStore 或文件加载预计算信号
        pass
'''
        return code


# ---------------------------------------------------------------------------
# 回测运行器
# ---------------------------------------------------------------------------


def run_lean_backtest(
    strategy: Union[KuantStrategyWrapper, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """配置并运行 LEAN 本地回测。

    通过 LEAN CLI (lean-cli) 在本地 Docker 中运行回测。
    需要预先安装: pip install lean

    Parameters
    ----------
    strategy : KuantStrategyWrapper 或 BaseStrategy
        Kuant 策略 (如果是 BaseStrategy 会自动包装)
    config : dict
        回测配置:
        - start_date: str, 如 '2020-01-01'
        - end_date: str, 如 '2024-12-31'
        - cash: float, 初始资金 (默认 1,000,000)
        - resolution: str, 'daily' / 'hour' / 'minute' (默认 'daily')
        - universe: list[str], 股票池 (可选, 从策略推断)
        - data_path: str, LEAN 数据目录 (可选)
        - output_dir: str, 输出目录 (默认 './lean_output')
        - lean_cli: bool, 是否用 lean-cli 运行 (默认 True)

    Returns
    -------
    dict
        标准化结果 (total_return, sharpe, max_drawdown, cagr 等)
    """
    # 自动包装 BaseStrategy
    if not isinstance(strategy, KuantStrategyWrapper):
        from qf.strategy import BaseStrategy
        if isinstance(strategy, BaseStrategy):
            strategy = KuantStrategyWrapper(strategy, config.get("data"))
        else:
            raise TypeError(f"strategy 须为 KuantStrategyWrapper 或 BaseStrategy, 收到 {type(strategy)}")

    output_dir = config.get("output_dir", "./lean_output")
    project_dir = os.path.join(output_dir, "project")

    # 生成 LEAN 项目
    generate_lean_project(strategy, project_dir, config)

    # 使用 lean-cli 运行
    if config.get("lean_cli", True):
        import subprocess

        lean_data = config.get("data_path", "")
        cmd = ["lean", "backtest", project_dir]
        if lean_data:
            cmd.extend(["--data", lean_data])

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600
            )
        except FileNotFoundError:
            raise RuntimeError(
                "lean-cli 未安装。请运行: pip install lean\n"
                "并配置: lean init"
            )

        if result.returncode != 0:
            raise RuntimeError(f"LEAN 回测失败:\n{result.stderr}")

        # 解析结果
        results_file = _find_results_json(project_dir)
        if results_file and os.path.exists(results_file):
            with open(results_file, "r") as f:
                raw = json.load(f)
            return lean_results_to_kuant(raw)

        # 从 stdout 解析
        return _parse_lean_stdout(result.stdout)

    # 非 CLI 模式：返回项目路径，用户自行运行
    return {
        "status": "project_generated",
        "project_dir": project_dir,
        "message": "LEAN 项目已生成，请手动运行或使用 lean-cli",
    }


def _find_results_json(project_dir: str) -> Optional[str]:
    """在 LEAN 输出目录中查找回测结果 JSON。"""
    backtests_dir = os.path.join(project_dir, "backtests")
    if not os.path.exists(backtests_dir):
        return None
    for root, dirs, files in os.walk(backtests_dir):
        for f in files:
            if f.endswith(".json"):
                return os.path.join(root, f)
    return None


def _parse_lean_stdout(stdout: str) -> Dict[str, Any]:
    """从 LEAN CLI 标准输出中解析关键指标。"""
    metrics = {}
    lines = stdout.split("\n")
    for line in lines:
        line = line.strip()
        for key, metric_name in [
            ("Total Net Profit", "total_return"),
            ("Sharpe Ratio", "sharpe"),
            ("Compounding Annual Return", "cagr"),
            ("Drawdown", "max_drawdown"),
            ("Win Rate", "win_rate"),
            ("Total Trades", "total_trades"),
        ]:
            if key in line:
                parts = line.split(key)
                if len(parts) > 1:
                    val_str = parts[-1].strip().strip("|").strip()
                    val_str = val_str.replace("%", "").replace("$", "").replace(",", "").strip()
                    try:
                        val = float(val_str)
                        if "%" in line:
                            val /= 100.0
                        metrics[metric_name] = val
                    except ValueError:
                        pass
    return metrics


# ---------------------------------------------------------------------------
# 实盘桥接
# ---------------------------------------------------------------------------


def run_lean_live(
    strategy: Union[KuantStrategyWrapper, Any],
    broker_config: Dict[str, Any],
) -> Dict[str, Any]:
    """连接券商实盘，通过 LEAN CLI 启动实盘交易。

    Parameters
    ----------
    strategy : KuantStrategyWrapper 或 BaseStrategy
        Kuant 策略
    broker_config : dict
        券商配置:
        - broker: str, 券商名称，目前支持 'alpaca' (默认)
        - api_key: str, API key
        - secret: str, API secret
        - paper_trading: bool, 是否纸盘 (默认 True)
        - base_url: str, API 地址 (可选)
        - environment: str, 'paper' / 'live' (可选, 从 paper_trading 推断)
        - data_feed: str, 数据源 (可选, 默认跟随 broker)
        - output_dir: str, 项目输出目录

    Returns
    -------
    dict
        启动状态信息
    """
    # 自动包装
    if not isinstance(strategy, KuantStrategyWrapper):
        from qf.strategy import BaseStrategy
        if isinstance(strategy, BaseStrategy):
            strategy = KuantStrategyWrapper(strategy)
        else:
            raise TypeError(f"strategy 须为 KuantStrategyWrapper 或 BaseStrategy")

    broker = broker_config.get("broker", "alpaca").lower()
    paper = broker_config.get("paper_trading", True)
    output_dir = broker_config.get("output_dir", "./lean_live")
    project_dir = os.path.join(output_dir, "project")

    # 生成项目
    live_config = {
        "start_date": datetime.now().strftime("%Y-%m-%d"),
        "end_date": (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d"),
        "cash": broker_config.get("cash", 100_000),
        "resolution": broker_config.get("resolution", "daily"),
    }
    generate_lean_project(strategy, project_dir, live_config)

    # 生成 LEAN 实盘配置
    lean_config = generate_lean_config(
        strategy_name=strategy.strategy.name,
        start=live_config["start_date"],
        end=live_config["end_date"],
        cash=live_config["cash"],
        resolution=live_config["resolution"],
        broker=broker,
    )

    # 注入券商密钥
    if broker == "alpaca":
        lean_config["environments"] = {
            "live-alpaca": {
                "live-mode": True,
                "live-mode-brokerage": "AlpacaBrokerage",
                "alpaca-api-key": broker_config.get("api_key", ""),
                "alpaca-api-secret": broker_config.get("secret", ""),
                "alpaca-trading-mode": "paper" if paper else "live",
                "alpaca-base-url": broker_config.get(
                    "base_url",
                    "https://paper-api.alpaca.markets"
                    if paper
                    else "https://api.alpaca.markets",
                ),
            }
        }

    config_path = os.path.join(project_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(lean_config, f, indent=2)

    # 使用 lean-cli 启动实盘
    import subprocess

    env_name = f"live-{broker}"
    cmd = [
        "lean", "live", project_dir,
        "--brokerage", broker.capitalize(),
        "--environment", env_name,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        return {
            "status": "error",
            "message": "lean-cli 未安装。请运行: pip install lean",
            "project_dir": project_dir,
        }

    return {
        "status": "started" if result.returncode == 0 else "error",
        "project_dir": project_dir,
        "broker": broker,
        "paper_trading": paper,
        "stdout": result.stdout,
        "stderr": result.stderr if result.returncode != 0 else "",
    }


# ---------------------------------------------------------------------------
# LEAN 配置生成
# ---------------------------------------------------------------------------

_RESOLUTION_MAP = {
    "daily": "Daily",
    "hour": "Hour",
    "minute": "Minute",
    "second": "Second",
    "tick": "Tick",
}


def generate_lean_config(
    strategy_name: str,
    start: str,
    end: str,
    cash: float = 1_000_000,
    resolution: str = "daily",
    broker: Optional[str] = None,
) -> Dict[str, Any]:
    """生成 LEAN 的 config.json 内容。

    Parameters
    ----------
    strategy_name : str
        策略名称
    start : str
        开始日期 'YYYY-MM-DD'
    end : str
        结束日期 'YYYY-MM-DD'
    cash : float
        初始资金
    resolution : str
        数据分辨率: 'daily', 'hour', 'minute'
    broker : str, optional
        券商: 'alpaca', None (回测模式)

    Returns
    -------
    dict
        LEAN config.json 内容
    """
    config = {
        "algorithm-type-name": strategy_name.replace(" ", ""),
        "algorithm-language": "Python",
        "algorithm-location": "main.py",
        "parameters": {},
        "start-date": start,
        "end-date": end,
        "cash-amount": cash,
        "data-folder": "../data",
        "results-destination-folder": "../results",
        "log-handler": "ConsoleLogHandler",
        "messaging-handler": "QuantConnect.Messaging.Messaging",
        "job-queue-handler": "QuantConnect.Queues.JobQueue",
        "api-handler": "QuantConnect.Api.Api",
        "map-file-provider": "QuantConnect.Data.Auxiliary.LocalDiskMapFileProvider",
        "factor-file-provider": "QuantConnect.Data.Auxiliary.LocalDiskFactorFileProvider",
    }

    if broker == "alpaca":
        config["live-mode"] = True
        config["live-mode-brokerage"] = "AlpacaBrokerage"

    return config


def generate_lean_project(
    strategy: Union[KuantStrategyWrapper, Any],
    output_dir: str,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """生成完整 LEAN 项目文件夹 (main.py + config.json)。

    Parameters
    ----------
    strategy : KuantStrategyWrapper
        包装后的 Kuant 策略
    output_dir : str
        输出目录路径
    config : dict, optional
        额外配置参数

    Returns
    -------
    str
        项目目录路径
    """
    if not isinstance(strategy, KuantStrategyWrapper):
        from qf.strategy import BaseStrategy
        if isinstance(strategy, BaseStrategy):
            strategy = KuantStrategyWrapper(strategy)
        else:
            raise TypeError("strategy 须为 KuantStrategyWrapper 或 BaseStrategy")

    config = config or {}
    os.makedirs(output_dir, exist_ok=True)

    # 生成 main.py
    main_code = strategy.generate_lean_algorithm_code()
    main_path = os.path.join(output_dir, "main.py")
    with open(main_path, "w", encoding="utf-8") as f:
        f.write(main_code)

    # 生成 config.json
    lean_config = generate_lean_config(
        strategy_name=strategy.strategy.name,
        start=config.get("start_date", "2020-01-01"),
        end=config.get("end_date", "2024-12-31"),
        cash=config.get("cash", 1_000_000),
        resolution=config.get("resolution", "daily"),
        broker=config.get("broker"),
    )
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(lean_config, f, indent=2, ensure_ascii=False)

    # 生成信号数据文件 (CSV)
    if strategy.signal_df is not None and not strategy.signal_df.empty:
        signal_path = os.path.join(output_dir, "signal_data.csv")
        strategy.signal_df.to_csv(signal_path)

    # 生成 requirements.txt
    req_path = os.path.join(output_dir, "requirements.txt")
    with open(req_path, "w") as f:
        f.write("numpy\npandas\n")

    return output_dir


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from qf.strategy import BaseStrategy

    # --- 示例策略 ---
    class MomentumStrategy(BaseStrategy):
        """动量因子策略示例"""
        name = "Momentum_12_1"
        description = "12个月动量 (跳过最近1个月)"
        long_n = 50
        short_n = 0

        def generate_signal(self, data: dict) -> pd.DataFrame:
            prices = data["prices"]
            # 12个月动量, 跳过最近1个月
            ret_12 = prices.pct_change(12)
            ret_1 = prices.pct_change(1)
            signal = ret_12 - ret_1
            return signal

    # --- 创建模拟数据 ---
    print("=" * 60)
    print("Kuant → LEAN 适配器演示")
    print("=" * 60)

    dates = pd.date_range("2020-01-01", "2024-12-31", freq="ME")
    stocks = [f"STOCK_{i}" for i in range(100)]
    np.random.seed(42)
    prices = pd.DataFrame(
        np.random.lognormal(0, 0.05, (len(dates), len(stocks))).cumprod(axis=0) * 100,
        index=dates,
        columns=stocks,
    )
    data = {"prices": prices, "returns": prices.pct_change()}

    # --- 包装策略 ---
    strat = MomentumStrategy()
    wrapper = KuantStrategyWrapper(strat, data, long_n=20, short_n=0, rebalance_days=21)

    print(f"\n策略: {strat.name}")
    print(f"股票池大小: {len(wrapper.get_universe())}")
    print(f"信号矩阵形状: {wrapper.signal_df.shape}")

    # --- 获取某天的目标仓位 ---
    sample_date = "2024-06-30"
    targets = wrapper.get_targets(sample_date)
    print(f"\n{sample_date} 目标持仓 (前5):")
    for sym, w in list(sorted(targets.items(), key=lambda x: -x[1]))[:5]:
        print(f"  {sym}: {w:.4f}")

    # --- 信号转换演示 ---
    signal = strat.generate_signal(data)
    targets_direct = signal_to_lean_targets(signal, "2024-06-30", long_n=20)
    print(f"\n直接信号转换: {len(targets_direct)} 只股票")

    # --- 生成 LEAN 项目 ---
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = os.path.join(tmpdir, "lean_project")
        generate_lean_project(
            wrapper,
            project_dir,
            config={
                "start_date": "2020-01-01",
                "end_date": "2024-12-31",
                "cash": 1_000_000,
            },
        )
        print(f"\nLEAN 项目已生成: {project_dir}")
        for f in os.listdir(project_dir):
            fpath = os.path.join(project_dir, f)
            size = os.path.getsize(fpath)
            print(f"  {f} ({size:,} bytes)")

    # --- LEAN 结果转换演示 ---
    mock_lean_results = {
        "Statistics": {
            "Total Net Profit": "45.23%",
            "Compounding Annual Return": "8.15%",
            "Sharpe Ratio": "1.23",
            "Sortino Ratio": "1.67",
            "Drawdown": "12.45%",
            "Win Rate": "55.3%",
            "Total Trades": "1234",
            "Alpha": "3.2%",
            "Beta": "0.85",
        }
    }
    kuant_metrics = lean_results_to_kuant(mock_lean_results)
    print(f"\nLEAN → Kuant 指标转换:")
    for k, v in kuant_metrics.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("注意: 实际运行 LEAN 回测/实盘需要安装 lean-cli:")
    print("  pip install lean")
    print("  lean init")
    print("  lean backtest <project_dir>")
    print("=" * 60)
