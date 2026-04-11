"""EasyTrader A股实盘交易适配器
EasyTrader: https://github.com/shidenggui/easytrader
支持同花顺、miniQMT 等券商客户端自动下单。
内置安全机制: dry_run 模式、仓位限制、风控检查。
"""
import warnings
import logging
import numpy as np
import pandas as pd
from typing import Optional

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    import easytrader

    _HAS_EASYTRADER = True
except ImportError:
    _HAS_EASYTRADER = False


def _check_easytrader():
    if not _HAS_EASYTRADER:
        raise ImportError(
            "easytrader 未安装。请运行: pip install easytrader"
        )


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_trader = None
_dry_run = True  # 默认 dry_run, 安全第一
_max_position_pct = 0.10  # 单只股票最大仓位占比
_max_total_position_pct = 0.95  # 总仓位最大占比


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def init_trader(
    broker: str = "tonghuashun",
    dry_run: bool = True,
    max_position_pct: float = 0.10,
    max_total_position_pct: float = 0.95,
    **kwargs,
) -> object:
    """初始化券商交易接口。

    Parameters
    ----------
    broker : str
        券商类型:
        - 'tonghuashun' : 同花顺客户端
        - 'miniqmt'     : miniQMT (迅投)
    dry_run : bool
        True = 只打印不下单 (默认开启, 安全第一)
    max_position_pct : float
        单只股票最大仓位比例 (0~1)
    max_total_position_pct : float
        总仓位最大比例 (0~1)
    **kwargs
        传给 easytrader 的额外参数, 如:
        - exe_path: 客户端路径
        - user, password: 登录凭证

    Returns
    -------
    easytrader instance or None (dry_run mode)
    """
    global _trader, _dry_run, _max_position_pct, _max_total_position_pct

    _dry_run = dry_run
    _max_position_pct = max_position_pct
    _max_total_position_pct = max_total_position_pct

    if dry_run:
        logger.info("[DRY RUN] 模拟模式, 不会实际下单")
        _trader = None
        return None

    _check_easytrader()

    broker_map = {
        "tonghuashun": "ths",
        "ths": "ths",
        "miniqmt": "xq",  # easytrader uses different names
        "xq": "xq",
    }

    broker_key = broker_map.get(broker.lower())
    if broker_key is None:
        raise ValueError(
            f"不支持的券商: {broker}。支持: {list(broker_map.keys())}"
        )

    _trader = easytrader.use(broker_key)

    # Connect
    if "exe_path" in kwargs:
        _trader.connect(kwargs["exe_path"])
    if "user" in kwargs and "password" in kwargs:
        _trader.prepare(user=kwargs["user"], password=kwargs["password"])

    logger.info(f"已连接券商: {broker}")
    return _trader


def get_positions() -> pd.DataFrame:
    """获取当前持仓。

    Returns
    -------
    pd.DataFrame
        含 code, name, amount, available, cost, market_value, pnl_pct 列
    """
    if _dry_run or _trader is None:
        logger.info("[DRY RUN] get_positions()")
        return pd.DataFrame(columns=[
            "code", "name", "amount", "available",
            "cost_price", "market_value", "pnl_pct",
        ])

    _check_easytrader()
    positions = _trader.position
    if not positions:
        return pd.DataFrame()

    df = pd.DataFrame(positions)
    # 标准化列名
    col_map = {
        "证券代码": "code",
        "证券名称": "name",
        "股票余额": "amount",
        "可用余额": "available",
        "成本价": "cost_price",
        "市值": "market_value",
        "盈亏比例(%)": "pnl_pct",
    }
    df = df.rename(columns=col_map)
    return df


def get_balance() -> dict:
    """获取账户资金信息。

    Returns
    -------
    dict
        {'total_assets', 'available_cash', 'market_value', 'frozen'}
    """
    if _dry_run or _trader is None:
        logger.info("[DRY RUN] get_balance()")
        return {
            "total_assets": 0.0,
            "available_cash": 0.0,
            "market_value": 0.0,
            "frozen": 0.0,
        }

    _check_easytrader()
    balance = _trader.balance
    if not balance:
        return {}

    b = balance[0] if isinstance(balance, list) else balance
    return {
        "total_assets": float(b.get("总资产", b.get("asset_balance", 0))),
        "available_cash": float(b.get("可用金额", b.get("enable_balance", 0))),
        "market_value": float(b.get("证券市值", b.get("market_value", 0))),
        "frozen": float(b.get("冻结资金", b.get("frozen_balance", 0))),
    }


def _safety_check(
    code: str,
    amount: int,
    price: float,
    total_assets: float,
) -> bool:
    """安全检查: 单只股票仓位限制。"""
    if total_assets <= 0:
        return True
    position_value = amount * price
    pct = position_value / total_assets
    if pct > _max_position_pct:
        logger.warning(
            f"[安全] {code} 仓位 {pct:.1%} 超过限制 {_max_position_pct:.1%}, 拒绝下单"
        )
        return False
    return True


def execute_orders(
    orders: list,
) -> list:
    """执行订单列表。

    Parameters
    ----------
    orders : list[dict]
        每个 dict 含:
        - code: str     股票代码
        - action: str   'buy' | 'sell'
        - amount: int   股数 (必须为 100 的整数倍)
        - price: float  委托价格 (0 = 市价)

    Returns
    -------
    list[dict]
        执行结果, 含 status, message 等
    """
    results = []

    for order in orders:
        code = order.get("code", "")
        action = order.get("action", "").lower()
        amount = int(order.get("amount", 0))
        price = float(order.get("price", 0))

        # 基本校验
        if not code or action not in ("buy", "sell") or amount <= 0:
            results.append({
                "code": code,
                "action": action,
                "status": "rejected",
                "message": "参数无效",
            })
            continue

        # 整手检查 (A股 100 股为一手)
        if amount % 100 != 0:
            amount = (amount // 100) * 100
            if amount == 0:
                results.append({
                    "code": code,
                    "action": action,
                    "status": "rejected",
                    "message": "数量不足一手 (100股)",
                })
                continue

        if _dry_run or _trader is None:
            logger.info(
                f"[DRY RUN] {action.upper()} {code} x {amount} @ {price}"
            )
            results.append({
                "code": code,
                "action": action,
                "amount": amount,
                "price": price,
                "status": "simulated",
                "message": "dry_run 模式",
            })
            continue

        _check_easytrader()

        # 安全检查
        balance = get_balance()
        total = balance.get("total_assets", 0)
        if action == "buy" and not _safety_check(code, amount, price, total):
            results.append({
                "code": code,
                "action": action,
                "status": "rejected",
                "message": "超过仓位限制",
            })
            continue

        try:
            if action == "buy":
                if price > 0:
                    resp = _trader.buy(code, price=price, amount=amount)
                else:
                    resp = _trader.market_buy(code, amount=amount)
            else:
                if price > 0:
                    resp = _trader.sell(code, price=price, amount=amount)
                else:
                    resp = _trader.market_sell(code, amount=amount)

            results.append({
                "code": code,
                "action": action,
                "amount": amount,
                "price": price,
                "status": "submitted",
                "response": str(resp),
            })
        except Exception as e:
            results.append({
                "code": code,
                "action": action,
                "status": "error",
                "message": str(e),
            })

    return results


def sync_from_signal(
    signal: pd.Series,
    capital: float = 1_000_000.0,
    top_n: int = 20,
    current_prices: pd.Series = None,
) -> list:
    """根据 KQ 信号自动生成并执行调仓订单。

    Parameters
    ----------
    signal : pd.Series
        最新一期截面信号 (stock -> value), 如 signal.iloc[-1]
    capital : float
        目标总资金
    top_n : int
        持仓数量
    current_prices : pd.Series
        当前价格 (stock -> price)。若为 None, 则只生成订单不执行。

    Returns
    -------
    list[dict]
        订单执行结果
    """
    # 选股
    signal_clean = signal.dropna().sort_values(ascending=False)
    target_stocks = signal_clean.head(top_n).index.tolist()

    if current_prices is None:
        logger.info(f"目标持仓: {target_stocks}")
        return [{"target_stocks": target_stocks, "status": "plan_only"}]

    # 当前持仓
    positions = get_positions()
    current_holdings = set(positions["code"].tolist()) if len(positions) > 0 else set()
    target_set = set(str(s) for s in target_stocks)

    # 等权分配
    per_stock_value = capital / top_n

    orders = []

    # 卖出不在目标中的
    for code in current_holdings - target_set:
        pos = positions[positions["code"] == code]
        if len(pos) > 0:
            amount = int(pos.iloc[0].get("available", 0))
            if amount > 0:
                orders.append({
                    "code": code,
                    "action": "sell",
                    "amount": amount,
                    "price": 0,  # 市价
                })

    # 买入新增的
    for code in target_set - current_holdings:
        code_str = str(code)
        if code_str in current_prices.index:
            price = float(current_prices[code_str])
            if price > 0:
                shares = int(per_stock_value / price / 100) * 100
                if shares >= 100:
                    orders.append({
                        "code": code_str,
                        "action": "buy",
                        "amount": shares,
                        "price": 0,  # 市价
                    })

    if not orders:
        logger.info("无需调仓")
        return []

    logger.info(f"生成 {len(orders)} 笔调仓订单")
    return execute_orders(orders)
