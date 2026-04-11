"""AKShare 增强数据适配器 — 免费开源中国金融数据
AKShare: https://github.com/akfamily/akshare
覆盖股票、基金/ETF、宏观指标、财务报表、指数成分等。
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

try:
    import akshare as ak

    _HAS_AKSHARE = True
except ImportError:
    _HAS_AKSHARE = False

# ---------------------------------------------------------------------------
# Cache (与 data_cn.py 保持一致)
# ---------------------------------------------------------------------------
CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"ak_{name}.pkl")


def _load_cache(name: str, max_age_days: int = 7):
    path = _cache_path(name)
    if os.path.exists(path):
        age = datetime.now().timestamp() - os.path.getmtime(path)
        if age < 86400 * max_age_days:
            with open(path, "rb") as f:
                return pickle.load(f)
    return None


def _save_cache(name: str, data):
    with open(_cache_path(name), "wb") as f:
        pickle.dump(data, f)


def _check_akshare():
    if not _HAS_AKSHARE:
        raise ImportError("akshare 未安装。请运行: pip install akshare")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_stock_hist(
    symbol: str,
    start: str = "20100101",
    end: str = "20251231",
    period: str = "daily",
    adjust: str = "hfq",
) -> pd.DataFrame:
    """获取个股历史行情 (后复权)。

    Parameters
    ----------
    symbol : str
        股票代码, 如 '000001'
    start, end : str
        日期范围, 格式 YYYYMMDD
    period : str
        'daily', 'weekly', 'monthly'
    adjust : str
        'hfq' 后复权 | 'qfq' 前复权 | '' 不复权

    Returns
    -------
    pd.DataFrame
        含 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 换手率
    """
    _check_akshare()
    cache_key = f"hist_{symbol}_{start}_{end}_{period}_{adjust}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period=period,
        start_date=start,
        end_date=end,
        adjust=adjust,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # 标准化列名
    rename_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_change",
        "涨跌额": "change",
        "换手率": "turnover",
    }
    df = df.rename(columns=rename_map)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    # 计算收益率
    if "close" in df.columns:
        df["ret"] = df["close"].pct_change()

    _save_cache(cache_key, df)
    return df


def get_fund_etf_hist(
    symbol: str,
    start: str = "20100101",
    end: str = "20251231",
    period: str = "daily",
    adjust: str = "hfq",
) -> pd.DataFrame:
    """获取场内ETF基金历史行情。

    Parameters
    ----------
    symbol : str
        ETF代码, 如 '510300' (沪深300ETF)
    start, end : str
        日期范围
    period : str
        'daily', 'weekly', 'monthly'
    adjust : str
        复权类型

    Returns
    -------
    pd.DataFrame
    """
    _check_akshare()
    cache_key = f"etf_{symbol}_{start}_{end}_{period}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    df = ak.fund_etf_hist_em(
        symbol=symbol,
        period=period,
        start_date=start,
        end_date=end,
        adjust=adjust,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    rename_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover",
    }
    df = df.rename(columns=rename_map)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    if "close" in df.columns:
        df["ret"] = df["close"].pct_change()

    _save_cache(cache_key, df)
    return df


def get_macro_data(indicator: str) -> pd.DataFrame:
    """获取宏观经济指标。

    Parameters
    ----------
    indicator : str
        指标名称:
        - 'gdp'       : GDP 季度数据
        - 'cpi'       : CPI 月度数据
        - 'pmi'       : PMI 月度数据
        - 'm2'        : M2 货币供应量
        - 'ppi'       : PPI 月度数据
        - 'shibor'    : Shibor 利率
        - 'lpr'       : LPR 利率

    Returns
    -------
    pd.DataFrame
    """
    _check_akshare()
    cache_key = f"macro_{indicator}"
    cached = _load_cache(cache_key, max_age_days=30)
    if cached is not None:
        return cached

    dispatch = {
        "gdp": lambda: ak.macro_china_gdp(),
        "cpi": lambda: ak.macro_china_cpi_monthly(),
        "pmi": lambda: ak.macro_china_pmi(),
        "m2": lambda: ak.macro_china_money_supply(),
        "ppi": lambda: ak.macro_china_ppi(),
        "shibor": lambda: ak.rate_interbank(market="上海银行间同业拆放利率(Shibor)", indicator="3月", need_page=""),
        "lpr": lambda: ak.macro_china_lpr(),
    }

    if indicator not in dispatch:
        raise ValueError(
            f"未知宏观指标: {indicator}。支持: {list(dispatch.keys())}"
        )

    try:
        df = dispatch[indicator]()
    except Exception as e:
        warnings.warn(f"获取宏观数据 '{indicator}' 失败: {e}")
        return pd.DataFrame()

    if df is not None and not df.empty:
        _save_cache(cache_key, df)
    return df if df is not None else pd.DataFrame()


def get_financial_statements(
    symbol: str,
    period: str = "yearly",
) -> dict:
    """获取财务报表数据 (利润表、资产负债表、现金流量表)。

    Parameters
    ----------
    symbol : str
        股票代码, 如 '000001'
    period : str
        'yearly' 年度 | 'quarterly' 季度

    Returns
    -------
    dict
        {'income': DataFrame, 'balance': DataFrame, 'cashflow': DataFrame}
    """
    _check_akshare()
    cache_key = f"fin_{symbol}_{period}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    result = {}

    # 利润表
    try:
        inc = ak.stock_financial_report_sina(stock=symbol, symbol="利润表")
        result["income"] = inc if inc is not None else pd.DataFrame()
    except Exception:
        result["income"] = pd.DataFrame()

    # 资产负债表
    try:
        bal = ak.stock_financial_report_sina(stock=symbol, symbol="资产负债表")
        result["balance"] = bal if bal is not None else pd.DataFrame()
    except Exception:
        result["balance"] = pd.DataFrame()

    # 现金流量表
    try:
        cf = ak.stock_financial_report_sina(stock=symbol, symbol="现金流量表")
        result["cashflow"] = cf if cf is not None else pd.DataFrame()
    except Exception:
        result["cashflow"] = pd.DataFrame()

    _save_cache(cache_key, result)
    return result


def get_index_components(index_name: str = "沪深300") -> list:
    """获取指数成分股列表。

    Parameters
    ----------
    index_name : str
        指数名称:
        - '沪深300' / 'csi300'
        - '中证500' / 'csi500'
        - '上证50'  / 'sse50'
        - '创业板指' / 'chinext'
        - '科创50'  / 'star50'

    Returns
    -------
    list[str]
        成分股代码列表
    """
    _check_akshare()
    name_map = {
        "csi300": "沪深300",
        "csi500": "中证500",
        "sse50": "上证50",
        "chinext": "创业板指",
        "star50": "科创50",
    }
    index_name = name_map.get(index_name, index_name)

    cache_key = f"comp_{index_name}"
    cached = _load_cache(cache_key, max_age_days=30)
    if cached is not None:
        return cached

    dispatch = {
        "沪深300": lambda: ak.index_stock_cons(symbol="000300"),
        "中证500": lambda: ak.index_stock_cons(symbol="000905"),
        "上证50": lambda: ak.index_stock_cons(symbol="000016"),
        "创业板指": lambda: ak.index_stock_cons(symbol="399006"),
        "科创50": lambda: ak.index_stock_cons(symbol="000688"),
    }

    if index_name not in dispatch:
        raise ValueError(
            f"未知指数: {index_name}。支持: {list(dispatch.keys())}"
        )

    try:
        df = dispatch[index_name]()
        codes = df["品种代码"].tolist() if "品种代码" in df.columns else []
    except Exception as e:
        warnings.warn(f"获取指数成分 '{index_name}' 失败: {e}")
        codes = []

    _save_cache(cache_key, codes)
    return codes
