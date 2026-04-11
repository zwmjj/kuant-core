"""Tushare Pro 数据适配器 — A股数据源
Tushare Pro: https://github.com/waditu/tushare
提供A股日频/财务/指数数据，需要注册获取 token。
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

try:
    import tushare as ts

    _HAS_TUSHARE = True
except ImportError:
    _HAS_TUSHARE = False

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

_pro = None


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"ts_{name}.pkl")


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def init_tushare(token: str):
    """初始化 Tushare Pro API。

    Parameters
    ----------
    token : str
        Tushare Pro 注册 token (https://tushare.pro/register)
    """
    global _pro
    if not _HAS_TUSHARE:
        raise ImportError(
            "tushare 未安装。请运行: pip install tushare"
        )
    ts.set_token(token)
    _pro = ts.pro_api()
    return _pro


def _ensure_pro():
    if _pro is None:
        raise RuntimeError(
            "请先调用 init_tushare(token) 初始化 Tushare Pro"
        )
    return _pro


def get_stock_daily(
    ts_code: str,
    start: str = "20100101",
    end: str = "20251231",
) -> pd.DataFrame:
    """获取个股日频行情（前复权）。

    Parameters
    ----------
    ts_code : str
        Tushare 股票代码, 如 '000001.SZ'
    start, end : str
        日期范围, 格式 YYYYMMDD

    Returns
    -------
    pd.DataFrame
        含 trade_date, open, high, low, close, vol, amount 等列
    """
    pro = _ensure_pro()
    cache_key = f"daily_{ts_code}_{start}_{end}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    df = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
    if df is None or df.empty:
        return pd.DataFrame()

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)

    # 计算收益率
    df["ret"] = df["close"].pct_change()

    _save_cache(cache_key, df)
    return df


def get_financials(
    ts_code: str,
    period: str = "",
) -> pd.DataFrame:
    """获取财务指标数据（利润表/资产负债表关键指标）。

    Parameters
    ----------
    ts_code : str
        Tushare 股票代码
    period : str
        报告期, 如 '20231231'；留空则获取最近所有报告期

    Returns
    -------
    pd.DataFrame
        含 eps, roe, roa, grossprofit_margin 等指标
    """
    pro = _ensure_pro()
    cache_key = f"fin_{ts_code}_{period}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    kwargs = {"ts_code": ts_code}
    if period:
        kwargs["period"] = period

    df = pro.fina_indicator(**kwargs)
    if df is None or df.empty:
        return pd.DataFrame()

    df["end_date"] = pd.to_datetime(df["end_date"])
    df = df.sort_values("end_date").reset_index(drop=True)
    _save_cache(cache_key, df)
    return df


def get_index_daily(
    index_code: str = "000300.SH",
    start: str = "20100101",
    end: str = "20251231",
) -> pd.DataFrame:
    """获取指数日频行情。

    Parameters
    ----------
    index_code : str
        指数代码, 如 '000300.SH' (沪深300), '000905.SH' (中证500)
    start, end : str
        日期范围, 格式 YYYYMMDD

    Returns
    -------
    pd.DataFrame
    """
    pro = _ensure_pro()
    cache_key = f"idx_{index_code}_{start}_{end}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    df = pro.index_daily(ts_code=index_code, start_date=start, end_date=end)
    if df is None or df.empty:
        return pd.DataFrame()

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["ret"] = df["close"].pct_change()
    _save_cache(cache_key, df)
    return df


def get_adj_factor(
    ts_code: str,
    start: str = "20100101",
    end: str = "20251231",
) -> pd.DataFrame:
    """获取复权因子。

    Parameters
    ----------
    ts_code : str
        Tushare 股票代码
    start, end : str
        日期范围

    Returns
    -------
    pd.DataFrame
        含 trade_date, adj_factor 列
    """
    pro = _ensure_pro()
    cache_key = f"adj_{ts_code}_{start}_{end}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    df = pro.adj_factor(ts_code=ts_code, start_date=start, end_date=end)
    if df is None or df.empty:
        return pd.DataFrame()

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    _save_cache(cache_key, df)
    return df
