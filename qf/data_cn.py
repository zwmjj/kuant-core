"""A-share data loading — baostock (free, unlimited)"""
import os, pickle, warnings, time
import baostock as bs
import pandas as pd
import numpy as np
from datetime import datetime
warnings.filterwarnings('ignore')

CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(name):
    return os.path.join(CACHE_DIR, f"cn_{name}.pkl")

def _load_cache(name, max_age_days=7):
    path = _cache_path(name)
    if os.path.exists(path):
        if datetime.now().timestamp() - os.path.getmtime(path) < 86400 * max_age_days:
            print(f"  使用缓存: cn_{name}")
            with open(path, 'rb') as f:
                return pickle.load(f)
    return None

def _save_cache(name, data):
    with open(_cache_path(name), 'wb') as f:
        pickle.dump(data, f)


def _bs_login():
    lg = bs.login()
    if lg.error_code != '0':
        raise RuntimeError(f"baostock login failed: {lg.error_msg}")
    return lg


def get_csi300_codes():
    """Fetch the CSI 300 constituents"""
    cached = _load_cache("bs_cons300")
    if cached is not None:
        return cached
    print("  下载沪深300成分股...")
    _bs_login()
    rs = bs.query_hs300_stocks()
    stocks = []
    while rs.error_code == '0' and rs.next():
        stocks.append(rs.get_row_data())
    bs.logout()
    df = pd.DataFrame(stocks, columns=rs.fields)
    codes = df['code'].tolist()
    _save_cache("bs_cons300", codes)
    print(f"  {len(codes)} 只成分股")
    return codes


def get_stock_monthly_bs(codes, start='2010-01-01', end='2025-12-31'):
    """Bulk-download monthly back-adjusted data via baostock"""
    cache_key = f"bs_monthly_{start}_{end}_{len(codes)}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    print(f"  下载A股月频数据 ({len(codes)} 只)...")
    _bs_login()
    all_data = []
    errors = 0
    for i, code in enumerate(codes):
        if i % 50 == 0 and i > 0:
            print(f"    {i}/{len(codes)}...")
        try:
            rs = bs.query_history_k_data_plus(
                code, 'date,code,close,volume,amount,turn,pctChg',
                start_date=start, end_date=end, frequency='m', adjustflag='2')
            rows = []
            while rs.error_code == '0' and rs.next():
                rows.append(rs.get_row_data())
            if len(rows) < 6:
                continue
            df = pd.DataFrame(rows, columns=rs.fields)
            df['date'] = pd.to_datetime(df['date'])
            for col in ['close', 'volume', 'amount', 'turn', 'pctChg']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['ret'] = df['pctChg'] / 100
            df['code_short'] = code.split('.')[1]
            all_data.append(df)
        except Exception:
            errors += 1

    bs.logout()

    if not all_data:
        raise RuntimeError("未能下载任何A股数据")

    raw = pd.concat(all_data, ignore_index=True)
    print(f"  完成: {raw['code'].nunique()} 只, {len(raw)} 条, {errors} 错误")
    _save_cache(cache_key, raw)
    return raw


def prepare_cn_data(start='2010-01-01', end='2025-12-31'):
    """Load the full A-share dataset"""
    print("加载A股数据 (baostock)...")

    codes = get_csi300_codes()
    raw = get_stock_monthly_bs(codes, start, end)

    # Pivot
    prices = raw.pivot_table(index='date', columns='code_short', values='close')
    returns = raw.pivot_table(index='date', columns='code_short', values='ret')
    amount = raw.pivot_table(index='date', columns='code_short', values='amount')  # 成交额
    turnover = raw.pivot_table(index='date', columns='code_short', values='turn')  # 换手率

    # Market cap proxy: amount / (turnover/100)
    mktcap = amount / (turnover.replace(0, np.nan) / 100)

    # Filter
    completeness = prices.notna().mean()
    valid = completeness[completeness > 0.5].index
    prices = prices[valid]
    returns = returns[[c for c in valid if c in returns.columns]]
    mktcap = mktcap[[c for c in valid if c in mktcap.columns]]
    amount = amount[[c for c in valid if c in amount.columns]]
    turnover = turnover[[c for c in valid if c in turnover.columns]]

    # CSI300 index benchmark
    try:
        import akshare as ak
        idx = ak.stock_zh_index_daily(symbol='sh000300')
        idx['date'] = pd.to_datetime(idx['date'])
        idx = idx.set_index('date')
        csi300_ret = idx['close'].resample('ME').last().pct_change().dropna()
    except Exception:
        csi300_ret = returns.mean(axis=1)  # fallback

    print(f"  A股: {prices.shape[1]} 只, {prices.shape[0]} 月 "
          f"({prices.index[0].strftime('%Y-%m')} ~ {prices.index[-1].strftime('%Y-%m')})")

    return {
        'prices': prices, 'returns': returns, 'mktcap': mktcap,
        'volume': amount, 'turnover': turnover,
        'market': 'CN', 'index_ret': csi300_ret,
        'rf': pd.Series(0.002, index=returns.index),
        'valid_stocks': valid,
    }
