"""数据加载 — WRDS + 缓存 + 退市处理"""
import os, pickle, warnings
import wrds
import pandas as pd
import numpy as np
from datetime import datetime
warnings.filterwarnings('ignore')


class DataLoader:
    def __init__(self):
        self.cache_dir = "data_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        user = os.environ.get('WRDS_USERNAME', '')
        pwd = os.environ.get('WRDS_PASSWORD', '')
        if not user or not pwd:
            print("  WRDS凭证未设置，使用纯缓存模式")
            self.db = None
            return
        try:
            print("连接WRDS...")
            self.db = wrds.Connection(wrds_username=user, wrds_password=pwd)
            print("连接成功")
        except Exception as e:
            print(f"  WRDS连接失败({e})，使用纯缓存模式")
            self.db = None

    def _cache_path(self, name):
        return os.path.join(self.cache_dir, f"{name}.pkl")

    def _load_cache(self, name):
        path = self._cache_path(name)
        if os.path.exists(path):
            if datetime.now().timestamp() - os.path.getmtime(path) < 86400 * 7:
                print(f"  使用缓存: {name}")
                with open(path, 'rb') as f:
                    return pickle.load(f)
        return None

    def _save_cache(self, name, data):
        with open(self._cache_path(name), 'wb') as f:
            pickle.dump(data, f)

    def get_sp500_prices(self, start='2010-01-01', end='2024-12-31'):
        cache_name = f"prices_{start}_{end}"
        cached = self._load_cache(cache_name)
        if cached is not None:
            return cached
        if self.db is None:
            raise RuntimeError(f"无缓存且无WRDS连接: {cache_name}")
        print("下载价格数据...")
        mid = '2017-12-31'
        sql_tpl = """
            SELECT a.permno, a.date, a.prc, a.ret, a.shrout, a.cfacpr, a.vol,
                   b.shrcd, b.exchcd
            FROM crsp_a_stock.msf a
            JOIN crsp_a_stock.msenames b
              ON a.permno = b.permno
             AND a.date BETWEEN b.namedt AND COALESCE(b.nameendt, '2099-12-31')
            WHERE a.date BETWEEN '{s}' AND '{e}'
            AND b.shrcd IN (10, 11) AND b.exchcd IN (1, 2, 3)
            AND a.prc IS NOT NULL AND a.shrout > 0
        """
        p1 = self.db.raw_sql(sql_tpl.format(s=start, e=mid))
        p2 = self.db.raw_sql(sql_tpl.format(s='2018-01-01', e=end))
        raw = pd.concat([p1, p2], ignore_index=True).drop_duplicates(subset=['permno', 'date'])
        raw['date'] = pd.to_datetime(raw['date'])
        raw['prc'] = raw['prc'].abs()
        raw['adj_price'] = raw['prc'] / raw['cfacpr'].replace(0, 1)
        price_wide = raw.pivot_table(index='date', columns='permno', values='adj_price')
        ret_wide = raw.pivot_table(index='date', columns='permno', values='ret')
        vol_wide = raw.pivot_table(index='date', columns='permno', values='vol')
        result = {'prices': price_wide, 'returns': ret_wide, 'raw': raw, 'volume': vol_wide}
        self._save_cache(cache_name, result)
        print(f"  完成: {len(price_wide.columns)} 只股票")
        return result

    def get_delisting_returns(self, start='2010-01-01', end='2024-12-31'):
        """下载退市收益 — 修复幸存者偏差 (Shumway 1997)"""
        cache_name = f"delist_{start}_{end}"
        cached = self._load_cache(cache_name)
        if cached is not None:
            return cached
        if self.db is None:
            print("  无WRDS连接，跳过退市数据")
            return pd.DataFrame(columns=['permno', 'dlstdt', 'dlstcd', 'dlret'])
        print("下载退市数据...")
        df = self.db.raw_sql("""
            SELECT permno, dlstdt, dlstcd, dlret
            FROM crsp_a_stock.msedelist
            WHERE dlstdt BETWEEN '{s}' AND '{e}'
        """.format(s=start, e=end))
        df['dlstdt'] = pd.to_datetime(df['dlstdt'])
        df['permno'] = df['permno'].astype(float)
        # Shumway (1997): performance-related delistings with missing returns
        perf_mask = df['dlstcd'].between(500, 599)
        df.loc[perf_mask & df['dlret'].isna(), 'dlret'] = -0.30
        # Mergers/exchanges: assume 0 if missing
        non_perf = ~perf_mask
        df.loc[non_perf & df['dlret'].isna(), 'dlret'] = 0.0
        self._save_cache(cache_name, df)
        print(f"  退市记录: {len(df)} 条")
        return df

    def get_ff5_factors(self, start='2010-01-01', end='2024-12-31'):
        cache_name = f"ff5_{start}_{end}"
        cached = self._load_cache(cache_name)
        if cached is not None:
            return cached
        if self.db is None:
            raise RuntimeError(f"无缓存且无WRDS连接: {cache_name}")
        print("下载FF5因子...")
        ff5 = self.db.raw_sql("""
            SELECT date, mktrf, smb, hml, umd, rf
            FROM ff.factors_monthly WHERE date BETWEEN '{s}' AND '{e}'
        """.format(s=start, e=end))
        ff5['date'] = pd.to_datetime(ff5['date'])
        ff5 = ff5.set_index('date')
        self._save_cache(cache_name, ff5)
        return ff5

    def get_rf_rate(self, start='2010-01-01', end='2024-12-31'):
        # Try to get from ff5 cache first
        ff5 = self.get_ff5_factors(start, end)
        if 'rf' in ff5.columns:
            return ff5['rf']
        if self.db is None:
            raise RuntimeError("无缓存且无WRDS连接: rf_rate")
        rf = self.db.raw_sql("""
            SELECT date, rf FROM ff.factors_monthly
            WHERE date BETWEEN '{s}' AND '{e}'
        """.format(s=start, e=end))
        rf['date'] = pd.to_datetime(rf['date'])
        return rf.set_index('date')['rf']

    def get_ccm_fundamentals(self, start='2010-01-01', end='2024-12-31'):
        cache_name = f"ccm_fund_{start}_{end}"
        cached = self._load_cache(cache_name)
        if cached is not None:
            return cached
        if self.db is None:
            raise RuntimeError(f"无缓存且无WRDS连接: {cache_name}")
        print("下载CCM基本面...")
        df = self.db.raw_sql("""
            SELECT b.lpermno AS permno, a.gvkey, a.datadate,
                   a.ni, a.ceq, a.at, a.sale, a.revt, a.gp,
                   a.prcc_f, a.csho, a.dltt, a.dlc
            FROM comp.funda a
            JOIN crsp.ccmxpf_lnkhist b ON a.gvkey = b.gvkey
             AND b.linktype IN ('LC','LU') AND b.linkprim IN ('P','C')
             AND a.datadate BETWEEN b.linkdt AND COALESCE(b.linkenddt,'2099-12-31')
            WHERE a.datadate BETWEEN '{s}' AND '{e}'
            AND a.indfmt='INDL' AND a.datafmt='STD' AND a.popsrc='D' AND a.consol='C'
            AND a.at > 0 AND a.ceq > 0
        """.format(s=start, e=end))
        df['datadate'] = pd.to_datetime(df['datadate'])
        df['permno'] = df['permno'].astype(float)
        df['roe'] = df['ni'] / df['ceq']
        df['roa'] = df['ni'] / df['at']
        df['asset_growth'] = df.groupby('permno')['at'].pct_change()
        df['leverage'] = (df['dltt'].fillna(0) + df['dlc'].fillna(0)) / df['at']
        # New fields for factor strategies
        mktcap_f = df['prcc_f'].abs() * df['csho']
        df['bm'] = df['ceq'] / mktcap_f.replace(0, np.nan)         # Book-to-Market
        df['ep'] = df['ni'] / mktcap_f.replace(0, np.nan)           # Earnings-to-Price
        df['gpa'] = df['gp'] / df['at'] if 'gp' in df.columns else np.nan  # Gross Profitability
        df['sp'] = df['sale'] / mktcap_f.replace(0, np.nan)         # Sales-to-Price
        df['year'] = df['datadate'].dt.year
        df = df.sort_values('datadate').drop_duplicates(subset=['permno', 'year'], keep='last')
        self._save_cache(cache_name, df)
        return df

    def get_permno_info(self):
        cache_name = "permno_info"
        cached = self._load_cache(cache_name)
        if cached is not None:
            return cached
        if self.db is None:
            raise RuntimeError("无缓存且无WRDS连接: permno_info")
        print("下载Ticker映射...")
        info = self.db.raw_sql("""
            SELECT permno, ticker, comnam, namedt, nameendt, shrcd, exchcd
            FROM crsp_a_stock.msenames WHERE shrcd IN (10,11) AND exchcd IN (1,2,3)
        """)
        info['namedt'] = pd.to_datetime(info['namedt'])
        info = info.sort_values('namedt').drop_duplicates(subset='permno', keep='last')
        info = info.set_index('permno')[['ticker', 'comnam', 'exchcd']]
        info['exchange'] = info['exchcd'].map({1: 'NYSE', 2: 'AMEX', 3: 'NASDAQ'})
        self._save_cache(cache_name, info)
        return info


def _apply_delisting_returns(returns, delist_df):
    """将退市收益合并到月度收益矩阵中"""
    if delist_df is None or len(delist_df) == 0:
        return returns
    applied = 0
    for _, row in delist_df.iterrows():
        p = row['permno']
        if p not in returns.columns:
            continue
        dl_ret = row['dlret']
        if pd.isna(dl_ret) or dl_ret == 0:
            continue
        # Match delisting date to the closest month-end in returns index
        dl_period = row['dlstdt'].to_period('M')
        matches = [d for d in returns.index if d.to_period('M') == dl_period]
        if matches:
            idx = matches[0]
            existing = returns.loc[idx, p]
            if pd.isna(existing):
                returns.loc[idx, p] = dl_ret
            else:
                returns.loc[idx, p] = (1 + existing) * (1 + dl_ret) - 1
            applied += 1
    if applied > 0:
        print(f"  应用退市收益: {applied} 条")
    return returns


def _extend_with_yfinance(prices, returns, mktcap, permno_info):
    """用yfinance补充WRDS之后的最新数据"""
    import yfinance as yf

    last_wrds = prices.index[-1]
    today = pd.Timestamp(datetime.now().strftime('%Y-%m-%d'))

    if (today - last_wrds).days < 35:
        return prices, returns, mktcap

    print(f"  yfinance补充: {str(last_wrds)[:10]} → {str(today)[:10]}")

    active = prices.iloc[-12:].notna().sum()
    top_permnos = active.nlargest(500).index
    ticker_map = {}
    for p in top_permnos:
        if p in permno_info.index:
            t = permno_info.loc[p, 'ticker']
            if isinstance(t, str) and len(t) <= 5:
                ticker_map[p] = t

    if len(ticker_map) < 50:
        print(f"  映射不足({len(ticker_map)}), 跳过补充")
        return prices, returns, mktcap

    tickers = list(ticker_map.values())
    inv_map = {v: k for k, v in ticker_map.items()}

    start_dl = (last_wrds - pd.DateOffset(months=1)).strftime('%Y-%m-%d')
    try:
        import logging
        # Suppress yfinance "N Failed download" warnings for delisted/invalid tickers
        yf_logger = logging.getLogger('yfinance')
        prev_level = yf_logger.level
        yf_logger.setLevel(logging.CRITICAL)
        try:
            yf_data = yf.download(tickers, start=start_dl, auto_adjust=True, progress=False)
        finally:
            yf_logger.setLevel(prev_level)
        if yf_data.empty:
            return prices, returns, mktcap
    except Exception as e:
        print(f"  yfinance下载失败: {e}")
        return prices, returns, mktcap

    if 'Close' in yf_data.columns.get_level_values(0):
        close = yf_data['Close']
    else:
        close = yf_data

    monthly_price = close.resample('ME').last()
    monthly_ret = monthly_price.pct_change()

    # Align to CRSP-style month-end dates (last business day)
    new_months = monthly_price.index[monthly_price.index > last_wrds]
    if len(new_months) == 0:
        print("  无新月份数据")
        return prices, returns, mktcap

    print(f"  补充 {len(new_months)} 个月, {len(ticker_map)} 只股票")

    for month in new_months:
        # Snap to last business day of the month for CRSP alignment
        bday_month_end = pd.Timestamp(month) - pd.tseries.offsets.BDay(0)
        if bday_month_end > month:
            bday_month_end = month - pd.tseries.offsets.BDay(1)
        use_date = bday_month_end

        new_price_row = {}
        new_ret_row = {}
        new_cap_row = {}
        for ticker in monthly_price.columns:
            if ticker not in inv_map:
                continue
            permno = inv_map[ticker]
            p = monthly_price.loc[month, ticker]
            r = monthly_ret.loc[month, ticker] if month in monthly_ret.index else np.nan
            if pd.notna(p):
                new_price_row[permno] = float(p)
                new_cap_row[permno] = float(p) * 1000
            if pd.notna(r):
                new_ret_row[permno] = float(r)

        if new_price_row:
            prices.loc[use_date] = pd.Series(new_price_row)
            returns.loc[use_date] = pd.Series(new_ret_row)
            mktcap.loc[use_date] = pd.Series(new_cap_row)

    prices = prices.sort_index()
    returns = returns.sort_index()
    mktcap = mktcap.sort_index()
    return prices, returns, mktcap


def _ensure_derived_fields(ccm_fund):
    """确保CCM数据包含所有衍生字段（兼容旧缓存）"""
    df = ccm_fund
    if 'roe' not in df.columns and 'ni' in df.columns and 'ceq' in df.columns:
        df['roe'] = df['ni'] / df['ceq']
    if 'roa' not in df.columns and 'ni' in df.columns and 'at' in df.columns:
        df['roa'] = df['ni'] / df['at']
    if 'asset_growth' not in df.columns and 'at' in df.columns:
        df['asset_growth'] = df.groupby('permno')['at'].pct_change()
    if 'leverage' not in df.columns and 'at' in df.columns:
        df['leverage'] = (df['dltt'].fillna(0) + df['dlc'].fillna(0)) / df['at']
    if 'bm' not in df.columns and 'ceq' in df.columns and 'prcc_f' in df.columns:
        mktcap_f = df['prcc_f'].abs() * df['csho']
        df['bm'] = df['ceq'] / mktcap_f.replace(0, np.nan)
    if 'ep' not in df.columns and 'ni' in df.columns and 'prcc_f' in df.columns:
        mktcap_f = df['prcc_f'].abs() * df['csho']
        df['ep'] = df['ni'] / mktcap_f.replace(0, np.nan)
    if 'gpa' not in df.columns:
        if 'gp' in df.columns and 'at' in df.columns:
            df['gpa'] = df['gp'] / df['at']
        elif 'revt' in df.columns and 'cogs' in df.columns and 'at' in df.columns:
            # Correct fallback: (revenue - COGS) / assets = gross profitability (Novy-Marx 2013)
            df['gpa'] = (df['revt'] - df['cogs']) / df['at'].replace(0, np.nan)
        elif 'revt' in df.columns and 'at' in df.columns:
            import warnings
            warnings.warn("gpa: cogs unavailable, using revt/at (asset turnover, NOT gross profitability)")
            df['gpa'] = df['revt'] / df['at']
    if 'sp' not in df.columns and 'sale' in df.columns and 'prcc_f' in df.columns:
        mktcap_f = df['prcc_f'].abs() * df['csho']
        df['sp'] = df['sale'] / mktcap_f.replace(0, np.nan)
    return df


def prepare_data(start='2000-01-01', end='2025-12-31'):
    """加载全套数据，返回标准数据字典（含退市处理）"""
    loader = DataLoader()
    data = loader.get_sp500_prices(start, end)

    # 退市数据（可能失败如果无WRDS连接）
    delist = None
    try:
        delist = loader.get_delisting_returns(start, end)
    except Exception as e:
        print(f"  退市数据加载跳过: {e}")

    ff5 = loader.get_ff5_factors(start, end)
    rf = loader.get_rf_rate(start, end)
    ccm_fund = loader.get_ccm_fundamentals(start, end)
    permno_info = loader.get_permno_info()

    # 确保衍生字段存在（兼容旧缓存）
    ccm_fund = _ensure_derived_fields(ccm_fund)

    prices = data['prices']
    returns = data['returns']
    volume = data.get('volume', None)
    adv_dollar = data.get('adv_dollar', None)

    # Apply delisting returns before any filtering
    if delist is not None and len(delist) > 0:
        returns = _apply_delisting_returns(returns, delist)

    # Filter: average price > $5, data completeness > 60%
    avg_price = prices.mean()
    completeness = prices.notna().mean()
    valid = avg_price[(avg_price > 5) & (completeness > 0.6)].index
    prices = prices[valid]
    returns = returns[[c for c in valid if c in returns.columns]]
    if volume is not None:
        volume = volume[[c for c in valid if c in volume.columns]]
    if adv_dollar is not None:
        adv_dollar = adv_dollar[[c for c in valid if c in adv_dollar.columns]]

    raw = data['raw']
    raw_valid = raw[raw['permno'].isin(valid)]
    mktcap = raw_valid.pivot_table(index='date', columns='permno', values='prc', aggfunc='first') * \
             raw_valid.pivot_table(index='date', columns='permno', values='shrout', aggfunc='first')

    # yfinance补充最新数据
    try:
        prices, returns, mktcap = _extend_with_yfinance(prices, returns, mktcap, permno_info)
    except Exception as e:
        print(f"  yfinance补充跳过: {e}")

    try:
        import yfinance as yf
        import logging
        yf_logger = logging.getLogger('yfinance')
        prev_level = yf_logger.level
        yf_logger.setLevel(logging.CRITICAL)
        try:
            spy_raw = yf.download('SPY', start=start, auto_adjust=True, progress=False)
        finally:
            yf_logger.setLevel(prev_level)
        spy_close = spy_raw['Close']
        if isinstance(spy_close, pd.DataFrame):
            spy_close = spy_close.iloc[:, 0]
        spy_ret = spy_close.resample('ME').last().pct_change().dropna().squeeze()
    except Exception:
        spy_ret = (ff5['mktrf'] + ff5['rf']).squeeze()

    return {
        'prices': prices, 'returns': returns, 'mktcap': mktcap,
        'ff5': ff5, 'rf': rf, 'ccm_fund': ccm_fund,
        'spy_ret': spy_ret, 'valid_stocks': valid, 'permno_info': permno_info,
        'volume': volume, 'adv_dollar': adv_dollar, 'delist': delist,
    }
