"""Signal computation toolkit — including factor strategies"""
import numpy as np
import pandas as pd


class SignalGenerator:
    """Collection of static methods for all factor signals"""

    # ── 动量类 ──

    @staticmethod
    def momentum_12_1(returns, lookback=12, skip=1):
        return returns.shift(skip).rolling(lookback).apply(
            lambda x: (1 + x).prod() - 1, raw=True)

    @staticmethod
    def multi_timeframe_momentum(returns):
        mom12 = returns.shift(1).rolling(12).apply(lambda x: (1 + x).prod() - 1, raw=True)
        mom6  = returns.shift(1).rolling(6).apply(lambda x: (1 + x).prod() - 1, raw=True)
        mom3  = returns.shift(1).rolling(3).apply(lambda x: (1 + x).prod() - 1, raw=True)
        vol = returns.shift(1).rolling(12).std().replace(0, np.nan)
        return (0.25 * mom12 / vol + 0.35 * mom6 / vol + 0.40 * mom3 / vol)

    @staticmethod
    def momentum_acceleration(returns):
        mom3_recent = returns.shift(1).rolling(3).apply(lambda x: (1 + x).prod() - 1, raw=True)
        mom3_older  = returns.shift(4).rolling(3).apply(lambda x: (1 + x).prod() - 1, raw=True)
        return mom3_recent - mom3_older

    @staticmethod
    def high_52_week(prices):
        """Proximity to the 52-week (12-month) high — George & Hwang (2004)"""
        high12 = prices.shift(1).rolling(12).max()
        current = prices.shift(1)
        ratio = current / high12.replace(0, np.nan)
        return ratio

    # ── 波动率/风险类 ──

    @staticmethod
    def volatility(returns, window=12):
        """Fixed: shift(1) avoids including the current month's data"""
        return returns.shift(1).rolling(window).std() * np.sqrt(12)

    @staticmethod
    def vol_of_vol(returns, vol_window=12, vov_window=12):
        """Vol-of-Vol — the volatility of volatility; less stable is worse"""
        vol = returns.shift(1).rolling(vol_window).std()
        return vol.rolling(vov_window).std()

    @staticmethod
    def downside_vol(returns, window=12):
        """Downside Volatility — downside-only volatility (semi-deviation)"""
        neg = returns.shift(1).clip(upper=0)
        return neg.rolling(window).std() * np.sqrt(12)

    @staticmethod
    def max_return(returns, window=12):
        """MAX — Bali, Cakici, Whitelaw (2011)
        Largest single-month return over the past 12 months. High MAX predicts low returns (lottery effect)"""
        return returns.shift(1).rolling(window).max()

    @staticmethod
    def beta_market(returns, spy_ret, window=36):
        """Market Beta — Frazzini & Pedersen (2014) BAB
        Low beta predicts higher risk-adjusted returns"""
        spy = spy_ret.copy()
        spy.index = spy.index.to_period('M')
        rets = returns.copy()
        rets.index = rets.index.to_period('M')
        common = rets.index.intersection(spy.index)
        rets = rets.loc[common]
        spy = spy.loc[common]
        beta_df = pd.DataFrame(np.nan, index=rets.index, columns=rets.columns)
        for i in range(window, len(common)):
            spy_win = spy.iloc[i - window:i].values.astype(float)
            spy_var = np.var(spy_win)
            if spy_var < 1e-10:
                continue
            for col in rets.columns:
                y = rets[col].iloc[i - window:i].values.astype(float)
                valid = ~np.isnan(y)
                if valid.sum() < window // 2:
                    continue
                cov = np.cov(y[valid], spy_win[valid])[0, 1]
                beta_df.iloc[i, beta_df.columns.get_loc(col)] = cov / spy_var
        # Map PeriodIndex back to original DatetimeIndex safely
        period_to_date = {d.to_period('M'): d for d in returns.index}
        beta_df.index = [period_to_date[p] for p in beta_df.index]
        return beta_df

    @staticmethod
    def skewness(returns, window=12):
        """Rolling Skewness — high skewness (lottery-like) predicts low returns"""
        return returns.shift(1).rolling(window).skew()

    @staticmethod
    def vol_term_structure(returns, short_window=3, long_window=12):
        """Vol Term Structure — short-term vol / long-term vol
        >1 means recent volatility is rising (risk increasing), <1 means it is calming down"""
        short_vol = returns.shift(1).rolling(short_window).std()
        long_vol = returns.shift(1).rolling(long_window).std()
        return short_vol / long_vol.replace(0, np.nan)

    @staticmethod
    def vol_momentum(returns, window=12):
        """Volatility Momentum — change in volatility
        Stocks with falling volatility are expected to do better (declining-risk trend)"""
        vol = returns.shift(1).rolling(window).std()
        vol_chg = vol.pct_change(3)  # 3-month change in vol
        return vol_chg

    @staticmethod
    def idiosyncratic_vol(returns, ff5, window=36):
        """Idiosyncratic volatility — Ang et al. (2006)
        Standard deviation of rolling FF3 regression residuals; low IVOL predicts higher returns
        """
        rets = returns.copy()
        ff = ff5[['mktrf', 'smb', 'hml']].copy()
        rf = ff5['rf'].copy()
        # Align to period
        rets_p = rets.copy()
        rets_p.index = rets_p.index.to_period('M')
        ff.index = ff.index.to_period('M')
        rf.index = rf.index.to_period('M')
        common = rets_p.index.intersection(ff.index)
        rets_p = rets_p.loc[common]
        ff = ff.loc[common]
        rf = rf.loc[common]

        ivol = pd.DataFrame(np.nan, index=rets_p.index, columns=rets_p.columns)
        if len(common) < window + 6:
            return ivol

        X = np.column_stack([np.ones(len(ff)), ff.values.astype(float)])
        for i in range(window, len(common)):
            X_win = X[i - window:i]
            for col in rets_p.columns:
                y = (rets_p[col].iloc[i - window:i] - rf.iloc[i - window:i]).values.astype(float)
                valid = ~np.isnan(y)
                if valid.sum() < window // 2:
                    continue
                try:
                    beta = np.linalg.lstsq(X_win[valid], y[valid], rcond=None)[0]
                    resid = y[valid] - X_win[valid] @ beta
                    ivol.iloc[i, ivol.columns.get_loc(col)] = resid.std() * np.sqrt(12)
                except Exception:
                    pass
        # Map PeriodIndex back to original DatetimeIndex safely
        period_to_date = {d.to_period('M'): d for d in rets.index}
        ivol.index = [period_to_date[p] for p in ivol.index]
        return ivol

    # ── 价值类 ──

    @staticmethod
    def _map_annual_to_monthly(ccm_fund, returns, field, ascending=True):
        """将年度基本面数据映射到月度信号网格（使用year-1数据避免前视）"""
        fund = ccm_fund.copy()
        if field not in fund.columns:
            return pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

        year_signals = {}
        for year in sorted(fund['year'].unique()):
            yr_data = fund[fund['year'] == year].drop_duplicates('permno').set_index('permno')
            vals = yr_data[field].dropna()
            # Winsorize at 1%/99% to handle outliers
            if len(vals) > 20:
                lo, hi = vals.quantile(0.01), vals.quantile(0.99)
                vals = vals.clip(lo, hi)
            if len(vals) < 10:
                continue
            ranked = vals.rank(pct=True)
            if not ascending:
                ranked = 1 - ranked
            ranked = ranked * 2 - 1  # scale to [-1, 1]
            year_signals[year] = ranked

        if not year_signals:
            return pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

        result = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
        for date in returns.index:
            # Fama-French convention: fiscal year Y data available after June Y+1
            # For months Jul-Dec of year Y+1: use fiscal year Y
            # For months Jan-Jun of year Y+1: use fiscal year Y-1
            # This ensures ~6 month minimum lag from fiscal year end to signal use
            if date.month >= 7:
                yr = date.year - 1
            else:
                yr = date.year - 2
            if yr in year_signals:
                sig = year_signals[yr]
                common = result.columns.intersection(sig.index)
                if len(common) > 0:
                    result.loc[date, common] = sig.reindex(common).values
        return result.astype(float)

    @staticmethod
    def book_to_market(ccm_fund, returns):
        """Book-to-market ratio — Fama & French (1992)"""
        return SignalGenerator._map_annual_to_monthly(
            ccm_fund, returns, 'bm', ascending=True)  # high BM = value = buy

    @staticmethod
    def earnings_to_price(ccm_fund, returns):
        """Earnings-to-price ratio — Basu (1977)"""
        return SignalGenerator._map_annual_to_monthly(
            ccm_fund, returns, 'ep', ascending=True)  # high EP = cheap = buy

    @staticmethod
    def return_on_equity(ccm_fund, returns):
        """Return on equity — Hou, Xue, Zhang (2015)"""
        return SignalGenerator._map_annual_to_monthly(
            ccm_fund, returns, 'roe', ascending=True)  # high ROE = buy

    @staticmethod
    def gross_profitability(ccm_fund, returns):
        """Gross profits-to-assets — Novy-Marx (2013)"""
        return SignalGenerator._map_annual_to_monthly(
            ccm_fund, returns, 'gpa', ascending=True)  # high GPA = buy

    @staticmethod
    def asset_growth(ccm_fund, returns):
        """Asset growth — Cooper, Gulen, Schill (2008)
        Low asset growth predicts higher returns (the investment anomaly)
        """
        return SignalGenerator._map_annual_to_monthly(
            ccm_fund, returns, 'asset_growth', ascending=False)  # low growth = buy

    # ── 基本面-新增因子 ──

    @staticmethod
    def turnover_ratio(ccm_fund, returns):
        """Asset turnover ratio — a measure of asset utilization efficiency
        A high turnover ratio indicates the firm generates revenue more efficiently from its assets
        """
        return SignalGenerator._map_annual_to_monthly(
            ccm_fund, returns, 'turnover_ratio', ascending=True)  # high turnover = buy

    @staticmethod
    def revenue_growth(ccm_fund, returns):
        """Revenue growth — Lakonishok, Shleifer, Vishny (1994)
        High revenue growth predicts higher returns (growth factor)
        """
        return SignalGenerator._map_annual_to_monthly(
            ccm_fund, returns, 'revt_growth', ascending=True)  # high growth = buy

    @staticmethod
    def debt_to_equity(ccm_fund, returns):
        """Debt-to-equity ratio — Bhandari (1988)
        Firms with lower leverage are financially sounder and perform better over the long run
        """
        return SignalGenerator._map_annual_to_monthly(
            ccm_fund, returns, 'de_ratio', ascending=False)  # low debt = buy

    @staticmethod
    def price_to_sales(ccm_fund, returns):
        """Inverse price-to-sales (Sales/Price) — Barbee et al. (1996)
        Firms with low P/S (high S/P) are undervalued and predict higher returns
        """
        return SignalGenerator._map_annual_to_monthly(
            ccm_fund, returns, 'sp', ascending=True)  # high S/P = cheap = buy

    @staticmethod
    def net_income_growth(ccm_fund, returns):
        """Net income growth — an earnings growth factor
        Sustained net income growth signals earnings quality and trend
        """
        return SignalGenerator._map_annual_to_monthly(
            ccm_fund, returns, 'ni_growth', ascending=True)  # high growth = buy

    # ── 因子模型类 ──

    @staticmethod
    def ff5_alpha(returns, ff5, window=36):
        """Rolling FF5 alpha — a momentum-alpha strategy"""
        rets = returns.copy()
        ff = ff5[['mktrf', 'smb', 'hml', 'umd']].copy()
        rf = ff5['rf'].copy()
        rets_p = rets.copy()
        rets_p.index = rets_p.index.to_period('M')
        ff.index = ff.index.to_period('M')
        rf.index = rf.index.to_period('M')
        common = rets_p.index.intersection(ff.index)
        rets_p = rets_p.loc[common]
        ff = ff.loc[common]
        rf = rf.loc[common]

        alpha_df = pd.DataFrame(np.nan, index=rets_p.index, columns=rets_p.columns)
        if len(common) < window + 6:
            return alpha_df

        X = np.column_stack([np.ones(len(ff)), ff.values.astype(float)])
        for i in range(window, len(common)):
            X_win = X[i - window:i]
            for col in rets_p.columns:
                y = (rets_p[col].iloc[i - window:i] - rf.iloc[i - window:i]).values.astype(float)
                valid = ~np.isnan(y)
                if valid.sum() < window // 2:
                    continue
                try:
                    beta = np.linalg.lstsq(X_win[valid], y[valid], rcond=None)[0]
                    alpha_df.iloc[i, alpha_df.columns.get_loc(col)] = beta[0] * 12
                except Exception:
                    pass
        period_to_date = {d.to_period('M'): d for d in rets.index}
        alpha_df.index = [period_to_date[p] for p in alpha_df.index]
        return alpha_df

    # ── 工具 ──

    @staticmethod
    def cross_sectional_rank(signal):
        def rank_row(row):
            valid = row.dropna()
            if len(valid) < 10:
                return row * np.nan
            return (valid.rank(pct=True) * 2 - 1).reindex(row.index)
        return signal.apply(rank_row, axis=1)

    @staticmethod
    def cross_sectional_zscore(signal, winsorize_sigma=3.0):
        """Cross-sectional z-score with winsorization at ±winsorize_sigma

        Unlike rank(), preserves magnitude information and handles outliers
        via clipping. Preferred for regression-based models.
        """
        def zscore_row(row):
            valid = row.dropna()
            if len(valid) < 10:
                return row * np.nan
            mu = valid.mean()
            sigma = valid.std()
            if sigma < 1e-10:
                return row * 0.0
            z = (valid - mu) / sigma
            z = z.clip(-winsorize_sigma, winsorize_sigma)
            return z.reindex(row.index)
        return signal.apply(zscore_row, axis=1)

    @staticmethod
    def crash_filter(returns):
        """Fixed: shift(1) uses the previous month's return to avoid look-ahead bias"""
        last_ret = returns.shift(1)
        pct10 = last_ret.quantile(0.10, axis=1)
        return last_ret.lt(pct10, axis=0)

    @staticmethod
    def quality_signal(ccm_fund, returns):
        """Composite quality signal (ROE + leverage + asset_growth)"""
        fund = ccm_fund.copy()
        frames = []
        for year in sorted(fund['year'].unique()):
            yr = fund[fund['year'] == year].set_index('permno')
            if len(yr) < 20:
                continue
            q = (0.5 * yr['roe'].rank(pct=True) +
                 0.25 * (1 - yr['leverage'].rank(pct=True)) +
                 0.25 * (1 - yr['asset_growth'].rank(pct=True)))
            q.name = 'quality'
            df = q.reset_index()
            df['year'] = year
            frames.append(df)
        if not frames:
            return pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
        qa = pd.concat(frames)
        qw = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
        for date in returns.index:
            yr = date.year - 1
            yq = qa[qa['year'] == yr].set_index('permno')['quality']
            common = qw.columns.intersection(yq.index)
            if len(common) > 0:
                qw.loc[date, common] = yq[common].values
        return qw.astype(float)


# ── 复合信号构建 ──

def build_signal(returns, prices, mktcap, ccm_fund,
                 w_mom=0.50, w_accel=0.20, w_quality=0.20, w_vol=0.10,
                 cap_quantile=0.75, verbose=True):
    """Build the composite momentum signal (original interface, kept for backward compatibility)"""
    sg = SignalGenerator()
    def to_float(df):
        return df.apply(pd.to_numeric, errors='coerce')

    mom_ranked   = to_float(sg.cross_sectional_rank(sg.multi_timeframe_momentum(returns)))
    accel_ranked = to_float(sg.cross_sectional_rank(sg.momentum_acceleration(returns)))
    vol_ranked   = to_float(sg.cross_sectional_rank(sg.volatility(returns, 12)))

    composite = w_mom * mom_ranked + w_accel * accel_ranked + w_vol * (-vol_ranked)
    if w_quality > 0:
        quality = to_float(sg.quality_signal(ccm_fund, returns))
        composite += w_quality * quality.fillna(0)

    pct = mktcap.quantile(cap_quantile, axis=1)
    cap_mask = mktcap.ge(pct, axis=0).reindex(index=composite.index, columns=composite.columns)
    crash = sg.crash_filter(returns).reindex(index=composite.index, columns=composite.columns).fillna(False)
    composite = composite.where(cap_mask & ~crash)

    if verbose:
        n = int(cap_mask.sum(axis=1).median())
        print(f"  信号: mom={w_mom} accel={w_accel} qual={w_quality} vol={w_vol} | 池~{n}")
    return composite


def build_factor_signal(factor_id, data, cap_quantile=0.75, verbose=True):
    """Build a single-factor signal

    Parameters
    ----------
    factor_id : str
        Factor ID: 'mom12', 'accel', 'high52', 'bm', 'ep', 'roe', 'gpa',
                 'ag', 'ivol', 'ff5alpha', 'composite'
    data : dict
        Data dictionary returned by prepare_data()
    """
    sg = SignalGenerator()
    returns = data['returns']
    prices = data['prices']
    mktcap = data['mktcap']
    ccm_fund = data['ccm_fund']

    # Generate raw signal
    if factor_id == 'mom12':
        raw = sg.multi_timeframe_momentum(returns)
    elif factor_id == 'accel':
        raw = sg.momentum_acceleration(returns)
    elif factor_id == 'high52':
        raw = sg.high_52_week(prices)
    elif factor_id == 'bm':
        raw = sg.book_to_market(ccm_fund, returns)
    elif factor_id == 'ep':
        raw = sg.earnings_to_price(ccm_fund, returns)
    elif factor_id == 'roe':
        raw = sg.return_on_equity(ccm_fund, returns)
    elif factor_id == 'gpa':
        raw = sg.gross_profitability(ccm_fund, returns)
    elif factor_id == 'ag':
        raw = sg.asset_growth(ccm_fund, returns)
    elif factor_id == 'turnover':
        raw = sg.turnover_ratio(ccm_fund, returns)
    elif factor_id == 'revgrowth':
        raw = sg.revenue_growth(ccm_fund, returns)
    elif factor_id == 'de':
        raw = sg.debt_to_equity(ccm_fund, returns)
    elif factor_id == 'ps':
        raw = sg.price_to_sales(ccm_fund, returns)
    elif factor_id == 'nigrowth':
        raw = sg.net_income_growth(ccm_fund, returns)
    elif factor_id == 'ivol':
        if 'ff5' not in data:
            raise ValueError("ivol 需要 ff5 因子数据")
        raw = sg.idiosyncratic_vol(returns, data['ff5'])
        raw = -raw  # 低IVOL更好，取负
    elif factor_id == 'lowvol':
        raw = -sg.volatility(returns, 12)  # 低vol更好
    elif factor_id == 'vov':
        raw = -sg.vol_of_vol(returns)  # 低vol-of-vol更好
    elif factor_id == 'downvol':
        raw = -sg.downside_vol(returns, 12)  # 低下行波动更好
    elif factor_id == 'maxret':
        raw = -sg.max_return(returns, 12)  # 低MAX更好 (Bali 2011)
    elif factor_id == 'beta':
        if 'spy_ret' not in data:
            raise ValueError("beta 需要 spy_ret 数据")
        raw = -sg.beta_market(returns, data['spy_ret'])  # 低beta更好 (BAB)
    elif factor_id == 'skew':
        raw = -sg.skewness(returns, 12)  # 低偏度更好
    elif factor_id == 'volts':
        raw = -sg.vol_term_structure(returns)  # 低vol term structure更好
    elif factor_id == 'volmom':
        raw = -sg.vol_momentum(returns)  # vol下降更好
    elif factor_id == 'ff5alpha':
        if 'ff5' not in data:
            raise ValueError("ff5alpha 需要 ff5 因子数据")
        raw = sg.ff5_alpha(returns, data['ff5'])
    elif factor_id == 'composite':
        return build_signal(returns, prices, mktcap, ccm_fund, verbose=verbose)
    else:
        raise ValueError(f"未知因子: {factor_id}")

    # Ensure float64 dtype (some operations produce nullable Float64)
    raw = raw.astype(float)

    # Cross-sectional rank
    ranked = sg.cross_sectional_rank(raw)

    # Apply universe filter
    pct = mktcap.quantile(cap_quantile, axis=1)
    cap_mask = mktcap.ge(pct, axis=0).reindex(index=ranked.index, columns=ranked.columns)
    crash = sg.crash_filter(returns).reindex(index=ranked.index, columns=ranked.columns).fillna(False)
    signal = ranked.where(cap_mask & ~crash)

    if verbose:
        n_valid = signal.notna().sum(axis=1).median()
        print(f"  因子 {factor_id}: 有效信号~{n_valid:.0f} 只/月")
    return signal


# ── 条件交互信号 (Round 1) ──

def build_interaction_signal(interaction_id, data, cap_quantile=0.75, verbose=True):
    """Conditional interaction signal — product of two factors; long only when both are favorable

    Parameters
    ----------
    interaction_id : str
        'quality_mom'   : GPA rank x momentum rank (high-quality momentum stocks only)
        'value_quality' : BM rank x ROE rank (profitable value stocks only)
        'profit_growth' : GPA rank x low-asset-growth rank (profitability + conservative investment)
        'roe_mom'       : ROE rank x momentum rank (profitability + trend confirmation)
        'alpha_quality' : FF5 alpha x GPA (alpha + quality double confirmation)
    """
    sg = SignalGenerator()
    returns = data['returns']
    mktcap = data['mktcap']
    ccm_fund = data['ccm_fund']

    if interaction_id == 'quality_mom':
        sig_a = sg.cross_sectional_rank(sg.gross_profitability(ccm_fund, returns))
        sig_b = sg.cross_sectional_rank(sg.multi_timeframe_momentum(returns))
    elif interaction_id == 'value_quality':
        sig_a = sg.cross_sectional_rank(sg.book_to_market(ccm_fund, returns))
        sig_b = sg.cross_sectional_rank(sg.return_on_equity(ccm_fund, returns))
    elif interaction_id == 'profit_growth':
        sig_a = sg.cross_sectional_rank(sg.gross_profitability(ccm_fund, returns))
        sig_b = sg.cross_sectional_rank(sg.asset_growth(ccm_fund, returns))
    elif interaction_id == 'roe_mom':
        sig_a = sg.cross_sectional_rank(sg.return_on_equity(ccm_fund, returns))
        sig_b = sg.cross_sectional_rank(sg.multi_timeframe_momentum(returns))
    elif interaction_id == 'alpha_quality':
        sig_a = sg.cross_sectional_rank(sg.ff5_alpha(returns, data['ff5']))
        sig_b = sg.cross_sectional_rank(sg.gross_profitability(ccm_fund, returns))
    else:
        raise ValueError(f"未知交互信号: {interaction_id}")

    # Align
    common_idx = sig_a.index.intersection(sig_b.index)
    common_cols = sig_a.columns.intersection(sig_b.columns)
    sig_a = sig_a.loc[common_idx, common_cols].astype(float)
    sig_b = sig_b.loc[common_idx, common_cols].astype(float)

    # Interaction: multiply rankings (both positive = very positive, mixed = near zero)
    raw = sig_a * sig_b

    # Re-rank
    ranked = sg.cross_sectional_rank(raw)

    # Universe filter
    pct = mktcap.quantile(cap_quantile, axis=1)
    cap_mask = mktcap.ge(pct, axis=0).reindex(index=ranked.index, columns=ranked.columns)
    crash = sg.crash_filter(returns).reindex(index=ranked.index, columns=ranked.columns).fillna(False)
    signal = ranked.where(cap_mask & ~crash)

    if verbose:
        n = signal.notna().sum(axis=1).median()
        print(f"  交互信号 {interaction_id}: ~{n:.0f} 只/月")
    return signal


def build_orthogonal_signal(factor_id, data, cap_quantile=0.75, verbose=True):
    """Orthogonalized signal — residual alpha after stripping out FF factor exposures

    Runs a monthly cross-sectional regression on the raw signal: signal_i = a + b1*beta_i + b2*size_i + b3*bm_i + e_i
    and uses the residual e_i as the orthogonalized signal
    """
    sg = SignalGenerator()
    returns = data['returns']
    mktcap = data['mktcap']
    ccm_fund = data['ccm_fund']

    # Get raw signal
    raw_signal = build_factor_signal(factor_id, data, cap_quantile=cap_quantile, verbose=False)

    # Get conditioning variables: log market cap and book-to-market
    log_cap = np.log(mktcap.replace(0, np.nan)).astype(float)
    bm_raw = sg.book_to_market(ccm_fund, returns)

    # Cross-sectional orthogonalization: each month, regress signal on cap+bm, take residual
    orthogonal = pd.DataFrame(np.nan, index=raw_signal.index, columns=raw_signal.columns)
    for date in raw_signal.index:
        y = raw_signal.loc[date].dropna()
        if len(y) < 50:
            continue
        # Get conditioning vars for this date
        x_cap = log_cap.loc[date].reindex(y.index) if date in log_cap.index else pd.Series(dtype=float)
        x_bm = bm_raw.loc[date].reindex(y.index) if date in bm_raw.index else pd.Series(dtype=float)

        # Build X matrix
        valid = y.index[y.notna() & x_cap.notna()]
        if len(valid) < 30:
            orthogonal.loc[date, y.index] = y.values
            continue

        X = np.column_stack([
            np.ones(len(valid)),
            x_cap.loc[valid].fillna(0).values,
            x_bm.loc[valid].fillna(0).values,
        ]).astype(float)
        y_vals = y.loc[valid].values.astype(float)

        try:
            beta = np.linalg.lstsq(X, y_vals, rcond=None)[0]
            resid = y_vals - X @ beta
            orthogonal.loc[date, valid] = resid
        except Exception:
            orthogonal.loc[date, y.index] = y.values

    # Re-rank residuals
    ranked = sg.cross_sectional_rank(orthogonal.astype(float))

    # Apply universe filter
    pct = mktcap.quantile(cap_quantile, axis=1)
    cap_mask = mktcap.ge(pct, axis=0).reindex(index=ranked.index, columns=ranked.columns)
    crash = sg.crash_filter(returns).reindex(index=ranked.index, columns=ranked.columns).fillna(False)
    signal = ranked.where(cap_mask & ~crash)

    if verbose:
        n = signal.notna().sum(axis=1).median()
        print(f"  正交化 {factor_id}: ~{n:.0f} 只/月")
    return signal


# ── 自适应择时 (Round 2) ──

def build_regime_adjusted_signal(factor_id, data, cap_quantile=0.75, verbose=True):
    """Adaptive timing signal — adjusts factor weights by market regime

    - Reduce momentum exposure in high-volatility regimes (momentum crash risk)
    - Increase momentum exposure in low-volatility regimes
    - Increase value exposure when the value spread is wide
    """
    sg = SignalGenerator()
    returns = data['returns']
    mktcap = data['mktcap']
    spy_ret = data['spy_ret']

    # Get base signal
    base = build_factor_signal(factor_id, data, cap_quantile=cap_quantile, verbose=False)

    # Market regime: rolling 6-month market vol
    mkt_vol = spy_ret.rolling(6).std() * np.sqrt(12)
    median_vol = mkt_vol.expanding(min_periods=12).median()

    # Regime scale: reduce exposure in high-vol (> 1.5x median), increase in low-vol
    regime_scale = pd.Series(1.0, index=base.index)
    for date in base.index:
        date_p = date.to_period('M') if hasattr(date, 'to_period') else date
        mv = mkt_vol.reindex(mkt_vol.index.to_period('M')).get(date_p, np.nan) if hasattr(mkt_vol.index, 'to_period') else np.nan
        med = median_vol.reindex(median_vol.index.to_period('M')).get(date_p, np.nan) if hasattr(median_vol.index, 'to_period') else np.nan
        if pd.notna(mv) and pd.notna(med) and med > 0:
            ratio = mv / med
            if factor_id in ('mom12', 'accel', 'high52', 'ff5alpha'):
                # Momentum factors: reduce in high vol
                if ratio > 1.5:
                    regime_scale[date] = 0.5
                elif ratio > 1.2:
                    regime_scale[date] = 0.75
                elif ratio < 0.8:
                    regime_scale[date] = 1.2
            else:
                # Value/quality factors: slightly increase in high vol (mean reversion)
                if ratio > 1.5:
                    regime_scale[date] = 1.1
                elif ratio < 0.7:
                    regime_scale[date] = 0.9

    # Apply regime scaling (scale signal magnitude, not the rank)
    adjusted = base.mul(regime_scale, axis=0)

    if verbose:
        n_reduced = (regime_scale < 1.0).sum()
        print(f"  择时 {factor_id}: {n_reduced} 月减仓")
    return adjusted


# 已实现因子列表
IMPLEMENTED_FACTORS = [
    'mom12', 'accel', 'high52',
    'bm', 'ep', 'roe', 'gpa', 'ag',
    'turnover', 'revgrowth', 'de', 'ps', 'nigrowth',
    'ivol', 'ff5alpha',
    'composite',
]

VOL_FACTORS = [
    'lowvol', 'vov', 'downvol', 'maxret', 'beta', 'skew', 'volts', 'volmom', 'ivol',
]

FUNDAMENTAL_FACTORS = [
    'turnover', 'revgrowth', 'de', 'ps', 'nigrowth',
]

ALL_FACTORS = (IMPLEMENTED_FACTORS
               + [f for f in VOL_FACTORS if f not in IMPLEMENTED_FACTORS]
               + [f for f in FUNDAMENTAL_FACTORS if f not in IMPLEMENTED_FACTORS])

INTERACTION_FACTORS = [
    'quality_mom', 'value_quality', 'profit_growth',
    'roe_mom', 'alpha_quality',
]

EFFECTIVE_FACTORS = ['gpa', 'roe', 'ff5alpha', 'mom12', 'ag', 'ep']
