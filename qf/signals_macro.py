"""Macro and cross-asset factor signal library - stock selection driven by macro indicators (gold, crude oil, bonds, volatility, sector ETFs)."""
import numpy as np
import pandas as pd


# 所需宏观ETF列表
MACRO_ETFS = [
    'SPY', 'GLD', 'USO', 'DBC', 'TLT', 'IEF', 'HYG', 'UUP',
    'XLE', 'XLF', 'XLK', 'XLV', 'XLU', 'XLP',
]

SECTOR_ETFS = ['XLE', 'XLF', 'XLK', 'XLV', 'XLU', 'XLP']


class MacroSignalGenerator:
    """Collection of static methods producing macro and cross-asset factor signals

    stock_returns: DataFrame with index=date, columns=ticker, values=daily returns
    Each column of macro_data holds a macro ETF's daily returns (or prices, depending
    on the method).
    All signal outputs pass through cross_sectional_rank and lie in [-1, 1].
    """

    # ══════════════════════════════════════════════════════════════
    #  工具
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def cross_sectional_rank(signal):
        """Cross-sectional rank, mapped to [-1, 1]."""
        def rank_row(row):
            valid = row.dropna()
            if len(valid) < 10:
                return row * np.nan
            return (valid.rank(pct=True) * 2 - 1).reindex(row.index)
        return signal.apply(rank_row, axis=1)

    @staticmethod
    def _rolling_beta(y, x, window):
        """滚动OLS beta: y对x的回归斜率

        y: DataFrame (T x N), x: Series (T,)
        返回 DataFrame (T x N)
        """
        x_aligned = x.reindex(y.index)
        x_mean = x_aligned.rolling(window, min_periods=max(window // 2, 20)).mean()
        x_var = x_aligned.rolling(window, min_periods=max(window // 2, 20)).var()

        betas = pd.DataFrame(np.nan, index=y.index, columns=y.columns)
        for col in y.columns:
            yi = y[col]
            cov = (yi * x_aligned).rolling(window, min_periods=max(window // 2, 20)).mean() \
                  - yi.rolling(window, min_periods=max(window // 2, 20)).mean() * x_mean
            betas[col] = cov / x_var.replace(0, np.nan)
        return betas

    @staticmethod
    def _safe_get(macro_data, key, fallback=None):
        """安全获取宏观数据列，缺失则用备选或返回None"""
        if key in macro_data.columns:
            return macro_data[key]
        if fallback and fallback in macro_data.columns:
            return macro_data[fallback]
        return None

    # ══════════════════════════════════════════════════════════════
    #  利率敏感性 (Interest Rate Sensitivity)
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def rate_sensitivity(stock_returns, tlt_returns, window=60):
        """Rate sensitivity - rolling beta of a stock to TLT (long-duration Treasuries)

        A negative beta marks a rate-sensitive growth stock that underperforms in
        hiking cycles.
        A positive beta marks a defensive stock that moves with bonds.
        Returns the cross-sectionally ranked signal; high values = high rate sensitivity.
        """
        betas = MacroSignalGenerator._rolling_beta(stock_returns, tlt_returns, window)
        return MacroSignalGenerator.cross_sectional_rank(betas)

    @staticmethod
    def real_yield_regime(gld_returns, tlt_returns, window=20):
        """Real yield regime - gold up + bonds down = rising real yields

        Returns a Series (T,) in [-1, 1]:
          Positive = rising real yield regime (unfavorable for growth stocks)
          Negative = falling real yield regime (favorable for growth stocks)
        Intended to be used with rate_sensitivity: avoid high TLT-beta stocks when
        real yields are rising.
        """
        # 黄金相对债券的强弱 = gold涨 + bond跌 → 实际收益率上升
        spread = gld_returns - tlt_returns
        regime = spread.rolling(window, min_periods=window // 2).mean()
        # 标准化到 [-1, 1]
        regime_z = (regime - regime.rolling(252, min_periods=60).mean()) \
                   / regime.rolling(252, min_periods=60).std().replace(0, np.nan)
        return regime_z.clip(-3, 3) / 3

    # ══════════════════════════════════════════════════════════════
    #  大宗商品暴露 (Commodity Exposure)
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def oil_beta(stock_returns, uso_returns, window=60):
        """Oil exposure - rolling beta of a stock to USO (crude oil ETF)

        High beta = large energy-related exposure. Favorable in energy bull markets.
        """
        betas = MacroSignalGenerator._rolling_beta(stock_returns, uso_returns, window)
        return MacroSignalGenerator.cross_sectional_rank(betas)

    @staticmethod
    def gold_beta(stock_returns, gld_returns, window=60):
        """Gold exposure - rolling beta of a stock to GLD (gold ETF)

        High beta = gold-hedge characteristics. Favorable in risk-off regimes.
        """
        betas = MacroSignalGenerator._rolling_beta(stock_returns, gld_returns, window)
        return MacroSignalGenerator.cross_sectional_rank(betas)

    @staticmethod
    def commodity_regime_tilt(stock_returns, dbc_returns, window=20):
        """Commodity trend timing tilt - favor high commodity-beta stocks in commodity uptrends

        First determines the direction of the DBC trend, then computes each stock's
        beta to DBC.
        In commodity uptrends high beta scores highly; in downtrends low beta scores
        highly (the signal flips).
        """
        # 商品趋势判断
        trend = dbc_returns.rolling(window, min_periods=window // 2).mean()
        regime = np.sign(trend)  # +1 上涨, -1 下跌

        # 股票对商品的beta
        betas = MacroSignalGenerator._rolling_beta(stock_returns, dbc_returns, window=60)

        # 趋势上涨时偏好高beta，下跌时偏好低beta
        tilted = betas.multiply(regime, axis=0)
        return MacroSignalGenerator.cross_sectional_rank(tilted)

    # ══════════════════════════════════════════════════════════════
    #  波动率环境 (Volatility Regime)
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def vix_beta(stock_returns, spy_returns, window=60):
        """Downside beta - beta computed only on SPY down days (a VIX proxy)

        High downside beta = crash-vulnerable stock that falls harder in panics.
        """
        # 仅保留SPY下跌日
        mask = spy_returns < 0
        spy_down = spy_returns.where(mask, np.nan)
        stock_down = stock_returns.where(mask, np.nan)

        betas = MacroSignalGenerator._rolling_beta(stock_down, spy_down, window)
        return MacroSignalGenerator.cross_sectional_rank(betas)

    @staticmethod
    def vol_regime_tilt(stock_returns, spy_returns, window=20):
        """Volatility regime timing tilt - favor low beta in high-vol regimes and high beta in low-vol regimes

        Uses SPY realized volatility to classify the regime and combines it with each
        stock's beta for timing.
        """
        # 已实现波动率
        realized_vol = spy_returns.rolling(window, min_periods=window // 2).std() * np.sqrt(252)
        vol_median = realized_vol.rolling(252, min_periods=60).median()
        high_vol = (realized_vol > vol_median).astype(float)  # 1=高波动, 0=低波动

        # 个股全样本beta
        full_beta = MacroSignalGenerator._rolling_beta(stock_returns, spy_returns, window=60)

        # 高波动: 偏好低beta (翻转); 低波动: 偏好高beta
        regime_sign = 1 - 2 * high_vol  # +1 低波动, -1 高波动
        tilted = full_beta.multiply(regime_sign, axis=0)
        return MacroSignalGenerator.cross_sectional_rank(tilted)

    @staticmethod
    def correlation_regime(stock_returns, spy_returns, window=60):
        """Rolling correlation regime - rolling correlation between a stock and the market

        In high-correlation regimes only low-correlation (idiosyncratic) stocks offer
        diversification value.
        Returns: the negated correlation rank -> high values = low correlation = more
        idiosyncratic value.
        """
        spy_aligned = spy_returns.reindex(stock_returns.index)
        corrs = pd.DataFrame(np.nan, index=stock_returns.index, columns=stock_returns.columns)
        min_obs = max(window // 2, 20)
        for col in stock_returns.columns:
            corrs[col] = stock_returns[col].rolling(window, min_periods=min_obs).corr(spy_aligned)

        # 取负: 低相关性=高分
        return MacroSignalGenerator.cross_sectional_rank(-corrs)

    # ══════════════════════════════════════════════════════════════
    #  行业轮动 (Sector Rotation)
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def sector_momentum(sector_etf_returns, window=20):
        """Sector momentum - cross-sectional momentum rank of sector ETFs

        Returns a DataFrame with index=date, columns=sector ETF ticker, values=[-1,1] ranks.
        Mapped down to individual stocks via each stock's beta to the sector ETF.
        """
        cum_ret = sector_etf_returns.rolling(window, min_periods=window // 2).sum()
        return MacroSignalGenerator.cross_sectional_rank(cum_ret)

    @staticmethod
    def sector_mean_reversion(sector_etf_returns, window=5):
        """Sector short-term reversal - contrarian signal on short-window sector ETF returns

        Sectors with the largest gains over the past 5 days tend to revert in the short run.
        """
        cum_ret = sector_etf_returns.rolling(window, min_periods=max(window // 2, 3)).sum()
        return MacroSignalGenerator.cross_sectional_rank(-cum_ret)

    @staticmethod
    def defensive_tilt(spy_returns, window=20):
        """Defensive tilt - favor defensive stocks when the SPY trend is negative

        Returns a Series (T,) in [-1, 1]:
          Negative = SPY downtrend -> overweight utilities/staples/healthcare (low beta)
          Positive = SPY uptrend -> overweight cyclicals (high beta)
        """
        trend = spy_returns.rolling(window, min_periods=window // 2).mean()
        trend_z = (trend - trend.rolling(252, min_periods=60).mean()) \
                  / trend.rolling(252, min_periods=60).std().replace(0, np.nan)
        return trend_z.clip(-3, 3) / 3

    # ══════════════════════════════════════════════════════════════
    #  跨资产动量 (Cross-Asset Momentum)
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def risk_appetite(spy_returns, gld_returns, window=20):
        """Risk appetite indicator - trend in the SPY/GLD ratio

        SPY strengthening against GLD = risk-on -> favor high-beta stocks
        SPY weakening against GLD = risk-off -> favor low-beta/defensive stocks
        Returns a Series (T,) in [-1, 1].
        """
        spread = spy_returns - gld_returns
        trend = spread.rolling(window, min_periods=window // 2).mean()
        trend_z = (trend - trend.rolling(252, min_periods=60).mean()) \
                  / trend.rolling(252, min_periods=60).std().replace(0, np.nan)
        return trend_z.clip(-3, 3) / 3

    @staticmethod
    def dollar_regime(uup_returns, window=20):
        """Dollar regime - a strong dollar hurts multinationals

        UUP in an uptrend = strong dollar -> favor domestically oriented stocks
        Returns a Series (T,) in [-1, 1]. Positive = strong-dollar regime.
        """
        trend = uup_returns.rolling(window, min_periods=window // 2).mean()
        trend_z = (trend - trend.rolling(252, min_periods=60).mean()) \
                  / trend.rolling(252, min_periods=60).std().replace(0, np.nan)
        return trend_z.clip(-3, 3) / 3

    @staticmethod
    def credit_spread_proxy(hyg_returns, tlt_returns, window=20):
        """Credit spread proxy - trend in the HYG-TLT spread

        HYG weakening against TLT = widening credit spreads = risk-off
        Returns a Series (T,) in [-1, 1]. Negative = risk-off (spreads widening).
        """
        spread = hyg_returns - tlt_returns
        trend = spread.rolling(window, min_periods=window // 2).mean()
        trend_z = (trend - trend.rolling(252, min_periods=60).mean()) \
                  / trend.rolling(252, min_periods=60).std().replace(0, np.nan)
        return trend_z.clip(-3, 3) / 3

    # ══════════════════════════════════════════════════════════════
    #  复合信号 (Composite)
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def build_macro_signal(stock_returns, macro_data, regime='auto'):
        """Detect the macro regime automatically and build a composite macro signal

        Args:
            stock_returns: DataFrame (T x N), daily stock returns
            macro_data: DataFrame whose columns are the daily returns of each macro ETF
            regime: 'auto' for automatic detection | 'risk_on' | 'risk_off' | 'neutral'

        Returns: DataFrame (T x N), composite macro stock-selection signal in [-1, 1]

        Automatic regime detection logic:
          - risk_on: SPY trending up + credit spreads tightening
          - risk_off: SPY trending down + credit spreads widening
          - neutral: everything else
        """
        sg = MacroSignalGenerator
        spy = sg._safe_get(macro_data, 'SPY')
        gld = sg._safe_get(macro_data, 'GLD')
        tlt = sg._safe_get(macro_data, 'TLT', 'IEF')
        hyg = sg._safe_get(macro_data, 'HYG')

        # ── 环境检测 ──
        if regime == 'auto':
            scores = []
            if spy is not None:
                def_tilt = sg.defensive_tilt(spy)
                scores.append(def_tilt)
            if hyg is not None and tlt is not None:
                credit = sg.credit_spread_proxy(hyg, tlt)
                scores.append(credit)
            if spy is not None and gld is not None:
                risk_app = sg.risk_appetite(spy, gld)
                scores.append(risk_app)

            if scores:
                avg_score = sum(scores) / len(scores)
                # 取最近有效值作为环境判断
                recent = avg_score.dropna().iloc[-1] if len(avg_score.dropna()) > 0 else 0.0
                if recent > 0.2:
                    regime = 'risk_on'
                elif recent < -0.2:
                    regime = 'risk_off'
                else:
                    regime = 'neutral'
            else:
                regime = 'neutral'

        # ── 根据环境构建信号 ──
        signals = []
        weights = []

        # 利率敏感性 (所有环境)
        if tlt is not None:
            rate_sig = sg.rate_sensitivity(stock_returns, tlt)
            if regime == 'risk_off':
                signals.append(-rate_sig)  # risk-off: 避开利率敏感股
                weights.append(2.0)
            else:
                signals.append(rate_sig)
                weights.append(1.0)

        # 下行beta (所有环境)
        if spy is not None:
            vix_sig = sg.vix_beta(stock_returns, spy)
            if regime == 'risk_off':
                signals.append(-vix_sig)  # risk-off: 避开崩盘易损股
                weights.append(3.0)
            elif regime == 'risk_on':
                signals.append(vix_sig)   # risk-on: 偏好高beta
                weights.append(1.0)
            else:
                signals.append(-vix_sig)
                weights.append(1.0)

        # 相关性 (高波动环境更重要)
        if spy is not None:
            corr_sig = sg.correlation_regime(stock_returns, spy)
            w = 2.0 if regime == 'risk_off' else 1.0
            signals.append(corr_sig)
            weights.append(w)

        # 商品倾斜
        dbc = sg._safe_get(macro_data, 'DBC')
        if dbc is not None:
            comm_sig = sg.commodity_regime_tilt(stock_returns, dbc)
            signals.append(comm_sig)
            weights.append(1.0)

        # 波动率倾斜
        if spy is not None:
            vol_sig = sg.vol_regime_tilt(stock_returns, spy)
            signals.append(vol_sig)
            weights.append(1.5)

        if not signals:
            return pd.DataFrame(0.0, index=stock_returns.index, columns=stock_returns.columns)

        # 加权平均
        total_w = sum(weights)
        composite = sum(s * w for s, w in zip(signals, weights)) / total_w
        return sg.cross_sectional_rank(composite)

    @staticmethod
    def prepare_macro_signals(stock_returns, macro_data):
        """Compute every macro factor signal and return them as a dict

        Args:
            stock_returns: DataFrame (T x N), daily stock returns
            macro_data: DataFrame whose columns are the daily returns of each macro ETF

        Returns: dict[str, DataFrame/Series]
            Keys are signal names, values are the corresponding signal DataFrame or Series.
            Signals whose macro inputs are missing are skipped.
        """
        sg = MacroSignalGenerator
        results = {}

        spy = sg._safe_get(macro_data, 'SPY')
        gld = sg._safe_get(macro_data, 'GLD')
        tlt = sg._safe_get(macro_data, 'TLT', 'IEF')
        uso = sg._safe_get(macro_data, 'USO')
        dbc = sg._safe_get(macro_data, 'DBC')
        hyg = sg._safe_get(macro_data, 'HYG')
        uup = sg._safe_get(macro_data, 'UUP')

        # ── 利率敏感性 ──
        if tlt is not None:
            results['rate_sensitivity'] = sg.rate_sensitivity(stock_returns, tlt)
        if gld is not None and tlt is not None:
            results['real_yield_regime'] = sg.real_yield_regime(gld, tlt)

        # ── 大宗商品暴露 ──
        if uso is not None:
            results['oil_beta'] = sg.oil_beta(stock_returns, uso)
        if gld is not None:
            results['gold_beta'] = sg.gold_beta(stock_returns, gld)
        if dbc is not None:
            results['commodity_regime_tilt'] = sg.commodity_regime_tilt(stock_returns, dbc)

        # ── 波动率环境 ──
        if spy is not None:
            results['vix_beta'] = sg.vix_beta(stock_returns, spy)
            results['vol_regime_tilt'] = sg.vol_regime_tilt(stock_returns, spy)
            results['correlation_regime'] = sg.correlation_regime(stock_returns, spy)

        # ── 行业轮动 ──
        available_sectors = [s for s in SECTOR_ETFS if s in macro_data.columns]
        if len(available_sectors) >= 3:
            sector_rets = macro_data[available_sectors]
            results['sector_momentum'] = sg.sector_momentum(sector_rets)
            results['sector_mean_reversion'] = sg.sector_mean_reversion(sector_rets)

        if spy is not None:
            results['defensive_tilt'] = sg.defensive_tilt(spy)

        # ── 跨资产动量 ──
        if spy is not None and gld is not None:
            results['risk_appetite'] = sg.risk_appetite(spy, gld)
        if uup is not None:
            results['dollar_regime'] = sg.dollar_regime(uup)
        if hyg is not None and tlt is not None:
            results['credit_spread_proxy'] = sg.credit_spread_proxy(hyg, tlt)

        # ── 复合信号 ──
        results['macro_composite'] = sg.build_macro_signal(stock_returns, macro_data)

        return results
