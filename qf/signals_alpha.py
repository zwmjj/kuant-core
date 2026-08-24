"""Alpha factor signal library - multi-factor models drawn from academic literature and quant fund research."""
import numpy as np
import pandas as pd


class AlphaSignalGenerator:
    """Collection of static methods producing alpha factor signals

    Inspired by academic work from Amihud, Roll, Da-Gurun-Warachka,
    Moskowitz-Ooi-Pedersen and others.
    All inputs are pandas DataFrames with index=date and columns=ticker.
    All signal outputs pass through cross_sectional_rank and lie in [-1, 1].
    """

    # ── 因子分类字典 ──

    ALPHA_CATEGORIES = {
        'liquidity': [
            'amihud_persistence', 'roll_spread', 'turnover_rate', 'liquidity_shock',
        ],
        'price_pattern': [
            'higher_highs', 'lower_lows', 'price_acceleration',
            'support_distance', 'resistance_distance', 'candlestick_body',
        ],
        'statistical': [
            'hurst_exponent', 'entropy', 'tail_risk',
            'autocorrelation', 'variance_ratio',
        ],
        'information_flow': [
            'delay_factor', 'co_movement', 'lead_lag',
        ],
        'momentum_refinement': [
            'frog_in_pan', 'max_return', 'time_series_momentum',
            'momentum_crash_filter',
        ],
    }

    # ── 工具 ──

    @staticmethod
    def cross_sectional_rank(signal):
        """Cross-sectional rank, mapped to [-1, 1]."""
        def rank_row(row):
            valid = row.dropna()
            if len(valid) < 10:
                return row * np.nan
            return (valid.rank(pct=True) * 2 - 1).reindex(row.index)
        return signal.apply(rank_row, axis=1)

    # ══════════════════════════════════════════════════════════════
    #  流动性因子 (Liquidity Factors)
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def amihud_persistence(returns, dollar_volume, window=20):
        """Amihud illiquidity autocorrelation - proxy for information asymmetry

        Amihud (2002) illiquidity = |ret| / dollar_volume.
        Computes its rolling window-day autocorrelation; persistent illiquidity
        signals information asymmetry and should be avoided.
        The signal is negated: persistently high illiquidity -> low rank.
        """
        illiq = returns.abs() / dollar_volume.replace(0, np.nan)
        illiq_lag = illiq.shift(1)

        def _rolling_autocorr(s, w):
            return s.rolling(w, min_periods=w // 2).corr(illiq_lag[s.name])

        raw = illiq.apply(lambda col: _rolling_autocorr(col, window))
        return AlphaSignalGenerator.cross_sectional_rank(-raw)

    @staticmethod
    def roll_spread(close, window=20):
        """Roll (1984) effective spread - liquidity measure

        Effective spread = 2 * sqrt(-cov(ret_t, ret_{t-1})).
        The spread is set to 0 when the covariance is positive. A low spread =
        good liquidity = favorable.
        The signal is negated: low spread -> high rank.
        """
        ret = close.pct_change()
        ret_lag = ret.shift(1)

        def _roll_col(col):
            cov_val = col.rolling(window, min_periods=window // 2).cov(ret_lag[col.name])
            spread = 2.0 * np.sqrt((-cov_val).clip(lower=0))
            return spread

        raw = ret.apply(_roll_col)
        return AlphaSignalGenerator.cross_sectional_rank(-raw)

    @staticmethod
    def turnover_rate(volume, shares_outstanding_proxy, window=20):
        """Turnover factor - attention and liquidity

        Turnover = volume / shares-outstanding proxy.
        Averaged over window days; high turnover = high attention.
        """
        turnover = volume / shares_outstanding_proxy.replace(0, np.nan)
        raw = turnover.rolling(window, min_periods=window // 2).mean()
        return AlphaSignalGenerator.cross_sectional_rank(raw)

    @staticmethod
    def liquidity_shock(dollar_volume, window=5, baseline=60):
        """Liquidity shock factor - abnormal volume changes

        (short-term average volume / long-term average volume). A sudden shift in
        liquidity is an event-driven signal.
        """
        short_avg = dollar_volume.rolling(window, min_periods=max(1, window // 2)).mean()
        long_avg = dollar_volume.rolling(baseline, min_periods=baseline // 2).mean()
        raw = short_avg / long_avg.replace(0, np.nan)
        return AlphaSignalGenerator.cross_sectional_rank(raw)

    # ══════════════════════════════════════════════════════════════
    #  价格形态因子 (Price Pattern Factors)
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def higher_highs(high, window=20):
        """Higher-highs count - bullish structure

        Number of days over the past window days where high > previous day's high.
        More higher-highs = uptrend structure = bullish.
        """
        hh = (high > high.shift(1)).astype(float)
        raw = hh.rolling(window, min_periods=window // 2).sum()
        return AlphaSignalGenerator.cross_sectional_rank(raw)

    @staticmethod
    def lower_lows(low, window=20):
        """Lower-lows count - bearish structure

        Number of days over the past window days where low < previous day's low.
        More lower-lows = downtrend = bearish. The signal is negated.
        """
        ll = (low < low.shift(1)).astype(float)
        raw = ll.rolling(window, min_periods=window // 2).sum()
        return AlphaSignalGenerator.cross_sectional_rank(-raw)

    @staticmethod
    def price_acceleration(close, short=5, long=20):
        """Price acceleration factor - second derivative of momentum

        Short-term momentum - long-term momentum = acceleration.
        Positive acceleration = strengthening trend = bullish.
        Inspired by Da, Gurun, Warachka (2014).
        """
        mom_short = close.pct_change(short)
        mom_long = close.pct_change(long)
        raw = mom_short - mom_long * (short / long)
        return AlphaSignalGenerator.cross_sectional_rank(raw)

    @staticmethod
    def support_distance(close, low, window=60):
        """Distance to support - overbought/oversold measure

        (close - window-day low) / close.
        A large distance = overextended = caution. The signal is negated.
        """
        rolling_low = low.rolling(window, min_periods=window // 2).min()
        raw = (close - rolling_low) / close.replace(0, np.nan)
        return AlphaSignalGenerator.cross_sectional_rank(-raw)

    @staticmethod
    def resistance_distance(close, high, window=60):
        """Distance to resistance - overhead pressure measure

        (window-day high - close) / close.
        Being close to resistance (small distance) = heavy selling pressure.
        The signal is negated.
        """
        rolling_high = high.rolling(window, min_periods=window // 2).max()
        raw = (rolling_high - close) / close.replace(0, np.nan)
        return AlphaSignalGenerator.cross_sectional_rank(-raw)

    @staticmethod
    def candlestick_body(open_, close, high, low):
        """Candlestick body ratio factor - conviction of trading

        |close - open| / (high - low).
        A large body = strong directional conviction = trend confirmation.
        """
        body = (close - open_).abs()
        shadow = (high - low).replace(0, np.nan)
        raw = body / shadow
        return AlphaSignalGenerator.cross_sectional_rank(raw)

    # ══════════════════════════════════════════════════════════════
    #  统计因子 (Statistical Factors)
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def hurst_exponent(close, window=100):
        """Hurst exponent proxy - R/S analysis

        H > 0.5 = trend persistence; H < 0.5 = mean reversion.
        Estimated with a simplified R/S method. Signal: H - 0.5
        (positive = trending, negative = mean-reverting).
        """
        ret = close.pct_change()

        def _hurst_col(col):
            result = pd.Series(np.nan, index=col.index)
            vals = col.values
            for i in range(window, len(vals)):
                segment = vals[i - window:i]
                valid = segment[~np.isnan(segment)]
                if len(valid) < window // 2:
                    continue
                mean_val = np.mean(valid)
                deviate = np.cumsum(valid - mean_val)
                r = np.max(deviate) - np.min(deviate)
                s = np.std(valid, ddof=1)
                if s > 1e-12 and r > 0:
                    result.iloc[i] = np.log(r / s) / np.log(len(valid))
            return result

        raw = ret.apply(_hurst_col)
        return AlphaSignalGenerator.cross_sectional_rank(raw - 0.5)

    @staticmethod
    def entropy(returns, window=20):
        """Shannon entropy - predictability of the return distribution

        Bins returns and computes the Shannon entropy. Low entropy = predictable = favorable.
        The signal is negated: low entropy -> high rank.
        """
        def _entropy_col(col):
            result = pd.Series(np.nan, index=col.index)
            vals = col.values
            for i in range(window, len(vals)):
                segment = vals[i - window:i]
                valid = segment[~np.isnan(segment)]
                if len(valid) < window // 2:
                    continue
                # 分10个箱
                counts, _ = np.histogram(valid, bins=10)
                probs = counts / counts.sum()
                probs = probs[probs > 0]
                result.iloc[i] = -np.sum(probs * np.log2(probs))
            return result

        raw = returns.apply(_entropy_col)
        return AlphaSignalGenerator.cross_sectional_rank(-raw)

    @staticmethod
    def tail_risk(returns, window=60):
        """Tail risk factor - Expected Shortfall (5%)

        Mean of returns below the 5th percentile over the past window days (CVaR).
        Low tail risk = safe = favorable. The signal is used as-is because ES is
        itself negative (a larger absolute ES = more dangerous).
        """
        def _es_col(col):
            result = pd.Series(np.nan, index=col.index)
            vals = col.values
            for i in range(window, len(vals)):
                segment = vals[i - window:i]
                valid = segment[~np.isnan(segment)]
                if len(valid) < window // 2:
                    continue
                threshold = np.percentile(valid, 5)
                tail = valid[valid <= threshold]
                if len(tail) > 0:
                    result.iloc[i] = np.mean(tail)
                else:
                    result.iloc[i] = threshold
            return result

        raw = returns.apply(_es_col)
        # ES值越大(越不负)=尾部风险越小=越好
        return AlphaSignalGenerator.cross_sectional_rank(raw)

    @staticmethod
    def autocorrelation(returns, lag=1, window=20):
        """Return autocorrelation factor

        Positive autocorrelation = momentum behavior; negative = reversal behavior.
        Cutler, Poterba, Summers (1989).
        """
        ret_lag = returns.shift(lag)

        def _autocorr_col(col):
            return col.rolling(window, min_periods=window // 2).corr(ret_lag[col.name])

        raw = returns.apply(_autocorr_col)
        return AlphaSignalGenerator.cross_sectional_rank(raw)

    @staticmethod
    def variance_ratio(returns, short=5, long=20):
        """Variance ratio factor - random walk test

        VR = Var(long_ret) / (Var(short_ret) * long/short).
        VR > 1 = positive autocorrelation (momentum); VR < 1 = negative
        autocorrelation (reversal).
        Lo & MacKinlay (1988).
        """
        var_short = returns.rolling(short, min_periods=max(2, short // 2)).var()
        ret_long = returns.rolling(long, min_periods=long // 2).sum()
        var_long = ret_long.rolling(long, min_periods=long // 2).var()
        ratio = long / short
        raw = var_long / (var_short * ratio).replace(0, np.nan)
        return AlphaSignalGenerator.cross_sectional_rank(raw - 1.0)

    # ══════════════════════════════════════════════════════════════
    #  信息流因子 (Information Flow Factors)
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def delay_factor(returns, market_returns, window=60):
        """Information delay factor - price discovery efficiency

        Hou & Moskowitz (2005).
        Compares the R² of a regression including lagged market returns against
        one using only the contemporaneous return.
        High delay = slow reaction to information = potential catch-up move ahead.
        """
        mkt = market_returns
        mkt_lag1 = mkt.shift(1)
        mkt_lag2 = mkt.shift(2)
        mkt_lag3 = mkt.shift(3)

        def _delay_col(col):
            result = pd.Series(np.nan, index=col.index)
            y = col.values
            x_contemp = mkt.values
            x_lag1 = mkt_lag1.values
            x_lag2 = mkt_lag2.values
            x_lag3 = mkt_lag3.values

            for i in range(window, len(y)):
                sl = slice(i - window, i)
                yi = y[sl]
                mask = ~(np.isnan(yi) | np.isnan(x_contemp[sl]) |
                         np.isnan(x_lag1[sl]) | np.isnan(x_lag2[sl]) |
                         np.isnan(x_lag3[sl]))
                if mask.sum() < window // 2:
                    continue
                yi_v = yi[mask]
                ss_tot = np.var(yi_v) * len(yi_v)
                if ss_tot < 1e-16:
                    continue
                # 仅同期回归 R²
                xc = x_contemp[sl][mask]
                beta_c = np.cov(yi_v, xc)[0, 1] / (np.var(xc) + 1e-16)
                ss_res_c = np.sum((yi_v - beta_c * xc) ** 2)
                r2_c = 1 - ss_res_c / ss_tot if ss_tot > 0 else 0

                # 含滞后回归 R²（简化：加3个滞后）
                X = np.column_stack([
                    x_contemp[sl][mask], x_lag1[sl][mask],
                    x_lag2[sl][mask], x_lag3[sl][mask],
                ])
                try:
                    beta_f, _, _, _ = np.linalg.lstsq(X, yi_v, rcond=None)
                    ss_res_f = np.sum((yi_v - X @ beta_f) ** 2)
                    r2_f = 1 - ss_res_f / ss_tot if ss_tot > 0 else 0
                except np.linalg.LinAlgError:
                    continue

                delay = (r2_f - r2_c) / (r2_f + 1e-16) if r2_f > 0 else 0
                result.iloc[i] = delay
            return result

        raw = returns.apply(_delay_col)
        return AlphaSignalGenerator.cross_sectional_rank(raw)

    @staticmethod
    def co_movement(returns, window=20):
        """Co-movement factor - correlation with the cross-sectional average

        Rolling correlation between a stock and the cross-sectional mean.
        Low co-movement = strongly idiosyncratic = a source of alpha.
        The signal is negated.
        """
        market_avg = returns.mean(axis=1)

        def _co_col(col):
            return col.rolling(window, min_periods=window // 2).corr(market_avg)

        raw = returns.apply(_co_col)
        return AlphaSignalGenerator.cross_sectional_rank(-raw)

    @staticmethod
    def lead_lag(returns, market_returns, window=20):
        """Lead-lag factor - degree of information leadership

        Correlation between a stock's current return and the market's next-day return.
        High correlation = the stock leads the market = an information advantage.
        """
        mkt_lead = market_returns.shift(-1)

        def _lead_col(col):
            return col.rolling(window, min_periods=window // 2).corr(mkt_lead)

        raw = returns.apply(_lead_col)
        return AlphaSignalGenerator.cross_sectional_rank(raw)

    # ══════════════════════════════════════════════════════════════
    #  动量改进因子 (Momentum Refinements)
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def frog_in_pan(returns, window=60):
        """Frog-in-the-pan factor - continuous vs. discrete momentum

        Da, Gurun, Warachka (2014).
        Decomposes momentum into steady small moves vs. a few large jumps.
        Continuous momentum is the stronger signal = investors under-react.
        FROG = sign(cum_ret) * (pct_positive - pct_negative).
        """
        cum_ret = returns.rolling(window, min_periods=window // 2).sum()
        pos_days = (returns > 0).astype(float).rolling(window, min_periods=window // 2).sum()
        neg_days = (returns < 0).astype(float).rolling(window, min_periods=window // 2).sum()
        total_days = pos_days + neg_days
        pct_pos = pos_days / total_days.replace(0, np.nan)
        pct_neg = neg_days / total_days.replace(0, np.nan)
        raw = np.sign(cum_ret) * (pct_pos - pct_neg)
        return AlphaSignalGenerator.cross_sectional_rank(raw)

    @staticmethod
    def max_return(returns, window=20):
        """Maximum daily return factor - lottery stock effect

        Bali, Cakici, Whitelaw (2011).
        Largest single-day return within the window.
        High max_ret = lottery-like stock = low expected future return.
        The signal is negated.
        """
        raw = returns.rolling(window, min_periods=window // 2).max()
        return AlphaSignalGenerator.cross_sectional_rank(-raw)

    @staticmethod
    def time_series_momentum(returns, window=252):
        """Time-series momentum - TSMOM

        Moskowitz, Ooi, Pedersen (2012).
        A stock's own cumulative return over the past window days.
        Unlike cross-sectional momentum, TSMOM looks at the stock's own trend
        rather than its relative rank.
        """
        raw = returns.rolling(window, min_periods=window // 2).sum()
        return AlphaSignalGenerator.cross_sectional_rank(raw)

    @staticmethod
    def momentum_crash_filter(returns, market_returns, window=60):
        """Momentum crash filter - conditional momentum

        Daniel & Moskowitz (2016).
        Momentum strategies are prone to crashing after large market declines.
        Signal = momentum * (1 - I(market_in_crash)).
        The momentum signal is damped when the market drawdown over the window
        exceeds -10%.
        """
        mom = returns.rolling(window, min_periods=window // 2).sum()
        mkt_cum = market_returns.rolling(window, min_periods=window // 2).sum()
        # 市场回撤严重时对动量信号打折
        crash_mask = (mkt_cum < -0.10).astype(float)
        dampened = mom * (1.0 - 0.5 * crash_mask)
        return AlphaSignalGenerator.cross_sectional_rank(dampened)

    # ══════════════════════════════════════════════════════════════
    #  复合因子构建 (Composite)
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def build_alpha_signal(data_dict, category='all', weights=None):
        """Build a composite alpha signal - equal-weight blend by category or across all factors

        Parameters
        ----------
        data_dict : dict
            Factor name -> already-ranked DataFrame signal.
        category : str
            Factor category name ('liquidity', 'price_pattern', 'statistical',
            'information_flow', 'momentum_refinement') or 'all'.
        weights : dict, optional
            Factor name -> weight. Equal weights by default.

        Returns
        ------
        DataFrame : Composite signal in [-1, 1].
        """
        cats = AlphaSignalGenerator.ALPHA_CATEGORIES

        if category == 'all':
            factor_names = [f for factors in cats.values() for f in factors]
        elif category in cats:
            factor_names = cats[category]
        else:
            raise ValueError(f"未知类别: {category}. 可选: {list(cats.keys())} 或 'all'")

        available = [f for f in factor_names if f in data_dict]
        if not available:
            raise ValueError(f"data_dict 中无可用因子。需要: {factor_names}")

        if weights is None:
            w = {f: 1.0 / len(available) for f in available}
        else:
            total = sum(weights.get(f, 0) for f in available)
            if total <= 0:
                raise ValueError("权重之和必须大于0")
            w = {f: weights.get(f, 0) / total for f in available}

        composite = None
        for f in available:
            contribution = data_dict[f] * w[f]
            if composite is None:
                composite = contribution
            else:
                composite = composite.add(contribution, fill_value=0)

        return AlphaSignalGenerator.cross_sectional_rank(composite)
