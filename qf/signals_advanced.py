"""Advanced alpha factor library - exploits under-used Alpaca data: microstructure, intraday patterns and volatility"""
import numpy as np
import pandas as pd


class AdvancedSignalGenerator:
    """A collection of static methods producing advanced factor signals.

    Every input is a pandas DataFrame indexed by date with tickers as columns.
    Every output signal is passed through cross_rank and lies in [-1, 1].

    Structure of data_dict:
        close, open, high, low, volume, trade_count, vwap,
        spy (Series of SPY closes), sector_returns (optional DataFrame)
    """

    # ═══════════════════════════════════════════════════════
    # 工具
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def cross_rank(signal):
        """Cross-sectional rank, mapped onto [-1, 1]."""
        def _rank_row(row):
            valid = row.dropna()
            if len(valid) < 10:
                return row * np.nan
            return (valid.rank(pct=True) * 2 - 1).reindex(row.index)
        return signal.apply(_rank_row, axis=1)

    # ═══════════════════════════════════════════════════════
    # 微观结构因子 (Microstructure)
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def kyle_lambda(returns, volume, window=20):
        """Kyle's lambda - the price-impact coefficient.

        Rolling regression of |returns| on volume; the slope is lambda.
        A high lambda means poor liquidity, so the signal is negated to
        favour liquid names.
        """
        abs_ret = returns.abs()

        def _regress(y, x):
            """单变量OLS斜率"""
            xm = x - x.mean()
            ym = y - y.mean()
            denom = (xm ** 2).sum()
            if denom == 0:
                return np.nan
            return (xm * ym).sum() / denom

        lam = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
        for col in returns.columns:
            r_col = abs_ret[col].dropna()
            v_col = volume[col].reindex(r_col.index).dropna()
            idx = r_col.index.intersection(v_col.index)
            if len(idx) < window:
                continue
            r_s = r_col.reindex(idx)
            v_s = v_col.reindex(idx)
            for i in range(window, len(idx)):
                sl = slice(i - window, i)
                lam.loc[idx[i], col] = _regress(r_s.iloc[sl], v_s.iloc[sl])

        raw = -lam  # 取负：偏好流动性好的
        return AdvancedSignalGenerator.cross_rank(raw)

    @staticmethod
    def order_flow_imbalance(close, high, low, volume, window=10):
        """Order flow imbalance (OFI).

        Estimated buy volume  = volume * (close - low) / (high - low)
        Estimated sell volume = volume - buy_vol
        OFI = rolling_sum(buy_vol - sell_vol)
        Positive values indicate buying pressure dominates.
        """
        hl_range = (high - low).replace(0, np.nan)
        buy_ratio = (close - low) / hl_range
        buy_ratio = buy_ratio.clip(0, 1)
        buy_vol = volume * buy_ratio
        sell_vol = volume * (1 - buy_ratio)
        ofi = (buy_vol - sell_vol).rolling(window).sum()
        return AdvancedSignalGenerator.cross_rank(ofi)

    @staticmethod
    def realized_spread(high, low, close, window=20):
        """Realized-spread indicator.

        Rolling mean of (2*close - high - low) / close.
        Positive values mean the close sits near the high, which reads bullish.
        """
        raw = (2 * close - high - low) / close
        smoothed = raw.rolling(window).mean()
        return AdvancedSignalGenerator.cross_rank(smoothed)

    @staticmethod
    def tick_size_clustering(close, window=20):
        """Price clustering - a proxy for retail behaviour.

        Fraction of the last `window` days whose close landed on a round
        ($X.00) or half-round ($X.50) price. Heavy clustering indicates retail
        dominance, since retail traders favour round quotes, so the signal is
        negated.
        """
        remainder = close % 1.0
        is_round = ((remainder < 0.02) | (remainder > 0.98) |
                    ((remainder > 0.48) & (remainder < 0.52)))
        is_round = is_round.astype(float)
        cluster_pct = is_round.rolling(window).mean()
        raw = -cluster_pct  # 取负：回避散户主导标的
        return AdvancedSignalGenerator.cross_rank(raw)

    # ═══════════════════════════════════════════════════════
    # 日内模式因子 (Intraday Pattern)
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def overnight_return(open_prices, prev_close):
        """Overnight return - an institutional pricing signal.

        overnight = open / prev_close - 1
        Persistent cumulative overnight returns indicate institutions building
        positions after hours or pre-market.
        prev_close may be passed as close.shift(1).
        """
        raw = open_prices / prev_close - 1
        cum_on = raw.rolling(20).sum()
        return AdvancedSignalGenerator.cross_rank(cum_on)

    @staticmethod
    def intraday_return(close, open_prices):
        """Intraday return - retail momentum.

        intraday = close / open - 1
        Cumulative intraday returns track retail flow.
        """
        raw = close / open_prices - 1
        cum_id = raw.rolling(20).sum()
        return AdvancedSignalGenerator.cross_rank(cum_id)

    @staticmethod
    def overnight_intraday_divergence(open_prices, close):
        """Overnight-versus-intraday divergence factor.

        When the overnight and intraday returns disagree in sign, follow the
        overnight direction as the better-informed flow.
        Signal = rolling sum of (overnight_ret - intraday_ret).
        """
        prev_close = close.shift(1)
        on_ret = open_prices / prev_close - 1
        id_ret = close / open_prices - 1
        divergence = (on_ret - id_ret).rolling(20).sum()
        return AdvancedSignalGenerator.cross_rank(divergence)

    # ═══════════════════════════════════════════════════════
    # 成交量模式因子 (Volume Pattern)
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def volume_autocorrelation(volume, window=20):
        """Volume autocorrelation - detects scheduled institutional execution.

        Rolling correlation between volume and its lag-1 value.
        High autocorrelation suggests institutions working a large order to a
        schedule (TWAP/VWAP algorithms).
        """
        vol_lag = volume.shift(1)
        corr = volume.rolling(window).corr(vol_lag)
        return AdvancedSignalGenerator.cross_rank(corr)

    @staticmethod
    def price_volume_correlation(returns, volume, window=20):
        """Price-volume correlation.

        Positive correlation confirms a trend (rising or falling on heavy volume).
        Negative correlation is a price-volume divergence and a possible reversal.
        The correlation coefficient is used directly as the signal.
        """
        corr = returns.rolling(window).corr(volume)
        return AdvancedSignalGenerator.cross_rank(corr)

    @staticmethod
    def abnormal_volume_return(returns, volume, window=20):
        """Returns on abnormal-volume days - a smart-money signal.

        A day is abnormal when volume > 2 * rolling_mean(volume).
        Returns on non-abnormal days are zeroed out and the rest accumulated.
        """
        vol_ma = volume.rolling(window).mean()
        is_abnormal = volume > (2 * vol_ma)
        abn_ret = returns.where(is_abnormal, 0.0)
        cum_abn = abn_ret.rolling(window).sum()
        return AdvancedSignalGenerator.cross_rank(cum_abn)

    # ═══════════════════════════════════════════════════════
    # 波幅与波动率因子 (Range & Volatility)
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def garman_klass_vol(open_prices, high, low, close, window=20):
        """Garman-Klass volatility estimator.

        GK = 0.5*ln(H/L)^2 - (2ln2-1)*ln(C/O)^2
        More efficient than close-to-close volatility because it uses the full
        OHLC bar. The signal is negated to capture the low-volatility premium.
        """
        log_hl = np.log(high / low)
        log_co = np.log(close / open_prices)
        gk_daily = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
        gk_vol = gk_daily.rolling(window).mean().apply(np.sqrt)
        raw = -gk_vol
        return AdvancedSignalGenerator.cross_rank(raw)

    @staticmethod
    def parkinson_vol(high, low, window=20):
        """Parkinson volatility - estimated from the intraday high-low range.

        PV = sqrt(1/(4*N*ln2) * sum(ln(H/L)^2))
        Captures intraday variation that close-to-close volatility misses.
        The signal is negated.
        """
        log_hl_sq = np.log(high / low) ** 2
        pv = np.sqrt(log_hl_sq.rolling(window).mean() / (4 * np.log(2)))
        raw = -pv
        return AdvancedSignalGenerator.cross_rank(raw)

    @staticmethod
    def vol_of_vol(returns, inner=10, outer=60):
        """Volatility of volatility (VoV).

        Computes rolling volatility over the inner window, then the standard
        deviation of that series over the outer window.
        High VoV means elevated uncertainty and reads as bearish.
        """
        inner_vol = returns.rolling(inner).std()
        vov = inner_vol.rolling(outer).std()
        raw = -vov
        return AdvancedSignalGenerator.cross_rank(raw)

    @staticmethod
    def range_expansion(high, low, close, window=20):
        """Range-expansion factor.

        Today's range divided by the 20-day average range.
        An expanding range signals a breakout; combining it with direction
        (where the close sits in the bar) yields a directional signal.
        """
        daily_range = high - low
        avg_range = daily_range.rolling(window).mean()
        expansion = daily_range / avg_range.replace(0, np.nan)
        # 结合方向: 收盘偏向高点为正，偏向低点为负
        hl_range = (high - low).replace(0, np.nan)
        direction = (close - low) / hl_range * 2 - 1  # [-1, 1]
        raw = expansion * direction
        return AdvancedSignalGenerator.cross_rank(raw)

    # ═══════════════════════════════════════════════════════
    # 跨资产因子 (Cross-Asset)
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def relative_strength_vs_spy(returns, spy_returns, window=20):
        """Relative strength versus SPY - pure single-name alpha momentum.

        Rolling alpha = cumulative(stock_ret - spy_ret)
        Momentum in excess return once market beta is stripped out.
        """
        # spy_returns 是 Series，广播到 DataFrame
        excess = returns.sub(spy_returns, axis=0)
        rolling_alpha = excess.rolling(window).sum()
        return AdvancedSignalGenerator.cross_rank(rolling_alpha)

    @staticmethod
    def beta_adjusted_momentum(returns, spy_returns, window=60):
        """Beta-adjusted momentum - residual momentum.

        Estimates a rolling beta, then accumulates the residual returns.
        Residual momentum predicts better than raw momentum.
        """
        result = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
        for col in returns.columns:
            r = returns[col].dropna()
            s = spy_returns.reindex(r.index).dropna()
            idx = r.index.intersection(s.index)
            if len(idx) < window:
                continue
            r_s = r.reindex(idx)
            s_s = s.reindex(idx)
            # 滚动beta
            cov = r_s.rolling(window).cov(s_s)
            var_spy = s_s.rolling(window).var()
            beta = cov / var_spy.replace(0, np.nan)
            # 残差收益
            residual = r_s - beta * s_s
            # 累计残差动量
            result.loc[idx, col] = residual.rolling(window).sum()
        return AdvancedSignalGenerator.cross_rank(result)

    @staticmethod
    def sector_relative_strength(returns, sector_returns, window=20):
        """Sector-relative strength.

        Rolling accumulation of (stock_ret - sector_ret), which removes sector
        beta and isolates within-sector alpha.
        sector_returns: DataFrame shaped like returns, each column carrying the
        return of the sector ETF that stock belongs to.
        """
        excess = returns - sector_returns
        rolling_rs = excess.rolling(window).sum()
        return AdvancedSignalGenerator.cross_rank(rolling_rs)

    # ═══════════════════════════════════════════════════════
    # 组合信号
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def prepare_advanced_signals(data_dict):
        """Compute every advanced factor and return them as dict[str, DataFrame].

        Parameters
        ----------
        data_dict : dict
            Required: close, open, high, low, volume
            Optional: trade_count, vwap, spy (Series of SPY closes),
                  sector_returns (DataFrame)

        Returns
        -------
        dict[str, DataFrame] - one cross_rank-ed DataFrame per factor name
        """
        G = AdvancedSignalGenerator
        close = data_dict['close']
        opn = data_dict['open']
        high = data_dict['high']
        low = data_dict['low']
        volume = data_dict['volume']
        returns = close.pct_change()
        prev_close = close.shift(1)

        signals = {}

        # ── 微观结构 ──
        signals['kyle_lambda'] = G.kyle_lambda(returns, volume)
        signals['order_flow_imbalance'] = G.order_flow_imbalance(close, high, low, volume)
        signals['realized_spread'] = G.realized_spread(high, low, close)
        signals['tick_size_clustering'] = G.tick_size_clustering(close)

        # ── 日内模式 ──
        signals['overnight_return'] = G.overnight_return(opn, prev_close)
        signals['intraday_return'] = G.intraday_return(close, opn)
        signals['overnight_intraday_divergence'] = G.overnight_intraday_divergence(opn, close)

        # ── 成交量模式 ──
        signals['volume_autocorrelation'] = G.volume_autocorrelation(volume)
        signals['price_volume_correlation'] = G.price_volume_correlation(returns, volume)
        signals['abnormal_volume_return'] = G.abnormal_volume_return(returns, volume)

        # ── 波幅与波动率 ──
        signals['garman_klass_vol'] = G.garman_klass_vol(opn, high, low, close)
        signals['parkinson_vol'] = G.parkinson_vol(high, low)
        signals['vol_of_vol'] = G.vol_of_vol(returns)
        signals['range_expansion'] = G.range_expansion(high, low, close)

        # ── 跨资产（需要SPY数据）──
        if 'spy' in data_dict:
            spy_close = data_dict['spy']
            spy_ret = spy_close.pct_change()
            signals['relative_strength_vs_spy'] = G.relative_strength_vs_spy(
                returns, spy_ret)
            signals['beta_adjusted_momentum'] = G.beta_adjusted_momentum(
                returns, spy_ret)

        if 'sector_returns' in data_dict:
            signals['sector_relative_strength'] = G.sector_relative_strength(
                returns, data_dict['sector_returns'])

        return signals

    @staticmethod
    def build_advanced_signal(data_dict, weights=None):
        """Blend every advanced factor by weight into a single composite signal.

        Parameters
        ----------
        data_dict : dict - as for prepare_advanced_signals
        weights : dict[str, float] | None
            Factor name -> weight. None means equal weights.

        Returns
        -------
        DataFrame - composite signal in [-1, 1]
        """
        G = AdvancedSignalGenerator
        signals = G.prepare_advanced_signals(data_dict)

        if weights is None:
            weights = {k: 1.0 for k in signals}

        # 归一化权重
        total_w = sum(weights.get(k, 0) for k in signals)
        if total_w == 0:
            raise ValueError("权重之和为零，无法构建组合信号")

        composite = None
        for name, sig in signals.items():
            w = weights.get(name, 0) / total_w
            if w == 0:
                continue
            if composite is None:
                composite = sig * w
            else:
                composite = composite.add(sig * w, fill_value=0)

        if composite is None:
            raise ValueError("无有效信号可组合")

        return G.cross_rank(composite)
