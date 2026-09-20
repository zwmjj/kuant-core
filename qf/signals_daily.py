"""Daily factor signal library — built on Alpaca daily OHLCV + VWAP + trade_count data"""
import numpy as np
import pandas as pd


class DailySignalGenerator:
    """Collection of static methods for daily factor signals

    All inputs are pandas DataFrames with index=date, columns=stock code.
    All signal outputs pass through cross_sectional_rank and are bounded to [-1, 1].
    """

    # ── 工具 ──

    @staticmethod
    def cross_sectional_rank(signal):
        """Cross-sectional rank mapped to [-1, 1]"""
        def rank_row(row):
            valid = row.dropna()
            if len(valid) < 10:
                return row * np.nan
            return (valid.rank(pct=True) * 2 - 1).reindex(row.index)
        return signal.apply(rank_row, axis=1)

    # ── 短期动量 (1-5日) ──

    @staticmethod
    def momentum_5d(prices):
        """5-day price momentum — short-term trend following
        (price / price_5d_ago) - 1, output after cross-sectional ranking
        """
        raw = prices / prices.shift(5) - 1
        return DailySignalGenerator.cross_sectional_rank(raw)

    @staticmethod
    def momentum_20d(prices):
        """20-day (roughly one month) price momentum — medium-short-term trend
        (price / price_20d_ago) - 1, output after cross-sectional ranking
        """
        raw = prices / prices.shift(20) - 1
        return DailySignalGenerator.cross_sectional_rank(raw)

    @staticmethod
    def momentum_reversal_5d(returns):
        """5-day reversal signal — contrarian factor
        Negative of the past 5-day cumulative return (short-term oversold bounce logic)
        """
        cum_5d = returns.rolling(5).sum()
        raw = -cum_5d
        return DailySignalGenerator.cross_sectional_rank(raw)

    # ── 成交量信号 ──

    @staticmethod
    def volume_surge(volume, lookback=20):
        """Volume surge — today's volume / 20-day average volume
        A volume spike often precedes a trend change or institutional participation
        """
        avg_vol = volume.rolling(lookback).mean()
        raw = volume / avg_vol.replace(0, np.nan)
        return DailySignalGenerator.cross_sectional_rank(raw)

    @staticmethod
    def dollar_volume_rank(volume, prices):
        """Dollar volume rank — liquidity signal
        volume x close; high liquidity is generally positively correlated with institutional attention
        """
        dollar_vol = volume * prices
        return DailySignalGenerator.cross_sectional_rank(dollar_vol)

    @staticmethod
    def volume_price_trend(prices, volume):
        """Volume-price trend (OBV-style) — cumulative signed volume
        Adds volume on up days and subtracts it on down days to measure the balance of buying
        and selling pressure. Uses a 20-day rolling sum to avoid unbounded accumulation.
        """
        price_chg = prices.diff()
        direction = np.sign(price_chg)
        directed_vol = direction * volume
        raw = directed_vol.rolling(20).sum()
        return DailySignalGenerator.cross_sectional_rank(raw)

    @staticmethod
    def volume_price_divergence(prices, volume, lookback=20):
        """Volume-price divergence factor — change in the rank correlation between price and volume

        Core logic:
        - Price makes a new high on shrinking volume -> weak upside momentum, bearish signal (negative)
        - Price makes a new low on expanding volume -> capitulation / bottom volume spike, bullish signal (positive)

        Method:
        1. Within a rolling lookback window (20 days), time-series rank each stock's price
           and volume separately
        2. Compute the Spearman correlation between the two rank series
        3. Negate it: positive correlation (price and volume rising together) is the norm, so only
           negative correlation (divergence) carries information
           The stronger the divergence, the larger the absolute signal
        4. Cross-sectionally rank and normalize to [-1, 1]

        Parameters
        ----------
        prices : pd.DataFrame
            Close prices, index=date, columns=stock code
        volume : pd.DataFrame
            Trading volume, index=date, columns=stock code
        lookback : int
            Rolling window length, default 20 days (roughly one month)

        Returns
        -------
        pd.DataFrame
            Volume-price divergence signal, index=date, columns=stock code, bounded to [-1, 1]
            Positive = bullish (bottom volume-spike divergence), negative = bearish (top volume-dryup divergence)
        """
        # 滚动窗口内价格排名与成交量排名的 Spearman 相关系数
        # 使用 rolling apply 逐列计算
        def _rolling_rank_corr(price_col, vol_col, window):
            """对单只股票计算滚动排名相关性"""
            result = pd.Series(np.nan, index=price_col.index)
            for i in range(window - 1, len(price_col)):
                p_window = price_col.iloc[i - window + 1: i + 1]
                v_window = vol_col.iloc[i - window + 1: i + 1]
                # 至少需要 window * 0.75 个有效数据点
                valid_mask = p_window.notna() & v_window.notna()
                if valid_mask.sum() < max(10, window * 0.75):
                    continue
                # Spearman 排名相关 = Pearson(rank(price), rank(volume))
                p_rank = p_window[valid_mask].rank()
                v_rank = v_window[valid_mask].rank()
                corr = p_rank.corr(v_rank)
                result.iloc[i] = corr
            return result

        # 逐列计算滚动排名相关性
        rank_corr = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
        for ticker in prices.columns:
            rank_corr[ticker] = _rolling_rank_corr(
                prices[ticker], volume[ticker], lookback
            )

        # 结合价格趋势方向，赋予背离正确的多空含义:
        #   上涨趋势 + rank_corr正(量价齐升) → 趋势健康(看涨) ✓
        #   上涨趋势 + rank_corr负(顶部缩量) → 背离看跌 ✓
        #   下跌趋势 + rank_corr负(底部放量) → 翻转信号后变为看涨 ✓
        #   下跌趋势 + rank_corr正(量价齐跌) → 翻转信号后变为看跌(持续下跌) ✓
        price_trend = prices / prices.shift(lookback) - 1  # 20日收益率
        raw = rank_corr.copy()
        # 下跌趋势中翻转信号: 背离(负corr) = 底部放量 = 看涨
        downtrend = price_trend < 0
        raw[downtrend] = -raw[downtrend]

        return DailySignalGenerator.cross_sectional_rank(raw)

    # ── VWAP信号 ──

    @staticmethod
    def vwap_deviation(close, vwap):
        """VWAP deviation — institutional buy/sell pressure indicator
        (close - vwap) / vwap
        close > vwap indicates buying pressure into the close (institutional accumulation); otherwise selling pressure
        """
        raw = (close - vwap) / vwap.replace(0, np.nan)
        return DailySignalGenerator.cross_sectional_rank(raw)

    @staticmethod
    def vwap_reversion(close, vwap, lookback=5):
        """Cumulative VWAP deviation — N-day mean-reversion signal on VWAP deviation
        The larger the cumulative deviation, the higher the probability of reversion (negated for reversal)
        """
        deviation = (close - vwap) / vwap.replace(0, np.nan)
        cum_dev = deviation.rolling(lookback).sum()
        raw = -cum_dev  # 累计偏离大 → 预期回落
        return DailySignalGenerator.cross_sectional_rank(raw)

    # ── 波动率 (日频) ──

    @staticmethod
    def realized_vol_20d(returns):
        """20-day realized volatility (annualized) — low-volatility anomaly
        Low-volatility stocks deliver better long-run risk-adjusted returns (Baker, Bradley, Wurgler 2011)
        Negated: low volatility -> high signal
        """
        raw = returns.rolling(20, min_periods=15).std() * np.sqrt(252)
        return DailySignalGenerator.cross_sectional_rank(-raw)

    @staticmethod
    def vol_breakout(prices, lookback=20):
        """Volatility breakout — today's range vs the historical average range
        Ideally needs high and low, but approximates them from prices (close): rolling max-min / close
        Detects breakouts; high values suggest a trend may be starting
        """
        # 使用收盘价的日收益绝对值作为振幅代理
        daily_range = prices.pct_change().abs()
        avg_range = daily_range.rolling(lookback).mean()
        raw = daily_range / avg_range.replace(0, np.nan)
        return DailySignalGenerator.cross_sectional_rank(raw)

    @staticmethod
    def vol_breakout_hl(high, low, close, lookback=20):
        """Volatility breakout (high-low version) — (high-low)/close vs its historical mean
        Detects abnormal expansion in the day's range, used to identify breakouts
        """
        intraday_range = (high - low) / close.replace(0, np.nan)
        avg_range = intraday_range.rolling(lookback).mean()
        raw = intraday_range / avg_range.replace(0, np.nan)
        return DailySignalGenerator.cross_sectional_rank(raw)

    @staticmethod
    def overnight_gap(open_prices, prev_close):
        """Overnight gap — (open / prev_close - 1)
        Captures overnight information shocks; a gap up may reflect good news but can also be an overreaction
        Pass prev_close = close.shift(1)
        """
        raw = open_prices / prev_close.replace(0, np.nan) - 1
        return DailySignalGenerator.cross_sectional_rank(raw)

    # ── 微观结构 ──

    @staticmethod
    def amihud_illiquidity(returns, dollar_volume):
        """Amihud illiquidity — |return| / dollar volume, Amihud (2002)
        High values = low liquidity = liquidity premium (negated: lower illiquidity is preferred)
        Smoothed with a 20-day rolling mean
        """
        illiq = returns.abs() / dollar_volume.replace(0, np.nan)
        raw = illiq.rolling(20, min_periods=10).mean()
        return DailySignalGenerator.cross_sectional_rank(-raw)

    @staticmethod
    def trade_intensity(trade_count, volume):
        """Average trade size — volume / trade_count (institutional footprint)
        Large average trade size suggests institutional participation, typically information-driven trading
        """
        raw = volume / trade_count.replace(0, np.nan)
        return DailySignalGenerator.cross_sectional_rank(raw)

    @staticmethod
    def high_low_spread(high, low, close):
        """High-low spread — (high - low) / close, an intraday spread proxy
        Corwin & Schultz (2012) high-low spread estimator, a measure of trading cost
        Negated: a narrow spread (low trading cost) is preferred
        """
        raw = (high - low) / close.replace(0, np.nan)
        return DailySignalGenerator.cross_sectional_rank(-raw)

    # ── 复合信号 ──

    @staticmethod
    def build_daily_signal(data_dict, weights=None):
        """Build the daily composite factor signal

        Parameters
        ----------
        data_dict : dict
            Must contain the 'close' and 'volume' keys; 'open', 'high', 'low',
            'vwap' and 'trade_count' are optional
        weights : dict or None
            Mapping of factor name → weight. Defaults to equal weights across every
            factor that can be computed.
            e.g. {'momentum_5d': 0.2, 'vwap_deviation': 0.3, 'volume_surge': 0.5}

        Returns
        -------
        pd.DataFrame
            Composite signal, date x symbol, bounded to [-1, 1]
        """
        sg = DailySignalGenerator
        signals = {}

        close = data_dict.get('close')
        volume = data_dict.get('volume')
        high = data_dict.get('high')
        low = data_dict.get('low')
        open_prices = data_dict.get('open')
        vwap = data_dict.get('vwap')
        trade_count = data_dict.get('trade_count')

        if close is None or volume is None:
            raise ValueError("data_dict 必须包含 'close' 和 'volume'")

        returns = close.pct_change()
        dollar_volume = close * volume

        # 动量类
        signals['momentum_5d'] = sg.momentum_5d(close)
        signals['momentum_20d'] = sg.momentum_20d(close)
        signals['momentum_reversal_5d'] = sg.momentum_reversal_5d(returns)

        # 成交量类
        signals['volume_surge'] = sg.volume_surge(volume)
        signals['dollar_volume_rank'] = sg.dollar_volume_rank(volume, close)
        signals['volume_price_trend'] = sg.volume_price_trend(close, volume)
        signals['volume_price_divergence'] = sg.volume_price_divergence(close, volume)

        # VWAP类
        if vwap is not None:
            signals['vwap_deviation'] = sg.vwap_deviation(close, vwap)
            signals['vwap_reversion'] = sg.vwap_reversion(close, vwap)

        # 波动率类
        signals['realized_vol_20d'] = sg.realized_vol_20d(returns)
        if high is not None and low is not None:
            signals['vol_breakout'] = sg.vol_breakout_hl(high, low, close)
        else:
            signals['vol_breakout'] = sg.vol_breakout(close)
        if open_prices is not None:
            signals['overnight_gap'] = sg.overnight_gap(open_prices, close.shift(1))

        # 微观结构类
        signals['amihud_illiquidity'] = sg.amihud_illiquidity(returns, dollar_volume)
        if trade_count is not None:
            signals['trade_intensity'] = sg.trade_intensity(trade_count, volume)
        if high is not None and low is not None:
            signals['high_low_spread'] = sg.high_low_spread(high, low, close)

        # 过滤只使用有权重的因子
        if weights is None:
            weights = {k: 1.0 / len(signals) for k in signals}

        # 归一化权重
        used = {k: v for k, v in weights.items() if k in signals}
        if not used:
            raise ValueError("无可用因子，请检查 weights 和 data_dict")
        total_w = sum(used.values())
        used = {k: v / total_w for k, v in used.items()}

        # 加权求和
        composite = None
        for name, w in used.items():
            sig = signals[name].astype(float)
            term = sig * w
            if composite is None:
                composite = term
            else:
                # 对齐后相加，NaN处只用非NaN的部分
                composite = composite.add(term, fill_value=0)

        return sg.cross_sectional_rank(composite)


def prepare_daily_signals(alpaca_data_dict):
    """Convenience function — compute every daily signal from Alpaca daily bars in one call

    Parameters
    ----------
    alpaca_data_dict : dict
        Keys are 'close', 'open', 'high', 'low', 'volume', 'vwap', 'trade_count'
        Values are DataFrames (date x symbol)

    Returns
    -------
    dict
        Mapping of signal name → DataFrame, each date x symbol and bounded to [-1, 1]
    """
    sg = DailySignalGenerator

    close = alpaca_data_dict.get('close')
    if close is None:
        close = alpaca_data_dict.get('prices')
    volume = alpaca_data_dict['volume']
    high = alpaca_data_dict.get('high')
    low = alpaca_data_dict.get('low')
    open_prices = alpaca_data_dict.get('open')
    vwap = alpaca_data_dict.get('vwap')
    trade_count = alpaca_data_dict.get('trade_count')

    returns = close.pct_change()
    dollar_volume = close * volume

    results = {}

    # 短期动量
    results['momentum_5d'] = sg.momentum_5d(close)
    results['momentum_20d'] = sg.momentum_20d(close)
    results['momentum_reversal_5d'] = sg.momentum_reversal_5d(returns)

    # 成交量
    results['volume_surge'] = sg.volume_surge(volume)
    results['dollar_volume_rank'] = sg.dollar_volume_rank(volume, close)
    results['volume_price_trend'] = sg.volume_price_trend(close, volume)
    results['volume_price_divergence'] = sg.volume_price_divergence(close, volume)

    # VWAP
    if vwap is not None:
        results['vwap_deviation'] = sg.vwap_deviation(close, vwap)
        results['vwap_reversion'] = sg.vwap_reversion(close, vwap)

    # 波动率
    results['realized_vol_20d'] = sg.realized_vol_20d(returns)
    if high is not None and low is not None:
        results['vol_breakout'] = sg.vol_breakout_hl(high, low, close)
    else:
        results['vol_breakout'] = sg.vol_breakout(close)
    if open_prices is not None:
        results['overnight_gap'] = sg.overnight_gap(open_prices, close.shift(1))

    # 微观结构
    results['amihud_illiquidity'] = sg.amihud_illiquidity(returns, dollar_volume)
    if trade_count is not None:
        results['trade_intensity'] = sg.trade_intensity(trade_count, volume)
    if high is not None and low is not None:
        results['high_low_spread'] = sg.high_low_spread(high, low, close)

    # 复合信号
    results['composite'] = sg.build_daily_signal(alpaca_data_dict)

    return results
