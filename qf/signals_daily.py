"""日频因子信号库 — 基于Alpaca日线OHLCV+VWAP+trade_count数据"""
import numpy as np
import pandas as pd


class DailySignalGenerator:
    """日频因子信号的静态方法集合

    所有输入均为 pandas DataFrame，index=日期，columns=股票代码。
    所有信号输出均经过 cross_sectional_rank 处理，值域 [-1, 1]。
    """

    # ── 工具 ──

    @staticmethod
    def cross_sectional_rank(signal):
        """截面排名，映射到 [-1, 1]"""
        def rank_row(row):
            valid = row.dropna()
            if len(valid) < 10:
                return row * np.nan
            return (valid.rank(pct=True) * 2 - 1).reindex(row.index)
        return signal.apply(rank_row, axis=1)

    # ── 短期动量 (1-5日) ──

    @staticmethod
    def momentum_5d(prices):
        """5日价格动量 — 短期趋势跟随
        (price / price_5d_ago) - 1，截面排名后输出
        """
        raw = prices / prices.shift(5) - 1
        return DailySignalGenerator.cross_sectional_rank(raw)

    @staticmethod
    def momentum_20d(prices):
        """20日（约1个月）价格动量 — 中短期趋势
        (price / price_20d_ago) - 1，截面排名后输出
        """
        raw = prices / prices.shift(20) - 1
        return DailySignalGenerator.cross_sectional_rank(raw)

    @staticmethod
    def momentum_reversal_5d(returns):
        """5日反转信号 — 逆势因子
        过去5日累计收益的负值（短期超卖反弹逻辑）
        """
        cum_5d = returns.rolling(5).sum()
        raw = -cum_5d
        return DailySignalGenerator.cross_sectional_rank(raw)

    # ── 成交量信号 ──

    @staticmethod
    def volume_surge(volume, lookback=20):
        """成交量突增 — 今日成交量 / 20日均量
        高量突增往往预示趋势变化或机构介入
        """
        avg_vol = volume.rolling(lookback).mean()
        raw = volume / avg_vol.replace(0, np.nan)
        return DailySignalGenerator.cross_sectional_rank(raw)

    @staticmethod
    def dollar_volume_rank(volume, prices):
        """美元成交额排名 — 流动性信号
        volume × close，高流动性通常与机构关注度正相关
        """
        dollar_vol = volume * prices
        return DailySignalGenerator.cross_sectional_rank(dollar_vol)

    @staticmethod
    def volume_price_trend(prices, volume):
        """量价趋势 (OBV-style) — 累计方向成交量
        price上涨日累加volume，下跌日累减，衡量买卖力量平衡
        取20日滚动累计避免无限累加
        """
        price_chg = prices.diff()
        direction = np.sign(price_chg)
        directed_vol = direction * volume
        raw = directed_vol.rolling(20).sum()
        return DailySignalGenerator.cross_sectional_rank(raw)

    @staticmethod
    def volume_price_divergence(prices, volume, lookback=20):
        """量价背离因子 — 价格与成交量的排名相关性变化

        核心逻辑:
        - 价格创新高但成交量萎缩 → 上涨动能不足，看跌信号 (负值)
        - 价格创新低但成交量放大 → 恐慌性抛售/底部放量，看涨信号 (正值)

        计算方法:
        1. 在 lookback(20日) 滚动窗口内，对每只股票的价格和成交量
           分别做时序排名 (rank)
        2. 计算两组排名的 Spearman 相关系数
        3. 取负值: 正相关(量价齐升)是常态，负相关(量价背离)才有信息量
           背离越严重 → 信号绝对值越大
        4. 截面排名归一化到 [-1, 1]

        Parameters
        ----------
        prices : pd.DataFrame
            收盘价，index=日期，columns=股票代码
        volume : pd.DataFrame
            成交量，index=日期，columns=股票代码
        lookback : int
            滚动窗口长度，默认20日（约1个月）

        Returns
        -------
        pd.DataFrame
            量价背离信号，index=日期，columns=股票代码，值域 [-1, 1]
            正值 = 看涨(底部放量型背离)，负值 = 看跌(顶部缩量型背离)
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
        """VWAP偏离度 — 机构买卖压力指标
        (close - vwap) / vwap
        close > vwap 说明收盘前有买压（机构扫货），反之为卖压
        """
        raw = (close - vwap) / vwap.replace(0, np.nan)
        return DailySignalGenerator.cross_sectional_rank(raw)

    @staticmethod
    def vwap_reversion(close, vwap, lookback=5):
        """VWAP累计偏离 — N日VWAP偏离均值回复信号
        累计偏离越大，回复概率越高（取负值做反转）
        """
        deviation = (close - vwap) / vwap.replace(0, np.nan)
        cum_dev = deviation.rolling(lookback).sum()
        raw = -cum_dev  # 累计偏离大 → 预期回落
        return DailySignalGenerator.cross_sectional_rank(raw)

    # ── 波动率 (日频) ──

    @staticmethod
    def realized_vol_20d(returns):
        """20日已实现波动率（年化） — 低波动异象
        低波动股票长期风险调整收益更优 (Baker, Bradley, Wurgler 2011)
        取负值：低波动 → 高信号
        """
        raw = returns.rolling(20, min_periods=15).std() * np.sqrt(252)
        return DailySignalGenerator.cross_sectional_rank(-raw)

    @staticmethod
    def vol_breakout(prices, lookback=20):
        """波动率突破 — 当日振幅 vs 历史平均振幅
        需要high和low，但这里用prices(close)近似：用rolling max-min / close
        检测突破行情，高值表示可能的趋势启动
        """
        # 使用收盘价的日收益绝对值作为振幅代理
        daily_range = prices.pct_change().abs()
        avg_range = daily_range.rolling(lookback).mean()
        raw = daily_range / avg_range.replace(0, np.nan)
        return DailySignalGenerator.cross_sectional_rank(raw)

    @staticmethod
    def vol_breakout_hl(high, low, close, lookback=20):
        """波动率突破 (高低价版) — (high-low)/close vs 历史均值
        检测当日振幅是否异常放大，用于识别breakout
        """
        intraday_range = (high - low) / close.replace(0, np.nan)
        avg_range = intraday_range.rolling(lookback).mean()
        raw = intraday_range / avg_range.replace(0, np.nan)
        return DailySignalGenerator.cross_sectional_rank(raw)

    @staticmethod
    def overnight_gap(open_prices, prev_close):
        """隔夜跳空 — (open / prev_close - 1)
        反映隔夜信息冲击，正跳空可能是利好，但也可能过度反应
        输入prev_close = close.shift(1)
        """
        raw = open_prices / prev_close.replace(0, np.nan) - 1
        return DailySignalGenerator.cross_sectional_rank(raw)

    # ── 微观结构 ──

    @staticmethod
    def amihud_illiquidity(returns, dollar_volume):
        """Amihud非流动性 — |return| / 美元成交额, Amihud (2002)
        高值 = 低流动性 = 流动性溢价（取负值：低非流动性更优）
        取20日滚动均值平滑
        """
        illiq = returns.abs() / dollar_volume.replace(0, np.nan)
        raw = illiq.rolling(20, min_periods=10).mean()
        return DailySignalGenerator.cross_sectional_rank(-raw)

    @staticmethod
    def trade_intensity(trade_count, volume):
        """单笔成交量 — volume / trade_count（机构足迹）
        大单笔成交量暗示机构参与，通常是信息驱动的交易
        """
        raw = volume / trade_count.replace(0, np.nan)
        return DailySignalGenerator.cross_sectional_rank(raw)

    @staticmethod
    def high_low_spread(high, low, close):
        """高低价差 — (high - low) / close，日内价差代理
        Corwin & Schultz (2012) 高低价差估计，衡量交易成本
        取负值：低价差（低交易成本）更优
        """
        raw = (high - low) / close.replace(0, np.nan)
        return DailySignalGenerator.cross_sectional_rank(-raw)

    # ── 复合信号 ──

    @staticmethod
    def build_daily_signal(data_dict, weights=None):
        """构建日频复合因子信号

        Parameters
        ----------
        data_dict : dict
            必须包含 'close', 'volume' 键；可选 'open', 'high', 'low',
            'vwap', 'trade_count'
        weights : dict or None
            因子名 → 权重的字典。默认等权使用所有可计算的因子。
            例: {'momentum_5d': 0.2, 'vwap_deviation': 0.3, 'volume_surge': 0.5}

        Returns
        -------
        pd.DataFrame
            复合信号，date × symbol，值域 [-1, 1]
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
    """便捷函数 — 从Alpaca日线数据一次性计算全部日频信号

    Parameters
    ----------
    alpaca_data_dict : dict
        键为 'close', 'open', 'high', 'low', 'volume', 'vwap', 'trade_count'
        值为 DataFrame (date × symbol)

    Returns
    -------
    dict
        信号名 → DataFrame 的字典，每个DataFrame为 date × symbol，值域 [-1, 1]
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
