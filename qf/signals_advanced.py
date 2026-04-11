"""高级Alpha因子库 — 利用Alpaca微观结构/日内模式/波动率等未开发数据源"""
import numpy as np
import pandas as pd


class AdvancedSignalGenerator:
    """高级因子信号的静态方法集合

    所有输入均为 pandas DataFrame，index=日期，columns=股票代码。
    所有信号输出均经过 cross_rank 处理，值域 [-1, 1]。

    data_dict 结构:
        close, open, high, low, volume, trade_count, vwap,
        spy (SPY收盘价 Series), sector_returns (可选 DataFrame)
    """

    # ═══════════════════════════════════════════════════════
    # 工具
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def cross_rank(signal):
        """截面排名，映射到 [-1, 1]"""
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
        """Kyle Lambda — 价格冲击系数

        对 |returns| 与 volume 做滚动回归，斜率即为 lambda。
        高 lambda = 流动性差。信号取负值，偏好流动性好的标的。
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
        """订单流失衡 (OFI)

        买量估计 = volume * (close - low) / (high - low)
        卖量估计 = volume - buy_vol
        OFI = rolling_sum(buy_vol - sell_vol)
        正值 = 买压主导
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
        """实现价差指标

        (2*close - high - low) / close 的滚动均值。
        正值 = 收盘偏向最高价 = 看涨倾向。
        """
        raw = (2 * close - high - low) / close
        smoothed = raw.rolling(window).mean()
        return AdvancedSignalGenerator.cross_rank(smoothed)

    @staticmethod
    def tick_size_clustering(close, window=20):
        """价格聚集度 — 散户行为代理

        统计过去window日中收盘价落在整数($X.00)或半整数($X.50)的比例。
        高聚集度 = 散户主导（散户偏好整数报价）。信号取负值。
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
        """隔夜收益 — 机构定价信号

        overnight = open / prev_close - 1
        累计隔夜收益持续性强 = 机构在盘后/盘前建仓。
        prev_close 可传入 close.shift(1)。
        """
        raw = open_prices / prev_close - 1
        cum_on = raw.rolling(20).sum()
        return AdvancedSignalGenerator.cross_rank(cum_on)

    @staticmethod
    def intraday_return(close, open_prices):
        """日内收益 — 散户动量

        intraday = close / open - 1
        累计日内收益 = 散户资金流向。
        """
        raw = close / open_prices - 1
        cum_id = raw.rolling(20).sum()
        return AdvancedSignalGenerator.cross_rank(cum_id)

    @staticmethod
    def overnight_intraday_divergence(open_prices, close):
        """隔夜-日内背离因子

        当隔夜收益与日内收益方向相反时，跟随隔夜方向（更聪明的钱）。
        信号 = overnight_ret - intraday_ret（滚动求和）。
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
        """成交量自相关 — 机构执行计划检测

        滚动成交量与lag-1的相关系数。
        高自相关 = 机构按计划执行大单（TWAP/VWAP算法）。
        """
        vol_lag = volume.shift(1)
        corr = volume.rolling(window).corr(vol_lag)
        return AdvancedSignalGenerator.cross_rank(corr)

    @staticmethod
    def price_volume_correlation(returns, volume, window=20):
        """量价相关性

        正相关 = 趋势确认（放量上涨或放量下跌）。
        负相关 = 量价背离 = 潜在反转。
        信号直接用相关系数。
        """
        corr = returns.rolling(window).corr(volume)
        return AdvancedSignalGenerator.cross_rank(corr)

    @staticmethod
    def abnormal_volume_return(returns, volume, window=20):
        """异常成交量日收益 — 捕捉"聪明钱"信号

        异常成交量 = volume > 2 * rolling_mean(volume)
        只保留异常日收益，其余置零，然后累计。
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
        """Garman-Klass 波动率估计

        GK = 0.5*ln(H/L)^2 - (2ln2-1)*ln(C/O)^2
        比收盘价波动率更准确，利用了OHLC全部信息。
        信号取负值（低波动率溢价）。
        """
        log_hl = np.log(high / low)
        log_co = np.log(close / open_prices)
        gk_daily = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
        gk_vol = gk_daily.rolling(window).mean().apply(np.sqrt)
        raw = -gk_vol
        return AdvancedSignalGenerator.cross_rank(raw)

    @staticmethod
    def parkinson_vol(high, low, window=20):
        """Parkinson 波动率 — 基于日内极差

        PV = sqrt(1/(4*N*ln2) * sum(ln(H/L)^2))
        捕捉日内波动率，收盘价波动率会遗漏这部分信息。
        信号取负值。
        """
        log_hl_sq = np.log(high / low) ** 2
        pv = np.sqrt(log_hl_sq.rolling(window).mean() / (4 * np.log(2)))
        raw = -pv
        return AdvancedSignalGenerator.cross_rank(raw)

    @staticmethod
    def vol_of_vol(returns, inner=10, outer=60):
        """波动率的波动率 (VoV)

        先算inner窗口的滚动波动率，再算outer窗口的波动率标准差。
        高VoV = 不确定性大 = 看空信号。
        """
        inner_vol = returns.rolling(inner).std()
        vov = inner_vol.rolling(outer).std()
        raw = -vov
        return AdvancedSignalGenerator.cross_rank(raw)

    @staticmethod
    def range_expansion(high, low, close, window=20):
        """波幅扩张因子

        当日波幅 / 20日平均波幅。
        波幅扩张 = 突破信号。结合方向（收盘偏向）给出方向性信号。
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
        """相对SPY强度 — 纯个股alpha动量

        滚动alpha = cumulative(stock_ret - spy_ret)
        去除市场beta后的纯超额收益动量。
        """
        # spy_returns 是 Series，广播到 DataFrame
        excess = returns.sub(spy_returns, axis=0)
        rolling_alpha = excess.rolling(window).sum()
        return AdvancedSignalGenerator.cross_rank(rolling_alpha)

    @staticmethod
    def beta_adjusted_momentum(returns, spy_returns, window=60):
        """Beta调整动量 — 残差动量

        先估计滚动beta，再取残差收益的累计值。
        残差动量比原始动量有更强的预测力。
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
        """板块相对强度

        stock_ret - sector_ret 的滚动累计。
        剥离行业beta，捕捉行业内alpha。
        sector_returns: DataFrame，与returns同结构（每列对应该股票所属板块ETF收益）。
        """
        excess = returns - sector_returns
        rolling_rs = excess.rolling(window).sum()
        return AdvancedSignalGenerator.cross_rank(rolling_rs)

    # ═══════════════════════════════════════════════════════
    # 组合信号
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def prepare_advanced_signals(data_dict):
        """计算全部高级因子，返回 dict[str, DataFrame]

        Parameters
        ----------
        data_dict : dict
            必须包含: close, open, high, low, volume
            可选: trade_count, vwap, spy (SPY收盘价 Series),
                  sector_returns (DataFrame)

        Returns
        -------
        dict[str, DataFrame] — 每个因子名对应一个 cross_rank 后的 DataFrame
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
        """加权组合全部高级因子，输出综合信号 DataFrame

        Parameters
        ----------
        data_dict : dict — 同 prepare_advanced_signals
        weights : dict[str, float] | None
            因子名 -> 权重。None 则等权。

        Returns
        -------
        DataFrame — 综合信号，值域 [-1, 1]
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
