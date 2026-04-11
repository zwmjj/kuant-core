"""Alpha因子信号库 — 基于学术文献与量化基金研究的多因子模型"""
import numpy as np
import pandas as pd


class AlphaSignalGenerator:
    """Alpha因子信号的静态方法集合

    灵感来源：Amihud, Roll, Da-Gurun-Warachka, Moskowitz-Ooi-Pedersen 等学术研究。
    所有输入均为 pandas DataFrame，index=日期，columns=股票代码。
    所有信号输出均经过 cross_sectional_rank 处理，值域 [-1, 1]。
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
        """截面排名，映射到 [-1, 1]"""
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
        """Amihud非流动性自相关 — 信息不对称代理

        Amihud (2002) 非流动性 = |ret| / dollar_volume。
        计算其 window 日滚动自相关；持续的非流动性暗示信息不对称，应回避。
        信号取负值：持续高非流动性 → 低排名。
        """
        illiq = returns.abs() / dollar_volume.replace(0, np.nan)
        illiq_lag = illiq.shift(1)

        def _rolling_autocorr(s, w):
            return s.rolling(w, min_periods=w // 2).corr(illiq_lag[s.name])

        raw = illiq.apply(lambda col: _rolling_autocorr(col, window))
        return AlphaSignalGenerator.cross_sectional_rank(-raw)

    @staticmethod
    def roll_spread(close, window=20):
        """Roll (1984) 有效价差 — 流动性度量

        有效价差 = 2 * sqrt(-cov(ret_t, ret_{t-1}))。
        协方差为正时价差设为0。低价差 = 流动性好 = 有利。
        信号取负值：低价差 → 高排名。
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
        """换手率因子 — 关注度与流动性

        换手率 = volume / 流通股本代理。
        取 window 日均值；高换手率 = 高关注度。
        """
        turnover = volume / shares_outstanding_proxy.replace(0, np.nan)
        raw = turnover.rolling(window, min_periods=window // 2).mean()
        return AlphaSignalGenerator.cross_sectional_rank(raw)

    @staticmethod
    def liquidity_shock(dollar_volume, window=5, baseline=60):
        """流动性冲击因子 — 异常成交量变化

        (短期均量 / 长期均量)。突然的流动性变化 = 事件驱动信号。
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
        """更高的高点计数 — 多头结构

        过去 window 日中，high > 前一日 high 的天数。
        更多 higher-highs = 上升趋势结构 = 看多。
        """
        hh = (high > high.shift(1)).astype(float)
        raw = hh.rolling(window, min_periods=window // 2).sum()
        return AlphaSignalGenerator.cross_sectional_rank(raw)

    @staticmethod
    def lower_lows(low, window=20):
        """更低的低点计数 — 空头结构

        过去 window 日中，low < 前一日 low 的天数。
        更多 lower-lows = 下跌趋势 = 看空。信号取负值。
        """
        ll = (low < low.shift(1)).astype(float)
        raw = ll.rolling(window, min_periods=window // 2).sum()
        return AlphaSignalGenerator.cross_sectional_rank(-raw)

    @staticmethod
    def price_acceleration(close, short=5, long=20):
        """价格加速度因子 — 动量二阶导

        短期动量 - 长期动量 = 加速度。
        加速度为正 = 趋势正在加强 = 看多。
        Da, Gurun, Warachka (2014) 启发。
        """
        mom_short = close.pct_change(short)
        mom_long = close.pct_change(long)
        raw = mom_short - mom_long * (short / long)
        return AlphaSignalGenerator.cross_sectional_rank(raw)

    @staticmethod
    def support_distance(close, low, window=60):
        """距支撑位距离 — 超买/超卖度量

        (close - window日最低价) / close。
        距离远 = 延伸过大 = 谨慎。信号取负值。
        """
        rolling_low = low.rolling(window, min_periods=window // 2).min()
        raw = (close - rolling_low) / close.replace(0, np.nan)
        return AlphaSignalGenerator.cross_sectional_rank(-raw)

    @staticmethod
    def resistance_distance(close, high, window=60):
        """距阻力位距离 — 上方压力度量

        (window日最高价 - close) / close。
        接近阻力位（距离小）= 卖出压力大。信号取负值。
        """
        rolling_high = high.rolling(window, min_periods=window // 2).max()
        raw = (rolling_high - close) / close.replace(0, np.nan)
        return AlphaSignalGenerator.cross_sectional_rank(-raw)

    @staticmethod
    def candlestick_body(open_, close, high, low):
        """K线实体比例因子 — 交易信念强度

        |close - open| / (high - low)。
        大实体 = 方向性信念强 = 趋势确认。
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
        """Hurst指数代理 — R/S分析

        H > 0.5 = 趋势持续性；H < 0.5 = 均值回复。
        使用简化的 R/S 方法估算。信号：H - 0.5（正=趋势，负=回复）。
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
        """Shannon熵 — 收益分布可预测性

        将收益分箱后计算Shannon熵。低熵 = 可预测 = 有利。
        信号取负值：低熵 → 高排名。
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
        """尾部风险因子 — 预期亏损 (Expected Shortfall, 5%)

        过去 window 日收益的5%分位数以下均值（CVaR）。
        尾部风险小 = 安全 = 有利。信号取负值的负值（ES本身为负，
        更大的ES绝对值=更危险）。
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
        """收益自相关因子

        正自相关 = 动量特征；负自相关 = 反转特征。
        Cutler, Poterba, Summers (1989)。
        """
        ret_lag = returns.shift(lag)

        def _autocorr_col(col):
            return col.rolling(window, min_periods=window // 2).corr(ret_lag[col.name])

        raw = returns.apply(_autocorr_col)
        return AlphaSignalGenerator.cross_sectional_rank(raw)

    @staticmethod
    def variance_ratio(returns, short=5, long=20):
        """方差比因子 — 随机游走检验

        VR = Var(long_ret) / (Var(short_ret) * long/short)。
        VR > 1 = 正自相关（动量）；VR < 1 = 负自相关（反转）。
        Lo & MacKinlay (1988)。
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
        """信息延迟因子 — 价格发现效率

        Hou & Moskowitz (2005)。
        比较含滞后市场收益回归的 R² 与仅同期回归的 R²。
        高延迟 = 信息反应慢 = 未来可能补涨/补跌。
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
        """共振因子 — 截面平均相关性

        个股与截面均值的滚动相关性。
        低共振 = 特质性强 = alpha来源。信号取负值。
        """
        market_avg = returns.mean(axis=1)

        def _co_col(col):
            return col.rolling(window, min_periods=window // 2).corr(market_avg)

        raw = returns.apply(_co_col)
        return AlphaSignalGenerator.cross_sectional_rank(-raw)

    @staticmethod
    def lead_lag(returns, market_returns, window=20):
        """领先滞后因子 — 信息领先度

        个股当期收益与市场未来1日收益的相关性。
        高相关 = 该股领先于市场 = 信息优势。
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
        """温水煮蛙因子 — 连续性动量分解

        Da, Gurun, Warachka (2014)。
        将动量分解为连续小幅上涨vs少数大幅跳跃。
        连续性动量信号更强 = 投资者关注度不足。
        FROG = sign(cum_ret) * (pct_positive - pct_negative)。
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
        """最大单日收益因子 — 彩票股效应

        Bali, Cakici, Whitelaw (2011)。
        window 日内最大单日收益。
        高 max_ret = 彩票型股票 = 未来预期回报低。信号取负值。
        """
        raw = returns.rolling(window, min_periods=window // 2).max()
        return AlphaSignalGenerator.cross_sectional_rank(-raw)

    @staticmethod
    def time_series_momentum(returns, window=252):
        """时间序列动量 — TSMOM

        Moskowitz, Ooi, Pedersen (2012)。
        个股自身的过去 window 日累计收益。
        与截面动量不同，TSMOM关注自身趋势而非相对排名。
        """
        raw = returns.rolling(window, min_periods=window // 2).sum()
        return AlphaSignalGenerator.cross_sectional_rank(raw)

    @staticmethod
    def momentum_crash_filter(returns, market_returns, window=60):
        """动量崩溃过滤器 — 条件动量

        Daniel & Moskowitz (2016)。
        在市场大幅下跌后，动量策略容易崩溃。
        信号 = momentum * (1 - I(market_in_crash))。
        市场在窗口期内回撤超过-10%时，动量信号衰减。
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
        """构建复合Alpha信号 — 按类别或全部因子等权合成

        参数
        ----------
        data_dict : dict
            各因子名称 → 已排名的 DataFrame 信号。
        category : str
            因子类别名（'liquidity', 'price_pattern', 'statistical',
            'information_flow', 'momentum_refinement'）或 'all'。
        weights : dict, optional
            因子名称 → 权重。默认等权。

        返回
        ------
        DataFrame : 复合信号，值域 [-1, 1]。
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
