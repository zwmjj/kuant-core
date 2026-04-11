"""宏观与跨资产因子信号库 — 利用宏观指标(黄金、原油、债券、波动率、行业ETF)进行股票选择"""
import numpy as np
import pandas as pd


# 所需宏观ETF列表
MACRO_ETFS = [
    'SPY', 'GLD', 'USO', 'DBC', 'TLT', 'IEF', 'HYG', 'UUP',
    'XLE', 'XLF', 'XLK', 'XLV', 'XLU', 'XLP',
]

SECTOR_ETFS = ['XLE', 'XLF', 'XLK', 'XLV', 'XLU', 'XLP']


class MacroSignalGenerator:
    """宏观与跨资产因子信号的静态方法集合

    stock_returns: DataFrame, index=日期, columns=股票代码, 值=日收益率
    macro_data中各列为宏观ETF的日收益率(或价格，视方法而定)。
    所有信号输出均经过 cross_sectional_rank 处理，值域 [-1, 1]。
    """

    # ══════════════════════════════════════════════════════════════
    #  工具
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def cross_sectional_rank(signal):
        """截面排名，映射到 [-1, 1]"""
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
        """利率敏感性 — 股票对TLT(长债)的滚动beta

        负beta表示利率敏感型成长股，加息周期中表现差。
        正beta表示与债券正相关的防御型股票。
        返回截面排名后的信号，高值=高利率敏感性。
        """
        betas = MacroSignalGenerator._rolling_beta(stock_returns, tlt_returns, window)
        return MacroSignalGenerator.cross_sectional_rank(betas)

    @staticmethod
    def real_yield_regime(gld_returns, tlt_returns, window=20):
        """实际收益率环境判断 — 黄金涨+债券跌=实际收益率上升

        返回 Series (T,)，值域 [-1, 1]:
          正值 = 实际收益率上升环境(不利于成长股)
          负值 = 实际收益率下降环境(有利于成长股)
        应配合 rate_sensitivity 使用：实际收益率上升时，避开高TLT-beta股票。
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
        """原油暴露 — 股票对USO(原油ETF)的滚动beta

        高beta = 能源相关敞口大。能源牛市中有利。
        """
        betas = MacroSignalGenerator._rolling_beta(stock_returns, uso_returns, window)
        return MacroSignalGenerator.cross_sectional_rank(betas)

    @staticmethod
    def gold_beta(stock_returns, gld_returns, window=60):
        """黄金暴露 — 股票对GLD(黄金ETF)的滚动beta

        高beta = 黄金对冲属性。避险环境中有利。
        """
        betas = MacroSignalGenerator._rolling_beta(stock_returns, gld_returns, window)
        return MacroSignalGenerator.cross_sectional_rank(betas)

    @staticmethod
    def commodity_regime_tilt(stock_returns, dbc_returns, window=20):
        """商品趋势择时倾斜 — 商品上涨趋势中偏好高商品beta股

        先判断DBC趋势方向，再计算股票对DBC的beta。
        商品上涨期：高beta得分高。商品下跌期：低beta得分高(信号翻转)。
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
        """下行beta — 仅用SPY下跌日计算的beta(VIX代理)

        高下行beta = 崩盘易损型股票。恐慌时跌幅更大。
        """
        # 仅保留SPY下跌日
        mask = spy_returns < 0
        spy_down = spy_returns.where(mask, np.nan)
        stock_down = stock_returns.where(mask, np.nan)

        betas = MacroSignalGenerator._rolling_beta(stock_down, spy_down, window)
        return MacroSignalGenerator.cross_sectional_rank(betas)

    @staticmethod
    def vol_regime_tilt(stock_returns, spy_returns, window=20):
        """波动率环境择时倾斜 — 高波动环境偏好低beta，低波动环境偏好高beta

        用SPY已实现波动率判断波动环境，结合个股beta做择时。
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
        """滚动相关性环境 — 个股与大盘的滚动相关系数

        高相关性环境下，只有低相关性(特异性)股票能提供分散化价值。
        返回: 负相关性排名 → 高值=低相关性=更有特异性价值。
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
        """行业动量 — 行业ETF的截面动量排名

        返回 DataFrame, index=日期, columns=行业ETF代码, 值=[-1,1]排名。
        通过个股对行业ETF的beta映射到个股层面。
        """
        cum_ret = sector_etf_returns.rolling(window, min_periods=window // 2).sum()
        return MacroSignalGenerator.cross_sectional_rank(cum_ret)

    @staticmethod
    def sector_mean_reversion(sector_etf_returns, window=5):
        """行业短期反转 — 短窗口行业ETF收益的反向信号

        过去5日涨幅最大的行业短期内倾向反转。
        """
        cum_ret = sector_etf_returns.rolling(window, min_periods=max(window // 2, 3)).sum()
        return MacroSignalGenerator.cross_sectional_rank(-cum_ret)

    @staticmethod
    def defensive_tilt(spy_returns, window=20):
        """防御性倾斜 — SPY趋势为负时偏好防御型股票

        返回 Series (T,)，值域 [-1, 1]:
          负值 = SPY下行趋势 → 应超配公用事业/必需消费/医疗(低beta)
          正值 = SPY上行趋势 → 可超配进攻型(高beta)
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
        """风险偏好指标 — SPY/GLD比值趋势

        SPY相对GLD走强 = risk-on → 偏好高beta股票
        SPY相对GLD走弱 = risk-off → 偏好低beta/防御股
        返回 Series (T,)，值域 [-1, 1]。
        """
        spread = spy_returns - gld_returns
        trend = spread.rolling(window, min_periods=window // 2).mean()
        trend_z = (trend - trend.rolling(252, min_periods=60).mean()) \
                  / trend.rolling(252, min_periods=60).std().replace(0, np.nan)
        return trend_z.clip(-3, 3) / 3

    @staticmethod
    def dollar_regime(uup_returns, window=20):
        """美元环境 — 强美元不利于跨国公司

        UUP上涨趋势 = 强美元 → 偏好内需型股票
        返回 Series (T,)，值域 [-1, 1]。正值=强美元环境。
        """
        trend = uup_returns.rolling(window, min_periods=window // 2).mean()
        trend_z = (trend - trend.rolling(252, min_periods=60).mean()) \
                  / trend.rolling(252, min_periods=60).std().replace(0, np.nan)
        return trend_z.clip(-3, 3) / 3

    @staticmethod
    def credit_spread_proxy(hyg_returns, tlt_returns, window=20):
        """信用利差代理 — HYG-TLT利差走势

        HYG相对TLT走弱 = 信用利差扩大 = risk-off
        返回 Series (T,)，值域 [-1, 1]。负值=risk-off(利差扩大)。
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
        """自动检测宏观环境并构建综合宏观信号

        参数:
            stock_returns: DataFrame (T x N), 个股日收益率
            macro_data: DataFrame, 列为各宏观ETF日收益率
            regime: 'auto'自动检测 | 'risk_on' | 'risk_off' | 'neutral'

        返回: DataFrame (T x N), 综合宏观择股信号 [-1, 1]

        自动环境检测逻辑:
          - risk_on: SPY趋势向上 + 信用利差收窄
          - risk_off: SPY趋势向下 + 信用利差扩大
          - neutral: 其他
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
        """计算所有宏观因子信号，返回字典

        参数:
            stock_returns: DataFrame (T x N), 个股日收益率
            macro_data: DataFrame, 列为各宏观ETF日收益率

        返回: dict[str, DataFrame/Series]
            键为信号名称，值为对应的信号DataFrame或Series。
            缺失的宏观数据对应的信号会被跳过。
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
