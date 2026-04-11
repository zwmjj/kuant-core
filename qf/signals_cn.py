"""A股特色因子信号库 — 北向资金、涨停板动量、龙虎榜机构热度"""
import numpy as np
import pandas as pd


class ChinaSignalGenerator:
    """A股特色因子信号的静态方法集合

    所有输入均为 pandas DataFrame，index=日期，columns=股票代码。
    所有信号输出均经过 cross_sectional_rank 处理，值域 [-1, 1]。

    因子列表:
        1. northbound_flow_factor  — 北向资金净流入因子
        2. limit_up_momentum       — 涨停板动量因子
        3. institutional_attention — 机构调研/龙虎榜热度因子
    """

    # ── 工具方法 ──

    @staticmethod
    def cross_sectional_rank(signal):
        """截面排名，映射到 [-1, 1]

        每日对所有股票进行百分比排名，然后线性映射到 [-1, 1]。
        当某日有效股票数不足 10 只时，该日信号设为 NaN。

        Parameters
        ----------
        signal : pd.DataFrame
            原始因子值，index=日期，columns=股票代码

        Returns
        -------
        pd.DataFrame
            截面排名后的信号，值域 [-1, 1]
        """
        def rank_row(row):
            valid = row.dropna()
            if len(valid) < 10:
                return row * np.nan
            return (valid.rank(pct=True) * 2 - 1).reindex(row.index)
        return signal.apply(rank_row, axis=1)

    # ── 因子 1: 北向资金净流入因子 ──

    @staticmethod
    def northbound_flow_factor(north_flow_df, prices):
        """北向资金净流入因子 — 沪/深股通资金流向信号

        核心逻辑:
            1. 计算每只股票的北向资金 5日均值 和 20日均值
            2. 短期/长期净流入变化率 = (5日均值 - 20日均值) / |20日均值|
               反映北向资金的加速流入/流出趋势
            3. 北向持续流入 = 外资看多信号

        Parameters
        ----------
        north_flow_df : pd.DataFrame
            北向资金每日净买入额，index=日期，columns=股票代码
            正值表示净买入，负值表示净卖出，单位: 元
        prices : pd.DataFrame
            收盘价，index=日期，columns=股票代码
            用于对齐日期和股票范围

        Returns
        -------
        pd.DataFrame
            北向资金信号，index=日期，columns=股票代码，值域 [-1, 1]
            正值 = 北向持续净买入(看多)，负值 = 北向持续净卖出(看空)
        """
        # 对齐股票和日期
        common_cols = north_flow_df.columns.intersection(prices.columns)
        common_idx = north_flow_df.index.intersection(prices.index)
        flow = north_flow_df.loc[common_idx, common_cols]

        # 5日和20日滚动均值
        flow_5d = flow.rolling(5, min_periods=3).mean()
        flow_20d = flow.rolling(20, min_periods=10).mean()

        # 短期相对长期的变化率: 反映北向资金的加速趋势
        # 分母取绝对值，避免负/负 = 正的符号问题
        denominator = flow_20d.abs().replace(0, np.nan)
        raw = (flow_5d - flow_20d) / denominator

        # 用 prices 的完整 index/columns 做 reindex，缺失的填 NaN
        raw = raw.reindex(index=prices.index, columns=prices.columns)

        return ChinaSignalGenerator.cross_sectional_rank(raw)

    # ── 因子 2: 涨停板动量因子 ──

    @staticmethod
    def limit_up_momentum(prices, volumes):
        """涨停板动量因子 — A股涨停板制度下的强势股识别

        核心逻辑:
            1. 涨停判断: 日涨幅 >= 9.8%（考虑四舍五入误差）
            2. 统计过去 20 个交易日内涨停次数（频率分量）
            3. 计算涨停后 3 日的平均涨幅（延续性分量）
               涨停后继续上涨 = 主力封板坚决，不是骗炮
            4. 两个分量等权合成

        Parameters
        ----------
        prices : pd.DataFrame
            收盘价，index=日期，columns=股票代码
        volumes : pd.DataFrame
            成交量，index=日期，columns=股票代码
            用于辅助判断涨停的有效性（预留，当前版本未使用）

        Returns
        -------
        pd.DataFrame
            涨停板动量信号，index=日期，columns=股票代码，值域 [-1, 1]
            正值 = 近期涨停多且延续性强(强势股)
        """
        # 日收益率
        daily_ret = prices.pct_change()

        # 涨停判断: 涨幅 >= 9.8%
        limit_up = (daily_ret >= 0.098).astype(float)

        # ---- 分量1: 20日涨停次数 ----
        limit_count_20d = limit_up.rolling(20, min_periods=1).sum()

        # ---- 分量2: 涨停后3日延续性 ----
        # 对每个涨停事件，计算其后3日的累计收益
        # forward_ret_3d[t] = prices[t+3] / prices[t] - 1
        forward_ret_3d = prices.shift(-3) / prices - 1

        # 仅保留涨停日的延续性收益，非涨停日填 NaN
        continuation = forward_ret_3d.where(limit_up == 1, np.nan)

        # 20日滚动窗口内涨停事件的平均延续性收益
        # 使用 expanding-like 逻辑: rolling sum / rolling count
        continuation_sum = continuation.fillna(0).rolling(20, min_periods=1).sum()
        continuation_cnt = limit_up.rolling(20, min_periods=1).sum().replace(0, np.nan)
        avg_continuation = continuation_sum / continuation_cnt

        # 无涨停的股票延续性设为 0（中性）
        avg_continuation = avg_continuation.fillna(0)

        # ---- 合成 ----
        # 两个分量分别截面排名后等权相加
        sg = ChinaSignalGenerator
        rank_count = sg.cross_sectional_rank(limit_count_20d)
        rank_cont = sg.cross_sectional_rank(avg_continuation)

        raw = (rank_count + rank_cont) / 2

        return sg.cross_sectional_rank(raw)

    # ── 因子 3: 机构调研/龙虎榜热度因子 ──

    @staticmethod
    def institutional_attention(dragon_tiger_df, prices):
        """机构调研/龙虎榜热度因子 — 基于龙虎榜数据的机构行为信号

        核心逻辑:
            1. 基于龙虎榜数据，提取机构买入净额和出现频次
            2. 20日指数衰减加权: 近期的龙虎榜事件权重更高
               衰减半衰期 = 5日，即 5 天前的事件权重降为一半
            3. 两个分量（净额 + 频次）等权合成
            4. 机构持续买入 = 看多信号

        Parameters
        ----------
        dragon_tiger_df : pd.DataFrame
            龙虎榜机构净买入额，index=日期，columns=股票代码
            正值表示机构净买入，负值表示机构净卖出
            非龙虎榜日期填 0 或 NaN

        prices : pd.DataFrame
            收盘价，index=日期，columns=股票代码
            用于对齐日期和股票范围

        Returns
        -------
        pd.DataFrame
            机构热度信号，index=日期，columns=股票代码，值域 [-1, 1]
            正值 = 机构持续买入(看多)，负值 = 机构持续卖出(看空)
        """
        # 对齐
        common_cols = dragon_tiger_df.columns.intersection(prices.columns)
        common_idx = dragon_tiger_df.index.intersection(prices.index)
        dt_data = dragon_tiger_df.loc[common_idx, common_cols].fillna(0)

        # 构建指数衰减权重: 半衰期5日, 窗口20日
        lookback = 20
        half_life = 5
        decay = np.exp(-np.log(2) / half_life * np.arange(lookback))
        # decay[0] = 最近一天(权重最大), decay[19] = 最远一天(权重最小)
        decay = decay / decay.sum()  # 归一化

        # ---- 分量1: 衰减加权净买入额 ----
        def _decay_weighted_sum(series, weights):
            """对单列计算衰减加权滚动和"""
            result = pd.Series(np.nan, index=series.index)
            arr = series.values
            w = weights
            n = len(w)
            for i in range(n - 1, len(arr)):
                window = arr[i - n + 1: i + 1]
                # 翻转使得最近的日期对应 weights[0]（最大权重）
                result.iloc[i] = np.nansum(window[::-1] * w)
            return result

        weighted_amount = pd.DataFrame(np.nan, index=dt_data.index, columns=dt_data.columns)
        for col in dt_data.columns:
            weighted_amount[col] = _decay_weighted_sum(dt_data[col], decay)

        # ---- 分量2: 衰减加权出现频次 ----
        # 龙虎榜出现 = 净额绝对值 > 0
        dt_occur = (dt_data.abs() > 0).astype(float)
        weighted_freq = pd.DataFrame(np.nan, index=dt_data.index, columns=dt_data.columns)
        for col in dt_data.columns:
            weighted_freq[col] = _decay_weighted_sum(dt_occur[col], decay)

        # ---- 合成 ----
        sg = ChinaSignalGenerator
        rank_amount = sg.cross_sectional_rank(
            weighted_amount.reindex(index=prices.index, columns=prices.columns)
        )
        rank_freq = sg.cross_sectional_rank(
            weighted_freq.reindex(index=prices.index, columns=prices.columns)
        )

        raw = (rank_amount + rank_freq) / 2

        return sg.cross_sectional_rank(raw)

    # ── 复合信号汇总 ──

    @staticmethod
    def build_cn_signals(prices, volumes, north_flow, dragon_tiger, weights=None):
        """构建A股特色复合因子信号

        将三个A股特色因子加权合成为综合信号。

        Parameters
        ----------
        prices : pd.DataFrame
            收盘价，index=日期，columns=股票代码
        volumes : pd.DataFrame
            成交量，index=日期，columns=股票代码
        north_flow : pd.DataFrame
            北向资金每日净买入额，index=日期，columns=股票代码
        dragon_tiger : pd.DataFrame
            龙虎榜机构净买入额，index=日期，columns=股票代码
        weights : dict or None
            因子名 → 权重的字典，默认等权。
            可用键: 'northbound_flow', 'limit_up_momentum', 'institutional_attention'

        Returns
        -------
        dict
            包含以下键:
            - 'northbound_flow': 北向资金因子信号 DataFrame
            - 'limit_up_momentum': 涨停板动量因子信号 DataFrame
            - 'institutional_attention': 机构热度因子信号 DataFrame
            - 'composite': 加权复合信号 DataFrame，值域 [-1, 1]
        """
        sg = ChinaSignalGenerator

        # 计算各因子
        signals = {}
        signals['northbound_flow'] = sg.northbound_flow_factor(north_flow, prices)
        signals['limit_up_momentum'] = sg.limit_up_momentum(prices, volumes)
        signals['institutional_attention'] = sg.institutional_attention(dragon_tiger, prices)

        # 默认等权
        if weights is None:
            weights = {k: 1.0 / 3 for k in signals}

        # 归一化权重
        used = {k: v for k, v in weights.items() if k in signals}
        if not used:
            raise ValueError("无可用因子，请检查 weights 键名")
        total_w = sum(used.values())
        used = {k: v / total_w for k, v in used.items()}

        # 加权合成
        composite = None
        for name, w in used.items():
            term = signals[name].astype(float) * w
            if composite is None:
                composite = term
            else:
                composite = composite.add(term, fill_value=0)

        signals['composite'] = sg.cross_sectional_rank(composite)
        return signals


# ── 便捷汇总函数 ──

def build_cn_signals(prices, volumes, north_flow, dragon_tiger, weights=None):
    """便捷函数 — 一次性计算全部A股特色因子

    Parameters
    ----------
    prices : pd.DataFrame
        收盘价，index=日期，columns=股票代码
    volumes : pd.DataFrame
        成交量，index=日期，columns=股票代码
    north_flow : pd.DataFrame
        北向资金每日净买入额，index=日期，columns=股票代码
    dragon_tiger : pd.DataFrame
        龙虎榜机构净买入额，index=日期，columns=股票代码
    weights : dict or None
        因子权重字典，默认等权

    Returns
    -------
    dict
        信号名 → DataFrame 的字典
    """
    return ChinaSignalGenerator.build_cn_signals(
        prices, volumes, north_flow, dragon_tiger, weights
    )


if __name__ == "__main__":
    # ── 用模拟数据演示三个因子 ──

    np.random.seed(42)
    n_days = 60       # 60 个交易日（约3个月）
    n_stocks = 30     # 30 只模拟股票

    dates = pd.bdate_range("2025-01-02", periods=n_days, freq="B")
    tickers = [f"{600000 + i}" for i in range(n_stocks)]

    # ---- 模拟价格数据 ----
    # 初始价格 10~50 元
    init_prices = np.random.uniform(10, 50, n_stocks)
    price_data = np.zeros((n_days, n_stocks))
    price_data[0] = init_prices
    for t in range(1, n_days):
        # 日收益率: 大部分正常波动，少数模拟涨停
        daily_rets = np.random.normal(0.001, 0.025, n_stocks)
        # 随机让某些股票涨停 (涨幅 10%)
        limit_mask = np.random.random(n_stocks) < 0.03  # 3%概率涨停
        daily_rets[limit_mask] = 0.10
        price_data[t] = price_data[t - 1] * (1 + daily_rets)

    prices = pd.DataFrame(price_data, index=dates, columns=tickers)
    volumes = pd.DataFrame(
        np.random.randint(100000, 5000000, (n_days, n_stocks)),
        index=dates, columns=tickers
    ).astype(float)

    # ---- 模拟北向资金数据 ----
    # 大部分为0（非北向标的），少数有持续流入/流出
    north_flow = pd.DataFrame(0.0, index=dates, columns=tickers)
    # 选 10 只作为北向标的
    north_tickers = tickers[:10]
    north_flow[north_tickers] = np.random.normal(5e6, 2e7, (n_days, 10))
    # 给前3只模拟持续流入
    north_flow[tickers[0]] = np.abs(np.random.normal(3e7, 1e7, n_days))
    north_flow[tickers[1]] = np.abs(np.random.normal(2e7, 8e6, n_days))

    # ---- 模拟龙虎榜数据 ----
    # 大部分为0（无龙虎榜事件），偶发有机构买卖
    dragon_tiger = pd.DataFrame(0.0, index=dates, columns=tickers)
    for t in range(n_days):
        # 每天随机 2~3 只上榜
        n_events = np.random.randint(1, 4)
        event_stocks = np.random.choice(n_stocks, n_events, replace=False)
        for s in event_stocks:
            # 机构净买入额: 正负随机
            dragon_tiger.iloc[t, s] = np.random.normal(1e7, 5e7)

    # ---- 计算因子 ----
    print("=" * 60)
    print("A股特色因子演示 (模拟数据)")
    print("=" * 60)

    sg = ChinaSignalGenerator

    print("\n[1] 北向资金净流入因子")
    sig_north = sg.northbound_flow_factor(north_flow, prices)
    print(f"    形状: {sig_north.shape}")
    print(f"    值域: [{sig_north.min().min():.4f}, {sig_north.max().max():.4f}]")
    print(f"    非NaN比例: {sig_north.notna().mean().mean():.1%}")
    print(f"    最新一日信号 (前5只):\n{sig_north.iloc[-1, :5]}")

    print("\n[2] 涨停板动量因子")
    sig_limit = sg.limit_up_momentum(prices, volumes)
    print(f"    形状: {sig_limit.shape}")
    print(f"    值域: [{sig_limit.min().min():.4f}, {sig_limit.max().max():.4f}]")
    print(f"    非NaN比例: {sig_limit.notna().mean().mean():.1%}")
    print(f"    最新一日信号 (前5只):\n{sig_limit.iloc[-1, :5]}")

    print("\n[3] 机构调研/龙虎榜热度因子")
    sig_inst = sg.institutional_attention(dragon_tiger, prices)
    print(f"    形状: {sig_inst.shape}")
    print(f"    值域: [{sig_inst.min().min():.4f}, {sig_inst.max().max():.4f}]")
    print(f"    非NaN比例: {sig_inst.notna().mean().mean():.1%}")
    print(f"    最新一日信号 (前5只):\n{sig_inst.iloc[-1, :5]}")

    # ---- 复合信号 ----
    print("\n[复合] build_cn_signals 汇总")
    all_signals = build_cn_signals(prices, volumes, north_flow, dragon_tiger)
    for name, sig in all_signals.items():
        valid_pct = sig.notna().mean().mean()
        print(f"    {name:30s} shape={sig.shape}  非NaN={valid_pct:.1%}")

    print("\n" + "=" * 60)
    print("演示完成")
