"""Screener信号生成器 — 基于Alpaca screener数据（最活跃/市场异动）"""
import numpy as np
import pandas as pd
from collections import Counter


def _cross_sectional_rank(s: pd.Series) -> pd.Series:
    """截面排名归一化到 [-1, 1]"""
    valid = s.dropna()
    if len(valid) == 0:
        return s
    ranked = valid.rank(pct=True)  # [0, 1]
    return (ranked * 2 - 1).reindex(s.index)


class ScreenerSignalGenerator:
    """基于Alpaca screener数据的信号集合

    Alpaca screener提供:
    - most_actives: [{'symbol': 'AAPL', 'volume': 1e8, 'trade_count': 5e5}, ...]
    - gainers/losers: [{'symbol': 'TSLA', 'percent_change': 5.2, 'price': 180.0}, ...]
    """

    # ── 注意力信号 ──

    @staticmethod
    def attention_signal(most_actives: list, universe: list) -> pd.Series:
        """注意力信号 — 高关注度股票倾向于短期动量（注意力驱动收益）

        Parameters
        ----------
        most_actives : list[dict]
            Alpaca most_actives数据，含 symbol, volume, trade_count
            按volume降序排列
        universe : list[str]
            投资宇宙中的标的列表

        Returns
        -------
        pd.Series : 截面排名归一化到 [-1, 1]，未在most_actives中的为0
        """
        signal = pd.Series(0.0, index=universe)

        if not most_actives:
            return signal

        n = len(most_actives)
        for rank_idx, item in enumerate(most_actives):
            sym = item.get('symbol', '')
            if sym in signal.index:
                # 排名越靠前（volume越大），信号越强
                # rank_idx=0 => 权重最大
                weight = (n - rank_idx) / n
                signal[sym] = weight

        # 仅对有信号的部分做截面排名
        active_mask = signal != 0
        if active_mask.sum() > 1:
            signal[active_mask] = _cross_sectional_rank(signal[active_mask])

        return signal

    # ── 动量信号 ──

    @staticmethod
    def momentum_signal(gainers: list, losers: list, universe: list) -> pd.Series:
        """短期趋势延续动量信号 — gainers给正信号，losers给负信号

        逻辑: 日内大涨/大跌股票在次日有延续效应（短期动量）

        Parameters
        ----------
        gainers : list[dict]
            含 symbol, percent_change, price
        losers : list[dict]
            含 symbol, percent_change, price（percent_change为负值）
        universe : list[str]
            投资宇宙

        Returns
        -------
        pd.Series : 截面排名归一化到 [-1, 1]
        """
        signal = pd.Series(0.0, index=universe)

        for item in (gainers or []):
            sym = item.get('symbol', '')
            pct = item.get('percent_change', 0.0)
            if sym in signal.index:
                signal[sym] = abs(pct)  # 正值

        for item in (losers or []):
            sym = item.get('symbol', '')
            pct = item.get('percent_change', 0.0)
            if sym in signal.index:
                signal[sym] = -abs(pct)  # 负值

        non_zero = signal != 0
        if non_zero.sum() > 1:
            signal[non_zero] = _cross_sectional_rank(signal[non_zero])

        return signal

    # ── 反转信号 ──

    @staticmethod
    def contrarian_signal(gainers: list, losers: list, universe: list) -> pd.Series:
        """均值回归反转信号 — losers给正信号（超卖反弹），gainers给负信号

        极端波动(>5%)时效果更佳（过度反应修正）

        Parameters
        ----------
        gainers : list[dict]
            含 symbol, percent_change, price
        losers : list[dict]
            含 symbol, percent_change, price
        universe : list[str]
            投资宇宙

        Returns
        -------
        pd.Series : 截面排名归一化到 [-1, 1]
        """
        signal = pd.Series(0.0, index=universe)

        # losers => 正信号（超卖反弹机会），极端跌幅加强
        for item in (losers or []):
            sym = item.get('symbol', '')
            pct = abs(item.get('percent_change', 0.0))
            if sym in signal.index:
                # >5%的跌幅给予额外权重（极端反转更可靠）
                multiplier = 1.5 if pct > 5.0 else 1.0
                signal[sym] = pct * multiplier

        # gainers => 负信号（超买回调风险），极端涨幅加强
        for item in (gainers or []):
            sym = item.get('symbol', '')
            pct = abs(item.get('percent_change', 0.0))
            if sym in signal.index:
                multiplier = 1.5 if pct > 5.0 else 1.0
                signal[sym] = -pct * multiplier

        non_zero = signal != 0
        if non_zero.sum() > 1:
            signal[non_zero] = _cross_sectional_rank(signal[non_zero])

        return signal

    # ── 交易强度信号 ──

    @staticmethod
    def trade_intensity_signal(most_actives: list, universe: list) -> pd.Series:
        """交易强度信号 — trade_count/volume比率区分散户vs机构

        高trade_count/volume = 小单多 = 散户兴趣（噪音交易者）
        低trade_count/volume = 大单多 = 机构大宗交易（信息交易者）

        信号方向: 机构参与度高（低比率）=> 正信号

        Parameters
        ----------
        most_actives : list[dict]
            含 symbol, volume, trade_count
        universe : list[str]
            投资宇宙

        Returns
        -------
        pd.Series : 截面排名归一化到 [-1, 1]
        """
        signal = pd.Series(np.nan, index=universe)

        if not most_actives:
            return pd.Series(0.0, index=universe)

        for item in (most_actives or []):
            sym = item.get('symbol', '')
            volume = item.get('volume', 0)
            trade_count = item.get('trade_count', 0)

            if sym in signal.index and volume > 0:
                # 低比率 = 机构大宗 => 信号为正
                # 取倒数使得低trade_count/volume => 高信号值
                ratio = trade_count / volume
                signal[sym] = -ratio  # 负ratio => 正信号（机构主导）

        valid = signal.dropna()
        if len(valid) > 1:
            signal[valid.index] = _cross_sectional_rank(valid)

        # 不在screener中的设为0
        signal = signal.fillna(0.0)
        return signal

    # ── 持续性信号 ──

    @staticmethod
    def persistence_signal(actives_history: list, days: int = 5) -> pd.Series:
        """持续关注度信号 — 多日连续出现在most_actives中表示持续机构兴趣

        Parameters
        ----------
        actives_history : list[list[dict]]
            最近N天的most_actives数据列表，每个元素是一天的most_actives
            actives_history[0] = 最近一天, actives_history[-1] = 最早一天
        days : int
            统计窗口天数（默认5天）

        Returns
        -------
        pd.Series : 出现频率越高信号越强，截面排名归一化到 [-1, 1]
        """
        if not actives_history:
            return pd.Series(dtype=float)

        # 截取最近days天
        history = actives_history[:days]
        n_days = len(history)

        # 统计每只股票出现的天数
        appearance_count = Counter()
        all_symbols = set()

        for day_actives in history:
            if not day_actives:
                continue
            day_symbols = set()
            for item in day_actives:
                sym = item.get('symbol', '')
                if sym:
                    day_symbols.add(sym)
                    all_symbols.add(sym)
            for sym in day_symbols:
                appearance_count[sym] += 1

        if not all_symbols:
            return pd.Series(dtype=float)

        # 构建信号: 出现频率
        signal = pd.Series(0.0, index=sorted(all_symbols))
        for sym, count in appearance_count.items():
            signal[sym] = count / n_days  # [0, 1]范围

        # 仅保留出现>=2天的（偶尔出现不算持续关注）
        persistent_mask = signal >= (2 / n_days)
        if persistent_mask.sum() > 1:
            signal[persistent_mask] = _cross_sectional_rank(signal[persistent_mask])
        signal[~persistent_mask] = 0.0

        return signal

    # ── 综合信号 ──

    @staticmethod
    def build_screener_signal(
        most_actives: list,
        gainers: list,
        losers: list,
        universe: list,
        w_attention: float = 0.3,
        w_momentum: float = 0.3,
        w_contrarian: float = 0.2,
        w_intensity: float = 0.2,
    ) -> pd.Series:
        """综合screener信号 — 加权合成所有子信号

        Parameters
        ----------
        most_actives : list[dict]
            Alpaca most_actives数据
        gainers, losers : list[dict]
            Alpaca市场异动数据
        universe : list[str]
            投资宇宙
        w_attention, w_momentum, w_contrarian, w_intensity : float
            各子信号权重，默认 0.3/0.3/0.2/0.2

        Returns
        -------
        pd.Series : 截面排名归一化到 [-1, 1] 的综合信号
        """
        gen = ScreenerSignalGenerator

        sig_att = gen.attention_signal(most_actives, universe)
        sig_mom = gen.momentum_signal(gainers, losers, universe)
        sig_con = gen.contrarian_signal(gainers, losers, universe)
        sig_int = gen.trade_intensity_signal(most_actives, universe)

        composite = (
            w_attention * sig_att
            + w_momentum * sig_mom
            + w_contrarian * sig_con
            + w_intensity * sig_int
        )

        return _cross_sectional_rank(composite)

    # ── 日度流水线 ──

    @staticmethod
    def daily_screener_pipeline(alpaca_loader) -> dict:
        """日度screener信号流水线 — 从Alpaca获取数据并计算所有信号

        Parameters
        ----------
        alpaca_loader : object
            需要提供以下方法:
            - get_most_actives() -> list[dict]
            - get_gainers() -> list[dict]
            - get_losers() -> list[dict]
            - get_universe() -> list[str]
            - get_actives_history(days) -> list[list[dict]]  (可选)

        Returns
        -------
        dict : {
            'attention': pd.Series,
            'momentum': pd.Series,
            'contrarian': pd.Series,
            'trade_intensity': pd.Series,
            'persistence': pd.Series or None,
            'composite': pd.Series,
        }
        """
        gen = ScreenerSignalGenerator

        # 获取screener数据
        most_actives = alpaca_loader.get_most_actives()
        gainers = alpaca_loader.get_gainers()
        losers = alpaca_loader.get_losers()
        universe = alpaca_loader.get_universe()

        # 计算各子信号
        signals = {
            'attention': gen.attention_signal(most_actives, universe),
            'momentum': gen.momentum_signal(gainers, losers, universe),
            'contrarian': gen.contrarian_signal(gainers, losers, universe),
            'trade_intensity': gen.trade_intensity_signal(most_actives, universe),
        }

        # 持续性信号（需要历史数据，可选）
        if hasattr(alpaca_loader, 'get_actives_history'):
            try:
                history = alpaca_loader.get_actives_history(days=5)
                signals['persistence'] = gen.persistence_signal(history, days=5)
            except Exception:
                signals['persistence'] = None
        else:
            signals['persistence'] = None

        # 综合信号
        signals['composite'] = gen.build_screener_signal(
            most_actives, gainers, losers, universe,
        )

        return signals
