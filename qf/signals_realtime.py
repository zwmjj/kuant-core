"""实时因子信号引擎 — 基于滚动窗口的增量计算，适用于逐bar流式数据"""
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Optional


class RealtimeSignalEngine:
    """实时因子计算引擎

    设计原则:
    - 增量更新: 每次收到新bar只做 O(1) 计算，不重算历史
    - 滚动窗口: 使用 deque 维护定长窗口，自动丢弃过期数据
    - warmup 保护: 数据不足时因子返回 None，避免错误信号

    使用方式:
        engine = RealtimeSignalEngine(window=20, rsi_period=14)
        engine.update("AAPL", bar_data)    # 逐bar推送
        signals = engine.get_signals("AAPL")  # 获取当前全部因子
        score = engine.get_composite_signal("AAPL", weights)  # 加权综合分

    bar_data 格式 (dict):
        必须字段: close, volume, timestamp
        可选字段: high, low, open, vwap, bid, ask
    """

    def __init__(self, window: int = 20, rsi_period: int = 14):
        """初始化引擎

        Parameters
        ----------
        window : int
            滚动窗口长度（bar数），用于动量、成交量均值等计算，默认20
        rsi_period : int
            RSI 计算周期，默认14
        """
        self.window = window
        self.rsi_period = rsi_period

        # 每个 symbol 独立维护的滚动数据缓冲区
        # 键: symbol, 值: dict of deque
        self._buffers: Dict[str, dict] = defaultdict(self._init_buffer)

        # RSI 增量状态（Wilder 平滑均值）
        self._rsi_state: Dict[str, dict] = {}

        # VWAP 当日累计状态（日内重置）
        self._vwap_state: Dict[str, dict] = {}

    def _init_buffer(self) -> dict:
        """为新 symbol 创建空缓冲区"""
        max_len = max(self.window, self.rsi_period) + 5  # 稍留余量
        return {
            'close': deque(maxlen=max_len),
            'volume': deque(maxlen=max_len),
            'high': deque(maxlen=max_len),
            'low': deque(maxlen=max_len),
            'open': deque(maxlen=max_len),
            'vwap': deque(maxlen=max_len),
            'bid': deque(maxlen=max_len),
            'ask': deque(maxlen=max_len),
            'timestamp': deque(maxlen=max_len),
            'bar_count': 0,
        }

    # ═══════════════════════════════════════════════════════
    # 核心接口
    # ═══════════════════════════════════════════════════════

    def update(self, symbol: str, bar_data: dict) -> Dict[str, Optional[float]]:
        """接收新 bar 数据，增量更新所有因子

        Parameters
        ----------
        symbol : str
            标的代码，如 "AAPL"
        bar_data : dict
            必须包含 'close' 和 'volume'；
            可选: 'high', 'low', 'open', 'vwap', 'bid', 'ask', 'timestamp'

        Returns
        -------
        dict
            当前所有因子值（同 get_signals）
        """
        buf = self._buffers[symbol]

        # 写入缓冲区
        buf['close'].append(bar_data['close'])
        buf['volume'].append(bar_data['volume'])
        buf['high'].append(bar_data.get('high', bar_data['close']))
        buf['low'].append(bar_data.get('low', bar_data['close']))
        buf['open'].append(bar_data.get('open', bar_data['close']))
        buf['vwap'].append(bar_data.get('vwap'))
        buf['bid'].append(bar_data.get('bid'))
        buf['ask'].append(bar_data.get('ask'))
        buf['timestamp'].append(bar_data.get('timestamp'))
        buf['bar_count'] += 1

        # 增量更新 RSI 的 Wilder 平滑状态
        self._update_rsi_state(symbol)

        # 增量更新 VWAP 累计（如果有日内 vwap 则跳过，直接用 bar 级 vwap）
        self._update_vwap_state(symbol, bar_data)

        return self.get_signals(symbol)

    def get_signals(self, symbol: str) -> Dict[str, Optional[float]]:
        """返回该 symbol 当前所有因子值

        Returns
        -------
        dict
            键为因子名，值为 float 或 None（warmup 期不足）
        """
        return {
            'realtime_momentum': self._calc_momentum(symbol),
            'realtime_vwap_deviation': self._calc_vwap_deviation(symbol),
            'realtime_volume_surge': self._calc_volume_surge(symbol),
            'realtime_spread_signal': self._calc_spread_signal(symbol),
            'realtime_rsi': self._calc_rsi(symbol),
        }

    def get_cross_sectional_signals(
        self,
        symbols: list,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Optional[float]]:
        """Cross-sectional normalized composite signals across all symbols

        Unlike get_composite_signal (absolute per-symbol), this ranks signals
        across the universe and maps to [-1, 1], matching the daily factor approach.

        Parameters
        ----------
        symbols : list of str
            Universe of symbols to rank across
        weights : dict or None
            Factor name -> weight. None = equal weight.

        Returns
        -------
        dict
            {symbol: normalized_signal} in [-1, 1], or None if insufficient data
        """
        # Collect raw composite scores for all symbols
        raw_scores = {}
        for sym in symbols:
            score = self.get_composite_signal(sym, weights)
            if score is not None:
                raw_scores[sym] = score

        if len(raw_scores) < 5:
            return {sym: None for sym in symbols}

        # Cross-sectional rank -> [-1, 1]
        import pandas as pd
        scores_series = pd.Series(raw_scores)
        ranked = scores_series.rank(pct=True) * 2 - 1

        result = {}
        for sym in symbols:
            result[sym] = float(ranked[sym]) if sym in ranked.index else None
        return result

    def get_all_factor_matrix(
        self,
        symbols: list,
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """Get per-factor cross-sectional ranked signals for all symbols

        Returns a dict of {factor_name: {symbol: ranked_value}}, where each
        factor is independently cross-sectionally ranked.

        Parameters
        ----------
        symbols : list of str

        Returns
        -------
        dict of dict
            {factor_name: {symbol: float in [-1,1] or None}}
        """
        import pandas as pd

        factor_names = ['realtime_momentum', 'realtime_vwap_deviation',
                        'realtime_volume_surge', 'realtime_spread_signal',
                        'realtime_rsi']

        # Collect raw values
        raw = {f: {} for f in factor_names}
        for sym in symbols:
            signals = self.get_signals(sym)
            for f in factor_names:
                val = signals.get(f)
                if val is not None:
                    raw[f][sym] = val

        # Cross-sectional rank each factor
        result = {}
        for f in factor_names:
            if len(raw[f]) < 5:
                result[f] = {sym: None for sym in symbols}
                continue
            series = pd.Series(raw[f])
            ranked = series.rank(pct=True) * 2 - 1
            result[f] = {sym: float(ranked.get(sym, float('nan')))
                         if sym in ranked.index else None
                         for sym in symbols}

        return result

    def get_composite_signal(
        self,
        symbol: str,
        weights: Optional[Dict[str, float]] = None,
    ) -> Optional[float]:
        """返回加权综合信号，值域 [-1, 1]

        Parameters
        ----------
        symbol : str
            标的代码
        weights : dict or None
            因子名 -> 权重。None 则等权。
            例: {'realtime_momentum': 0.3, 'realtime_rsi': 0.3,
                 'realtime_volume_surge': 0.2, 'realtime_vwap_deviation': 0.1,
                 'realtime_spread_signal': 0.1}

        Returns
        -------
        float or None
            综合信号，[-1, 1]。如果所有因子均为 None 则返回 None。
        """
        signals = self.get_signals(symbol)

        if weights is None:
            # 默认等权
            weights = {k: 1.0 for k in signals}

        # 只使用有值的因子
        valid_pairs = []
        for name, w in weights.items():
            val = signals.get(name)
            if val is not None:
                valid_pairs.append((val, w))

        if not valid_pairs:
            return None

        # 归一化权重并加权求和
        total_w = sum(w for _, w in valid_pairs)
        if total_w == 0:
            return None

        raw = sum(v * w for v, w in valid_pairs) / total_w

        # 裁剪到 [-1, 1]
        return float(np.clip(raw, -1.0, 1.0))

    # ═══════════════════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════════════════

    def reset(self, symbol: Optional[str] = None):
        """重置指定 symbol 的全部状态，或重置所有"""
        if symbol is None:
            self._buffers.clear()
            self._rsi_state.clear()
            self._vwap_state.clear()
        else:
            self._buffers.pop(symbol, None)
            self._rsi_state.pop(symbol, None)
            self._vwap_state.pop(symbol, None)

    def get_bar_count(self, symbol: str) -> int:
        """返回已接收的 bar 数量"""
        return self._buffers[symbol]['bar_count']

    # ═══════════════════════════════════════════════════════
    # 因子实现 — 增量计算
    # ═══════════════════════════════════════════════════════

    def _calc_momentum(self, symbol: str) -> Optional[float]:
        """realtime_momentum: 最近 N 根 bar 的收益率动量

        计算方式: (当前价 / N根bar前的价) - 1
        warmup: 需要至少 window + 1 根 bar

        返回值语义:
            正值 = 近期上涨趋势, 负值 = 近期下跌趋势
        """
        buf = self._buffers[symbol]
        closes = buf['close']

        if len(closes) < self.window + 1:
            return None

        current = closes[-1]
        past = closes[-(self.window + 1)]

        if past == 0 or past is None:
            return None

        return (current / past) - 1.0

    def _calc_vwap_deviation(self, symbol: str) -> Optional[float]:
        """realtime_vwap_deviation: 当前价偏离 VWAP 的程度

        计算方式:
        - 如果 bar 自带 vwap 字段 → 使用累计 VWAP
        - 否则 → 用 sum(price*volume) / sum(volume) 手动计算

        返回值: (close - vwap) / vwap
            正值 = 价格在 VWAP 上方（买压）
            负值 = 价格在 VWAP 下方（卖压）

        warmup: 至少 2 根 bar
        """
        buf = self._buffers[symbol]

        if len(buf['close']) < 2:
            return None

        current_close = buf['close'][-1]

        # 优先使用 VWAP 累计状态
        vwap_st = self._vwap_state.get(symbol)
        if vwap_st and vwap_st['cum_volume'] > 0:
            vwap = vwap_st['cum_pv'] / vwap_st['cum_volume']
        else:
            # 没有 vwap 数据时，用缓冲区里的 close*volume 近似
            closes = list(buf['close'])
            volumes = list(buf['volume'])
            n = min(len(closes), self.window)
            pv_sum = sum(closes[-n + i] * volumes[-n + i] for i in range(n))
            v_sum = sum(volumes[-n + i] for i in range(n))
            if v_sum == 0:
                return None
            vwap = pv_sum / v_sum

        if vwap == 0:
            return None

        return (current_close - vwap) / vwap

    def _calc_volume_surge(self, symbol: str) -> Optional[float]:
        """realtime_volume_surge: 当前 bar 成交量相对近期均值的倍数

        计算方式: current_volume / mean(最近 window 根 bar 的 volume)
        输出做了中心化处理: surge - 1.0，使得:
            0 = 成交量正常
            正值 = 放量（如 1.5 表示当前是均值的 2.5 倍）
            负值 = 缩量

        warmup: 至少 window 根 bar
        """
        buf = self._buffers[symbol]
        volumes = buf['volume']

        if len(volumes) < self.window:
            return None

        # 用前 window-1 根 bar 算均值（排除当前 bar），避免自身影响
        hist_volumes = list(volumes)[-self.window:-1]
        avg = sum(hist_volumes) / len(hist_volumes)

        if avg == 0:
            return None

        current = volumes[-1]
        return (current / avg) - 1.0

    def _calc_spread_signal(self, symbol: str) -> Optional[float]:
        """realtime_spread_signal: 买卖价差异常信号

        计算方式:
        1. spread = ask - bid
        2. relative_spread = spread / mid_price
        3. 与近期均值比较: z_score = (当前spread - 均值) / 标准差

        返回值语义:
            正值（高价差）= 流动性恶化 / 不确定性上升 → 偏看空
            负值（低价差）= 流动性充裕 → 偏看多

        注意: 取负值输出，使高流动性对应正信号

        warmup: 至少 window 根包含 bid/ask 的 bar
        """
        buf = self._buffers[symbol]

        # 检查是否有 bid/ask 数据
        bids = buf['bid']
        asks = buf['ask']

        if len(bids) < self.window:
            return None

        # 收集最近 window 个有效 spread
        spreads = []
        for i in range(-self.window, 0):
            b = bids[i]
            a = asks[i]
            if b is not None and a is not None and b > 0 and a > 0:
                mid = (a + b) / 2.0
                spreads.append((a - b) / mid)

        if len(spreads) < self.window // 2:
            # 有效 quote 数据不足
            return None

        current_spread = spreads[-1]
        mean_spread = sum(spreads) / len(spreads)
        var_spread = sum((s - mean_spread) ** 2 for s in spreads) / len(spreads)
        std_spread = var_spread ** 0.5

        if std_spread < 1e-12:
            return 0.0

        z = (current_spread - mean_spread) / std_spread

        # 取负值: 价差收窄（流动性好）为正信号
        # 裁剪到 [-1, 1] 区间（z_score / 3 做归一化近似）
        return float(np.clip(-z / 3.0, -1.0, 1.0))

    def _calc_rsi(self, symbol: str) -> Optional[float]:
        """realtime_rsi: 流式增量 RSI (Wilder 平滑)

        增量算法:
        1. 首次满足 warmup 时，用简单均值初始化 avg_gain / avg_loss
        2. 后续每根 bar 用指数平滑更新:
           avg_gain = (avg_gain * (period-1) + gain) / period
           avg_loss = (avg_loss * (period-1) + loss) / period
        3. RS = avg_gain / avg_loss
        4. RSI = 100 - 100 / (1 + RS)

        返回值做了归一化: (RSI - 50) / 50，映射到 [-1, 1]
            正值 = 超买方向, 负值 = 超卖方向

        warmup: 至少 rsi_period + 1 根 bar
        """
        state = self._rsi_state.get(symbol)

        if state is None or not state.get('initialized'):
            return None

        avg_gain = state['avg_gain']
        avg_loss = state['avg_loss']

        if avg_loss == 0:
            # 全部上涨，RSI = 100
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - 100.0 / (1.0 + rs)

        # 归一化到 [-1, 1]: RSI 50 → 0, RSI 100 → 1, RSI 0 → -1
        return (rsi - 50.0) / 50.0

    # ═══════════════════════════════════════════════════════
    # 内部状态更新
    # ═══════════════════════════════════════════════════════

    def _update_rsi_state(self, symbol: str):
        """增量更新 RSI 的 Wilder 平滑状态

        Wilder 平滑（指数移动均值）:
        - 初始化: avg = SMA(first N values)
        - 更新:   avg = (prev_avg * (N-1) + new_value) / N
        """
        buf = self._buffers[symbol]
        closes = buf['close']
        period = self.rsi_period

        if len(closes) < 2:
            return

        # 计算本次价格变动
        change = closes[-1] - closes[-2]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        if symbol not in self._rsi_state:
            self._rsi_state[symbol] = {
                'initialized': False,
                'gains': [],
                'losses': [],
                'avg_gain': 0.0,
                'avg_loss': 0.0,
            }

        state = self._rsi_state[symbol]

        if not state['initialized']:
            # 收集初始期数据
            state['gains'].append(gain)
            state['losses'].append(loss)

            if len(state['gains']) >= period:
                # 用简单均值初始化
                state['avg_gain'] = sum(state['gains'][:period]) / period
                state['avg_loss'] = sum(state['losses'][:period]) / period
                state['initialized'] = True
                # 释放临时列表
                del state['gains']
                del state['losses']
        else:
            # Wilder 平滑更新 — O(1)
            state['avg_gain'] = (state['avg_gain'] * (period - 1) + gain) / period
            state['avg_loss'] = (state['avg_loss'] * (period - 1) + loss) / period

    def _update_vwap_state(self, symbol: str, bar_data: dict):
        """增量更新 VWAP 累计状态

        累计 price*volume 和 volume，用于计算日内 VWAP。
        如果 bar_data 自带 timestamp，检测日期切换时重置累计。
        """
        close = bar_data['close']
        volume = bar_data['volume']
        timestamp = bar_data.get('timestamp')

        if symbol not in self._vwap_state:
            self._vwap_state[symbol] = {
                'cum_pv': 0.0,
                'cum_volume': 0.0,
                'last_date': None,
            }

        state = self._vwap_state[symbol]

        # 检测日期切换 → 重置累计（新交易日）
        current_date = None
        if timestamp is not None:
            if hasattr(timestamp, 'date'):
                current_date = timestamp.date()
            elif hasattr(timestamp, 'day'):
                current_date = timestamp

        if current_date is not None and state['last_date'] is not None:
            if current_date != state['last_date']:
                state['cum_pv'] = 0.0
                state['cum_volume'] = 0.0

        state['last_date'] = current_date

        # 使用 typical price 作为 VWAP 的价格基准
        # typical = (high + low + close) / 3，更贴近真实 VWAP
        high = bar_data.get('high', close)
        low = bar_data.get('low', close)
        typical = (high + low + close) / 3.0

        state['cum_pv'] += typical * volume
        state['cum_volume'] += volume
