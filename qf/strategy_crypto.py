"""Crypto trading strategies - a strategy family built on Alpaca crypto data

Supported pairs: BTC/USD, ETH/USD, SOL/USD, DOGE/USD, AVAX/USD, LINK/USD, DOT/USD, ADA/USD
Data format: DataFrame with columns open, high, low, close, volume, trade_count, vwap (daily)
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


# ── 常量 ──────────────────────────────────────────────────────────────────────

CRYPTO_PAIRS = [
    'BTC/USD', 'ETH/USD', 'SOL/USD', 'DOGE/USD',
    'AVAX/USD', 'LINK/USD', 'DOT/USD', 'ADA/USD',
]

TRADING_DAYS_YEAR = 365  # 加密货币全年无休


# ── 基类 ──────────────────────────────────────────────────────────────────────

class CryptoBaseStrategy(ABC):
    """
    Crypto strategy base class. Every strategy implements generate_signal().

    Usage:
        class MyStrategy(CryptoBaseStrategy):
            name = "My Crypto Strategy"
            def generate_signal(self, data_dict):
                return signal_df  # (date x symbol), positive=long, negative=short, 0=flat
    """
    name: str = "未命名加密策略"
    description: str = ""

    def __init__(self, symbols=None, capital=10000):
        """
        初始化策略。

        参数:
            symbols: 交易对列表, 默认全部8个
            capital: 初始资金
        """
        self.symbols = symbols or CRYPTO_PAIRS.copy()
        self.capital = capital

    @abstractmethod
    def generate_signal(self, data_dict: dict) -> pd.DataFrame:
        """
        Generate the signal matrix (date x symbol).

        Args:
            data_dict: Dict keyed by trading pair, with DataFrames holding close/volume and other columns

        Returns:
            pd.DataFrame - rows=dates, columns=trading pairs, values=signal strength
        """
        raise NotImplementedError

    def get_params(self) -> dict:
        """Return the strategy parameters (used for logging and reporting)."""
        return {
            'name': self.name,
            'symbols': self.symbols,
            'capital': self.capital,
        }


# ── 动量策略 ──────────────────────────────────────────────────────────────────

class CryptoMomentumStrategy(CryptoBaseStrategy):
    """
    Cross-sectional crypto momentum strategy.

    Logic:
    - Use the 20-day return as the momentum signal
    - Rank across pairs: long the top 3, short the bottom 2
    - Inverse-volatility weighting: position weight is inversely proportional to volatility
    """
    name = "加密动量策略"
    description = "20日动量截面排名, 多空组合, 反波动率加权"

    def __init__(self, symbols=None, capital=10000, lookback=20,
                 long_n=3, short_n=2):
        """
        参数:
            lookback: 动量回看窗口 (天)
            long_n: 做多数量
            short_n: 做空数量
        """
        super().__init__(symbols, capital)
        self.lookback = lookback
        self.long_n = long_n
        self.short_n = short_n

    def generate_signal(self, data_dict: dict) -> pd.DataFrame:
        """Generate cross-sectional momentum signals with inverse-volatility weighting."""
        # 构建收盘价矩阵
        close = _build_close_matrix(data_dict, self.symbols)

        # 动量: lookback日收益率
        momentum = close.pct_change(self.lookback)

        # 滚动波动率 (用于反波动率加权)
        daily_ret = close.pct_change()
        vol = daily_ret.rolling(self.lookback).std()
        inv_vol = 1.0 / vol.replace(0, np.nan)

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        for date in close.index:
            mom_row = momentum.loc[date].dropna()
            if len(mom_row) < self.long_n + self.short_n:
                continue

            ranked = mom_row.sort_values(ascending=False)
            long_syms = ranked.head(self.long_n).index.tolist()
            short_syms = ranked.tail(self.short_n).index.tolist()

            # 反波动率权重
            iv_row = inv_vol.loc[date]
            long_iv = iv_row.reindex(long_syms).dropna()
            short_iv = iv_row.reindex(short_syms).dropna()

            if len(long_iv) > 0:
                long_w = long_iv / long_iv.sum()
                for s in long_w.index:
                    signals.loc[date, s] = long_w[s]

            if len(short_iv) > 0:
                short_w = short_iv / short_iv.sum()
                for s in short_w.index:
                    signals.loc[date, s] = -short_w[s]

        return signals

    def get_params(self) -> dict:
        base = super().get_params()
        base.update({
            'lookback': self.lookback,
            'long_n': self.long_n,
            'short_n': self.short_n,
        })
        return base


# ── 均值回归策略 ─────────────────────────────────────────────────────────────

class CryptoMeanReversionStrategy(CryptoBaseStrategy):
    """
    Crypto mean-reversion strategy.

    Logic:
    - Z-score = (close - 20-day SMA) / 20-day standard deviation
    - Z < -2: oversold, buy signal
    - Z > 2: overbought, sell signal
    - Signals are generated independently per asset
    """
    name = "加密均值回归策略"
    description = "Z-score均值回归, 超买超卖阈值触发"

    def __init__(self, symbols=None, capital=10000, window=20,
                 buy_threshold=-2.0, sell_threshold=2.0):
        """
        参数:
            window: SMA和标准差窗口
            buy_threshold: 买入Z分数阈值 (默认 -2)
            sell_threshold: 卖出Z分数阈值 (默认 2)
        """
        super().__init__(symbols, capital)
        self.window = window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def generate_signal(self, data_dict: dict) -> pd.DataFrame:
        """Generate mean-reversion signals from the Z-score."""
        close = _build_close_matrix(data_dict, self.symbols)

        sma = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()
        zscore = (close - sma) / std.replace(0, np.nan)

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        # 超卖买入, 超买卖出; 信号强度用z-score的负值 (越低越买)
        signals[zscore < self.buy_threshold] = -zscore[zscore < self.buy_threshold]
        signals[zscore > self.sell_threshold] = -zscore[zscore > self.sell_threshold]

        return signals

    def get_params(self) -> dict:
        base = super().get_params()
        base.update({
            'window': self.window,
            'buy_threshold': self.buy_threshold,
            'sell_threshold': self.sell_threshold,
        })
        return base


# ── 趋势跟踪策略 ─────────────────────────────────────────────────────────────

class CryptoTrendFollowStrategy(CryptoBaseStrategy):
    """
    Crypto dual-moving-average trend-following strategy.

    Logic:
    - Fast line: 10-day EMA; slow line: 50-day EMA
    - Fast > slow -> go long
    - Fast < slow -> go short / stay flat
    - Volume confirmation: volume on the crossover day must exceed the 20-day average volume
    """
    name = "加密趋势跟踪策略"
    description = "双EMA趋势跟踪, 成交量确认"

    def __init__(self, symbols=None, capital=10000, fast_period=10,
                 slow_period=50, vol_confirm_window=20, long_only=False):
        """
        参数:
            fast_period: 快线EMA周期
            slow_period: 慢线EMA周期
            vol_confirm_window: 成交量确认均值窗口
            long_only: 是否纯多头 (True则空头信号为0)
        """
        super().__init__(symbols, capital)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.vol_confirm_window = vol_confirm_window
        self.long_only = long_only

    def generate_signal(self, data_dict: dict) -> pd.DataFrame:
        """Generate dual-moving-average trend signals with volume confirmation."""
        close = _build_close_matrix(data_dict, self.symbols)
        volume = _build_field_matrix(data_dict, self.symbols, 'volume')

        ema_fast = close.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_period, adjust=False).mean()

        # 趋势方向: 快线在上=1, 在下=-1
        trend = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        trend[ema_fast > ema_slow] = 1.0
        trend[ema_fast < ema_slow] = -1.0 if not self.long_only else 0.0

        # 检测交叉点
        prev_trend = trend.shift(1)
        crossover = (trend != prev_trend) & prev_trend.notna()

        # 成交量确认: 交叉日成交量 > 均量
        avg_vol = volume.rolling(self.vol_confirm_window).mean()
        vol_confirm = volume > avg_vol

        # 只在有效交叉处更新信号, 之后保持信号直到下次交叉
        signals = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

        for col in close.columns:
            current_signal = 0.0
            for i, date in enumerate(close.index):
                if crossover.loc[date, col]:
                    # 交叉发生, 检查成交量
                    if vol_confirm.loc[date, col]:
                        current_signal = trend.loc[date, col]
                    # 成交量不足, 保持旧信号
                else:
                    # 非交叉日, 跟随趋势 (允许慢慢跟进)
                    if not np.isnan(trend.loc[date, col]) and i >= self.slow_period:
                        current_signal = trend.loc[date, col]
                signals.loc[date, col] = current_signal

        return signals.fillna(0.0)

    def get_params(self) -> dict:
        base = super().get_params()
        base.update({
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'vol_confirm_window': self.vol_confirm_window,
            'long_only': self.long_only,
        })
        return base


# ── BTC Beta策略 ──────────────────────────────────────────────────────────────

class CryptoBTCBetaStrategy(CryptoBaseStrategy):
    """
    Crypto BTC beta strategy.

    Logic:
    - Compute each altcoin's rolling beta to BTC
    - BTC uptrend (20-day SMA > 50-day SMA): long high-beta altcoins (amplify returns)
    - BTC downtrend: long low-beta altcoins or short high-beta ones
    """
    name = "加密BTC Beta策略"
    description = "基于BTC趋势和山寨币beta的轮动策略"

    def __init__(self, symbols=None, capital=10000, beta_window=60,
                 btc_fast=20, btc_slow=50, top_n=3):
        """
        参数:
            beta_window: 滚动beta计算窗口
            btc_fast: BTC趋势快线SMA
            btc_slow: BTC趋势慢线SMA
            top_n: 持仓数量
        """
        super().__init__(symbols, capital)
        self.beta_window = beta_window
        self.btc_fast = btc_fast
        self.btc_slow = btc_slow
        self.top_n = top_n

    def generate_signal(self, data_dict: dict) -> pd.DataFrame:
        """Generate signals from the BTC trend and each coin's beta."""
        close = _build_close_matrix(data_dict, self.symbols)
        daily_ret = close.pct_change()

        # BTC趋势判断
        btc_col = 'BTC/USD'
        if btc_col not in close.columns:
            raise ValueError("数据中必须包含BTC/USD")

        btc_close = close[btc_col]
        btc_sma_fast = btc_close.rolling(self.btc_fast).mean()
        btc_sma_slow = btc_close.rolling(self.btc_slow).mean()
        btc_uptrend = btc_sma_fast > btc_sma_slow  # True=上升趋势

        # 山寨币列表 (排除BTC)
        alt_cols = [c for c in close.columns if c != btc_col]
        btc_ret = daily_ret[btc_col]

        # 计算滚动beta
        beta_df = pd.DataFrame(index=close.index, columns=alt_cols, dtype=float)
        for col in alt_cols:
            alt_ret = daily_ret[col]
            cov = alt_ret.rolling(self.beta_window).cov(btc_ret)
            var = btc_ret.rolling(self.beta_window).var()
            beta_df[col] = cov / var.replace(0, np.nan)

        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        for date in close.index:
            betas = beta_df.loc[date].dropna()
            if len(betas) < self.top_n:
                continue

            ranked = betas.sort_values(ascending=False)

            if btc_uptrend.loc[date]:
                # BTC上升: 做多高beta山寨币
                top = ranked.head(self.top_n).index
                w = 1.0 / self.top_n
                for s in top:
                    signals.loc[date, s] = w
                # BTC本身也做多
                signals.loc[date, btc_col] = w * 0.5
            else:
                # BTC下降: 做多低beta, 做空高beta
                low_beta = ranked.tail(self.top_n).index
                high_beta = ranked.head(self.top_n).index
                w = 1.0 / self.top_n
                for s in low_beta:
                    signals.loc[date, s] = w * 0.5
                for s in high_beta:
                    signals.loc[date, s] = -w * 0.5

        return signals

    def get_params(self) -> dict:
        base = super().get_params()
        base.update({
            'beta_window': self.beta_window,
            'btc_fast': self.btc_fast,
            'btc_slow': self.btc_slow,
            'top_n': self.top_n,
        })
        return base


# ── 波动率目标策略 ────────────────────────────────────────────────────────────

class CryptoVolTargetStrategy(CryptoBaseStrategy):
    """
    Crypto volatility-targeting strategy.

    Logic:
    - Target 15% annualized volatility
    - Scale positions inversely to realized volatility
    - Layered on top of a momentum or trend signal: cut positions when volatility is high,
      scale them up when it is low
    """
    name = "加密波动率目标策略"
    description = "目标年化波动率15%, 仓位随实际波动率反向调整"

    def __init__(self, symbols=None, capital=10000, target_vol=0.15,
                 vol_window=20, inner_strategy=None, max_leverage=2.0):
        """
        参数:
            target_vol: 目标年化波动率
            vol_window: 波动率估计窗口
            inner_strategy: 内层信号策略 (默认用动量策略)
            max_leverage: 最大杠杆倍数
        """
        super().__init__(symbols, capital)
        self.target_vol = target_vol
        self.vol_window = vol_window
        self.max_leverage = max_leverage
        self.inner_strategy = inner_strategy or CryptoMomentumStrategy(
            symbols=symbols, capital=capital
        )

    def generate_signal(self, data_dict: dict) -> pd.DataFrame:
        """Apply volatility-target scaling on top of the inner strategy's signal."""
        # 获取内层策略信号
        raw_signal = self.inner_strategy.generate_signal(data_dict)

        close = _build_close_matrix(data_dict, self.symbols)
        daily_ret = close.pct_change()

        # 投资组合收益 (按信号加权)
        port_ret = (raw_signal.shift(1) * daily_ret).sum(axis=1)

        # 实际年化波动率
        realized_vol = port_ret.rolling(self.vol_window).std() * np.sqrt(TRADING_DAYS_YEAR)

        # 缩放因子 = 目标波动率 / 实际波动率
        scale = self.target_vol / realized_vol.replace(0, np.nan)
        scale = scale.clip(upper=self.max_leverage).fillna(1.0)

        # 逐行缩放信号
        signals = raw_signal.multiply(scale, axis=0)

        return signals

    def get_params(self) -> dict:
        base = super().get_params()
        base.update({
            'target_vol': self.target_vol,
            'vol_window': self.vol_window,
            'max_leverage': self.max_leverage,
            'inner_strategy': self.inner_strategy.name,
        })
        return base


# ── 回测函数 ──────────────────────────────────────────────────────────────────

def run_crypto_backtest(strategy, data_dict, initial_capital=10000):
    """
    Simple vectorized daily crypto backtest.

    Args:
        strategy: A CryptoBaseStrategy instance
        data_dict: Dict {symbol: DataFrame(close, volume, ...)}
        initial_capital: Initial capital

    Returns:
        dict with total_return, sharpe, max_drawdown, daily_returns, equity_curve
    """
    signals = strategy.generate_signal(data_dict)
    close = _build_close_matrix(data_dict, strategy.symbols)
    daily_ret = close.pct_change()

    # 对齐
    common_idx = signals.index.intersection(daily_ret.index)
    signals = signals.loc[common_idx]
    daily_ret = daily_ret.loc[common_idx]

    # 投资组合日收益 (信号滞后一天, 避免前视偏差)
    port_ret = (signals.shift(1) * daily_ret).sum(axis=1)
    port_ret = port_ret.fillna(0)

    # 净值曲线
    equity = initial_capital * (1 + port_ret).cumprod()

    # 最大回撤
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd = drawdown.min()

    # 夏普比率 (年化, 365天)
    if port_ret.std() > 0:
        sharpe = port_ret.mean() / port_ret.std() * np.sqrt(TRADING_DAYS_YEAR)
    else:
        sharpe = 0.0

    total_ret = equity.iloc[-1] / initial_capital - 1 if len(equity) > 0 else 0.0

    # 年化收益
    n_days = (common_idx[-1] - common_idx[0]).days if len(common_idx) > 1 else 1
    years = max(n_days / 365.25, 0.01)
    cagr = (1 + total_ret) ** (1 / years) - 1

    return {
        'total_return': total_ret,
        'cagr': cagr,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'daily_returns': port_ret,
        'equity_curve': equity,
        'n_days': n_days,
        'strategy': strategy.name,
    }


# ── 数据准备函数 ─────────────────────────────────────────────────────────────

def prepare_crypto_data(alpaca_loader, lookback_days=365):
    """
    Fetch and normalize crypto data from Alpaca.

    Args:
        alpaca_loader: A loader exposing get_crypto_bars(symbol, timeframe, start, end)
        lookback_days: Number of lookback days

    Returns:
        dict - keyed by trading pair, with DataFrames holding close/open/high/low/volume/vwap/returns
    """
    end = pd.Timestamp.now(tz='UTC').normalize()
    start = end - pd.Timedelta(days=lookback_days)

    data_dict = {}

    for symbol in CRYPTO_PAIRS:
        try:
            bars = alpaca_loader.get_crypto_bars(
                symbol,
                timeframe='1Day',
                start=start.isoformat(),
                end=end.isoformat(),
            )

            if bars is None or len(bars) == 0:
                print(f"  跳过 {symbol}: 无数据")
                continue

            df = bars if isinstance(bars, pd.DataFrame) else bars.df

            # 确保列名统一
            col_map = {}
            for col in ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']:
                for c in df.columns:
                    if c.lower() == col:
                        col_map[c] = col
            df = df.rename(columns=col_map)

            # 计算日收益率
            df['returns'] = df['close'].pct_change()

            data_dict[symbol] = df
            print(f"  {symbol}: {len(df)} 日数据")

        except Exception as e:
            print(f"  {symbol} 获取失败: {e}")

    if len(data_dict) == 0:
        raise RuntimeError("未获取到任何加密货币数据")

    print(f"\n成功获取 {len(data_dict)}/{len(CRYPTO_PAIRS)} 个交易对数据")
    return data_dict


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _build_close_matrix(data_dict, symbols):
    """从data_dict构建收盘价矩阵 (date x symbol)。"""
    frames = {}
    for sym in symbols:
        if sym in data_dict:
            df = data_dict[sym]
            frames[sym] = df['close'] if 'close' in df.columns else df.iloc[:, 3]

    if not frames:
        raise ValueError("data_dict 中无有效数据")

    close = pd.DataFrame(frames)
    close.index = pd.DatetimeIndex(close.index)
    close = close.sort_index()
    return close


def _build_field_matrix(data_dict, symbols, field):
    """从data_dict构建指定字段的矩阵 (date x symbol)。"""
    frames = {}
    for sym in symbols:
        if sym in data_dict and field in data_dict[sym].columns:
            frames[sym] = data_dict[sym][field]

    if not frames:
        return pd.DataFrame()

    mat = pd.DataFrame(frames)
    mat.index = pd.DatetimeIndex(mat.index)
    mat = mat.sort_index()
    return mat
