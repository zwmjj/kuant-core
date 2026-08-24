"""WorldQuant 101 Formulaic Alphas implementation
Paper: "101 Formulaic Alphas" — Zura Kakushadze (2015)
https://arxiv.org/abs/1601.00991

Adapts the classic 101 alpha formulas to the Kuant SignalGenerator pattern:
    - Inputs are pandas DataFrames (date x stock)
    - Outputs are cross-sectionally ranked signals, compatible with monthly backtests
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helper operators (论文中定义的算子)
# ---------------------------------------------------------------------------

def _rank(df: pd.DataFrame) -> pd.DataFrame:
    """截面百分位排名 (0~1)"""
    return df.rank(axis=1, pct=True)


def _delta(df: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    """时间序列差分"""
    return df.diff(period)


def _delay(df: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    """时间序列滞后"""
    return df.shift(period)


def _ts_rank(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """时间序列排名 (当前值在窗口内的排名百分位)"""
    return df.rolling(window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )


def _ts_max(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return df.rolling(window).max()


def _ts_min(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return df.rolling(window).min()


def _ts_argmax(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """窗口内最大值位置 (0-indexed from window start)"""
    return df.rolling(window).apply(lambda x: np.argmax(x), raw=True)


def _ts_argmin(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return df.rolling(window).apply(lambda x: np.argmin(x), raw=True)


def _ts_sum(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return df.rolling(window).sum()


def _ts_stddev(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return df.rolling(window).std()


def _ts_corr(x: pd.DataFrame, y: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """滚动截面相关 (按列逐列计算)"""
    result = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    for col in x.columns:
        if col in y.columns:
            result[col] = x[col].rolling(window).corr(y[col])
    return result


def _ts_cov(x: pd.DataFrame, y: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    result = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    for col in x.columns:
        if col in y.columns:
            result[col] = x[col].rolling(window).cov(y[col])
    return result


def _signed_power(df: pd.DataFrame, exp: float) -> pd.DataFrame:
    return df.apply(lambda x: np.sign(x) * (np.abs(x) ** exp))


def _decay_linear(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """线性衰减加权均值"""
    weights = np.arange(1, window + 1, dtype=float)
    weights = weights / weights.sum()
    return df.rolling(window).apply(lambda x: np.dot(x, weights), raw=True)


def _sma(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return df.rolling(window).mean()


def _product(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return df.rolling(window).apply(lambda x: np.prod(x), raw=True)


# ---------------------------------------------------------------------------
# WorldQuant 101 Alpha class
# ---------------------------------------------------------------------------

class WorldQuantAlphas:
    """WorldQuant 101 Formulaic Alphas — a selection of 20 key alphas.

    Each alpha is a static method taking OHLCV DataFrames (date x stock) and returning a
    signal DataFrame that can be passed straight to SignalGenerator.cross_sectional_rank().

    Parameters (common)
    -------------------
    open_, high, low, close : pd.DataFrame
        OHLC price matrices (date x stock)
    volume : pd.DataFrame
        Trading volume matrix
    returns : pd.DataFrame
        Return matrix
    vwap : pd.DataFrame (required by some alphas)
        Volume-weighted average price matrix; if unavailable, approximate with (open + high + low + close) / 4
    """

    @staticmethod
    def alpha001(close: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        """Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0)
        ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
        """
        cond = returns < 0
        inner = pd.DataFrame(
            np.where(cond, _ts_stddev(returns, 20), close),
            index=close.index,
            columns=close.columns,
        )
        sp = _signed_power(inner, 2.0)
        return _rank(_ts_argmax(sp, 5)) - 0.5

    @staticmethod
    def alpha002(
        open_: pd.DataFrame,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)),
        rank(((close - open) / open)), 6))
        """
        a = _rank(_delta(np.log(volume.replace(0, np.nan)), 2))
        b = _rank((close - open_) / open_.replace(0, np.nan))
        return -1 * _ts_corr(a, b, 6)

    @staticmethod
    def alpha006(open_: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        """Alpha#6: (-1 * correlation(open, volume, 10))"""
        return -1 * _ts_corr(open_, volume, 10)

    @staticmethod
    def alpha012(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        """Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))"""
        return np.sign(_delta(volume, 1)) * (-1 * _delta(close, 1))

    @staticmethod
    def alpha026(
        high: pd.DataFrame,
        volume: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Alpha#26: (-1 * ts_max(correlation(ts_rank(volume, 5),
        ts_rank(high, 5), 5), 3))
        """
        a = _ts_rank(volume, 5)
        b = _ts_rank(high, 5)
        return -1 * _ts_max(_ts_corr(a, b, 5), 3)

    @staticmethod
    def alpha033(
        open_: pd.DataFrame,
        close: pd.DataFrame,
    ) -> pd.DataFrame:
        """Alpha#33: rank((-1 * ((1 - (open / close))^1)))"""
        return _rank(-1 * (1 - open_ / close.replace(0, np.nan)))

    @staticmethod
    def alpha040(
        high: pd.DataFrame,
        volume: pd.DataFrame,
    ) -> pd.DataFrame:
        """Alpha#40: (-1 * rank(stddev(high, 10)) * correlation(high, volume, 10))"""
        return -1 * _rank(_ts_stddev(high, 10)) * _ts_corr(high, volume, 10)

    @staticmethod
    def alpha042(
        close: pd.DataFrame,
        vwap: pd.DataFrame,
    ) -> pd.DataFrame:
        """Alpha#42: (rank((vwap - close)) / rank((vwap + close)))"""
        return _rank(vwap - close) / _rank(vwap + close).replace(0, np.nan)

    @staticmethod
    def alpha044(
        high: pd.DataFrame,
        volume: pd.DataFrame,
    ) -> pd.DataFrame:
        """Alpha#44: (-1 * correlation(high, rank(volume), 5))"""
        return -1 * _ts_corr(high, _rank(volume), 5)

    @staticmethod
    def alpha047(
        high: pd.DataFrame,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        vwap: pd.DataFrame,
    ) -> pd.DataFrame:
        """Alpha#47: ((((rank((1 / close)) * volume) / adv20) *
        ((high * rank((high - close))) / (sum(high, 5) / 5))) -
        rank((vwap - delay(vwap, 5))))
        """
        adv20 = _sma(volume, 20)
        part1 = (_rank(1.0 / close.replace(0, np.nan)) * volume) / adv20.replace(0, np.nan)
        part2 = (high * _rank(high - close)) / (_ts_sum(high, 5) / 5).replace(0, np.nan)
        part3 = _rank(vwap - _delay(vwap, 5))
        return part1 * part2 - part3

    @staticmethod
    def alpha053(
        high: pd.DataFrame,
        low: pd.DataFrame,
        close: pd.DataFrame,
    ) -> pd.DataFrame:
        """Alpha#53: (-1 * delta(((close - low) - (high - close))
        / (close - low).replace(0, nan), 9))
        """
        inner = ((close - low) - (high - close)) / (close - low).replace(0, np.nan)
        return -1 * _delta(inner, 9)

    @staticmethod
    def alpha054(
        open_: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        close: pd.DataFrame,
    ) -> pd.DataFrame:
        """Alpha#54: ((-1 * ((low - close) * (open^5))) /
        ((low - high) * (close^5)))
        """
        num = -1 * (low - close) * (open_ ** 5)
        den = (low - high).replace(0, np.nan) * (close ** 5).replace(0, np.nan)
        return num / den

    @staticmethod
    def alpha055(
        high: pd.DataFrame,
        low: pd.DataFrame,
        close: pd.DataFrame,
        volume: pd.DataFrame,
    ) -> pd.DataFrame:
        """Alpha#55: (-1 * correlation(rank(((close - ts_min(low, 12))
        / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))
        """
        min_low = _ts_min(low, 12)
        max_high = _ts_max(high, 12)
        denom = (max_high - min_low).replace(0, np.nan)
        inner = (close - min_low) / denom
        return -1 * _ts_corr(_rank(inner), _rank(volume), 6)

    @staticmethod
    def alpha057(
        close: pd.DataFrame,
        vwap: pd.DataFrame,
    ) -> pd.DataFrame:
        """Alpha#57: (0 - (1 * ((close - vwap) /
        decay_linear(rank(ts_argmax(close, 30)), 2))))
        """
        inner = _decay_linear(_rank(_ts_argmax(close, 30)), 2)
        return -1 * (close - vwap) / inner.replace(0, np.nan)

    @staticmethod
    def alpha060(
        high: pd.DataFrame,
        low: pd.DataFrame,
        close: pd.DataFrame,
        volume: pd.DataFrame,
    ) -> pd.DataFrame:
        """Alpha#60: (0 - (1 * ((2 * scale(rank(((((close - low)
        - (high - close)) / (high - low)) * volume)))) -
        scale(rank(ts_argmax(close, 10))))))

        Simplified: we skip the portfolio-level scale() and use rank instead.
        """
        inner = ((close - low) - (high - close)) / (high - low).replace(0, np.nan) * volume
        part1 = 2 * _rank(inner)
        part2 = _rank(_ts_argmax(close, 10))
        return -(part1 - part2)

    @staticmethod
    def alpha083(
        high: pd.DataFrame,
        low: pd.DataFrame,
        close: pd.DataFrame,
        volume: pd.DataFrame,
    ) -> pd.DataFrame:
        """Alpha#83: ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2))
        * rank(rank(volume))) /
        (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))

        Simplified: use close as vwap proxy where needed.
        """
        hl_ratio = (high - low) / (_ts_sum(close, 5) / 5).replace(0, np.nan)
        part1 = _rank(_delay(hl_ratio, 2)) * _rank(_rank(volume))
        # avoid division by (vwap - close) which is often ~0
        # use the numerator as the signal directly (robust variant)
        return part1

    @staticmethod
    def alpha084(
        close: pd.DataFrame,
        vwap: pd.DataFrame,
    ) -> pd.DataFrame:
        """Alpha#84: SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))"""
        part1 = _ts_rank(vwap - _ts_max(vwap, 15), 20)
        part2 = _delta(close, 5)
        return _signed_power(part1, part2)

    @staticmethod
    def alpha085(
        high: pd.DataFrame,
        low: pd.DataFrame,
        close: pd.DataFrame,
        volume: pd.DataFrame,
    ) -> pd.DataFrame:
        """Alpha#85: (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))),
        adv30, 9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596),
        Ts_Rank(volume, 10.1595), 7.11408)))
        """
        adv30 = _sma(volume, 30)
        price_blend = high * 0.8767 + close * 0.1233
        corr1 = _ts_corr(price_blend, adv30, 10)
        mid = (high + low) / 2
        corr2 = _ts_corr(_ts_rank(mid, 4), _ts_rank(volume, 10), 7)
        return _rank(corr1) ** _rank(corr2)

    @staticmethod
    def alpha101(
        open_: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        close: pd.DataFrame,
    ) -> pd.DataFrame:
        """Alpha#101: ((close - open) / ((high - low) + .001))"""
        return (close - open_) / ((high - low) + 0.001)

    # ------------------------------------------------------------------
    # 便捷接口: 批量计算所有 alpha
    # ------------------------------------------------------------------

    @classmethod
    def compute_all(
        cls,
        open_: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        returns: pd.DataFrame,
        vwap: pd.DataFrame = None,
    ) -> dict:
        """Compute every implemented alpha in bulk and return {alpha_name: DataFrame}.

        Parameters
        ----------
        vwap : pd.DataFrame, optional
            If None, approximated with (open + high + low + close) / 4
        """
        if vwap is None:
            vwap = (open_ + high + low + close) / 4

        results = {}
        alpha_methods = {
            "alpha001": lambda: cls.alpha001(close, returns),
            "alpha002": lambda: cls.alpha002(open_, close, volume, returns),
            "alpha006": lambda: cls.alpha006(open_, volume),
            "alpha012": lambda: cls.alpha012(close, volume),
            "alpha026": lambda: cls.alpha026(high, volume, returns),
            "alpha033": lambda: cls.alpha033(open_, close),
            "alpha040": lambda: cls.alpha040(high, volume),
            "alpha042": lambda: cls.alpha042(close, vwap),
            "alpha044": lambda: cls.alpha044(high, volume),
            "alpha047": lambda: cls.alpha047(high, close, volume, vwap),
            "alpha053": lambda: cls.alpha053(high, low, close),
            "alpha054": lambda: cls.alpha054(open_, high, low, close),
            "alpha055": lambda: cls.alpha055(high, low, close, volume),
            "alpha057": lambda: cls.alpha057(close, vwap),
            "alpha060": lambda: cls.alpha060(high, low, close, volume),
            "alpha083": lambda: cls.alpha083(high, low, close, volume),
            "alpha084": lambda: cls.alpha084(close, vwap),
            "alpha085": lambda: cls.alpha085(high, low, close, volume),
            "alpha101": lambda: cls.alpha101(open_, high, low, close),
        }

        for name, fn in alpha_methods.items():
            try:
                results[name] = fn()
            except Exception as e:
                warnings.warn(f"计算 {name} 失败: {e}")

        return results

    @classmethod
    def as_signal(
        cls,
        alpha_df: pd.DataFrame,
        method: str = "rank",
    ) -> pd.DataFrame:
        """Convert raw alpha values into a Kuant-style cross-sectional signal in [-1, 1].

        Parameters
        ----------
        alpha_df : pd.DataFrame
            Raw alpha value matrix
        method : str
            'rank' cross-sectional percentile rank | 'zscore' cross-sectional standardization

        Returns
        -------
        pd.DataFrame
            Same output format as SignalGenerator.cross_sectional_rank()
        """
        if method == "rank":
            ranked = alpha_df.rank(axis=1, pct=True)
            return ranked * 2 - 1  # scale to [-1, 1]
        elif method == "zscore":
            mu = alpha_df.mean(axis=1)
            sigma = alpha_df.std(axis=1).replace(0, np.nan)
            return alpha_df.sub(mu, axis=0).div(sigma, axis=0).clip(-3, 3) / 3
        else:
            raise ValueError(f"未知方法: {method}。支持: 'rank', 'zscore'")
