"""市场体制(Regime)检测系统 — 动态识别趋势/波动率/相关性/流动性/利率/风险偏好体制，
用于策略权重的自适应调整。

数据源: yfinance
输出: 各维度当前体制标签 + 历史体制序列 + 策略权重建议 + 回测比较
"""
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════
#  常量
# ════════════════════════════════════════════════════════════════════════

SECTOR_ETFS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
REQUIRED_TICKERS = ["SPY", "GLD", "TLT", "SHY", "HYG"] + SECTOR_ETFS

# 体制枚举
TREND_REGIMES = ("bull", "bear", "sideways")
VOL_REGIMES = ("low_vol", "normal_vol", "high_vol", "crisis")
CORR_REGIMES = ("low_corr", "normal_corr", "high_corr")
LIQ_REGIMES = ("ample", "normal_liq", "tight")
RATE_REGIMES = ("easing", "neutral_rate", "tightening")
RISK_REGIMES = ("risk_on", "neutral_risk", "risk_off")


# ════════════════════════════════════════════════════════════════════════
#  数据获取
# ════════════════════════════════════════════════════════════════════════

def fetch_market_data(
    start: str = "2006-01-01",
    end: Optional[str] = None,
    tickers: Optional[list] = None,
) -> Dict[str, pd.DataFrame]:
    """通过 yfinance 下载所需 ETF 的 OHLCV 数据。

    返回:
        dict 包含 'close', 'high', 'low', 'volume' 四个 DataFrame,
        index=日期, columns=ticker
    """
    tickers = tickers or REQUIRED_TICKERS
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    # yfinance 返回 MultiIndex columns: (field, ticker)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    high = raw["High"] if isinstance(raw.columns, pd.MultiIndex) else raw[["High"]]
    low = raw["Low"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Low"]]
    volume = raw["Volume"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Volume"]]
    # 确保 columns 是扁平 ticker
    for df in (close, high, low, volume):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)
    return {"close": close.ffill(), "high": high.ffill(), "low": low.ffill(), "volume": volume.ffill()}


# ════════════════════════════════════════════════════════════════════════
#  体制检测器
# ════════════════════════════════════════════════════════════════════════

class RegimeDetector:
    """多维度市场体制检测器

    支持六大维度: 趋势、波动率、相关性、流动性、利率、风险偏好。
    所有方法均为纯函数——传入 market_data (dict of DataFrames), 输出体制标签。
    """

    # ──────────────────────────────────────────────────────────
    #  1. 趋势体制: SPY 50d/200d SMA 交叉 + ADX 代理
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _adx_proxy(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """简化版 ADX 代理: 用 True Range 的指数移动均值衡量趋势强度。

        非标准 ADX, 但与价格波幅成正比, 在纯价格数据下足够区分趋势/震荡。
        """
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=window, min_periods=window).mean()
        # 用价格变化的绝对值 vs ATR 的比率作为趋势强度
        price_change = close.diff(window).abs()
        strength = price_change / (atr * np.sqrt(window) + 1e-10)
        return strength

    @staticmethod
    def detect_trend(market_data: dict) -> pd.Series:
        """趋势体制序列: bull / bear / sideways

        逻辑:
        - SMA50 > SMA200 且 ADX代理 > 阈值 → bull
        - SMA50 < SMA200 且 ADX代理 > 阈值 → bear
        - 其余 → sideways
        """
        close = market_data["close"]["SPY"]
        high = market_data["high"]["SPY"]
        low = market_data["low"]["SPY"]

        sma50 = close.rolling(50, min_periods=50).mean()
        sma200 = close.rolling(200, min_periods=200).mean()
        adx = RegimeDetector._adx_proxy(high, low, close, window=14)
        adx_thresh = adx.rolling(252, min_periods=60).median()

        regime = pd.Series("sideways", index=close.index)
        regime[(sma50 > sma200) & (adx > adx_thresh)] = "bull"
        regime[(sma50 < sma200) & (adx > adx_thresh)] = "bear"
        regime.name = "trend"
        return regime

    # ──────────────────────────────────────────────────────────
    #  2. 波动率体制: 20d 已实现波动率的 252d 百分位排名
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def detect_volatility(market_data: dict) -> pd.Series:
        """波动率体制序列: low_vol / normal_vol / high_vol / crisis

        20d 年化已实现波动率在过去 252 个交易日中的百分位:
        - < 25%  → low_vol
        - 25-75% → normal_vol
        - 75-95% → high_vol
        - > 95%  → crisis
        """
        ret = market_data["close"]["SPY"].pct_change()
        rvol = ret.rolling(20, min_periods=15).std() * np.sqrt(252)

        def pct_rank(s):
            """滚动百分位排名"""
            out = s.copy() * np.nan
            vals = s.values
            for i in range(252, len(vals)):
                window = vals[i - 252 : i + 1]
                valid = window[~np.isnan(window)]
                if len(valid) < 60:
                    continue
                out.iloc[i] = (valid < vals[i]).sum() / len(valid)
            return out

        prank = pct_rank(rvol)

        regime = pd.Series("normal_vol", index=rvol.index)
        regime[prank < 0.25] = "low_vol"
        regime[(prank >= 0.75) & (prank < 0.95)] = "high_vol"
        regime[prank >= 0.95] = "crisis"
        regime.name = "volatility"
        return regime

    # ──────────────────────────────────────────────────────────
    #  3. 相关性体制: 行业 ETF 20d 滚动平均成对相关
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def detect_correlation(market_data: dict) -> pd.Series:
        """相关性体制序列: low_corr / normal_corr / high_corr

        行业 ETF 20d 滚动收益率成对相关的均值:
        - < 0.30 → low_corr
        - 0.30-0.65 → normal_corr
        - > 0.65 → high_corr
        """
        available = [t for t in SECTOR_ETFS if t in market_data["close"].columns]
        if len(available) < 4:
            # 数据不足，返回全 normal
            return pd.Series("normal_corr", index=market_data["close"].index, name="correlation")

        rets = market_data["close"][available].pct_change()
        window = 20
        n = len(available)
        n_pairs = n * (n - 1) / 2

        avg_corr = pd.Series(np.nan, index=rets.index)
        for i in range(window, len(rets)):
            block = rets.iloc[i - window : i].dropna(axis=1, how="all")
            if block.shape[1] < 4:
                continue
            cm = block.corr()
            # 上三角均值
            mask = np.triu(np.ones(cm.shape, dtype=bool), k=1)
            avg_corr.iloc[i] = cm.values[mask].mean()

        regime = pd.Series("normal_corr", index=rets.index)
        regime[avg_corr < 0.30] = "low_corr"
        regime[avg_corr > 0.65] = "high_corr"
        regime.name = "correlation"
        return regime

    # ──────────────────────────────────────────────────────────
    #  4. 流动性体制: High-Low 范围代理 + 成交量趋势
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def detect_liquidity(market_data: dict) -> pd.Series:
        """流动性体制序列: ample / normal_liq / tight

        代理指标:
        - Bid-ask 代理: (High-Low)/Close 的 20d 均值 (越大=越差)
        - 成交量趋势: 20d 成交量均值 / 60d 成交量均值
        综合评分的百分位:
        - < 30% → ample
        - 30-70% → normal_liq
        - > 70% → tight
        """
        spy_close = market_data["close"]["SPY"]
        spy_high = market_data["high"]["SPY"]
        spy_low = market_data["low"]["SPY"]
        spy_vol = market_data["volume"]["SPY"]

        spread_proxy = ((spy_high - spy_low) / spy_close).rolling(20, min_periods=15).mean()
        vol_ratio = spy_vol.rolling(20, min_periods=15).mean() / spy_vol.rolling(60, min_periods=40).mean()

        # 标准化 (滚动 z-score)
        def rolling_z(s, w=252):
            m = s.rolling(w, min_periods=60).mean()
            sd = s.rolling(w, min_periods=60).std()
            return (s - m) / (sd + 1e-10)

        # 高 spread = 差, 低 volume ratio = 差 → 综合评分越高=流动性越紧
        score = rolling_z(spread_proxy) - rolling_z(vol_ratio)
        # 百分位排名
        prank = score.rolling(252, min_periods=60).apply(
            lambda x: (x.iloc[:-1] < x.iloc[-1]).sum() / (len(x) - 1), raw=False
        )

        regime = pd.Series("normal_liq", index=spy_close.index)
        regime[prank < 0.30] = "ample"
        regime[prank > 0.70] = "tight"
        regime.name = "liquidity"
        return regime

    # ──────────────────────────────────────────────────────────
    #  5. 利率体制: TLT/SHY 比率趋势 (收益率曲线代理)
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def detect_rate(market_data: dict) -> pd.Series:
        """利率体制序列: easing / neutral_rate / tightening

        TLT/SHY 比率反映长端 vs 短端债券价格:
        - 比率上升 (长端涨、收益率降) → easing
        - 比率下降 (长端跌、收益率升) → tightening
        - 持平 → neutral_rate

        用 60d 变化率的 z-score 判断。
        """
        close = market_data["close"]
        if "TLT" not in close.columns or "SHY" not in close.columns:
            return pd.Series("neutral_rate", index=close.index, name="rate")

        ratio = close["TLT"] / close["SHY"]
        change = ratio.pct_change(60)  # 60个交易日变化率
        z = (change - change.rolling(252, min_periods=60).mean()) / (
            change.rolling(252, min_periods=60).std() + 1e-10
        )

        regime = pd.Series("neutral_rate", index=close.index)
        regime[z > 0.8] = "easing"
        regime[z < -0.8] = "tightening"
        regime.name = "rate"
        return regime

    # ──────────────────────────────────────────────────────────
    #  6. 风险偏好: SPY/GLD + HYG/TLT 综合信号
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def detect_risk_appetite(market_data: dict) -> pd.Series:
        """风险偏好体制序列: risk_on / neutral_risk / risk_off

        两个比率的 z-score 均值:
        - SPY/GLD: 股票 vs 避险黄金
        - HYG/TLT: 高收益债 vs 国债

        z > 0.5 → risk_on, z < -0.5 → risk_off, 其余 → neutral_risk
        """
        close = market_data["close"]
        available_pairs = []
        if "SPY" in close.columns and "GLD" in close.columns:
            available_pairs.append(close["SPY"] / close["GLD"])
        if "HYG" in close.columns and "TLT" in close.columns:
            available_pairs.append(close["HYG"] / close["TLT"])

        if not available_pairs:
            return pd.Series("neutral_risk", index=close.index, name="risk_appetite")

        def ratio_z(ratio, lookback=60, z_window=252):
            mom = ratio.pct_change(lookback)
            z = (mom - mom.rolling(z_window, min_periods=60).mean()) / (
                mom.rolling(z_window, min_periods=60).std() + 1e-10
            )
            return z

        z_scores = pd.concat([ratio_z(r) for r in available_pairs], axis=1)
        composite = z_scores.mean(axis=1)

        regime = pd.Series("neutral_risk", index=close.index)
        regime[composite > 0.5] = "risk_on"
        regime[composite < -0.5] = "risk_off"
        regime.name = "risk_appetite"
        return regime

    # ══════════════════════════════════════════════════════════════
    #  综合接口
    # ══════════════════════════════════════════════════════════════

    def detect_all(self, market_data: dict) -> dict:
        """检测所有维度的当前体制。

        参数:
            market_data: fetch_market_data() 的返回值
        返回:
            dict, key=维度名, value=当前体制标签 (str)
        """
        regimes = {}
        regimes["trend"] = self.detect_trend(market_data).dropna().iloc[-1]
        regimes["volatility"] = self.detect_volatility(market_data).dropna().iloc[-1]
        regimes["correlation"] = self.detect_correlation(market_data).dropna().iloc[-1]
        regimes["liquidity"] = self.detect_liquidity(market_data).dropna().iloc[-1]
        regimes["rate"] = self.detect_rate(market_data).dropna().iloc[-1]
        regimes["risk_appetite"] = self.detect_risk_appetite(market_data).dropna().iloc[-1]
        return regimes

    def get_regime_history(self, market_data: dict, lookback: int = 252) -> pd.DataFrame:
        """返回最近 lookback 个交易日的体制历史 DataFrame。

        参数:
            market_data: fetch_market_data() 的返回值
            lookback: 回看天数
        返回:
            DataFrame, index=日期, columns=[trend, volatility, correlation, liquidity, rate, risk_appetite]
        """
        trend = self.detect_trend(market_data)
        vol = self.detect_volatility(market_data)
        corr = self.detect_correlation(market_data)
        liq = self.detect_liquidity(market_data)
        rate = self.detect_rate(market_data)
        risk = self.detect_risk_appetite(market_data)

        df = pd.DataFrame({
            "trend": trend,
            "volatility": vol,
            "correlation": corr,
            "liquidity": liq,
            "rate": rate,
            "risk_appetite": risk,
        })
        return df.iloc[-lookback:]

    # ══════════════════════════════════════════════════════════════
    #  策略权重映射
    # ══════════════════════════════════════════════════════════════

    # 基础策略桶
    STRATEGY_NAMES = [
        "stocks",          # 股票多头 (SPY)
        "bonds",           # 国债 (TLT)
        "gold",            # 黄金 (GLD)
        "vol_arb",         # 波动率套利 / 期权策略
        "mean_reversion",  # 均值回归
        "calendar_spread", # 日历价差
        "cash",            # 现金
    ]

    # 默认静态等权 (用于回测基准)
    STATIC_WEIGHTS = {
        "stocks": 0.30,
        "bonds": 0.20,
        "gold": 0.10,
        "vol_arb": 0.10,
        "mean_reversion": 0.10,
        "calendar_spread": 0.10,
        "cash": 0.10,
    }

    @staticmethod
    def get_strategy_weights(regimes: dict) -> dict:
        """根据当前多维体制输出策略权重。

        核心逻辑:
        - Bull + Low Vol → 超配股票, 低配黄金
        - Bear + High Vol → 超配黄金 + vol_arb, 低配股票
        - Crisis → 最大现金, 买保护性看跌
        - Sideways → 超配日历价差 + 均值回归

        参数:
            regimes: detect_all() 的返回值
        返回:
            dict, key=策略名, value=权重 (总和=1)
        """
        trend = regimes.get("trend", "sideways")
        vol = regimes.get("volatility", "normal_vol")
        risk = regimes.get("risk_appetite", "neutral_risk")

        # ── 危机模式: 最高优先级 ──
        if vol == "crisis":
            w = {
                "stocks": 0.05,
                "bonds": 0.15,
                "gold": 0.20,
                "vol_arb": 0.10,  # 含保护性看跌
                "mean_reversion": 0.00,
                "calendar_spread": 0.00,
                "cash": 0.50,
            }
            return w

        # ── 牛市 + 低波 ──
        if trend == "bull" and vol in ("low_vol", "normal_vol"):
            base = {
                "stocks": 0.50,
                "bonds": 0.15,
                "gold": 0.05,
                "vol_arb": 0.05,
                "mean_reversion": 0.10,
                "calendar_spread": 0.05,
                "cash": 0.10,
            }
            # risk_on 进一步增配股票
            if risk == "risk_on":
                base["stocks"] += 0.05
                base["cash"] -= 0.05
            return base

        # ── 熊市 + 高波 ──
        if trend == "bear" and vol in ("high_vol", "crisis"):
            return {
                "stocks": 0.05,
                "bonds": 0.15,
                "gold": 0.30,
                "vol_arb": 0.20,
                "mean_reversion": 0.05,
                "calendar_spread": 0.00,
                "cash": 0.25,
            }

        # ── 熊市 (一般) ──
        if trend == "bear":
            return {
                "stocks": 0.10,
                "bonds": 0.20,
                "gold": 0.20,
                "vol_arb": 0.15,
                "mean_reversion": 0.10,
                "calendar_spread": 0.05,
                "cash": 0.20,
            }

        # ── 震荡市 ──
        if trend == "sideways":
            base = {
                "stocks": 0.15,
                "bonds": 0.15,
                "gold": 0.10,
                "vol_arb": 0.10,
                "mean_reversion": 0.25,
                "calendar_spread": 0.15,
                "cash": 0.10,
            }
            if vol == "high_vol":
                base["mean_reversion"] -= 0.05
                base["cash"] += 0.05
            return base

        # ── 牛市 + 高波 (动量但颠簸) ──
        return {
            "stocks": 0.30,
            "bonds": 0.15,
            "gold": 0.15,
            "vol_arb": 0.10,
            "mean_reversion": 0.10,
            "calendar_spread": 0.05,
            "cash": 0.15,
        }

    # ══════════════════════════════════════════════════════════════
    #  回测: 体制切换 vs 静态配置
    # ══════════════════════════════════════════════════════════════

    def backtest_regime_switching(
        self,
        market_data: dict,
        sub_strategy_returns: Optional[pd.DataFrame] = None,
        rebalance_freq: int = 5,
    ) -> dict:
        """回测体制切换策略 vs 静态配置。

        参数:
            market_data: fetch_market_data() 的返回值
            sub_strategy_returns: DataFrame, index=日期, columns=策略名, 值=日收益率
                若为 None 则使用简单代理 (SPY=stocks, TLT=bonds, GLD=gold, 其余=合成)
            rebalance_freq: 再平衡频率 (交易日)
        返回:
            dict 包含:
            - 'regime_sharpe': 体制切换策略年化 Sharpe
            - 'static_sharpe': 静态配置年化 Sharpe
            - 'regime_returns': Series (日收益)
            - 'static_returns': Series (日收益)
            - 'regime_cumulative': Series (累计净值)
            - 'static_cumulative': Series (累计净值)
            - 'regime_weights_history': DataFrame (权重时间序列)
        """
        # ── 子策略收益代理 ──
        if sub_strategy_returns is None:
            sub_strategy_returns = self._build_proxy_returns(market_data)

        # ── 计算历史体制 ──
        trend = self.detect_trend(market_data)
        vol = self.detect_volatility(market_data)
        corr = self.detect_correlation(market_data)
        liq = self.detect_liquidity(market_data)
        rate = self.detect_rate(market_data)
        risk = self.detect_risk_appetite(market_data)

        regime_df = pd.DataFrame({
            "trend": trend, "volatility": vol, "correlation": corr,
            "liquidity": liq, "rate": rate, "risk_appetite": risk,
        })

        # 对齐日期
        common_idx = regime_df.index.intersection(sub_strategy_returns.index)
        common_idx = common_idx.sort_values()
        regime_df = regime_df.loc[common_idx]
        sub_ret = sub_strategy_returns.loc[common_idx]

        # ── 逐日计算体制切换组合收益 ──
        strategies = [s for s in self.STRATEGY_NAMES if s in sub_ret.columns]
        regime_weights_hist = pd.DataFrame(0.0, index=common_idx, columns=strategies)
        regime_port_ret = pd.Series(0.0, index=common_idx)
        static_port_ret = pd.Series(0.0, index=common_idx)

        current_weights = {s: self.STATIC_WEIGHTS.get(s, 0) for s in strategies}

        for i, date in enumerate(common_idx):
            # 再平衡
            if i % rebalance_freq == 0:
                row = regime_df.loc[date]
                r = {
                    "trend": row["trend"],
                    "volatility": row["volatility"],
                    "correlation": row["correlation"],
                    "liquidity": row["liquidity"],
                    "rate": row["rate"],
                    "risk_appetite": row["risk_appetite"],
                }
                w = self.get_strategy_weights(r)
                current_weights = {s: w.get(s, 0) for s in strategies}

            for s in strategies:
                regime_weights_hist.loc[date, s] = current_weights[s]

            # 体制切换组合日收益
            day_ret = sum(current_weights[s] * sub_ret.loc[date, s] for s in strategies)
            regime_port_ret.iloc[i] = day_ret

            # 静态组合日收益
            static_ret = sum(
                self.STATIC_WEIGHTS.get(s, 0) * sub_ret.loc[date, s] for s in strategies
            )
            static_port_ret.iloc[i] = static_ret

        # ── 计算 Sharpe ──
        def ann_sharpe(r: pd.Series) -> float:
            r = r.dropna()
            if r.std() == 0:
                return 0.0
            return r.mean() / r.std() * np.sqrt(252)

        regime_cum = (1 + regime_port_ret).cumprod()
        static_cum = (1 + static_port_ret).cumprod()

        return {
            "regime_sharpe": ann_sharpe(regime_port_ret),
            "static_sharpe": ann_sharpe(static_port_ret),
            "regime_returns": regime_port_ret,
            "static_returns": static_port_ret,
            "regime_cumulative": regime_cum,
            "static_cumulative": static_cum,
            "regime_weights_history": regime_weights_hist,
        }

    @staticmethod
    def _build_proxy_returns(market_data: dict) -> pd.DataFrame:
        """用 ETF 数据构建各子策略的日收益率代理。

        代理逻辑:
        - stocks: SPY 日收益
        - bonds: TLT 日收益
        - gold: GLD 日收益
        - vol_arb: -0.5 × |SPY日收益| + 微正漂移 (卖波动率近似)
        - mean_reversion: -SPY 1d 收益 × 衰减 (反转信号)
        - calendar_spread: SPY 低波时正收益 (theta 收入代理)
        - cash: 常数日收益 ≈ 年化 2%
        """
        close = market_data["close"]
        spy_ret = close["SPY"].pct_change()
        tlt_ret = close["TLT"].pct_change() if "TLT" in close.columns else spy_ret * 0
        gld_ret = close["GLD"].pct_change() if "GLD" in close.columns else spy_ret * 0

        # vol_arb 代理: 收取 theta 但在大波动日亏损
        spy_vol20 = spy_ret.rolling(20, min_periods=15).std()
        vol_arb = 0.0003 - 0.5 * spy_ret.abs()  # 每日微正漂移 - 尾部风险

        # mean_reversion 代理: 1 日反转
        mean_rev = -spy_ret.shift(1) * 0.3  # 假设反转 alpha 衰减

        # calendar_spread 代理: theta 收入, 在低波时更稳定
        cal_spread = pd.Series(0.0002, index=close.index) - spy_ret.abs() * 0.15

        cash = pd.Series(0.02 / 252, index=close.index)

        df = pd.DataFrame({
            "stocks": spy_ret,
            "bonds": tlt_ret,
            "gold": gld_ret,
            "vol_arb": vol_arb,
            "mean_reversion": mean_rev,
            "calendar_spread": cal_spread,
            "cash": cash,
        })
        return df.dropna()


# ════════════════════════════════════════════════════════════════════════
#  命令行入口: 快速演示
# ════════════════════════════════════════════════════════════════════════

def main():
    """下载数据 → 检测体制 → 输出权重 → 回测比较"""
    print("=" * 60)
    print("  市场体制检测系统")
    print("=" * 60)

    print("\n[1/4] 下载市场数据 ...")
    md = fetch_market_data(start="2008-01-01")
    print(f"  数据范围: {md['close'].index[0].date()} ~ {md['close'].index[-1].date()}")
    print(f"  标的数量: {md['close'].shape[1]}")

    detector = RegimeDetector()

    print("\n[2/4] 检测当前体制 ...")
    regimes = detector.detect_all(md)
    for dim, label in regimes.items():
        print(f"  {dim:20s} → {label}")

    print("\n[3/4] 推荐策略权重 ...")
    weights = detector.get_strategy_weights(regimes)
    for strat, w in sorted(weights.items(), key=lambda x: -x[1]):
        bar = "█" * int(w * 50)
        print(f"  {strat:20s} {w:5.1%}  {bar}")

    print("\n[4/4] 回测体制切换 vs 静态配置 ...")
    result = detector.backtest_regime_switching(md, rebalance_freq=5)
    print(f"  体制切换 Sharpe: {result['regime_sharpe']:.3f}")
    print(f"  静态配置 Sharpe: {result['static_sharpe']:.3f}")
    print(f"  体制切换终值:   {result['regime_cumulative'].iloc[-1]:.2f}")
    print(f"  静态配置终值:   {result['static_cumulative'].iloc[-1]:.2f}")

    # 体制分布统计
    print("\n── 体制历史分布 (最近 252 日) ──")
    hist = detector.get_regime_history(md, lookback=252)
    for col in hist.columns:
        dist = hist[col].value_counts(normalize=True)
        items = ", ".join(f"{k}: {v:.0%}" for k, v in dist.items())
        print(f"  {col:20s} | {items}")

    print("\n完成。")


if __name__ == "__main__":
    main()
