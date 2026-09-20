"""Option-derived factor signals, built on Alpaca option-chain data.

Constructs stock-selection and timing factors from implied volatility, skew
and term structure.
"""
import re
import math
from datetime import datetime, date
from typing import Optional

import numpy as np
import pandas as pd

# ── 尝试加载 QuantLib 用于更精确的 IV 计算 ──
try:
    import QuantLib as ql

    _HAS_QUANTLIB = True
except ImportError:
    _HAS_QUANTLIB = False


class OptionsSignalGenerator:
    """Factor generator for option-chain data.

    Alpaca returns an option chain in this shape:
        chain_dict[symbol] -> obj, where obj.latest_quote has .bid_price, .ask_price
        symbol format: {UNDERLYING}{YYMMDD}{C/P}{STRIKE*1000 zero-padded to 8}
        e.g. 'AAPL260424P00295000' => AAPL, 2026-04-24, Put, 295.0
    """

    # 正则匹配期权符号: 字母部分(underlying) + 6位日期 + C/P + 8位行权价
    _SYM_RE = re.compile(
        r'^([A-Z]+)'       # underlying ticker
        r'(\d{6})'         # YYMMDD
        r'([CP])'          # call / put
        r'(\d{8})$'        # strike * 1000, zero-padded
    )

    # ------------------------------------------------------------------
    # 1. 解析期权符号
    # ------------------------------------------------------------------
    @staticmethod
    def parse_option_symbol(sym: str) -> dict:
        """Parse an Alpaca option symbol into a structured dictionary.

        Parameters
        ----------
        sym : str
            Option symbol, e.g. 'AAPL260424P00295000'

        Returns
        -------
        dict
            Keys underlying, expiry, type, strike
        """
        m = OptionsSignalGenerator._SYM_RE.match(sym)
        if m is None:
            raise ValueError(f"无法解析期权符号: {sym}")

        underlying = m.group(1)
        date_str = m.group(2)          # YYMMDD
        opt_type = 'call' if m.group(3) == 'C' else 'put'
        strike = int(m.group(4)) / 1000.0

        yy, mm, dd = int(date_str[:2]), int(date_str[2:4]), int(date_str[4:6])
        year = 2000 + yy
        expiry = f"{year}-{mm:02d}-{dd:02d}"

        return {
            'underlying': underlying,
            'expiry': expiry,
            'type': opt_type,
            'strike': strike,
        }

    # ------------------------------------------------------------------
    # 2. 处理整条期权链
    # ------------------------------------------------------------------
    def process_chain(
        self,
        chain_dict: dict,
        underlying_price: float,
        ref_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Convert an Alpaca option-chain dictionary into a DataFrame.

        Parameters
        ----------
        chain_dict : dict
            The chain as returned by Alpaca: key=symbol, value=object with
            a .latest_quote attribute
        underlying_price : float
            Current price of the underlying
        ref_date : date, optional
            Reference date, defaulting to today; used to compute days to expiry

        Returns
        -------
        pd.DataFrame
            Columns strike, expiry, type, bid, ask, mid, moneyness, dte.
            Contracts quoted with bid=0 are dropped.
        """
        if ref_date is None:
            ref_date = date.today()

        # 如果已经是DataFrame (来自data_alpaca), 直接补充缺失列
        if isinstance(chain_dict, pd.DataFrame):
            df = chain_dict.copy()
            if 'moneyness' not in df.columns and underlying_price > 0:
                df['moneyness'] = df['strike'] / underlying_price
            if 'dte' not in df.columns and 'expiry' in df.columns:
                df['dte'] = df['expiry'].apply(
                    lambda x: (datetime.strptime(str(x)[:10], '%Y-%m-%d').date() - ref_date).days
                    if pd.notna(x) else 0
                )
            df = df[df['bid'] > 0] if 'bid' in df.columns else df
            df = df[df['dte'] > 0] if 'dte' in df.columns else df
            return df

        # 原始Alpaca API dict格式
        rows = []
        for sym, obj in chain_dict.items():
            try:
                info = self.parse_option_symbol(sym)
            except ValueError:
                continue

            q = obj.latest_quote
            bid = float(q.bid_price)
            ask = float(q.ask_price)

            if bid <= 0:
                continue

            mid = (bid + ask) / 2.0
            strike = info['strike']
            expiry_date = datetime.strptime(info['expiry'], '%Y-%m-%d').date()
            dte = (expiry_date - ref_date).days

            if dte <= 0:
                continue

            rows.append({
                'symbol': sym,
                'strike': strike,
                'expiry': info['expiry'],
                'type': info['type'],
                'bid': bid,
                'ask': ask,
                'mid': mid,
                'moneyness': strike / underlying_price,
                'dte': dte,
            })

        df = pd.DataFrame(rows)
        return df

    # ------------------------------------------------------------------
    # 3. 隐含波动率近似 (Brenner-Subrahmanyam 1988)
    # ------------------------------------------------------------------
    @staticmethod
    def implied_vol_proxy(
        mid_price: float,
        underlying_price: float,
        strike: float,
        days_to_expiry: int,
        option_type: str = 'call',
        risk_free_rate: float = 0.05,
    ) -> float:
        """Approximate implied volatility.

        Uses QuantLib where available (Newton-Raphson inversion of Black-Scholes);
        otherwise falls back to the Brenner-Subrahmanyam (1988) approximation:
            IV ~= mid / (0.4 * S * sqrt(T))

        Parameters
        ----------
        mid_price : float
            Option mid price
        underlying_price : float
            Price of the underlying
        strike : float
            Strike price
        days_to_expiry : int
            Days remaining to expiry
        option_type : str
            'call' or 'put'
        risk_free_rate : float
            Risk-free rate, default 5%

        Returns
        -------
        float
            Estimated implied volatility, annualized and expressed as a decimal
        """
        if days_to_expiry <= 0 or mid_price <= 0 or underlying_price <= 0:
            return np.nan

        T = days_to_expiry / 365.0

        # ── 尝试 QuantLib 精确反解 ──
        if _HAS_QUANTLIB:
            try:
                today = ql.Date.todaysDate()
                ql.Settings.instance().evaluationDate = today

                expiry_ql = today + ql.Period(days_to_expiry, ql.Days)
                payoff = ql.PlainVanillaPayoff(
                    ql.Option.Call if option_type == 'call' else ql.Option.Put,
                    strike,
                )
                exercise = ql.EuropeanExercise(expiry_ql)
                option = ql.VanillaOption(payoff, exercise)

                S = ql.SimpleQuote(underlying_price)
                r = ql.SimpleQuote(risk_free_rate)
                sigma = ql.SimpleQuote(0.20)  # 初始猜测

                spot_handle = ql.QuoteHandle(S)
                rate_handle = ql.YieldTermStructureHandle(
                    ql.FlatForward(today, ql.QuoteHandle(r), ql.Actual365Fixed())
                )
                div_handle = ql.YieldTermStructureHandle(
                    ql.FlatForward(today, 0.0, ql.Actual365Fixed())
                )
                vol_handle = ql.BlackVolTermStructureHandle(
                    ql.BlackConstantVol(today, ql.NullCalendar(),
                                        ql.QuoteHandle(sigma), ql.Actual365Fixed())
                )

                bsm = ql.BlackScholesMertonProcess(
                    spot_handle, div_handle, rate_handle, vol_handle
                )
                option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm))

                iv = option.impliedVolatility(mid_price, bsm)
                if 0.01 < iv < 5.0:
                    return iv
            except Exception:
                pass  # 回退到近似公式

        # ── Brenner-Subrahmanyam 近似 ──
        iv_approx = mid_price / (0.4 * underlying_price * math.sqrt(T))

        # 基本合理性检查
        if iv_approx <= 0 or iv_approx > 5.0:
            return np.nan
        return iv_approx

    # ------------------------------------------------------------------
    # 4. ATM 隐含波动率
    # ------------------------------------------------------------------
    def atm_iv(
        self,
        chain_df: pd.DataFrame,
        underlying_price: float,
        days_range: tuple = (20, 60),
    ) -> float:
        """Mean implied volatility near the money.

        Selects contracts with moneyness in [0.95, 1.05] and DTE inside
        days_range, computes IV separately for calls and puts, and averages.

        Parameters
        ----------
        chain_df : pd.DataFrame
            Output of process_chain
        underlying_price : float
            Current price of the underlying
        days_range : tuple
            (min DTE, max DTE), default (20, 60)

        Returns
        -------
        float
            Mean ATM implied volatility
        """
        if chain_df.empty:
            return np.nan

        mask = (
            (chain_df['moneyness'] >= 0.95)
            & (chain_df['moneyness'] <= 1.05)
            & (chain_df['dte'] >= days_range[0])
            & (chain_df['dte'] <= days_range[1])
        )
        atm = chain_df.loc[mask].copy()

        if atm.empty:
            return np.nan

        atm['iv'] = atm.apply(
            lambda r: self.implied_vol_proxy(
                r['mid'], underlying_price, r['strike'], r['dte'], r['type']
            ),
            axis=1,
        )
        return atm['iv'].mean()

    # ------------------------------------------------------------------
    # 5. IV 偏度 (25-delta skew)
    # ------------------------------------------------------------------
    def iv_skew(
        self,
        chain_df: pd.DataFrame,
        underlying_price: float,
        days_range: tuple = (20, 60),
    ) -> float:
        """Implied-volatility skew - 25-delta put versus 25-delta call.

        Delta is approximated by moneyness:
            25-delta put  ~= moneyness 0.88-0.92 (OTM put)
            25-delta call ~= moneyness 1.08-1.12 (OTM call)

        Positive skew means downside protection is expensive, which reads as
        bearish sentiment. The returned signal is -skew, so steep skew maps to
        a bearish score.

        Parameters
        ----------
        chain_df : pd.DataFrame
            Output of process_chain
        underlying_price : float
            Current price of the underlying
        days_range : tuple
            DTE filter range

        Returns
        -------
        float
            Skew signal; negative values are bearish
        """
        if chain_df.empty:
            return np.nan

        dte_mask = (chain_df['dte'] >= days_range[0]) & (chain_df['dte'] <= days_range[1])

        # OTM puts — 25-delta 近似
        put_mask = (
            dte_mask
            & (chain_df['type'] == 'put')
            & (chain_df['moneyness'] >= 0.88)
            & (chain_df['moneyness'] <= 0.92)
        )
        puts = chain_df.loc[put_mask]

        # OTM calls — 25-delta 近似
        call_mask = (
            dte_mask
            & (chain_df['type'] == 'call')
            & (chain_df['moneyness'] >= 1.08)
            & (chain_df['moneyness'] <= 1.12)
        )
        calls = chain_df.loc[call_mask]

        if puts.empty or calls.empty:
            return np.nan

        put_iv = puts.apply(
            lambda r: self.implied_vol_proxy(
                r['mid'], underlying_price, r['strike'], r['dte'], 'put'
            ),
            axis=1,
        ).mean()

        call_iv = calls.apply(
            lambda r: self.implied_vol_proxy(
                r['mid'], underlying_price, r['strike'], r['dte'], 'call'
            ),
            axis=1,
        ).mean()

        if np.isnan(put_iv) or np.isnan(call_iv) or call_iv == 0:
            return np.nan

        skew = (put_iv - call_iv) / call_iv
        return -skew  # 高偏度 => 信号为负(看空)

    # ------------------------------------------------------------------
    # 6. IV 期限结构
    # ------------------------------------------------------------------
    def iv_term_structure(
        self,
        chain_df: pd.DataFrame,
        underlying_price: float,
    ) -> float:
        """Implied-volatility term structure - near-dated versus far-dated.

        Near-dated is DTE < 30, far-dated is DTE > 60, both restricted to
        near-the-money contracts (moneyness 0.95-1.05). A ratio above 1
        indicates near-term stress and reads as bearish.

        Parameters
        ----------
        chain_df : pd.DataFrame
            Output of process_chain
        underlying_price : float
            Current price of the underlying

        Returns
        -------
        float
            Near-dated IV / far-dated IV; values above 1 are bearish
        """
        if chain_df.empty:
            return np.nan

        atm_mask = (chain_df['moneyness'] >= 0.95) & (chain_df['moneyness'] <= 1.05)

        short_mask = atm_mask & (chain_df['dte'] < 30) & (chain_df['dte'] > 5)
        long_mask = atm_mask & (chain_df['dte'] > 60)

        short_opts = chain_df.loc[short_mask]
        long_opts = chain_df.loc[long_mask]

        if short_opts.empty or long_opts.empty:
            return np.nan

        short_iv = short_opts.apply(
            lambda r: self.implied_vol_proxy(
                r['mid'], underlying_price, r['strike'], r['dte'], r['type']
            ),
            axis=1,
        ).mean()

        long_iv = long_opts.apply(
            lambda r: self.implied_vol_proxy(
                r['mid'], underlying_price, r['strike'], r['dte'], r['type']
            ),
            axis=1,
        ).mean()

        if np.isnan(short_iv) or np.isnan(long_iv) or long_iv == 0:
            return np.nan

        return short_iv / long_iv

    # ------------------------------------------------------------------
    # 7. Put/Call 比率
    # ------------------------------------------------------------------
    @staticmethod
    def put_call_ratio(chain_df: pd.DataFrame) -> float:
        """Put/call ratio, using a volume proxy weighted by the reciprocal of mid price.

        1/mid stands in for open interest, on the reasoning that cheaper
        contracts attract more speculative trading. Puts and calls are summed
        separately and the ratio taken. A high put/call ratio reads as bearish.

        Parameters
        ----------
        chain_df : pd.DataFrame
            Output of process_chain

        Returns
        -------
        float
            Put/call ratio
        """
        if chain_df.empty:
            return np.nan

        puts = chain_df.loc[chain_df['type'] == 'put']
        calls = chain_df.loc[chain_df['type'] == 'call']

        if puts.empty or calls.empty:
            return np.nan

        # 用 bid-ask spread 的倒数作为流动性/活跃度代理
        put_activity = (1.0 / puts['mid'].replace(0, np.nan)).sum()
        call_activity = (1.0 / calls['mid'].replace(0, np.nan)).sum()

        if call_activity == 0 or np.isnan(call_activity):
            return np.nan

        return put_activity / call_activity

    # ------------------------------------------------------------------
    # 8. IV 排名 (百分位)
    # ------------------------------------------------------------------
    @staticmethod
    def iv_rank(current_iv: float, iv_history: pd.Series) -> float:
        """Percentile rank of the current IV within its own history.

        Parameters
        ----------
        current_iv : float
            Current ATM implied volatility
        iv_history : pd.Series
            Historical IV series (252 trading days / 52 weeks recommended)

        Returns
        -------
        float
            Percentile in [0, 1]; 0 is the historical low, 1 the historical high
        """
        if np.isnan(current_iv) or iv_history.dropna().empty:
            return np.nan

        clean = iv_history.dropna().values
        rank = np.sum(clean < current_iv) / len(clean)
        return float(rank)

    # ------------------------------------------------------------------
    # 9. 组合期权信号
    # ------------------------------------------------------------------
    def build_options_signal(
        self,
        chain_data: dict,
        prices: dict,
    ) -> pd.Series:
        """Cross-sectional option signal across multiple instruments.

        Computes three sub-factors per instrument, blends them by weight, then
        cross-sectionally ranks the result onto [-1, 1].

        Weights: 40% iv_skew + 30% put_call_ratio + 30% iv_term_structure

        Parameters
        ----------
        chain_data : dict
            {ticker: alpaca_chain_dict, ...}
        prices : dict
            {ticker: underlying_price, ...}

        Returns
        -------
        pd.Series
            index=ticker, values in [-1, 1]; positive is bullish, negative bearish
        """
        records = {}

        for ticker, chain_dict in chain_data.items():
            up = prices.get(ticker)
            if up is None or up <= 0:
                continue

            df = self.process_chain(chain_dict, up)
            if df.empty:
                continue

            skew = self.iv_skew(df, up)
            pcr = self.put_call_ratio(df)
            term = self.iv_term_structure(df, up)

            records[ticker] = {
                'iv_skew': skew,
                'put_call_ratio': pcr,
                'iv_term_structure': term,
            }

        if not records:
            return pd.Series(dtype=float)

        raw = pd.DataFrame(records).T  # index = tickers

        # 截面标准化各子因子
        def _zscore(s: pd.Series) -> pd.Series:
            """截面 z-score, 处理 NaN"""
            mu = s.mean()
            sd = s.std()
            if sd == 0 or np.isnan(sd):
                return s * 0.0
            return (s - mu) / sd

        z_skew = _zscore(raw['iv_skew'])        # 已取负, 负=看空
        z_pcr = -_zscore(raw['put_call_ratio'])  # 高PCR=看空 => 取负
        z_term = -_zscore(raw['iv_term_structure'])  # 高比值=短期恐慌 => 取负

        composite = 0.40 * z_skew + 0.30 * z_pcr + 0.30 * z_term

        # 截面排名映射到 [-1, 1]
        n = composite.dropna().shape[0]
        if n <= 1:
            return composite.apply(lambda x: 0.0 if not np.isnan(x) else np.nan)

        ranked = composite.rank(pct=True)       # 0~1
        signal = ranked * 2.0 - 1.0             # [-1, 1]

        return signal
