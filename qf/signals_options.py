"""期权衍生因子信号 — 基于 Alpaca 期权链数据

利用期权隐含波动率、偏度、期限结构等信息构建选股/择时因子。
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
    """期权链衍生因子生成器

    Alpaca 期权链返回格式:
        chain_dict[symbol] -> obj, 其中 obj.latest_quote 具有 .bid_price, .ask_price
        symbol 格式: {UNDERLYING}{YYMMDD}{C/P}{STRIKE*1000 padded 8位}
        例: 'AAPL260424P00295000' => AAPL, 2026-04-24, Put, 295.0
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
        """解析 Alpaca 期权符号，返回结构化字典

        Parameters
        ----------
        sym : str
            期权符号, 例如 'AAPL260424P00295000'

        Returns
        -------
        dict
            包含 underlying, expiry, type, strike 四个字段
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
        """将 Alpaca 期权链字典转为 DataFrame

        Parameters
        ----------
        chain_dict : dict
            Alpaca 返回的期权链, key=symbol, value=对象(含 .latest_quote)
        underlying_price : float
            标的当前价格
        ref_date : date, optional
            参考日期, 默认今天; 用于计算距到期日天数

        Returns
        -------
        pd.DataFrame
            包含 strike, expiry, type, bid, ask, mid, moneyness, dte 列;
            已过滤 bid=0 的合约
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
        """近似隐含波动率

        优先使用 QuantLib (Newton-Raphson BSM反解); 不可用时退回
        Brenner-Subrahmanyam (1988) 近似:
            IV ≈ mid / (0.4 * S * sqrt(T))

        Parameters
        ----------
        mid_price : float
            期权中间价
        underlying_price : float
            标的价格
        strike : float
            行权价
        days_to_expiry : int
            距到期天数
        option_type : str
            'call' 或 'put'
        risk_free_rate : float
            无风险利率, 默认 5%

        Returns
        -------
        float
            隐含波动率估计值 (年化, 小数形式)
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
        """平值附近的隐含波动率均值

        筛选 moneyness 在 [0.95, 1.05] 且 DTE 在 days_range 内的期权,
        对看涨和看跌分别计算 IV 后取均值。

        Parameters
        ----------
        chain_df : pd.DataFrame
            process_chain 输出
        underlying_price : float
            标的当前价格
        days_range : tuple
            (最小DTE, 最大DTE), 默认 (20, 60)

        Returns
        -------
        float
            ATM IV 均值
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
        """隐含波动率偏度 — 25-delta put vs 25-delta call

        用 moneyness 近似:
            25-delta put  ≈ moneyness 0.88~0.92 (OTM put)
            25-delta call ≈ moneyness 1.08~1.12 (OTM call)

        正偏度 = 下行保护昂贵 = 看空情绪
        返回信号: -skew (高偏度 => 看空标的)

        Parameters
        ----------
        chain_df : pd.DataFrame
            process_chain 输出
        underlying_price : float
            标的当前价格
        days_range : tuple
            DTE 筛选范围

        Returns
        -------
        float
            偏度信号 (负数 = 看空)
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
        """隐含波动率期限结构 — 短期 vs 长期

        短期: DTE < 30, 长期: DTE > 60, 筛选近 ATM (0.95~1.05)
        比值 > 1 表示短期恐慌, 看空信号

        Parameters
        ----------
        chain_df : pd.DataFrame
            process_chain 输出
        underlying_price : float
            标的当前价格

        Returns
        -------
        float
            短期IV / 长期IV 比值; > 1 看空
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
        """看跌/看涨比率 — 以成交量代理 (mid 倒数加权)

        用 1/mid 作为持仓量代理 (价格越低说明交易越投机),
        分别对 put/call 求和并取比值。
        高 PC 比 = 看空情绪。

        Parameters
        ----------
        chain_df : pd.DataFrame
            process_chain 输出

        Returns
        -------
        float
            Put/Call 比率
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
        """当前 IV 在历史区间中的百分位排名

        Parameters
        ----------
        current_iv : float
            当前 ATM IV
        iv_history : pd.Series
            历史 IV 序列 (建议 252 个交易日 / 52 周)

        Returns
        -------
        float
            0~1 之间的百分位; 0=历史最低, 1=历史最高
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
        """多标的组合期权信号

        对每个标的计算三个子因子, 加权合成后做截面排名映射到 [-1, 1]。

        权重: 40% iv_skew + 30% put_call_ratio + 30% iv_term_structure

        Parameters
        ----------
        chain_data : dict
            {ticker: alpaca_chain_dict, ...}
        prices : dict
            {ticker: underlying_price, ...}

        Returns
        -------
        pd.Series
            index=ticker, values in [-1, 1], 正数看多 / 负数看空
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
