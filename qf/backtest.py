"""回测引擎 (事件驱动)"""
import numpy as np
import pandas as pd
from qf.costs import ExecutionHandler, SignalEvent, OrderEvent
from qf.portfolio import CovarianceEstimator, PortfolioOptimizer


class BacktestResult:
    def __init__(self, portfolio_values, returns, trades):
        self.pv = portfolio_values
        self.returns = returns
        self.trades = trades

    def metrics(self, rf=None, benchmark_returns=None):
        rets = self.returns.dropna()
        if len(rets) < 2:
            return {'cagr': 0, 'sharpe': 0, 'sortino': 0, 'max_drawdown': 0,
                    'win_rate': 0, 'final_value': self.pv.iloc[-1] if len(self.pv) > 0 else 0}
        rf_rate = rf.reindex(rets.index).fillna(0) if rf is not None else pd.Series(0, index=rets.index)
        excess = rets - rf_rate
        years = max((rets.index[-1] - rets.index[0]).days / 365.25, 0.01)
        total_ret = self.pv.iloc[-1] / self.pv.iloc[0] - 1
        cagr = (1 + total_ret) ** (1 / years) - 1
        sharpe = excess.mean() / excess.std() * np.sqrt(12) if excess.std() > 0 else 0
        ds = rets[rets < 0].std() * np.sqrt(12)
        sortino = excess.mean() * 12 / ds if ds > 0 else 0
        dd = (self.pv - self.pv.cummax()) / self.pv.cummax()
        m = {'total_return': total_ret, 'cagr': cagr, 'sharpe': sharpe, 'sortino': sortino,
             'max_drawdown': dd.min(), 'win_rate': (rets > 0).mean(),
             'final_value': self.pv.iloc[-1], 'n_months': len(rets)}
        if benchmark_returns is not None:
            bm = benchmark_returns.squeeze().copy()
            bm.index = bm.index.to_period('M')
            rp = rets.copy(); rp.index = rp.index.to_period('M')
            common = rp.index.intersection(bm.index)
            bm = bm.loc[common]; aligned = rp.loc[common]
            cov_m = np.cov(aligned.values.flatten(), bm.values.flatten())
            beta = cov_m[0, 1] / cov_m[1, 1] if cov_m[1, 1] > 0 else 1
            bm_total = (1 + bm).prod() - 1
            bm_cagr = (1 + bm_total) ** (1 / years) - 1
            m['alpha'] = cagr - beta * ((1 + bm.mean()) ** 12 - 1)
            m['beta'] = beta
            m['benchmark_cagr'] = bm_cagr
            m['excess_return'] = cagr - bm_cagr
        return m


class DataHandler:
    def __init__(self, prices, returns, volume=None, adv_dollar=None):
        self.prices = prices
        self.returns = returns
        self.volume = volume
        self.adv_dollar = adv_dollar
        self.dates = prices.index.tolist()

    def get_bar(self, date):
        bar = {'date': date}
        bar['prices'] = self.prices.loc[date].dropna() if date in self.prices.index else pd.Series()
        bar['returns'] = self.returns.loc[date].dropna() if date in self.returns.index else pd.Series()
        tradable = set(bar['prices'][bar['prices'] > 0].index) & set(bar['returns'].dropna().index)
        bar['tradable'] = tradable
        bar['suspended'] = set(self.prices.columns) - tradable
        bar['volume'] = self.volume.loc[date].dropna() if (self.volume is not None and date in self.volume.index) else pd.Series()
        bar['adv_dollar'] = self.adv_dollar.loc[date].dropna() if (self.adv_dollar is not None and date in self.adv_dollar.index) else pd.Series()
        return bar

    def iter_bars(self):
        for date in self.dates:
            yield self.get_bar(date)


class StrategyEngine:
    """信号->权重 引擎 (事件驱动回测用)"""
    def __init__(self, signal_df, long_n=20, short_n=20, long_pct=1.15, short_pct=0.15,
                 weight_mode='inv_vol', inv_vol_df=None, turnover_penalty=0.0,
                 optimizer=None, cov_estimator=None):
        self.signal = signal_df
        self.long_n = long_n; self.short_n = short_n
        self.long_pct = long_pct; self.short_pct = short_pct
        self.weight_mode = weight_mode; self.inv_vol = inv_vol_df
        self.turnover_penalty = turnover_penalty
        self.optimizer = optimizer; self.cov_estimator = cov_estimator
        self._prev_long = set(); self._prev_short = set()

    def on_market(self, bar):
        date = bar['date']
        if date not in self.signal.index: return None
        sig_row = self.signal.loc[date].dropna()
        valid = sig_row[sig_row.index.isin(bar['tradable'])]
        if len(valid) < self.long_n + self.short_n: return None

        adj = valid.copy()
        if self.turnover_penalty > 0:
            for p in self._prev_long:
                if p in adj.index: adj[p] += self.turnover_penalty
            for p in self._prev_short:
                if p in adj.index: adj[p] -= self.turnover_penalty

        long_list = adj.nlargest(self.long_n).index.tolist()
        short_list = adj.nsmallest(self.short_n).index.tolist()

        # 加权
        if self.weight_mode == 'inv_vol' and self.inv_vol is not None and date in self.inv_vol.index:
            iv = self.inv_vol.loc[date, long_list].dropna()
            long_w = (iv / iv.sum() * self.long_pct).to_dict() if len(iv) > 0 else {t: self.long_pct/self.long_n for t in long_list}
        else:
            long_w = {t: self.long_pct / self.long_n for t in long_list}
        short_w = {t: self.short_pct / self.short_n for t in short_list}

        self._prev_long = set(long_w.keys())
        self._prev_short = set(short_w.keys())
        return SignalEvent(date=date, long_targets=long_w, short_targets=short_w)


class Portfolio:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.long_positions = {}
        self.short_positions = {}
        self.pv_history = []
        self.return_history = []
        self.trade_count = 0

    def on_signal(self, sig, bar):
        orders = []
        for p in list(self.long_positions):
            if p not in sig.long_targets: orders.append((p, 0, 'CLOSE_LONG'))
        for p in list(self.short_positions):
            if p not in sig.short_targets: orders.append((p, 0, 'CLOSE_SHORT'))
        for p, w in sig.long_targets.items(): orders.append((p, w, 'LONG'))
        for p, w in sig.short_targets.items(): orders.append((p, w, 'SHORT'))
        return OrderEvent(date=sig.date, orders=orders)

    def update_pv(self, month_ret):
        self.capital *= (1 + month_ret)
        self.pv_history.append(self.capital)
        self.return_history.append(month_ret)


class EventDrivenBacktester:
    def __init__(self, data_handler, strategy_engine, portfolio, execution):
        self.data = data_handler
        self.strategy = strategy_engine
        self.portfolio = portfolio
        self.execution = execution

    def run(self, verbose=True):
        dates_with_ret = []
        if verbose:
            print(f"\n  运行事件驱动回测...")
        for bar in self.data.iter_bars():
            # 1. 现有持仓赚取本期回报
            month_ret = self.execution.compute_holding_return(bar, self.portfolio)

            # 2. 生成新信号，执行换仓（新持仓下期才赚回报）
            sig = self.strategy.on_market(bar)
            if sig is not None:
                order = self.portfolio.on_signal(sig, bar)
                _, cost = self.execution.on_order(order, bar, self.portfolio)
                month_ret -= cost

            # 3. 只有有持仓时才记录回报
            if self.portfolio.long_positions or self.portfolio.short_positions or month_ret != 0:
                self.portfolio.update_pv(month_ret)
                dates_with_ret.append(bar['date'])
        if verbose:
            print(f"  完成: {len(dates_with_ret)} 个月, {self.portfolio.trade_count} 笔交易")
        pv = pd.Series([self.portfolio.initial_capital] + self.portfolio.pv_history,
                       index=[self.data.dates[0]] + dates_with_ret)
        rets = pd.Series(self.portfolio.return_history, index=dates_with_ret)
        return BacktestResult(pv, rets, pd.DataFrame())


# ── 旧接口兼容
Backtester = EventDrivenBacktester


def run_event_driven(d, signal, long_n=20, short_n=20, long_pct=1.15, short_pct=0.15,
                     weight_mode='inv_vol', turnover_penalty=0.25,
                     cost_model='sqrt', verbose=True, **cost_kw):
    """便捷函数"""
    inv_vol = 1.0 / d['returns'].rolling(12).std().replace(0, np.nan)
    dh = DataHandler(d['prices'], d['returns'], d.get('volume'), d.get('adv_dollar'))
    se = StrategyEngine(signal, long_n=long_n, short_n=short_n,
                        long_pct=long_pct, short_pct=short_pct,
                        weight_mode=weight_mode, inv_vol_df=inv_vol,
                        turnover_penalty=turnover_penalty)
    port = Portfolio(initial_capital=10000)
    exe = ExecutionHandler(cost_model=cost_model, **cost_kw)
    eng = EventDrivenBacktester(dh, se, port, exe)
    result = eng.run(verbose=verbose)
    m = result.metrics(rf=d['rf'], benchmark_returns=d['spy_ret'])
    return result, m
