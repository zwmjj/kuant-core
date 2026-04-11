"""压力测试 + 情景分析"""
import numpy as np
import pandas as pd
from qf.backtest import run_event_driven


class StressTest:
    """可插拔压力测试套件"""
    def __init__(self, d, signal, base_kw=None):
        self.d = d
        self.signal = signal
        self.base_kw = base_kw or dict(cost_model='fixed', commission_bps=1.0,
                                        spread_bps=3.0, turnover_penalty=0.25)

    def _run(self, **extra_kw):
        kw = {**self.base_kw, **extra_kw}
        r, m = run_event_driven(self.d, self.signal, verbose=False, **kw)
        return r, m

    def crisis_replay(self, crises=None):
        """历史危机回放"""
        if crises is None:
            crises = [
                ('Dot-Com Crash',       '2000-03', '2002-10'),
                ('GFC 2008-09',         '2007-10', '2009-03'),
                ('Flash Crash 2010',    '2010-04', '2010-07'),
                ('2011 Euro Crisis',    '2011-06', '2011-10'),
                ('Taper Tantrum 2013',  '2013-05', '2013-09'),
                ('2015 Fed Hike',       '2015-07', '2015-10'),
                ('2018 Q4 Selloff',     '2018-10', '2018-12'),
                ('2020 COVID',          '2020-02', '2020-04'),
                ('2022 Fed Tightening', '2022-01', '2022-10'),
                ('2023 Bank Crisis',    '2023-02', '2023-04'),
            ]
        r, _ = self._run()
        rets = r.returns.copy()
        rets.index = rets.index.to_period('M')
        spy = self.d['spy_ret'].copy()
        spy.index = spy.index.to_period('M')
        results = []
        for name, s, e in crises:
            sp, ep = pd.Period(s, 'M'), pd.Period(e, 'M')
            mask = (rets.index >= sp) & (rets.index <= ep)
            smask = (spy.index >= sp) & (spy.index <= ep)
            if mask.sum() == 0: continue
            sr = (1 + rets[mask]).prod() - 1
            br = (1 + spy[smask]).prod() - 1 if smask.sum() > 0 else 0
            cpv = (1 + rets[mask]).cumprod()
            cmdd = ((cpv - cpv.cummax()) / cpv.cummax()).min()
            results.append({'crisis': name, 'strategy': sr, 'benchmark': br,
                           'excess': sr - br, 'crisis_mdd': cmdd})
        return pd.DataFrame(results)

    def synthetic_shocks(self):
        """合成压力场景"""
        r, _ = self._run()
        rets = r.returns
        results = []
        for label, shock in [('Baseline', 0), ('-10%', -0.10), ('-20%', -0.20),
                              ('-30%', -0.30), ('-50%', -0.50)]:
            sr = rets.copy()
            if shock != 0: sr.iloc[len(sr)//2] = shock
            pv = 10000 * (1 + sr).cumprod()
            mdd = ((pv - pv.cummax()) / pv.cummax()).min()
            sharpe = sr.mean() / sr.std() * np.sqrt(12) if sr.std() > 0 else 0
            results.append({'scenario': label, 'sharpe': sharpe, 'mdd': mdd})
        # Vol double
        sr = rets.copy()
        mean_r = sr.mean()
        sr = mean_r + (sr - mean_r) * 2
        pv = 10000 * (1 + sr).cumprod()
        mdd = ((pv - pv.cummax()) / pv.cummax()).min()
        sharpe = sr.mean() / sr.std() * np.sqrt(12) if sr.std() > 0 else 0
        results.append({'scenario': 'Vol x2', 'sharpe': sharpe, 'mdd': mdd})
        return pd.DataFrame(results)

    def parameter_sensitivity(self, param_grid=None):
        """参数扰动测试"""
        if param_grid is None:
            param_grid = [
                ('Baseline L20/S20', {}),
                ('L15/S20', dict(long_n=15)),
                ('L30/S20', dict(long_n=30)),
                ('S10', dict(short_n=10)),
                ('S30', dict(short_n=30)),
                ('Pure Long', dict(long_n=20, short_n=1, long_pct=1.0, short_pct=0.0)),
                ('130/30', dict(long_pct=1.3, short_pct=0.3)),
            ]
        results = []
        for name, kw in param_grid:
            _, m = self._run(**kw)
            results.append({'config': name, 'cagr': m['cagr'], 'sharpe': m['sharpe'],
                           'max_drawdown': m['max_drawdown'], 'alpha': m.get('alpha', 0)})
        return pd.DataFrame(results)

    def cost_sensitivity(self, cost_levels=None):
        """交易成本压力测试"""
        if cost_levels is None:
            cost_levels = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
        results = []
        for bps in cost_levels:
            _, m = self._run(commission_bps=bps, spread_bps=bps*1.5)
            results.append({'cost_bps': bps, 'cagr': m['cagr'], 'sharpe': m['sharpe'],
                           'final_value': m['final_value']})
        return pd.DataFrame(results)

    def triple_cost_scenario(self):
        """3x成本场景 (SOP Phase 3 要求)"""
        base = self.base_kw.copy()
        triple_kw = {
            'commission_bps': base.get('commission_bps', 1.0) * 3,
            'spread_bps': base.get('spread_bps', 5.0) * 3,
            'impact_coeff': base.get('impact_coeff', 0.3) * 3,
            'short_borrow_bps': base.get('short_borrow_bps', 30.0) * 3,
        }
        _, m = self._run(**triple_kw)
        return m
