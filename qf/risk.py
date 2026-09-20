"""Risk analytics + gate checks"""
import numpy as np
import pandas as pd
from scipy import stats


class RiskAnalyzer:
    """Risk metric computation"""
    def __init__(self, returns, pv=None):
        self.returns = returns
        self.pv = pv

    def var(self, confidence=0.95):
        return self.returns.quantile(1 - confidence)

    def cvar(self, confidence=0.95):
        v = self.var(confidence)
        tail = self.returns[self.returns <= v]
        return tail.mean() if len(tail) > 0 else v

    def max_drawdown(self):
        if self.pv is None:
            self.pv = (1 + self.returns).cumprod()
        dd = (self.pv - self.pv.cummax()) / self.pv.cummax()
        return dd.min()

    def drawdown_series(self):
        if self.pv is None:
            self.pv = (1 + self.returns).cumprod()
        return (self.pv - self.pv.cummax()) / self.pv.cummax()

    def calmar(self, cagr):
        mdd = self.max_drawdown()
        return cagr / abs(mdd) if mdd != 0 else 0

    def tail_stats(self):
        r = self.returns
        return {
            'skew': r.skew(), 'kurtosis': r.kurtosis(),
            'var_95': self.var(0.95), 'cvar_95': self.cvar(0.95),
            'var_99': self.var(0.99), 'cvar_99': self.cvar(0.99),
            'worst_month': r.min(), 'best_month': r.max(),
            'loss_pct': (r < 0).mean(),
            'profit_loss_ratio': r[r > 0].mean() / abs(r[r < 0].mean()) if (r < 0).sum() > 0 else np.inf,
            'max_consecutive_loss': self._max_consec_loss(),
        }

    def _max_consec_loss(self):
        neg = (self.returns < 0).astype(int)
        mx = cur = 0
        for v in neg:
            cur = cur + 1 if v else 0
            mx = max(mx, cur)
        return mx

    def dsr(self, n_trials=20):
        """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)"""
        T = len(self.returns)
        if T < 12:
            return {'z_score': 0, 'p_value': 0.5, 'dsr': 0.5}
        sr_m = self.returns.mean() / (self.returns.std() + 1e-10)
        skew = self.returns.skew()
        kurt = self.returns.kurtosis()
        # Expected max monthly SR under null (N independent trials)
        e_max = np.sqrt(2 * np.log(max(n_trials, 2))) - \
                (np.log(np.pi) + 0.5772) / (2 * np.sqrt(2 * np.log(max(n_trials, 2))))
        sr0 = e_max / np.sqrt(T)  # Scale to monthly SR units
        # Std of SR estimator (Lo, 2002)
        # Lo (2002): pandas .kurtosis() returns excess kurtosis, formula needs raw kurtosis
        sr_std = np.sqrt((1 + 0.5 * sr_m**2 - skew * sr_m +
                          ((kurt + 3) / 4) * sr_m**2) / max(T - 1, 1))
        z = (sr_m - sr0) / (sr_std + 1e-10)
        return {'z_score': z, 'p_value': 1 - stats.norm.cdf(z), 'dsr': stats.norm.cdf(z)}


class GateCheck:
    """Gate check manager"""
    def __init__(self):
        self.results = []

    def check(self, name, value, threshold, comparison='>', category=''):
        """Add a gate check"""
        if comparison == '>':
            passed = value > threshold
        elif comparison == '<':
            passed = value < threshold
        elif comparison == '>=':
            passed = value >= threshold
        elif comparison == '==':
            passed = value == threshold
        else:
            passed = False

        self.results.append({
            'name': name,
            'value': value,
            'threshold': threshold,
            'comparison': comparison,
            'passed': bool(passed),
            'category': category,
        })
        return passed

    def summary(self):
        """Return the summary"""
        n_pass = sum(1 for r in self.results if r['passed'])
        n_total = len(self.results)
        categories = {}
        for r in self.results:
            cat = r.get('category', 'Other')
            if cat not in categories:
                categories[cat] = {'pass': 0, 'total': 0}
            categories[cat]['total'] += 1
            if r['passed']:
                categories[cat]['pass'] += 1
        return {
            'n_pass': n_pass, 'n_total': n_total,
            'all_pass': n_pass == n_total,
            'categories': categories,
            'details': self.results,
        }

    def run_standard_gates(self, metrics, risk_analyzer, is_sharpe=None, oos_sharpe=None, n_trials=20):
        """Run the standard gate checks"""
        m = metrics
        self.check('Sharpe > 1.5', m['sharpe'], 1.5, '>', 'Backtest')
        self.check('MDD > -20%', m['max_drawdown'], -0.20, '>', 'Backtest')
        self.check('Calmar > 1.0', risk_analyzer.calmar(m['cagr']), 1.0, '>', 'Backtest')
        self.check('Sortino > 2.0', m['sortino'], 2.0, '>', 'Backtest')
        self.check('Win Rate > 55%', m['win_rate'], 0.55, '>', 'Backtest')

        if is_sharpe is not None and oos_sharpe is not None:
            self.check('OOS Sharpe > 1.0', oos_sharpe, 1.0, '>', 'OOS')
            decay = 1 - oos_sharpe / is_sharpe if is_sharpe > 0 else 1
            self.check('Sharpe Decay < 50%', decay, 0.50, '<', 'OOS')

        dsr = risk_analyzer.dsr(n_trials)
        self.check('DSR > 95%', dsr['dsr'], 0.95, '>', 'Statistical')

        return self.summary()
