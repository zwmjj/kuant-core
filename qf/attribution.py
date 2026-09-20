"""Factor attribution analysis."""
import numpy as np
import pandas as pd


class AttributionAnalyzer:
    def __init__(self, ff5_factors):
        self.ff5 = ff5_factors

    def ff5_regression(self, strategy_returns, name="策略"):
        rets = strategy_returns.dropna()
        ff = self.ff5.copy()
        rets.index = rets.index.to_period('M')
        ff.index = ff.index.to_period('M')
        common = rets.index.intersection(ff.index)
        rets = rets.loc[common]; ff = ff.loc[common]
        excess = rets - ff['rf']
        X = ff[['mktrf', 'smb', 'hml', 'umd']]
        y = excess
        X_c = np.column_stack([np.ones(len(X)), X.values.astype(float)])
        coeffs = np.linalg.lstsq(X_c, y.values.astype(float), rcond=None)[0]
        alpha = coeffs[0] * 12
        betas = coeffs[1:]
        y_vals = y.values.astype(float)
        y_pred = X_c @ coeffs
        r2 = 1 - np.sum((y_vals - y_pred)**2) / np.sum((y_vals - y_vals.mean())**2)
        names = ['MKT', 'SMB', 'HML', 'UMD']
        return {'alpha': alpha, 'betas': dict(zip(names, betas)), 'r2': r2, 'n_months': len(common)}

    def yearly_attribution(self, strategy_returns, benchmark_returns):
        strat = strategy_returns.copy(); bm = benchmark_returns.copy()
        strat.index = strat.index.to_period('M')
        bm.index = bm.index.to_period('M')
        df = pd.DataFrame({'strategy': strat, 'benchmark': bm}).dropna()
        df['excess'] = df['strategy'] - df['benchmark']
        df['year'] = df.index.year
        return df.groupby('year').agg({
            'strategy': lambda x: (1+x).prod()-1,
            'benchmark': lambda x: (1+x).prod()-1,
            'excess': lambda x: (1+x).prod()-1,
        })
