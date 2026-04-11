"""协方差估计 + 组合优化"""
import numpy as np
import pandas as pd


class CovarianceEstimator:
    def __init__(self, returns_df, method='ledoit_wolf', lookback=60):
        self.returns = returns_df
        self.method = method
        self.lookback = lookback

    def estimate(self, date, asset_list):
        idx = self.returns.index.get_loc(date)
        start = max(0, idx - self.lookback)
        hist = self.returns.iloc[start:idx][asset_list].dropna(axis=1, how='all')
        valid_assets = [a for a in asset_list if a in hist.columns]
        hist = hist[valid_assets].dropna()
        n_obs, n_assets = hist.shape
        if n_obs < max(12, n_assets) or n_assets < 2:
            vols = self.returns[valid_assets].iloc[start:idx].std().fillna(0.05)
            return np.diag(vols.values ** 2), valid_assets
        X = hist.values.astype(float)
        if self.method == 'ledoit_wolf':
            cov = self._ledoit_wolf(X)
        elif self.method == 'exponential':
            cov = self._exponential(X)
        elif self.method == 'factor':
            cov = self._single_factor(X)
        else:
            cov = np.cov(X, rowvar=False)
        cov += np.eye(n_assets) * 1e-8
        return cov, valid_assets

    def _ledoit_wolf(self, X):
        n, p = X.shape
        S = np.cov(X, rowvar=False)
        mu = np.trace(S) / p
        delta = S - mu * np.eye(p)
        X_c = X - X.mean(axis=0)
        gamma = sum(np.sum((X_c[i:i+1].T @ X_c[i:i+1] - S)**2) for i in range(n)) / n**2
        kappa = (np.sum(delta**2) + mu**2 * p) / n
        shrinkage = max(0, min(1, gamma / (kappa + 1e-10)))
        return (1 - shrinkage) * S + shrinkage * mu * np.eye(p)

    def _exponential(self, X):
        n, p = X.shape
        w = np.array([2**(-(n-1-i)/36) for i in range(n)])
        w /= w.sum()
        X_c = X - np.average(X, axis=0, weights=w)
        cov = (X_c * w[:, None]).T @ X_c
        mu = np.trace(cov) / p
        return 0.9 * cov + 0.1 * mu * np.eye(p)

    def _single_factor(self, X):
        n, p = X.shape
        mkt = X.mean(axis=1)
        mkt_var = np.var(mkt)
        betas = np.array([np.cov(X[:, i], mkt)[0, 1] / (mkt_var + 1e-10) for i in range(p)])
        resid = X - np.outer(mkt, betas)
        return np.outer(betas, betas) * mkt_var + np.diag(np.var(resid, axis=0))


class PortfolioOptimizer:
    def __init__(self, method='max_sharpe', max_pos_weight=0.10, min_pos_weight=0.005):
        self.method = method
        self.max_w = max_pos_weight
        self.min_w = min_pos_weight

    def optimize(self, expected_returns, cov_matrix, target_exposure, side='long'):
        from scipy.optimize import minimize
        n = len(expected_returns)
        if n < 2:
            return np.ones(n) * target_exposure / max(n, 1)
        mu = np.array(expected_returns, dtype=float)
        Sigma = np.array(cov_matrix, dtype=float)
        bounds = [(self.min_w, self.max_w)] * n
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - target_exposure}]
        vols = np.sqrt(np.diag(Sigma))
        iv = 1.0 / np.maximum(vols, 1e-8)
        w0 = np.clip(iv / iv.sum() * target_exposure, self.min_w, self.max_w)
        w0 = w0 / w0.sum() * target_exposure
        try:
            if self.method == 'max_sharpe':
                obj = lambda w: -(w @ mu) / np.sqrt(w @ Sigma @ w + 1e-10)
            elif self.method == 'min_variance':
                obj = lambda w: w @ Sigma @ w
            elif self.method == 'risk_parity':
                def obj(w):
                    pv = w @ Sigma @ w + 1e-10
                    rc = w * (Sigma @ w) / pv
                    return np.sum((rc - target_exposure / n) ** 2)
            elif self.method == 'max_diversification':
                obj = lambda w: -(w @ vols) / np.sqrt(w @ Sigma @ w + 1e-10)
            else:
                return w0
            res = minimize(obj, w0, method='SLSQP', bounds=bounds, constraints=constraints,
                          options={'maxiter': 500, 'ftol': 1e-10})
            return res.x if res.success else w0
        except Exception:
            return w0
