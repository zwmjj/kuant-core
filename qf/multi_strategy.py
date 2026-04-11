"""
Multi-Strategy Multi-Asset Portfolio Engine
============================================
Integrates all validated strategies into a single portfolio.

Each strategy is an independent "sleeve" with its own signal, asset, and frequency.
The engine combines them with equal or risk-parity weights.

Validated strategies (4 rounds of scan + strict validation):
  1. safe_haven     — Equity/Gold/Bond rotation on vol regime
  2. dual_mom_gld   — Gold dual momentum (absolute+relative, 120d)
  3. dual_mom_spy   — SPY dual momentum (60d)
  4. reversal_tlt   — TLT extreme reversal (5d z-score > 1.5)
  5. overnight      — SPY overnight premium (close→open)
  6. corr_regime    — Stock-bond correlation regime allocation
  7. vol_regime_gld — Gold trend/MR switch on vol regime
  8. crypto_xmom    — Crypto risk-adjusted cross-sectional momentum
  9. curve_rot      — Yield curve rotation (TLT/SHY → equity/bond)
  10. xasset_mom    — Multi-asset dual momentum ensemble

All use shift(1) to avoid lookahead. All validated with WFCV OOS > 0.
"""
import numpy as np
import pandas as pd


def sigmoid_trend(price, period, steepness=0.5):
    sma = price.rolling(period).mean()
    dev = (price / sma - 1) * steepness * 100
    return 2 / (1 + np.exp(-dev)) - 1


class MultiStrategyPortfolio:
    """Runs all validated strategies and combines into a single portfolio."""

    def __init__(self, prices: dict, weighting='equal'):
        """
        Parameters
        ----------
        prices : dict
            {ticker: pd.Series of daily close prices}
            Required: SPY, GLD, TLT, SHY, QQQ, IWM
            Optional: BTC-USD, ETH-USD, SOL-USD, EEM, VNQ, HYG
        weighting : str
            'equal' or 'risk_parity'
        """
        self.prices = prices
        self.weighting = weighting
        self.strategies = {}

    def run_all(self):
        """Run all strategies, return dict of {name: daily_returns}"""
        p = self.prices

        # 1. Safe haven rotation
        if all(k in p for k in ['SPY', 'GLD', 'TLT']):
            spy_ret = p['SPY'].pct_change()
            stress = spy_ret.rolling(20).std() * np.sqrt(252)
            is_stress = (stress > stress.rolling(252).quantile(0.8)).astype(float)
            safe = p['GLD'].pct_change() * 0.5 + p['TLT'].pct_change() * 0.5
            r = ((1 - is_stress).shift(1) * spy_ret + is_stress.shift(1) * safe).dropna()
            self.strategies['safe_haven'] = r

        # 2. Gold dual momentum (120d)
        if all(k in p for k in ['GLD', 'SHY']):
            gld = p['GLD']; shy = p['SHY']
            abs_mom = gld / gld.shift(120) - 1
            shy_mom = shy / shy.shift(120) - 1
            sig = ((abs_mom > 0) & (abs_mom > shy_mom.reindex(abs_mom.index))).astype(float)
            self.strategies['dual_mom_gld'] = (sig.shift(1) * gld.pct_change()).dropna()

        # 3. SPY dual momentum (60d)
        if all(k in p for k in ['SPY', 'SHY']):
            spy = p['SPY']; shy = p['SHY']
            abs_mom = spy / spy.shift(60) - 1
            shy_mom = shy / shy.shift(60) - 1
            sig = ((abs_mom > 0) & (abs_mom > shy_mom.reindex(abs_mom.index))).astype(float)
            self.strategies['dual_mom_spy'] = (sig.shift(1) * spy.pct_change()).dropna()

        # 4. TLT extreme reversal
        if 'TLT' in p:
            tlt = p['TLT']; tlt_ret = tlt.pct_change()
            r5 = tlt_ret.rolling(5).sum()
            z = (r5 - r5.rolling(252).mean()) / r5.rolling(252).std()
            sig = pd.Series(0.0, index=z.index)
            sig[z < -1.5] = 1.0   # oversold → long
            sig[z > 1.5] = -1.0   # overbought → short
            self.strategies['reversal_tlt'] = (sig.shift(1) * tlt_ret).dropna()

        # 5. Overnight premium
        if 'SPY' in p and 'SPY_open' in p:
            overnight = p['SPY_open'] / p['SPY'].shift(1) - 1
            self.strategies['overnight'] = overnight.dropna()

        # 6. Correlation regime
        if all(k in p for k in ['SPY', 'TLT', 'GLD', 'SHY']):
            spy_r = p['SPY'].pct_change()
            tlt_r = p['TLT'].pct_change()
            gld_r = p['GLD'].pct_change()
            shy_r = p['SHY'].pct_change()
            sb_corr = spy_r.rolling(60).corr(tlt_r)
            normal = (sb_corr < 0).astype(float)
            stressed = (sb_corr > 0.3).astype(float)
            r = (normal.shift(1) * (0.6 * spy_r + 0.4 * tlt_r) +
                 stressed.shift(1) * (0.3 * spy_r + 0.3 * gld_r + 0.4 * shy_r)).dropna()
            self.strategies['corr_regime'] = r

        # 7. Gold vol regime switch
        if 'GLD' in p:
            gld = p['GLD']; gld_ret = gld.pct_change()
            gld_vol = gld_ret.rolling(20).std() * np.sqrt(252)
            gld_hv = gld_vol > gld_vol.rolling(252).median()
            trend = ((gld / gld.rolling(50).mean() - 1) > 0).astype(float) * 2 - 1
            mr = -(gld - gld.rolling(20).mean()) / gld.rolling(20).std()
            mr = mr.clip(-2, 2) / 2
            sig = pd.Series(0.0, index=gld.index)
            sig[~gld_hv] = trend[~gld_hv]
            sig[gld_hv] = mr[gld_hv]
            self.strategies['vol_regime_gld'] = (sig.shift(1) * gld_ret).dropna()

        # 8. Crypto risk-adjusted momentum
        crypto_tks = [k for k in ['BTC-USD', 'ETH-USD', 'SOL-USD'] if k in p]
        if len(crypto_tks) >= 2:
            cr_p = pd.DataFrame({k: p[k] for k in crypto_tks})
            cr_r = cr_p.pct_change()
            cr_vol = cr_r.rolling(20).std()
            cr_mom = cr_p / cr_p.shift(20) - 1
            cr_ram = cr_mom / cr_vol.replace(0, np.nan)
            pnl = []
            for d in cr_ram.index[60:]:
                prev = cr_ram.index[cr_ram.index.get_loc(d) - 1]
                s = cr_ram.loc[prev].dropna()
                r = cr_r.loc[d].dropna()
                o = s.index.intersection(r.index)
                if len(o) < 2:
                    pnl.append(0)
                    continue
                n = max(1, len(o) // 3)
                pnl.append(r[s.nlargest(n).index].mean() - r[s.nsmallest(n).index].mean())
            self.strategies['crypto_xmom'] = pd.Series(pnl, index=cr_ram.index[60:])

        # 9. Yield curve rotation
        if all(k in p for k in ['SPY', 'TLT', 'SHY']):
            curve = p['TLT'] / p['SHY']
            curve_mom = curve.pct_change(60)
            eq_sig = (curve_mom > 0).astype(float)
            r = (eq_sig.shift(1) * p['SPY'].pct_change() +
                 (1 - eq_sig.shift(1)) * p['TLT'].pct_change()).dropna()
            self.strategies['curve_rot'] = r

        # 10. Multi-asset dual momentum ensemble
        if all(k in p for k in ['SPY', 'GLD', 'TLT', 'SHY']):
            assets = {k: p[k] for k in ['SPY', 'GLD', 'TLT'] if k in p}
            if 'EEM' in p:
                assets['EEM'] = p['EEM']
            if 'VNQ' in p:
                assets['VNQ'] = p['VNQ']
            shy = p['SHY']
            all_r = []
            for tk, price in assets.items():
                ret = price.pct_change()
                mom = price / price.shift(60) - 1
                shy_mom = shy / shy.shift(60) - 1
                sig = ((mom > 0) & (mom > shy_mom.reindex(mom.index))).astype(float)
                vol = ret.rolling(60).std().replace(0, np.nan)
                weighted = sig / vol
                all_r.append((weighted.shift(1) * ret).dropna())
            if all_r:
                self.strategies['xasset_mom'] = pd.concat(all_r, axis=1).mean(axis=1).dropna()

        return self.strategies

    def get_portfolio_returns(self):
        """Combine all strategy returns into a single portfolio."""
        if not self.strategies:
            self.run_all()

        df = pd.DataFrame(self.strategies).dropna()
        if df.empty:
            return pd.Series(dtype=float)

        if self.weighting == 'risk_parity':
            vols = df.std()
            w = (1 / vols) / (1 / vols).sum()
            return (df * w).sum(axis=1)
        else:
            return df.mean(axis=1)

    def get_stats(self):
        """Portfolio statistics."""
        r = self.get_portfolio_returns()
        if len(r) < 50:
            return {}
        sr = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
        cum = (1 + r).cumprod()
        mdd = ((cum - cum.cummax()) / cum.cummax()).min()
        return {
            'sharpe': sr,
            'annual_return': r.mean() * 252,
            'annual_vol': r.std() * np.sqrt(252),
            'max_drawdown': mdd,
            'win_rate': (r > 0).mean(),
            'n_strategies': len(self.strategies),
            'n_days': len(r),
        }

    def get_allocation_signals(self):
        """Return current allocation signals for each strategy.

        Returns dict of {ticker: weight} ready for order generation.
        """
        if not self.strategies:
            self.run_all()

        # Each strategy implies positions in specific assets
        # This is a simplified version — production would track exact positions
        signals = {}
        n = len(self.strategies)
        w = 1.0 / n if n > 0 else 0

        # Map strategy → asset direction
        for name, rets in self.strategies.items():
            if len(rets) < 2:
                continue
            last_ret = rets.iloc[-1]
            direction = 1 if last_ret >= 0 else -1

            if 'gld' in name or 'gold' in name:
                signals.setdefault('GLD', 0)
                signals['GLD'] += w * direction
            elif 'spy' in name or 'safe_haven' in name or 'curve' in name or 'corr' in name:
                signals.setdefault('SPY', 0)
                signals['SPY'] += w * direction
            elif 'tlt' in name or 'reversal_tlt' in name:
                signals.setdefault('TLT', 0)
                signals['TLT'] += w * direction
            elif 'crypto' in name:
                signals.setdefault('BTC-USD', 0)
                signals['BTC-USD'] += w * direction

        return signals
