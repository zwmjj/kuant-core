"""Strategy optimizer — multi-factor blending, volatility targeting, drawdown control"""
import numpy as np
import pandas as pd


# ── 1. 多因子信号混合 ──

def build_combo_signal(combo_name, data, cap_quantile=0.75, verbose=True):
    """Build an optimized combo signal by name"""
    cfg = OPTIMIZED_COMBOS.get(combo_name)
    if cfg is None:
        raise ValueError(f"未知组合: {combo_name}")

    stype = cfg.get('type', 'standard')

    if stype == 'interaction':
        from qf.signals import build_interaction_signal
        signals = {}
        for iid in cfg['interactions']:
            signals[iid] = build_interaction_signal(iid, data, cap_quantile=cap_quantile, verbose=False)
        return _blend_signal_dict(signals, cfg.get('method', 'equal'),
                                  cfg.get('weights'), verbose=verbose)

    elif stype == 'orthogonal':
        from qf.signals import build_orthogonal_signal
        signals = {}
        for fid in cfg['factors']:
            signals[fid] = build_orthogonal_signal(fid, data, cap_quantile=cap_quantile, verbose=False)
        return _blend_signal_dict(signals, cfg.get('method', 'equal'),
                                  cfg.get('weights'), verbose=verbose)

    elif stype == 'regime':
        from qf.signals import build_regime_adjusted_signal
        signals = {}
        for fid in cfg['factors']:
            signals[fid] = build_regime_adjusted_signal(fid, data, cap_quantile=cap_quantile, verbose=False)
        return _blend_signal_dict(signals, cfg.get('method', 'risk_parity'),
                                  cfg.get('weights'), lookback=36, verbose=verbose)

    elif stype == 'ultimate':
        return _build_ultimate_signal(data, cap_quantile=cap_quantile, verbose=verbose)

    else:
        # Standard factor blend
        return blend_factors(data, cfg['factors'], weights=cfg.get('weights'),
                            method=cfg.get('method', 'risk_parity'),
                            cap_quantile=cap_quantile, verbose=verbose)


def _blend_signal_dict(signals, method='equal', weights=None, lookback=36, verbose=True):
    """内部函数: 混合一组已生成的信号"""
    from qf.signals import SignalGenerator
    sg = SignalGenerator()

    common_idx = list(signals.values())[0].index
    common_cols = list(signals.values())[0].columns
    for sig in signals.values():
        common_idx = common_idx.intersection(sig.index)
        common_cols = common_cols.intersection(sig.columns)
    aligned = {k: v.loc[common_idx, common_cols].astype(float) for k, v in signals.items()}

    if method == 'fixed' and weights:
        w = {k: weights.get(k, 0) for k in aligned}
        total = sum(w.values())
        w = {k: v / total for k, v in w.items()}
    elif method == 'equal':
        w = {k: 1.0 / len(aligned) for k in aligned}
    else:
        w = {k: 1.0 / len(aligned) for k in aligned}

    blended = pd.DataFrame(0.0, index=common_idx, columns=common_cols)
    for fid, sig in aligned.items():
        blended += w[fid] * sig.fillna(0)

    blended = sg.cross_sectional_rank(blended)
    if verbose:
        wstr = ", ".join(f"{k}={v:.0%}" for k, v in w.items())
        n = blended.notna().sum(axis=1).median()
        print(f"  混合: {wstr} | ~{n:.0f}只/月")
    return blended


def _build_ultimate_signal(data, cap_quantile=0.75, verbose=True):
    """终极组合: 交互+正交+择时+最佳因子 四层融合"""
    from qf.signals import (build_interaction_signal, build_orthogonal_signal,
                             build_regime_adjusted_signal, build_factor_signal, SignalGenerator)
    sg = SignalGenerator()
    signals = {}

    # Layer 1: Best interaction signals (30%)
    try:
        signals['qm_interact'] = build_interaction_signal('quality_mom', data, cap_quantile=cap_quantile, verbose=False)
    except: pass
    try:
        signals['aq_interact'] = build_interaction_signal('alpha_quality', data, cap_quantile=cap_quantile, verbose=False)
    except: pass

    # Layer 2: Orthogonalized best factors (30%)
    try:
        signals['gpa_orth'] = build_orthogonal_signal('gpa', data, cap_quantile=cap_quantile, verbose=False)
    except: pass
    try:
        signals['roe_orth'] = build_orthogonal_signal('roe', data, cap_quantile=cap_quantile, verbose=False)
    except: pass

    # Layer 3: Regime-adjusted momentum (20%)
    try:
        signals['mom_regime'] = build_regime_adjusted_signal('mom12', data, cap_quantile=cap_quantile, verbose=False)
    except: pass

    # Layer 4: Raw best factor (20%)
    try:
        signals['gpa_raw'] = build_factor_signal('gpa', data, cap_quantile=cap_quantile, verbose=False)
    except: pass

    if len(signals) < 3:
        raise ValueError(f"终极组合需要至少3个信号层，仅生成{len(signals)}个")

    weights = {
        'qm_interact': 0.15, 'aq_interact': 0.15,
        'gpa_orth': 0.15, 'roe_orth': 0.15,
        'mom_regime': 0.20,
        'gpa_raw': 0.20,
    }

    return _blend_signal_dict(signals, 'fixed', weights, verbose=verbose)


def blend_factors(data, factor_ids, weights=None, method='risk_parity',
                  lookback=36, cap_quantile=0.75, verbose=True):
    """Blend multiple factor signals at the stock level

    Parameters
    ----------
    method : str
        'equal' - equal-weighted blend
        'risk_parity' - weight by the inverse of each factor's volatility (from rolling backtest returns)
        'sharpe_weighted' - weight by each factor's rolling Sharpe ratio
        'fixed' - use the fixed weights given by the weights argument
    """
    from qf.signals import build_factor_signal, SignalGenerator

    sg = SignalGenerator()
    signals = {}
    for fid in factor_ids:
        try:
            sig = build_factor_signal(fid, data, cap_quantile=cap_quantile, verbose=False)
            signals[fid] = sig
        except Exception as e:
            if verbose:
                print(f"  跳过因子 {fid}: {e}")

    if not signals:
        raise ValueError("无可用因子信号")

    # Align all signals to common index
    common_idx = signals[list(signals.keys())[0]].index
    common_cols = signals[list(signals.keys())[0]].columns
    for fid, sig in signals.items():
        common_idx = common_idx.intersection(sig.index)
        common_cols = common_cols.intersection(sig.columns)

    aligned = {fid: sig.loc[common_idx, common_cols].astype(float) for fid, sig in signals.items()}

    if method == 'fixed' and weights:
        w = {fid: weights.get(fid, 0) for fid in aligned}
        total = sum(w.values())
        w = {k: v / total for k, v in w.items()}
    elif method == 'equal':
        w = {fid: 1.0 / len(aligned) for fid in aligned}
    elif method == 'risk_parity':
        w = _risk_parity_weights(aligned, common_idx, lookback)
    elif method == 'sharpe_weighted':
        w = _sharpe_weights(aligned, common_idx, lookback)
    else:
        w = {fid: 1.0 / len(aligned) for fid in aligned}

    # Blend — detect if weights are static (float) or time-varying (Series)
    blended = pd.DataFrame(0.0, index=common_idx, columns=common_cols)
    sample_w = next(iter(w.values()))
    is_static = isinstance(sample_w, (int, float, np.floating))

    if is_static:
        for fid, sig in aligned.items():
            blended += w[fid] * sig.fillna(0)
    else:
        # Time-varying weights (dict of Series)
        for fid, sig in aligned.items():
            wt = w[fid].reindex(common_idx).fillna(1.0 / len(aligned))
            sig_filled = sig.fillna(0)
            for col in common_cols:
                blended[col] += wt.values * sig_filled[col].values

    # Re-rank the blended signal cross-sectionally
    blended = sg.cross_sectional_rank(blended)

    if verbose:
        if isinstance(w, dict):
            wstr = ", ".join(f"{k}={v:.1%}" for k, v in w.items())
        else:
            wstr = f"time-varying {method}"
        print(f"  混合信号: {len(aligned)} 因子, 方法={method}")
        print(f"  权重: {wstr}")
        n = blended.notna().sum(axis=1).median()
        print(f"  有效信号: ~{n:.0f} 只/月")

    return blended


def _risk_parity_weights(aligned_signals, index, lookback):
    """基于因子信号截面均值收益率的波动率倒数加权"""
    # Compute factor "return" as cross-sectional mean of signal changes
    factor_means = {}
    for fid, sig in aligned_signals.items():
        factor_means[fid] = sig.mean(axis=1)

    means_df = pd.DataFrame(factor_means)
    vols = means_df.rolling(lookback, min_periods=12).std()
    inv_vols = 1.0 / vols.replace(0, np.nan)
    weights = inv_vols.div(inv_vols.sum(axis=1), axis=0)

    # Return time-varying weights
    result = {}
    for fid in aligned_signals:
        result[fid] = weights[fid] if fid in weights.columns else pd.Series(
            1.0 / len(aligned_signals), index=index)
    return result


def _sharpe_weights(aligned_signals, index, lookback):
    """基于滚动夏普的权重"""
    factor_means = {}
    for fid, sig in aligned_signals.items():
        factor_means[fid] = sig.mean(axis=1)

    means_df = pd.DataFrame(factor_means)
    rolling_mean = means_df.rolling(lookback, min_periods=12).mean()
    rolling_std = means_df.rolling(lookback, min_periods=12).std()
    sharpes = rolling_mean / rolling_std.replace(0, np.nan)
    sharpes = sharpes.clip(lower=0)  # Only positive Sharpe contributors
    weights = sharpes.div(sharpes.sum(axis=1), axis=0)

    result = {}
    for fid in aligned_signals:
        result[fid] = weights[fid] if fid in weights.columns else pd.Series(
            1.0 / len(aligned_signals), index=index)
    return result


# ── 2. 预定义多因子组合 ──

OPTIMIZED_COMBOS = {
    # ── Round 0: 基础多因子 ──
    'quality_momentum': {
        'factors': ['gpa', 'roe', 'mom12', 'ff5alpha'],
        'method': 'risk_parity',
        'description': '质量+动量 4因子',
    },
    'concentrated': {
        'factors': ['gpa', 'ff5alpha', 'mom12'],
        'method': 'fixed',
        'weights': {'gpa': 0.40, 'ff5alpha': 0.35, 'mom12': 0.25},
        'description': '集中3因子 (GPA40% FF5α35% MOM25%)',
    },
    'all_weather': {
        'factors': ['gpa', 'roe', 'mom12', 'bm', 'ag', 'ff5alpha'],
        'method': 'risk_parity',
        'description': '全天候6因子',
    },
    # ── Round 1: 纯净alpha (剔除拥挤因子, 聚焦低R²高alpha) ──
    'pure_alpha': {
        'factors': ['gpa', 'roe', 'ff5alpha'],
        'method': 'fixed',
        'weights': {'gpa': 0.45, 'roe': 0.30, 'ff5alpha': 0.25},
        'description': '纯Alpha三因子(仅低R²高Alpha因子)',
    },
    # ── Round 1: 交互信号组合 ──
    'interaction_blend': {
        'type': 'interaction',
        'interactions': ['quality_mom', 'value_quality', 'profit_growth'],
        'method': 'equal',
        'description': '交互信号混合(质量×动量 + 价值×质量 + 盈利×低增长)',
    },
    'best_interaction': {
        'type': 'interaction',
        'interactions': ['quality_mom', 'alpha_quality'],
        'method': 'fixed',
        'weights': {'quality_mom': 0.50, 'alpha_quality': 0.50},
        'description': '最佳交互(质量×动量50% + Alpha×质量50%)',
    },
    # ── Round 2: 正交化 ──
    'orthogonal_blend': {
        'type': 'orthogonal',
        'factors': ['gpa', 'roe', 'ff5alpha'],
        'method': 'equal',
        'description': '正交化三因子(剥离size+value暴露)',
    },
    # ── Round 2: 择时 ──
    'regime_blend': {
        'type': 'regime',
        'factors': ['gpa', 'roe', 'mom12', 'ff5alpha'],
        'method': 'risk_parity',
        'description': '择时四因子(高波动减动量)',
    },
    # ── statistically optimized (2026-04) ──
    'gpa_mom_v2': {
        'factors': ['gpa', 'mom12'],
        'method': 'fixed',
        'weights': {'gpa': 0.70, 'mom12': 0.30},
        'description': '统计最优: GPA70+Mom30, Sharpe 0.95, MDD -35%',
    },
    'gpa_mom_ep': {
        'factors': ['gpa', 'mom12', 'ep'],
        'method': 'fixed',
        'weights': {'gpa': 0.63, 'mom12': 0.27, 'ep': 0.10},
        'description': '统计最优3因子: GPA63+Mom27+EP10, Sharpe 0.96, MDD -27.5%',
    },
    'gpa_mom_roe': {
        'factors': ['gpa', 'mom12', 'roe'],
        'method': 'fixed',
        'weights': {'gpa': 0.63, 'mom12': 0.27, 'roe': 0.10},
        'description': '质量动量: GPA63+Mom27+ROE10, Sharpe 0.96, MDD -32%',
    },
    # ── Round 3: 最终组合 ──
    'ultimate': {
        'type': 'ultimate',
        'description': '终极组合: 交互信号+正交化+择时+最佳因子',
    },
}


# ── 3. 波动率目标 + 回撤控制 ──

def vol_target_scale(returns_history, target_vol=0.10, lookback=6, max_leverage=1.5):
    """Compute the position scaling factor for volatility targeting

    Parameters
    ----------
    target_vol : float
        Annualized target volatility (default 10%)
    lookback : int
        Lookback window in months (default 6)
    max_leverage : float
        Maximum leverage (default 1.5)
    """
    if len(returns_history) < lookback:
        return 1.0
    recent = np.array(returns_history[-lookback:])
    realized_vol = np.std(recent, ddof=1) * np.sqrt(12)
    if realized_vol < 0.01:
        return 1.0
    scale = target_vol / realized_vol
    return min(max(scale, 0.3), max_leverage)


def drawdown_scale(pv_history, threshold_1=0.10, threshold_2=0.20, threshold_3=0.25):
    """Compute the position scaling factor for drawdown control

    Parameters
    ----------
    threshold_1 : float
        Mild drawdown threshold → cut positions to 80% (default 10%)
    threshold_2 : float
        Moderate drawdown threshold → cut positions to 50% (default 20%)
    threshold_3 : float
        Severe drawdown threshold → cut positions to 25% (default 25%)
    """
    if len(pv_history) < 3:
        return 1.0
    peak = max(pv_history)
    current = pv_history[-1]
    dd = (current - peak) / peak  # negative number
    if dd < -threshold_3:
        return 0.25
    elif dd < -threshold_2:
        return 0.50
    elif dd < -threshold_1:
        return 0.80
    return 1.0


def sigmoid_regime_scale(returns_history, vix_value=None,
                         center=25.0, steepness=0.3, min_weight=0.2):
    """Smooth sigmoid position scaling based on VIX/vol regime

    Replaces hard VIX threshold with smooth transition:
    weight = min_weight + (1-min_weight) / (1 + exp(steepness*(vol-center)))

    Falls back to rolling realized vol proxy if vix_value is None.
    """
    if vix_value is not None:
        x = vix_value
    elif len(returns_history) >= 6:
        # Use annualized realized vol as VIX proxy
        x = np.std(returns_history[-6:], ddof=1) * np.sqrt(12) * 100  # scale to VIX-like
    else:
        return 1.0

    weight = min_weight + (1.0 - min_weight) / (1.0 + np.exp(steepness * (x - center)))
    return float(weight)


def run_optimized_backtest(d, signal, target_vol=0.10, dd_control=True,
                           long_n=20, short_n=20, base_long_pct=1.15, base_short_pct=0.15,
                           weight_mode='inv_vol', turnover_penalty=0.25,
                           cost_model='sqrt', verbose=True, initial_capital=10000, **cost_kw):
    """Run the optimized backtest with volatility targeting and drawdown control"""
    from qf.backtest import DataHandler, StrategyEngine, Portfolio, EventDrivenBacktester, BacktestResult
    from qf.costs import ExecutionHandler

    inv_vol = 1.0 / d['returns'].rolling(12).std().replace(0, np.nan)
    dh = DataHandler(d['prices'], d['returns'], d.get('volume'), d.get('adv_dollar'))

    # We'll manually drive the backtest loop for dynamic position sizing
    port = Portfolio(initial_capital=initial_capital)
    exe = ExecutionHandler(cost_model=cost_model, **cost_kw)

    dates_with_ret = []
    if verbose:
        print(f"\n  运行优化回测 (vol_target={target_vol:.0%}, dd_control={dd_control})...")

    for bar in dh.iter_bars():
        date = bar['date']
        if date not in signal.index:
            continue
        sig_row = signal.loc[date].dropna()
        valid = sig_row[sig_row.index.isin(bar['tradable'])]
        if len(valid) < long_n + short_n:
            continue

        # Compute dynamic scale
        v_scale = vol_target_scale(port.return_history, target_vol=target_vol) if target_vol > 0 else 1.0
        d_scale = drawdown_scale(port.pv_history) if dd_control else 1.0
        scale = v_scale * d_scale

        # Adjusted exposure
        adj_long_pct = base_long_pct * scale
        adj_short_pct = base_short_pct * scale

        # Build signal manually with adjusted exposure
        adj = valid.copy()
        if turnover_penalty > 0:
            for p in getattr(port, '_prev_long', set()):
                if p in adj.index:
                    adj[p] += turnover_penalty
            for p in getattr(port, '_prev_short', set()):
                if p in adj.index:
                    adj[p] -= turnover_penalty

        long_list = adj.nlargest(long_n).index.tolist()
        short_list = adj.nsmallest(short_n).index.tolist()

        # Weighting
        if weight_mode == 'inv_vol' and date in inv_vol.index:
            iv_vals = inv_vol.loc[date, long_list].dropna()
            if len(iv_vals) > 0:
                long_w = (iv_vals / iv_vals.sum() * adj_long_pct).to_dict()
            else:
                long_w = {t: adj_long_pct / long_n for t in long_list}
        else:
            long_w = {t: adj_long_pct / long_n for t in long_list}
        short_w = {t: adj_short_pct / short_n for t in short_list}

        port._prev_long = set(long_w.keys())
        port._prev_short = set(short_w.keys())

        from qf.costs import SignalEvent, OrderEvent
        sig_evt = SignalEvent(date=date, long_targets=long_w, short_targets=short_w)
        order = port.on_signal(sig_evt, bar)
        _, month_ret = exe.on_order(order, bar, port)
        port.update_pv(month_ret)
        dates_with_ret.append(date)

    if verbose:
        n = len(dates_with_ret)
        print(f"  完成: {n} 个月, {port.trade_count} 笔交易")

    pv = pd.Series([port.initial_capital] + port.pv_history,
                   index=[dh.dates[0]] + dates_with_ret)
    rets = pd.Series(port.return_history, index=dates_with_ret)
    result = BacktestResult(pv, rets, pd.DataFrame())
    m = result.metrics(rf=d['rf'], benchmark_returns=d['spy_ret'])
    return result, m
