"""Alphalens 因子分析适配器
Alphalens: https://github.com/quantopian/alphalens
提供因子 IC 分析、分组收益、换手率等量化因子评估工具。
结果以 dict 返回, 方便 JSON 序列化传给 API 层。
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import alphalens
    from alphalens.utils import get_clean_factor_and_forward_returns
    from alphalens import performance as al_perf
    from alphalens import tears as al_tears

    _HAS_ALPHALENS = True
except ImportError:
    _HAS_ALPHALENS = False


def _check_alphalens():
    if not _HAS_ALPHALENS:
        raise ImportError(
            "alphalens 未安装。请运行: pip install alphalens-reloaded"
        )


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _prepare_factor(
    factor_data: pd.DataFrame,
    prices: pd.DataFrame,
    periods: tuple = (1, 5, 10),
    quantiles: int = 5,
    max_loss: float = 0.35,
) -> pd.DataFrame:
    """将 Kuant 信号矩阵 (date x stock) 转为 Alphalens 格式。

    Parameters
    ----------
    factor_data : pd.DataFrame
        信号矩阵 (date x stock), 如 SignalGenerator 输出
    prices : pd.DataFrame
        价格矩阵 (date x stock)
    periods : tuple
        前向收益计算期 (交易日)
    quantiles : int
        分组数
    max_loss : float
        允许的最大数据缺失比例

    Returns
    -------
    pd.DataFrame
        Alphalens 标准格式的 factor + forward returns
    """
    _check_alphalens()

    # Stack factor to MultiIndex (date, asset)
    factor_stacked = factor_data.stack()
    factor_stacked.index.names = ["date", "asset"]
    factor_stacked = factor_stacked.dropna()

    clean = get_clean_factor_and_forward_returns(
        factor=factor_stacked,
        prices=prices,
        quantiles=quantiles,
        periods=periods,
        max_loss=max_loss,
    )
    return clean


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_factor_analysis(
    factor_data: pd.DataFrame,
    prices: pd.DataFrame,
    periods: tuple = (1, 5, 10),
    quantiles: int = 5,
) -> dict:
    """运行完整因子分析, 返回 JSON-serializable 结果。

    Parameters
    ----------
    factor_data : pd.DataFrame
        信号矩阵 (date x stock)
    prices : pd.DataFrame
        价格矩阵 (date x stock)
    periods : tuple
        前向收益计算期
    quantiles : int
        分组数

    Returns
    -------
    dict
        {
            'ic_summary': {...},
            'quantile_returns': {...},
            'turnover': {...},
            'factor_stats': {...},
        }
    """
    clean = _prepare_factor(factor_data, prices, periods, quantiles)

    # IC 分析
    ic = al_perf.factor_information_coefficient(clean)
    ic_summary = {
        "ic_mean": ic.mean().to_dict(),
        "ic_std": ic.std().to_dict(),
        "ic_ir": (ic.mean() / ic.std()).to_dict(),
        "ic_positive_pct": (ic > 0).mean().to_dict(),
    }

    # 分组收益
    mean_ret_by_q = al_perf.mean_return_by_quantile(clean)[0]
    q_returns = {}
    for col in mean_ret_by_q.columns:
        q_returns[str(col)] = mean_ret_by_q[col].to_dict()

    # 换手率
    try:
        turnover_data = {}
        for p in periods:
            col_name = f"{p}D"
            auto = al_perf.factor_autocorrelation(clean, period=p)
            turnover_data[col_name] = {
                "autocorrelation_mean": float(auto.mean()),
                "autocorrelation_std": float(auto.std()),
            }
    except Exception:
        turnover_data = {}

    # 统计摘要
    factor_vals = clean["factor"]
    stats = {
        "n_observations": int(len(factor_vals)),
        "n_dates": int(factor_vals.index.get_level_values(0).nunique()),
        "n_assets": int(factor_vals.index.get_level_values(1).nunique()),
        "factor_mean": float(factor_vals.mean()),
        "factor_std": float(factor_vals.std()),
    }

    return {
        "ic_summary": ic_summary,
        "quantile_returns": q_returns,
        "turnover": turnover_data,
        "factor_stats": stats,
    }


def compute_ic_series(
    factor: pd.DataFrame,
    returns: pd.DataFrame,
    method: str = "spearman",
) -> pd.Series:
    """计算截面 IC (Information Coefficient) 时间序列。

    不依赖 Alphalens, 纯 pandas 实现。

    Parameters
    ----------
    factor : pd.DataFrame
        信号矩阵 (date x stock)
    returns : pd.DataFrame
        下期收益矩阵 (date x stock)
    method : str
        'spearman' (rank IC) 或 'pearson'

    Returns
    -------
    pd.Series
        每期截面相关系数
    """
    common_dates = factor.index.intersection(returns.index)
    common_cols = factor.columns.intersection(returns.columns)

    ic_list = []
    for date in common_dates:
        f = factor.loc[date, common_cols].dropna()
        r = returns.loc[date, common_cols].dropna()
        common = f.index.intersection(r.index)
        if len(common) < 20:
            ic_list.append(np.nan)
            continue
        if method == "spearman":
            ic_list.append(f[common].rank().corr(r[common].rank()))
        else:
            ic_list.append(f[common].corr(r[common]))

    return pd.Series(ic_list, index=common_dates, name="IC")


def factor_tear_sheet(
    factor: pd.DataFrame,
    returns: pd.DataFrame,
    quantiles: int = 5,
) -> dict:
    """生成因子 tear sheet 摘要 (纯 pandas 实现, 不依赖 Alphalens)。

    Parameters
    ----------
    factor : pd.DataFrame
        信号矩阵
    returns : pd.DataFrame
        下期收益矩阵
    quantiles : int
        分组数

    Returns
    -------
    dict
        JSON-serializable tear sheet
    """
    common_dates = factor.index.intersection(returns.index)
    common_cols = factor.columns.intersection(returns.columns)
    factor_a = factor.loc[common_dates, common_cols]
    returns_a = returns.loc[common_dates, common_cols]

    # IC series
    ic = compute_ic_series(factor_a, returns_a)

    # Quantile returns
    quantile_rets = {f"Q{i+1}": [] for i in range(quantiles)}
    for date in common_dates:
        f = factor_a.loc[date].dropna()
        r = returns_a.loc[date].dropna()
        common = f.index.intersection(r.index)
        if len(common) < quantiles * 5:
            continue
        f_c = f[common]
        r_c = r[common]
        q_labels = pd.qcut(f_c, quantiles, labels=False, duplicates="drop")
        for q in range(quantiles):
            mask = q_labels == q
            if mask.sum() > 0:
                quantile_rets[f"Q{q+1}"].append(r_c[mask].mean())
            else:
                quantile_rets[f"Q{q+1}"].append(np.nan)

    q_mean = {k: float(np.nanmean(v)) if v else np.nan for k, v in quantile_rets.items()}
    q_std = {k: float(np.nanstd(v)) if v else np.nan for k, v in quantile_rets.items()}

    # Long-short return
    ls_rets = []
    for i in range(len(quantile_rets[f"Q{quantiles}"])):
        top = quantile_rets[f"Q{quantiles}"][i]
        bot = quantile_rets["Q1"][i]
        if pd.notna(top) and pd.notna(bot):
            ls_rets.append(top - bot)

    return {
        "ic_mean": float(ic.mean()),
        "ic_std": float(ic.std()),
        "ic_ir": float(ic.mean() / ic.std()) if ic.std() > 0 else 0.0,
        "ic_positive_pct": float((ic > 0).mean()),
        "quantile_mean_returns": q_mean,
        "quantile_std_returns": q_std,
        "long_short_mean": float(np.nanmean(ls_rets)) if ls_rets else 0.0,
        "long_short_sharpe": (
            float(np.nanmean(ls_rets) / np.nanstd(ls_rets) * np.sqrt(12))
            if ls_rets and np.nanstd(ls_rets) > 0
            else 0.0
        ),
        "n_periods": len(common_dates),
    }


def compute_turnover(factor: pd.DataFrame, top_n: int = 20) -> pd.Series:
    """计算因子信号的换手率 (top N 持仓变化比例)。

    Parameters
    ----------
    factor : pd.DataFrame
        信号矩阵
    top_n : int
        持仓数量

    Returns
    -------
    pd.Series
        每期换手率 [0, 1]
    """
    turnover_list = []
    prev_holdings = set()
    for date in factor.index:
        row = factor.loc[date].dropna().nlargest(top_n)
        current = set(row.index)
        if prev_holdings:
            changed = len(current - prev_holdings) + len(prev_holdings - current)
            turnover_list.append(changed / (2 * top_n))
        else:
            turnover_list.append(np.nan)
        prev_holdings = current

    return pd.Series(turnover_list, index=factor.index, name="turnover")


def compute_quantile_returns(
    factor: pd.DataFrame,
    returns: pd.DataFrame,
    quantiles: int = 5,
) -> pd.DataFrame:
    """计算每个分位组的平均收益时间序列。

    Parameters
    ----------
    factor : pd.DataFrame
        信号矩阵
    returns : pd.DataFrame
        下期收益矩阵
    quantiles : int
        分组数

    Returns
    -------
    pd.DataFrame
        index=date, columns=Q1..Q{quantiles}
    """
    common_dates = factor.index.intersection(returns.index)
    common_cols = factor.columns.intersection(returns.columns)

    records = []
    for date in common_dates:
        f = factor.loc[date, common_cols].dropna()
        r = returns.loc[date, common_cols].dropna()
        common = f.index.intersection(r.index)
        if len(common) < quantiles * 5:
            continue
        f_c = f[common]
        r_c = r[common]
        q_labels = pd.qcut(f_c, quantiles, labels=False, duplicates="drop")
        row = {"date": date}
        for q in range(quantiles):
            mask = q_labels == q
            row[f"Q{q+1}"] = r_c[mask].mean() if mask.sum() > 0 else np.nan
        records.append(row)

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).set_index("date")
    return df
