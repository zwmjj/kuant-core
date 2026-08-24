"""quantstats-lumi performance reporting adapter
quantstats-lumi: https://github.com/Lumiwealth/quantstats_lumi
Provides strategy performance reports, benchmark comparison, drawdown analysis, monthly heatmaps and more.
Results are returned as dicts for easy JSON serialization to the API layer.
"""
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Union

warnings.filterwarnings("ignore")

try:
    import quantstats_lumi as qs

    _HAS_QS = True
except ImportError:
    try:
        import quantstats as qs

        _HAS_QS = True
    except ImportError:
        _HAS_QS = False


def _check_qs():
    if not _HAS_QS:
        raise ImportError(
            "quantstats-lumi 未安装。请运行: pip install quantstats-lumi"
        )


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _to_returns_series(data: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    """将输入转为 pd.Series 收益率序列。"""
    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            return data.iloc[:, 0].dropna()
        raise ValueError("DataFrame 需为单列, 或请传入 pd.Series")
    return data.dropna()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_report(
    returns: Union[pd.Series, pd.DataFrame],
    benchmark: Optional[Union[pd.Series, pd.DataFrame]] = None,
    output_file: Optional[str] = None,
    title: str = "Kuant Strategy Report",
) -> Optional[str]:
    """Generate a full HTML performance report.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Strategy return series (daily)
    benchmark : pd.Series or pd.DataFrame, optional
        Benchmark return series
    output_file : str, optional
        Output HTML file path. If None, the HTML string is returned
    title : str
        Report title

    Returns
    -------
    str or None
        HTML string (if output_file=None), otherwise the file is written and None is returned
    """
    _check_qs()

    ret = _to_returns_series(returns)
    bench = _to_returns_series(benchmark) if benchmark is not None else None

    if output_file:
        qs.reports.html(
            ret,
            benchmark=bench,
            output=output_file,
            title=title,
        )
        return None
    else:
        return qs.reports.html(
            ret,
            benchmark=bench,
            title=title,
            output=True,
        )


def compute_metrics(
    returns: Union[pd.Series, pd.DataFrame],
    benchmark: Optional[Union[pd.Series, pd.DataFrame]] = None,
    risk_free_rate: float = 0.0,
) -> dict:
    """Compute a comprehensive set of performance metrics.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Strategy return series
    benchmark : pd.Series or pd.DataFrame, optional
        Benchmark return series
    risk_free_rate : float
        Risk-free rate (annualized)

    Returns
    -------
    dict
        {
            'total_return': float,
            'cagr': float,
            'sharpe': float,
            'sortino': float,
            'max_drawdown': float,
            'calmar': float,
            'volatility': float,
            'win_rate': float,
            ...
        }
    """
    _check_qs()

    ret = _to_returns_series(returns)

    metrics = {
        "total_return": float(qs.stats.comp(ret)),
        "cagr": float(qs.stats.cagr(ret)),
        "sharpe": float(qs.stats.sharpe(ret, rf=risk_free_rate)),
        "sortino": float(qs.stats.sortino(ret, rf=risk_free_rate)),
        "max_drawdown": float(qs.stats.max_drawdown(ret)),
        "calmar": float(qs.stats.calmar(ret)),
        "volatility": float(qs.stats.volatility(ret)),
        "win_rate": float(qs.stats.win_rate(ret)),
        "avg_win": float(qs.stats.avg_win(ret)),
        "avg_loss": float(qs.stats.avg_loss(ret)),
        "payoff_ratio": float(qs.stats.payoff_ratio(ret)),
        "profit_factor": float(qs.stats.profit_factor(ret)),
        "skew": float(qs.stats.skew(ret)),
        "kurtosis": float(qs.stats.kurtosis(ret)),
        "var_95": float(qs.stats.var(ret)),
        "cvar_95": float(qs.stats.cvar(ret)),
    }

    # Benchmark-relative metrics
    if benchmark is not None:
        bench = _to_returns_series(benchmark)
        try:
            metrics["alpha"] = float(qs.stats.greeks(ret, bench).get("alpha", 0.0))
            metrics["beta"] = float(qs.stats.greeks(ret, bench).get("beta", 0.0))
        except Exception:
            metrics["alpha"] = 0.0
            metrics["beta"] = 0.0
        try:
            metrics["information_ratio"] = float(qs.stats.information_ratio(ret, bench))
        except Exception:
            metrics["information_ratio"] = 0.0

    return metrics


def compute_drawdown_analysis(
    returns: Union[pd.Series, pd.DataFrame],
    top_n: int = 5,
) -> dict:
    """Drawdown analysis.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Strategy return series
    top_n : int
        Number of largest drawdowns to return

    Returns
    -------
    dict
        {
            'max_drawdown': float,
            'avg_drawdown': float,
            'drawdown_series': list,
            'worst_drawdowns': list[dict],
        }
    """
    _check_qs()

    ret = _to_returns_series(returns)

    # Compute drawdown series
    cum = (1 + ret).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max

    # Find worst drawdowns
    dd_details = qs.stats.to_drawdown_series(ret)

    result = {
        "max_drawdown": float(dd.min()),
        "avg_drawdown": float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0,
        "drawdown_series": dd.tolist(),
        "drawdown_dates": [str(d) for d in dd.index],
    }

    return result


def compute_monthly_returns(
    returns: Union[pd.Series, pd.DataFrame],
) -> dict:
    """Compute the monthly return matrix (for heatmaps).

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Strategy return series (daily)

    Returns
    -------
    dict
        {
            'monthly_returns': {year: {month: return}},
            'annual_returns': {year: return},
        }
    """
    _check_qs()

    ret = _to_returns_series(returns)

    monthly = qs.stats.monthly_returns(ret)

    # Convert to nested dict
    monthly_dict = {}
    if isinstance(monthly, pd.DataFrame):
        for year in monthly.index:
            monthly_dict[int(year)] = {}
            for month in monthly.columns:
                val = monthly.loc[year, month]
                if pd.notna(val):
                    monthly_dict[int(year)][str(month)] = float(val)

    # Annual returns
    annual = ret.resample("Y").apply(lambda x: (1 + x).prod() - 1)
    annual_dict = {int(d.year): float(v) for d, v in annual.items() if pd.notna(v)}

    return {
        "monthly_returns": monthly_dict,
        "annual_returns": annual_dict,
    }


def compare_strategies(
    strategies: dict,
    benchmark: Optional[Union[pd.Series, pd.DataFrame]] = None,
    risk_free_rate: float = 0.0,
) -> dict:
    """Compare the performance of multiple strategies.

    Parameters
    ----------
    strategies : dict
        {strategy_name: returns_series}
    benchmark : pd.Series or pd.DataFrame, optional
        Benchmark returns
    risk_free_rate : float
        Risk-free rate

    Returns
    -------
    dict
        {strategy_name: metrics_dict}
    """
    _check_qs()

    results = {}
    for name, ret in strategies.items():
        try:
            metrics = compute_metrics(ret, benchmark=benchmark, risk_free_rate=risk_free_rate)
            results[name] = metrics
        except Exception as e:
            results[name] = {"error": str(e)}

    return results


def snapshot(
    returns: Union[pd.Series, pd.DataFrame],
    benchmark: Optional[Union[pd.Series, pd.DataFrame]] = None,
    title: str = "Strategy Snapshot",
    output_file: Optional[str] = None,
) -> dict:
    """Generate a strategy snapshot (key metrics plus a brief performance summary).

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Strategy return series
    benchmark : pd.Series or pd.DataFrame, optional
        Benchmark returns
    title : str
        Report title
    output_file : str, optional
        If provided, the chart is saved to this file

    Returns
    -------
    dict
        Snapshot summary
    """
    _check_qs()

    ret = _to_returns_series(returns)

    metrics = compute_metrics(ret, benchmark=benchmark)
    dd = compute_drawdown_analysis(ret)
    monthly = compute_monthly_returns(ret)

    snapshot_data = {
        "title": title,
        "metrics": metrics,
        "max_drawdown": dd["max_drawdown"],
        "avg_drawdown": dd["avg_drawdown"],
        "annual_returns": monthly["annual_returns"],
        "start_date": str(ret.index[0]),
        "end_date": str(ret.index[-1]),
        "n_days": len(ret),
    }

    # Optionally save plot
    if output_file:
        try:
            qs.plots.snapshot(ret, title=title, savefig=output_file)
        except Exception:
            pass

    return snapshot_data
