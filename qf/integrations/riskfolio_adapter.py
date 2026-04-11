"""Riskfolio-Lib 投资组合优化适配器
Riskfolio-Lib: https://github.com/dcajasn/Riskfolio-Lib
提供均值-方差、风险平价、HRP 等多种组合优化方法,
支持多种风险度量 (CVaR, CDaR, EVaR 等)。
"""
import warnings
import numpy as np
import pandas as pd
from typing import Optional

warnings.filterwarnings("ignore")

try:
    import riskfolio as rp

    _HAS_RISKFOLIO = True
except ImportError:
    _HAS_RISKFOLIO = False


def _check_riskfolio():
    if not _HAS_RISKFOLIO:
        raise ImportError(
            "Riskfolio-Lib 未安装。请运行: pip install Riskfolio-Lib"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def optimize_portfolio(
    returns: pd.DataFrame,
    method: str = "MV",
    risk_measure: str = "MV",
    objective: str = "Sharpe",
    risk_free_rate: float = 0.0,
    target_return: Optional[float] = None,
    constraints: Optional[dict] = None,
) -> dict:
    """使用 Riskfolio-Lib 优化投资组合权重。

    Parameters
    ----------
    returns : pd.DataFrame
        资产收益矩阵 (date x asset)
    method : str
        优化方法:
        - 'MV' : 均值-方差 (Mean-Variance)
        - 'RP' : 风险平价 (Risk Parity)
        - 'HRP': 层次风险平价 (Hierarchical Risk Parity)
    risk_measure : str
        风险度量:
        - 'MV'   : 方差 (Variance)
        - 'CVaR' : 条件VaR
        - 'CDaR' : 条件最大回撤
        - 'EVaR' : 熵VaR
        - 'MAD'  : 平均绝对偏差
    objective : str
        优化目标 (仅 MV 方法):
        - 'Sharpe'    : 最大化夏普比率
        - 'MinRisk'   : 最小化风险
        - 'MaxRet'    : 最大化收益 (给定风险约束)
        - 'Utility'   : 最大化效用函数
    risk_free_rate : float
        无风险利率
    target_return : float, optional
        目标收益率 (仅 MaxRet 使用)
    constraints : dict, optional
        额外约束, 如 {'upper_bound': 0.1} 限制单资产最大权重

    Returns
    -------
    dict
        {
            'weights': {asset: weight},
            'expected_return': float,
            'risk': float,
            'sharpe_ratio': float,
            'method': str,
            'risk_measure': str,
        }
    """
    _check_riskfolio()

    returns_clean = returns.dropna(axis=1, how="all").dropna()

    if method == "HRP":
        return _optimize_hrp(returns_clean, risk_measure)
    elif method == "RP":
        return _optimize_risk_parity(returns_clean, risk_measure)
    else:
        return _optimize_mean_variance(
            returns_clean,
            risk_measure=risk_measure,
            objective=objective,
            risk_free_rate=risk_free_rate,
            target_return=target_return,
            constraints=constraints,
        )


def _optimize_mean_variance(
    returns: pd.DataFrame,
    risk_measure: str = "MV",
    objective: str = "Sharpe",
    risk_free_rate: float = 0.0,
    target_return: Optional[float] = None,
    constraints: Optional[dict] = None,
) -> dict:
    """均值-方差优化。"""
    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu="hist", method_cov="hist")

    # Apply constraints
    if constraints:
        if "upper_bound" in constraints:
            port.upperlng = constraints["upper_bound"]
        if "lower_bound" in constraints:
            port.lowerlng = constraints["lower_bound"]

    # Optimize
    w = port.optimization(
        model="Classic",
        rm=risk_measure,
        obj=objective,
        rf=risk_free_rate,
        hist=True,
    )

    if w is None:
        raise ValueError("优化失败, 请检查输入数据或放宽约束")

    weights = w["weights"].to_dict() if isinstance(w, pd.DataFrame) else w.squeeze().to_dict()

    # Compute stats
    mu = returns.mean()
    cov = returns.cov()
    w_arr = np.array([weights.get(c, 0.0) for c in returns.columns])
    exp_ret = float(np.dot(w_arr, mu.values))
    risk = float(np.sqrt(np.dot(w_arr, np.dot(cov.values, w_arr))))
    sharpe = float((exp_ret - risk_free_rate) / risk) if risk > 0 else 0.0

    return {
        "weights": weights,
        "expected_return": exp_ret,
        "risk": risk,
        "sharpe_ratio": sharpe,
        "method": "MV",
        "risk_measure": risk_measure,
        "objective": objective,
    }


def _optimize_risk_parity(
    returns: pd.DataFrame,
    risk_measure: str = "MV",
) -> dict:
    """风险平价优化。"""
    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu="hist", method_cov="hist")

    w = port.rp_optimization(
        model="Classic",
        rm=risk_measure,
        hist=True,
    )

    if w is None:
        raise ValueError("风险平价优化失败")

    weights = w["weights"].to_dict() if isinstance(w, pd.DataFrame) else w.squeeze().to_dict()

    mu = returns.mean()
    cov = returns.cov()
    w_arr = np.array([weights.get(c, 0.0) for c in returns.columns])
    exp_ret = float(np.dot(w_arr, mu.values))
    risk = float(np.sqrt(np.dot(w_arr, np.dot(cov.values, w_arr))))

    return {
        "weights": weights,
        "expected_return": exp_ret,
        "risk": risk,
        "sharpe_ratio": float(exp_ret / risk) if risk > 0 else 0.0,
        "method": "RP",
        "risk_measure": risk_measure,
    }


def _optimize_hrp(
    returns: pd.DataFrame,
    risk_measure: str = "MV",
) -> dict:
    """层次风险平价 (HRP) 优化。"""
    port = rp.HCPortfolio(returns=returns)

    w = port.optimization(
        model="HRP",
        rm=risk_measure,
    )

    if w is None:
        raise ValueError("HRP 优化失败")

    weights = w["weights"].to_dict() if isinstance(w, pd.DataFrame) else w.squeeze().to_dict()

    mu = returns.mean()
    cov = returns.cov()
    w_arr = np.array([weights.get(c, 0.0) for c in returns.columns])
    exp_ret = float(np.dot(w_arr, mu.values))
    risk = float(np.sqrt(np.dot(w_arr, np.dot(cov.values, w_arr))))

    return {
        "weights": weights,
        "expected_return": exp_ret,
        "risk": risk,
        "sharpe_ratio": float(exp_ret / risk) if risk > 0 else 0.0,
        "method": "HRP",
        "risk_measure": risk_measure,
    }


def compute_efficient_frontier(
    returns: pd.DataFrame,
    risk_measure: str = "MV",
    n_points: int = 50,
    risk_free_rate: float = 0.0,
) -> dict:
    """计算有效前沿。

    Parameters
    ----------
    returns : pd.DataFrame
        资产收益矩阵
    risk_measure : str
        风险度量
    n_points : int
        有效前沿点数
    risk_free_rate : float
        无风险利率

    Returns
    -------
    dict
        {
            'frontier_risk': list[float],
            'frontier_return': list[float],
            'frontier_weights': list[dict],
        }
    """
    _check_riskfolio()

    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu="hist", method_cov="hist")

    # Get min and max return portfolios
    w_min = port.optimization(model="Classic", rm=risk_measure, obj="MinRisk", rf=risk_free_rate, hist=True)
    w_max = port.optimization(model="Classic", rm=risk_measure, obj="Sharpe", rf=risk_free_rate, hist=True)

    if w_min is None or w_max is None:
        raise ValueError("有效前沿计算失败")

    mu = returns.mean().values
    cov = returns.cov().values

    w_min_arr = w_min.values.flatten()
    w_max_arr = w_max.values.flatten()

    ret_min = np.dot(w_min_arr, mu)
    ret_max = np.dot(w_max_arr, mu)

    frontier_risk = []
    frontier_return = []
    frontier_weights = []

    target_returns = np.linspace(ret_min, ret_max, n_points)
    for target in target_returns:
        try:
            port_temp = rp.Portfolio(returns=returns)
            port_temp.assets_stats(method_mu="hist", method_cov="hist")
            w = port_temp.optimization(
                model="Classic", rm=risk_measure, obj="MinRisk",
                rf=risk_free_rate, hist=True,
            )
            if w is not None:
                w_arr = w.values.flatten()
                risk = float(np.sqrt(np.dot(w_arr, np.dot(cov, w_arr))))
                ret = float(np.dot(w_arr, mu))
                frontier_risk.append(risk)
                frontier_return.append(ret)
                cols = returns.columns.tolist()
                frontier_weights.append(
                    {cols[i]: float(w_arr[i]) for i in range(len(cols))}
                )
        except Exception:
            continue

    return {
        "frontier_risk": frontier_risk,
        "frontier_return": frontier_return,
        "frontier_weights": frontier_weights,
    }


def compare_optimization_methods(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> dict:
    """比较不同优化方法的结果。

    Parameters
    ----------
    returns : pd.DataFrame
        资产收益矩阵
    risk_free_rate : float
        无风险利率

    Returns
    -------
    dict
        {method_name: {weights, expected_return, risk, sharpe_ratio}}
    """
    _check_riskfolio()

    results = {}

    methods = [
        ("MV_Sharpe", "MV", "MV", "Sharpe"),
        ("MV_MinRisk", "MV", "MV", "MinRisk"),
        ("MV_CVaR", "MV", "CVaR", "Sharpe"),
        ("RiskParity", "RP", "MV", None),
        ("HRP", "HRP", "MV", None),
    ]

    for name, method, rm, obj in methods:
        try:
            if method == "MV":
                r = optimize_portfolio(
                    returns, method=method, risk_measure=rm,
                    objective=obj, risk_free_rate=risk_free_rate,
                )
            else:
                r = optimize_portfolio(
                    returns, method=method, risk_measure=rm,
                    risk_free_rate=risk_free_rate,
                )
            results[name] = r
        except Exception as e:
            results[name] = {"error": str(e)}

    return results
