"""Qlib 风格 ML 模型适配器
Microsoft Qlib: https://github.com/microsoft/qlib
封装 LightGBM / XGBoost 树模型训练与预测, 支持 walk-forward 验证。
不直接依赖 Qlib, 而是复现其核心 ML pipeline 模式。
"""
import warnings
import numpy as np
import pandas as pd
from typing import Any, Optional

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb

    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    import xgboost as xgb

    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Default hyper-parameters (Qlib LGBModel style)
# ---------------------------------------------------------------------------

_DEFAULT_LGB_PARAMS = {
    "objective": "regression",
    "metric": "mse",
    "boosting_type": "gbdt",
    "num_leaves": 128,
    "learning_rate": 0.05,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "min_child_samples": 20,
    "verbose": -1,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
}

_DEFAULT_XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "verbosity": 0,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_lightgbm_model(
    features: pd.DataFrame,
    returns: pd.Series,
    params: Optional[dict] = None,
    val_ratio: float = 0.15,
) -> Any:
    """训练 LightGBM 回归模型 (Qlib LGBModel 风格)。

    Parameters
    ----------
    features : pd.DataFrame
        特征矩阵 (n_samples x n_features)
    returns : pd.Series
        目标变量 (下期收益率)
    params : dict, optional
        LightGBM 超参数; None 则使用默认值
    val_ratio : float
        验证集比例 (从尾部切分, 保持时间序列顺序)

    Returns
    -------
    lgb.LGBMRegressor
        训练好的模型
    """
    if not _HAS_LGB:
        raise ImportError("lightgbm 未安装。请运行: pip install lightgbm")

    p = {**_DEFAULT_LGB_PARAMS, **(params or {})}

    # Align
    common = features.index.intersection(returns.index)
    X = features.loc[common].values.astype(np.float32)
    y = returns.loc[common].values.astype(np.float32)

    # Remove NaN rows
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[mask], y[mask]

    # Train / val split (time-series order)
    split = int(len(X) * (1 - val_ratio))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    n_est = p.pop("n_estimators", 500)
    early = p.pop("early_stopping_rounds", 50)

    model = lgb.LGBMRegressor(n_estimators=n_est, **p)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(early, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def train_xgboost_model(
    features: pd.DataFrame,
    returns: pd.Series,
    params: Optional[dict] = None,
    val_ratio: float = 0.15,
) -> Any:
    """训练 XGBoost 回归模型。

    Parameters
    ----------
    features : pd.DataFrame
        特征矩阵
    returns : pd.Series
        目标变量
    params : dict, optional
        XGBoost 超参数
    val_ratio : float
        验证集比例

    Returns
    -------
    xgb.XGBRegressor
    """
    if not _HAS_XGB:
        raise ImportError("xgboost 未安装。请运行: pip install xgboost")

    p = {**_DEFAULT_XGB_PARAMS, **(params or {})}

    common = features.index.intersection(returns.index)
    X = features.loc[common].values.astype(np.float32)
    y = returns.loc[common].values.astype(np.float32)

    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[mask], y[mask]

    split = int(len(X) * (1 - val_ratio))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    n_est = p.pop("n_estimators", 500)
    early = p.pop("early_stopping_rounds", 50)

    model = xgb.XGBRegressor(n_estimators=n_est, **p)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def predict_alpha(
    model: Any,
    features: pd.DataFrame,
) -> pd.Series:
    """使用训练好的模型预测 alpha。

    Parameters
    ----------
    model : LGBMRegressor | XGBRegressor
        训练好的树模型
    features : pd.DataFrame
        特征矩阵

    Returns
    -------
    pd.Series
        预测 alpha, index 与 features 一致
    """
    X = features.values.astype(np.float32)
    # Fill NaN with 0 for prediction (model can handle it via missing)
    X = np.nan_to_num(X, nan=0.0)
    preds = model.predict(X)
    return pd.Series(preds, index=features.index, name="pred_alpha")


def feature_importance(
    model: Any,
    feature_names: list = None,
    importance_type: str = "gain",
) -> pd.Series:
    """提取特征重要性。

    Parameters
    ----------
    model : LGBMRegressor | XGBRegressor
    feature_names : list, optional
        特征名称列表
    importance_type : str
        'gain', 'split' (LGB) 或 'weight', 'gain', 'cover' (XGB)

    Returns
    -------
    pd.Series
        特征重要性, 降序排列
    """
    if _HAS_LGB and isinstance(model, lgb.LGBMRegressor):
        imp = model.feature_importances_
    elif _HAS_XGB and isinstance(model, xgb.XGBRegressor):
        imp = model.feature_importances_
    else:
        raise ValueError("不支持的模型类型")

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(imp))]

    return pd.Series(imp, index=feature_names, name="importance").sort_values(ascending=False)


def walk_forward_ml(
    features: pd.DataFrame,
    returns: pd.Series,
    n_splits: int = 5,
    model_type: str = "lightgbm",
    params: Optional[dict] = None,
) -> dict:
    """Walk-forward ML 验证 (Qlib rolling-window style)。

    Parameters
    ----------
    features : pd.DataFrame
        特征矩阵 (MultiIndex (date, stock) 或 flat index)
    returns : pd.Series
        目标变量
    n_splits : int
        折数 (时间序列交叉验证)
    model_type : str
        'lightgbm' 或 'xgboost'
    params : dict, optional
        模型超参数

    Returns
    -------
    dict
        {
            'ic_per_fold': [float, ...],
            'mse_per_fold': [float, ...],
            'ic_mean': float,
            'ic_std': float,
            'mse_mean': float,
            'predictions': pd.Series,  # OOS 预测拼接
        }
    """
    if not _HAS_SKLEARN:
        raise ImportError("scikit-learn 未安装。请运行: pip install scikit-learn")

    common = features.index.intersection(returns.index)
    X_all = features.loc[common]
    y_all = returns.loc[common]

    # Remove NaN rows
    mask = ~(X_all.isna().any(axis=1) | y_all.isna())
    X_all = X_all[mask]
    y_all = y_all[mask]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    ic_list = []
    mse_list = []
    all_preds = []

    train_fn = (
        train_lightgbm_model if model_type == "lightgbm" else train_xgboost_model
    )

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_all)):
        X_train = X_all.iloc[train_idx]
        y_train = y_all.iloc[train_idx]
        X_test = X_all.iloc[test_idx]
        y_test = y_all.iloc[test_idx]

        model = train_fn(X_train, y_train, params=params, val_ratio=0.1)
        preds = predict_alpha(model, X_test)

        # IC (rank correlation)
        ic = preds.corr(y_test, method="spearman")
        mse = float(mean_squared_error(y_test, preds))

        ic_list.append(float(ic))
        mse_list.append(mse)
        all_preds.append(preds)

    oos_preds = pd.concat(all_preds)

    return {
        "ic_per_fold": ic_list,
        "mse_per_fold": mse_list,
        "ic_mean": float(np.mean(ic_list)),
        "ic_std": float(np.std(ic_list)),
        "mse_mean": float(np.mean(mse_list)),
        "predictions": oos_preds,
    }
