"""Qlib-style ML model adapter
Microsoft Qlib: https://github.com/microsoft/qlib
Wraps LightGBM / XGBoost tree model training and prediction, with walk-forward validation support.
Does not depend on Qlib directly; it reproduces the core Qlib ML pipeline pattern.
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
    """Train a LightGBM regression model (Qlib LGBModel style).

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix (n_samples x n_features)
    returns : pd.Series
        Target variable (next-period return)
    params : dict, optional
        LightGBM hyperparameters; defaults are used when None
    val_ratio : float
        Validation set fraction (split from the tail, preserving time-series order)

    Returns
    -------
    lgb.LGBMRegressor
        The trained model
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
    """Train an XGBoost regression model.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix
    returns : pd.Series
        Target variable
    params : dict, optional
        XGBoost hyperparameters
    val_ratio : float
        Validation set fraction

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
    """Predict alpha with a trained model.

    Parameters
    ----------
    model : LGBMRegressor | XGBRegressor
        A trained tree model
    features : pd.DataFrame
        Feature matrix

    Returns
    -------
    pd.Series
        Predicted alpha, indexed the same as features
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
    """Extract feature importances.

    Parameters
    ----------
    model : LGBMRegressor | XGBRegressor
    feature_names : list, optional
        List of feature names
    importance_type : str
        'gain', 'split' (LGB) or 'weight', 'gain', 'cover' (XGB)

    Returns
    -------
    pd.Series
        Feature importances, sorted descending
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
    """Walk-forward ML validation (Qlib rolling-window style).

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix (MultiIndex (date, stock) or flat index)
    returns : pd.Series
        Target variable
    n_splits : int
        Number of folds (time-series cross-validation)
    model_type : str
        'lightgbm' or 'xgboost'
    params : dict, optional
        Model hyperparameters

    Returns
    -------
    dict
        {
            'ic_per_fold': [float, ...],
            'mse_per_fold': [float, ...],
            'ic_mean': float,
            'ic_std': float,
            'mse_mean': float,
            'predictions': pd.Series,  # concatenated OOS predictions
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
