"""策略表现异常检测模块

提供基于统计方法和机器学习的策略运行时异常检测，
可与 RiskManager 和 RiskAnalyzer 配合使用，实现更细粒度的策略健康监控。

检测方法:
    - Z-score 异常检测（滚动均值/标准差）
    - Isolation Forest 多维异常检测
    - 回撤阈值告警
    - 市场状态突变检测（均值突变 CUSUM）

依赖: numpy, pandas, scikit-learn (已在项目中安装)
"""

import datetime
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 告警严重级别
# ---------------------------------------------------------------------------
WARNING = "warning"
CRITICAL = "critical"

# 监控指标名称常量
METRIC_RETURN = "return"              # 收益率
METRIC_VOLATILITY = "volatility"      # 波动率
METRIC_MAX_DRAWDOWN = "max_drawdown"  # 最大回撤
METRIC_TURNOVER = "turnover"          # 换手率
METRIC_FACTOR_EXPOSURE = "factor_exposure"  # 因子暴露变化


class StrategyAnomalyDetector:
    """策略表现异常检测器

    实时接收策略运行指标，通过多种检测方法识别异常行为并生成告警。

    Parameters
    ----------
    window : int
        滚动窗口大小（数据点数），用于 Z-score 计算，默认 60
    z_threshold : float
        Z-score 异常阈值，默认 2.5（超过则 warning），3.5 则 critical
    drawdown_warning : float
        回撤 warning 阈值（正值，如 0.08 表示 8%），默认 0.08
    drawdown_critical : float
        回撤 critical 阈值，默认 0.15
    isolation_contamination : float
        IsolationForest 的 contamination 参数，默认 0.05
    cusum_threshold : float
        CUSUM 检测的阈值系数（相对标准差的倍数），默认 4.0
    max_history : int
        保留的最大历史数据点数，默认 500

    Examples
    --------
    >>> detector = StrategyAnomalyDetector(window=30, z_threshold=2.5)
    >>> # 模拟每日更新
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> for i in range(100):
    ...     metrics = {
    ...         "return": np.random.normal(0.001, 0.02),
    ...         "volatility": abs(np.random.normal(0.15, 0.03)),
    ...         "max_drawdown": -abs(np.random.normal(0.03, 0.02)),
    ...         "turnover": abs(np.random.normal(0.05, 0.02)),
    ...         "factor_exposure": np.random.normal(0, 0.5),
    ...     }
    ...     ts = datetime.datetime(2025, 1, 1) + datetime.timedelta(days=i)
    ...     detector.update(ts, metrics)
    >>> # 注入一个异常点
    >>> detector.update(
    ...     datetime.datetime(2025, 4, 11),
    ...     {"return": -0.15, "volatility": 0.60, "max_drawdown": -0.20,
    ...      "turnover": 0.30, "factor_exposure": 3.0},
    ... )
    >>> alerts = detector.check_alerts()
    >>> for a in alerts:
    ...     print(f"[{a['severity']}] {a['type']}: {a['message']}")
    >>> print(f"策略健康分: {detector.get_health_score()}")
    """

    def __init__(
        self,
        window: int = 60,
        z_threshold: float = 2.5,
        drawdown_warning: float = 0.08,
        drawdown_critical: float = 0.15,
        isolation_contamination: float = 0.05,
        cusum_threshold: float = 4.0,
        max_history: int = 500,
    ):
        self.window = window
        self.z_threshold = z_threshold
        self.z_critical = z_threshold + 1.0  # critical 阈值 = z_threshold + 1
        self.drawdown_warning = drawdown_warning
        self.drawdown_critical = drawdown_critical
        self.isolation_contamination = isolation_contamination
        self.cusum_threshold = cusum_threshold
        self.max_history = max_history

        # 监控的指标名列表
        self._metric_names = [
            METRIC_RETURN, METRIC_VOLATILITY, METRIC_MAX_DRAWDOWN,
            METRIC_TURNOVER, METRIC_FACTOR_EXPOSURE,
        ]

        # 历史数据存储: {指标名: deque([值, ...])}
        self._history: Dict[str, deque] = {
            name: deque(maxlen=max_history) for name in self._metric_names
        }
        # 时间戳序列
        self._timestamps: deque = deque(maxlen=max_history)

        # CUSUM 状态
        self._cusum_pos: Dict[str, float] = {name: 0.0 for name in self._metric_names}
        self._cusum_neg: Dict[str, float] = {name: 0.0 for name in self._metric_names}

        # IsolationForest 模型（延迟拟合）
        self._iso_forest: Optional[IsolationForest] = None
        self._scaler: Optional[StandardScaler] = None
        self._iso_fitted = False
        self._iso_refit_interval = 50  # 每 50 个新数据点重新拟合
        self._updates_since_fit = 0

        # 最近一次告警结果缓存
        self._last_alerts: List[Dict[str, Any]] = []

    # ==================================================================
    # 数据更新
    # ==================================================================

    def update(self, timestamp: Any, metrics_dict: Dict[str, float]) -> None:
        """实时更新一个数据点

        Parameters
        ----------
        timestamp : datetime 或任意可序列化的时间标识
            当前数据点的时间戳
        metrics_dict : dict
            指标字典，键为指标名，值为浮点数。
            支持的键: "return", "volatility", "max_drawdown", "turnover", "factor_exposure"
            不存在的键会用 NaN 填充。
        """
        self._timestamps.append(timestamp)
        for name in self._metric_names:
            value = metrics_dict.get(name, np.nan)
            self._history[name].append(value)

        # 更新 CUSUM 状态
        self._update_cusum(metrics_dict)

        # 标记需要重新拟合 IsolationForest
        self._updates_since_fit += 1

    def _update_cusum(self, metrics_dict: Dict[str, float]) -> None:
        """更新 CUSUM 累积和（用于均值突变检测）"""
        for name in self._metric_names:
            values = list(self._history[name])
            if len(values) < self.window:
                continue

            # 用窗口内数据的均值和标准差作为参考
            window_vals = np.array(values[-self.window:], dtype=float)
            valid = window_vals[~np.isnan(window_vals)]
            if len(valid) < 10:
                continue

            mu = np.mean(valid[:-1]) if len(valid) > 1 else np.mean(valid)
            sigma = np.std(valid[:-1]) if len(valid) > 1 else np.std(valid)
            if sigma < 1e-10:
                continue

            current = metrics_dict.get(name, np.nan)
            if np.isnan(current):
                continue

            # 标准化偏差
            z = (current - mu) / sigma
            # 累积正向和负向偏差
            self._cusum_pos[name] = max(0, self._cusum_pos[name] + z - 0.5)
            self._cusum_neg[name] = max(0, self._cusum_neg[name] - z - 0.5)

    # ==================================================================
    # 检测方法
    # ==================================================================

    def z_score_detector(self) -> List[Dict[str, Any]]:
        """基于滚动均值/标准差的 Z-score 异常检测

        对每个监控指标计算当前值相对于滚动窗口的 Z-score，
        超过阈值则产生告警。

        Returns
        -------
        list of dict
            告警列表
        """
        alerts = []
        for name in self._metric_names:
            values = list(self._history[name])
            if len(values) < self.window:
                continue

            arr = np.array(values, dtype=float)
            valid_mask = ~np.isnan(arr)
            if valid_mask.sum() < self.window:
                continue

            # 滚动窗口: 不含最新值
            window_vals = arr[-self.window - 1:-1]
            window_valid = window_vals[~np.isnan(window_vals)]
            if len(window_valid) < 10:
                continue

            current = arr[-1]
            if np.isnan(current):
                continue

            mu = np.mean(window_valid)
            sigma = np.std(window_valid)
            if sigma < 1e-10:
                continue

            z = (current - mu) / sigma
            abs_z = abs(z)

            if abs_z >= self.z_critical:
                alerts.append(self._make_alert(
                    alert_type="z_score",
                    severity=CRITICAL,
                    message=(
                        f"指标 [{name}] Z-score={z:.2f} 严重异常 "
                        f"(当前={current:.4f}, 均值={mu:.4f}, 标准差={sigma:.4f})"
                    ),
                    details={"metric": name, "z_score": z, "value": current,
                             "mean": mu, "std": sigma},
                ))
            elif abs_z >= self.z_threshold:
                alerts.append(self._make_alert(
                    alert_type="z_score",
                    severity=WARNING,
                    message=(
                        f"指标 [{name}] Z-score={z:.2f} 异常偏离 "
                        f"(当前={current:.4f}, 均值={mu:.4f}, 标准差={sigma:.4f})"
                    ),
                    details={"metric": name, "z_score": z, "value": current,
                             "mean": mu, "std": sigma},
                ))
        return alerts

    def isolation_forest_detector(self) -> List[Dict[str, Any]]:
        """基于 IsolationForest 的多维异常检测

        将所有监控指标作为特征向量，训练 IsolationForest 模型，
        检测最新数据点是否为多维空间中的异常。

        Returns
        -------
        list of dict
            告警列表
        """
        alerts = []
        n_points = len(self._timestamps)
        if n_points < max(self.window, 30):
            return alerts

        # 构建特征矩阵
        X = self._build_feature_matrix()
        if X is None or len(X) < 30:
            return alerts

        # 按需重新拟合模型
        if not self._iso_fitted or self._updates_since_fit >= self._iso_refit_interval:
            self._fit_isolation_forest(X[:-1])  # 用历史数据拟合（不含最新点）

        if not self._iso_fitted:
            return alerts

        # 对最新点打分
        latest = X[-1:].copy()
        latest_scaled = self._scaler.transform(latest)
        prediction = self._iso_forest.predict(latest_scaled)
        score = self._iso_forest.decision_function(latest_scaled)[0]

        if prediction[0] == -1:
            # 异常点: score 越负越异常
            severity = CRITICAL if score < -0.3 else WARNING
            alerts.append(self._make_alert(
                alert_type="isolation_forest",
                severity=severity,
                message=(
                    f"IsolationForest 检测到多维异常 (异常分={score:.3f})，"
                    f"多个指标同时偏离正常范围"
                ),
                details={"anomaly_score": score, "prediction": int(prediction[0])},
            ))
        return alerts

    def drawdown_alert(self) -> List[Dict[str, Any]]:
        """回撤超阈值告警

        检查最新的 max_drawdown 指标是否超过预设阈值。

        Returns
        -------
        list of dict
            告警列表
        """
        alerts = []
        dd_values = list(self._history[METRIC_MAX_DRAWDOWN])
        if not dd_values:
            return alerts

        current_dd = dd_values[-1]
        if np.isnan(current_dd):
            return alerts

        # max_drawdown 为负值，取绝对值比较
        abs_dd = abs(current_dd)

        if abs_dd >= self.drawdown_critical:
            alerts.append(self._make_alert(
                alert_type="drawdown",
                severity=CRITICAL,
                message=f"回撤 {current_dd:.2%} 超过严重阈值 -{self.drawdown_critical:.0%}，建议立即减仓",
                details={"drawdown": current_dd, "threshold": -self.drawdown_critical},
            ))
        elif abs_dd >= self.drawdown_warning:
            alerts.append(self._make_alert(
                alert_type="drawdown",
                severity=WARNING,
                message=f"回撤 {current_dd:.2%} 超过预警阈值 -{self.drawdown_warning:.0%}，请关注风险",
                details={"drawdown": current_dd, "threshold": -self.drawdown_warning},
            ))
        return alerts

    def regime_change_detector(self) -> List[Dict[str, Any]]:
        """市场状态突变检测（基于 CUSUM 算法）

        CUSUM (Cumulative Sum) 通过累积偏差来检测均值突变。
        当累积和超过阈值时，认为发生了状态突变。

        相比 HMM，CUSUM 更轻量、无需额外依赖，且对在线检测更友好。

        Returns
        -------
        list of dict
            告警列表
        """
        alerts = []
        for name in self._metric_names:
            values = list(self._history[name])
            if len(values) < self.window:
                continue

            arr = np.array(values[-self.window:], dtype=float)
            valid = arr[~np.isnan(arr)]
            if len(valid) < 10:
                continue

            sigma = np.std(valid)
            if sigma < 1e-10:
                continue

            threshold = self.cusum_threshold * sigma
            cusum_p = self._cusum_pos[name]
            cusum_n = self._cusum_neg[name]

            if cusum_p > threshold or cusum_n > threshold:
                direction = "上升" if cusum_p > cusum_n else "下降"
                cusum_val = max(cusum_p, cusum_n)
                severity = CRITICAL if cusum_val > threshold * 1.5 else WARNING

                alerts.append(self._make_alert(
                    alert_type="regime_change",
                    severity=severity,
                    message=(
                        f"指标 [{name}] 检测到均值{direction}突变 "
                        f"(CUSUM={cusum_val:.2f}, 阈值={threshold:.2f})"
                    ),
                    details={"metric": name, "cusum_pos": cusum_p,
                             "cusum_neg": cusum_n, "threshold": threshold,
                             "direction": direction},
                ))

                # 重置 CUSUM（突变已检测，避免持续告警）
                self._cusum_pos[name] = 0.0
                self._cusum_neg[name] = 0.0

        return alerts

    # ==================================================================
    # 综合接口
    # ==================================================================

    def check_alerts(self) -> List[Dict[str, Any]]:
        """运行所有检测方法，返回合并的告警列表

        Returns
        -------
        list of dict
            每个告警包含:
            - "type": str  检测方法名 ("z_score" | "isolation_forest" | "drawdown" | "regime_change")
            - "severity": str  严重级别 ("warning" | "critical")
            - "message": str  告警描述
            - "timestamp": Any  告警关联的时间戳
            - "details": dict  附加信息
        """
        alerts = []
        alerts.extend(self.z_score_detector())
        alerts.extend(self.isolation_forest_detector())
        alerts.extend(self.drawdown_alert())
        alerts.extend(self.regime_change_detector())

        # 按严重级别排序: critical 优先
        severity_order = {CRITICAL: 0, WARNING: 1}
        alerts.sort(key=lambda a: severity_order.get(a["severity"], 99))

        self._last_alerts = alerts
        return alerts

    def get_health_score(self) -> float:
        """计算策略健康分 (0-100)

        综合多个维度评估策略当前健康状态:
        - Z-score 偏离程度 (权重 30%)
        - 回撤状态 (权重 30%)
        - IsolationForest 异常分 (权重 20%)
        - CUSUM 突变程度 (权重 20%)

        Returns
        -------
        float
            健康分，100 为完全健康，0 为极度异常
        """
        scores = []
        weights = []

        # --- 1. Z-score 维度 (权重 30%) ---
        z_score_penalty = self._calc_z_score_penalty()
        scores.append(max(0, 100 - z_score_penalty))
        weights.append(0.30)

        # --- 2. 回撤维度 (权重 30%) ---
        dd_score = self._calc_drawdown_score()
        scores.append(dd_score)
        weights.append(0.30)

        # --- 3. IsolationForest 维度 (权重 20%) ---
        iso_score = self._calc_isolation_score()
        scores.append(iso_score)
        weights.append(0.20)

        # --- 4. CUSUM 维度 (权重 20%) ---
        cusum_score = self._calc_cusum_score()
        scores.append(cusum_score)
        weights.append(0.20)

        # 加权平均
        if not scores:
            return 100.0

        total = sum(s * w for s, w in zip(scores, weights))
        total_w = sum(weights)
        health = total / total_w if total_w > 0 else 100.0

        return round(float(np.clip(health, 0, 100)), 1)

    # ==================================================================
    # 内部辅助方法
    # ==================================================================

    def _make_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """构造标准化告警字典"""
        ts = self._timestamps[-1] if self._timestamps else None
        return {
            "type": alert_type,
            "severity": severity,
            "message": message,
            "timestamp": ts,
            "details": details or {},
        }

    def _build_feature_matrix(self) -> Optional[np.ndarray]:
        """将历史数据构建为特征矩阵 (n_samples, n_features)"""
        n = len(self._timestamps)
        if n == 0:
            return None

        cols = []
        for name in self._metric_names:
            vals = list(self._history[name])
            # 补齐长度
            padded = [np.nan] * (n - len(vals)) + list(vals)
            cols.append(padded)

        X = np.column_stack(cols)

        # 去掉含 NaN 的行（但保留最后一行即使有 NaN）
        valid_mask = ~np.any(np.isnan(X[:-1]), axis=1)
        valid_rows = X[:-1][valid_mask]
        last_row = X[-1:]

        # 如果最新行有 NaN，用列均值填充
        if np.any(np.isnan(last_row)):
            col_means = np.nanmean(X, axis=0)
            for j in range(last_row.shape[1]):
                if np.isnan(last_row[0, j]):
                    last_row[0, j] = col_means[j] if not np.isnan(col_means[j]) else 0.0

        if len(valid_rows) < 20:
            return None

        return np.vstack([valid_rows, last_row])

    def _fit_isolation_forest(self, X: np.ndarray) -> None:
        """拟合 IsolationForest 模型"""
        if len(X) < 20:
            return

        try:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)

            self._iso_forest = IsolationForest(
                contamination=self.isolation_contamination,
                n_estimators=100,
                max_samples="auto",
                random_state=42,
            )
            self._iso_forest.fit(X_scaled)
            self._iso_fitted = True
            self._updates_since_fit = 0
        except Exception:
            # 拟合失败时静默处理，不影响其他检测
            self._iso_fitted = False

    def _calc_z_score_penalty(self) -> float:
        """计算 Z-score 维度的扣分（0=无异常, 100=极度异常）"""
        max_abs_z = 0.0
        for name in self._metric_names:
            values = list(self._history[name])
            if len(values) < self.window:
                continue

            arr = np.array(values, dtype=float)
            window_vals = arr[-self.window - 1:-1]
            window_valid = window_vals[~np.isnan(window_vals)]
            if len(window_valid) < 10:
                continue

            current = arr[-1]
            if np.isnan(current):
                continue

            mu = np.mean(window_valid)
            sigma = np.std(window_valid)
            if sigma < 1e-10:
                continue

            abs_z = abs((current - mu) / sigma)
            max_abs_z = max(max_abs_z, abs_z)

        # 映射: z=0 -> 0分, z=z_threshold -> 30分, z=z_critical -> 60分, z=5+ -> 100分
        if max_abs_z <= 1.0:
            return 0.0
        elif max_abs_z <= self.z_threshold:
            return (max_abs_z - 1.0) / (self.z_threshold - 1.0) * 30
        elif max_abs_z <= self.z_critical:
            return 30 + (max_abs_z - self.z_threshold) / (self.z_critical - self.z_threshold) * 30
        else:
            return min(100, 60 + (max_abs_z - self.z_critical) * 20)

    def _calc_drawdown_score(self) -> float:
        """计算回撤维度的健康分 (0-100)"""
        dd_values = list(self._history[METRIC_MAX_DRAWDOWN])
        if not dd_values:
            return 100.0

        current_dd = dd_values[-1]
        if np.isnan(current_dd):
            return 100.0

        abs_dd = abs(current_dd)
        if abs_dd <= 0.02:
            return 100.0
        elif abs_dd <= self.drawdown_warning:
            # 线性衰减: 2% -> 100, warning -> 60
            return 100 - (abs_dd - 0.02) / (self.drawdown_warning - 0.02) * 40
        elif abs_dd <= self.drawdown_critical:
            # 60 -> 20
            return 60 - (abs_dd - self.drawdown_warning) / (self.drawdown_critical - self.drawdown_warning) * 40
        else:
            # 超过 critical: 快速降到 0
            return max(0, 20 - (abs_dd - self.drawdown_critical) * 200)

    def _calc_isolation_score(self) -> float:
        """计算 IsolationForest 维度的健康分 (0-100)"""
        if not self._iso_fitted:
            return 100.0  # 模型未拟合，默认健康

        X = self._build_feature_matrix()
        if X is None or len(X) < 2:
            return 100.0

        try:
            latest = X[-1:].copy()
            latest_scaled = self._scaler.transform(latest)
            score = self._iso_forest.decision_function(latest_scaled)[0]
            # score > 0 表示正常, < 0 表示异常
            # 映射: score=0.1+ -> 100, score=0 -> 70, score=-0.3 -> 0
            if score >= 0.1:
                return 100.0
            elif score >= 0:
                return 70 + score / 0.1 * 30
            elif score >= -0.3:
                return max(0, 70 + score / 0.3 * 70)
            else:
                return 0.0
        except Exception:
            return 100.0

    def _calc_cusum_score(self) -> float:
        """计算 CUSUM 维度的健康分 (0-100)"""
        max_ratio = 0.0
        for name in self._metric_names:
            values = list(self._history[name])
            if len(values) < self.window:
                continue

            arr = np.array(values[-self.window:], dtype=float)
            valid = arr[~np.isnan(arr)]
            if len(valid) < 10:
                continue

            sigma = np.std(valid)
            if sigma < 1e-10:
                continue

            threshold = self.cusum_threshold * sigma
            cusum_max = max(self._cusum_pos[name], self._cusum_neg[name])
            ratio = cusum_max / threshold if threshold > 0 else 0.0
            max_ratio = max(max_ratio, ratio)

        # 映射: ratio=0 -> 100, ratio=0.5 -> 80, ratio=1.0 -> 40, ratio=1.5+ -> 0
        if max_ratio <= 0.5:
            return 100 - max_ratio * 40
        elif max_ratio <= 1.0:
            return 80 - (max_ratio - 0.5) * 80
        else:
            return max(0, 40 - (max_ratio - 1.0) * 80)

    # ==================================================================
    # 实用方法
    # ==================================================================

    def get_metric_summary(self) -> Dict[str, Dict[str, float]]:
        """获取各指标的统计摘要

        Returns
        -------
        dict
            {指标名: {"current": ..., "mean": ..., "std": ..., "z_score": ..., "min": ..., "max": ...}}
        """
        summary = {}
        for name in self._metric_names:
            values = list(self._history[name])
            if not values:
                continue
            arr = np.array(values, dtype=float)
            valid = arr[~np.isnan(arr)]
            if len(valid) == 0:
                continue

            current = valid[-1]
            mu = np.mean(valid)
            sigma = np.std(valid)
            z = (current - mu) / sigma if sigma > 1e-10 else 0.0

            summary[name] = {
                "current": float(current),
                "mean": float(mu),
                "std": float(sigma),
                "z_score": float(z),
                "min": float(np.min(valid)),
                "max": float(np.max(valid)),
                "n_points": len(valid),
            }
        return summary

    def reset(self) -> None:
        """重置检测器状态"""
        for name in self._metric_names:
            self._history[name].clear()
            self._cusum_pos[name] = 0.0
            self._cusum_neg[name] = 0.0
        self._timestamps.clear()
        self._iso_fitted = False
        self._iso_forest = None
        self._scaler = None
        self._updates_since_fit = 0
        self._last_alerts = []


# ======================================================================
# 使用示例
# ======================================================================
if __name__ == "__main__":
    import datetime

    print("=" * 60)
    print("  策略异常检测器 - 使用示例")
    print("=" * 60)

    # 创建检测器
    detector = StrategyAnomalyDetector(
        window=30,
        z_threshold=2.5,
        drawdown_warning=0.08,
        drawdown_critical=0.15,
    )

    np.random.seed(42)

    # --- 第一阶段: 喂入 80 天正常数据 ---
    print("\n[阶段1] 输入 80 天正常交易数据...")
    for i in range(80):
        metrics = {
            "return": np.random.normal(0.001, 0.015),
            "volatility": abs(np.random.normal(0.15, 0.02)),
            "max_drawdown": -abs(np.random.normal(0.03, 0.01)),
            "turnover": abs(np.random.normal(0.05, 0.01)),
            "factor_exposure": np.random.normal(0, 0.3),
        }
        ts = datetime.datetime(2025, 1, 1) + datetime.timedelta(days=i)
        detector.update(ts, metrics)

    alerts = detector.check_alerts()
    health = detector.get_health_score()
    print(f"  告警数: {len(alerts)}")
    print(f"  健康分: {health}")

    # --- 第二阶段: 注入异常数据（模拟策略崩溃） ---
    print("\n[阶段2] 注入异常数据（模拟策略崩溃）...")
    anomaly_metrics = {
        "return": -0.12,          # 单日巨亏
        "volatility": 0.55,       # 波动率飙升
        "max_drawdown": -0.18,    # 严重回撤
        "turnover": 0.35,         # 换手率异常高
        "factor_exposure": 3.5,   # 因子暴露突变
    }
    ts = datetime.datetime(2025, 3, 22)
    detector.update(ts, anomaly_metrics)

    alerts = detector.check_alerts()
    health = detector.get_health_score()
    print(f"  告警数: {len(alerts)}")
    for a in alerts:
        print(f"  [{a['severity'].upper():8s}] {a['type']:20s} | {a['message']}")
    print(f"  健康分: {health}")

    # --- 第三阶段: 查看指标摘要 ---
    print("\n[阶段3] 指标统计摘要:")
    summary = detector.get_metric_summary()
    for name, stats in summary.items():
        print(f"  {name:20s} | 当前={stats['current']:+.4f}  "
              f"均值={stats['mean']:+.4f}  Z={stats['z_score']:+.2f}  "
              f"N={stats['n_points']}")

    print("\n" + "=" * 60)
    print("  示例完成")
    print("=" * 60)
