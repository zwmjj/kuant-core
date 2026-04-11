"""实时风险管理系统 — 多资产组合的全面风控"""
import os
import csv
import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# 预定义压力情景
# ---------------------------------------------------------------------------
STRESS_SCENARIOS: Dict[str, Dict[str, float]] = {
    "COVID_2020": {
        "description": "2020年3月新冠暴跌",
        "SPY": -0.34,
        "QQQ": -0.28,
        "TLT": 0.15,
        "VIX": 4.00,
        "GLD": 0.03,
        "BTC": -0.40,
        "duration_days": 23,
    },
    "BEAR_2022": {
        "description": "2022年熊市(美联储加息)",
        "SPY": -0.25,
        "QQQ": -0.33,
        "TLT": -0.30,
        "VIX": 1.50,
        "GLD": -0.03,
        "BTC": -0.65,
        "duration_days": 270,
    },
    "FLASH_CRASH": {
        "description": "闪崩 — 单日暴跌",
        "SPY": -0.10,
        "QQQ": -0.12,
        "TLT": 0.05,
        "VIX": 3.00,
        "GLD": 0.02,
        "BTC": -0.15,
        "duration_days": 1,
    },
    "BOND_CRASH": {
        "description": "债券崩盘",
        "SPY": -0.08,
        "QQQ": -0.10,
        "TLT": -0.20,
        "VIX": 1.00,
        "GLD": 0.05,
        "BTC": -0.05,
        "duration_days": 60,
    },
    "CRYPTO_CRASH": {
        "description": "加密货币崩盘",
        "SPY": -0.03,
        "QQQ": -0.05,
        "TLT": 0.02,
        "VIX": 0.30,
        "GLD": 0.01,
        "BTC": -0.50,
        "duration_days": 14,
    },
}

# 风险级别常量
GREEN = "GREEN"
YELLOW = "YELLOW"
ORANGE = "ORANGE"
RED = "RED"


# ---------------------------------------------------------------------------
# 希腊字母默认限制
# ---------------------------------------------------------------------------
_DEFAULT_GREEK_LIMITS = {
    "delta": (-0.30, 0.30),       # 组合 delta 范围
    "gamma": (-0.05, 0.05),       # 组合 gamma 范围
    "theta": (-500.0, float("inf")),  # 每日 theta 下限
    "vega":  (-1000.0, 1000.0),   # vega 范围
}


class RiskManager:
    """
    实时风险管理器

    集成持仓级、组合级、动态限额、期权希腊字母、告警与执行动作。
    可与 AutoTrader 对接: ``evaluate()`` 返回的动作字典可直接执行。

    Parameters
    ----------
    risk_log_path : str
        风险日志 CSV 路径，默认 ``risk_log.csv``
    greek_limits : dict, optional
        期权希腊字母限额覆盖
    """

    def __init__(
        self,
        risk_log_path: str = "risk_log.csv",
        greek_limits: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.risk_log_path = risk_log_path
        self.greek_limits = {**_DEFAULT_GREEK_LIMITS, **(greek_limits or {})}
        self._ensure_log_header()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _ensure_log_header(self):
        """确保日志文件存在且有表头"""
        if not os.path.exists(self.risk_log_path):
            with open(self.risk_log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "level", "message", "action",
                    "portfolio_var", "portfolio_cvar", "max_dd",
                ])

    def _log(self, level: str, message: str, action: str,
             portfolio_var: float = 0.0, portfolio_cvar: float = 0.0,
             max_dd: float = 0.0):
        """写一行到 risk_log.csv"""
        with open(self.risk_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.datetime.now().isoformat(),
                level, message, action,
                f"{portfolio_var:.6f}", f"{portfolio_cvar:.6f}", f"{max_dd:.4f}",
            ])

    # ==================================================================
    # 1. 持仓级风险
    # ==================================================================

    def check_position_size(
        self,
        positions: Dict[str, float],
        equity: float,
    ) -> List[Tuple[str, str, float]]:
        """
        检查单一持仓是否超过组合净值的 10%

        Parameters
        ----------
        positions : dict
            {ticker: market_value, ...}
        equity : float
            组合净值

        Returns
        -------
        list of (ticker, level, weight)
            超限持仓列表
        """
        alerts: List[Tuple[str, str, float]] = []
        if equity <= 0:
            return alerts
        for ticker, mv in positions.items():
            weight = abs(mv) / equity
            if weight > 0.10:
                alerts.append((ticker, RED if weight > 0.20 else ORANGE, weight))
            elif weight > 0.08:
                alerts.append((ticker, YELLOW, weight))
        return alerts

    def check_sector_concentration(
        self,
        positions: Dict[str, Dict[str, Any]],
    ) -> List[Tuple[str, str, float]]:
        """
        检查行业集中度，单一行业不超过 30%

        Parameters
        ----------
        positions : dict
            {ticker: {"market_value": float, "sector": str}, ...}

        Returns
        -------
        list of (sector, level, weight)
        """
        total = sum(abs(p["market_value"]) for p in positions.values())
        if total <= 0:
            return []
        sector_expo: Dict[str, float] = {}
        for info in positions.values():
            sec = info.get("sector", "Unknown")
            sector_expo[sec] = sector_expo.get(sec, 0.0) + abs(info["market_value"])

        alerts: List[Tuple[str, str, float]] = []
        for sec, mv in sector_expo.items():
            w = mv / total
            if w > 0.30:
                alerts.append((sec, ORANGE, w))
            elif w > 0.25:
                alerts.append((sec, YELLOW, w))
        return alerts

    def check_correlation_risk(
        self,
        positions: Dict[str, float],
        returns: pd.DataFrame,
        threshold: float = 0.70,
        top_n: int = 10,
    ) -> List[Tuple[str, str, float]]:
        """
        检查头部持仓间相关性，高度相关(>0.7)则告警

        Parameters
        ----------
        positions : dict
            {ticker: market_value, ...}
        returns : DataFrame
            历史收益率, 列为 ticker
        threshold : float
            相关性告警阈值
        top_n : int
            只检查最大的 N 个持仓

        Returns
        -------
        list of (pair_str, level, corr)
        """
        sorted_pos = sorted(positions.items(), key=lambda x: abs(x[1]), reverse=True)
        tickers = [t for t, _ in sorted_pos[:top_n]]
        available = [t for t in tickers if t in returns.columns]
        if len(available) < 2:
            return []

        corr_matrix = returns[available].dropna().corr()
        alerts: List[Tuple[str, str, float]] = []
        seen = set()
        for i, t1 in enumerate(available):
            for j, t2 in enumerate(available):
                if i >= j:
                    continue
                pair = (t1, t2)
                if pair in seen:
                    continue
                seen.add(pair)
                c = corr_matrix.loc[t1, t2]
                if abs(c) > threshold:
                    lvl = RED if abs(c) > 0.90 else (ORANGE if abs(c) > 0.80 else YELLOW)
                    alerts.append((f"{t1}/{t2}", lvl, c))
        return alerts

    def estimate_position_var(
        self,
        position_value: float,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """
        估算单一持仓的 VaR（历史法）

        Parameters
        ----------
        position_value : float
            持仓市值
        returns : Series
            该资产的历史日收益率
        confidence : float
            置信度

        Returns
        -------
        float
            VaR 金额（负值表示损失）
        """
        if len(returns) < 20:
            return 0.0
        var_pct = float(returns.quantile(1 - confidence))
        return position_value * var_pct

    # ==================================================================
    # 2. 组合级风险
    # ==================================================================

    def portfolio_var(
        self,
        portfolio_returns: pd.Series,
        confidence: float = 0.99,
    ) -> float:
        """
        组合历史 VaR

        Parameters
        ----------
        portfolio_returns : Series
            组合日收益率
        confidence : float
            置信度

        Returns
        -------
        float
            VaR（负值）
        """
        if len(portfolio_returns) < 20:
            return 0.0
        return float(portfolio_returns.quantile(1 - confidence))

    def portfolio_cvar(
        self,
        portfolio_returns: pd.Series,
        confidence: float = 0.99,
    ) -> float:
        """
        组合条件 VaR（预期尾部损失）

        Parameters
        ----------
        portfolio_returns : Series
            组合日收益率
        confidence : float
            置信度

        Returns
        -------
        float
            CVaR（负值）
        """
        var = self.portfolio_var(portfolio_returns, confidence)
        tail = portfolio_returns[portfolio_returns <= var]
        return float(tail.mean()) if len(tail) > 0 else var

    def max_drawdown_check(
        self,
        equity_curve: pd.Series,
        threshold: float = 0.08,
    ) -> Tuple[float, float]:
        """
        回撤检查，返回缩放因子

        - DD <= threshold   → scale = 1.0（正常）
        - DD > threshold    → scale = 0.5（减仓）
        - DD > 15%          → scale = 0.2（大幅减仓）

        Parameters
        ----------
        equity_curve : Series
            净值曲线
        threshold : float
            开始减仓的回撤阈值

        Returns
        -------
        (current_dd, scale_factor)
        """
        if len(equity_curve) < 2:
            return 0.0, 1.0
        peak = equity_curve.cummax()
        dd = (equity_curve - peak) / peak
        current_dd = float(dd.iloc[-1])

        if current_dd < -0.15:
            scale = 0.2
        elif current_dd < -threshold:
            scale = 0.5
        else:
            scale = 1.0
        return current_dd, scale

    def stress_test(
        self,
        portfolio_returns: pd.Series,
        scenarios: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> pd.DataFrame:
        """
        针对预定义情景的压力测试

        通过对历史收益分布施加情景冲击来估算组合损失。

        Parameters
        ----------
        portfolio_returns : Series
            组合日收益率
        scenarios : dict, optional
            情景字典，默认使用 STRESS_SCENARIOS

        Returns
        -------
        DataFrame
            每个情景的预计损失
        """
        if scenarios is None:
            scenarios = STRESS_SCENARIOS

        results = []
        daily_vol = float(portfolio_returns.std()) if len(portfolio_returns) > 1 else 0.01
        daily_mean = float(portfolio_returns.mean()) if len(portfolio_returns) > 0 else 0.0
        # 计算组合与 SPY 的 beta
        # 简单近似: 用历史波动率和情景中 SPY 的冲击来估计损失
        for name, params in scenarios.items():
            spy_shock = params.get("SPY", -0.10)
            duration = params.get("duration_days", 1)
            description = params.get("description", name)
            # 假设 beta ≈ 1，按波动率调整
            vol_ratio = daily_vol / 0.01 if 0.01 > 0 else 1.0
            estimated_loss = spy_shock * min(vol_ratio, 2.0)
            # 回撤期间累积
            if duration > 1:
                # 分散到多日, 线性近似
                daily_shock = estimated_loss / max(duration, 1)
                worst_day = daily_shock * 3  # 最坏单日
            else:
                worst_day = estimated_loss

            results.append({
                "scenario": name,
                "description": description,
                "spy_shock": spy_shock,
                "estimated_loss": estimated_loss,
                "worst_day": worst_day,
                "duration_days": duration,
            })
        return pd.DataFrame(results)

    # ==================================================================
    # 3. 动态风险限额
    # ==================================================================

    def vol_target_scale(
        self,
        portfolio_returns: pd.Series,
        target: float = 0.10,
        lookback: int = 20,
    ) -> float:
        """
        波动率目标缩放 — 维持目标年化波动率

        Parameters
        ----------
        portfolio_returns : Series
            组合日收益率
        target : float
            目标年化波动率
        lookback : int
            回看窗口（交易日）

        Returns
        -------
        float
            仓位缩放系数（限制在 [0.2, 2.0]）
        """
        if len(portfolio_returns) < lookback:
            return 1.0
        recent = portfolio_returns.iloc[-lookback:]
        realized_vol = float(recent.std()) * np.sqrt(252)
        if realized_vol < 1e-8:
            return 1.0
        scale = target / realized_vol
        return float(np.clip(scale, 0.2, 2.0))

    def kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Kelly 准则仓位比例

        f* = (p * b - q) / b
        其中 p = 胜率, q = 败率, b = 盈亏比

        实际使用半 Kelly (f*/2) 以降低波动。

        Parameters
        ----------
        win_rate : float
            胜率 (0~1)
        avg_win : float
            平均盈利（正值）
        avg_loss : float
            平均亏损（正值）

        Returns
        -------
        float
            建议仓位比例 (半 Kelly)，限制在 [0, 0.25]
        """
        if avg_loss <= 0 or avg_win <= 0 or not (0 < win_rate < 1):
            return 0.0
        b = avg_win / avg_loss
        q = 1.0 - win_rate
        kelly = (win_rate * b - q) / b
        half_kelly = kelly / 2.0
        return float(np.clip(half_kelly, 0.0, 0.25))

    def risk_budget(
        self,
        sub_strategy_returns: pd.DataFrame,
        total_risk: float = 0.10,
    ) -> Dict[str, float]:
        """
        风险预算 — 按等风险贡献分配

        Parameters
        ----------
        sub_strategy_returns : DataFrame
            各子策略的日收益率，列为策略名
        total_risk : float
            总风险预算（年化波动率）

        Returns
        -------
        dict
            {strategy_name: allocation_weight}
        """
        if sub_strategy_returns.empty or len(sub_strategy_returns) < 20:
            n = max(sub_strategy_returns.shape[1], 1)
            return {c: 1.0 / n for c in sub_strategy_returns.columns}

        vols = sub_strategy_returns.std() * np.sqrt(252)
        vols = vols.replace(0, np.nan).fillna(vols.mean())
        if (vols <= 0).all():
            n = len(vols)
            return {c: 1.0 / n for c in vols.index}

        # 等风险贡献: 权重 ∝ 1/vol
        inv_vol = 1.0 / vols
        raw_weights = inv_vol / inv_vol.sum()

        # 缩放到目标总风险
        port_vol = float(np.sqrt(
            raw_weights.values @ sub_strategy_returns.cov().values * 252
            @ raw_weights.values
        ))
        if port_vol > 0:
            scale = total_risk / port_vol
            raw_weights *= min(scale, 2.0)

        return {str(c): float(w) for c, w in raw_weights.items()}

    # ==================================================================
    # 4. 期权特有风险
    # ==================================================================

    def check_greeks(
        self,
        positions: List[Dict[str, float]],
    ) -> Tuple[Dict[str, float], List[Tuple[str, str, str]]]:
        """
        检查期权组合希腊字母限额

        Parameters
        ----------
        positions : list of dict
            每个字典包含 {"ticker", "delta", "gamma", "theta", "vega", "quantity"}

        Returns
        -------
        (portfolio_greeks, alerts)
            portfolio_greeks: {"delta": x, "gamma": x, "theta": x, "vega": x}
            alerts: list of (greek, level, message)
        """
        totals = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
        for pos in positions:
            qty = pos.get("quantity", 1)
            for g in totals:
                totals[g] += pos.get(g, 0.0) * qty

        alerts: List[Tuple[str, str, str]] = []
        for greek, value in totals.items():
            lo, hi = self.greek_limits.get(greek, (-float("inf"), float("inf")))
            if value < lo:
                alerts.append((
                    greek, RED if value < lo * 2 else ORANGE,
                    f"组合{greek}={value:.4f} 低于下限 {lo:.4f}"
                ))
            elif value > hi:
                alerts.append((
                    greek, RED if value > hi * 2 else ORANGE,
                    f"组合{greek}={value:.4f} 超过上限 {hi:.4f}"
                ))
        return totals, alerts

    def max_loss_scenario(
        self,
        option_positions: List[Dict[str, float]],
        price_shock: float = 0.10,
    ) -> Dict[str, float]:
        """
        极端行情下的期权组合最大损失估算

        使用 delta-gamma 近似: ΔP ≈ Δ·S·shock + 0.5·Γ·S²·shock²

        Parameters
        ----------
        option_positions : list of dict
            每个字典包含 {"ticker", "delta", "gamma", "vega", "underlying_price", "quantity"}
        price_shock : float
            标的价格冲击幅度（如 0.10 = 下跌10%）

        Returns
        -------
        dict
            {"total_pnl", "delta_pnl", "gamma_pnl", "vega_pnl", "per_position": [...]}
        """
        total_delta_pnl = 0.0
        total_gamma_pnl = 0.0
        total_vega_pnl = 0.0
        per_pos = []

        # 假设大跌时 VIX 飙升，IV 上涨约 shock * 100 个 vol points
        iv_change = price_shock * 50  # 简化：下跌10%，IV升5个点

        for pos in option_positions:
            qty = pos.get("quantity", 1)
            S = pos.get("underlying_price", 100.0)
            delta = pos.get("delta", 0.0) * qty
            gamma = pos.get("gamma", 0.0) * qty
            vega = pos.get("vega", 0.0) * qty

            dS = -S * price_shock  # 下跌
            delta_pnl = delta * dS
            gamma_pnl = 0.5 * gamma * dS ** 2
            vega_pnl = vega * iv_change

            total_delta_pnl += delta_pnl
            total_gamma_pnl += gamma_pnl
            total_vega_pnl += vega_pnl

            per_pos.append({
                "ticker": pos.get("ticker", "?"),
                "delta_pnl": delta_pnl,
                "gamma_pnl": gamma_pnl,
                "vega_pnl": vega_pnl,
                "total_pnl": delta_pnl + gamma_pnl + vega_pnl,
            })

        total = total_delta_pnl + total_gamma_pnl + total_vega_pnl
        return {
            "total_pnl": total,
            "delta_pnl": total_delta_pnl,
            "gamma_pnl": total_gamma_pnl,
            "vega_pnl": total_vega_pnl,
            "per_position": per_pos,
        }

    def margin_check(
        self,
        positions: Dict[str, Dict[str, float]],
        equity: float,
    ) -> Tuple[float, bool, str]:
        """
        保证金充足性检查

        使用简化 Reg-T 规则:
        - 股票多头: 50% 初始保证金
        - 股票空头: 50% + 额外 max(0, 空头市值-多头市值)
        - 期权卖方: 20% 标的市值

        Parameters
        ----------
        positions : dict
            {ticker: {"market_value": float, "type": "stock"|"option_long"|"option_short"}, ...}
        equity : float
            账户净值

        Returns
        -------
        (margin_used, is_ok, message)
        """
        margin = 0.0
        for ticker, info in positions.items():
            mv = abs(info.get("market_value", 0))
            ptype = info.get("type", "stock")
            if ptype == "stock":
                margin += mv * 0.50
            elif ptype == "option_long":
                margin += mv  # 全额（已付出权利金）
            elif ptype == "option_short":
                margin += mv * 0.20 + mv  # 简化
            else:
                margin += mv * 0.50

        ratio = margin / equity if equity > 0 else float("inf")
        if ratio > 0.90:
            return ratio, False, f"保证金使用率 {ratio:.1%} — 超过90%，面临追保风险"
        elif ratio > 0.70:
            return ratio, True, f"保证金使用率 {ratio:.1%} — 偏高，建议减仓"
        else:
            return ratio, True, f"保证金使用率 {ratio:.1%} — 正常"

    # ==================================================================
    # 5. 告警与执行
    # ==================================================================

    def evaluate(
        self,
        portfolio_state: Dict[str, Any],
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        综合评估组合风险，返回 (级别, 消息, 动作) 列表

        动作字典可被 AutoTrader 直接执行。

        Parameters
        ----------
        portfolio_state : dict
            必需键:
            - "equity": float
            - "positions": dict  {ticker: market_value}
            - "portfolio_returns": pd.Series
            - "equity_curve": pd.Series
            可选键:
            - "positions_detail": dict  {ticker: {"market_value", "sector"}}
            - "asset_returns": pd.DataFrame
            - "option_positions": list of dict
            - "sub_strategy_returns": pd.DataFrame

        Returns
        -------
        list of (level, message, action_dict)
            action_dict 含:
            - "type": "none" | "log" | "reduce" | "close_all"
            - "scale": float  缩放因子
            - "targets": list  受影响的标的
        """
        equity = portfolio_state.get("equity", 0)
        positions = portfolio_state.get("positions", {})
        port_ret = portfolio_state.get("portfolio_returns", pd.Series(dtype=float))
        eq_curve = portfolio_state.get("equity_curve", pd.Series(dtype=float))

        alerts: List[Tuple[str, str, Dict[str, Any]]] = []

        # --- 回撤检查 ---
        if len(eq_curve) >= 2:
            dd, scale = self.max_drawdown_check(eq_curve)
            if scale < 0.5:
                alerts.append((
                    RED, f"最大回撤 {dd:.1%}，触发清仓",
                    {"type": "close_all", "scale": 0.0, "targets": list(positions.keys())},
                ))
            elif scale < 1.0:
                alerts.append((
                    ORANGE, f"回撤 {dd:.1%}，减仓至 {scale:.0%}",
                    {"type": "reduce", "scale": scale, "targets": list(positions.keys())},
                ))

        # --- 单一持仓集中度 ---
        size_alerts = self.check_position_size(positions, equity)
        for ticker, lvl, w in size_alerts:
            action_type = "reduce" if lvl in (ORANGE, RED) else "log"
            alerts.append((
                lvl, f"持仓集中: {ticker} 占比 {w:.1%}",
                {"type": action_type, "scale": 0.70, "targets": [ticker]},
            ))

        # --- 行业集中度 ---
        pos_detail = portfolio_state.get("positions_detail", {})
        if pos_detail:
            sec_alerts = self.check_sector_concentration(pos_detail)
            for sec, lvl, w in sec_alerts:
                alerts.append((
                    lvl, f"行业集中: {sec} 占比 {w:.1%}",
                    {"type": "reduce" if lvl == ORANGE else "log", "scale": 0.70,
                     "targets": [t for t, info in pos_detail.items()
                                 if info.get("sector") == sec]},
                ))

        # --- 相关性 ---
        asset_ret = portfolio_state.get("asset_returns", pd.DataFrame())
        if not asset_ret.empty and len(positions) >= 2:
            corr_alerts = self.check_correlation_risk(positions, asset_ret)
            for pair, lvl, c in corr_alerts:
                alerts.append((
                    lvl, f"高相关性: {pair} corr={c:.2f}",
                    {"type": "log", "scale": 1.0, "targets": pair.split("/")},
                ))

        # --- VaR / CVaR ---
        if len(port_ret) >= 20:
            var99 = self.portfolio_var(port_ret, 0.99)
            cvar99 = self.portfolio_cvar(port_ret, 0.99)
            if var99 < -0.05:
                alerts.append((
                    ORANGE, f"日VaR(99%)={var99:.2%}, CVaR={cvar99:.2%}",
                    {"type": "reduce", "scale": 0.70, "targets": list(positions.keys())},
                ))
            elif var99 < -0.03:
                alerts.append((
                    YELLOW, f"日VaR(99%)={var99:.2%}, CVaR={cvar99:.2%}",
                    {"type": "log", "scale": 1.0, "targets": []},
                ))

        # --- 期权希腊字母 ---
        opt_positions = portfolio_state.get("option_positions", [])
        if opt_positions:
            _greeks, greek_alerts = self.check_greeks(opt_positions)
            for greek, lvl, msg in greek_alerts:
                alerts.append((
                    lvl, msg,
                    {"type": "reduce" if lvl == RED else "log", "scale": 0.50,
                     "targets": [p.get("ticker", "?") for p in opt_positions]},
                ))

        # --- 如果一切正常 ---
        if not alerts:
            alerts.append((
                GREEN, "所有风控指标正常",
                {"type": "none", "scale": 1.0, "targets": []},
            ))

        # --- 记录日志 ---
        worst_level = GREEN
        level_order = {GREEN: 0, YELLOW: 1, ORANGE: 2, RED: 3}
        for lvl, msg, act in alerts:
            if level_order.get(lvl, 0) > level_order.get(worst_level, 0):
                worst_level = lvl
            self._log(
                lvl, msg, act.get("type", "none"),
                portfolio_var=self.portfolio_var(port_ret) if len(port_ret) >= 20 else 0.0,
                portfolio_cvar=self.portfolio_cvar(port_ret) if len(port_ret) >= 20 else 0.0,
                max_dd=float((eq_curve / eq_curve.cummax() - 1).min()) if len(eq_curve) >= 2 else 0.0,
            )

        return alerts

    def generate_risk_report(
        self,
        portfolio_state: Dict[str, Any],
    ) -> str:
        """
        生成格式化风险报告

        Parameters
        ----------
        portfolio_state : dict
            与 ``evaluate()`` 相同的组合状态字典

        Returns
        -------
        str
            格式化的风险报告文本
        """
        equity = portfolio_state.get("equity", 0)
        positions = portfolio_state.get("positions", {})
        port_ret = portfolio_state.get("portfolio_returns", pd.Series(dtype=float))
        eq_curve = portfolio_state.get("equity_curve", pd.Series(dtype=float))

        lines = []
        sep = "=" * 60
        lines.append(sep)
        lines.append("  风险管理报告")
        lines.append(f"  生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(sep)

        # --- 组合概览 ---
        lines.append("\n[1] 组合概览")
        lines.append(f"  净值:         ${equity:,.2f}")
        lines.append(f"  持仓数:       {len(positions)}")
        if len(port_ret) >= 2:
            ann_vol = float(port_ret.std()) * np.sqrt(252)
            lines.append(f"  年化波动率:   {ann_vol:.2%}")

        # --- VaR / CVaR ---
        if len(port_ret) >= 20:
            var95 = self.portfolio_var(port_ret, 0.95)
            cvar95 = self.portfolio_cvar(port_ret, 0.95)
            var99 = self.portfolio_var(port_ret, 0.99)
            cvar99 = self.portfolio_cvar(port_ret, 0.99)
            lines.append("\n[2] 风险指标")
            lines.append(f"  VaR(95%):     {var95:.4f}  (${equity * abs(var95):,.0f})")
            lines.append(f"  CVaR(95%):    {cvar95:.4f}  (${equity * abs(cvar95):,.0f})")
            lines.append(f"  VaR(99%):     {var99:.4f}  (${equity * abs(var99):,.0f})")
            lines.append(f"  CVaR(99%):    {cvar99:.4f}  (${equity * abs(cvar99):,.0f})")

        # --- 回撤 ---
        if len(eq_curve) >= 2:
            dd, scale = self.max_drawdown_check(eq_curve)
            lines.append(f"\n[3] 回撤状态")
            lines.append(f"  当前回撤:     {dd:.2%}")
            lines.append(f"  仓位缩放:     {scale:.0%}")

        # --- 持仓集中度 ---
        lines.append(f"\n[4] 持仓集中度")
        size_alerts = self.check_position_size(positions, equity)
        if size_alerts:
            for ticker, lvl, w in size_alerts:
                lines.append(f"  [{lvl}] {ticker}: {w:.1%}")
        else:
            lines.append("  所有持仓在限额内")

        # --- 波动率目标 ---
        if len(port_ret) >= 20:
            vol_scale = self.vol_target_scale(port_ret)
            lines.append(f"\n[5] 波动率目标")
            lines.append(f"  建议缩放:     {vol_scale:.2f}x")

        # --- 压力测试 ---
        if len(port_ret) >= 20:
            st = self.stress_test(port_ret)
            lines.append(f"\n[6] 压力测试")
            for _, row in st.iterrows():
                lines.append(
                    f"  {row['scenario']:<20s} 预计损失: {row['estimated_loss']:.2%}"
                    f"  ({row['description']})"
                )

        # --- 综合评估 ---
        alerts = self.evaluate(portfolio_state)
        lines.append(f"\n[7] 综合评估")
        for lvl, msg, act in alerts:
            lines.append(f"  [{lvl}] {msg}  -> {act['type']}")

        lines.append(sep)
        return "\n".join(lines)
