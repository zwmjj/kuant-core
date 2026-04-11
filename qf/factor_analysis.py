"""因子 IC/IR 衰减分析模块

对所有日频因子进行 Information Coefficient (IC) 和 Information Ratio (IR) 分析，
包括 IC 衰减曲线、因子排名、热力图和完整 tearsheet 报告。

institutional-grade statistical rigor:
- Newey-West HAC standard errors for IC t-stats
- IC autocorrelation detection (Ljung-Box) + effective N adjustment
- Multiple testing correction (Bonferroni / BH-FDR)
- VIF multicollinearity diagnostics
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无头模式，避免弹窗
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as sp_stats
try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.stats.multitest import multipletests
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# 设置中文字体（优先 SimHei，回退 sans-serif）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class FactorAnalyzer:
    """因子 IC/IR 衰减分析器

    提供滚动 IC 计算、IR 指标、多滞后期 IC 衰减分析、
    全因子排名比较、热力图和单因子 tearsheet 等功能。

    Parameters
    ----------
    method : str
        IC 计算方法，'spearman'（秩相关，默认）或 'pearson'
    min_obs : int
        每期截面最少有效观测数，少于此数跳过该期
    """

    def __init__(self, method: str = "spearman", min_obs: int = 20):
        self.method = method
        self.min_obs = min_obs

    # ------------------------------------------------------------------
    # 核心计算
    # ------------------------------------------------------------------

    def compute_ic_series(
        self,
        factor: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.Series:
        """计算滚动 IC (Information Coefficient) 时间序列

        每期对因子截面值与下期收益做截面相关（Spearman / Pearson），
        得到逐期 IC 序列。

        Parameters
        ----------
        factor : pd.DataFrame
            因子值矩阵，index=日期，columns=股票代码
        returns : pd.DataFrame
            下期收益矩阵（需与 factor 对齐，即 returns[t] 是 factor[t] 的下期收益）

        Returns
        -------
        pd.Series
            以日期为索引的 IC 时间序列
        """
        common_dates = factor.index.intersection(returns.index)
        common_cols = factor.columns.intersection(returns.columns)

        ic_values = []
        valid_dates = []
        for date in common_dates:
            f = factor.loc[date, common_cols].dropna()
            r = returns.loc[date, common_cols].dropna()
            overlap = f.index.intersection(r.index)
            if len(overlap) < self.min_obs:
                continue
            if self.method == "spearman":
                ic = f[overlap].rank().corr(r[overlap].rank())
            else:
                ic = f[overlap].corr(r[overlap])
            ic_values.append(ic)
            valid_dates.append(date)

        return pd.Series(ic_values, index=valid_dates, name="IC")

    @staticmethod
    def compute_ir(ic_series: pd.Series) -> float:
        """计算 Information Ratio = mean(IC) / std(IC)

        IR 衡量因子预测能力的稳定性，|IR| > 0.5 通常认为是有效因子。

        Parameters
        ----------
        ic_series : pd.Series
            IC 时间序列（由 compute_ic_series 产出）

        Returns
        -------
        float
            Information Ratio；若 std(IC)==0 则返回 0.0
        """
        ic_clean = ic_series.dropna()
        if len(ic_clean) == 0 or ic_clean.std() == 0:
            return 0.0
        return float(ic_clean.mean() / ic_clean.std())

    # ------------------------------------------------------------------
    # statistical Statistical Rigor: Newey-West, Autocorrelation, Multiple Testing, VIF
    # ------------------------------------------------------------------

    def ic_tstat_newey_west(self, ic_series: pd.Series) -> dict:
        """Newey-West HAC t-stat for IC series mean != 0

        Monthly IC series typically exhibit autocorrelation, making naive
        t-stats (assuming i.i.d.) inflated by 2-3x. Newey-West corrects
        standard errors for heteroskedasticity and autocorrelation.

        Parameters
        ----------
        ic_series : pd.Series
            IC time series

        Returns
        -------
        dict with keys: ic_mean, ic_std, t_naive, t_nw, nw_se, n_obs, lag_m
        """
        ic = ic_series.dropna()
        n = len(ic)
        if n < 10:
            return {'ic_mean': np.nan, 't_naive': np.nan, 't_nw': np.nan,
                    'nw_se': np.nan, 'n_obs': n, 'lag_m': 0}

        ic_mean = ic.mean()
        ic_std = ic.std(ddof=1)
        t_naive = ic_mean / (ic_std / np.sqrt(n))

        if not HAS_STATSMODELS:
            return {'ic_mean': ic_mean, 't_naive': t_naive, 't_nw': t_naive,
                    'nw_se': ic_std / np.sqrt(n), 'n_obs': n, 'lag_m': 0}

        # Newey-West with m = T^(1/3) lags
        lag_m = max(1, int(n ** (1/3)))
        y = np.array(ic)
        X = add_constant(np.ones(n))
        model = OLS(y, X[:, :1]).fit(cov_type='HAC',
                                      cov_kwds={'maxlags': lag_m})
        t_nw = float(model.tvalues[0])
        nw_se = float(model.bse[0])

        return {
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            't_naive': t_naive,
            't_nw': t_nw,
            'nw_se': nw_se,
            'n_obs': n,
            'lag_m': lag_m,
        }

    def ic_autocorrelation_test(self, ic_series: pd.Series, max_lag: int = 5) -> dict:
        """Test IC series for autocorrelation and compute effective N

        Uses Ljung-Box test and first-order autocorrelation to adjust
        the effective sample size: N_eff = N * (1-rho) / (1+rho)

        Parameters
        ----------
        ic_series : pd.Series
            IC time series
        max_lag : int
            Maximum lag for Ljung-Box test

        Returns
        -------
        dict with: rho1, n_eff, n_obs, ljung_box_stat, ljung_box_pvalue,
                   has_autocorrelation, t_adjusted
        """
        ic = ic_series.dropna()
        n = len(ic)
        if n < 20:
            return {'rho1': np.nan, 'n_eff': n, 'n_obs': n,
                    'ljung_box_pvalue': np.nan, 'has_autocorrelation': False,
                    't_adjusted': np.nan}

        # First-order autocorrelation
        rho1 = float(ic.autocorr(lag=1))

        # Effective N (Bayley & David adjustment)
        if abs(rho1) < 1.0:
            n_eff = n * (1 - rho1) / (1 + rho1)
        else:
            n_eff = n

        n_eff = max(n_eff, 2)  # floor

        # Ljung-Box test
        lb_stat, lb_pval = np.nan, np.nan
        if HAS_STATSMODELS:
            try:
                lb_result = acorr_ljungbox(ic, lags=[max_lag], return_df=True)
                lb_stat = float(lb_result['lb_stat'].iloc[-1])
                lb_pval = float(lb_result['lb_pvalue'].iloc[-1])
            except Exception:
                pass

        has_autocorr = lb_pval < 0.05 if not np.isnan(lb_pval) else abs(rho1) > 0.15

        # Adjusted t-stat using N_eff
        ic_mean = ic.mean()
        ic_std = ic.std(ddof=1)
        t_adjusted = ic_mean / (ic_std / np.sqrt(n_eff)) if ic_std > 0 else 0.0

        return {
            'rho1': rho1,
            'n_obs': n,
            'n_eff': n_eff,
            'ljung_box_stat': lb_stat,
            'ljung_box_pvalue': lb_pval,
            'has_autocorrelation': has_autocorr,
            't_adjusted': t_adjusted,
        }

    @staticmethod
    def multiple_testing_correction(
        factor_pvalues: dict,
        method: str = 'fdr_bh',
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Apply multiple testing correction across all tested factors

        When testing N factors, the expected number of false positives = N * alpha.
        This corrects p-values using Bonferroni, BH-FDR, or Holm methods.

        Parameters
        ----------
        factor_pvalues : dict
            {factor_name: p_value} from IC t-tests
        method : str
            'bonferroni', 'holm', 'fdr_bh' (Benjamini-Hochberg), 'fdr_by'
        alpha : float
            Family-wise error rate

        Returns
        -------
        pd.DataFrame with columns: factor, p_raw, p_adjusted, reject, method
        """
        if not HAS_STATSMODELS:
            df = pd.DataFrame({
                'factor': list(factor_pvalues.keys()),
                'p_raw': list(factor_pvalues.values()),
            })
            df['p_adjusted'] = df['p_raw']  # no correction
            df['reject'] = df['p_raw'] < alpha
            df['method'] = 'none (statsmodels not installed)'
            return df

        names = list(factor_pvalues.keys())
        pvals = np.array(list(factor_pvalues.values()))

        # Handle NaN p-values
        valid_mask = ~np.isnan(pvals)
        if valid_mask.sum() == 0:
            return pd.DataFrame({'factor': names, 'p_raw': pvals,
                                 'p_adjusted': np.nan, 'reject': False, 'method': method})

        reject_arr = np.full(len(pvals), False)
        padj_arr = np.full(len(pvals), np.nan)

        reject_valid, padj_valid, _, _ = multipletests(
            pvals[valid_mask], alpha=alpha, method=method
        )
        reject_arr[valid_mask] = reject_valid
        padj_arr[valid_mask] = padj_valid

        df = pd.DataFrame({
            'factor': names,
            'p_raw': pvals,
            'p_adjusted': padj_arr,
            'reject': reject_arr,
            'method': method,
        })
        return df.sort_values('p_adjusted')

    @staticmethod
    def compute_vif(factors_df: pd.DataFrame, max_factors: int = 30) -> pd.DataFrame:
        """Compute Variance Inflation Factor for factor multicollinearity

        VIF_j = 1 / (1 - R²_j) where R²_j is from regressing factor j on all others.
        VIF > 5: concern. VIF > 10: severe multicollinearity.

        Parameters
        ----------
        factors_df : pd.DataFrame
            Cross-sectional factor values (rows=stocks, columns=factors)
            Typically computed on a single date or averaged across dates.
        max_factors : int
            Max factors to include (truncate for speed)

        Returns
        -------
        pd.DataFrame with columns: factor, vif, concern_level
        """
        df = factors_df.dropna().copy()
        cols = list(df.columns)[:max_factors]
        df = df[cols].dropna()

        if len(df) < len(cols) + 2:
            return pd.DataFrame({'factor': cols, 'vif': np.nan, 'concern_level': 'insufficient_data'})

        if not HAS_STATSMODELS:
            # Manual VIF via R² from OLS
            results = []
            for i, col in enumerate(cols):
                y = df[col].values
                X = df[[c for c in cols if c != col]].values
                X = np.column_stack([np.ones(len(X)), X])
                try:
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    y_hat = X @ beta
                    ss_res = np.sum((y - y_hat) ** 2)
                    ss_tot = np.sum((y - y.mean()) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                    vif = 1 / (1 - r2) if r2 < 1 else np.inf
                except Exception:
                    vif = np.nan
                results.append({'factor': col, 'vif': vif})
        else:
            X = add_constant(df[cols].values)
            results = []
            for i, col in enumerate(cols):
                try:
                    vif = variance_inflation_factor(X, i + 1)  # +1 for constant
                except Exception:
                    vif = np.nan
                results.append({'factor': col, 'vif': vif})

        out = pd.DataFrame(results)
        out['concern_level'] = out['vif'].apply(
            lambda v: 'SEVERE (>10)' if v > 10
            else 'WARNING (>5)' if v > 5
            else 'OK' if not np.isnan(v)
            else 'N/A'
        )
        return out.sort_values('vif', ascending=False)

    def full_statistical_audit(
        self,
        factors_dict: dict,
        returns: pd.DataFrame,
        correction_method: str = 'fdr_bh',
        alpha: float = 0.05,
    ) -> dict:
        """Run complete institutional-grade statistical audit on all factors

        Combines: Newey-West t-stats, autocorrelation tests, multiple testing
        correction, and VIF multicollinearity check.

        Parameters
        ----------
        factors_dict : dict
            {factor_name: factor_df}
        returns : pd.DataFrame
            Forward returns (already shifted if needed)
        correction_method : str
            Multiple testing method ('bonferroni', 'fdr_bh', 'holm')
        alpha : float
            Significance level

        Returns
        -------
        dict with keys: factor_stats (DataFrame), multiple_testing (DataFrame),
                        vif (DataFrame), summary (dict)
        """
        fwd_ret = returns.shift(-1)
        factor_rows = []
        pvalues = {}

        for name, factor in factors_dict.items():
            ic_series = self.compute_ic_series(factor, fwd_ret)

            # Newey-West
            nw = self.ic_tstat_newey_west(ic_series)

            # Autocorrelation
            ac = self.ic_autocorrelation_test(ic_series)

            # p-value from Newey-West t-stat (two-sided)
            if not np.isnan(nw['t_nw']) and nw.get('n_obs', 0) > 10:
                p_nw = 2 * (1 - sp_stats.norm.cdf(abs(nw['t_nw'])))
            else:
                p_nw = np.nan
            pvalues[name] = p_nw

            factor_rows.append({
                'factor': name,
                'ic_mean': nw['ic_mean'],
                'ir': nw['ic_mean'] / nw['ic_std'] if nw.get('ic_std', 0) > 0 else 0,
                't_naive': nw['t_naive'],
                't_newey_west': nw['t_nw'],
                'nw_se': nw['nw_se'],
                'nw_lags': nw['lag_m'],
                'rho1': ac['rho1'],
                'n_eff': ac['n_eff'],
                'n_obs': ac['n_obs'],
                'ljung_box_p': ac['ljung_box_pvalue'],
                'has_autocorr': ac['has_autocorrelation'],
                't_neff_adjusted': ac['t_adjusted'],
                'p_newey_west': p_nw,
            })

        factor_stats = pd.DataFrame(factor_rows).set_index('factor')

        # Multiple testing correction
        mt = self.multiple_testing_correction(pvalues, method=correction_method, alpha=alpha)

        # VIF — use median cross-section
        vif_df = pd.DataFrame()
        try:
            # Build cross-sectional factor matrix at median date
            dates = list(factors_dict.values())[0].index
            mid_idx = len(dates) // 2
            sample_dates = dates[max(0, mid_idx-5):mid_idx+5]
            cs_frames = []
            for d in sample_dates:
                row = {}
                for name, fdf in factors_dict.items():
                    if d in fdf.index:
                        row[name] = fdf.loc[d]
                if row:
                    cs_frames.append(pd.DataFrame(row))
            if cs_frames:
                cs_all = pd.concat(cs_frames, ignore_index=True)
                vif_df = self.compute_vif(cs_all)
        except Exception:
            pass

        # Summary
        n_significant_raw = (factor_stats['p_newey_west'] < alpha).sum()
        n_significant_adj = mt['reject'].sum() if len(mt) > 0 else 0
        n_autocorr = factor_stats['has_autocorr'].sum()
        n_high_vif = (vif_df['vif'] > 5).sum() if len(vif_df) > 0 else 0

        summary = {
            'n_factors_tested': len(factors_dict),
            'n_significant_raw': int(n_significant_raw),
            'n_significant_adjusted': int(n_significant_adj),
            'expected_false_positives': len(factors_dict) * alpha,
            'n_with_autocorrelation': int(n_autocorr),
            'n_high_vif': int(n_high_vif),
            'correction_method': correction_method,
            'median_t_inflation': float(
                (factor_stats['t_naive'].abs() / factor_stats['t_newey_west'].abs().replace(0, np.nan)).median()
            ) if len(factor_stats) > 0 else np.nan,
        }

        return {
            'factor_stats': factor_stats,
            'multiple_testing': mt,
            'vif': vif_df,
            'summary': summary,
        }

    def ic_decay_analysis(
        self,
        factor: pd.DataFrame,
        returns: pd.DataFrame,
        lags: list = None,
    ) -> pd.DataFrame:
        """计算不同滞后期的 IC 衰减

        对因子 factor[t] 分别与 returns[t+lag] 计算截面 IC，
        观察随着持有期拉长，因子预测力如何衰减。

        Parameters
        ----------
        factor : pd.DataFrame
            因子值矩阵，index=日期，columns=股票代码
        returns : pd.DataFrame
            日收益矩阵（原始日收益，函数内部会按 lag 做 shift）
        lags : list of int
            滞后天数列表，默认 [1, 2, 3, 5, 10, 20]

        Returns
        -------
        pd.DataFrame
            index=lag, columns=['IC_mean', 'IC_std', 'IR', 'IC_positive_pct']
        """
        if lags is None:
            lags = [1, 2, 3, 5, 10, 20]

        records = []
        for lag in lags:
            # 将日收益按 lag 滚动累积，得到 lag 天后的累积收益
            fwd_returns = returns.shift(-lag).rolling(lag).apply(
                lambda x: (1 + x).prod() - 1, raw=True
            ) if lag > 1 else returns.shift(-1)

            ic_series = self.compute_ic_series(factor, fwd_returns)
            ic_clean = ic_series.dropna()
            records.append({
                "lag": lag,
                "IC_mean": ic_clean.mean() if len(ic_clean) > 0 else np.nan,
                "IC_std": ic_clean.std() if len(ic_clean) > 0 else np.nan,
                "IR": self.compute_ir(ic_series),
                "IC_positive_pct": float((ic_clean > 0).mean()) if len(ic_clean) > 0 else np.nan,
            })

        return pd.DataFrame(records).set_index("lag")

    def _compute_turnover(
        self,
        factor: pd.DataFrame,
        top_n: int = 20,
    ) -> float:
        """计算因子信号的平均换手率

        Parameters
        ----------
        factor : pd.DataFrame
            因子值矩阵
        top_n : int
            持仓数量（取信号最高的 top_n 只）

        Returns
        -------
        float
            平均换手率 [0, 1]
        """
        turnover_list = []
        prev_holdings = set()
        for date in factor.index:
            row = factor.loc[date].dropna()
            if len(row) < top_n:
                prev_holdings = set(row.index)
                continue
            current = set(row.nlargest(top_n).index)
            if prev_holdings:
                changed = len(current - prev_holdings) + len(prev_holdings - current)
                turnover_list.append(changed / (2 * top_n))
            prev_holdings = current

        return float(np.nanmean(turnover_list)) if turnover_list else np.nan

    def rank_all_factors(
        self,
        factors_dict: dict,
        returns: pd.DataFrame,
        lags: list = None,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """对所有因子计算 IC / IR / IC衰减 / 换手率，返回排名 DataFrame

        Parameters
        ----------
        factors_dict : dict
            因子名 -> 因子值 DataFrame 的字典
        returns : pd.DataFrame
            日收益矩阵
        lags : list of int
            IC 衰减滞后期列表
        top_n : int
            换手率计算用的持仓数

        Returns
        -------
        pd.DataFrame
            index=因子名，columns 包含各项指标，按 |IC_mean| 降序排列
        """
        if lags is None:
            lags = [1, 2, 3, 5, 10, 20]

        rows = []
        for name, factor in factors_dict.items():
            # 1日前向收益的 IC
            fwd_ret_1d = returns.shift(-1)
            ic_series = self.compute_ic_series(factor, fwd_ret_1d)
            ic_clean = ic_series.dropna()

            row = {
                "factor": name,
                "IC_mean": ic_clean.mean() if len(ic_clean) > 0 else np.nan,
                "IC_std": ic_clean.std() if len(ic_clean) > 0 else np.nan,
                "IR": self.compute_ir(ic_series),
                "IC_positive_pct": float((ic_clean > 0).mean()) if len(ic_clean) > 0 else np.nan,
                "turnover": self._compute_turnover(factor, top_n),
            }

            # 各滞后期 IC
            decay = self.ic_decay_analysis(factor, returns, lags)
            for lag in lags:
                if lag in decay.index:
                    row[f"IC_lag{lag}"] = decay.loc[lag, "IC_mean"]
                else:
                    row[f"IC_lag{lag}"] = np.nan

            rows.append(row)

        result = pd.DataFrame(rows).set_index("factor")
        # 按 |IC_mean| 降序排列
        result["abs_IC"] = result["IC_mean"].abs()
        result = result.sort_values("abs_IC", ascending=False).drop(columns=["abs_IC"])
        return result

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------

    def plot_ic_heatmap(
        self,
        results: pd.DataFrame,
        save_path: str = None,
    ) -> plt.Figure:
        """生成 IC 衰减热力图

        横轴为滞后天数，纵轴为因子名称，颜色深浅表示 IC 大小。

        Parameters
        ----------
        results : pd.DataFrame
            rank_all_factors 的输出
        save_path : str or None
            若指定则保存图片

        Returns
        -------
        matplotlib.figure.Figure
        """
        # 提取 IC_lag 列
        lag_cols = [c for c in results.columns if c.startswith("IC_lag")]
        if not lag_cols:
            raise ValueError("results 中未找到 IC_lag 列，请先运行 rank_all_factors")

        heatmap_data = results[lag_cols].copy()
        # 列名美化: IC_lag1 -> 1
        heatmap_data.columns = [c.replace("IC_lag", "") for c in heatmap_data.columns]

        fig, ax = plt.subplots(figsize=(10, max(4, len(heatmap_data) * 0.45)))
        im = ax.imshow(
            heatmap_data.values.astype(float),
            aspect="auto",
            cmap="RdYlGn",
            vmin=-0.15,
            vmax=0.15,
        )

        # 标注数值
        for i in range(heatmap_data.shape[0]):
            for j in range(heatmap_data.shape[1]):
                val = heatmap_data.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

        ax.set_xticks(range(len(heatmap_data.columns)))
        ax.set_xticklabels(heatmap_data.columns)
        ax.set_yticks(range(len(heatmap_data.index)))
        ax.set_yticklabels(heatmap_data.index)
        ax.set_xlabel("滞后天数 (lag)")
        ax.set_ylabel("因子")
        ax.set_title("IC 衰减热力图")
        fig.colorbar(im, ax=ax, label="IC")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_factor_tearsheet(
        self,
        factor_name: str,
        ic_series: pd.Series,
        save_path: str = None,
    ) -> plt.Figure:
        """单因子分析图 — IC 时间序列 + IC 分布 + 累积IC

        三张子图:
        1. IC 时间序列折线图 + 20期滚动均值
        2. IC 分布直方图
        3. 累积 IC 曲线

        Parameters
        ----------
        factor_name : str
            因子名称（用于标题）
        ic_series : pd.Series
            IC 时间序列
        save_path : str or None
            若指定则保存图片

        Returns
        -------
        matplotlib.figure.Figure
        """
        ic_clean = ic_series.dropna()
        if len(ic_clean) == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"{factor_name}: 无有效 IC 数据", ha="center", va="center")
            return fig

        ir = self.compute_ir(ic_series)

        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

        # ── 子图 1: IC 时间序列 ──
        ax1 = fig.add_subplot(gs[0, :])
        ax1.bar(range(len(ic_clean)), ic_clean.values, color="steelblue", alpha=0.6, width=1.0)
        # 滚动均值
        rolling_window = min(20, len(ic_clean) // 3) if len(ic_clean) > 6 else len(ic_clean)
        if rolling_window > 1:
            rolling_ic = ic_clean.rolling(rolling_window).mean()
            ax1.plot(range(len(ic_clean)), rolling_ic.values, color="red", linewidth=1.5,
                     label=f"{rolling_window}期滚动均值")
        ax1.axhline(0, color="black", linewidth=0.5)
        ax1.set_title(f"{factor_name} — IC 时间序列 (mean={ic_clean.mean():.4f}, IR={ir:.3f})")
        ax1.set_xlabel("期数")
        ax1.set_ylabel("IC")
        ax1.legend()

        # ── 子图 2: IC 分布 ──
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(ic_clean.values, bins=30, color="steelblue", alpha=0.7, edgecolor="white")
        ax2.axvline(ic_clean.mean(), color="red", linestyle="--", label=f"均值={ic_clean.mean():.4f}")
        ax2.axvline(0, color="black", linewidth=0.5)
        ax2.set_title(f"{factor_name} — IC 分布")
        ax2.set_xlabel("IC")
        ax2.set_ylabel("频数")
        ax2.legend()

        # ── 子图 3: 累积 IC ──
        ax3 = fig.add_subplot(gs[1, 1])
        cum_ic = ic_clean.cumsum()
        ax3.plot(range(len(cum_ic)), cum_ic.values, color="steelblue", linewidth=1.5)
        ax3.fill_between(range(len(cum_ic)), 0, cum_ic.values, alpha=0.2, color="steelblue")
        ax3.axhline(0, color="black", linewidth=0.5)
        ax3.set_title(f"{factor_name} — 累积 IC")
        ax3.set_xlabel("期数")
        ax3.set_ylabel("累积 IC")

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # 报告生成
    # ------------------------------------------------------------------

    def generate_report(
        self,
        factors_dict: dict,
        returns: pd.DataFrame,
        output_dir: str,
        lags: list = None,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """生成完整因子分析报告

        包括:
        - 全因子排名表（CSV）
        - IC 衰减热力图（PNG）
        - 每个因子的 tearsheet 图（PNG）

        Parameters
        ----------
        factors_dict : dict
            因子名 -> 因子值 DataFrame
        returns : pd.DataFrame
            日收益矩阵
        output_dir : str
            输出目录路径
        lags : list of int
            IC 衰减滞后期列表
        top_n : int
            换手率持仓数

        Returns
        -------
        pd.DataFrame
            全因子排名表
        """
        if lags is None:
            lags = [1, 2, 3, 5, 10, 20]

        os.makedirs(output_dir, exist_ok=True)

        # 1. 计算全因子排名
        print("=" * 60)
        print("因子 IC/IR 衰减分析报告")
        print("=" * 60)

        ranking = self.rank_all_factors(factors_dict, returns, lags, top_n)
        csv_path = os.path.join(output_dir, "factor_ranking.csv")
        ranking.to_csv(csv_path, float_format="%.6f")
        print(f"\n[1/3] 因子排名表已保存: {csv_path}")
        print(ranking.to_string(float_format=lambda x: f"{x:.4f}"))

        # 2. IC 衰减热力图
        heatmap_path = os.path.join(output_dir, "ic_decay_heatmap.png")
        self.plot_ic_heatmap(ranking, save_path=heatmap_path)
        plt.close()
        print(f"\n[2/3] IC 衰减热力图已保存: {heatmap_path}")

        # 3. 各因子 tearsheet
        fwd_ret_1d = returns.shift(-1)
        print(f"\n[3/3] 生成单因子 tearsheet ...")
        for name, factor in factors_dict.items():
            ic_series = self.compute_ic_series(factor, fwd_ret_1d)
            fig_path = os.path.join(output_dir, f"tearsheet_{name}.png")
            self.plot_factor_tearsheet(name, ic_series, save_path=fig_path)
            plt.close()
            ir = self.compute_ir(ic_series)
            print(f"  {name:30s}  IC={ic_series.mean():.4f}  IR={ir:.3f}  -> {fig_path}")

        print("\n" + "=" * 60)
        print("报告生成完毕!")
        print("=" * 60)

        return ranking


# ======================================================================
# 模拟数据演示
# ======================================================================

def _generate_mock_data(n_dates=500, n_stocks=100, seed=42):
    """生成模拟的价格/收益/因子数据用于演示

    Parameters
    ----------
    n_dates : int
        交易日数
    n_stocks : int
        股票数
    seed : int
        随机种子

    Returns
    -------
    tuple
        (factors_dict, returns) — 因子字典和日收益 DataFrame
    """
    np.random.seed(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_dates, freq="B")
    tickers = [f"STOCK_{i:03d}" for i in range(n_stocks)]

    # 模拟日收益（含微弱因子效应）
    base_returns = np.random.randn(n_dates, n_stocks) * 0.02
    returns = pd.DataFrame(base_returns, index=dates, columns=tickers)

    # 模拟日频因子（与 signals_daily.py 中的因子对应）
    daily_factor_names = [
        "momentum_5d", "momentum_20d", "momentum_reversal_5d",
        "volume_surge", "dollar_volume_rank", "volume_price_trend",
        "volume_price_divergence",
        "vwap_deviation", "vwap_reversion",
        "realized_vol_20d", "vol_breakout",
        "overnight_gap",
        "amihud_illiquidity", "trade_intensity", "high_low_spread",
    ]

    factors_dict = {}
    for i, name in enumerate(daily_factor_names):
        # 因子 = 噪声 + 微弱的与收益相关的信号
        signal_strength = 0.02 + 0.01 * (i % 5)  # 不同因子不同强度
        noise = np.random.randn(n_dates, n_stocks)
        # 用下期收益的 shift 来模拟因子的预测力
        signal = np.roll(base_returns, -1, axis=0) * signal_strength * 50
        raw = signal + noise
        factors_dict[name] = pd.DataFrame(raw, index=dates, columns=tickers)

    return factors_dict, returns


if __name__ == "__main__":
    print("生成模拟数据 ...")
    factors_dict, returns = _generate_mock_data()

    print(f"因子数: {len(factors_dict)}")
    print(f"日期范围: {returns.index[0].date()} ~ {returns.index[-1].date()}")
    print(f"股票数: {returns.shape[1]}")

    # 创建分析器并生成报告
    analyzer = FactorAnalyzer(method="spearman", min_obs=20)

    output_dir = os.path.join(os.path.dirname(__file__), "factor_analysis_output")
    ranking = analyzer.generate_report(
        factors_dict, returns, output_dir=output_dir,
        lags=[1, 2, 3, 5, 10, 20],
    )

    # 单独演示 IC 衰减分析
    print("\n--- 单因子 IC 衰减示例 (momentum_5d) ---")
    decay = analyzer.ic_decay_analysis(
        factors_dict["momentum_5d"], returns, lags=[1, 2, 3, 5, 10, 20]
    )
    print(decay.to_string(float_format=lambda x: f"{x:.4f}"))
