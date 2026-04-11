"""因子重要性分析与最优Alpha组合筛选

使用 LightGBM / XGBoost / Permutation Importance 对因子进行重要性排序，
并通过逐步前向选择和组合枚举找到最强 alpha 因子组合。

日频因子 (signals_daily.py):
    momentum_5d, momentum_20d, momentum_reversal_5d,
    volume_surge, dollar_volume_rank, volume_price_trend, volume_price_divergence,
    vwap_deviation, vwap_reversion,
    realized_vol_20d, vol_breakout, overnight_gap,
    amihud_illiquidity, trade_intensity, high_low_spread

月频因子 (signals.py):
    mom12, accel, high52, bm, ep, roe, gpa, ag,
    turnover, revgrowth, de, ps, nigrowth,
    ivol, ff5alpha, lowvol, vov, downvol, maxret, beta, skew, volts, volmom
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import xgboost as xgb
from sklearn.inspection import permutation_importance as sk_permutation_importance
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class FactorSelector:
    """因子重要性分析与最优组合筛选器

    工作流程:
    1. prepare_features: 因子矩阵 + 前瞻收益 → 对齐的 X, y
    2. lgbm_importance / xgb_importance: 树模型特征重要性
    3. permutation_importance: 置换重要性（模型无关）
    4. stepwise_selection: 逐步前向选择最优因子组合
    5. find_best_combination: 枚举 top N 因子的最优子集
    6. generate_report: 完整分析报告
    """

    def __init__(self, train_ratio: float = 0.7, random_state: int = 42):
        """
        Parameters
        ----------
        train_ratio : float
            训练集占比，默认 0.7（前 70% 训练，后 30% 测试）
        random_state : int
            随机种子
        """
        self.train_ratio = train_ratio
        self.random_state = random_state

    # ──────────────────────────────────────────────
    # 数据准备
    # ──────────────────────────────────────────────

    @staticmethod
    def prepare_features(
        factors_dict: Dict[str, pd.DataFrame],
        returns: pd.DataFrame,
        forward_period: int = 5,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """将因子矩阵和前瞻收益对齐，构建训练用 X, y

        Parameters
        ----------
        factors_dict : dict
            因子名 → DataFrame (index=日期, columns=股票代码)
        returns : pd.DataFrame
            日收益率 DataFrame (index=日期, columns=股票代码)
        forward_period : int
            前瞻收益天数，默认 5 (一周)

        Returns
        -------
        X : pd.DataFrame
            特征矩阵，每行对应 (日期, 股票) 的因子值
        y : pd.Series
            前瞻累计收益
        """
        # 计算前瞻收益: 未来 forward_period 天的累计收益
        forward_ret = returns.shift(-forward_period).rolling(forward_period).sum()

        # 找到所有因子的公共日期和股票
        common_dates = returns.index.copy()
        common_symbols = returns.columns.copy()
        for name, factor_df in factors_dict.items():
            common_dates = common_dates.intersection(factor_df.index)
            common_symbols = common_symbols.intersection(factor_df.columns)

        common_dates = common_dates.intersection(forward_ret.index)
        common_dates = sorted(common_dates)

        if len(common_dates) == 0 or len(common_symbols) == 0:
            raise ValueError("因子矩阵与收益数据无重叠的日期或股票")

        # 构建 panel 数据: 将 (date, symbol) 展平为行
        records = []
        for dt in common_dates:
            for sym in common_symbols:
                row = {}
                y_val = forward_ret.loc[dt, sym] if dt in forward_ret.index else np.nan
                if pd.isna(y_val):
                    continue
                all_valid = True
                for fname, fdf in factors_dict.items():
                    val = fdf.loc[dt, sym] if (dt in fdf.index and sym in fdf.columns) else np.nan
                    row[fname] = val
                    # 允许部分因子缺失，后续用 NaN 处理
                row['__y__'] = y_val
                row['__date__'] = dt
                row['__symbol__'] = sym
                records.append(row)

        if len(records) == 0:
            raise ValueError("对齐后无有效样本，请检查数据")

        panel = pd.DataFrame(records)
        # 按日期排序，确保时间切分正确
        panel = panel.sort_values('__date__').reset_index(drop=True)

        factor_names = list(factors_dict.keys())
        X = panel[factor_names].copy()
        y = panel['__y__'].copy()
        y.index = panel.index

        # 保存日期信息供时间切分使用
        X.attrs['dates'] = panel['__date__'].values
        X.attrs['symbols'] = panel['__symbol__'].values

        return X, y

    def _time_split(self, X: pd.DataFrame, y: pd.Series):
        """严格时间切分: 前 train_ratio 训练，后面测试

        Returns
        -------
        X_train, X_test, y_train, y_test
        """
        n = len(X)
        split_idx = int(n * self.train_ratio)
        X_train = X.iloc[:split_idx].copy()
        X_test = X.iloc[split_idx:].copy()
        y_train = y.iloc[:split_idx].copy()
        y_test = y.iloc[split_idx:].copy()
        return X_train, X_test, y_train, y_test

    # ──────────────────────────────────────────────
    # LightGBM 因子重要性
    # ──────────────────────────────────────────────

    def lgbm_importance(
        self, X: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """用 LightGBM 训练并返回特征重要性排序 (gain + split 两种)

        Parameters
        ----------
        X : pd.DataFrame
            特征矩阵
        y : pd.Series
            目标变量（前瞻收益）

        Returns
        -------
        pd.DataFrame
            columns=['feature', 'gain', 'split', 'gain_rank', 'split_rank']
            按 gain 降序排列
        """
        X_train, X_test, y_train, y_test = self._time_split(X, y)

        params = {
            'objective': 'regression',
            'metric': 'mse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': self.random_state,
            'n_jobs': -1,
        }

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_test, label=y_test, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        # 提取两种重要性
        gain_imp = model.feature_importance(importance_type='gain')
        split_imp = model.feature_importance(importance_type='split')
        features = model.feature_name()

        result = pd.DataFrame({
            'feature': features,
            'gain': gain_imp,
            'split': split_imp,
        })
        result['gain_rank'] = result['gain'].rank(ascending=False).astype(int)
        result['split_rank'] = result['split'].rank(ascending=False).astype(int)
        result = result.sort_values('gain', ascending=False).reset_index(drop=True)

        # 计算测试集 MSE
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"[LightGBM] 测试集 MSE: {mse:.6f}  |  训练 {len(X_train)} 样本, 测试 {len(X_test)} 样本")

        self._lgbm_model = model
        return result

    # ──────────────────────────────────────────────
    # XGBoost 因子重要性（对比）
    # ──────────────────────────────────────────────

    def xgb_importance(
        self, X: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """用 XGBoost 训练并返回特征重要性排序，作为 LightGBM 的对比

        Returns
        -------
        pd.DataFrame
            columns=['feature', 'gain', 'weight', 'gain_rank', 'weight_rank']
        """
        X_train, X_test, y_train, y_test = self._time_split(X, y)

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X.columns))
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=list(X.columns))

        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': self.random_state,
            'verbosity': 0,
        }

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dtest, 'test')],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        # gain 和 weight (分裂次数) 两种重要性
        gain_imp = model.get_score(importance_type='gain')
        weight_imp = model.get_score(importance_type='weight')

        features = list(X.columns)
        result = pd.DataFrame({
            'feature': features,
            'gain': [gain_imp.get(f, 0) for f in features],
            'weight': [weight_imp.get(f, 0) for f in features],
        })
        result['gain_rank'] = result['gain'].rank(ascending=False).astype(int)
        result['weight_rank'] = result['weight'].rank(ascending=False).astype(int)
        result = result.sort_values('gain', ascending=False).reset_index(drop=True)

        y_pred = model.predict(dtest)
        mse = mean_squared_error(y_test, y_pred)
        print(f"[XGBoost]  测试集 MSE: {mse:.6f}  |  训练 {len(X_train)} 样本, 测试 {len(X_test)} 样本")

        self._xgb_model = model
        return result

    # ──────────────────────────────────────────────
    # 置换重要性（模型无关）
    # ──────────────────────────────────────────────

    def permutation_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model=None,
        n_repeats: int = 10,
    ) -> pd.DataFrame:
        """用 sklearn permutation_importance 计算因子重要性

        Parameters
        ----------
        X : pd.DataFrame
        y : pd.Series
        model : 已训练的 sklearn 兼容模型，默认用 LightGBM
        n_repeats : int
            置换重复次数

        Returns
        -------
        pd.DataFrame
            columns=['feature', 'importance_mean', 'importance_std', 'rank']
        """
        X_train, X_test, y_train, y_test = self._time_split(X, y)

        if model is None:
            # 使用 sklearn 接口的 LightGBM
            model = lgb.LGBMRegressor(
                n_estimators=300,
                num_leaves=31,
                learning_rate=0.05,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                verbose=-1,
                random_state=self.random_state,
                n_jobs=-1,
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )

        # 在测试集上计算置换重要性
        perm_result = sk_permutation_importance(
            model, X_test, y_test,
            n_repeats=n_repeats,
            random_state=self.random_state,
            n_jobs=-1,
            scoring='neg_mean_squared_error',
        )

        result = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': perm_result.importances_mean,
            'importance_std': perm_result.importances_std,
        })
        result['rank'] = result['importance_mean'].rank(ascending=False).astype(int)
        result = result.sort_values('importance_mean', ascending=False).reset_index(drop=True)

        print(f"[Permutation] 基于测试集 {len(X_test)} 样本, 重复 {n_repeats} 次")
        return result

    # ──────────────────────────────────────────────
    # 逐步前向选择
    # ──────────────────────────────────────────────

    def stepwise_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        max_factors: int = 8,
    ) -> List[dict]:
        """逐步前向选择最优因子组合

        每步加入使 IC (信息系数 = rank correlation) 提升最大的因子。
        使用测试集 IC 避免过拟合。

        Parameters
        ----------
        X : pd.DataFrame
        y : pd.Series
        max_factors : int
            最多选择的因子数

        Returns
        -------
        list of dict
            每步的结果: {'step', 'added_factor', 'selected', 'train_ic', 'test_ic'}
        """
        X_train, X_test, y_train, y_test = self._time_split(X, y)

        all_factors = list(X.columns)
        selected = []
        remaining = list(all_factors)
        history = []

        for step in range(1, max_factors + 1):
            if not remaining:
                break

            best_ic = -np.inf
            best_factor = None

            for candidate in remaining:
                trial = selected + [candidate]
                # 用选中因子的等权组合计算 IC
                combo_train = X_train[trial].mean(axis=1)
                combo_test = X_test[trial].mean(axis=1)

                # Rank IC (Spearman相关)
                ic_test = combo_test.corr(y_test, method='spearman')

                if ic_test > best_ic:
                    best_ic = ic_test
                    best_factor = candidate

            if best_factor is None:
                break

            selected.append(best_factor)
            remaining.remove(best_factor)

            # 计算当前组合的训练集和测试集 IC
            combo_train = X_train[selected].mean(axis=1)
            combo_test = X_test[selected].mean(axis=1)
            train_ic = combo_train.corr(y_train, method='spearman')
            test_ic = combo_test.corr(y_test, method='spearman')

            record = {
                'step': step,
                'added_factor': best_factor,
                'selected': list(selected),
                'train_ic': round(train_ic, 6),
                'test_ic': round(test_ic, 6),
            }
            history.append(record)
            print(f"  步骤 {step}: +{best_factor:<30s}  训练IC={train_ic:.4f}  测试IC={test_ic:.4f}")

        return history

    # ──────────────────────────────────────────────
    # 最优组合枚举
    # ──────────────────────────────────────────────

    def find_best_combination(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        top_n: int = 10,
        min_size: int = 2,
        max_size: int = 5,
    ) -> pd.DataFrame:
        """枚举 top N 因子的所有组合，找出最优子集

        Parameters
        ----------
        X : pd.DataFrame
        y : pd.Series
        top_n : int
            只考虑重要性排名前 top_n 的因子
        min_size : int
            最小组合大小
        max_size : int
            最大组合大小

        Returns
        -------
        pd.DataFrame
            columns=['combination', 'size', 'train_ic', 'test_ic', 'ic_decay']
            按 test_ic 降序排列
        """
        # 先用 LightGBM 获取 top N 因子
        lgbm_result = self.lgbm_importance(X, y)
        top_factors = lgbm_result['feature'].head(top_n).tolist()
        print(f"\n[组合枚举] 基于 top {len(top_factors)} 因子: {top_factors}")

        X_train, X_test, y_train, y_test = self._time_split(X, y)
        results = []

        for size in range(min_size, min(max_size + 1, len(top_factors) + 1)):
            for combo in combinations(top_factors, size):
                combo_train = X_train[list(combo)].mean(axis=1)
                combo_test = X_test[list(combo)].mean(axis=1)

                train_ic = combo_train.corr(y_train, method='spearman')
                test_ic = combo_test.corr(y_test, method='spearman')
                ic_decay = train_ic - test_ic  # IC 衰减，越小越稳

                results.append({
                    'combination': combo,
                    'size': size,
                    'train_ic': round(train_ic, 6),
                    'test_ic': round(test_ic, 6),
                    'ic_decay': round(ic_decay, 6),
                })

        df = pd.DataFrame(results)
        df = df.sort_values('test_ic', ascending=False).reset_index(drop=True)

        # 打印 top 5
        print("\n  === 测试集 IC 排名前 5 的因子组合 ===")
        for i, row in df.head(5).iterrows():
            combo_str = ' + '.join(row['combination'])
            print(f"  #{i+1}  IC={row['test_ic']:.4f} (衰减={row['ic_decay']:.4f})  {combo_str}")

        return df

    # ──────────────────────────────────────────────
    # 可视化
    # ──────────────────────────────────────────────

    @staticmethod
    def plot_importance(
        results: pd.DataFrame,
        title: str = '因子重要性排序',
        importance_col: str = 'gain',
        top_n: int = 20,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None,
    ):
        """因子重要性柱状图

        Parameters
        ----------
        results : pd.DataFrame
            lgbm_importance / xgb_importance 的输出
        title : str
        importance_col : str
            使用哪列作为重要性度量
        top_n : int
            显示前 N 个因子
        figsize : tuple
        save_path : str or None
            保存路径，None 则显示
        """
        df = results.head(top_n).copy()
        df = df.sort_values(importance_col, ascending=True)  # 水平柱状图从下到上

        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(df)))[::-1]
        ax.barh(df['feature'], df[importance_col], color=colors)
        ax.set_xlabel(importance_col.capitalize())
        ax.set_title(title)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  图表已保存: {save_path}")
        else:
            plt.show()
        plt.close()

    @staticmethod
    def plot_stepwise(history: List[dict], save_path: Optional[str] = None):
        """逐步选择过程的 IC 变化图"""
        steps = [h['step'] for h in history]
        train_ics = [h['train_ic'] for h in history]
        test_ics = [h['test_ic'] for h in history]
        labels = [h['added_factor'] for h in history]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, train_ics, 'o-', label='训练集 IC', color='steelblue')
        ax.plot(steps, test_ics, 's-', label='测试集 IC', color='coral')
        ax.set_xticks(steps)
        ax.set_xticklabels([f"+{l}" for l in labels], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Rank IC (Spearman)')
        ax.set_title('逐步前向选择 — IC 变化')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  图表已保存: {save_path}")
        else:
            plt.show()
        plt.close()

    # ──────────────────────────────────────────────
    # 完整报告
    # ──────────────────────────────────────────────

    def generate_report(
        self,
        factors_dict: Dict[str, pd.DataFrame],
        returns: pd.DataFrame,
        forward_period: int = 5,
        top_n: int = 10,
        max_factors: int = 8,
        save_dir: Optional[str] = None,
    ) -> dict:
        """生成完整因子重要性分析报告

        流程:
        1. 数据准备与对齐
        2. LightGBM 特征重要性
        3. XGBoost 特征重要性（对比）
        4. 置换重要性
        5. 逐步前向选择
        6. 组合枚举（带外测试）
        7. 可视化输出

        Parameters
        ----------
        factors_dict : dict
            因子名 → DataFrame
        returns : pd.DataFrame
            日收益率
        forward_period : int
            前瞻天数
        top_n : int
            枚举最优组合时考虑的 top N 因子
        max_factors : int
            逐步选择最大因子数
        save_dir : str or None
            图表保存目录

        Returns
        -------
        dict
            包含所有分析结果的字典
        """
        print("=" * 70)
        print("  因子重要性分析报告")
        print("=" * 70)

        # 1. 数据准备
        print("\n[1/6] 数据准备...")
        X, y = self.prepare_features(factors_dict, returns, forward_period)
        n_samples = len(X)
        n_features = len(X.columns)
        n_train = int(n_samples * self.train_ratio)
        n_test = n_samples - n_train
        print(f"  样本数: {n_samples}  |  因子数: {n_features}")
        print(f"  训练集: {n_train} ({self.train_ratio:.0%})  |  测试集: {n_test} ({1-self.train_ratio:.0%})")
        print(f"  前瞻期: {forward_period} 天")

        # 2. LightGBM
        print(f"\n[2/6] LightGBM 特征重要性...")
        lgbm_result = self.lgbm_importance(X, y)
        print(lgbm_result[['feature', 'gain', 'gain_rank']].to_string(index=False))

        # 3. XGBoost
        print(f"\n[3/6] XGBoost 特征重要性...")
        xgb_result = self.xgb_importance(X, y)
        print(xgb_result[['feature', 'gain', 'gain_rank']].to_string(index=False))

        # 4. 置换重要性
        print(f"\n[4/6] 置换重要性...")
        perm_result = self.permutation_importance(X, y)
        print(perm_result[['feature', 'importance_mean', 'rank']].to_string(index=False))

        # 5. 逐步选择
        print(f"\n[5/6] 逐步前向选择 (最多 {max_factors} 因子)...")
        stepwise_result = self.stepwise_selection(X, y, max_factors=max_factors)

        # 6. 最优组合
        print(f"\n[6/6] 枚举 top {top_n} 因子的最优组合...")
        combo_result = self.find_best_combination(X, y, top_n=top_n)

        # 汇总
        print("\n" + "=" * 70)
        print("  汇总")
        print("=" * 70)

        # 综合排名: 三种方法的平均排名
        merged = lgbm_result[['feature', 'gain_rank']].copy()
        merged = merged.merge(
            xgb_result[['feature', 'gain_rank']].rename(columns={'gain_rank': 'xgb_rank'}),
            on='feature', how='left'
        )
        merged = merged.merge(
            perm_result[['feature', 'rank']].rename(columns={'rank': 'perm_rank'}),
            on='feature', how='left'
        )
        merged['avg_rank'] = merged[['gain_rank', 'xgb_rank', 'perm_rank']].mean(axis=1)
        merged = merged.sort_values('avg_rank').reset_index(drop=True)
        print("\n  综合排名 (LightGBM + XGBoost + Permutation 平均):")
        print(merged.to_string(index=False))

        # 最优逐步选择结果
        if stepwise_result:
            best_step = max(stepwise_result, key=lambda x: x['test_ic'])
            print(f"\n  最优逐步选择: {best_step['selected']}")
            print(f"  测试集 IC = {best_step['test_ic']:.4f}")

        # 最优组合
        if len(combo_result) > 0:
            best_combo = combo_result.iloc[0]
            print(f"\n  最优组合: {list(best_combo['combination'])}")
            print(f"  测试集 IC = {best_combo['test_ic']:.4f}  |  IC衰减 = {best_combo['ic_decay']:.4f}")

        # 可视化
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            self.plot_importance(lgbm_result, title='LightGBM 因子重要性 (Gain)',
                                save_path=f'{save_dir}/lgbm_importance.png')
            self.plot_importance(xgb_result, title='XGBoost 因子重要性 (Gain)',
                                save_path=f'{save_dir}/xgb_importance.png')
            self.plot_importance(perm_result, title='置换重要性',
                                importance_col='importance_mean',
                                save_path=f'{save_dir}/perm_importance.png')
            if stepwise_result:
                self.plot_stepwise(stepwise_result,
                                   save_path=f'{save_dir}/stepwise_selection.png')
            print(f"\n  所有图表已保存到 {save_dir}/")

        report = {
            'X': X,
            'y': y,
            'lgbm_importance': lgbm_result,
            'xgb_importance': xgb_result,
            'perm_importance': perm_result,
            'stepwise_history': stepwise_result,
            'combination_results': combo_result,
            'comprehensive_rank': merged,
        }
        return report


# ──────────────────────────────────────────────
# 模拟数据演示
# ──────────────────────────────────────────────

def _generate_mock_data(
    n_dates: int = 500,
    n_symbols: int = 50,
    n_factors: int = 12,
    seed: int = 42,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """生成模拟因子和收益数据，用于演示

    构造方式:
    - 前 3 个因子对收益有真实预测力 (alpha 因子)
    - 中间 4 个因子有弱预测力
    - 后 5 个因子是纯噪声
    """
    np.random.seed(seed)
    dates = pd.bdate_range('2020-01-01', periods=n_dates, freq='B')
    symbols = [f'SYM_{i:03d}' for i in range(n_symbols)]

    # 模拟因子名称（参照 signals_daily.py）
    factor_names = [
        'momentum_5d', 'volume_surge', 'vwap_deviation',      # 强 alpha
        'realized_vol_20d', 'dollar_volume_rank',              # 弱 alpha
        'volume_price_trend', 'amihud_illiquidity',            # 弱 alpha
        'momentum_reversal_5d', 'overnight_gap',               # 噪声
        'high_low_spread', 'vol_breakout', 'trade_intensity',  # 噪声
    ][:n_factors]

    # 生成因子值
    factors_dict = {}
    factor_matrices = {}
    for i, name in enumerate(factor_names):
        mat = np.random.randn(n_dates, n_symbols) * 0.5
        # 加入一些自相关使其更像真实因子
        for t in range(1, n_dates):
            mat[t] = 0.7 * mat[t - 1] + 0.3 * mat[t]
        df = pd.DataFrame(mat, index=dates, columns=symbols)
        factors_dict[name] = df
        factor_matrices[name] = mat

    # 生成收益: 部分因子有预测力
    returns_mat = np.zeros((n_dates, n_symbols))
    # 强 alpha 因子 (前3个)
    for i in range(min(3, n_factors)):
        weight = 0.003 * (3 - i)  # 递减权重
        returns_mat += weight * factor_matrices[factor_names[i]]
    # 弱 alpha 因子 (3-6)
    for i in range(3, min(7, n_factors)):
        returns_mat += 0.0005 * factor_matrices[factor_names[i]]
    # 加噪声
    returns_mat += np.random.randn(n_dates, n_symbols) * 0.02

    returns = pd.DataFrame(returns_mat, index=dates, columns=symbols)

    return factors_dict, returns


if __name__ == '__main__':
    print("因子重要性分析 — 模拟数据演示")
    print("-" * 50)

    # 生成模拟数据
    factors_dict, returns = _generate_mock_data(
        n_dates=500, n_symbols=50, n_factors=12
    )
    print(f"模拟数据: {len(returns)} 天 × {len(returns.columns)} 只股票 × {len(factors_dict)} 因子")
    print(f"因子列表: {list(factors_dict.keys())}")
    print()

    # 运行完整报告
    selector = FactorSelector(train_ratio=0.7, random_state=42)
    report = selector.generate_report(
        factors_dict=factors_dict,
        returns=returns,
        forward_period=5,
        top_n=8,
        max_factors=6,
    )

    print("\n完成! 预期结果: momentum_5d, volume_surge, vwap_deviation 排名最高")
