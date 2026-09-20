"""Framework entry point - swap strategies with a single line."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import warnings; warnings.filterwarnings('ignore')

from qf.data import prepare_data
from qf.backtest import run_event_driven, BacktestResult
from qf.attribution import AttributionAnalyzer
from qf.risk import RiskAnalyzer, GateCheck
from qf.stress import StressTest
from qf.viz import generate_full_report


class Framework:
    """
    Quant research framework - switching strategies takes one line

    Usage:
        from qf import Framework
        from strategies.momentum import MomentumStrategy

        fw = Framework()
        fw.run(MomentumStrategy())
    """

    def __init__(self, start='2010-01-01', end='2024-12-31'):
        self.start = start
        self.end = end
        self.data = None

    def _load_data(self):
        if self.data is None:
            self.data = prepare_data(self.start, self.end)
        return self.data

    def run(self, strategy, visualize=True, stress_test=True, gate_check=True):
        """Run the full pipeline: backtest + attribution + risk control + stress test + visualization."""
        d = self._load_data()
        params = strategy.get_params()

        print("=" * 60)
        print(f"  {params['name']}")
        print("=" * 60)

        # 1. 生成信号
        print("\n[1] 生成信号...")
        signal = strategy.generate_signal(d)
        print(f"  信号形状: {signal.shape}")

        # 2. 回测
        print("\n[2] 运行回测...")
        result, metrics = run_event_driven(
            d, signal,
            long_n=params['long_n'], short_n=params['short_n'],
            long_pct=params['long_pct'], short_pct=params['short_pct'],
            weight_mode=params['weight_mode'],
            turnover_penalty=params['turnover_penalty'],
            cost_model='sqrt', commission_bps=1.0, spread_bps=5.0,
            impact_coeff=0.3, short_borrow_bps=30.0,
        )

        # 打印核心指标
        m = metrics
        print(f"\n  {'CAGR':<12} {m['cagr']:.1%}")
        print(f"  {'Sharpe':<12} {m['sharpe']:.2f}")
        print(f"  {'Sortino':<12} {m['sortino']:.2f}")
        print(f"  {'MaxDD':<12} {m['max_drawdown']:.1%}")
        print(f"  {'Win Rate':<12} {m['win_rate']:.1%}")
        print(f"  {'Alpha':<12} {m.get('alpha', 0):.1%}")
        print(f"  {'Final':<12} ${m['final_value']:,.0f}")

        # 3. 因子归因
        print("\n[3] 因子归因...")
        analyzer = AttributionAnalyzer(d['ff5'])
        attr = analyzer.ff5_regression(result.returns, params['name'])
        yearly = analyzer.yearly_attribution(result.returns, d['spy_ret'])
        print(f"  Alpha={attr['alpha']:.2%}  R2={attr['r2']:.3f}")

        # 4. 门控检查
        gate_summary = None
        if gate_check:
            print("\n[4] 门控检查...")
            risk = RiskAnalyzer(result.returns, result.pv)
            gc = GateCheck()

            # IS/OOS
            is_rets = result.returns[result.returns.index <= '2017-12-31']
            oos_rets = result.returns[result.returns.index > '2017-12-31']
            is_sr = is_rets.mean() / is_rets.std() * 12**0.5 if is_rets.std() > 0 else 0
            oos_sr = oos_rets.mean() / oos_rets.std() * 12**0.5 if oos_rets.std() > 0 else 0

            gate_summary = gc.run_standard_gates(m, risk, is_sr, oos_sr)
            print(f"  通过: {gate_summary['n_pass']}/{gate_summary['n_total']}")

        # 5. 压力测试
        crisis_df = cost_df = None
        if stress_test:
            print("\n[5] 压力测试...")
            st = StressTest(d, signal)
            crisis_df = st.crisis_replay()
            n_beat = (crisis_df['excess'] > 0).sum()
            print(f"  危机回放: {n_beat}/{len(crisis_df)} 跑赢基准")

            cost_df = st.cost_sensitivity()
            min_sr = cost_df['sharpe'].min()
            print(f"  成本敏感度: 最差Sharpe={min_sr:.2f}")

        # 6. 可视化
        if visualize:
            print("\n[6] 生成可视化...")
            generate_full_report(
                result, d,
                gate_summary=gate_summary,
                crisis_df=crisis_df,
                cost_df=cost_df,
                yearly_df=yearly,
                output_dir='reports'
            )

        return {
            'result': result, 'metrics': metrics, 'attribution': attr,
            'yearly': yearly, 'gate_summary': gate_summary,
            'crisis': crisis_df, 'cost': cost_df,
        }
