"""可视化模块 — 回测/风控/压力测试/门控 图表"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# 全局风格
plt.rcParams.update({
    'figure.figsize': (12, 6), 'figure.dpi': 120, 'figure.facecolor': 'white',
    'axes.grid': True, 'axes.spines.top': False, 'axes.spines.right': False,
    'grid.alpha': 0.3, 'font.size': 10,
})
COLORS = {'strategy': '#2196F3', 'benchmark': '#9E9E9E', 'positive': '#4CAF50',
          'negative': '#F44336', 'neutral': '#FF9800', 'accent': '#9C27B0'}


def plot_equity_curve(result, benchmark_returns=None, title="Equity Curve", save_path=None):
    """净值曲线 + 基准对比"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True)

    # 净值
    norm_pv = result.pv / result.pv.iloc[0]
    ax1.plot(norm_pv.index, norm_pv, color=COLORS['strategy'], linewidth=2, label='Strategy')

    if benchmark_returns is not None:
        bm = benchmark_returns.copy()
        bm.index = pd.to_datetime(bm.index.to_timestamp()) if hasattr(bm.index, 'to_timestamp') else bm.index
        bm_pv = (1 + bm).cumprod()
        bm_pv = bm_pv.reindex(norm_pv.index, method='nearest')
        ax1.plot(bm_pv.index, bm_pv / bm_pv.iloc[0], color=COLORS['benchmark'],
                linewidth=1.5, linestyle='--', label='SPY', alpha=0.7)

    ax1.set_ylabel('Cumulative Return')
    ax1.legend(loc='upper left')
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:.1f}x'))

    # 回撤
    dd = (result.pv - result.pv.cummax()) / result.pv.cummax()
    ax2.fill_between(dd.index, dd, 0, color=COLORS['negative'], alpha=0.4)
    ax2.set_ylabel('Drawdown')
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()
    return fig


def plot_monthly_heatmap(result, title="Monthly Returns", save_path=None):
    """月度收益热力图"""
    rets = result.returns.copy()
    rets.index = pd.to_datetime(rets.index)
    df = pd.DataFrame({'year': rets.index.year, 'month': rets.index.month, 'ret': rets.values})
    pivot = df.pivot_table(index='year', columns='month', values='ret', aggfunc='first')
    pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.5)))
    im = ax.imshow(pivot.values * 100, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)

    ax.set_xticks(range(12)); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot))); ax.set_yticklabels(pivot.index)

    for i in range(len(pivot)):
        for j in range(12):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f'{val*100:.1f}', ha='center', va='center',
                       fontsize=8, color='black' if abs(val) < 0.05 else 'white')

    plt.colorbar(im, ax=ax, label='Return (%)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return fig


def plot_return_distribution(returns, title="Return Distribution", save_path=None):
    """收益分布 + VaR/CVaR标注"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(returns * 100, bins=30, color=COLORS['strategy'], alpha=0.6, edgecolor='white')

    var95 = returns.quantile(0.05) * 100
    cvar95 = returns[returns <= returns.quantile(0.05)].mean() * 100
    ax.axvline(var95, color=COLORS['negative'], linestyle='--', linewidth=2, label=f'VaR(95%) = {var95:.1f}%')
    ax.axvline(cvar95, color=COLORS['negative'], linestyle=':', linewidth=2, label=f'CVaR(95%) = {cvar95:.1f}%')
    ax.axvline(returns.mean() * 100, color=COLORS['positive'], linewidth=2, label=f'Mean = {returns.mean()*100:.1f}%')

    ax.set_xlabel('Monthly Return (%)')
    ax.set_ylabel('Frequency')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return fig


def plot_rolling_sharpe(returns, window=12, title="Rolling Sharpe", save_path=None):
    """滚动Sharpe"""
    rolling_sr = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(12)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(rolling_sr.index, rolling_sr, color=COLORS['strategy'], linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axhline(1, color=COLORS['positive'], linestyle='--', alpha=0.5, label='Sharpe=1')
    ax.axhline(2, color=COLORS['accent'], linestyle='--', alpha=0.5, label='Sharpe=2')
    ax.fill_between(rolling_sr.index, rolling_sr, 0,
                    where=rolling_sr > 0, color=COLORS['positive'], alpha=0.1)
    ax.fill_between(rolling_sr.index, rolling_sr, 0,
                    where=rolling_sr < 0, color=COLORS['negative'], alpha=0.1)
    ax.set_ylabel(f'{window}M Rolling Sharpe')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return fig


def plot_yearly_attribution(yearly_df, title="Yearly Attribution", save_path=None):
    """年度收益对比柱状图"""
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(yearly_df))
    w = 0.35
    ax.bar(x - w/2, yearly_df['strategy'] * 100, w, color=COLORS['strategy'], label='Strategy')
    ax.bar(x + w/2, yearly_df['benchmark'] * 100, w, color=COLORS['benchmark'], label='Benchmark')

    for i, exc in enumerate(yearly_df['excess']):
        color = COLORS['positive'] if exc > 0 else COLORS['negative']
        ax.annotate(f'{exc*100:+.1f}', (x[i], max(yearly_df['strategy'].iloc[i], yearly_df['benchmark'].iloc[i]) * 100 + 2),
                   ha='center', fontsize=7, color=color)

    ax.set_xticks(x); ax.set_xticklabels(yearly_df.index)
    ax.set_ylabel('Return (%)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.axhline(0, color='black', linewidth=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return fig


def plot_gate_dashboard(gate_summary, title="Gate Check Dashboard", save_path=None):
    """门控检查仪表盘 (交通灯)"""
    details = gate_summary['details']
    n = len(details)
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.45)))

    for i, g in enumerate(reversed(details)):
        color = COLORS['positive'] if g['passed'] else COLORS['negative']
        ax.barh(i, 1, color=color, alpha=0.8, height=0.7)

        val = g['value']
        val_str = f'{val:.2f}' if isinstance(val, float) else str(val)
        thr = g['threshold']
        thr_str = f'{thr:.2f}' if isinstance(thr, float) else str(thr)
        label = f"{g['name']}  |  {val_str} {g['comparison']} {thr_str}"
        status = "PASS" if g['passed'] else "FAIL"

        ax.text(0.02, i, label, va='center', fontsize=9, fontweight='bold', color='white')
        ax.text(0.95, i, status, va='center', ha='right', fontsize=9,
               fontweight='bold', color='white')

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(f"{title}  ({gate_summary['n_pass']}/{gate_summary['n_total']})",
                fontsize=14, fontweight='bold')
    ax.spines['left'].set_visible(False); ax.spines['bottom'].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return fig


def plot_stress_comparison(crisis_df, title="Crisis Replay", save_path=None):
    """危机回放对比图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(crisis_df))
    w = 0.35
    ax.bar(x - w/2, crisis_df['strategy'] * 100, w, color=COLORS['strategy'], label='Strategy')
    ax.bar(x + w/2, crisis_df['benchmark'] * 100, w, color=COLORS['benchmark'], label='Benchmark')
    ax.set_xticks(x)
    ax.set_xticklabels(crisis_df['crisis'], rotation=30, ha='right')
    ax.set_ylabel('Return (%)')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return fig


def plot_cost_sensitivity(cost_df, title="Cost Sensitivity", save_path=None):
    """成本敏感度曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(cost_df['cost_bps'], cost_df['cagr'] * 100, 'o-', color=COLORS['strategy'], linewidth=2)
    ax1.set_xlabel('Transaction Cost (bps)'); ax1.set_ylabel('CAGR (%)')
    ax1.set_title('CAGR vs Cost')
    ax1.axhline(0, color=COLORS['negative'], linestyle='--', alpha=0.5)

    ax2.plot(cost_df['cost_bps'], cost_df['sharpe'], 's-', color=COLORS['accent'], linewidth=2)
    ax2.set_xlabel('Transaction Cost (bps)'); ax2.set_ylabel('Sharpe')
    ax2.set_title('Sharpe vs Cost')
    ax2.axhline(1.0, color=COLORS['neutral'], linestyle='--', alpha=0.5, label='Sharpe=1')
    ax2.legend()

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return fig


def generate_full_report(result, d, gate_summary=None, crisis_df=None,
                          cost_df=None, yearly_df=None, output_dir='reports'):
    """一键生成全套可视化报告"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  生成可视化报告 -> {output_dir}/")
    plot_equity_curve(result, d.get('spy_ret'), save_path=f'{output_dir}/01_equity_curve.png')
    plot_monthly_heatmap(result, save_path=f'{output_dir}/02_monthly_heatmap.png')
    plot_return_distribution(result.returns, save_path=f'{output_dir}/03_return_dist.png')
    plot_rolling_sharpe(result.returns, save_path=f'{output_dir}/04_rolling_sharpe.png')

    if yearly_df is not None:
        plot_yearly_attribution(yearly_df, save_path=f'{output_dir}/05_yearly_attr.png')
    if gate_summary is not None:
        plot_gate_dashboard(gate_summary, save_path=f'{output_dir}/06_gate_check.png')
    if crisis_df is not None:
        plot_stress_comparison(crisis_df, save_path=f'{output_dir}/07_crisis_replay.png')
    if cost_df is not None:
        plot_cost_sensitivity(cost_df, save_path=f'{output_dir}/08_cost_sensitivity.png')

    print(f"  完成: {len(os.listdir(output_dir))} 张图表")
