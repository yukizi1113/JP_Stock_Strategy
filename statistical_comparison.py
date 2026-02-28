"""
statistical_comparison.py ― 全戦略の統計的優位性分析

機能:
  - 各戦略の月次超過リターン vs TOPIX に対するt検定
    (H0: mean_excess_return = 0, 有意水準 1%/5%/10%)
  - Sharpe ratio（年率、Rf=0.1%）
  - 情報比率 (Information Ratio)
  - 最大ドローダウン (MDD)
  - カルマー比率 (Calmar Ratio)
  - TOPIX beta / alpha（Jensen's alpha）
  - ブートストラップ 95% 信頼区間（1000 draws）
  - 月次リターン相関マトリクス（戦略間）
  - ローリング Sharpe比の時系列推移

使い方:
  python statistical_comparison.py
  python statistical_comparison.py --results-dir results/
  python statistical_comparison.py --benchmark-ticker ^N225

出力:
  results/statistical_summary.csv  ― 統計サマリーCSV
  results/comparison_chart.png     ― 比較チャート（4パネル）
  results/correlation_matrix.png   ― 月次リターン相関マトリクス
  results/rolling_sharpe.png       ― ローリングSharpe推移
"""
import os, sys, json, glob, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats as scipy_stats

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from data_loader import fetch_benchmark, performance_stats


# ─────────────────────────────────────────────────────────────────────────────
# 日本語フォント設定
# ─────────────────────────────────────────────────────────────────────────────

def setup_japanese_font():
    for font in ["Yu Gothic", "Meiryo", "MS Gothic", "IPAGothic"]:
        try:
            fm.findfont(fm.FontProperties(family=font))
            plt.rcParams["font.family"] = font
            return
        except Exception:
            pass

setup_japanese_font()


# ─────────────────────────────────────────────────────────────────────────────
# 統計計算
# ─────────────────────────────────────────────────────────────────────────────

def load_portfolio_returns(results_dir: str) -> dict:
    """
    results/ ディレクトリから各戦略のポートフォリオCSVを読み込み、
    日次リターン Series の辞書を返す。
    """
    portfolio_returns = {}
    csv_files = glob.glob(os.path.join(results_dir, "*_portfolio.csv"))

    for fpath in sorted(csv_files):
        fname = os.path.basename(fpath)
        try:
            df = pd.read_csv(fpath, index_col=0, parse_dates=True)
            if df.empty or df.shape[1] < 1:
                continue
            # 1列目がポートフォリオ価値
            port_val = df.iloc[:, 0].dropna()
            if len(port_val) < 50:
                continue
            daily_ret = port_val.pct_change().dropna()
            # 戦略名を抽出（ファイル名から）
            strat_name = fname.replace("_portfolio.csv", "").replace("_", " ")
            portfolio_returns[strat_name] = daily_ret
        except Exception as e:
            print(f"読み込みエラー {fname}: {e}")

    return portfolio_returns


def compute_extended_stats(
    daily_ret: pd.Series,
    benchmark_ret: pd.Series,
    rf_daily: float = 0.001 / 252,
    n_bootstrap: int = 1000,
) -> dict:
    """
    拡張統計指標を計算する。

    Parameters
    ----------
    daily_ret      : 日次リターン系列
    benchmark_ret  : ベンチマーク日次リターン（^N225 or TOPIX）
    rf_daily       : 無リスク金利（日次、デフォルト: 0.1%/年）
    n_bootstrap    : ブートストラップ繰り返し数

    Returns
    -------
    dict: 各種統計指標
    """
    r = daily_ret.dropna()
    n = len(r)
    if n < 20:
        return {}

    ann_factor = 252

    # ── 基本指標 ────────────────────────────────────────────────────────────
    ann_ret  = (1 + r).prod() ** (ann_factor / n) - 1
    ann_vol  = r.std() * np.sqrt(ann_factor)
    sharpe   = (ann_ret - rf_daily * ann_factor) / ann_vol if ann_vol > 0 else np.nan
    cum      = (1 + r).cumprod()
    drawdown = cum / cum.cummax() - 1
    max_dd   = drawdown.min()
    calmar   = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    stats = {
        "ann_return":   round(ann_ret, 4),
        "ann_vol":      round(ann_vol, 4),
        "sharpe":       round(sharpe,  4),
        "max_drawdown": round(max_dd,  4),
        "calmar":       round(calmar,  4) if not np.isnan(calmar) else np.nan,
    }

    # ── ベンチマーク対比 ─────────────────────────────────────────────────────
    b = benchmark_ret.reindex(r.index).dropna()
    r_aligned = r.reindex(b.index).dropna()
    b_aligned = b.reindex(r_aligned.index)

    if len(b_aligned) >= 30:
        # Beta / Alpha（Jensen）
        cov_mat  = np.cov(r_aligned.values, b_aligned.values)
        beta     = cov_mat[0, 1] / np.var(b_aligned.values)
        b_ann    = (1 + b_aligned.mean()) ** ann_factor - 1
        alpha    = ann_ret - (rf_daily * ann_factor) - beta * (b_ann - rf_daily * ann_factor)
        stats["beta"]  = round(float(beta),  4)
        stats["alpha"] = round(float(alpha), 4)

        # 月次超過リターン t 検定
        port_monthly  = (1 + r_aligned).resample("ME").prod() - 1
        bench_monthly = (1 + b_aligned).resample("ME").prod() - 1
        excess_monthly = (port_monthly - bench_monthly.reindex(port_monthly.index)).dropna()

        if len(excess_monthly) >= 6:
            t_stat, p_val = scipy_stats.ttest_1samp(excess_monthly.values, 0.0)
            n_months = len(excess_monthly)
            tracking_err = excess_monthly.std() * np.sqrt(12)
            ann_excess   = excess_monthly.mean() * 12
            ir = ann_excess / tracking_err if tracking_err > 0 else np.nan

            stats.update({
                "n_months":          n_months,
                "ann_excess_return": round(ann_excess, 4),
                "t_stat_vs_bench":   round(float(t_stat), 4),
                "p_val_vs_bench":    round(float(p_val),  4),
                "information_ratio": round(ir, 4) if not np.isnan(ir) else np.nan,
            })

            # 有意性フラグ
            if p_val < 0.01:
                stats["significance"] = "*** (1%)"
            elif p_val < 0.05:
                stats["significance"] = "** (5%)"
            elif p_val < 0.10:
                stats["significance"] = "* (10%)"
            else:
                stats["significance"] = "ns"

            # ブートストラップ 95% CI（月次超過リターンの年率換算）
            rng = np.random.default_rng(42)
            boot_means = [
                rng.choice(excess_monthly.values, size=n_months, replace=True).mean()
                for _ in range(n_bootstrap)
            ]
            stats["boot_ci_lo_ann"] = round(float(np.percentile(boot_means, 2.5))  * 12, 4)
            stats["boot_ci_hi_ann"] = round(float(np.percentile(boot_means, 97.5)) * 12, 4)

    # ── 高次モーメント ───────────────────────────────────────────────────────
    from scipy.stats import skew, kurtosis
    stats["skewness"] = round(float(skew(r.values)),     4)
    stats["kurtosis"] = round(float(kurtosis(r.values)), 4)  # excess kurtosis

    # ── 勝率 ─────────────────────────────────────────────────────────────────
    stats["win_rate_daily"]   = round(float((r > 0).mean()), 4)
    monthly = (1 + r).resample("ME").prod() - 1
    stats["win_rate_monthly"] = round(float((monthly > 0).mean()), 4)

    return stats


def compute_rolling_sharpe(daily_ret: pd.Series,
                             window_months: int = 12,
                             rf_daily: float = 0.001 / 252) -> pd.Series:
    """月次ローリング Sharpe 比を計算"""
    window_days = window_months * 21  # 1ヶ月≒21営業日
    rolling_sharpe = (
        (daily_ret.rolling(window_days).mean() - rf_daily)
        / daily_ret.rolling(window_days).std()
        * np.sqrt(252)
    )
    return rolling_sharpe


# ─────────────────────────────────────────────────────────────────────────────
# 可視化
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_chart(
    portfolio_returns: dict,
    benchmark_ret: pd.Series,
    summary_df: pd.DataFrame,
    save_path: str,
) -> None:
    """4パネルの比較チャートを描画"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(portfolio_returns), 1)))

    # ── Panel 1: 累積リターン ────────────────────────────────────────────────
    ax = axes[0, 0]
    bench_cum = (1 + benchmark_ret).cumprod()
    bench_cum = bench_cum / bench_cum.iloc[0]
    bench_cum.plot(ax=ax, label="日経225", color="black", linewidth=2.5,
                   linestyle="--", zorder=10)
    for i, (name, ret) in enumerate(sorted(portfolio_returns.items())):
        cum = (1 + ret).cumprod()
        cum = cum / cum.iloc[0]
        cum.plot(ax=ax, label=name[:20], color=colors[i], linewidth=1.5, alpha=0.85)
    ax.set_title("累積リターン比較", fontsize=12, fontweight="bold")
    ax.set_ylabel("累積倍率 (初期=1.0)")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)

    # ── Panel 2: Sharpe vs TOPIX超過リターン 散布図 ─────────────────────────
    ax2 = axes[0, 1]
    if not summary_df.empty and "sharpe" in summary_df.columns:
        for i, row in summary_df.iterrows():
            sharpe = row.get("sharpe", np.nan)
            excess = row.get("ann_excess_return", np.nan)
            if pd.isna(sharpe) or pd.isna(excess):
                continue
            sig = row.get("significance", "ns")
            marker = "^" if "***" in str(sig) else ("o" if "**" in str(sig) else "s")
            ax2.scatter(sharpe, excess * 100, s=100, marker=marker,
                        color=colors[i % len(colors)], zorder=5)
            ax2.annotate(row.get("strategy_name", f"S{i}")[:12],
                         (sharpe, excess * 100),
                         textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax2.axhline(0, color="red", linestyle="--", alpha=0.5, label="超過リターン=0")
    ax2.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Sharpe比 (年率)")
    ax2.set_ylabel("TOPIX超過リターン (年率%)")
    ax2.set_title("Sharpe比 vs TOPIX超過リターン", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: ドローダウン比較 ────────────────────────────────────────────
    ax3 = axes[1, 0]
    for i, (name, ret) in enumerate(sorted(portfolio_returns.items())):
        cum = (1 + ret).cumprod()
        dd = (cum / cum.cummax() - 1) * 100
        dd.plot(ax=ax3, label=name[:15], color=colors[i], linewidth=1.0, alpha=0.7)
    ax3.set_title("ドローダウン推移 (%)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("ドローダウン (%)")
    ax3.legend(fontsize=7, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.fill_between(ax3.get_lines()[0].get_xdata() if ax3.get_lines() else [],
                     [], alpha=0)  # dummy

    # ── Panel 4: 統計サマリーバーチャート ───────────────────────────────────
    ax4 = axes[1, 1]
    if not summary_df.empty and "sharpe" in summary_df.columns:
        names = [str(r.get("strategy_name", f"S{i}"))[:15]
                 for i, r in summary_df.iterrows()]
        sharpes = [r.get("sharpe", 0) for _, r in summary_df.iterrows()]
        bar_colors = ["steelblue" if s >= 0 else "tomato" for s in sharpes]
        bars = ax4.barh(names, sharpes, color=bar_colors)
        ax4.set_xlabel("Sharpe比 (年率)")
        ax4.set_title("戦略別 Sharpe比", fontsize=12, fontweight="bold")
        ax4.axvline(0, color="black", linewidth=0.8)
        ax4.grid(True, alpha=0.3, axis="x")
        for bar, val in zip(bars, sharpes):
            ax4.text(val + 0.02 if val >= 0 else val - 0.02,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", ha="left" if val >= 0 else "right",
                     fontsize=8)

    plt.suptitle("JP Stock Strategy ― バックテスト統計比較 (2018-2024)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"比較チャート保存: {save_path}")


def plot_correlation_matrix(portfolio_returns: dict, save_path: str) -> None:
    """月次リターンの相関マトリクスを描画"""
    if not portfolio_returns:
        return

    monthly_dict = {}
    for name, ret in portfolio_returns.items():
        monthly = (1 + ret).resample("ME").prod() - 1
        monthly_dict[name[:20]] = monthly

    monthly_df = pd.DataFrame(monthly_dict).dropna(how="all")
    if monthly_df.shape[1] < 2:
        return

    corr = monthly_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, ax=ax, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("戦略間 月次リターン相関マトリクス", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"相関マトリクス保存: {save_path}")


def plot_rolling_sharpe(portfolio_returns: dict, save_path: str,
                         window_months: int = 12) -> None:
    """ローリング Sharpe 比の時系列推移を描画"""
    if not portfolio_returns:
        return

    fig, ax = plt.subplots(figsize=(16, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(portfolio_returns), 1)))

    for i, (name, ret) in enumerate(sorted(portfolio_returns.items())):
        rs = compute_rolling_sharpe(ret, window_months=window_months)
        rs.plot(ax=ax, label=name[:20], color=colors[i], linewidth=1.2, alpha=0.8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.axhline(0.5, color="green", linewidth=0.8, linestyle="--", alpha=0.5, label="Sharpe=0.5")
    ax.set_title(f"ローリング Sharpe 比 ({window_months}ヶ月ウィンドウ)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Sharpe 比 (年率)")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ローリングSharpe保存: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="全戦略 統計的優位性分析")
    parser.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    parser.add_argument("--benchmark-ticker", default="^N225")
    parser.add_argument("--start",  default="2018-01-01")
    parser.add_argument("--end",    default="2024-12-31")
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"results ディレクトリが存在しません: {args.results_dir}")
        print("先に master_backtest.py を実行してください。")
        return

    print("ベンチマーク取得中...")
    try:
        bench = fetch_benchmark(args.start, args.end, args.benchmark_ticker)
        bench_ret = bench.pct_change().dropna()
        print(f"ベンチマーク ({args.benchmark_ticker}): {len(bench_ret)}日分")
    except Exception as e:
        print(f"ベンチマーク取得失敗: {e}")
        bench_ret = pd.Series(dtype=float)

    # ── JSONファイルからも stats を読み込む ────────────────────────────────
    json_stats = {}
    for fpath in glob.glob(os.path.join(args.results_dir, "*_stats.json")):
        if "master" in os.path.basename(fpath):
            continue
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            name = os.path.basename(fpath).replace("_stats.json", "").replace("_", " ")
            json_stats[name] = data
        except Exception:
            pass

    # ── ポートフォリオリターンを読み込んで統計を再計算 ─────────────────────
    portfolio_returns = load_portfolio_returns(args.results_dir)

    if not portfolio_returns:
        print("ポートフォリオCSVが見つかりません。master_backtest.py を先に実行してください。")
        if json_stats:
            print(f"JSONファイルから {len(json_stats)} 戦略分の統計を読み込みました。")
        return

    print(f"\n{len(portfolio_returns)} 戦略分のデータを分析中...")

    # 詳細統計計算
    summary_rows = []
    for name, ret in sorted(portfolio_returns.items()):
        print(f"  分析中: {name}")
        ext_stats = compute_extended_stats(
            ret, bench_ret, n_bootstrap=args.bootstrap_n
        )
        ext_stats["strategy_name"] = name
        summary_rows.append(ext_stats)

    # ベンチマーク自体の統計も追加
    if not bench_ret.empty:
        b_stats = compute_extended_stats(bench_ret, bench_ret, n_bootstrap=args.bootstrap_n)
        b_stats["strategy_name"] = f"ベンチマーク ({args.benchmark_ticker})"
        b_stats["significance"] = "基準"
        summary_rows.insert(0, b_stats)

    summary_df = pd.DataFrame(summary_rows)

    # ── コンソール出力 ───────────────────────────────────────────────────────
    print("\n" + "="*100)
    print("  統計的優位性分析サマリー")
    print("="*100)

    display_cols = [
        "strategy_name", "ann_return", "ann_vol", "sharpe", "max_drawdown",
        "calmar", "ann_excess_return", "t_stat_vs_bench", "p_val_vs_bench",
        "information_ratio", "significance", "boot_ci_lo_ann", "boot_ci_hi_ann",
    ]
    available_cols = [c for c in display_cols if c in summary_df.columns]

    # フォーマット整形
    display_df = summary_df[available_cols].copy()
    fmt_pct = ["ann_return", "ann_vol", "max_drawdown", "ann_excess_return",
               "boot_ci_lo_ann", "boot_ci_hi_ann"]
    for col in fmt_pct:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.1%}" if pd.notnull(x) and isinstance(x, float) else x
            )
    fmt_3f = ["sharpe", "calmar", "t_stat_vs_bench", "p_val_vs_bench", "information_ratio"]
    for col in fmt_3f:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.3f}" if pd.notnull(x) and isinstance(x, float) else x
            )

    print(display_df.to_string(index=False))

    # ── CSV 保存 ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.results_dir, "statistical_summary.csv")
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n統計サマリーCSV保存: {csv_path}")

    # ── グラフ生成 ───────────────────────────────────────────────────────────
    try:
        plot_comparison_chart(
            portfolio_returns, bench_ret, summary_df,
            save_path=os.path.join(args.results_dir, "comparison_chart.png"),
        )
    except Exception as e:
        print(f"比較チャートエラー: {e}")

    try:
        plot_correlation_matrix(
            portfolio_returns,
            save_path=os.path.join(args.results_dir, "correlation_matrix.png"),
        )
    except Exception as e:
        print(f"相関マトリクスエラー: {e}")

    try:
        plot_rolling_sharpe(
            portfolio_returns,
            save_path=os.path.join(args.results_dir, "rolling_sharpe.png"),
        )
    except Exception as e:
        print(f"ローリングSharpeエラー: {e}")

    # ── 解釈のコメント ───────────────────────────────────────────────────────
    print("\n【解釈の注意】")
    print("  - サンプル数は約84ヶ月（2018-2024）。統計的検出力は限定的。")
    print("  - p < 0.05 でも in-sample の結果。将来パフォーマンスを保証しない。")
    print("  - ブートストラップCIはデータの時系列依存性を考慮していない点に注意。")
    print("  - 取引コスト・スリッページは一部のみ考慮（0.1%片道）。")


if __name__ == "__main__":
    main()
