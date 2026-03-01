"""
master_backtest.py ― 全9戦略 一括バックテスト実行スクリプト

機能:
  1. 全3800銘柄の価格データを一括取得・キャッシュ
  2. 全9戦略を順番に実行
  3. 結果を results/ フォルダに保存
  4. TOPIX比較チャートと統計サマリーを出力

使い方:
  python master_backtest.py
  python master_backtest.py --start 2018-01-01 --end 2024-12-31
  python master_backtest.py --strategies 1,2,3   # 特定戦略のみ
  python master_backtest.py --force-refresh       # キャッシュを無視して再取得
  python master_backtest.py --max-stocks 500      # 銘柄数を制限（テスト用）

所要時間の目安:
  価格取得（3800銘柄）: 30〜60分
  戦略1 (ML Momentum):         1〜3時間
  戦略2 (PCA Eigen):           15〜30分
  戦略3 (Mean Reversion):      30〜60分
  戦略4 (RL Portfolio):        30〜60分
  戦略5 (Absorption Ratio):    15〜30分
  戦略6 (Multi Factor ML):     2〜5時間
  戦略7 (Black-Litterman):     30〜60分
  戦略8 (ABCD Forecast):       15〜30分 ※別途マルチアセット取得
  戦略9 (Deep Portfolio DPO):  2〜4時間 ※PyTorch必要
"""
import os, sys, time, json, traceback, argparse, warnings
from datetime import datetime

# sklearn/yfinance の大量警告を抑制（ログ肥大化を防ぐ）
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 日本語フォント設定（Windowsの場合）
def setup_japanese_font():
    for font in ["Yu Gothic", "Meiryo", "MS Gothic", "IPAGothic", "Noto Sans CJK JP"]:
        try:
            fm.findfont(fm.FontProperties(family=font))
            plt.rcParams["font.family"] = font
            return
        except Exception:
            pass
    plt.rcParams["font.family"] = "sans-serif"

setup_japanese_font()

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from data_loader import (
    fetch_prices_cached,
    build_universe_from_edinet,
    fetch_benchmark,
)
from backtest_engine import Backtester


# ─────────────────────────────────────────────────────────────────────────────
# 戦略インポート（importlib で絶対パス指定・sys.path 汚染なし）
# ─────────────────────────────────────────────────────────────────────────────

import importlib.util as _importlib_util

def _load_module_from_file(module_name: str, file_path: str):
    """importlib でファイルを直接インポートする（sys.path 非汚染）"""
    spec = _importlib_util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"ファイルが見つかりません: {file_path}")
    mod = _importlib_util.module_from_spec(spec)
    # 戦略ファイルが同一ディレクトリの他ファイル(backtest_engine等)を
    # import できるよう、そのディレクトリをsys.pathに一時追加
    strat_dir = os.path.dirname(file_path)
    added = False
    if strat_dir not in sys.path:
        sys.path.insert(0, strat_dir)
        added = True
    try:
        spec.loader.exec_module(mod)
    finally:
        # strat_dir を削除し sys.path を元に戻す
        if added and strat_dir in sys.path:
            sys.path.remove(strat_dir)
    return mod

_STRAT_DIR_MAP = {
    1: "01_ml_momentum",
    2: "02_eigen_portfolio",
    3: "03_mean_reversion",
    4: "04_rl_portfolio",
    5: "05_absorption_ratio",
    6: "06_multi_factor_ml",
    7: "07_black_litterman",
    8: "08_abcd_forecast",
    9: "09_deep_portfolio",
}

# モジュールキャッシュ（同一戦略を複数回ロードしない）
_loaded_modules: dict = {}

def load_strategy(strategy_id: int, top_n: int = 20, **kwargs):
    """
    戦略インスタンスを生成して返す。

    importlib.util.spec_from_file_location を使い、絶対パスで各戦略の
    strategy.py を直接ロード。sys.path 汚染を避けることで
    複数戦略のロード時に名前衝突が発生しない。
    """
    if strategy_id not in _STRAT_DIR_MAP:
        return None

    strat_subdir = _STRAT_DIR_MAP[strategy_id]
    strat_file   = os.path.join(ROOT, "strategies", strat_subdir, "strategy.py")
    module_name  = f"strategy_{strategy_id:02d}"

    if module_name not in _loaded_modules:
        _loaded_modules[module_name] = _load_module_from_file(module_name, strat_file)

    mod = _loaded_modules[module_name]

    if strategy_id == 1:
        # n_jobs=1: 並列ツリー学習を無効化してRAM使用量を大幅削減
        # n_estimators=100, max_samples=0.6: さらにメモリ・速度を節約
        return mod.MLMomentumStrategy(top_n=top_n, rf_params={
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_leaf": 5,
            "random_state": 42,
            "n_jobs": 1,
            "max_samples": 0.6,
        })
    elif strategy_id == 2:
        return mod.EigenPortfolioStrategy(pc_index=1, top_n=top_n, window_months=36, long_only=True)
    elif strategy_id == 3:
        return mod.MeanReversionStrategy(hurst_threshold=0.45, entry_z=1.5, exit_z=0.3)
    elif strategy_id == 4:
        return mod.RLPortfolioStrategy(top_n=min(top_n, 15), train_episodes=300)
    elif strategy_id == 5:
        return mod.AbsorptionRatioStrategy(ar_window=252, ar_n_components=5, base_top_n=30)
    elif strategy_id == 6:
        return mod.MultiFactorMLStrategy(top_n=top_n, model_type="rf")
    elif strategy_id == 7:
        return mod.BlackLittermanStrategy(top_n=top_n, view_confidence=0.3)
    elif strategy_id == 8:
        return mod.ABCDForecastStrategy(n_matrices=10, top_n=5, long_only=False)
    elif strategy_id == 9:
        return mod.DeepPortfolioStrategy(
            top_n=top_n,
            H=128,
            C=20,
            L=4,
            Q=16,
            K=64,
            retrain_interval=6,
            max_train_stocks=500,
            n_epochs=8,
        )
    return None


STRATEGY_INFO = {
    1: {"name": "MLモメンタム",          "uses_jpx": True,  "max_stocks_override": 1500},
    2: {"name": "PCA固有ポートフォリオ",  "uses_jpx": True,  "max_stocks_override": None},
    3: {"name": "OU平均回帰",            "uses_jpx": True,  "max_stocks_override": None},
    4: {"name": "RL（Q学習）",           "uses_jpx": True,  "max_stocks_override": 100},
    5: {"name": "吸収比率タイミング",     "uses_jpx": True,  "max_stocks_override": None},
    6: {"name": "マルチファクターML",     "uses_jpx": True,  "max_stocks_override": 1000},
    7: {"name": "ブラック・リターマン",   "uses_jpx": True,  "max_stocks_override": None},
    8: {"name": "ABCD-Forecast",         "uses_jpx": False, "max_stocks_override": None},
    9: {"name": "Deep Portfolio (DPO)",   "uses_jpx": True,  "max_stocks_override": 800},
}


def run_all_backtests(
    start: str = "2018-01-01",
    end: str = "2024-12-31",
    strategy_ids: list = None,
    max_stocks: int = None,
    force_refresh: bool = False,
    results_dir: str = None,
    top_n: int = 20,
) -> dict:
    """
    全8戦略を順番に実行し、結果辞書を返す。

    Parameters
    ----------
    start, end      : バックテスト期間
    strategy_ids    : 実行する戦略IDリスト（None=全8戦略）
    max_stocks      : 最大銘柄数（None=全銘柄）
    force_refresh   : キャッシュを無視して再取得
    results_dir     : 結果保存ディレクトリ
    top_n           : 各戦略のポートフォリオ銘柄数

    Returns
    -------
    {strategy_id: {"stats": dict, "result": DataFrame, "error": str}}
    """
    if strategy_ids is None:
        strategy_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    results_dir = results_dir or os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    cache_path = os.path.join(ROOT, "data", "prices_cache.pkl")

    # ── Step 1: 日本株価格データ一括取得・キャッシュ ──────────────────────────
    jpx_prices = None
    jpx_strategies = [sid for sid in strategy_ids if STRATEGY_INFO[sid]["uses_jpx"]]
    if jpx_strategies:
        print("\n" + "="*60)
        print(" ユニバース & 価格データ取得")
        print("="*60)

        print("ユニバース取得中（EDINETデータベース）...")
        try:
            universe = build_universe_from_edinet()
            tickers = universe["ticker"].tolist()
            print(f"EDINETユニバース: {len(tickers)}銘柄")
        except Exception as e:
            print(f"EDINET失敗（{e}）→ JPXウェブから取得")
            from data_loader import build_jpx_universe
            universe = build_jpx_universe()
            tickers = universe["ticker"].tolist()

        if max_stocks:
            tickers = tickers[:max_stocks]

        print(f"\n全銘柄価格取得中（{len(tickers)}銘柄, {start}〜{end}）...")
        t0 = time.time()
        jpx_prices = fetch_prices_cached(
            tickers, start, end,
            cache_path=cache_path,
            force_refresh=force_refresh,
            batch_size=200,
        )
        jpx_prices = jpx_prices.dropna(axis=1, thresh=int(len(jpx_prices) * 0.5))
        print(f"価格取得完了: {jpx_prices.shape[1]}銘柄 / {len(jpx_prices)}日 "
              f"({time.time()-t0:.0f}秒)")

    # ── Step 2: マルチアセット価格取得（戦略8用）───────────────────────────────
    multi_prices = None
    if 8 in strategy_ids:
        print("\n戦略8用マルチアセット価格取得中...")
        try:
            strat8_dir = os.path.join(ROOT, "strategies", "08_abcd_forecast")
            if strat8_dir not in sys.path:
                sys.path.insert(0, strat8_dir)
            from strategy import fetch_multi_asset_prices
            multi_prices = fetch_multi_asset_prices(start, end)
            multi_prices = multi_prices.dropna(axis=1, thresh=int(len(multi_prices) * 0.3))
            print(f"マルチアセット取得完了: {multi_prices.shape[1]}アセット")
        except Exception as e:
            print(f"マルチアセット価格取得失敗: {e}")

    # ── Step 3: 各戦略のバックテスト実行 ─────────────────────────────────────
    all_results = {}

    for sid in strategy_ids:
        info = STRATEGY_INFO[sid]
        print(f"\n{'='*60}")
        print(f" 戦略{sid}: {info['name']}")
        print(f"{'='*60}")

        t_start = time.time()
        try:
            # 価格データ選択
            if not STRATEGY_INFO[sid]["uses_jpx"]:
                if multi_prices is None or multi_prices.empty:
                    raise ValueError("マルチアセット価格データが利用不可")
                prices_for_bt = multi_prices
            else:
                if jpx_prices is None or jpx_prices.empty:
                    raise ValueError("日本株価格データが利用不可")
                # RL戦略はmax_stocks=100に制限
                override = info["max_stocks_override"]
                if override and override < jpx_prices.shape[1]:
                    prices_for_bt = jpx_prices.iloc[:, :override]
                    print(f"注: RL戦略のためmax_stocks={override}に制限")
                else:
                    prices_for_bt = jpx_prices

            # 戦略インスタンス生成
            strategy = load_strategy(sid, top_n=top_n)
            if strategy is None:
                raise ValueError("戦略インスタンス生成失敗")

            # バックテスト実行
            bt = Backtester(prices_for_bt, strategy, rebal_freq="ME")
            result = bt.run()
            stats = bt.summary()

            elapsed = time.time() - t_start
            print(f"完了: {elapsed:.0f}秒")
            print(f"  年率リターン: {stats.get('ann_return', 'N/A'):.2%}")
            print(f"  Sharpe比:     {stats.get('sharpe', 'N/A'):.3f}")
            print(f"  最大DD:       {stats.get('max_drawdown', 'N/A'):.2%}")
            if "t_stat_vs_bench" in stats:
                sig = "**" if stats.get("p_val_vs_bench", 1) < 0.05 else ""
                print(f"  t統計量:      {stats['t_stat_vs_bench']:.3f} (p={stats['p_val_vs_bench']:.3f}){sig}")

            # 結果保存
            result.to_csv(os.path.join(results_dir, f"{sid:02d}_{info['name']}_portfolio.csv"))
            bt.plot(save_path=os.path.join(results_dir, f"{sid:02d}_{info['name']}_chart.png"))
            with open(os.path.join(results_dir, f"{sid:02d}_{info['name']}_stats.json"),
                      "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2, default=str)

            all_results[sid] = {"stats": stats, "result": result, "error": None}

        except Exception as e:
            elapsed = time.time() - t_start
            err_msg = traceback.format_exc()
            print(f"エラー ({elapsed:.0f}秒): {e}")
            print(err_msg[:500])
            all_results[sid] = {"stats": {}, "result": None, "error": str(e)}
            # エラーログ保存
            with open(os.path.join(results_dir, f"{sid:02d}_{info['name']}_error.log"),
                      "w", encoding="utf-8") as f:
                f.write(err_msg)

    return all_results


def print_summary_table(all_results: dict) -> pd.DataFrame:
    """全戦略の統計サマリーテーブルを出力・返す"""
    rows = []
    for sid, res in sorted(all_results.items()):
        info = STRATEGY_INFO[sid]
        s = res["stats"]
        err = res["error"]
        if err:
            rows.append({
                "戦略ID": sid,
                "戦略名": info["name"],
                "年率リターン": "ERROR",
                "年率ボラ": "-",
                "Sharpe": "-",
                "最大DD": "-",
                "TOPIX超過(年率)": "-",
                "t値": "-",
                "p値": "-",
                "情報比率": "-",
                "有意性": "ERROR",
            })
        else:
            ann_ret = s.get("ann_return", float("nan"))
            ann_vol = s.get("ann_vol", float("nan"))
            sharpe  = s.get("sharpe",   float("nan"))
            max_dd  = s.get("max_drawdown", float("nan"))
            exc_ret = s.get("ann_excess_return", float("nan"))
            t_stat  = s.get("t_stat_vs_bench", float("nan"))
            p_val   = s.get("p_val_vs_bench",  float("nan"))
            ir      = s.get("information_ratio", float("nan"))

            if not pd.isna(p_val):
                if p_val < 0.01:
                    sig = "*** (1%)"
                elif p_val < 0.05:
                    sig = "** (5%)"
                elif p_val < 0.10:
                    sig = "* (10%)"
                else:
                    sig = "ns"
            else:
                sig = "-"

            rows.append({
                "戦略ID": sid,
                "戦略名": info["name"],
                "年率リターン": f"{ann_ret:.1%}" if not pd.isna(ann_ret) else "-",
                "年率ボラ":     f"{ann_vol:.1%}" if not pd.isna(ann_vol) else "-",
                "Sharpe":       f"{sharpe:.3f}"  if not pd.isna(sharpe)  else "-",
                "最大DD":       f"{max_dd:.1%}"  if not pd.isna(max_dd)  else "-",
                "TOPIX超過(年率)": f"{exc_ret:+.1%}" if not pd.isna(exc_ret) else "-",
                "t値":          f"{t_stat:.3f}"  if not pd.isna(t_stat)  else "-",
                "p値":          f"{p_val:.3f}"   if not pd.isna(p_val)   else "-",
                "情報比率":     f"{ir:.3f}"      if not pd.isna(ir)      else "-",
                "有意性":       sig,
            })

    df = pd.DataFrame(rows)
    print("\n" + "="*80)
    print("  全戦略 パフォーマンスサマリー")
    print("="*80)
    print(df.to_string(index=False))
    return df


def plot_comparison_chart(all_results: dict, benchmark_data: pd.Series,
                          save_path: str) -> None:
    """全戦略の累積リターン比較チャートを描画"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                              gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results) + 1))

    # ベンチマーク（TOPIX代替: ^N225）
    if benchmark_data is not None and not benchmark_data.empty:
        bench_norm = benchmark_data / benchmark_data.iloc[0]
        bench_norm.plot(ax=ax, label="日経225 (ベンチマーク)", color="black",
                        linewidth=2.5, linestyle="--", zorder=10)

    for i, (sid, res) in enumerate(sorted(all_results.items())):
        if res["result"] is None:
            continue
        result = res["result"]
        info = STRATEGY_INFO[sid]
        port_norm = result.iloc[:, 0] / result.iloc[0, 0]
        port_norm.plot(ax=ax, label=f"戦略{sid}: {info['name']}", color=colors[i],
                       linewidth=1.5, alpha=0.85)

    ax.set_title("全戦略 累積リターン比較 (2018〜2024)", fontsize=14, fontweight="bold")
    ax.set_ylabel("累積倍率 (初期値=1.0)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)

    # 年次リターン比較バーチャート
    ax2 = axes[1]
    strategy_names = []
    ann_returns = []
    for sid, res in sorted(all_results.items()):
        if res["stats"]:
            strategy_names.append(f"S{sid}")
            ann_returns.append(res["stats"].get("ann_return", 0) * 100)

    bars = ax2.bar(strategy_names, ann_returns,
                   color=["steelblue" if r >= 0 else "tomato" for r in ann_returns])
    ax2.set_title("年率リターン比較 (%)")
    ax2.set_ylabel("年率リターン (%)")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, ann_returns):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"比較チャート保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="全9戦略 一括バックテスト")
    parser.add_argument("--start",    default="2018-01-01")
    parser.add_argument("--end",      default="2024-12-31")
    parser.add_argument("--strategies", default=None,
                        help="実行する戦略番号（カンマ区切り, 例: 1,2,3）")
    parser.add_argument("--max-stocks", type=int, default=None,
                        help="最大銘柄数（テスト用。None=全銘柄）")
    parser.add_argument("--top-n",    type=int, default=20)
    parser.add_argument("--force-refresh", action="store_true",
                        help="キャッシュを無視して価格データを再取得")
    parser.add_argument("--out-dir",  default=os.path.join(ROOT, "results"))
    args = parser.parse_args()

    strategy_ids = None
    if args.strategies:
        strategy_ids = [int(x.strip()) for x in args.strategies.split(",")]

    print(f"【全戦略バックテスト開始】")
    print(f"  期間: {args.start} 〜 {args.end}")
    print(f"  戦略: {strategy_ids or '全9戦略'}")
    print(f"  最大銘柄数: {args.max_stocks or '全銘柄'}")
    print(f"  開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    t_total = time.time()

    all_results = run_all_backtests(
        start=args.start,
        end=args.end,
        strategy_ids=strategy_ids,
        max_stocks=args.max_stocks,
        force_refresh=args.force_refresh,
        results_dir=args.out_dir,
        top_n=args.top_n,
    )

    # サマリーテーブル出力・保存
    summary_df = print_summary_table(all_results)
    summary_df.to_csv(os.path.join(args.out_dir, "master_summary.csv"),
                      index=False, encoding="utf-8-sig")

    # 全戦略JSONサマリー保存
    master_stats = {
        sid: res["stats"] for sid, res in all_results.items()
    }
    with open(os.path.join(args.out_dir, "master_stats.json"), "w", encoding="utf-8") as f:
        json.dump(master_stats, f, ensure_ascii=False, indent=2, default=str)

    # 比較チャート生成
    try:
        bench = fetch_benchmark(args.start, args.end, "^N225")
        chart_path = os.path.join(args.out_dir, "master_comparison_chart.png")
        plot_comparison_chart(all_results, bench, chart_path)
    except Exception as e:
        print(f"比較チャート生成失敗: {e}")

    elapsed_total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  全バックテスト完了!")
    print(f"  総所要時間: {elapsed_total/60:.1f}分")
    print(f"  結果保存先: {args.out_dir}")
    print(f"  完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # 次のステップ案内
    print(f"""
次のステップ:
  python statistical_comparison.py   # 統計的有意性の詳細分析
  python generate_orders.py          # 本日の発注指示生成
""")


if __name__ == "__main__":
    main()
