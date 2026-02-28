"""
戦略2: PCA固有ポートフォリオ バックテスト + 吸収比率分析

使い方:
  python backtest.py --start 2018-01-01 --end 2024-12-31 --pc 1
  python backtest.py --absorption-only  # 吸収比率の可視化のみ
  python backtest.py --prices-cache data/prices_cache.pkl
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import fetch_prices_cached, build_universe_from_edinet, daily_returns
from backtest_engine import Backtester
from strategy import EigenPortfolioStrategy, AbsorptionRatioIndicator

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def main():
    parser = argparse.ArgumentParser(description="固有ポートフォリオ バックテスト")
    parser.add_argument("--start",  default="2018-01-01")
    parser.add_argument("--end",    default="2024-12-31")
    parser.add_argument("--pc",     type=int, default=1,
                        help="使用する主成分のインデックス（0=PC1, 1=PC2...）")
    parser.add_argument("--top-n",  type=int, default=20)
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--absorption-only", action="store_true")
    parser.add_argument("--source", default="yfinance")
    parser.add_argument("--prices-cache", default=None)
    parser.add_argument("--out-dir", default=os.path.dirname(__file__))
    args = parser.parse_args()

    print("ユニバース取得中（EDINETデータベース）...")
    try:
        universe = build_universe_from_edinet()
        tickers = universe["ticker"].tolist()
    except Exception as e:
        print(f"EDINET失敗: {e}")
        from data_loader import build_jpx_universe
        universe = build_jpx_universe()
        tickers = universe["ticker"].tolist()
    if args.max_stocks:
        tickers = tickers[:args.max_stocks]

    cache = args.prices_cache or os.path.join(ROOT, "data", "prices_cache.pkl")
    prices = fetch_prices_cached(tickers, args.start, args.end,
                                  source=args.source, cache_path=cache)
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.5))
    print(f"有効銘柄: {prices.shape[1]}")

    if args.absorption_only:
        dr = daily_returns(prices)
        ar = AbsorptionRatioIndicator(n_components=5, window=250).compute(dr)
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        ar.plot(ax=axes[0], title="吸収比率 (Absorption Ratio) - システミックリスク指標")
        axes[0].set_ylabel("AR")
        axes[0].axhline(ar.mean(), color="red", linestyle="--", label=f"平均: {ar.mean():.2f}")
        axes[0].legend()
        ar_change = ar.rolling(60).mean().pct_change(20)
        ar_change.plot(ax=axes[1], title="吸収比率60日平均の20日変化率", color="orange")
        axes[1].axhline(0, color="black", linestyle="-")
        plt.tight_layout()
        os.makedirs(args.out_dir, exist_ok=True)
        out = os.path.join(args.out_dir, "02_absorption_ratio.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        ar.to_csv(os.path.join(args.out_dir, "02_absorption_ratio.csv"))
        print(f"吸収比率グラフ保存: {out}")
        return

    strategy = EigenPortfolioStrategy(
        pc_index=args.pc, top_n=args.top_n, window_months=36, long_only=True
    )
    bt = Backtester(prices, strategy, rebal_freq="ME")
    print("バックテスト実行中...")
    result = bt.run()

    stats = bt.summary()
    print("\n=== パフォーマンス指標 ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    os.makedirs(args.out_dir, exist_ok=True)
    bt.plot(save_path=os.path.join(args.out_dir, "02_eigen_portfolio_result.png"))
    result.to_csv(os.path.join(args.out_dir, "02_eigen_portfolio_portfolio.csv"))

    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "02_eigen_portfolio_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    print(f"結果を保存: {args.out_dir}")


if __name__ == "__main__":
    main()
