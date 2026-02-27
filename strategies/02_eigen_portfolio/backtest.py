"""
戦略2: PCA固有ポートフォリオ バックテスト + 吸収比率分析

使い方:
  python backtest.py --start 2015-01-01 --end 2024-12-31 --pc 1
  python backtest.py --absorption-only  # 吸収比率の可視化のみ
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import fetch_prices, build_jpx_universe, daily_returns
from backtest_engine import Backtester
from strategy import EigenPortfolioStrategy, AbsorptionRatioIndicator


def main():
    parser = argparse.ArgumentParser(description="固有ポートフォリオ バックテスト")
    parser.add_argument("--start",  default="2015-01-01")
    parser.add_argument("--end",    default="2024-12-31")
    parser.add_argument("--pc",     type=int, default=1,
                        help="使用する主成分のインデックス（0=PC1, 1=PC2...）")
    parser.add_argument("--top-n",  type=int, default=20)
    parser.add_argument("--max-stocks", type=int, default=200)
    parser.add_argument("--absorption-only", action="store_true")
    parser.add_argument("--source", default="yfinance")
    args = parser.parse_args()

    print("ユニバース取得...")
    try:
        universe = build_jpx_universe()
        tickers = universe["ticker"].tolist()[:args.max_stocks]
    except Exception:
        tickers = [
            "7203","6758","9432","8306","6861","4063","7974","9984",
            "6902","8035","9433","8316","6954","4502","7751","6367",
        ]

    print(f"価格取得中（{len(tickers)}銘柄）...")
    prices = fetch_prices(tickers, args.start, args.end, source=args.source)
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.5))
    print(f"有効銘柄: {prices.shape[1]}")

    if args.absorption_only:
        # 吸収比率の可視化
        dr = daily_returns(prices)
        ar = AbsorptionRatioIndicator(n_components=5, window=250).compute(dr)

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        ar.plot(ax=axes[0], title="吸収比率 (Absorption Ratio) - システミックリスク指標")
        axes[0].set_ylabel("AR")
        axes[0].axhline(ar.mean(), color="red", linestyle="--", label=f"平均: {ar.mean():.2f}")
        axes[0].legend()

        # AR の変化率（急増 = 市場ストレス増大）
        ar_change = ar.rolling(60).mean().pct_change(20)
        ar_change.plot(ax=axes[1], title="吸収比率60日平均の20日変化率", color="orange")
        axes[1].axhline(0, color="black", linestyle="-")

        plt.tight_layout()
        out = os.path.join(os.path.dirname(__file__), "absorption_ratio.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        ar.to_csv(os.path.join(os.path.dirname(__file__), "absorption_ratio.csv"))
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

    out_dir = os.path.dirname(__file__)
    bt.plot(save_path=os.path.join(out_dir, "backtest_result.png"))
    result.to_csv(os.path.join(out_dir, "backtest_portfolio.csv"))


if __name__ == "__main__":
    main()
