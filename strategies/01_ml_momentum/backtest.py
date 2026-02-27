"""
戦略1: MLファクターモメンタム バックテスト実行スクリプト

使い方:
  python backtest.py --start 2015-01-01 --end 2024-12-31 --top-n 20
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
from data_loader import fetch_prices, build_jpx_universe
from backtest_engine import Backtester
from strategy import MLMomentumStrategy


def main():
    parser = argparse.ArgumentParser(description="MLモメンタム戦略バックテスト")
    parser.add_argument("--start",  default="2015-01-01")
    parser.add_argument("--end",    default="2024-12-31")
    parser.add_argument("--top-n",  type=int, default=20)
    parser.add_argument("--max-stocks", type=int, default=300,
                        help="ユニバースの最大銘柄数（取得時間削減のため）")
    parser.add_argument("--source", default="yfinance", choices=["yfinance", "stooq"])
    args = parser.parse_args()

    print("JPXユニバース取得中...")
    try:
        universe = build_jpx_universe()
        tickers = universe["ticker"].tolist()[:args.max_stocks]
    except Exception as e:
        print(f"JPXユニバース取得失敗: {e}")
        # 代替: 日本の主要銘柄
        tickers = [
            "7203","6758","9432","8306","6861","4063","7974","9984",
            "6902","8035","9433","8316","6954","4502","7751","6367",
            "4568","7267","8411","3382","9020","8058","6701","5108",
        ]

    print(f"価格データ取得中（{len(tickers)}銘柄）...")
    prices = fetch_prices(tickers, args.start, args.end, source=args.source)
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.5))
    print(f"有効銘柄数: {prices.shape[1]}")

    strategy = MLMomentumStrategy(top_n=args.top_n)
    bt = Backtester(prices, strategy, rebal_freq="ME")

    print("バックテスト実行中...")
    result = bt.run()

    stats = bt.summary()
    print("\n=== パフォーマンス指標 ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # グラフ保存
    out_dir = os.path.dirname(__file__)
    bt.plot(save_path=os.path.join(out_dir, "backtest_result.png"))
    result.to_csv(os.path.join(out_dir, "backtest_portfolio.csv"))
    print(f"\n結果を保存: {out_dir}")


if __name__ == "__main__":
    main()
