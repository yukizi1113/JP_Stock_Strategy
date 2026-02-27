"""
戦略7: ブラック・リターマン + ML ビュー バックテスト

使い方:
  python backtest.py --start 2015-01-01 --end 2024-12-31
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from data_loader import fetch_prices, build_jpx_universe
from backtest_engine import Backtester
from strategy import BlackLittermanStrategy


def main():
    parser = argparse.ArgumentParser(description="B-L + MLビュー バックテスト")
    parser.add_argument("--start",  default="2015-01-01")
    parser.add_argument("--end",    default="2024-12-31")
    parser.add_argument("--top-n",  type=int, default=20)
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--max-stocks", type=int, default=100)
    parser.add_argument("--source", default="yfinance")
    args = parser.parse_args()

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

    strategy = BlackLittermanStrategy(top_n=args.top_n, view_confidence=args.confidence)
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
