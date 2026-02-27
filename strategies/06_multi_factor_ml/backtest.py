"""
戦略6: マルチファクターML バックテスト

使い方:
  python backtest.py --start 2015-01-01 --end 2024-12-31
  python backtest.py --model xgboost
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from data_loader import fetch_prices, build_jpx_universe
from backtest_engine import Backtester
from strategy import MultiFactorMLStrategy


def main():
    parser = argparse.ArgumentParser(description="マルチファクターML バックテスト")
    parser.add_argument("--start",  default="2015-01-01")
    parser.add_argument("--end",    default="2024-12-31")
    parser.add_argument("--top-n",  type=int, default=20)
    parser.add_argument("--model",  default="rf", choices=["rf", "xgboost"])
    parser.add_argument("--max-stocks", type=int, default=200)
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

    strategy = MultiFactorMLStrategy(top_n=args.top_n, model_type=args.model)
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
