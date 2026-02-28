"""
戦略5: 吸収比率リスクタイミング バックテスト

使い方:
  python backtest.py --start 2018-01-01 --end 2024-12-31
  python backtest.py --prices-cache data/prices_cache.pkl
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from data_loader import fetch_prices_cached, build_universe_from_edinet
from backtest_engine import Backtester
from strategy import AbsorptionRatioStrategy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def main():
    parser = argparse.ArgumentParser(description="吸収比率リスクタイミング バックテスト")
    parser.add_argument("--start",  default="2018-01-01")
    parser.add_argument("--end",    default="2024-12-31")
    parser.add_argument("--max-stocks", type=int, default=None)
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

    strategy = AbsorptionRatioStrategy(ar_window=252, ar_n_components=5, base_top_n=30)
    bt = Backtester(prices, strategy, rebal_freq="ME")

    print("バックテスト実行中...")
    result = bt.run()

    stats = bt.summary()
    print("\n=== パフォーマンス指標 ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    os.makedirs(args.out_dir, exist_ok=True)
    bt.plot(save_path=os.path.join(args.out_dir, "05_absorption_ratio_result.png"))
    result.to_csv(os.path.join(args.out_dir, "05_absorption_ratio_portfolio.csv"))

    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "05_absorption_ratio_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    print(f"結果を保存: {args.out_dir}")


if __name__ == "__main__":
    main()
