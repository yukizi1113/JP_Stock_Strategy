"""
戦略4: Q学習ポートフォリオ最適化 バックテスト

使い方:
  python backtest.py --start 2018-01-01 --end 2024-12-31
  python backtest.py --prices-cache data/prices_cache.pkl

注意: Q-tableの次元数制約により max-stocks は 100 が上限。
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from data_loader import fetch_prices_cached, build_universe_from_edinet
from backtest_engine import Backtester
from strategy import RLPortfolioStrategy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Q学習ポートフォリオ バックテスト")
    parser.add_argument("--start",   default="2018-01-01")
    parser.add_argument("--end",     default="2024-12-31")
    parser.add_argument("--top-n",   type=int, default=15)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-stocks", type=int, default=100,
                        help="Q-table次元数制約により100以下を推奨")
    parser.add_argument("--source",  default="yfinance")
    parser.add_argument("--prices-cache", default=None)
    parser.add_argument("--out-dir", default=os.path.dirname(__file__))
    args = parser.parse_args()

    print("ユニバース取得中（EDINETデータベース）...")
    try:
        universe = build_universe_from_edinet()
        tickers = universe["ticker"].tolist()[:args.max_stocks]
    except Exception as e:
        print(f"EDINET失敗: {e}")
        tickers = [
            "7203","6758","9432","8306","6861","4063","7974","9984",
            "6902","8035","9433","8316","6954","4502","7751","6367",
            "4568","7267","8411","3382","9020","8058","6701","5108",
            "2914","4911","8591","9022","7011","6981",
        ][:args.max_stocks]

    cache = args.prices_cache or os.path.join(ROOT, "data", "prices_cache.pkl")
    prices = fetch_prices_cached(tickers, args.start, args.end,
                                  source=args.source, cache_path=cache)
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.5))
    print(f"有効銘柄: {prices.shape[1]}")

    strategy = RLPortfolioStrategy(
        top_n=args.top_n,
        train_episodes=args.episodes,
    )
    bt = Backtester(prices, strategy, rebal_freq="ME")

    print("バックテスト実行中（Q学習含む）...")
    result = bt.run()

    stats = bt.summary()
    print("\n=== パフォーマンス指標 ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    os.makedirs(args.out_dir, exist_ok=True)
    bt.plot(save_path=os.path.join(args.out_dir, "04_rl_portfolio_result.png"))
    result.to_csv(os.path.join(args.out_dir, "04_rl_portfolio_portfolio.csv"))

    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "04_rl_portfolio_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    print(f"結果を保存: {args.out_dir}")


if __name__ == "__main__":
    main()
