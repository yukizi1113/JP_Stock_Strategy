"""
戦略1: MLファクターモメンタム バックテスト実行スクリプト

使い方:
  python backtest.py --start 2018-01-01 --end 2024-12-31 --top-n 20
  python backtest.py --prices-cache data/prices_cache.pkl   # キャッシュ使用
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
from data_loader import fetch_prices_cached, build_universe_from_edinet
from backtest_engine import Backtester
from strategy import MLMomentumStrategy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def main():
    parser = argparse.ArgumentParser(description="MLモメンタム戦略バックテスト")
    parser.add_argument("--start",  default="2018-01-01")
    parser.add_argument("--end",    default="2024-12-31")
    parser.add_argument("--top-n",  type=int, default=20)
    parser.add_argument("--max-stocks", type=int, default=None,
                        help="ユニバースの最大銘柄数（None=全銘柄）")
    parser.add_argument("--source", default="yfinance", choices=["yfinance", "stooq"])
    parser.add_argument("--prices-cache", default=None,
                        help="価格データキャッシュファイルパス (.pkl)")
    parser.add_argument("--out-dir", default=os.path.dirname(__file__))
    args = parser.parse_args()

    # ユニバース取得（EDINETデータベース優先）
    print("ユニバース取得中（EDINETデータベース）...")
    try:
        universe = build_universe_from_edinet()
        tickers = universe["ticker"].tolist()
        print(f"EDINETユニバース: {len(tickers)}銘柄")
    except Exception as e:
        print(f"EDINET失敗: {e}, JPXウェブから取得...")
        from data_loader import build_jpx_universe
        universe = build_jpx_universe()
        tickers = universe["ticker"].tolist()
    if args.max_stocks:
        tickers = tickers[:args.max_stocks]

    # 価格取得（キャッシュ活用）
    cache = args.prices_cache or os.path.join(ROOT, "data", "prices_cache.pkl")
    prices = fetch_prices_cached(
        tickers, args.start, args.end,
        source=args.source, cache_path=cache
    )
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

    os.makedirs(args.out_dir, exist_ok=True)
    bt.plot(save_path=os.path.join(args.out_dir, "01_ml_momentum_result.png"))
    result.to_csv(os.path.join(args.out_dir, "01_ml_momentum_portfolio.csv"))

    # 統計をJSONで保存（master_backtest.pyが読み込む）
    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "01_ml_momentum_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n結果を保存: {args.out_dir}")


if __name__ == "__main__":
    main()
