"""
戦略3: OU平均回帰 バックテスト実行スクリプト

使い方:
  python backtest.py --start 2018-01-01 --end 2024-12-31
  python backtest.py --pairs   # ペア取引モード
  python backtest.py --hurst-only  # Hurst指数の分布確認
  python backtest.py --prices-cache data/prices_cache.pkl
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data_loader import fetch_prices_cached, build_universe_from_edinet
from backtest_engine import Backtester
from strategy import MeanReversionStrategy, PairsTradingStrategy, hurst_exponent

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def plot_hurst_distribution(prices: pd.DataFrame, lookback: int = 252, out_dir: str = ".") -> None:
    hursts = {}
    recent = prices.iloc[-lookback:]
    for ticker in recent.columns:
        ts = recent[ticker].dropna()
        if len(ts) > 50:
            log_p = np.log(ts.values)
            hursts[ticker] = hurst_exponent(log_p, max_lag=50)
    if not hursts:
        return
    h_series = pd.Series(hursts)
    fig, ax = plt.subplots(figsize=(10, 5))
    h_series.hist(bins=30, ax=ax, color="steelblue", edgecolor="white")
    ax.axvline(0.5, color="red", linestyle="--", label="H=0.5 (ランダムウォーク)")
    ax.axvline(0.45, color="orange", linestyle="--", label="H=0.45 (選別しきい値)")
    ax.set_title("Hurst指数の分布（日本株）")
    ax.set_xlabel("Hurst指数 H")
    ax.set_ylabel("銘柄数")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(out_dir, "03_hurst_distribution.png")
    plt.savefig(out, dpi=150)
    print(f"Hurst分布: {out}")
    print(f"  H < 0.45 (平均回帰): {(h_series < 0.45).sum()} 銘柄")
    print(f"  H = 0.45-0.55 (中間): {((h_series >= 0.45) & (h_series <= 0.55)).sum()} 銘柄")
    print(f"  H > 0.55 (トレンド): {(h_series > 0.55).sum()} 銘柄")


def main():
    parser = argparse.ArgumentParser(description="OU平均回帰 バックテスト")
    parser.add_argument("--start",  default="2018-01-01")
    parser.add_argument("--end",    default="2024-12-31")
    parser.add_argument("--pairs",  action="store_true", help="ペア取引モード")
    parser.add_argument("--hurst-only", action="store_true")
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

    os.makedirs(args.out_dir, exist_ok=True)

    if args.hurst_only:
        plot_hurst_distribution(prices, out_dir=args.out_dir)
        return

    if args.pairs:
        strategy = PairsTradingStrategy()
        strat_name = "03_pairs_trading"
    else:
        strategy = MeanReversionStrategy(hurst_threshold=0.45, entry_z=1.5, exit_z=0.3)
        strat_name = "03_mean_reversion"

    bt = Backtester(prices, strategy, rebal_freq="ME")
    print("バックテスト実行中...")
    result = bt.run()

    stats = bt.summary()
    print("\n=== パフォーマンス指標 ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    bt.plot(save_path=os.path.join(args.out_dir, f"{strat_name}_result.png"))
    result.to_csv(os.path.join(args.out_dir, f"{strat_name}_portfolio.csv"))

    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{strat_name}_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    print(f"結果を保存: {args.out_dir}")


if __name__ == "__main__":
    main()
