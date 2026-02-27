"""
戦略3: OU平均回帰 バックテスト実行スクリプト

使い方:
  python backtest.py --start 2015-01-01 --end 2024-12-31
  python backtest.py --pairs   # ペア取引モード
  python backtest.py --hurst-only  # Hurst指数の分布確認
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data_loader import fetch_prices, build_jpx_universe
from backtest_engine import Backtester
from strategy import MeanReversionStrategy, PairsTradingStrategy, hurst_exponent


def plot_hurst_distribution(prices: pd.DataFrame, lookback: int = 252) -> None:
    """Hurst指数の分布を銘柄ごとに可視化"""
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
    out = os.path.join(os.path.dirname(__file__), "hurst_distribution.png")
    plt.savefig(out, dpi=150)
    print(f"Hurst分布: {out}")
    print(f"  H < 0.45 (平均回帰): {(h_series < 0.45).sum()} 銘柄")
    print(f"  H = 0.45-0.55 (中間): {((h_series >= 0.45) & (h_series <= 0.55)).sum()} 銘柄")
    print(f"  H > 0.55 (トレンド): {(h_series > 0.55).sum()} 銘柄")


def main():
    parser = argparse.ArgumentParser(description="OU平均回帰 バックテスト")
    parser.add_argument("--start",  default="2015-01-01")
    parser.add_argument("--end",    default="2024-12-31")
    parser.add_argument("--pairs",  action="store_true", help="ペア取引モード")
    parser.add_argument("--hurst-only", action="store_true")
    parser.add_argument("--max-stocks", type=int, default=150)
    parser.add_argument("--source", default="yfinance")
    args = parser.parse_args()

    tickers = [
        "7203","7267","6758","6752","9432","9433","8306","8316",
        "4502","4568","6861","4063","7974","9984","6902","8035",
        "9020","8058","6701","5108","3382","2914","7751","6367",
        "6954","4523","2802","2503","4911","8591","9022","7011",
    ]
    try:
        universe = build_jpx_universe()
        tickers = universe["ticker"].tolist()[:args.max_stocks]
    except Exception:
        pass

    print(f"価格取得中（{len(tickers)}銘柄）...")
    prices = fetch_prices(tickers, args.start, args.end, source=args.source)
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.5))

    if args.hurst_only:
        plot_hurst_distribution(prices)
        return

    if args.pairs:
        strategy = PairsTradingStrategy()
    else:
        strategy = MeanReversionStrategy(hurst_threshold=0.45, entry_z=1.5, exit_z=0.3)

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
