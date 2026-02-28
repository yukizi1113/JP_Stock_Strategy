"""
戦略8: ABCD-Forecast マルチアセット バックテスト

使い方:
  python backtest.py --start 2018-01-01 --end 2024-12-31
  python backtest.py --long-only   # ロング専用モード

対象: マルチアセット（日本ETF、コモディティ先物、為替、仮想通貨、株価指数）
データソース: yfinance（MULTI_ASSET_UNIVERSEのすべてのティッカーに対応）

注意:
  - 日本個別株ではなくグローバルマルチアセットが対象
  - 仮想通貨（BTC/ETH）は高スプレッドのためウェイト制限あり
  - 為替・コモディティは証拠金取引が必要（現物の場合はETF代替を推奨）
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
from backtest_engine import Backtester
from strategy import ABCDForecastStrategy, fetch_multi_asset_prices, MULTI_ASSET_UNIVERSE

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def main():
    parser = argparse.ArgumentParser(description="ABCD-Forecast マルチアセット バックテスト")
    parser.add_argument("--start",       default="2018-01-01")
    parser.add_argument("--end",         default="2024-12-31")
    parser.add_argument("--n-matrices",  type=int, default=10,
                        help="ABCD行列の数（増やすほど安定・時間がかかる）")
    parser.add_argument("--top-n",       type=int, default=5,
                        help="ロング/ショート各銘柄数")
    parser.add_argument("--long-only",   action="store_true")
    parser.add_argument("--out-dir",     default=os.path.dirname(__file__))
    args = parser.parse_args()

    print(f"マルチアセット価格データ取得中（{len(MULTI_ASSET_UNIVERSE)}アセット）...")
    prices = fetch_multi_asset_prices(args.start, args.end)
    if prices.empty:
        print("価格データ取得失敗")
        return

    # 欠損率 70% 超のアセットを除外
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.3))
    print(f"有効アセット数: {prices.shape[1]}")
    print(f"期間: {prices.index[0].date()} 〜 {prices.index[-1].date()}")

    # 利用可能なアセットとその分類を表示
    print("\n【利用可能アセット一覧】")
    for tk in prices.columns:
        if tk in MULTI_ASSET_UNIVERSE:
            asset_class, desc = MULTI_ASSET_UNIVERSE[tk]
            print(f"  {tk:12s} [{asset_class:12s}] {desc}")

    strategy = ABCDForecastStrategy(
        n_matrices=args.n_matrices,
        top_n=args.top_n,
        long_only=args.long_only,
    )
    bt = Backtester(
        prices,
        strategy,
        rebal_freq="ME",
        benchmark_ticker="^N225",
        transaction_cost=0.001,
    )

    print("\nバックテスト実行中（ABCD-Forecast）...")
    result = bt.run()

    stats = bt.summary()
    print("\n=== パフォーマンス指標 ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    os.makedirs(args.out_dir, exist_ok=True)
    bt.plot(save_path=os.path.join(args.out_dir, "08_abcd_forecast_result.png"))
    result.to_csv(os.path.join(args.out_dir, "08_abcd_forecast_portfolio.csv"))

    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "08_abcd_forecast_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n結果を保存: {args.out_dir}")


if __name__ == "__main__":
    main()
