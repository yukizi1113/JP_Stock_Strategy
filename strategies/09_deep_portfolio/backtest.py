"""
戦略9: Deep Portfolio Optimization (DPO) バックテスト
Imajo, Minami, Ito, Nakagawa (AAAI 2021)

使い方:
  python backtest.py --start 2018-01-01 --end 2024-12-31
  python backtest.py --max-stocks 500   # 銘柄数を制限（高速テスト用）
  python backtest.py --top-n 30         # 保有銘柄数を変更

所要時間目安（CPU）:
  500銘柄:  30〜60分（6ヶ月ごとに再学習）
  1000銘柄: 1〜2時間

注意:
  - PyTorch が必要。`pip install torch --index-url https://download.pytorch.org/whl/cpu`
  - GPU がある場合は自動的に GPU を使用
  - メモリ不足の場合は --max-train-stocks を減らすこと
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
from backtest_engine import Backtester
from data_loader import fetch_prices_cached, build_universe_from_edinet

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Deep Portfolio Optimization バックテスト")
    parser.add_argument("--start",           default="2018-01-01")
    parser.add_argument("--end",             default="2024-12-31")
    parser.add_argument("--max-stocks",      type=int, default=None,
                        help="使用する銘柄数上限（None=全銘柄）")
    parser.add_argument("--top-n",           type=int, default=20,
                        help="ロングポートフォリオの銘柄数")
    parser.add_argument("--H",               type=int, default=128,
                        help="ルックバック窓（日数、論文は256）")
    parser.add_argument("--C",               type=int, default=20,
                        help="除去する主成分数（日本株最適=20）")
    parser.add_argument("--L",               type=int, default=4,
                        help="フラクタルスケール数（論文は22）")
    parser.add_argument("--Q",               type=int, default=16,
                        help="分位点数（論文は32）")
    parser.add_argument("--K",               type=int, default=64,
                        help="FractalNet 中間層次元（論文は256）")
    parser.add_argument("--retrain-interval", type=int, default=6,
                        help="再学習間隔（月数）")
    parser.add_argument("--max-train-stocks", type=int, default=500,
                        help="学習時の最大銘柄数（メモリ節約）")
    parser.add_argument("--n-epochs",        type=int, default=8)
    parser.add_argument("--cache-path",      default=os.path.join(ROOT, "data", "prices_cache.pkl"))
    parser.add_argument("--out-dir",         default=os.path.dirname(__file__))
    parser.add_argument("--force-refresh",   action="store_true")
    args = parser.parse_args()

    # ── 戦略インポート ────────────────────────────────────────────────────────
    sys.path.insert(0, os.path.dirname(__file__))
    from strategy import DeepPortfolioStrategy

    # ── ユニバース取得 ────────────────────────────────────────────────────────
    print("ユニバース取得中（EDINETデータベース）...")
    try:
        universe = build_universe_from_edinet()
        tickers = universe["ticker"].tolist()
    except Exception as e:
        print(f"EDINET失敗 ({e}) → データキャッシュから取得")
        tickers = None

    # ── 価格データ取得 ────────────────────────────────────────────────────────
    print(f"価格データ取得中（{args.start} 〜 {args.end}）...")
    prices = fetch_prices_cached(
        tickers, args.start, args.end,
        cache_path=args.cache_path,
        force_refresh=args.force_refresh,
        batch_size=200,
    )

    # 欠損率 50% 超の銘柄を除外
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.5))

    if args.max_stocks and args.max_stocks < prices.shape[1]:
        prices = prices.iloc[:, :args.max_stocks]
        print(f"銘柄数を {args.max_stocks} に制限")

    print(f"価格データ: {prices.shape[1]}銘柄 / {len(prices)}日")
    print(f"期間: {prices.index[0].date()} 〜 {prices.index[-1].date()}")

    # ── 戦略・バックテスト ────────────────────────────────────────────────────
    strategy = DeepPortfolioStrategy(
        top_n=args.top_n,
        H=args.H,
        C=args.C,
        L=args.L,
        Q=args.Q,
        K=args.K,
        retrain_interval=args.retrain_interval,
        max_train_stocks=args.max_train_stocks,
        n_epochs=args.n_epochs,
    )

    bt = Backtester(
        prices,
        strategy,
        rebal_freq="ME",
        benchmark_ticker="^N225",
        transaction_cost=0.001,
    )

    print("\nバックテスト実行中（Deep Portfolio Optimization）...")
    print(f"  パラメータ: H={args.H}, C={args.C}, L={args.L}, Q={args.Q}, K={args.K}")
    print(f"  再学習間隔: {args.retrain_interval}ヶ月, 学習銘柄数上限: {args.max_train_stocks}")
    result = bt.run()

    stats = bt.summary()
    print("\n=== パフォーマンス指標 ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    os.makedirs(args.out_dir, exist_ok=True)
    bt.plot(save_path=os.path.join(args.out_dir, "09_deep_portfolio_result.png"))
    result.to_csv(os.path.join(args.out_dir, "09_deep_portfolio_portfolio.csv"))

    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "09_deep_portfolio_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n結果を保存: {args.out_dir}")


if __name__ == "__main__":
    main()
