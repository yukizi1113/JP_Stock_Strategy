"""
ライブ取引実行スクリプト（kabu STATION API連携）

指定した戦略を実行し、kabu STATIONを通じてリバランス注文を発注する。

使い方:
  python live_trader.py --strategy ml_momentum --budget 5000000 --dry-run
  python live_trader.py --strategy eigen_portfolio --budget 5000000
  python live_trader.py --strategy multi_factor --budget 5000000

重要:
  --dry-run フラグを付けると注文は発注されず、計画のみ表示される。
  実際の注文は --dry-run を外したときのみ発注される。
"""
import sys, os, argparse, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from datetime import date, timedelta
from data_loader import fetch_prices, build_jpx_universe
from kabu_api import KabuClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


STRATEGY_MAP = {
    "ml_momentum": {
        "module": "strategies.01_ml_momentum.strategy",
        "class":  "MLMomentumStrategy",
        "kwargs": {"top_n": 20},
    },
    "eigen_portfolio": {
        "module": "strategies.02_eigen_portfolio.strategy",
        "class":  "EigenPortfolioStrategy",
        "kwargs": {"pc_index": 1, "top_n": 20},
    },
    "mean_reversion": {
        "module": "strategies.03_mean_reversion.strategy",
        "class":  "MeanReversionStrategy",
        "kwargs": {"hurst_threshold": 0.45, "top_n": 15},
    },
    "rl_portfolio": {
        "module": "strategies.04_rl_portfolio.strategy",
        "class":  "RLPortfolioStrategy",
        "kwargs": {"top_n": 15},
    },
    "absorption_ratio": {
        "module": "strategies.05_absorption_ratio.strategy",
        "class":  "AbsorptionRatioStrategy",
        "kwargs": {"base_top_n": 30},
    },
    "multi_factor": {
        "module": "strategies.06_multi_factor_ml.strategy",
        "class":  "MultiFactorMLStrategy",
        "kwargs": {"top_n": 20},
    },
    "black_litterman": {
        "module": "strategies.07_black_litterman.strategy",
        "class":  "BlackLittermanStrategy",
        "kwargs": {"top_n": 20},
    },
}


def load_strategy(name: str):
    """戦略名から戦略インスタンスを動的にロードする"""
    info = STRATEGY_MAP[name]
    # モジュールパスを Python import 形式に変換
    mod_path = info["module"].replace(".", "/") + ".py"
    base_dir = os.path.dirname(os.path.dirname(__file__))
    full_path = os.path.join(base_dir, mod_path)

    import importlib.util
    spec = importlib.util.spec_from_file_location(info["class"], full_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, info["class"])
    return cls(**info["kwargs"])


def main():
    parser = argparse.ArgumentParser(description="kabu STATION ライブ取引実行")
    parser.add_argument("--strategy", required=True, choices=list(STRATEGY_MAP.keys()))
    parser.add_argument("--budget", type=float, default=5_000_000,
                        help="総投資予算（円）")
    parser.add_argument("--dry-run", action="store_true",
                        help="注文を発注せず計画のみ表示")
    parser.add_argument("--lookback", type=int, default=365,
                        help="価格データのルックバック日数")
    parser.add_argument("--max-stocks", type=int, default=200)
    args = parser.parse_args()

    logger.info(f"戦略: {args.strategy}")
    logger.info(f"予算: ¥{args.budget:,.0f}")
    logger.info(f"ドライラン: {args.dry_run}")

    # ユニバース取得
    logger.info("JPXユニバース取得中...")
    try:
        universe = build_jpx_universe()
        tickers = universe["ticker"].tolist()[:args.max_stocks]
    except Exception as e:
        logger.warning(f"ユニバース取得失敗: {e} → デフォルト銘柄を使用")
        tickers = [
            "7203","6758","9432","8306","6861","4063","7974","9984",
            "6902","8035","9433","8316","6954","4502","7751","6367",
        ]

    # 価格データ取得
    end   = date.today().isoformat()
    start = (date.today() - timedelta(days=args.lookback)).isoformat()
    logger.info(f"価格取得中（{len(tickers)}銘柄, {start} ~ {end}）...")
    prices = fetch_prices(tickers, start, end, source="yfinance")
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.7))
    logger.info(f"有効銘柄: {prices.shape[1]}")

    # 戦略シグナル生成
    strategy = load_strategy(args.strategy)
    logger.info(f"シグナル生成中: {strategy.name}")
    as_of = pd.Timestamp(prices.index[-1])
    signals = strategy.generate_signals(prices, as_of)

    if not signals:
        logger.warning("シグナルなし → 注文なし")
        return

    # シグナルの表示
    logger.info("\n=== ポートフォリオ目標ウェイト ===")
    for ticker, weight in sorted(signals.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {ticker}: {weight:.2%}  (¥{args.budget * weight:,.0f})")

    # kabu STATION API 経由でリバランス
    if not args.dry_run:
        logger.info("kabu STATION API に接続...")
        client = KabuClient()
        client.get_token()

        logger.info("リバランス注文発注中...")
        orders = client.rebalance(
            target_weights=signals,
            total_budget=args.budget,
            min_trade_unit=100,
            dry_run=False,
        )
        logger.info(f"発注完了: {len(orders)}件")
        for o in orders:
            logger.info(f"  {o['ticker']}: {o['qty']}株 @ ¥{o['price']:,.0f}")
    else:
        logger.info("[ドライラン] 注文は発注されていません。")
        # ドライランでも kabu API から現在価格を確認
        try:
            client = KabuClient()
            client.get_token()
            plans = client.rebalance(
                target_weights=signals,
                total_budget=args.budget,
                dry_run=True,
            )
            logger.info("\n=== 注文計画（ドライラン） ===")
            for p in plans:
                logger.info(f"  {p['ticker']}: {p['qty']}株 × ¥{p['price']:,.0f}")
        except Exception as e:
            logger.warning(f"API接続失敗（ドライラン確認スキップ）: {e}")


if __name__ == "__main__":
    main()
