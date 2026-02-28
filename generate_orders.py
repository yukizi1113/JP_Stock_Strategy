"""
generate_orders.py ― 全8戦略の本日発注指示を自動生成

機能:
  1. 当日の価格データを取得（yfinance）
  2. 全8戦略の generate_signals() を実行
  3. 各戦略の具体的発注指示を生成:
     - 日本株戦略（1-7）: ticker, 会社名, BUY/SELL, 推奨ウェイト%,
       予算換算の株数・金額, 参考株価（yfinance終値）
     - マルチアセット（8）: ticker, 資産名, BUY/SELL, ウェイト%,
       円換算金額, スプレッドコスト, 取引所/取引方法
  4. 出力形式:
     - コンソール: 見やすい日本語テーブル
     - orders/YYYY-MM-DD_orders.md: マークダウン形式
     - orders/YYYY-MM-DD_orders.csv: CSV形式

使い方:
  python generate_orders.py
  python generate_orders.py --budget 10000000        # 予算1000万円
  python generate_orders.py --strategies 1,2,8       # 特定戦略のみ
  python generate_orders.py --lookback-months 36     # データ取得期間（月）
  python generate_orders.py --prices-cache data/prices_cache.pkl

発注タイミング:
  - 戦略1-7: 月末リバランス（翌営業日市場価格で成行）
  - 戦略8:   月末リバランス（各アセットの指定取引所・証券会社）
"""
import os, sys, json, argparse
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from data_loader import (
    fetch_prices_cached,
    build_universe_from_edinet,
    EDINET_DB_PATH,
)

# ─────────────────────────────────────────────────────────────────────────────
# 定数・設定
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_BUDGET = 5_000_000  # 5百万円

# 取引方法の案内（戦略8 マルチアセット）
TRADING_METHOD = {
    "equity_etf": "東証ETF（SBI/楽天証券）",
    "reit":       "東証REIT ETF（SBI/楽天証券）",
    "index":      "指数ETF経由 or CFD（IG証券）",
    "commodity":  "コモディティCFD（IG/GMO外貨ex）or 先物口座",
    "forex":      "FX口座（SBI FX/GMOクリック証券）",
    "crypto":     "暗号資産取引所（bitFlyer/Coincheck）",
    "default":    "証券会社にご確認ください",
}

# 戦略ID → 表示名
STRATEGY_NAMES = {
    1: "MLモメンタム",
    2: "PCA固有ポートフォリオ",
    3: "OU平均回帰",
    4: "RL（Q学習）ポートフォリオ",
    5: "吸収比率リスクタイミング",
    6: "マルチファクターML",
    7: "ブラック・リターマン",
    8: "ABCD-Forecast (マルチアセット)",
}

# ─────────────────────────────────────────────────────────────────────────────
# 会社名取得（EDINET DB）
# ─────────────────────────────────────────────────────────────────────────────

_company_name_cache: dict = {}

def get_company_name(ticker: str) -> str:
    """EDINETデータベースまたはキャッシュから会社名を取得"""
    global _company_name_cache
    if not _company_name_cache:
        try:
            import sqlite3
            con = sqlite3.connect(EDINET_DB_PATH)
            rows = con.execute(
                "SELECT ticker, company_name FROM company_master WHERE is_active=1"
            ).fetchall()
            con.close()
            _company_name_cache = {r[0]: r[1] for r in rows}
        except Exception:
            pass
    return _company_name_cache.get(ticker, ticker)


# ─────────────────────────────────────────────────────────────────────────────
# 戦略シグナル生成
# ─────────────────────────────────────────────────────────────────────────────

def get_strategy_instance(sid: int, top_n: int = 20):
    """戦略インスタンスを生成（インポート失敗時はNone）"""
    strat_dirs = {
        1: "01_ml_momentum",
        2: "02_eigen_portfolio",
        3: "03_mean_reversion",
        4: "04_rl_portfolio",
        5: "05_absorption_ratio",
        6: "06_multi_factor_ml",
        7: "07_black_litterman",
        8: "08_abcd_forecast",
    }
    strat_dir = os.path.join(ROOT, "strategies", strat_dirs[sid])
    if strat_dir not in sys.path:
        sys.path.insert(0, strat_dir)
    try:
        if sid == 1:
            from strategy import MLMomentumStrategy
            return MLMomentumStrategy(top_n=top_n)
        elif sid == 2:
            from strategy import EigenPortfolioStrategy
            return EigenPortfolioStrategy(pc_index=1, top_n=top_n, long_only=True)
        elif sid == 3:
            from strategy import MeanReversionStrategy
            return MeanReversionStrategy(hurst_threshold=0.45, entry_z=1.5, exit_z=0.3)
        elif sid == 4:
            from strategy import RLPortfolioStrategy
            return RLPortfolioStrategy(top_n=min(top_n, 15), train_episodes=300)
        elif sid == 5:
            from strategy import AbsorptionRatioStrategy
            return AbsorptionRatioStrategy(ar_window=252, ar_n_components=5, base_top_n=30)
        elif sid == 6:
            from strategy import MultiFactorMLStrategy
            return MultiFactorMLStrategy(top_n=top_n, model_type="xgboost")
        elif sid == 7:
            from strategy import BlackLittermanStrategy
            return BlackLittermanStrategy(top_n=top_n, view_confidence=0.3)
        elif sid == 8:
            from strategy import ABCDForecastStrategy
            return ABCDForecastStrategy(n_matrices=10, top_n=5, long_only=False)
    except Exception as e:
        print(f"  戦略{sid}インスタンス生成失敗: {e}")
        return None


def generate_strategy_signals(
    sid: int,
    prices: pd.DataFrame,
    top_n: int = 20,
) -> dict:
    """
    戦略のシグナルを生成して重みを返す。

    Parameters
    ----------
    sid    : 戦略ID (1-8)
    prices : 価格データ DataFrame
    top_n  : ポートフォリオ銘柄数

    Returns
    -------
    {ticker: weight} の辞書。空の場合は空辞書。
    """
    strategy = get_strategy_instance(sid, top_n=top_n)
    if strategy is None:
        return {}

    as_of = pd.Timestamp(prices.index[-1])

    try:
        weights = strategy.generate_signals(prices, as_of=as_of)
        return weights or {}
    except Exception as e:
        print(f"  戦略{sid} シグナル生成エラー: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# 現在価格の取得
# ─────────────────────────────────────────────────────────────────────────────

def fetch_current_prices(tickers: list, suffix: str = ".T") -> dict:
    """
    yfinanceから各銘柄の直近終値（JPY）を取得する。

    Parameters
    ----------
    tickers : 銘柄コードリスト
    suffix  : ティッカーサフィックス（日本株: ".T"）

    Returns
    -------
    {ticker: price} の辞書
    """
    prices = {}
    yf_tickers = [f"{t}{suffix}" for t in tickers]
    try:
        raw = yf.download(yf_tickers, period="5d", auto_adjust=True, progress=False)
        if raw.empty:
            return prices

        close = raw["Close"]
        if hasattr(close, "columns"):
            for t, yft in zip(tickers, yf_tickers):
                if yft in close.columns:
                    last = close[yft].dropna().iloc[-1] if not close[yft].dropna().empty else None
                    if last is not None:
                        prices[t] = float(last)
        else:
            if len(tickers) == 1:
                last = close.dropna().iloc[-1] if not close.dropna().empty else None
                if last is not None:
                    prices[tickers[0]] = float(last)
    except Exception as e:
        print(f"  現在価格取得エラー: {e}")
    return prices


def fetch_multi_asset_current_prices(tickers: list) -> dict:
    """マルチアセットの現在価格を取得（.T サフィックスなし）"""
    prices = {}
    try:
        raw = yf.download(tickers, period="5d", auto_adjust=True, progress=False)
        if raw.empty:
            return prices
        close = raw["Close"]
        if hasattr(close, "columns"):
            for t in tickers:
                if t in close.columns:
                    last = close[t].dropna().iloc[-1] if not close[t].dropna().empty else None
                    if last is not None:
                        prices[t] = float(last)
        else:
            if len(tickers) == 1:
                last = close.dropna().iloc[-1] if not close.dropna().empty else None
                if last is not None:
                    prices[tickers[0]] = float(last)
    except Exception as e:
        print(f"  マルチアセット現在価格取得エラー: {e}")
    return prices


# ─────────────────────────────────────────────────────────────────────────────
# 発注指示の生成・フォーマット
# ─────────────────────────────────────────────────────────────────────────────

def make_jpx_orders(
    sid: int,
    weights: dict,
    current_prices: dict,
    budget: float,
    universe_df: pd.DataFrame = None,
) -> list:
    """
    日本株戦略（1-7）の発注指示リストを生成する。

    Returns
    -------
    list of dict: 発注指示レコード
    """
    orders = []
    total_w = sum(abs(v) for v in weights.values())
    if total_w == 0:
        return orders

    for ticker, weight in sorted(weights.items(), key=lambda x: -abs(x[1])):
        price = current_prices.get(ticker)
        if price is None or price <= 0:
            continue

        weight_pct = weight / total_w
        alloc_jpy = abs(weight_pct) * budget
        lot_unit  = 100  # 日本株の標準売買単位（100株）
        shares_raw = alloc_jpy / price
        shares = max(lot_unit, int(shares_raw / lot_unit) * lot_unit)  # 売買単位に切り下げ
        actual_jpy = shares * price

        company = get_company_name(ticker)
        direction = "BUY" if weight > 0 else "SELL"

        orders.append({
            "ticker":       ticker,
            "company":      company,
            "direction":    direction,
            "weight_pct":   round(weight_pct * 100, 2),
            "price_jpy":    int(price),
            "shares":       shares,
            "amount_jpy":   int(actual_jpy),
            "lot_unit":     lot_unit,
            "note":         "成行（翌営業日寄付き）",
        })

    return orders


def make_multi_asset_orders(
    weights: dict,
    budget: float,
    usdjpy_rate: float = 150.0,
) -> list:
    """
    マルチアセット戦略（戦略8）の発注指示リストを生成する。

    Parameters
    ----------
    weights     : {ticker: weight} 辞書
    budget      : 予算（円）
    usdjpy_rate : USD/JPY レート（価格がUSD建ての場合の換算に使用）

    Returns
    -------
    list of dict: 発注指示レコード
    """
    # strat8_dir インポート
    strat8_dir = os.path.join(ROOT, "strategies", "08_abcd_forecast")
    if strat8_dir not in sys.path:
        sys.path.insert(0, strat8_dir)

    try:
        from strategy import (
            MULTI_ASSET_UNIVERSE, SPREAD_COSTS, get_asset_class, get_spread_cost
        )
    except Exception:
        MULTI_ASSET_UNIVERSE = {}
        SPREAD_COSTS = {}
        def get_asset_class(t, u=None): return "default"
        def get_spread_cost(t, u=None): return 0.001

    total_w = sum(abs(v) for v in weights.values())
    if total_w == 0:
        return []

    # USD/JPYレートを取得
    try:
        fx = yf.download("USDJPY=X", period="3d", progress=False)
        if not fx.empty:
            usdjpy_rate = float(fx["Close"].dropna().iloc[-1])
    except Exception:
        pass

    orders = []
    for ticker, weight in sorted(weights.items(), key=lambda x: -abs(x[1])):
        weight_pct = weight / total_w
        alloc_jpy  = abs(weight_pct) * budget
        direction  = "BUY" if weight > 0 else "SELL"

        asset_class = get_asset_class(ticker)
        spread_cost = get_spread_cost(ticker)
        spread_cost_pct = spread_cost * 100

        # 資産名・取引方法
        if ticker in MULTI_ASSET_UNIVERSE:
            _, asset_name = MULTI_ASSET_UNIVERSE[ticker]
        else:
            asset_name = ticker

        trading_method = TRADING_METHOD.get(asset_class, TRADING_METHOD["default"])

        # 為替建て資産の換算
        unit_note = ""
        if asset_class == "forex":
            lot_size_jpy = alloc_jpy
            fx_lot = alloc_jpy / (usdjpy_rate * 10000)  # 標準ロット（1lot=10万通貨換算の概算）
            unit_note = f"≈{fx_lot:.1f}万通貨"
        elif asset_class == "crypto":
            unit_note = "取引所で数量確認"
        elif asset_class in ("commodity", "index"):
            unit_note = "CFDまたは先物"
        else:
            unit_note = "ETF口数"

        orders.append({
            "ticker":          ticker,
            "asset_name":      asset_name,
            "asset_class":     asset_class,
            "direction":       direction,
            "weight_pct":      round(weight_pct * 100, 2),
            "amount_jpy":      int(alloc_jpy),
            "spread_cost_pct": round(spread_cost_pct, 3),
            "unit_note":       unit_note,
            "trading_method":  trading_method,
            "note":            "月末リバランス",
        })

    return orders


# ─────────────────────────────────────────────────────────────────────────────
# 出力フォーマット
# ─────────────────────────────────────────────────────────────────────────────

def format_console_table(sid: int, orders: list, budget: float) -> str:
    """コンソール用テーブル文字列を生成"""
    if not orders:
        return f"  ⚠ 戦略{sid}のシグナルなし（データ不足またはエラー）\n"

    lines = []
    is_multi = sid == 8

    if is_multi:
        header = f"  {'ティッカー':<12} {'資産名':<25} {'方向':<5} {'ウェイト':>7} {'概算金額(円)':>12} {'スプレッド':>8}  取引方法"
        sep    = "  " + "-"*100
    else:
        header = f"  {'銘柄コード':<8} {'会社名':<22} {'方向':<5} {'ウェイト':>7} {'参考株価':>8} {'株数':>6} {'概算金額(円)':>12}"
        sep    = "  " + "-"*75

    lines.append(header)
    lines.append(sep)

    total_buy = 0
    total_sell = 0
    for o in orders:
        if is_multi:
            amt = o["amount_jpy"] if o["direction"] == "BUY" else -o["amount_jpy"]
            line = (
                f"  {o['ticker']:<12} {o['asset_name']:<25} "
                f"{'買' if o['direction']=='BUY' else '売':<5} "
                f"{o['weight_pct']:>6.1f}% "
                f"{amt:>+12,}円 "
                f"{o['spread_cost_pct']:>7.3f}%  "
                f"{o['trading_method']}"
            )
        else:
            amt = o["amount_jpy"] if o["direction"] == "BUY" else -o["amount_jpy"]
            line = (
                f"  {o['ticker']:<8} {o['company'][:20]:<22} "
                f"{'買' if o['direction']=='BUY' else '売':<5} "
                f"{o['weight_pct']:>6.1f}% "
                f"{o['price_jpy']:>8,}円 "
                f"{o['shares']:>5}株 "
                f"{o['amount_jpy']:>12,}円"
            )
        lines.append(line)
        if o["direction"] == "BUY":
            total_buy += o["amount_jpy"]
        else:
            total_sell += o["amount_jpy"]

    lines.append(sep)
    lines.append(f"  合計BUY: {total_buy:,}円  合計SELL: {total_sell:,}円  "
                 f"（予算{budget:,}円ベース）")
    if not is_multi:
        lines.append(f"  ※ 株数は{orders[0].get('lot_unit',100)}株単位で計算。端株は切り捨て。")
    return "\n".join(lines)


def format_markdown(
    all_orders: dict,
    budget: float,
    generated_at: str,
) -> str:
    """マークダウン形式の発注指示レポートを生成"""
    lines = [
        "# JP Stock Strategy - 発注指示レポート",
        "",
        f"- **生成日時**: {generated_at}",
        f"- **推奨予算**: ¥{budget:,}",
        f"- **発注タイミング**: 月末翌営業日（成行・寄付き）",
        "",
        "---",
        "",
    ]

    for sid in sorted(all_orders.keys()):
        orders = all_orders[sid]
        strat_name = STRATEGY_NAMES.get(sid, f"戦略{sid}")
        lines.append(f"## 戦略{sid}: {strat_name}")
        lines.append("")

        if not orders:
            lines.append("> シグナルなし（データ不足またはエラー）")
            lines.append("")
            continue

        is_multi = sid == 8
        if is_multi:
            lines.append("| ティッカー | 資産名 | 方向 | ウェイト | 概算金額 | スプレッド | 取引方法 |")
            lines.append("|-----------|--------|------|---------|---------|----------|---------|")
            for o in orders:
                amt = o["amount_jpy"] if o["direction"] == "BUY" else -o["amount_jpy"]
                direction_jp = "買" if o["direction"] == "BUY" else "売"
                lines.append(
                    f"| {o['ticker']} | {o['asset_name']} | {direction_jp} | "
                    f"{o['weight_pct']:.1f}% | ¥{amt:+,} | {o['spread_cost_pct']:.3f}% | "
                    f"{o['trading_method']} |"
                )
        else:
            lines.append("| 銘柄コード | 会社名 | 方向 | ウェイト | 参考株価 | 株数 | 概算金額 |")
            lines.append("|-----------|--------|------|---------|---------|------|---------|")
            for o in orders:
                direction_jp = "買" if o["direction"] == "BUY" else "売"
                lines.append(
                    f"| {o['ticker']} | {o['company']} | {direction_jp} | "
                    f"{o['weight_pct']:.1f}% | ¥{o['price_jpy']:,} | "
                    f"{o['shares']:,}株 | ¥{o['amount_jpy']:,} |"
                )

        lines.append("")
        if is_multi:
            lines.append("**注意**: 為替・コモディティ・仮想通貨は専用口座が必要。")
        else:
            lines.append(
                f"> 予算 ¥{budget:,} ベース。"
                f"発注は{orders[0].get('note','')}。株数は{orders[0].get('lot_unit',100)}株単位。"
            )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("**免責事項**: このレポートはアルゴリズムが生成した参考情報です。")
    lines.append("投資判断は自己責任で行ってください。")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="全8戦略 本日発注指示生成")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET,
                        help=f"予算（円）デフォルト: {DEFAULT_BUDGET:,}")
    parser.add_argument("--strategies", default=None,
                        help="戦略番号（カンマ区切り, 例: 1,2,8）")
    parser.add_argument("--top-n", type=int, default=20,
                        help="ポートフォリオ銘柄数（デフォルト: 20）")
    parser.add_argument("--lookback-months", type=int, default=36,
                        help="シグナル生成に使用する月数（デフォルト: 36ヶ月）")
    parser.add_argument("--prices-cache", default=None)
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "orders"))
    args = parser.parse_args()

    strategy_ids = list(range(1, 9))
    if args.strategies:
        strategy_ids = [int(x.strip()) for x in args.strategies.split(",")]

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_str = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 価格データ取得期間 ────────────────────────────────────────────────────
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=args.lookback_months * 31)).strftime("%Y-%m-%d")

    print("=" * 60)
    print("  JP Stock Strategy - 発注指示生成")
    print(f"  生成日時: {generated_at}")
    print(f"  予算:     ¥{args.budget:,.0f}")
    print(f"  戦略:     {strategy_ids}")
    print(f"  データ期間: {start_date} 〜 {end_date}")
    print("=" * 60)

    # ── ユニバース取得 ────────────────────────────────────────────────────────
    jpx_strategies = [sid for sid in strategy_ids if sid != 8]
    jpx_prices = None
    if jpx_strategies:
        print("\nユニバース取得中...")
        try:
            universe = build_universe_from_edinet()
            tickers = universe["ticker"].tolist()
            print(f"EDINETユニバース: {len(tickers)}銘柄")
        except Exception as e:
            print(f"EDINET失敗: {e}")
            from data_loader import build_jpx_universe
            universe = build_jpx_universe()
            tickers = universe["ticker"].tolist()

        cache = args.prices_cache or os.path.join(ROOT, "data", "prices_cache.pkl")
        print(f"価格取得中（{len(tickers)}銘柄, {start_date}〜{end_date}）...")
        jpx_prices = fetch_prices_cached(
            tickers, start_date, end_date,
            cache_path=cache, batch_size=200,
        )
        jpx_prices = jpx_prices.dropna(axis=1, thresh=int(len(jpx_prices) * 0.5))
        print(f"有効銘柄: {jpx_prices.shape[1]}")

        # 現在価格の取得（発注用）
        print("現在価格取得中...")
        jpx_current = fetch_current_prices(jpx_prices.columns.tolist())
        print(f"現在価格取得: {len(jpx_current)}銘柄")

    # ── マルチアセット価格取得（戦略8用）───────────────────────────────────
    multi_prices = None
    multi_current = {}
    if 8 in strategy_ids:
        print("\n戦略8用マルチアセット価格取得中...")
        strat8_dir = os.path.join(ROOT, "strategies", "08_abcd_forecast")
        if strat8_dir not in sys.path:
            sys.path.insert(0, strat8_dir)
        try:
            from strategy import fetch_multi_asset_prices, MULTI_ASSET_UNIVERSE
            multi_prices = fetch_multi_asset_prices(start_date, end_date)
            multi_prices = multi_prices.dropna(axis=1, thresh=int(len(multi_prices) * 0.3))
            print(f"マルチアセット: {multi_prices.shape[1]}アセット")
            multi_current = fetch_multi_asset_current_prices(multi_prices.columns.tolist())
        except Exception as e:
            print(f"マルチアセット価格取得失敗: {e}")

    # ── 各戦略のシグナル生成 ─────────────────────────────────────────────────
    all_orders = {}
    all_weights = {}

    for sid in strategy_ids:
        print(f"\n[戦略{sid}: {STRATEGY_NAMES.get(sid, '')}] シグナル生成中...")

        if sid == 8:
            if multi_prices is None or multi_prices.empty:
                print("  マルチアセット価格なし → スキップ")
                all_orders[sid] = []
                continue
            weights = generate_strategy_signals(sid, multi_prices, top_n=args.top_n)
        else:
            if jpx_prices is None or jpx_prices.empty:
                print("  日本株価格なし → スキップ")
                all_orders[sid] = []
                continue
            # RL戦略は100銘柄に制限
            if sid == 4:
                prices_for_sig = jpx_prices.iloc[:, :100]
            else:
                prices_for_sig = jpx_prices
            weights = generate_strategy_signals(sid, prices_for_sig, top_n=args.top_n)

        all_weights[sid] = weights
        print(f"  シグナル: {len(weights)}銘柄 (非ゼロウェイト)")
        if weights:
            top3 = sorted(weights.items(), key=lambda x: -abs(x[1]))[:3]
            for t, w in top3:
                print(f"    {t}: {w:+.3f}")

        # 発注指示生成
        if sid == 8:
            orders = make_multi_asset_orders(weights, args.budget)
        else:
            orders = make_jpx_orders(sid, weights, jpx_current, args.budget)

        all_orders[sid] = orders

    # ── コンソール出力 ────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  JP Stock Strategy - 発注指示レポート")
    print(f"  生成日時: {generated_at}")
    print(f"  推奨予算: ¥{args.budget:,.0f}")
    print("=" * 70)

    for sid in sorted(all_orders.keys()):
        strat_name = STRATEGY_NAMES.get(sid, f"戦略{sid}")
        print(f"\n【戦略{sid}: {strat_name}】")
        if sid in (1, 2, 3, 5, 6, 7):
            print("  発注方式: 月末翌営業日 成行（寄付き）/ ロング専用")
        elif sid == 4:
            print("  発注方式: 月末翌営業日 成行（寄付き）/ RL制御 100銘柄以内")
        else:
            print("  発注方式: 月末翌営業日 / 各アセットの指定口座で執行")
        print()
        print(format_console_table(sid, all_orders[sid], args.budget))

    # ── ファイル出力 ──────────────────────────────────────────────────────────
    # Markdown
    md_path = os.path.join(args.out_dir, f"{date_str}_orders.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(format_markdown(all_orders, args.budget, generated_at))
    print(f"\nMarkdown発注指示保存: {md_path}")

    # CSV（全戦略まとめ）
    csv_rows = []
    for sid, orders in all_orders.items():
        strat_name = STRATEGY_NAMES.get(sid, f"戦略{sid}")
        for o in orders:
            row = {"strategy_id": sid, "strategy_name": strat_name}
            row.update(o)
            csv_rows.append(row)
    if csv_rows:
        csv_path = os.path.join(args.out_dir, f"{date_str}_orders.csv")
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"CSV発注指示保存: {csv_path}")

    # JSON（シグナルデータ）
    json_path = os.path.join(args.out_dir, f"{date_str}_signals.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {str(k): v for k, v in all_weights.items()},
            f, ensure_ascii=False, indent=2, default=str
        )
    print(f"シグナルJSON保存: {json_path}")

    print(f"\n{'='*70}")
    print("  発注指示生成完了!")
    print(f"  保存先: {args.out_dir}")
    print(f"{'='*70}")
    print("""
【発注手順】
  1. 月末リバランス日（各月最終営業日）の翌営業日に発注
  2. 発注方式: 成行（寄付き）または指値（前日終値±1%）
  3. 戦略1-7 (日本株): SBI証券、楽天証券などで通常株式発注
  4. 戦略8 (マルチアセット): 各アセットクラスの取引所・口座で別途執行
  5. 仮想通貨・FXは証拠金リスクに注意（ポジションサイズを適切に調整）
""")


if __name__ == "__main__":
    main()
