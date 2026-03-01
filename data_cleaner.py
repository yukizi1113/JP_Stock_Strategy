"""
data_cleaner.py ― 価格データの厳密品質管理モジュール

【設計思想】
  「学習を開始する前に何度もデータの正確性をチェックする」

  不完全なデータで機械学習モデルを学習すると、スプリアスなパターンを
  学習してしまう（いわゆる「ガベージイン・ガベージアウト」問題）。
  特に日本株yfinanceデータには以下の既知の問題がある:
    (a) 上場廃止銘柄のbfill → 上場前期間に仮想の価格が入る
    (b) データプロバイダの一時的なエラー → 1日に-99%→翌日+∞% の往復
    (c) 長期停止銘柄 → 同一価格が数ヶ月続く（出来高0相当）
    (d) 単位変更（株式分割・併合）が反映されていない場合 → 1日で数倍の変動

【クリーニング手順 (3ラウンド)】
  Round 1: 上場前NaN保持・ゼロ価格除去・明らかな異常値除去
  Round 2: 価格スパイク（上昇→即時反転）検出・停滞価格検出
  Round 3: 最終確認・カバレッジ確認・品質レポート

【リークに関する注意】
  clean_price_data() は過去データにのみ適用する。
  この関数はバックテスト開始前に価格系列全体を一括クリーニングするが、
  クリーニング基準（閾値）は固定パラメータであり、将来情報を参照しない。
  各戦略の generate_signals() には as_of 以前のデータのみ渡されるため、
  クリーニング後もリーク構造は変わらない。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 定数
# ─────────────────────────────────────────────────────────────────────────────

# 東証の値幅制限（price limit）: 株価帯別に±15%〜±30%
# データエラーの判定には保守的に ±50% を使用（制限値の約2倍）
DAILY_RETURN_HARD_CAP  = 0.50   # Round 1: これ以上は確実にエラー
DAILY_RETURN_SOFT_CAP  = 0.40   # Round 2: スパイク検出の補助閾値
SPIKE_REVERSAL_RATIO   = 0.60   # スパイクの反転率: 翌日リターンが -60%以上 = スパイク疑い
STALE_WINDOW           = 30     # 停滞検出ウィンドウ（日数）
STALE_ZERO_THRESHOLD   = 25     # ウィンドウ内でゼロリターンが25日以上 → 停滞
FFILL_LIMIT            = 10     # 連続欠損を許容するffill最大日数（祝日等）
MIN_TRADING_DAYS       = 252    # 最低取引日数（約1年）
MIN_DATA_COVERAGE      = 0.80   # 期間全体でのデータカバレッジ率（80%以上必須）


# ─────────────────────────────────────────────────────────────────────────────
# メインクリーニング関数
# ─────────────────────────────────────────────────────────────────────────────

def clean_price_data(
    prices: pd.DataFrame,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    3ラウンドの反復クリーニングで価格データを品質管理する。

    Parameters
    ----------
    prices  : 生の価格DataFrame (index=date, columns=ticker)
              ※ yfinanceから取得したそのままのデータを想定
    verbose : True の場合、各ラウンドの詳細を出力

    Returns
    -------
    cleaned_prices : クリーニング済み価格DataFrame
    report         : 品質レポート辞書
    """
    report: Dict = {}
    n_stocks_orig = prices.shape[1]
    n_days        = len(prices)

    def _log(msg: str) -> None:
        if verbose:
            print(f"  [DataCleaner] {msg}")

    _log(f"クリーニング開始: {n_stocks_orig}銘柄 × {n_days}日")
    _log(f"期間: {prices.index[0].date()} 〜 {prices.index[-1].date()}")

    # ─────────────────────────────────────────────────────────────────────
    # 前処理: bfillなし・ffillのみ（上場前をNaNのまま保持）
    # ─────────────────────────────────────────────────────────────────────
    # dropna(how='all') → すべての銘柄がNaNの日付（非営業日）を除去
    # ffill(limit=FFILL_LIMIT) → 祝日など短期欠損のみ前日終値で補完
    # bfill は「上場前」に IPO価格を埋めてしまうため使用しない
    df = prices.dropna(how="all").ffill(limit=FFILL_LIMIT)

    # 各銘柄の「上場日（初回有効価格日）」を記録
    ipo_dates = df.apply(lambda col: col.first_valid_index())
    report["ipo_dates"] = ipo_dates

    _log(f"上場前NaN保持完了（bfillなし）")

    # ─────────────────────────────────────────────────────────────────────
    # Round 1: 確実なエラーの除去
    # ─────────────────────────────────────────────────────────────────────
    _log("── Round 1: 明確なエラー除去 ──────────────────────────────")

    removed_r1 = 0

    # (1a) ゼロ・負の価格を除去
    zero_neg = df <= 0
    n_zero = int(zero_neg.sum().sum())
    if n_zero > 0:
        df = df.where(~zero_neg)
        _log(f"  ゼロ/負の価格: {n_zero}データ点 → NaN化")
    removed_r1 += n_zero

    # (1b) ハードキャップ超えの日次リターン → その日の価格をNaN化
    ret = df.pct_change()
    hard_bad = ret.abs() > DAILY_RETURN_HARD_CAP
    n_hard = int(hard_bad.sum().sum())
    if n_hard > 0:
        # 異常リターンの日（終値が跳んでいる日）と
        # その前日（参照価格が誤っている可能性）をNaN化
        df = df.where(~hard_bad)
        df = df.where(~hard_bad.shift(-1, fill_value=False))
        _log(f"  ±{DAILY_RETURN_HARD_CAP*100:.0f}%超の日次リターン: "
             f"{n_hard}データ点 → 当日+前日をNaN化")
    removed_r1 += n_hard * 2

    # ffillで短期欠損を補完
    df = df.ffill(limit=FFILL_LIMIT)

    report["round1_removed"] = removed_r1
    _log(f"  Round 1 完了: 合計{removed_r1}データ点を除去")

    # ─────────────────────────────────────────────────────────────────────
    # Round 2: 高度なパターン検出
    # ─────────────────────────────────────────────────────────────────────
    _log("── Round 2: スパイク・停滞パターン検出 ─────────────────────")

    removed_r2 = 0

    # (2a) 価格スパイク検出:
    # 「ある日に+X%上昇し、翌日に-X%以上下落」または逆
    # = 誤ったデータが1日だけ混入しているケース（データプロバイダのエラー）
    ret2 = df.pct_change()

    # スパイクアップ: 今日 > +SOFT_CAP かつ 翌日 < -SOFT_CAP (元に戻る)
    spike_up_today    = ret2 > DAILY_RETURN_SOFT_CAP
    spike_up_next     = ret2.shift(-1) < -DAILY_RETURN_SOFT_CAP
    spike_up          = spike_up_today & spike_up_next

    # スパイクダウン: 今日 < -SOFT_CAP かつ 翌日 > +SOFT_CAP (元に戻る)
    spike_dn_today    = ret2 < -DAILY_RETURN_SOFT_CAP
    spike_dn_next     = ret2.shift(-1) > DAILY_RETURN_SOFT_CAP
    spike_dn          = spike_dn_today & spike_dn_next

    spike_mask = spike_up | spike_dn
    n_spikes = int(spike_mask.sum().sum())
    if n_spikes > 0:
        df = df.where(~spike_mask)
        df = df.where(~spike_mask.shift(-1, fill_value=False))
        _log(f"  価格スパイク: {n_spikes}件 → スパイク日+翌日をNaN化")
    removed_r2 += n_spikes * 2

    # (2b) 長期停滞価格検出:
    # 30日間のうち25日以上、リターンがゼロの銘柄 → 取引停止または誤データ
    ret3 = df.pct_change()
    zero_ret = (ret3 == 0) | ret3.isna()
    stale_score = zero_ret.rolling(STALE_WINDOW, min_periods=STALE_WINDOW // 2).sum()
    stale_mask  = stale_score >= STALE_ZERO_THRESHOLD

    n_stale = int(stale_mask.sum().sum())
    if n_stale > 0:
        df = df.where(~stale_mask)
        _log(f"  長期停滞価格（{STALE_WINDOW}日中{STALE_ZERO_THRESHOLD}日ゼロ返): "
             f"{n_stale}データ点 → NaN化")
    removed_r2 += n_stale

    # ffill再適用
    df = df.ffill(limit=FFILL_LIMIT)

    report["round2_removed"] = removed_r2
    _log(f"  Round 2 完了: 合計{removed_r2}データ点を除去")

    # ─────────────────────────────────────────────────────────────────────
    # Round 3: 最終確認・銘柄レベルのフィルタリング
    # ─────────────────────────────────────────────────────────────────────
    _log("── Round 3: 最終確認・カバレッジフィルタ ───────────────────")

    # (3a) 残存する異常リターンを再チェック
    ret_final = df.pct_change()
    remaining_extreme = (ret_final.abs() > DAILY_RETURN_HARD_CAP).sum().sum()
    _log(f"  残存する±{DAILY_RETURN_HARD_CAP*100:.0f}%超リターン: {remaining_extreme}件")

    # (3b) データカバレッジ（有効日数）が少ない銘柄を除去
    valid_days_per_stock = df.notna().sum()
    total_days = len(df)

    # 最低取引日数チェック
    has_min_days = valid_days_per_stock >= MIN_TRADING_DAYS
    # 最低カバレッジ率チェック
    coverage = valid_days_per_stock / total_days
    has_min_coverage = coverage >= MIN_DATA_COVERAGE

    keep_mask = has_min_days & has_min_coverage
    n_removed_coverage = int((~keep_mask).sum())

    if n_removed_coverage > 0:
        df = df.loc[:, keep_mask]
        _log(f"  カバレッジ不足で除去: {n_removed_coverage}銘柄 "
             f"(最低{MIN_TRADING_DAYS}日 or {MIN_DATA_COVERAGE*100:.0f}%未満)")

    # (3c) IPO後の期間が短すぎる銘柄の警告（除去はしないが記録）
    ipo_dates_final = df.apply(lambda col: col.first_valid_index())
    last_date = df.index[-1]
    ipo_recency = pd.to_datetime(last_date) - pd.to_datetime(ipo_dates_final)
    recent_ipo = ipo_recency < pd.Timedelta(days=365)
    n_recent = int(recent_ipo.sum())
    if n_recent > 0:
        _log(f"  直近1年以内にIPOした銘柄（訓練データ少）: {n_recent}銘柄 ※利用可能だが注意")

    report["round3_removed_coverage"] = n_removed_coverage
    report["round3_remaining_extreme"] = int(remaining_extreme)
    report["round3_recent_ipo"] = n_recent

    # ─────────────────────────────────────────────────────────────────────
    # 品質サマリー
    # ─────────────────────────────────────────────────────────────────────
    n_stocks_final  = df.shape[1]
    n_removed_total = n_stocks_orig - n_stocks_final

    # 銘柄ごとの品質スコア計算
    final_ret = df.pct_change()
    stock_quality = pd.DataFrame({
        "ipo_date":       df.apply(lambda c: c.first_valid_index()),
        "last_date":      df.apply(lambda c: c.last_valid_index()),
        "valid_days":     df.notna().sum(),
        "coverage_pct":   (df.notna().sum() / total_days * 100).round(1),
        "max_daily_ret":  final_ret.max().round(4),
        "min_daily_ret":  final_ret.min().round(4),
        "ret_std_ann":    (final_ret.std() * np.sqrt(252)).round(4),
    })
    report["stock_quality"] = stock_quality

    report["summary"] = {
        "n_orig":           n_stocks_orig,
        "n_final":          n_stocks_final,
        "n_removed":        n_removed_total,
        "removed_pct":      round(n_removed_total / n_stocks_orig * 100, 1),
        "total_days":       n_days,
        "round1_removed":   report["round1_removed"],
        "round2_removed":   report["round2_removed"],
        "round3_removed":   n_removed_coverage,
        "remaining_extreme_returns": int(remaining_extreme),
    }

    _log(f"── クリーニング完了 ─────────────────────────────────────────")
    _log(f"  最終銘柄数: {n_stocks_final} / {n_stocks_orig} "
         f"({n_removed_total}銘柄除去, {n_removed_total/n_stocks_orig*100:.1f}%)")
    _log(f"  残存異常リターン: {remaining_extreme}件")

    # 年率ボラティリティの分布を確認（正常なら15%〜50%に集中するはず）
    vol_arr = stock_quality["ret_std_ann"].dropna()
    _log(f"  年率ボラ分布: 中央値={vol_arr.median():.1%}, "
         f"95%ile={vol_arr.quantile(0.95):.1%}, "
         f"最大値={vol_arr.max():.1%}")

    if vol_arr.max() > 3.0:
        n_high_vol = int((vol_arr > 3.0).sum())
        _log(f"  ⚠ 年率ボラ300%超の銘柄: {n_high_vol}銘柄 → 要確認")

    return df, report


# ─────────────────────────────────────────────────────────────────────────────
# 追加ユーティリティ
# ─────────────────────────────────────────────────────────────────────────────

def filter_by_ipo_date(
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    min_months_since_ipo: int = 13,
) -> pd.DataFrame:
    """
    as_of 時点において上場から min_months_since_ipo ヶ月以上経過した銘柄のみを返す。

    各戦略の generate_signals() 内で呼ぶことで、
    「上場直後でデータが少ない銘柄」を自動的に除外できる。

    Parameters
    ----------
    prices                : 価格DataFrame
    as_of                 : 評価基準日
    min_months_since_ipo  : 最低上場月数（デフォルト13: 12ヶ月モメンタム計算に必要）

    Returns
    -------
    フィルタ済みDataFrame
    """
    ipo_dates = prices.apply(lambda col: col.first_valid_index())
    cutoff    = as_of - pd.DateOffset(months=min_months_since_ipo)
    eligible  = ipo_dates[pd.to_datetime(ipo_dates) <= cutoff].index
    return prices[eligible]


def clip_returns_for_backtest(
    daily_ret: pd.DataFrame,
    hard_cap: float = 0.50,
) -> pd.DataFrame:
    """
    バックテスト用の日次リターンをクリッピング。

    clean_price_data() でほぼすべての異常値は除去されているが、
    万が一残った場合の最終安全網として機能する。

    Parameters
    ----------
    daily_ret : pct_change()で計算した日次リターン
    hard_cap  : 片方向の最大リターン（デフォルト50%）

    Returns
    -------
    クリッピング済みリターン（NaNはそのまま保持）
    """
    return daily_ret.clip(lower=-hard_cap, upper=hard_cap)


def save_quality_report(report: dict, path: str) -> None:
    """品質レポートをCSVで保存（デバッグ・記録用）"""
    if "stock_quality" in report:
        report["stock_quality"].to_csv(path, encoding="utf-8-sig")
        print(f"品質レポート保存: {path}")
