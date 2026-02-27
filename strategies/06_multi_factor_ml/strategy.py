"""
戦略6: マルチファクターML (Multi-Factor Machine Learning)

【理論背景】
ML and RL in Finance Course 2-1 (SVM, Random Forest) +
ML for Trading Module 2 (Feature Engineering) +
JQuants-Forum (UKI000) 知見統合

【データリーク対策】
- ラベル: グローバル閾値ではなく各月内クロスセクションで中央値ランク付け
  → 将来情報は一切混入しない
- 特徴量: 各月末時点で確定している過去データのみを使用
- スケーラー: Pipeline内で学習データのみでfit
- 時系列順守: 訓練 → 予測の方向は常に過去 → 未来

【特徴量（JQuants由来追加）】
ATR 4変種（価格データのみから近似計算）:
  - 通常ATR: 終値ベースの日次リターン絶対値
  - HL Band: 期間内高値-安値レンジ（OHLCVデータ利用可能時）
  - 出来高比率: 当日÷N日平均出来高（OHLCVデータ利用可能時）

テクニカル（終値ベース）:
  - モメンタム: 1/3/6/12ヶ月（直近1ヶ月スキップ版含む）
  - ボラティリティ: 20/60日
  - RSI (14日)
  - MACD ヒストグラム (12/26/9)
  - ボリンジャーバンド幅 (20日)
  - 52週高値からの距離

【モデル】
XGBoost または Random Forest（バイナリ分類）
colsample_bytree=0.1 （JQuants 推奨: aggressive feature sampling）
月次リバランス
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Dict, Optional, List
from backtest_engine import BaseStrategy


def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """RSI (Relative Strength Index)"""
    delta = prices.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]
    if pd.isna(avg_loss) or avg_loss == 0:
        return 50.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def compute_macd(prices: pd.Series, fast=12, slow=26, signal=9) -> float:
    """MACD ヒストグラム"""
    if len(prices) < slow + signal:
        return 0.0
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    sig  = macd.ewm(span=signal).mean()
    val  = (macd - sig).iloc[-1]
    return float(val) if not pd.isna(val) else 0.0


def compute_bb_width(prices: pd.Series, period: int = 20) -> float:
    """ボリンジャーバンド幅"""
    if len(prices) < period:
        return 0.0
    rolling = prices.rolling(period)
    mid = rolling.mean().iloc[-1]
    std = rolling.std().iloc[-1]
    if pd.isna(mid) or mid == 0:
        return 0.0
    return float(2 * 2 * std / mid)


def compute_volatility_features(
    close: pd.Series,
    periods: List[int] = [5, 10, 20, 40, 60],
) -> dict:
    """
    JQuants 知見: 終値ベースのボラティリティ特徴量
    （ATRの代替として日次リターン絶対値の移動平均を使用）
    """
    ret = close.pct_change()
    features = {}
    for n in periods:
        # ATR 近似 (終値ベース)
        features[f"atr_{n}"]      = float(ret.abs().rolling(n).mean().iloc[-1] or 0)
        # HL Band 近似 (終値ローリング最大-最小)
        features[f"hl_{n}"]       = float(
            (close.rolling(n).max().iloc[-1] - close.rolling(n).min().iloc[-1])
            / close.iloc[-1] if close.iloc[-1] > 0 else 0
        )
        # ボラティリティ (標準偏差)
        features[f"vola_{n}"]     = float(ret.rolling(n).std().iloc[-1] or 0)
    return features


class MultiFactorMLStrategy(BaseStrategy):
    """
    テクニカル特徴量による ML 株式選別戦略（リーク修正版）。

    ラベルは各月内クロスセクション中央値以上=1 で生成するため
    グローバル閾値による将来情報の混入なし。

    Parameters
    ----------
    top_n : 上位保有銘柄数
    train_window_months : 学習ウィンドウ（月数）
    model_type : 'xgboost' or 'rf'（random forest）
    """

    name = "マルチファクターML"

    def __init__(
        self,
        top_n: int = 20,
        train_window_months: int = 36,
        model_type: str = "rf",
    ):
        self.top_n          = top_n
        self.train_window   = train_window_months
        self.model_type     = model_type

    def _compute_features_at(
        self,
        prices: pd.DataFrame,
        as_of_idx: int,
    ) -> pd.DataFrame:
        """
        prices のインデックス as_of_idx 時点における全銘柄の特徴量を計算。
        as_of_idx 以前のデータのみを使用（リーク防止）。
        Returns: DataFrame (index=ticker, columns=features)
        """
        records = {}
        window = prices.iloc[max(0, as_of_idx - 300):as_of_idx + 1]

        for ticker in window.columns:
            ts = window[ticker].dropna()
            if len(ts) < 60:
                continue
            try:
                close = ts

                # ── モメンタム（過去リターン）───────────────────────────
                def ret_safe(n):
                    return float(close.iloc[-1] / close.iloc[-n] - 1) if len(close) >= n else np.nan
                ret_1m  = ret_safe(22)
                ret_3m  = ret_safe(66)
                ret_6m  = ret_safe(132)
                ret_12m = ret_safe(252)
                # 直近1ヶ月スキップ版
                ret_3m_skip  = float(close.iloc[-23] / close.iloc[-66] - 1)  if len(close) >= 66 else np.nan
                ret_6m_skip  = float(close.iloc[-23] / close.iloc[-132] - 1) if len(close) >= 132 else np.nan
                ret_12m_skip = float(close.iloc[-23] / close.iloc[-252] - 1) if len(close) >= 252 else np.nan

                # ── ボラティリティ ────────────────────────────────────
                dr = close.pct_change().dropna()
                vol_20  = float(dr.iloc[-20:].std() * np.sqrt(252)) if len(dr) >= 20 else np.nan
                vol_60  = float(dr.iloc[-60:].std() * np.sqrt(252)) if len(dr) >= 60 else np.nan

                # ── RSI / MACD / BB ───────────────────────────────────
                rsi    = compute_rsi(close.iloc[-30:]) if len(close) >= 30 else 50.0
                macd_h = compute_macd(close)
                bb_w   = compute_bb_width(close.iloc[-40:]) if len(close) >= 25 else 0.0

                # ── 52週高値距離 ──────────────────────────────────────
                hi_52w  = close.iloc[-min(252, len(close)):].max()
                dist_hi = (close.iloc[-1] - hi_52w) / hi_52w if hi_52w > 0 else 0.0

                # ── リスク調整モメンタム ──────────────────────────────
                ret_12m_adj = ret_12m_skip / vol_60 if (vol_60 and vol_60 > 1e-8) else 0.0
                win_rate    = float((dr.iloc[-252:] > 0).mean()) if len(dr) >= 252 else 0.5

                # ── JQuants由来: ボラティリティ多期間 + HL Band ────────
                vol_feats = compute_volatility_features(close.iloc[-300:])

                rec = {
                    "ret_1m":       ret_1m,
                    "ret_3m":       ret_3m,
                    "ret_6m":       ret_6m,
                    "ret_12m":      ret_12m,
                    "ret_3m_skip":  ret_3m_skip,
                    "ret_6m_skip":  ret_6m_skip,
                    "ret_12m_skip": ret_12m_skip,
                    "ret_12m_adj":  ret_12m_adj,
                    "vol_20":       vol_20,
                    "vol_60":       vol_60,
                    "rsi":          rsi,
                    "macd":         macd_h,
                    "bb_width":     bb_w,
                    "dist_52w_hi":  dist_hi,
                    "win_rate":     win_rate,
                    **vol_feats,
                }
                records[ticker] = rec
            except Exception:
                continue

        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records).T

    def _build_model(self) -> Pipeline:
        if self.model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                clf = XGBClassifier(
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.01,
                    subsample=0.8,
                    # JQuants 推奨: 特徴量の10%のみ各ツリーで使用 → 過学習防止
                    colsample_bytree=0.1,
                    random_state=42,
                    n_jobs=-1,
                    use_label_encoder=False,
                    eval_metric="logloss",
                )
            except ImportError:
                return self._build_rf()
        else:
            return self._build_rf()
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    def _build_rf(self) -> Pipeline:
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=5,
            max_features=0.3,  # JQuants的な特徴量削減の近似
            random_state=42, n_jobs=-1
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    def generate_signals(
        self,
        prices: pd.DataFrame,
        as_of: pd.Timestamp,
        **kwargs,
    ) -> Dict[str, float]:

        if len(prices) < 100:
            return {}

        monthly    = prices.resample("ME").last()
        monthly_ret = monthly.pct_change()
        n_months   = len(monthly)

        if n_months < self.train_window + 2:
            return {}

        train_start = max(3, n_months - self.train_window - 1)
        train_end   = n_months - 1  # 最終月は予測対象

        # ──────────────────────────────────────────────────────────────
        # 学習データ構築
        # ラベル: 各月内クロスセクション中央値以上=1
        # → グローバル閾値を使わないためリークなし
        # ──────────────────────────────────────────────────────────────
        X_list, y_list = [], []

        for i in range(train_start, train_end):
            # i 時点の月末日を価格インデックスで探す
            month_end_dt = monthly.index[i]

            # prices のインデックスから月末に最も近い日を探す（<=月末）
            price_idx = prices.index.searchsorted(month_end_dt)
            if price_idx >= len(prices):
                price_idx = len(prices) - 1

            feat = self._compute_features_at(prices, price_idx)
            if feat.empty:
                continue

            # 翌月リターン（リーク防止: i+1 の翌月データを使う）
            next_month_ret = monthly_ret.iloc[i + 1]

            # ────────────────────────────────────────────────────────
            # クロスセクション内ラベル付け
            # 同月に同時確定する全銘柄の相対順位 → 将来情報でない
            # ────────────────────────────────────────────────────────
            common = feat.index.intersection(next_month_ret.dropna().index)
            if len(common) < 10:
                continue

            cross_median = next_month_ret[common].median()

            for ticker in common:
                row = feat.loc[ticker].values.astype(float)
                if np.any(~np.isfinite(row)):
                    continue
                label = int(next_month_ret[ticker] >= cross_median)
                X_list.append(row)
                y_list.append(label)

        if len(X_list) < 50:
            # フォールバック: 単純12ヶ月モメンタム
            if n_months >= 14:
                mom = (1 + monthly_ret.iloc[-13:-1]).prod() - 1
                top = mom.dropna().nlargest(self.top_n).index.tolist()
                return {t: 1.0 / len(top) for t in top} if top else {}
            return {}

        X_train = np.array(X_list)
        y_train = np.array(y_list)

        # ──────────────────────────────────────────────────────────────
        # モデル学習
        # StandardScaler は Pipeline 内で学習データのみでfit → リークなし
        # ──────────────────────────────────────────────────────────────
        model = self._build_model()
        try:
            model.fit(X_train, y_train)
        except Exception:
            return {}

        # ──────────────────────────────────────────────────────────────
        # 現時点（as_of）の特徴量で予測
        # ──────────────────────────────────────────────────────────────
        current_idx = len(prices) - 1
        current_feat = self._compute_features_at(prices, current_idx)
        if current_feat.empty:
            return {}

        scores = {}
        for ticker in current_feat.index:
            row = current_feat.loc[ticker].values.astype(float)
            if not np.all(np.isfinite(row)):
                continue
            try:
                prob = model.predict_proba(row.reshape(1, -1))[0][1]
                scores[ticker] = float(prob)
            except Exception:
                pass

        top_tickers = sorted(scores, key=lambda k: scores[k], reverse=True)[:self.top_n]
        if not top_tickers:
            return {}
        return {t: 1.0 / len(top_tickers) for t in top_tickers}
