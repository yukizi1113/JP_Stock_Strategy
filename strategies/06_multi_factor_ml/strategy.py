"""
戦略6: マルチファクターML (Multi-Factor Machine Learning)

【理論背景】
ML and RL in Finance Course 2-1 (SVM, Random Forest) +
ML for Trading Module 2 (Feature Engineering) に基づく。

ファンダメンタル要因（EDINET由来）とテクニカル要因を組み合わせ、
XGBoost/Random Forest で翌月リターンの相対的な強さを予測する。

【特徴量（テクニカル）】
- モメンタム: 過去1/3/6/12ヶ月リターン
- ボラティリティ: 過去20/60日標準偏差
- RSI (14日)
- MACD (12/26/9)
- ボリンジャーバンド幅
- 価格÷52週高値

【特徴量（ファンダメンタル - EDINET利用可能時）】
- ROE, ROA
- P/E, P/B
- 売上高成長率
- 利益成長率
- 負債比率

【アルゴリズム】
1. 毎月末、各銘柄の特徴量を計算
2. XGBoostで翌月リターンの上位/下位分類（5分位）
3. 予測スコア Top-N 銘柄を均等保有
4. 月次リバランス
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Dict, Optional
from backtest_engine import BaseStrategy


def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """RSI (Relative Strength Index) を計算する"""
    delta = prices.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def compute_macd(prices: pd.Series, fast=12, slow=26, signal=9) -> float:
    """MACD ヒストグラムを計算する"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    sig  = macd.ewm(span=signal).mean()
    return float((macd - sig).iloc[-1])


def compute_bb_width(prices: pd.Series, period: int = 20) -> float:
    """ボリンジャーバンド幅 = (上限-下限)/中央"""
    rolling = prices.rolling(period)
    upper = rolling.mean() + 2 * rolling.std()
    lower = rolling.mean() - 2 * rolling.std()
    mid   = rolling.mean()
    bb_w  = ((upper - lower) / mid).iloc[-1]
    return float(bb_w) if not np.isnan(bb_w) else 0.0


class MultiFactorMLStrategy(BaseStrategy):
    """
    テクニカル + ファンダメンタル特徴量による ML 株式選別戦略。

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
        self.top_n = top_n
        self.train_window = train_window_months
        self.model_type = model_type
        self._model = None
        self._scaler = StandardScaler()

    def _compute_technical_features(
        self,
        prices: pd.DataFrame,
        as_of_idx: int,
    ) -> pd.DataFrame:
        """
        prices の as_of_idx 時点における全銘柄のテクニカル特徴量を返す。
        Returns: DataFrame (index=ticker, columns=features)
        """
        records = {}
        window_prices = prices.iloc[max(0, as_of_idx - 260):as_of_idx + 1]

        for ticker in prices.columns:
            ts = window_prices[ticker].dropna()
            if len(ts) < 60:
                continue

            try:
                # モメンタム
                ret_1m  = ts.iloc[-1] / ts.iloc[-22] - 1  if len(ts) >= 22 else np.nan
                ret_3m  = ts.iloc[-1] / ts.iloc[-66] - 1  if len(ts) >= 66 else np.nan
                ret_6m  = ts.iloc[-1] / ts.iloc[-132] - 1 if len(ts) >= 132 else np.nan
                ret_12m = ts.iloc[-1] / ts.iloc[-252] - 1 if len(ts) >= 252 else np.nan

                # モメンタムスキップ（直近1ヶ月除く）
                ret_12m_skip = ts.iloc[-23] / ts.iloc[-252] - 1 if len(ts) >= 252 else np.nan

                # ボラティリティ
                daily_ret = ts.pct_change().dropna()
                vol_20 = daily_ret.iloc[-20:].std() * np.sqrt(252) if len(daily_ret) >= 20 else np.nan
                vol_60 = daily_ret.iloc[-60:].std() * np.sqrt(252) if len(daily_ret) >= 60 else np.nan

                # RSI
                rsi = compute_rsi(ts.iloc[-30:]) if len(ts) >= 30 else 50.0

                # MACD
                macd_h = compute_macd(ts) if len(ts) >= 30 else 0.0

                # ボリンジャーバンド幅
                bb_w = compute_bb_width(ts.iloc[-40:]) if len(ts) >= 25 else 0.0

                # 52週高値からの距離
                hi_52w = ts.iloc[-min(252, len(ts)):].max()
                dist_hi = (ts.iloc[-1] - hi_52w) / hi_52w if hi_52w > 0 else 0.0

                # リスク調整モメンタム
                ret_12m_adj = ret_12m_skip / vol_60 if (vol_60 and vol_60 > 0) else 0.0

                # 勝率（過去12ヶ月の上昇日割合）
                win_rate = (daily_ret.iloc[-252:] > 0).mean() if len(daily_ret) >= 252 else 0.5

                records[ticker] = {
                    "ret_1m": ret_1m,
                    "ret_3m": ret_3m,
                    "ret_6m": ret_6m,
                    "ret_12m": ret_12m,
                    "ret_12m_skip": ret_12m_skip,
                    "ret_12m_adj": ret_12m_adj,
                    "vol_20": vol_20,
                    "vol_60": vol_60,
                    "rsi": rsi,
                    "macd": macd_h,
                    "bb_width": bb_w,
                    "dist_52w_hi": dist_hi,
                    "win_rate": win_rate,
                }
            except Exception:
                continue

        return pd.DataFrame(records).T

    def _build_model(self) -> Pipeline:
        if self.model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                clf = XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    use_label_encoder=False,
                    eval_metric="logloss",
                )
            except ImportError:
                clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
        else:
            clf = RandomForestClassifier(
                n_estimators=200, max_depth=5, min_samples_leaf=5,
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

        monthly = prices.resample("ME").last()
        monthly_ret = monthly.pct_change()

        if len(monthly) < self.train_window + 2:
            return {}

        # 月末インデックスを日次に対応させる
        monthly_idx = monthly.index

        # 学習データ構築
        X_list, y_list = [], []
        train_months = monthly_idx[-self.train_window - 1:-1]

        for i, dt in enumerate(train_months[:-1]):
            next_dt = train_months[i + 1]

            # dt 時点での価格インデックスを取得
            if dt not in prices.index:
                price_idx = prices.index.searchsorted(dt) - 1
            else:
                price_idx = prices.index.get_loc(dt)

            if price_idx < 60:
                continue

            feat = self._compute_technical_features(prices, price_idx)
            if feat.empty:
                continue

            if next_dt not in monthly_ret.index:
                continue
            next_month_ret = monthly_ret.loc[next_dt]

            for ticker in feat.index:
                if ticker not in next_month_ret.index or pd.isna(next_month_ret[ticker]):
                    continue
                row = feat.loc[ticker].values
                if np.any(np.isnan(row)):
                    continue
                X_list.append(row)
                y_list.append(next_month_ret[ticker])

        if len(X_list) < 50:
            # フォールバック: 単純モメンタム
            if len(prices) >= 252:
                mom = (prices.iloc[-1] / prices.iloc[-252] - 1).dropna()
                top = mom.nlargest(self.top_n).index.tolist()
                return {t: 1.0 / len(top) for t in top}
            return {}

        X = np.array(X_list)
        y = np.array(y_list)

        # バイナリラベル（中央値以上 = 上位グループ）
        threshold = np.nanmedian(y)
        y_bin = (y >= threshold).astype(int)

        # モデル学習
        model = self._build_model()
        try:
            model.fit(X, y_bin)
        except Exception as e:
            return {}

        # 現時点の特徴量で予測
        current_idx = len(prices) - 1
        current_feat = self._compute_technical_features(prices, current_idx)
        if current_feat.empty:
            return {}

        scores = {}
        for ticker in current_feat.index:
            row = current_feat.loc[ticker].values
            if np.any(np.isnan(row)):
                continue
            try:
                prob = model.predict_proba(row.reshape(1, -1))[0][1]
                scores[ticker] = prob
            except Exception:
                pass

        top_tickers = sorted(scores, key=lambda k: scores[k], reverse=True)[:self.top_n]
        if not top_tickers:
            return {}
        w = 1.0 / len(top_tickers)
        return {t: w for t in top_tickers}
