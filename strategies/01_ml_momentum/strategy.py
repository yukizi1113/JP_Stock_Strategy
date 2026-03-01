"""
戦略1: MLファクターモメンタム (Cross-Sectional Momentum via Random Forest)

【理論背景】
ML for Trading Module 2-4 / ML and RL in Finance Course 2-1 に基づく。

クロスセクショナルモメンタムは、過去リターンの強い銘柄を買い、
弱い銘柄を売る戦略。単純モメンタムをランダムフォレストで拡張し、
複数の特徴量を組み合わせてランキングスコアを生成する。

【データリーク対策】
- ラベル生成: グローバル閾値を使わず、各月の横断面（クロスセクション）内で
  中央値を計算してバイナリラベルを生成。将来情報は混入しない。
- 学習ウィンドウ: 訓練データは予測対象月より過去のみを使用。
- スケーラー: Pipeline内で学習データのみでfit。
- 時系列順守: 各月のラベルはその月の翌月リターンで生成し、
  訓練→検証の方向は常に過去→未来。

【特徴量】
- 過去1/3/6/12ヶ月リターン（直近1ヶ月スキップ版含む）
- 過去3/12ヶ月ボラティリティ
- リスク調整モメンタム（12ヶ月リターン÷ボラティリティ）
- 勝率（過去12ヶ月の上昇月割合）

【アルゴリズム】
1. 各月末、全銘柄の特徴量を計算
2. 学習ラベル: 各月内でクロスセクショナルに中央値以上=1を付与
3. Random Forest で上昇確率スコアを予測
4. 予測スコア上位 Top-N 銘柄を等ウェイトで保有
5. 月次リバランス
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Dict, Optional
from backtest_engine import BaseStrategy


class MLMomentumStrategy(BaseStrategy):
    """
    ランダムフォレストを用いたクロスセクショナルモメンタム戦略。

    ラベルはグローバル閾値でなく各月内クロスセクションで生成するため、
    データリークなし。

    Parameters
    ----------
    top_n : 保有銘柄数（上位N銘柄）
    train_window : 学習に使う月数（ローリングウィンドウ）
    rf_params : RandomForestClassifier のパラメータ
    """

    name = "MLモメンタム"

    def __init__(
        self,
        top_n: int = 20,
        train_window: int = 36,
        rf_params: Optional[dict] = None,
    ):
        self.top_n        = top_n
        self.train_window = train_window
        self.rf_params    = rf_params or {
            "n_estimators": 200,
            "max_depth": 5,
            "min_samples_leaf": 5,
            "random_state": 42,
            "n_jobs": -1,
        }

    def _compute_features_at(
        self,
        monthly_ret: pd.DataFrame,
        i: int,
    ) -> pd.DataFrame:
        """
        月次リターン行列のインデックス i 時点の横断面特徴量を計算する。
        i 以前のデータのみを使用（リーク防止）。
        Returns: DataFrame (index=ticker, columns=features)
        """
        records = {}
        for ticker in monthly_ret.columns:
            s = monthly_ret[ticker].iloc[:i+1].dropna()
            if len(s) < 12:
                continue
            try:
                m1  = float(s.iloc[-1])
                m3  = float((1 + s.iloc[-3:]).prod() - 1)
                m6  = float((1 + s.iloc[-6:]).prod() - 1)
                m12 = float((1 + s.iloc[-12:]).prod() - 1)
                # 直近1ヶ月スキップ版（過去2〜4ヶ月、過去2〜7ヶ月、過去2〜13ヶ月）
                m3_skip  = float((1 + s.iloc[-4:-1]).prod() - 1) if len(s) >= 4 else np.nan
                m6_skip  = float((1 + s.iloc[-7:-1]).prod() - 1) if len(s) >= 7 else np.nan
                m12_skip = float((1 + s.iloc[-13:-1]).prod() - 1) if len(s) >= 13 else np.nan

                vol3  = float(s.iloc[-3:].std())
                vol12 = float(s.iloc[-12:].std())
                m12_adj = m12_skip / vol12 if vol12 > 1e-8 else 0.0
                win_rate = float((s.iloc[-12:] > 0).mean())

                records[ticker] = {
                    "ret_1m": m1,
                    "ret_3m": m3,
                    "ret_6m": m6,
                    "ret_12m": m12,
                    "ret_3m_skip": m3_skip,
                    "ret_6m_skip": m6_skip,
                    "ret_12m_skip": m12_skip,
                    "vol_3m": vol3,
                    "vol_12m": vol12,
                    "ret_12m_adj": m12_adj,
                    "win_rate": win_rate,
                }
            except Exception:
                continue
        return pd.DataFrame(records).T if records else pd.DataFrame()

    def generate_signals(
        self,
        prices: pd.DataFrame,
        as_of: pd.Timestamp,
        **kwargs,
    ) -> Dict[str, float]:

        monthly = prices.resample("ME").last()
        monthly_ret = monthly.pct_change()

        # ── 学習前データ品質チェック ──────────────────────────────────────
        # 月次リターンも±80%でクリッピング（日次±50%でも月複利で大きくなりうる）
        # これにより外れ値が特徴量・ラベルに混入しないことを保証する
        monthly_ret = monthly_ret.clip(lower=-0.80, upper=0.80)

        # 上場前（NaN）の月はそのまま保持し、特徴量計算時に dropna() で除外
        # 各銘柄の有効月数を確認（12ヶ月未満は後段で自動除外）

        n_months = len(monthly_ret)
        if n_months < 14:
            return {}

        # ──────────────────────────────────────────────────────────────
        # 学習データ構築
        # ラベル: 各月内のクロスセクションで翌月リターンの中央値以上=1
        # → グローバル閾値を使わないためリークなし
        # ──────────────────────────────────────────────────────────────
        train_start = max(12, n_months - self.train_window - 1)
        train_end   = n_months - 1   # 最後の月は予測対象（ラベル未確定）

        X_train, y_train = [], []

        for i in range(train_start, train_end):
            # i 時点の特徴量（i 以前のデータのみ使用）
            feat_i = self._compute_features_at(monthly_ret, i)
            if feat_i.empty:
                continue

            # 翌月リターン（i+1 時点）
            next_ret = monthly_ret.iloc[i + 1]

            # ────────────────────────────────────────────────────────
            # クロスセクション内ラベル付け
            # その月の全銘柄を見てから中央値を決めることに注意:
            # これは「この月に同時に観察できる全銘柄の相対順位」であり
            # 将来情報ではない（全銘柄は i+1 時点で同時確定）。
            # ────────────────────────────────────────────────────────
            common_tickers = feat_i.index.intersection(next_ret.dropna().index)
            if len(common_tickers) < 10:
                continue

            cross_median = next_ret[common_tickers].median()

            for ticker in common_tickers:
                row = feat_i.loc[ticker].values
                if np.any(np.isnan(row)):
                    continue
                label = int(next_ret[ticker] >= cross_median)
                X_train.append(row)
                y_train.append(label)

        if len(X_train) < 50:
            return self._simple_momentum(monthly_ret)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # ──────────────────────────────────────────────────────────────
        # モデル学習
        # StandardScaler は Pipeline 内で学習データのみでfit → リークなし
        # ──────────────────────────────────────────────────────────────
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(**self.rf_params)),
        ])
        try:
            pipe.fit(X_train, y_train)
        except Exception:
            return self._simple_momentum(monthly_ret)

        # ──────────────────────────────────────────────────────────────
        # 現在時点（as_of = 最終月）の特徴量で予測
        # as_of 以前のデータのみを使用
        # ──────────────────────────────────────────────────────────────
        current_feat = self._compute_features_at(monthly_ret, n_months - 1)
        if current_feat.empty:
            return self._simple_momentum(monthly_ret)

        scores = {}
        for ticker in current_feat.index:
            row = current_feat.loc[ticker].values
            if np.any(np.isnan(row)):
                continue
            try:
                prob = pipe.predict_proba(row.reshape(1, -1))[0][1]
                scores[ticker] = prob
            except Exception:
                pass

        top_tickers = sorted(scores, key=lambda k: scores[k], reverse=True)[:self.top_n]
        if not top_tickers:
            return {}
        w = 1.0 / len(top_tickers)
        return {t: w for t in top_tickers}

    def _simple_momentum(self, monthly_ret: pd.DataFrame) -> Dict[str, float]:
        """フォールバック: 単純12ヶ月モメンタム（直近1ヶ月スキップ）"""
        if len(monthly_ret) < 14:
            return {}
        # 過去13〜2ヶ月前のリターン（直近1ヶ月スキップ）
        mom = (1 + monthly_ret.iloc[-13:-1]).prod() - 1
        mom = mom.dropna().sort_values(ascending=False)
        top = mom.head(self.top_n).index.tolist()
        if not top:
            return {}
        return {t: 1.0 / len(top) for t in top}
