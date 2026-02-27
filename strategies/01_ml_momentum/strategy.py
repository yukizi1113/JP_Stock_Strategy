"""
戦略1: MLファクターモメンタム (Cross-Sectional Momentum via Random Forest)

【理論背景】
ML for Trading Module 2 / ML and RL in Finance Course 2-1 に基づく。

クロスセクショナルモメンタムは、過去リターンの強い銘柄を買い、
弱い銘柄を売る戦略。単純モメンタムをランダムフォレストで拡張し、
複数の特徴量を組み合わせてランキングスコアを生成する。

【特徴量】
- 過去1/3/6/12ヶ月リターン（スキップ直近1ヶ月）
- 過去20/60日ボラティリティ
- 過去20日出来高比率（流動性）
- 52週高値からの距離

【アルゴリズム】
1. 各月末、全銘柄の特徴量を計算
2. Random Forest でリターンの四分位数を予測（分類）
3. 予測スコア上位 Top-N 銘柄を等ウェイトで保有
4. 月次リバランス
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
        train_window: int = 36,  # 月
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

    def _compute_features(self, monthly_ret: pd.DataFrame) -> pd.DataFrame:
        """
        各月末における横断的な特徴量を計算する。
        Input: monthly_ret (index=month_end, columns=ticker)
        Output: MultiIndex DataFrame (month_end, ticker) → features
        """
        records = []
        for i in range(12, len(monthly_ret)):
            dt = monthly_ret.index[i]
            row = monthly_ret.iloc[:i+1]

            for ticker in monthly_ret.columns:
                s = row[ticker].dropna()
                if len(s) < 12:
                    continue

                # モメンタム特徴量（直近1ヶ月スキップ）
                m1  = s.iloc[-1]               # 直近1ヶ月
                m3  = (1 + s.iloc[-3:]).prod() - 1
                m6  = (1 + s.iloc[-6:]).prod() - 1
                m12 = (1 + s.iloc[-12:]).prod() - 1
                m3_skip  = (1 + s.iloc[-4:-1]).prod() - 1
                m6_skip  = (1 + s.iloc[-7:-1]).prod() - 1
                m12_skip = (1 + s.iloc[-13:-1]).prod() - 1

                # ボラティリティ
                vol3  = s.iloc[-3:].std()
                vol12 = s.iloc[-12:].std()

                # リスク調整モメンタム
                m12_adj = m12_skip / vol12 if vol12 > 0 else 0

                # 上昇月数比率（勝率）
                win_rate = (s.iloc[-12:] > 0).mean()

                records.append({
                    "date":   dt,
                    "ticker": ticker,
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
                })

        df = pd.DataFrame(records)
        if df.empty:
            return df
        return df.set_index(["date", "ticker"])

    def generate_signals(
        self,
        prices: pd.DataFrame,
        as_of: pd.Timestamp,
        **kwargs,
    ) -> Dict[str, float]:

        monthly = prices.resample("ME").last()
        monthly_ret = monthly.pct_change()

        # 特徴量行列を構築
        feat_df = self._compute_features(monthly_ret)
        if feat_df.empty:
            return {}

        # 目的変数: 翌月リターンの四分位数（0=下位, 1=上位）
        all_dates = feat_df.index.get_level_values("date").unique().sort_values()

        X_list, y_list = [], []
        for i, dt in enumerate(all_dates[:-1]):
            next_dt = all_dates[i + 1]
            try:
                x_row = feat_df.loc[dt]
            except KeyError:
                continue

            next_ret = monthly_ret.loc[next_dt] if next_dt in monthly_ret.index else None
            if next_ret is None:
                continue

            for ticker in x_row.index:
                if ticker not in next_ret.index or pd.isna(next_ret[ticker]):
                    continue
                X_list.append(x_row.loc[ticker].values)
                y_list.append(next_ret[ticker])

        if len(X_list) < 50:
            # データ不足 → 単純12ヶ月モメンタムにフォールバック
            return self._simple_momentum(monthly_ret)

        X = np.array(X_list)
        y = np.array(y_list)

        # バイナリラベル（中央値以上=1）
        threshold = np.median(y)
        y_bin = (y >= threshold).astype(int)

        # ローリング学習（最新 train_window ヶ月分）
        n_dates = len(all_dates)
        train_dates = all_dates[max(0, n_dates - self.train_window - 1):-1]

        X_train, y_train = [], []
        for dt in train_dates:
            try:
                xr = feat_df.loc[dt]
                idx_offset = list(all_dates).index(dt)
                if idx_offset + 1 >= len(all_dates):
                    continue
                next_dt = all_dates[idx_offset + 1]
                if next_dt not in monthly_ret.index:
                    continue
                nr = monthly_ret.loc[next_dt]
                for ticker in xr.index:
                    if ticker not in nr.index or pd.isna(nr[ticker]):
                        continue
                    X_train.append(xr.loc[ticker].values)
                    y_train.append(int(nr[ticker] >= threshold))
            except Exception:
                continue

        if len(X_train) < 30:
            return self._simple_momentum(monthly_ret)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # モデル学習
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(**self.rf_params)),
        ])
        pipe.fit(X_train, y_train)

        # 現在時点の特徴量で予測
        if as_of not in feat_df.index.get_level_values("date"):
            return self._simple_momentum(monthly_ret)

        current_feat = feat_df.loc[as_of]
        feature_cols = current_feat.columns.tolist()
        scores = {}
        for ticker in current_feat.index:
            try:
                x = current_feat.loc[ticker].values.reshape(1, -1)
                prob = pipe.predict_proba(x)[0][1]  # 上昇確率
                scores[ticker] = prob
            except Exception:
                pass

        # 上位 top_n を等ウェイト
        top_tickers = sorted(scores, key=lambda k: scores[k], reverse=True)[:self.top_n]
        if not top_tickers:
            return {}
        w = 1.0 / len(top_tickers)
        return {t: w for t in top_tickers}

    def _simple_momentum(self, monthly_ret: pd.DataFrame) -> Dict[str, float]:
        """フォールバック: 単純12ヶ月モメンタム"""
        if len(monthly_ret) < 13:
            return {}
        mom = (1 + monthly_ret.iloc[-13:-1]).prod() - 1
        mom = mom.dropna().sort_values(ascending=False)
        top = mom.head(self.top_n).index.tolist()
        if not top:
            return {}
        w = 1.0 / len(top)
        return {t: w for t in top}
