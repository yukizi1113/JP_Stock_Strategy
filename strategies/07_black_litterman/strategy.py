"""
戦略7: ブラック・リターマン + 逆最適化 (Black-Litterman + Inverse Optimization)

【理論背景】
ML and RL in Finance Course 3-4 に基づく。
"Inverse Optimization - A New Perspective on the Black-Litterman Model"

【データリーク対策】
- ML シグナル生成: 訓練データは予測月より過去のみ使用
- 訓練ウィンドウ: monthly_ret.iloc[-(train_window+1):-1] で最終月を除外
- ラベル: 各月内クロスセクション中央値（将来情報混入なし）
- スケーラー: 訓練データのみでfit、予測時は同スケーラーで変換

【Black-Litterman モデルの概要】
標準B-Lモデル:
1. 均衡リターン π = λ Σ w_mkt （逆最適化 = Inverse RL）
   λ: リスク回避係数
   Σ: 共分散行列
   w_mkt: 市場時価総額ウェイト

2. 投資家ビューの組み込み:
   P μ = Q + ε,   ε ~ N(0, Ω)
   P: ビュー行列, Q: ビューベクトル, Ω: ビューの不確実性

3. B-L 事後リターン:
   μ_BL = [(τΣ)^-1 + P'Ω^-1 P]^-1 [(τΣ)^-1 π + P'Ω^-1 Q]

4. 最適ウェイト:
   w* = (λΣ)^-1 μ_BL
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from typing import Dict, Optional
from backtest_engine import BaseStrategy


class BlackLittermanOptimizer:
    """ブラック・リターマンポートフォリオ最適化器"""

    def __init__(self, risk_aversion: float = 2.5, tau: float = 0.05):
        self.lam = risk_aversion
        self.tau = tau

    def equilibrium_returns(self, cov_matrix, market_weights):
        return self.lam * cov_matrix @ market_weights

    def posterior_returns(self, cov_matrix, market_weights, P, Q, omega=None):
        pi = self.equilibrium_returns(cov_matrix, market_weights)
        tau_sigma = self.tau * cov_matrix

        if omega is None:
            omega = self.tau * (P @ cov_matrix @ P.T)

        ts_inv      = np.linalg.inv(tau_sigma)
        omega_inv   = np.linalg.inv(omega)
        post_cov_inv = ts_inv + P.T @ omega_inv @ P
        post_cov     = np.linalg.inv(post_cov_inv)
        return post_cov @ (ts_inv @ pi + P.T @ omega_inv @ Q)

    def optimal_weights(self, cov_matrix, expected_returns, constraints=None):
        n = len(expected_returns)
        if constraints is None:
            constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        bounds = [(0.0, 0.3)] * n
        w0 = np.ones(n) / n

        def neg_utility(w):
            return -(expected_returns @ w - 0.5 * self.lam * w @ cov_matrix @ w)

        result = minimize(
            neg_utility, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        if result.success:
            weights = np.clip(result.x, 0, None)
            return weights / weights.sum() if weights.sum() > 0 else np.ones(n) / n
        return np.ones(n) / n


class BlackLittermanStrategy(BaseStrategy):
    """
    ブラック・リターマン + ML ビュー戦略（リーク修正版）。

    データリーク対策:
    - 訓練ウィンドウは [-(train_window+1):-1] で最終月を除外
    - ラベルは各月クロスセクション中央値以上=1（グローバル閾値なし）
    - スケーラーは訓練データのみでfit

    Parameters
    ----------
    top_n : 保有銘柄数
    tau : B-L の τ パラメータ
    risk_aversion : リスク回避係数 λ
    view_confidence : ビューへの確信度（0-1）
    train_window_months : ML 訓練ウィンドウ（月数）
    """

    name = "ブラック・リターマン + MLビュー"

    def __init__(
        self,
        top_n: int = 20,
        tau: float = 0.05,
        risk_aversion: float = 2.5,
        view_confidence: float = 0.3,
        train_window_months: int = 24,
    ):
        self.top_n  = top_n
        self.tau    = tau
        self.lam    = risk_aversion
        self.conf   = view_confidence
        self.train_window = train_window_months
        self.optimizer = BlackLittermanOptimizer(risk_aversion, tau)

    def _ml_signals(self, monthly_ret: pd.DataFrame) -> pd.Series:
        """
        Random Forest で上昇確率を予測し、B-L ビューとして返す。

        リーク対策:
        - 訓練データ: monthly_ret.iloc[-(train_window+1):-1] （最終月を除外）
        - ラベル: 各月クロスセクション中央値以上=1（グローバル閾値なし）
        - スケーラー: 訓練データのみでfit
        """
        # 最終月を除いた訓練データ
        # （最終月のリターンは予測対象月の翌月情報になりうるため除外）
        train_data = monthly_ret.iloc[-(self.train_window + 1):-1]

        if len(train_data) < 14:
            return pd.Series()

        X_list, y_list = [], []

        # 月次イテレーション（各月の特徴量で翌月を予測）
        for i in range(12, len(train_data) - 1):
            next_ret   = train_data.iloc[i + 1]
            # ────────────────────────────────────────────────────────
            # クロスセクション中央値でラベル付け
            # 同月に同時確定する全銘柄の相対順位 → 将来情報でない
            # ────────────────────────────────────────────────────────
            cross_median = next_ret.median()

            for ticker in train_data.columns:
                ts = train_data[ticker].iloc[:i+1].dropna()
                if len(ts) < 12 or ticker not in next_ret.index:
                    continue
                if pd.isna(next_ret[ticker]):
                    continue
                try:
                    f = [
                        float((1 + ts.iloc[-3:]).prod() - 1),
                        float((1 + ts.iloc[-6:]).prod() - 1),
                        float((1 + ts.iloc[-12:]).prod() - 1),
                        float(ts.iloc[-12:].std()),
                        # JQuants 由来: ボラティリティ多期間
                        float(ts.iloc[-3:].std()),
                        float(ts.iloc[-6:].std()),
                        # 勝率
                        float((ts.iloc[-12:] > 0).mean()),
                    ]
                    if all(np.isfinite(v) for v in f):
                        X_list.append(f)
                        y_list.append(int(next_ret[ticker] >= cross_median))
                except Exception:
                    pass

        if len(X_list) < 30:
            return pd.Series()

        X = np.array(X_list)
        y = np.array(y_list)

        # 訓練データのみでスケーラーをfit（リークなし）
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = RandomForestClassifier(
            n_estimators=100, max_depth=4,
            max_features=0.3, random_state=42, n_jobs=-1
        )
        clf.fit(X_scaled, y)

        # 現時点の特徴量（最終月を含む全データから計算）
        current_scores = {}
        for ticker in monthly_ret.columns:
            ts = monthly_ret[ticker].dropna()
            if len(ts) < 12:
                continue
            try:
                f = [
                    float((1 + ts.iloc[-3:]).prod() - 1),
                    float((1 + ts.iloc[-6:]).prod() - 1),
                    float((1 + ts.iloc[-12:]).prod() - 1),
                    float(ts.iloc[-12:].std()),
                    float(ts.iloc[-3:].std()),
                    float(ts.iloc[-6:].std()),
                    float((ts.iloc[-12:] > 0).mean()),
                ]
                if all(np.isfinite(v) for v in f):
                    # 同じ scaler で変換（訓練データ統計を使用）
                    x = scaler.transform([f])
                    prob = clf.predict_proba(x)[0][1]
                    # ビュー期待値: (確率 - 0.5) × スケール → ±6%レンジ
                    current_scores[ticker] = (prob - 0.5) * 0.12
            except Exception:
                pass

        return pd.Series(current_scores)

    def generate_signals(
        self,
        prices: pd.DataFrame,
        as_of: pd.Timestamp,
        **kwargs,
    ) -> Dict[str, float]:

        monthly    = prices.resample("ME").last()
        monthly_ret = monthly.pct_change()

        if len(monthly_ret) < 24:
            return {}

        valid = monthly_ret.dropna(
            thresh=int(len(monthly_ret) * 0.5), axis=1
        ).columns
        if len(valid) < 10:
            return {}

        monthly_ret  = monthly_ret[valid]
        prices_valid = prices[valid]

        # 共分散行列（直近252日の日次リターン）
        daily_ret = prices_valid.pct_change().dropna(how="all")
        if len(daily_ret) < 60:
            return {}

        cov_matrix   = daily_ret.iloc[-252:].cov().values * 252
        n            = len(valid)
        market_weights = np.ones(n) / n

        ml_signals = self._ml_signals(monthly_ret)

        if ml_signals.empty:
            pi = self.optimizer.equilibrium_returns(cov_matrix, market_weights)
            weights_raw = self.optimizer.optimal_weights(cov_matrix, pi)
        else:
            view_tickers = [t for t in ml_signals.index if t in valid]
            if not view_tickers:
                weights_raw = market_weights
            else:
                ticker_list = list(valid)
                K = len(view_tickers)
                P = np.zeros((K, n))
                Q = np.zeros(K)
                Omega = np.zeros((K, K))

                for k, vt in enumerate(view_tickers):
                    if vt in ticker_list:
                        P[k, ticker_list.index(vt)] = 1.0
                    Q[k] = float(ml_signals[vt])
                    cov_idx = ticker_list.index(vt) if vt in ticker_list else 0
                    view_var = max(
                        (1 - self.conf) * self.tau * float(cov_matrix[cov_idx, cov_idx]),
                        1e-6
                    )
                    Omega[k, k] = view_var

                try:
                    post_ret = self.optimizer.posterior_returns(
                        cov_matrix, market_weights, P, Q, Omega
                    )
                    weights_raw = self.optimizer.optimal_weights(cov_matrix, post_ret)
                except Exception:
                    weights_raw = market_weights

        weights_series = pd.Series(weights_raw, index=valid)
        top = weights_series.nlargest(self.top_n)
        total = top.sum()
        if total <= 0:
            return {}
        return dict(top / total)
