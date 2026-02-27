"""
戦略7: ブラック・リターマン + 逆最適化 (Black-Litterman + Inverse Optimization)

【理論背景】
ML and RL in Finance Course 3-4 に基づく。
"Inverse Optimization - A New Perspective on the Black-Litterman Model"

【Black-Litterman モデルの概要】
標準B-Lモデル:
1. 均衡リターン π = λ Σ w_mkt （逆最適化）
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

【逆最適化（Inverse RL的アプローチ）】
観測されたポートフォリオウェイト（例: 市場時価総額比率）から、
その背後にある目的関数（均衡リターン）を逆算する。
これは逆強化学習（IRL）の金融応用。

【ビューの生成】
MLモデルのシグナルをB-Lビューとして組み込む:
  - ML予測が強気 → 正のビュー (Q > 0)
  - ML予測が弱気 → 負のビュー (Q < 0)
  - ビューの確信度 → Ω の対角要素（小さい = 確信度高）
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from typing import Dict, Optional, List
from backtest_engine import BaseStrategy


class BlackLittermanOptimizer:
    """
    ブラック・リターマンポートフォリオ最適化器。

    Parameters
    ----------
    risk_aversion : リスク回避係数 λ（デフォルト 2.5）
    tau : 事前確率への信頼度（デフォルト 0.05）
    """

    def __init__(self, risk_aversion: float = 2.5, tau: float = 0.05):
        self.lam = risk_aversion
        self.tau = tau

    def equilibrium_returns(
        self,
        cov_matrix: np.ndarray,
        market_weights: np.ndarray,
    ) -> np.ndarray:
        """
        逆最適化による均衡リターン π = λ Σ w_mkt
        """
        return self.lam * cov_matrix @ market_weights

    def posterior_returns(
        self,
        cov_matrix: np.ndarray,
        market_weights: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        omega: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        B-L 事後リターンを計算する。

        Parameters
        ----------
        cov_matrix : (N, N) 共分散行列
        market_weights : (N,) 市場ウェイト
        P : (K, N) ビュー行列（K=ビュー数）
        Q : (K,) ビュー期待リターン
        omega : (K, K) ビューの不確実性（None なら比例デフォルト）
        """
        pi = self.equilibrium_returns(cov_matrix, market_weights)
        tau_sigma = self.tau * cov_matrix

        if omega is None:
            # Idzorek (2005) のデフォルト: Ω = τ * P Σ P'
            omega = self.tau * (P @ cov_matrix @ P.T)

        # B-L 公式（Sherman-Morrison-Woodbury 形式）
        ts_inv = np.linalg.inv(tau_sigma)
        omega_inv = np.linalg.inv(omega)

        posterior_cov_inv = ts_inv + P.T @ omega_inv @ P
        posterior_cov = np.linalg.inv(posterior_cov_inv)

        posterior_ret = posterior_cov @ (ts_inv @ pi + P.T @ omega_inv @ Q)
        return posterior_ret

    def optimal_weights(
        self,
        cov_matrix: np.ndarray,
        expected_returns: np.ndarray,
        constraints: list = None,
    ) -> np.ndarray:
        """
        平均分散最適化: max(μ' w - λ/2 * w' Σ w)
        制約: Σw = 1, w >= 0 (ロング専用)
        """
        n = len(expected_returns)
        if constraints is None:
            constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

        bounds = [(0.0, 0.3)] * n  # 各銘柄最大30%
        w0 = np.ones(n) / n

        def neg_utility(w):
            ret  = expected_returns @ w
            risk = 0.5 * self.lam * w @ cov_matrix @ w
            return -(ret - risk)

        result = minimize(
            neg_utility, w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )

        if result.success:
            weights = np.clip(result.x, 0, None)
            return weights / weights.sum()
        else:
            return np.ones(n) / n


class BlackLittermanStrategy(BaseStrategy):
    """
    ブラック・リターマン + ML ビュー戦略。

    1. 市場時価総額ウェイト → 逆最適化で均衡リターン取得
    2. MLモデル（Random Forest）のシグナルをビューとして組み込む
    3. B-L 事後リターンで最適ウェイトを計算

    Parameters
    ----------
    n_stocks : ユニバースサイズ（最大銘柄数）
    top_n : 実際に保有する銘柄数
    tau : B-L のτパラメータ
    risk_aversion : リスク回避係数
    view_confidence : ビューへの確信度（0-1, 大きいほど ML を信頼）
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

    def _ml_signals(
        self,
        monthly_ret: pd.DataFrame,
    ) -> pd.Series:
        """
        ランダムフォレストでモメンタムスコアを計算し、
        B-L ビューの期待値として使用する。
        Returns: pd.Series (index=ticker, values=expected_excess_return)
        """
        if len(monthly_ret) < self.train_window + 2:
            return pd.Series()

        # 特徴量: 過去3/6/12ヶ月モメンタム
        X_list, y_list, tickers_list = [], [], []
        train_data = monthly_ret.iloc[-(self.train_window + 1):]

        for i in range(12, len(train_data) - 1):
            next_ret = train_data.iloc[i + 1]
            threshold = next_ret.median()
            for ticker in train_data.columns:
                ts = train_data[ticker].iloc[:i+1]
                if ts.isna().sum() > 3:
                    continue
                try:
                    f = [
                        float((1 + ts.iloc[-3:]).prod() - 1),
                        float((1 + ts.iloc[-6:]).prod() - 1),
                        float((1 + ts.iloc[-12:]).prod() - 1),
                        float(ts.iloc[-12:].std()),
                    ]
                    if not any(np.isnan(v) for v in f):
                        X_list.append(f)
                        y_list.append(int(next_ret[ticker] >= threshold) if ticker in next_ret.index else 0)
                        tickers_list.append(ticker)
                except Exception:
                    pass

        if len(X_list) < 30:
            return pd.Series()

        X = np.array(X_list)
        y = np.array(y_list)

        pipe = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pipe.fit(X_scaled, y)

        # 現時点の予測
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
                ]
                if not any(np.isnan(v) for v in f):
                    x = scaler.transform([f])
                    prob = pipe.predict_proba(x)[0][1]
                    # ビュー期待値: 確率を超過リターン期待値に変換
                    current_scores[ticker] = (prob - 0.5) * 0.12  # ±6%レンジ
            except Exception:
                pass

        return pd.Series(current_scores)

    def generate_signals(
        self,
        prices: pd.DataFrame,
        as_of: pd.Timestamp,
        **kwargs,
    ) -> Dict[str, float]:

        monthly = prices.resample("ME").last()
        monthly_ret = monthly.pct_change()

        if len(monthly_ret) < 24:
            return {}

        # 有効銘柄
        valid = monthly_ret.dropna(thresh=int(len(monthly_ret) * 0.5), axis=1).columns
        if len(valid) < 10:
            return {}

        monthly_ret = monthly_ret[valid]
        prices_valid = prices[valid]

        # 日次リターンから共分散行列を推定（252日）
        daily_ret = prices_valid.pct_change().dropna(how="all")
        if len(daily_ret) < 60:
            return {}

        daily_window = daily_ret.iloc[-252:]
        cov_matrix = daily_window.cov().values * 252  # 年率換算

        # 市場ウェイト: 等ウェイト（時価総額データなし）
        n = len(valid)
        market_weights = np.ones(n) / n

        # ML シグナルを B-L ビューとして使用
        ml_signals = self._ml_signals(monthly_ret)

        if ml_signals.empty:
            # ビューなしの均衡ポートフォリオ
            pi = self.optimizer.equilibrium_returns(cov_matrix, market_weights)
            weights_raw = self.optimizer.optimal_weights(cov_matrix, pi)
        else:
            # ビューを持つ銘柄のみ使用
            view_tickers = [t for t in ml_signals.index if t in valid]
            if not view_tickers:
                weights_raw = market_weights
            else:
                K = len(view_tickers)
                ticker_list = list(valid)
                P = np.zeros((K, n))
                Q = np.zeros(K)
                Omega = np.zeros((K, K))

                for k, vt in enumerate(view_tickers):
                    if vt in ticker_list:
                        P[k, ticker_list.index(vt)] = 1.0
                    Q[k] = float(ml_signals[vt])
                    # ビューの不確実性: 確信度が高いほど小さい
                    view_var = (1 - self.conf) * self.tau * float(cov_matrix[
                        ticker_list.index(vt), ticker_list.index(vt)
                    ]) if vt in ticker_list else 0.01
                    Omega[k, k] = max(view_var, 1e-6)

                try:
                    post_ret = self.optimizer.posterior_returns(
                        cov_matrix, market_weights, P, Q, Omega
                    )
                    weights_raw = self.optimizer.optimal_weights(cov_matrix, post_ret)
                except Exception:
                    weights_raw = market_weights

        # ウェイトの小さい銘柄をカット、top_n に絞る
        weights_series = pd.Series(weights_raw, index=valid)
        top = weights_series.nlargest(self.top_n)
        top = top / top.sum()
        return dict(top)
