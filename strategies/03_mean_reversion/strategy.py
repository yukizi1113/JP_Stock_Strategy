"""
戦略3: 統計的裁定・平均回帰戦略 (Statistical Arbitrage via OU Process)

【理論背景】
ML for Trading Module 2-4 / ML and RL in Finance Course 3-4 に基づく。

Hurst指数（H < 0.5 → 平均回帰、H > 0.5 → トレンド追随）を用いて
平均回帰性を持つ銘柄・ペアを特定し、Ornstein-Uhlenbeck (OU) プロセスで
最適エントリー・エグジットポイントを決定する。

【Ornstein-Uhlenbeck プロセス】
dX_t = θ(μ - X_t)dt + σ dW_t

パラメータ:
  θ: 平均回帰速度（大きいほど速く平均に戻る）
  μ: 長期平均
  σ: ボラティリティ

離散近似: X_t - X_{t-1} = a + bX_{t-1} + ε_t
  → b = -θΔt, a = θμΔt

【Hurst指数の計算】
R/S法 (Rescaled Range Analysis):
E[R(n)/S(n)] ∝ n^H
H < 0.5: 平均回帰
H = 0.5: ランダムウォーク
H > 0.5: トレンド

【戦略】
1. Hurst指数 < 0.45 の銘柄を選別
2. OU プロセスのパラメータを推定
3. z-score = (価格 - μ_hat) / σ_hat を計算
4. z-score < -entry_z → 買い
   z-score > entry_z → 売り（ロング専用では空売りなし）
   |z-score| < exit_z → 決済
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional
from backtest_engine import BaseStrategy


def hurst_exponent(ts: np.ndarray, max_lag: int = 100) -> float:
    """
    R/S法でHurst指数を計算する。
    ts: 対数価格系列 or リターン系列
    Returns: H (0~1)
    """
    ts = np.asarray(ts, dtype=float)
    ts = ts[~np.isnan(ts)]
    if len(ts) < 20:
        return 0.5

    lags = range(2, min(max_lag, len(ts) // 2))
    rs_values = []
    for lag in lags:
        # R/S統計量を計算
        subseries = [ts[i:i+lag] for i in range(0, len(ts) - lag, lag)]
        rs = []
        for sub in subseries:
            if len(sub) < 2:
                continue
            m = np.mean(sub)
            deviation = np.cumsum(sub - m)
            r = np.max(deviation) - np.min(deviation)
            s = np.std(sub, ddof=1)
            if s > 0:
                rs.append(r / s)
        if rs:
            rs_values.append((np.log(lag), np.log(np.mean(rs))))

    if len(rs_values) < 5:
        return 0.5

    x = [v[0] for v in rs_values]
    y = [v[1] for v in rs_values]
    slope, _, _, _, _ = stats.linregress(x, y)
    return float(np.clip(slope, 0.0, 1.0))


def fit_ou_params(log_prices: np.ndarray) -> Tuple[float, float, float]:
    """
    OU パラメータ（θ, μ, σ）を最小二乗で推定する。
    dX = θ(μ - X)dt + σ dW
    離散化: ΔX = a + bX_{t-1} + ε
    θ = -b/Δt, μ = a/θΔt, σ = std(ε)/sqrt(Δt)
    """
    x = np.asarray(log_prices, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 10:
        return 0.0, float(np.nanmean(x)), float(np.nanstd(x))

    dx = np.diff(x)
    x_lag = x[:-1]

    # OLS
    X = np.column_stack([np.ones(len(x_lag)), x_lag])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, dx, rcond=None)
    except Exception:
        return 0.0, float(np.nanmean(x)), float(np.nanstd(x))

    a, b = beta
    dt = 1.0  # 1日

    theta  = max(-b / dt, 1e-6)
    mu     = a / (theta * dt) if theta > 0 else float(np.nanmean(x))
    resid  = dx - (a + b * x_lag)
    sigma  = float(np.std(resid)) / np.sqrt(dt)

    return theta, mu, sigma


class MeanReversionStrategy(BaseStrategy):
    """
    OU平均回帰戦略（シングル銘柄版・ロング専用）。

    Parameters
    ----------
    hurst_threshold : この値未満の銘柄のみ対象（デフォルト 0.45）
    entry_z : エントリーのz-scoreしきい値（デフォルト 1.5）
    exit_z  : エグジットのz-scoreしきい値（デフォルト 0.3）
    lookback : OU推定に使う日数（デフォルト 252）
    top_n   : 最大保有銘柄数
    """

    name = "OU平均回帰"

    def __init__(
        self,
        hurst_threshold: float = 0.45,
        entry_z: float = 1.5,
        exit_z: float = 0.3,
        lookback: int = 252,
        top_n: int = 15,
    ):
        self.hurst_threshold = hurst_threshold
        self.entry_z  = entry_z
        self.exit_z   = exit_z
        self.lookback = lookback
        self.top_n    = top_n

    def generate_signals(
        self,
        prices: pd.DataFrame,
        as_of: pd.Timestamp,
        **kwargs,
    ) -> Dict[str, float]:

        if len(prices) < self.lookback + 10:
            return {}

        recent = prices.iloc[-self.lookback:]

        buy_signals = {}

        for ticker in recent.columns:
            ts = recent[ticker].dropna()
            if len(ts) < 60:
                continue

            log_p = np.log(ts.values)

            # Hurst指数で平均回帰性を確認
            # 【修正】log価格（非定常）ではなく対数リターン（定常）で計算する。
            # log価格に適用すると小標本バイアスにより H≈1.0 となり全銘柄を排除してしまう。
            log_ret = np.diff(log_p)
            H = hurst_exponent(log_ret, max_lag=50)
            if H >= self.hurst_threshold:
                continue

            # OU パラメータ推定
            theta, mu, sigma = fit_ou_params(log_p)
            if sigma <= 0 or theta <= 0:
                continue

            # z-score 計算
            current_log_p = log_p[-1]
            z = (current_log_p - mu) / sigma

            # エントリー条件: 価格が平均より entry_z 以上下にある（過売れ）
            if z < -self.entry_z:
                # 収束スコア: 平均回帰速度 × |z-score|（大きいほど魅力的）
                score = theta * abs(z)
                buy_signals[ticker] = score

        if not buy_signals:
            return {}

        # スコア上位 top_n を等ウェイト
        top = sorted(buy_signals, key=lambda k: buy_signals[k], reverse=True)[:self.top_n]
        w = 1.0 / len(top)
        return {t: w for t in top}


class PairsTradingStrategy(BaseStrategy):
    """
    コインテグレーションに基づくペア取引戦略。

    共和分した2銘柄のスプレッドに対してOU平均回帰を適用する。
    日本株では同一セクター内のペアが有効。

    Parameters
    ----------
    pairs : [(ticker_a, ticker_b), ...] 事前定義ペア
    entry_z : エントリーz-score
    exit_z  : エグジットz-score
    lookback : スプレッド計算ウィンドウ
    """

    name = "ペア取引"

    def __init__(
        self,
        pairs: Optional[list] = None,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        lookback: int = 120,
    ):
        # デフォルトペア: 日本の競合企業ペア
        self.pairs = pairs or [
            ("7203", "7267"),  # トヨタ vs ホンダ
            ("9432", "9433"),  # NTT vs KDDI
            ("8306", "8316"),  # 三菱UFJ vs 三井住友
            ("6758", "6752"),  # ソニー vs パナソニック
        ]
        self.entry_z  = entry_z
        self.exit_z   = exit_z
        self.lookback = lookback

    def generate_signals(
        self,
        prices: pd.DataFrame,
        as_of: pd.Timestamp,
        **kwargs,
    ) -> Dict[str, float]:

        signals = {}

        for tkA, tkB in self.pairs:
            if tkA not in prices.columns or tkB not in prices.columns:
                continue

            p = prices[[tkA, tkB]].dropna()
            if len(p) < self.lookback + 10:
                continue

            recent = p.iloc[-self.lookback:]

            # OLS でヘッジ比率推定（log価格）
            logA = np.log(recent[tkA].values)
            logB = np.log(recent[tkB].values)

            X = np.column_stack([np.ones(len(logA)), logA])
            try:
                beta = np.linalg.lstsq(X, logB, rcond=None)[0]
            except Exception:
                continue

            hedge_ratio = beta[1]
            spread = logB - hedge_ratio * logA

            # z-score
            mu_s  = spread.mean()
            sd_s  = spread.std()
            if sd_s <= 0:
                continue

            z = (spread[-1] - mu_s) / sd_s

            # Hurst 確認
            H = hurst_exponent(spread, max_lag=30)
            if H >= 0.5:
                continue

            if z < -self.entry_z:
                # スプレッド縮小 → A売り B買い（ロング専用モードでは B のみ買い）
                signals[tkB] = 0.5
            elif z > self.entry_z:
                # スプレッド拡大 → A買い B売り
                signals[tkA] = 0.5

        # 正規化
        total = sum(signals.values())
        if total > 0:
            signals = {k: v / total for k, v in signals.items()}
        return signals
