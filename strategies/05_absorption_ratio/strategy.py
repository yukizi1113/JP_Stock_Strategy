"""
戦略5: 吸収比率リスクタイミング (Absorption Ratio Risk Management)

【理論背景】
ML and RL in Finance Course 2-4 (Absorption Ratio via PCA) に基づく。
Kritzman et al. (2011) の研究を日本株に適用。

吸収比率（AR）: 上位K個の主成分がリスク全体の何割を説明するか。
AR = Σ_k Var(PC_k) / Σ_i Var(r_i)

ARの意味:
  AR が高い: リスクが少数因子に集中 = 市場の脆弱性が高い = システミックリスク増大
  AR が低い: リスクが分散 = 市場が健全

【戦略のロジック】
1. ARのローリング計算（500日ウィンドウ、K=5主成分）
2. ARの標準化スコアを計算
3. AR_zscore > AR_threshold → リスクオフ（ポジション縮小）
4. AR_zscore < -AR_threshold → リスクオン（ポジション拡大）

オーバーレイとして使用:
  任意のベース戦略にAR調整を組み合わせてリスク管理を強化できる。

具体的なポジションサイジング:
  position_multiplier = 1 - sigmoid(AR_zscore * scale_factor)
  multiplier ∈ [0.2, 1.0]
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional
from backtest_engine import BaseStrategy


def compute_absorption_ratio(
    returns: pd.DataFrame,
    n_components: int = 5,
) -> float:
    """
    特定時点の吸収比率を計算する。
    returns: (T, N) 行列
    """
    filled = returns.fillna(returns.mean())
    if filled.shape[0] < 20 or filled.shape[1] < n_components:
        return np.nan

    scaler = StandardScaler()
    X = scaler.fit_transform(filled)
    n_comp = min(n_components, X.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(X)
    return float(pca.explained_variance_ratio_[:n_comp].sum())


class AbsorptionRatioStrategy(BaseStrategy):
    """
    吸収比率を活用したリスクタイミング戦略。

    ベース戦略: 日経225採用銘柄等ウェイト（パッシブ）
    AR調整: 高ARのとき現金比率を上げ、低ARのとき株式比率を上げる。

    Parameters
    ----------
    ar_window : AR計算ウィンドウ（日）
    ar_n_components : AR計算に使う主成分数
    ar_zscore_window : ARのz-score計算ウィンドウ
    ar_high_threshold : リスクオフの z-score しきい値（デフォルト 1.0）
    base_top_n : ベースポートフォリオの銘柄数
    min_invest_ratio : 最小投資比率（キャッシュ最大時）
    """

    name = "吸収比率リスクタイミング"

    def __init__(
        self,
        ar_window: int = 252,
        ar_n_components: int = 5,
        ar_zscore_window: int = 500,
        ar_high_threshold: float = 1.0,
        ar_low_threshold: float = -0.5,
        base_top_n: int = 30,
        min_invest_ratio: float = 0.3,
        max_invest_ratio: float = 1.0,
    ):
        self.ar_window          = ar_window
        self.ar_n_components    = ar_n_components
        self.ar_zscore_window   = ar_zscore_window
        self.ar_high_threshold  = ar_high_threshold
        self.ar_low_threshold   = ar_low_threshold
        self.base_top_n         = base_top_n
        self.min_invest_ratio   = min_invest_ratio
        self.max_invest_ratio   = max_invest_ratio
        self._ar_history: list = []

    def _compute_current_ar(self, daily_ret: pd.DataFrame) -> float:
        """直近 ar_window 日のリターンで AR を計算"""
        window_ret = daily_ret.iloc[-self.ar_window:]
        return compute_absorption_ratio(window_ret, self.ar_n_components)

    def _ar_zscore(self) -> float:
        """AR の z-score を履歴から計算"""
        if len(self._ar_history) < 20:
            return 0.0
        history = np.array(self._ar_history[-self.ar_zscore_window:])
        mu, sigma = history.mean(), history.std()
        if sigma < 1e-6:
            return 0.0
        return float((history[-1] - mu) / sigma)

    def _invest_ratio(self, ar_z: float) -> float:
        """
        AR z-score からポジション比率を決定。
        高AR → 比率↓  低AR → 比率↑
        sigmoid ベースのスムーズな調整。
        """
        # sigmoid: z が大きい（高リスク）→ 比率を下げる
        raw = 1.0 / (1.0 + np.exp(ar_z * 1.5))
        # スケール調整: [min_invest_ratio, max_invest_ratio] に収める
        ratio = self.min_invest_ratio + (self.max_invest_ratio - self.min_invest_ratio) * raw
        return float(np.clip(ratio, self.min_invest_ratio, self.max_invest_ratio))

    def generate_signals(
        self,
        prices: pd.DataFrame,
        as_of: pd.Timestamp,
        **kwargs,
    ) -> Dict[str, float]:

        # 日次リターン計算
        daily_ret = prices.pct_change().dropna(how="all")
        daily_ret = daily_ret.dropna(axis=1, thresh=int(len(daily_ret) * 0.5))

        if len(daily_ret) < self.ar_window + 10:
            return {}

        # 吸収比率を計算して履歴追記
        ar = self._compute_current_ar(daily_ret)
        if not np.isnan(ar):
            self._ar_history.append(ar)

        ar_z = self._ar_zscore()
        invest_ratio = self._invest_ratio(ar_z)

        # ベースポートフォリオ: 直近60日モメンタム上位 base_top_n
        if len(prices) >= 60:
            mom = (prices.iloc[-1] / prices.iloc[-60] - 1).dropna()
        else:
            mom = (prices.iloc[-1] / prices.iloc[0] - 1).dropna()

        # ボラティリティで調整（リスクパリティ的）
        vol = daily_ret.iloc[-60:].std()
        iv_weights = (1 / vol).replace([np.inf, -np.inf], np.nan).dropna()

        top_tickers = mom.nlargest(self.base_top_n).index.tolist()
        top_tickers = [t for t in top_tickers if t in iv_weights.index]

        if not top_tickers:
            return {}

        # 逆ボラティリティウェイト
        w_series = iv_weights[top_tickers]
        w_normalized = w_series / w_series.sum()

        # AR調整: invest_ratio の分だけ株式、残りをキャッシュ
        signals = {t: float(w) * invest_ratio for t, w in w_normalized.items()}

        return signals
