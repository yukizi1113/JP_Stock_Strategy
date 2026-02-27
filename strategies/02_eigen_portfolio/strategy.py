"""
戦略2: PCA固有ポートフォリオ (Eigen Portfolio via PCA)

【理論背景】
ML and RL in Finance Course 2-2 に基づく。

PCAをリターン行列に適用し、市場の主要リスク因子を抽出する。
- 第1主成分（PC1）: 市場全体の動き（ベータリスク）
- 第2主成分（PC2）: バリュー vs グロース等のスタイルリスク
- 第3主成分（PC3）: セクター間の分散

固有ポートフォリオは各主成分に対する固有ベクトルのウェイト。
PC2/PC3を活用したマーケットニュートラル戦略を構築する。

【数学的定義】
リターン行列 R ∈ ℝ^(T×N) に対し、PCA:
  R = U Σ V^T  （SVD分解）
  V の各列 = 固有ポートフォリオのウェイトベクトル

【戦略】
- PC1: 市場ベータに連動 → スキップ
- PC2: 長（高PC2負荷） + 短（低PC2負荷）→ マーケットニュートラル
- ロング専用の場合: PC2上位 Top-N の等ウェイト
- 月次リバランス、60ヶ月ローリングウィンドウでPCA更新
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, Tuple
from backtest_engine import BaseStrategy


class EigenPortfolioStrategy(BaseStrategy):
    """
    PCA固有ポートフォリオ戦略。

    Parameters
    ----------
    pc_index : 使用する主成分のインデックス（0=PC1, 1=PC2...）
    top_n : 保有銘柄数
    window_months : PCA学習ウィンドウ（月数）
    long_only : True=ロング専用, False=ロングショート
    """

    name = "固有ポートフォリオ(PCA)"

    def __init__(
        self,
        pc_index: int = 1,
        top_n: int = 20,
        window_months: int = 36,
        long_only: bool = True,
    ):
        self.pc_index      = pc_index
        self.top_n         = top_n
        self.window_months = window_months
        self.long_only     = long_only

    def _fit_pca(self, monthly_ret: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        月次リターン行列にPCAを適用し、固有ベクトルと寄与率を返す。
        Returns: (eigenvectors shape=(N, N), explained_variance_ratio shape=(N,))
        """
        # 欠損値を列の平均で補完
        filled = monthly_ret.fillna(monthly_ret.mean())
        scaler = StandardScaler()
        X = scaler.fit_transform(filled)

        pca = PCA(n_components=min(10, filled.shape[1]))
        pca.fit(X)

        return pca.components_, pca.explained_variance_ratio_

    def generate_signals(
        self,
        prices: pd.DataFrame,
        as_of: pd.Timestamp,
        **kwargs,
    ) -> Dict[str, float]:

        monthly = prices.resample("ME").last()
        monthly_ret = monthly.pct_change()

        # 有効銘柄を絞る（欠損 50% 以上をドロップ）
        valid_cols = monthly_ret.dropna(thresh=int(len(monthly_ret) * 0.5), axis=1).columns
        monthly_ret = monthly_ret[valid_cols]

        if len(monthly_ret) < 24 or len(valid_cols) < 10:
            return {}

        # ローリングウィンドウ
        window = min(self.window_months, len(monthly_ret) - 1)
        ret_window = monthly_ret.iloc[-window:]

        try:
            components, var_ratio = self._fit_pca(ret_window)
        except Exception:
            return {}

        if self.pc_index >= len(components):
            return {}

        pc = components[self.pc_index]  # shape=(N,)
        tickers = monthly_ret.columns.tolist()

        # 固有ベクトルの各銘柄への負荷量
        loadings = pd.Series(pc, index=tickers)

        if self.long_only:
            # ロング専用: 最も高い負荷量の上位 top_n を等ウェイト
            top = loadings.nlargest(self.top_n).index.tolist()
            w = 1.0 / len(top)
            return {t: w for t in top}
        else:
            # ロングショート: 上位を買い、下位を売り（空売り制限を考慮）
            from data_loader import build_no_short_tickers
            try:
                no_short = build_no_short_tickers()
            except Exception:
                no_short = frozenset()

            top_long = loadings.nlargest(self.top_n // 2).index.tolist()

            # 空売り不可銘柄（制度信用銘柄・非制度信用銘柄）を除外
            shortable = loadings[~loadings.index.isin(no_short)]
            top_short = shortable.nsmallest(self.top_n // 2).index.tolist()

            weights = {}
            for t in top_long:
                weights[t] = 1.0 / len(top_long) * 0.5
            if top_short:
                for t in top_short:
                    weights[t] = -1.0 / len(top_short) * 0.5
            return weights


class AbsorptionRatioIndicator:
    """
    吸収比率（Absorption Ratio）の計算。

    AR = Σ Var(PC_k) / Σ Var(r_i)   for k=1..K
    K個の主成分でリスクの何割が説明されるかを示す。
    ARが高い ≒ 市場リスクが1つの方向に集中 ≒ システミックリスク高。

    参考: Kritzman et al. (2011) "Principal Components as a Measure of Systemic Risk"
    """

    def __init__(self, n_components: int = 5, window: int = 500):
        """
        n_components : 使用する主成分数
        window : ローリングウィンドウ（日数）
        """
        self.n_components = n_components
        self.window = window

    def compute(self, daily_ret: pd.DataFrame) -> pd.Series:
        """
        日次リターン行列から吸収比率の時系列を計算する。
        Returns: pd.Series (index=date, values=absorption_ratio)
        """
        daily_ret = daily_ret.dropna(axis=1, how="all").fillna(0)
        n = len(daily_ret)
        result = {}

        for i in range(self.window, n):
            window_ret = daily_ret.iloc[i - self.window:i]
            filled = window_ret.fillna(window_ret.mean())
            scaler = StandardScaler()
            X = scaler.fit_transform(filled)

            n_comp = min(self.n_components, X.shape[1])
            pca = PCA(n_components=n_comp)
            pca.fit(X)

            ar = pca.explained_variance_ratio_[:n_comp].sum()
            result[daily_ret.index[i]] = ar

        return pd.Series(result, name="absorption_ratio")
