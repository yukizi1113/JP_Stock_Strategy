"""
戦略9: Deep Portfolio Optimization via Distributional Prediction of Residual Factors
(Imajo, Minami, Ito, Nakagawa; AAAI 2021)

【論文の核心アルゴリズム】

① スペクトル残差の抽出（Spectral Residuals）
   - リターン行列を SVD 分解し、上位 C 個の主成分（市場ファクター・業種ファクター）を除去
   - 残差が銘柄固有の idiosyncratic リターン → 共分散行列がほぼ対角行列
   - 日本株で最適な C=20（論文 Table 4 より）

② フラクタルネット（FractalNet）による分布予測
   【振幅不変性（Amplitude Invariance）】
     ψ(x) = ||x|| · ψ̃(x / ||x||)
     → スケールが変わっても同じ形状を認識（ボラティリティクラスタリングに対応）
   【時間スケール不変性（Time-scale Invariance）】
     複数のスケールパラメータ τ で時系列をリサンプリングし、
     共通ネットワーク ψ₁ に通して平均 → フラクタル構造（自己相似性）を活用
   【分位点回帰（Quantile Regression）】
     Q-1 個の分位点を同時予測 → 分布の平均・分散を推定（非対称分布に対応）

③ ポートフォリオ最適化
   - 残差ファクターの共分散が対角行列 → Markowitz 最適解が解析的に得られる
   - 銘柄 j の重み: w_j = μ̂_j / σ̂²_j （平均をリスクで割る）
   - A_t^T で元のリターン空間に逆変換

【実装上の工夫・論文からの変更点】
   - H=128 日（論文は256日）: CPU での計算速度とのトレードオフ
   - L=4 スケール（論文は22スケール）: 主要スケールのみカバー
   - Q=16 分位点（論文は32分位点）
   - K=64, ψ₁=2層, ψ₂=3層（論文は K=256, ψ₁=3層, ψ₂=8層）
   - 6ヶ月ごとに再学習（毎月は計算コスト超過）
   - ロングオンリー: 信号上位 top_n 銘柄を等ウェイト保有
"""
from __future__ import annotations

import os, sys, warnings, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from backtest_engine import BaseStrategy


# ──────────────────────────────────────────────────────────────────────────────
# ① スペクトル残差の計算
# ──────────────────────────────────────────────────────────────────────────────

def compute_spectral_residuals(
    returns: np.ndarray,
    C: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SVD によるスペクトル残差の抽出。

    Parameters
    ----------
    returns : (T, S) 日次リターン行列
    C       : 除去する主成分数（市場・業種ファクター数）

    Returns
    -------
    residuals : (T, S) スペクトル残差行列
    V_res     : (S, S-C) 残差空間の基底行列
    """
    T, S = returns.shape
    C = min(C, S - 1)

    # 行平均を引いた行列に SVD を適用
    X = returns - returns.mean(axis=0, keepdims=True)

    # SVD: X = U Σ V^T  (Vt は S×S 直交行列)
    # numpy の svd は X = U @ diag(s) @ Vt を返す
    try:
        # S < T の場合は economy SVD
        if S <= T:
            _, s, Vt = np.linalg.svd(X, full_matrices=False)
        else:
            # T < S の場合、転置で計算
            _, s, Vt = np.linalg.svd(X.T, full_matrices=False)
            # Vt の形状調整（この場合 Vt は T×T なので X の列空間の基底は別）
            # 転置後に再計算
            _, s, Vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        # SVD が収束しない場合はゼロを返す
        return returns.copy(), np.eye(S)

    # Vt: (min(T,S), S)  → 上位 C 個を除去して投影行列を構成
    V = Vt.T  # (S, min(T,S))
    n_components = V.shape[1]
    C_actual = min(C, n_components - 1)

    # 残差空間の基底: 上位 C 個以外の列
    V_res = V[:, C_actual:]  # (S, n_components - C_actual)

    # 射影行列: A_res = V_res @ V_res^T
    A_res = V_res @ V_res.T  # (S, S)

    # スペクトル残差
    residuals = X @ A_res  # (T, S)

    return residuals, V_res


# ──────────────────────────────────────────────────────────────────────────────
# ② フラクタルネット
# ──────────────────────────────────────────────────────────────────────────────

class FractalNet(nn.Module):
    """
    フラクタルネット: 振幅不変性 × 時間スケール不変性を持つ分位点予測器。

    入力: (batch, H) 残差リターン系列
    出力: (batch, Q-1) 分位点予測値

    【振幅不変性の実装】
    ψ(x) = ||x|| · ψ̃(x / ||x||)
    ψ̃ は単位球面上で定義される任意の関数 → どんな ψ̃ でも全体が正斉次関数になる

    【時間スケール不変性の実装（リサンプリング機構）】
    各スケール τ について:
      (i)  累積和: z = cumsum(x)
      (ii) リサンプリング: z の末尾 ceil(τ·H) 点を H'+1 点に線形補間
      (iii) 差分: diff → H' 点のリターン列
      (iv)  スケーリング: τ^{-1/2} を乗算（Hurst 指数 H≈0.5 を仮定）
    全スケールに共通 ψ₁ を適用し平均 → ψ₂ で分位点出力
    """

    def __init__(
        self,
        H_in: int = 128,      # 入力系列長
        H_prime: int = 32,    # リサンプリング後の系列長
        K: int = 64,          # ψ₁ の出力次元
        Q_minus_1: int = 15,  # 推定する分位点数
        L: int = 4,           # スケールパラメータ数
        dropout_p: float = 0.3,
    ):
        super().__init__()
        self.H_in = H_in
        self.H_prime = H_prime
        self.Q_minus_1 = Q_minus_1

        # スケールパラメータ: τ_l = 4^{-l/(L-1)}, l=0,...,L-1
        # τ_0=1.0 (フルレゾリューション) から τ_{L-1}≈0.25 (1/4 レゾリューション) まで
        if L <= 1:
            self.taus = [1.0]
        else:
            self.taus = [4.0 ** (-l / (L - 1)) for l in range(L)]

        # ψ₁: H_prime → K (2 FC 層)
        # bias=True でも可（ψ(x)=||x||·ψ̃(x/||x||) の構造で全体の正斉次性を保証）
        self.psi1 = nn.Sequential(
            nn.Linear(H_prime, K),
            nn.BatchNorm1d(K),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(K, K),
            nn.BatchNorm1d(K),
            nn.ReLU(),
        )

        # ψ₂: K → Q-1 (3 FC 層)
        hidden = max(K // 2, Q_minus_1 * 2)
        self.psi2 = nn.Sequential(
            nn.Linear(K, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden, Q_minus_1),
        )

    def _resample(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        """
        スペクトル残差リターン系列 x を時間スケール τ でリサンプリング。

        Parameters
        ----------
        x   : (batch, H_in) リターン系列
        tau : スケールパラメータ (0 < tau ≤ 1)

        Returns
        -------
        (batch, H_prime) リサンプリング済みリターン系列
        """
        H = self.H_in
        H_prime = self.H_prime

        # (i) 累積和: z = cumsum(x)  → 対数価格に対応
        z = torch.cumsum(x, dim=1)  # (batch, H)

        # (ii) リサンプリング: 末尾 ceil(τH) 点を取り出し、H'+1 点に線形補間
        start_idx = max(0, int(math.floor((1.0 - tau) * H)))
        z_sub = z[:, start_idx:]  # (batch, len_sub)

        if z_sub.shape[1] < 2:
            # サブ系列が短すぎる場合はゼロを返す
            return torch.zeros(x.shape[0], H_prime, device=x.device, dtype=x.dtype)

        # F.interpolate expects (batch, channels, length)
        z_3d = z_sub.unsqueeze(1)                            # (batch, 1, len_sub)
        z_interp = F.interpolate(z_3d, size=H_prime + 1,
                                  mode='linear', align_corners=True)
        z_interp = z_interp.squeeze(1)                       # (batch, H_prime+1)

        # (iii) 差分 → リターン列
        dz = z_interp[:, 1:] - z_interp[:, :-1]             # (batch, H_prime)

        # (iv) スケーリング: τ^{-1/2} を乗算（Hurst 指数 ≈ 0.5 を仮定）
        dz = dz * (tau ** -0.5)

        return dz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, H_in) 残差リターン系列

        Returns
        -------
        quantiles : (batch, Q-1) 分位点予測値（元のスケールで）
        """
        # 振幅不変性: ψ(x) = ||x|| · ψ̃(x / ||x||)
        norms = x.norm(dim=1, keepdim=True).clamp(min=1e-8)  # (batch, 1)
        x_norm = x / norms                                    # 単位球面上に正規化

        # 各スケールで ψ₁ を適用して平均
        psi1_outputs = []
        for tau in self.taus:
            x_resampled = self._resample(x_norm, tau)        # (batch, H_prime)
            psi1_out = self.psi1(x_resampled)                # (batch, K)
            psi1_outputs.append(psi1_out)

        avg_out = torch.stack(psi1_outputs, dim=0).mean(dim=0)  # (batch, K)

        # ψ₂: K → Q-1
        quantiles_norm = self.psi2(avg_out)                  # (batch, Q-1)

        # スケールを元に戻す: ψ(x) = ||x|| · ψ̃(x/||x||)
        quantiles = quantiles_norm * norms                    # (batch, Q-1)

        return quantiles


# ──────────────────────────────────────────────────────────────────────────────
# ピンボールロス（分位点回帰損失）
# ──────────────────────────────────────────────────────────────────────────────

def pinball_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    alphas: torch.Tensor,
) -> torch.Tensor:
    """
    分位点回帰のピンボールロス。

    Parameters
    ----------
    y_pred  : (batch, Q-1) 予測分位点
    y_true  : (batch,) 真値
    alphas  : (Q-1,) 分位点レベル [α₁, ..., α_{Q-1}]

    Returns
    -------
    scalar loss
    """
    y_true_exp = y_true.unsqueeze(1)               # (batch, 1)
    errors = y_true_exp - y_pred                   # (batch, Q-1)
    alphas_exp = alphas.unsqueeze(0)               # (1, Q-1)
    loss = torch.where(
        errors >= 0,
        alphas_exp * errors,
        (alphas_exp - 1.0) * errors,
    )
    return loss.mean()


# ──────────────────────────────────────────────────────────────────────────────
# ③ DPO 戦略クラス
# ──────────────────────────────────────────────────────────────────────────────

class DeepPortfolioStrategy(BaseStrategy):
    """
    Deep Portfolio Optimization (DPO) 戦略。
    Imajo et al. (AAAI 2021) の実装。

    Parameters
    ----------
    top_n              : 保有銘柄数（ロングオンリー）
    H                  : ルックバック窓（日数）
    C                  : 除去する主成分数（日本株最適値=20）
    L                  : フラクタルネットのスケール数
    Q                  : 分位点数（Q-1 個を予測）
    H_prime            : リサンプリング後の系列長
    K                  : FractalNet 中間層次元
    train_window_months: 学習に使う月数
    retrain_interval   : 再学習間隔（月数）
    max_train_stocks   : 学習に使う最大銘柄数（メモリ節約）
    n_epochs           : 学習エポック数
    batch_size         : ミニバッチサイズ
    lr                 : Adam 学習率
    """

    name = "Deep Portfolio (DPO)"

    def __init__(
        self,
        top_n: int = 20,
        H: int = 128,
        C: int = 20,
        L: int = 4,
        Q: int = 16,
        H_prime: int = 32,
        K: int = 64,
        train_window_months: int = 36,
        retrain_interval: int = 6,
        max_train_stocks: int = 500,
        n_epochs: int = 8,
        batch_size: int = 1024,
        lr: float = 1e-3,
        dropout_p: float = 0.3,
    ):
        self.top_n = top_n
        self.H = H
        self.C = C
        self.L = L
        self.Q = Q
        self.H_prime = H_prime
        self.K = K
        self.train_window_months = train_window_months
        self.retrain_interval = retrain_interval
        self.max_train_stocks = max_train_stocks
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout_p = dropout_p

        # 分位点レベル: j/Q for j=1,...,Q-1
        self.alphas = np.array([j / Q for j in range(1, Q)])  # (Q-1,)

        # モデルとキャッシュ
        self._model: Optional[FractalNet] = None
        self._last_train_date: Optional[pd.Timestamp] = None
        self._call_count: int = 0
        self.device = torch.device("cpu")

    def _build_samples(
        self,
        residuals: np.ndarray,
        H: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        スペクトル残差行列からモデル学習用サンプルを生成。

        Parameters
        ----------
        residuals : (T, S) 残差リターン行列
        H         : ルックバック窓長

        Returns
        -------
        X : (N, H)  入力系列
        y : (N,)    翌日残差リターン（ラベル）
        """
        T, S = residuals.shape
        X_list, y_list = [], []

        for t in range(H, T - 1):
            x_t = residuals[t - H:t, :]    # (H, S)
            y_t = residuals[t + 1, :]      # (S,) 翌日残差

            # 各銘柄を独立サンプルとして追加（共通モデル ψ を銘柄間で共有）
            X_list.append(x_t.T)           # (S, H)
            y_list.append(y_t)             # (S,)

        if not X_list:
            return np.empty((0, H)), np.empty((0,))

        X = np.concatenate(X_list, axis=0).astype(np.float32)  # (N*S, H)
        y = np.concatenate(y_list, axis=0).astype(np.float32)  # (N*S,)
        return X, y

    def _train(self, residuals_train: np.ndarray) -> None:
        """
        FractalNet を残差リターン系列で学習する。

        Parameters
        ----------
        residuals_train : (T, S) 学習期間の残差リターン行列
        """
        T, S = residuals_train.shape

        # 銘柄数を制限（メモリ節約）
        if S > self.max_train_stocks:
            idx = np.random.choice(S, self.max_train_stocks, replace=False)
            residuals_train = residuals_train[:, idx]

        # サンプル生成
        X, y = self._build_samples(residuals_train, self.H)
        if len(X) < self.batch_size:
            return

        # データローダー
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)
        alphas_t = torch.tensor(self.alphas, dtype=torch.float32)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size,
                            shuffle=True, drop_last=True)

        # モデル初期化
        self._model = FractalNet(
            H_in=self.H,
            H_prime=self.H_prime,
            K=self.K,
            Q_minus_1=len(self.alphas),
            L=self.L,
            dropout_p=self.dropout_p,
        ).to(self.device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_epochs
        )

        self._model.train()
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            n_batches = 0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                pred = self._model(xb)                      # (batch, Q-1)
                loss = pinball_loss(pred, yb, alphas_t)
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            scheduler.step()

        self._model.eval()

    def _predict_signals(
        self,
        residuals_current: np.ndarray,
        H: int,
    ) -> np.ndarray:
        """
        現時点の残差リターン系列からシグナル（μ/σ²）を計算。

        Parameters
        ----------
        residuals_current : (T, S) 現在までの残差リターン
        H                 : ルックバック窓

        Returns
        -------
        signals : (S,) 各銘柄のシグナル値（大きいほど買い）
        """
        T, S = residuals_current.shape
        if T < H:
            return np.zeros(S)

        # 現在の窓: 最新 H 日分の残差リターン
        x_current = residuals_current[-H:, :].T.astype(np.float32)  # (S, H)
        x_t = torch.from_numpy(x_current).to(self.device)

        self._model.eval()
        with torch.no_grad():
            # バッチ処理（全銘柄を一括予測）
            batch_size = 512
            quantiles_list = []
            for i in range(0, S, batch_size):
                xb = x_t[i:i + batch_size]
                qb = self._model(xb)                        # (batch, Q-1)
                quantiles_list.append(qb.cpu().numpy())
            quantiles = np.concatenate(quantiles_list, axis=0)  # (S, Q-1)

        # 平均（期待リターン）と分散（リスク）を分位点から推定
        mu = quantiles.mean(axis=1)                         # (S,)
        var = quantiles.var(axis=1) + 1e-8                  # (S,)

        # シグナル: μ / σ² （Markowitz 最適化の解析解）
        signals = mu / var                                   # (S,)
        return signals

    def generate_signals(
        self,
        prices: pd.DataFrame,
        as_of: pd.Timestamp,
        **kwargs,
    ) -> Dict[str, float]:
        """
        DPO による銘柄選択シグナルを生成。

        Parameters
        ----------
        prices : (T, S) as_of 以前の全価格データ
        as_of  : 現在の評価日

        Returns
        -------
        {ticker: weight} 上位 top_n 銘柄を等ウェイトで保有
        """
        self._call_count += 1

        # ──────────────────────────────────────────────────────
        # データ準備
        # ──────────────────────────────────────────────────────
        # 学習窓を制限して計算量を削減
        train_days = self.train_window_months * 21 + self.H
        prices_win = prices.iloc[-min(train_days, len(prices)):]

        # 欠損率が高い銘柄を除外（75%以上データがある銘柄のみ）
        valid_cols = prices_win.columns[
            prices_win.notna().mean() >= 0.75
        ]
        if len(valid_cols) < 10:
            return {}

        prices_clean = prices_win[valid_cols].ffill(limit=10)
        # 日次リターンをクリッピング（±50%超は外れ値）してからSVDへ
        returns = (
            prices_clean.pct_change()
            .clip(lower=-0.50, upper=0.50)
            .dropna(how="all")
            .values.astype(np.float32)
        )
        tickers = list(valid_cols)

        if returns.shape[0] < self.H + 30:
            return {}

        # ──────────────────────────────────────────────────────
        # スペクトル残差の抽出
        # ──────────────────────────────────────────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            residuals, _ = compute_spectral_residuals(returns, C=self.C)

        # ──────────────────────────────────────────────────────
        # モデルの学習（初回 or retrain_interval 月ごと）
        # ──────────────────────────────────────────────────────
        needs_train = (
            self._model is None
            or self._last_train_date is None
            or (as_of - self._last_train_date).days >= self.retrain_interval * 30
        )

        if needs_train:
            # 学習データ: 直近 train_window_months の残差
            train_days_res = self.train_window_months * 21
            residuals_train = residuals[-min(train_days_res, len(residuals)):]

            self._train(residuals_train)
            self._last_train_date = as_of

        if self._model is None:
            return {}

        # ──────────────────────────────────────────────────────
        # シグナル生成
        # ──────────────────────────────────────────────────────
        signals = self._predict_signals(residuals, self.H)  # (S,)

        # シグナル上位 top_n 銘柄を等ウェイトでロング
        if len(signals) == 0 or not np.isfinite(signals).any():
            return {}

        # NaN・inf を除去
        valid_mask = np.isfinite(signals)
        if valid_mask.sum() < self.top_n:
            return {}

        signals_valid = np.where(valid_mask, signals, -np.inf)
        top_indices = np.argsort(signals_valid)[-self.top_n:]

        # シグナルがプラスの銘柄のみ選択（ネガティブシグナルは除外）
        top_indices = [i for i in top_indices if signals_valid[i] > 0]
        if not top_indices:
            # 全部ネガティブの場合は上位をシグナル比でウェイト
            top_indices = list(np.argsort(signals_valid)[-self.top_n:])

        w = 1.0 / len(top_indices)
        result = {tickers[i]: w for i in top_indices}

        return result
