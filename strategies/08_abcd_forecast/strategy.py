"""
戦略8: ABCD-Forecast ― データ拡張バギングによるマルチアセット予測

Augmentation and Bagging method for Confidential Data series Forecasting

【出典】
  伊藤克哉ら「ABCD-Forecast: 機密金融時系列予測のためのデータ拡張バギング手法」
  JSAI 2023 (3Xin4-09) / マケデコ勉強会 (2023-08-24)

【理論的背景】
  クラウドソーシング型コンペティションを模倣した新しい時系列予測枠組み:

  1. データ拡張 (Augmentation): N種類のスパース可逆行列 A_n を用い、
     多様な「ペアスプレッド問題」を生成する。
       D^n = (A_n Y, f_n(X))

  2. バギング (Bagging): 各データセット D^n に独立に弱学習器 p_n を学習。
     弱学習器: 決定木系（Random Forest）

  3. アンサンブル (Ensemble): 逆変換後の予測を単純平均で統合。
       ŷ = (1/N) Σ A_n^{-1} p_n(X^n)

【IMOS定理】
  n個のシグナル S_i ~ N(μ, σ²) と n²個のノイズ N_i ~ N(0, σ²) があっても、
  標準化した総和 (n²+n)^{-1/2} X は N(μ, σ²) に分布収束する。
  → データ数・モデル数を増やすほど予測精度が向上。

【ABCD行列（Algorithm 2）】
  基底行列 A (N×N):
    対角 = 1, 下対角 = -1 → 隣接ペアの差分
  行・列をランダム置換 → 任意のペア組み合わせを生成。
  det(A) = ±1 → 常に正則（逆行列が存在）。

【独立性の確保 (f^n 変換)】
  各行列ごとに異なる:
  - 特徴量ラグ（1/2/3/5日）
  - ルックバックウィンドウ（15/20/30日）
  → モデル間の相関を低減し、IMOS定理の独立条件に近づける。

【対象アセット】
  日本個別株以外のグローバル資産に対応:
  - 日本ETF・REIT（JPX上場、スプレッド低）
  - コモディティ先物（金・原油・銅・小麦など、ロールオーバーコスト考慮）
  - 為替（USD/JPY等主要通貨ペア、スプレッド最小）
  - 仮想通貨（BTC/ETH、スプレッド高・注意）
  - 株価指数（日経225・S&P500等、スプレッド低）

【スプレッドへの対応】
  - 資産クラス別にスプレッドコストを明示的に設定
  - 月次リバランスで回転率を抑制
  - 高スプレッド資産（仮想通貨・コモディティ）のウェイトを相対的に抑制

【データリーク対策】
  - generate_signals() は as_of 以前のデータのみ受け取る（バックテストエンジン保証）
  - 特徴量は lag 日分のラグを適用し、予測日当日のデータを使用しない
  - 学習は X[:T-1] → y[1:T] の形式（常に過去 → 未来）
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import warnings
import numpy as np
import pandas as pd
from datetime import date
from typing import Dict, List, Optional, Tuple

from backtest_engine import BaseStrategy

warnings.filterwarnings("ignore", category=FutureWarning)


# ─────────────────────────────────────────────────────────────────────────────
# マルチアセット ユニバース定義
# ─────────────────────────────────────────────────────────────────────────────

# {ticker: (asset_class, description)}
MULTI_ASSET_UNIVERSE: Dict[str, Tuple[str, str]] = {
    # ── 日本ETF（JPX上場、yfinance: ticker + ".T"）──────────────────────────
    "1321.T": ("equity_etf", "日経225 ETF (BlackRock iShares)"),
    "1306.T": ("equity_etf", "TOPIX ETF (野村アセット)"),
    "1343.T": ("reit",       "東証REIT指数 ETF (野村アセット)"),
    "1348.T": ("equity_etf", "MAXIS TOPIX ETF (三菱UFJ)"),
    "1615.T": ("equity_etf", "NEXT FUNDS 銀行業ETF"),
    "1570.T": ("equity_etf", "日経レバレッジETF (×2)"),
    # ── コモディティ先物（S&P GSCI構成）──────────────────────────────────────
    "GC=F":   ("commodity",  "Gold Futures (COMEX)"),
    "SI=F":   ("commodity",  "Silver Futures (COMEX)"),
    "CL=F":   ("commodity",  "WTI Crude Oil Futures"),
    "NG=F":   ("commodity",  "Natural Gas Futures (NYMEX)"),
    "HG=F":   ("commodity",  "Copper Futures"),
    "ZW=F":   ("commodity",  "Wheat Futures (CBOT)"),
    "ZC=F":   ("commodity",  "Corn Futures (CBOT)"),
    "KC=F":   ("commodity",  "Coffee Futures (ICE)"),
    # ── 為替（yfinance で終値取得可能）────────────────────────────────────────
    "USDJPY=X": ("forex", "USD/JPY"),
    "EURJPY=X": ("forex", "EUR/JPY"),
    "GBPJPY=X": ("forex", "GBP/JPY"),
    "AUDJPY=X": ("forex", "AUD/JPY"),
    "CADJPY=X": ("forex", "CAD/JPY"),
    # ── 仮想通貨（スプレッド高: 注意）────────────────────────────────────────
    "BTC-USD": ("crypto", "Bitcoin / USD"),
    "ETH-USD": ("crypto", "Ethereum / USD"),
    # ── 株価指数（個別株ではなく指数）────────────────────────────────────────
    "^N225":  ("index", "日経225"),
    "^GSPC":  ("index", "S&P 500"),
    "^HSI":   ("index", "Hang Seng Index"),
    "^FTSE":  ("index", "FTSE 100"),
}

# 資産クラス別スプレッドコスト（片道、比率）
# 参考: FX 0.01-0.05%、ETF 0.03-0.1%、先物 0.05-0.2%、仮想通貨 0.1-0.5%
SPREAD_COSTS: Dict[str, float] = {
    "equity_etf": 0.0005,   # 0.05%: 日本ETF（流動性高）
    "reit":       0.0008,   # 0.08%: REIT（やや流動性低）
    "index":      0.0003,   # 0.03%: 指数（スプレッド最小）
    "commodity":  0.0010,   # 0.10%: コモディティ先物（ロールオーバーコスト含む）
    "forex":      0.0002,   # 0.02%: 為替主要通貨ペア
    "crypto":     0.0030,   # 0.30%: 仮想通貨（取引所手数料 + スプレッド）
    "default":    0.0010,   # デフォルト
}


def get_asset_class(ticker: str, universe: Optional[Dict] = None) -> str:
    """ティッカーの資産クラスを返す"""
    uni = universe or MULTI_ASSET_UNIVERSE
    if ticker in uni:
        return uni[ticker][0]
    # ヒューリスティック判定
    if ticker.endswith(".T"):
        return "equity_etf"
    if ticker.endswith("=X"):
        return "forex"
    if ticker.endswith("=F"):
        return "commodity"
    if "BTC" in ticker or "ETH" in ticker or "XRP" in ticker:
        return "crypto"
    if ticker.startswith("^"):
        return "index"
    return "default"


def get_spread_cost(ticker: str, universe: Optional[Dict] = None) -> float:
    """ティッカーのスプレッドコスト（片道）を返す"""
    asset_class = get_asset_class(ticker, universe)
    return SPREAD_COSTS.get(asset_class, SPREAD_COSTS["default"])


def fetch_multi_asset_prices(
    start: str,
    end: Optional[str] = None,
    universe: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    マルチアセットユニバースの日次終値をyfinanceで取得する。

    Parameters
    ----------
    start   : 開始日 ('YYYY-MM-DD')
    end     : 終了日 (None の場合は今日)
    universe: {ticker: (asset_class, name)} の辞書

    Returns
    -------
    pd.DataFrame (index=date, columns=tickers)
    """
    import yfinance as yf

    uni = universe or MULTI_ASSET_UNIVERSE
    tickers = list(uni.keys())
    end = end or date.today().isoformat()

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()

    if len(tickers) == 1:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        prices = raw["Close"].copy()
        if hasattr(prices.columns, "tolist"):
            pass  # already correct

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    # 欠損を前値で補完（異なるマーケット間の祝日差異）
    prices = prices.ffill().dropna(how="all")
    return prices.sort_index()


# ─────────────────────────────────────────────────────────────────────────────
# ABCDForecastStrategy
# ─────────────────────────────────────────────────────────────────────────────

class ABCDForecastStrategy(BaseStrategy):
    """
    ABCD-Forecast: データ拡張バギングによるマルチアセット予測戦略。

    クラウドソーシング型コンペを模倣し、N 個の「問題設定（ペアスプレッド）」を
    生成し、各問題に独立なモデルを当てる。逆変換後のアンサンブルで
    最終シグナルを得る。

    Parameters
    ----------
    n_matrices      : ABCD行列の数 N（増やすほど安定・精度向上）
    window          : 特徴量ルックバック（日数）
    lag             : 実行ラグ（日数）。当日データを使用しない安全マージン
    top_n           : ロング/ショート各銘柄数
    long_only       : True=ロング専用, False=ロングショート
    min_history     : 学習に必要な最低日数
    vol_normalize   : True=ボラティリティ正規化（異なる資産クラスの混合に必須）
    universe_cfg    : 資産設定辞書。None の場合は MULTI_ASSET_UNIVERSE を使用
    """

    name = "ABCD-Forecast (マルチアセット)"

    def __init__(
        self,
        n_matrices: int = 10,
        window: int = 30,
        lag: int = 1,
        top_n: int = 5,
        long_only: bool = False,
        min_history: int = 90,
        vol_normalize: bool = True,
        universe_cfg: Optional[Dict] = None,
    ):
        self.n_matrices   = n_matrices
        self.window       = window
        self.lag          = lag
        self.top_n        = top_n
        self.long_only    = long_only
        self.min_history  = min_history
        self.vol_normalize = vol_normalize
        self.universe_cfg  = universe_cfg or MULTI_ASSET_UNIVERSE

    # ── Algorithm 2: ABCD行列生成 ────────────────────────────────────────────

    def _generate_abcd_matrix(self, K: int, seed: int) -> np.ndarray:
        """
        スパースかつ正則な K×K 行列 A を生成する（Algorithm 2 in paper）。

        基底行列: 対角=1, 下対角=-1（連続差分行列、det=1）
        行・列をランダム置換 → 任意のペア組合せを実現。
        det(A) = ±1 → 常に正則。
        """
        # 下三角差分行列
        A = np.eye(K, dtype=float)
        for i in range(1, K):
            A[i, i - 1] = -1.0
        # ランダム置換（行・列）
        rng = np.random.default_rng(seed)
        row_perm = rng.permutation(K)
        col_perm = rng.permutation(K)
        return A[row_perm, :][:, col_perm]

    # ── f^n 変換: 特徴量生成（各行列ごとに異なる設定）────────────────────────

    def _build_features_from_spread(
        self,
        spread: np.ndarray,   # (T, K) スプレッドリターン
        raw_ret: np.ndarray,  # (T, K) 元リターン
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        学習用 (X_train, y_train) を生成する。

        特徴量（各時刻 t）:
          - 直近 5 日のスプレッドリターン（フラット）
          - 5 日・10 日ローリング平均
          - 5 日ローリング z-スコア
          - 元リターン直近 1 日
          - seed ごとに異なる追加ラグ（独立性向上）

        ターゲット:
          - 翌日スプレッドリターンの符号（3値: -1, 0, +1 は符号関数で近似）

        リーク防止:
          - 時刻 t の特徴量は t-lag-window:t-lag の範囲のみ使用
          - ターゲットは t+1 のスプレッドリターン（tに対して未来だが訓練データ内）
        """
        T, K = spread.shape
        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []

        # 行列ごとに異なるラグ設定（独立性のための多様化）
        rng = np.random.default_rng(seed * 137 + 1)
        extra_lags = rng.choice([1, 2, 3, 5], size=3, replace=False).tolist()
        win = min(self.window, 20)  # 特徴量ウィンドウ（最大20日）

        for t in range(win + self.lag + 5, T - 1):
            s_end   = t - self.lag
            s_start = s_end - win
            if s_start < 0:
                continue

            window_data = spread[s_start:s_end]  # (win, K)

            # ローリング統計
            n5  = min(5, s_end)
            n10 = min(10, s_end)
            mean_5  = spread[s_end - n5: s_end].mean(axis=0)
            mean_10 = spread[s_end - n10:s_end].mean(axis=0)
            std_5   = spread[s_end - n5: s_end].std(axis=0) + 1e-8

            # 追加ラグ特徴量（独立性確保）
            lag_feats = []
            for lg in extra_lags:
                idx = s_end - lg - 1
                lag_feats.append(spread[idx] if idx >= 0 else np.zeros(K))

            feat = np.concatenate([
                window_data[-5:].flatten(),   # 直近5日スプレッドリターン
                mean_5,                        # 5日平均
                mean_10,                       # 10日平均
                mean_5 / std_5,               # z-score
                raw_ret[s_end - 1],           # 元リターン直近1日
                *lag_feats,
            ]).astype(np.float32)

            # ターゲット: 翌日スプレッドリターンの符号
            y = np.sign(spread[t + 1]).astype(np.float32)

            X_list.append(feat)
            y_list.append(y)

        if not X_list:
            return np.empty((0, 0)), np.empty((0, 0))

        return np.array(X_list), np.array(y_list)

    # ── 弱学習器: Random Forest （決定木ベース）──────────────────────────────

    def _train_and_predict(
        self,
        X_train: np.ndarray,  # (T_train, feat_dim)
        y_train: np.ndarray,  # (T_train, K)
        x_pred: np.ndarray,   # (feat_dim,)
        seed: int,
    ) -> np.ndarray:
        """
        各スプレッド資産 k ごとに Random Forest を学習し、
        翌日スプレッドリターンの方向性スコアを予測する。

        y_train の各列は符号 {-1, 0, +1} だが、
        実際のスプレッドは連続値のため 0 はほぼ発生しない → 事実上 2 値分類。
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            raise ImportError("scikit-learn が必要です: pip install scikit-learn")

        K = y_train.shape[1]
        pred = np.zeros(K, dtype=float)

        # NaN/Inf 除去
        valid = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train).all(axis=1)
        X_tr, y_tr = X_train[valid], y_train[valid]

        if len(X_tr) < 20:
            return pred

        x_p = x_pred.reshape(1, -1)
        if not np.isfinite(x_p).all():
            return pred

        for k in range(K):
            labels = y_tr[:, k]
            unique = np.unique(labels)
            if len(unique) < 2:
                continue

            # 時系列ランダムサブサンプリング（モデル多様性向上）
            rng = np.random.default_rng(seed * 1000 + k)
            n = len(X_tr)
            idx = np.sort(rng.choice(n, size=max(20, int(n * 0.8)), replace=False))

            rf = RandomForestClassifier(
                n_estimators=30,    # 計算効率のため少なめ（N matrices で補完）
                max_depth=4,
                max_features=0.3,
                random_state=seed + k,
                n_jobs=1,
            )
            try:
                rf.fit(X_tr[idx], labels[idx])
            except Exception:
                continue

            # 確率予測 → 方向性スコア（E[sign] の近似）
            proba = rf.predict_proba(x_p)[0]
            classes = rf.classes_
            score = float(sum(c * p for c, p in zip(classes, proba)))
            pred[k] = score

        return pred

    # ── メインシグナル生成 ────────────────────────────────────────────────────

    def generate_signals(
        self,
        prices: pd.DataFrame,
        as_of: pd.Timestamp,
        **kwargs,
    ) -> Dict[str, float]:
        """
        ABCD-Forecast アルゴリズムでシグナルを生成する。

        処理フロー:
          1. 日次リターン計算 → ボラティリティ正規化
          2. N 個の ABCD 行列を生成
          3. 各行列でスプレッドリターンを計算 → 特徴量 + 学習 → 予測
          4. 逆変換 (A_n^{-1} @ pred_spread) → 元資産の予測スコアへ変換
          5. N 個の予測を単純平均（アンサンブル）
          6. 逆ボラティリティ加重でポートフォリオを構築

        リーク防止:
          - prices は as_of 以前のみ（バックテストエンジンが保証）
          - 特徴量は lag 日分のラグを適用（当日データ不使用）
          - 学習は常に X[:-1] → y[1:] の時系列順序を守る
        """
        if len(prices) < self.min_history:
            return {}

        # 日次リターン（ffill で欠損を補完してから計算）
        ret = prices.ffill().pct_change().dropna(how="all")

        # 有効銘柄フィルタ（欠損 30% 超は除外）
        valid_cols = ret.dropna(thresh=int(len(ret) * 0.7), axis=1).columns
        if len(valid_cols) < 3:
            return {}
        ret = ret[valid_cols].fillna(0.0)

        K = len(valid_cols)
        T = len(ret)

        if T < self.window + self.lag + 10:
            return {}

        raw_arr = ret.values.astype(np.float32)  # (T, K)

        # ── ボラティリティ正規化 ──────────────────────────────────────────────
        # 異なる資産クラス（BTC vs 日本ETF等）の価格スケール差を吸収
        if self.vol_normalize:
            vol = ret.rolling(20).std().replace(0, np.nan).ffill().fillna(1.0)
            vol_arr = vol.values.astype(np.float32)
            vol_norm = (raw_arr / vol_arr).clip(-5, 5)
            vol_norm = np.nan_to_num(vol_norm, nan=0.0)
        else:
            vol_norm = raw_arr.copy()
            vol_arr  = np.ones_like(raw_arr)

        # ── N 個の ABCD 行列でデータ拡張 → 予測 ─────────────────────────────
        all_preds: List[np.ndarray] = []

        for n in range(self.n_matrices):
            A = self._generate_abcd_matrix(K, seed=n)

            # スプレッドリターン: (T, K) @ A.T = スプレッド問題 n の目的変数
            spread = vol_norm @ A.T  # (T, K)

            # 学習データ生成（リーク防止: 最後の lag 日を除外して学習）
            X_tr, y_tr = self._build_features_from_spread(
                spread[: T - self.lag],
                raw_arr[: T - self.lag],
                seed=n,
            )
            if len(X_tr) < 20:
                continue

            # 現時点の特徴量（予測用）
            s_end   = T - self.lag
            s_start = max(0, s_end - self.window)
            if s_start >= s_end:
                continue

            window_data = spread[s_start:s_end]
            n5  = min(5, s_end)
            n10 = min(10, s_end)
            mean_5  = spread[max(0, s_end - n5): s_end].mean(axis=0)
            mean_10 = spread[max(0, s_end - n10):s_end].mean(axis=0)
            std_5   = spread[max(0, s_end - n5): s_end].std(axis=0) + 1e-8

            rng_f = np.random.default_rng(n * 137 + 1)
            extra_lags = rng_f.choice([1, 2, 3, 5], size=3, replace=False).tolist()
            lag_feats = []
            for lg in extra_lags:
                idx = s_end - lg - 1
                lag_feats.append(spread[idx] if idx >= 0 else np.zeros(K))

            x_pred = np.concatenate([
                window_data[-5:].flatten() if len(window_data) >= 5 else np.zeros(5 * K),
                mean_5,
                mean_10,
                mean_5 / std_5,
                raw_arr[s_end - 1],
                *lag_feats,
            ]).astype(np.float32)

            # 弱学習器で予測
            pred_spread = self._train_and_predict(X_tr, y_tr, x_pred, seed=n)  # (K,)

            # 逆変換: スプレッド予測 → 元資産の予測スコアへ
            try:
                A_inv = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                continue
            pred_original = A_inv @ pred_spread  # (K,)

            all_preds.append(pred_original)

        if not all_preds:
            return {}

        # ── アンサンブル（単純平均）──────────────────────────────────────────
        # IMOS定理: 平均によってノイズが打ち消され、シグナルが浮き出る
        ensemble_pred = np.mean(all_preds, axis=0)  # (K,)
        pred_series   = pd.Series(ensemble_pred, index=valid_cols)

        # ── ポートフォリオ構築（逆ボラティリティ加重）───────────────────────
        # 最近 20 日のボラティリティで逆数ウェイト（高リスク資産のウェイトを抑制）
        recent_vol = ret[valid_cols].rolling(20).std().iloc[-1].fillna(1.0).clip(1e-6, None)
        inv_vol    = 1.0 / recent_vol

        # スプレッドコスト考慮: 高スプレッド資産のウェイトを追加で割引
        spread_penalty = pd.Series(
            {t: 1.0 / (1.0 + 50 * get_spread_cost(t, self.universe_cfg))
             for t in valid_cols},
            dtype=float,
        )

        weights: Dict[str, float] = {}
        n_side = max(1, self.top_n)

        if self.long_only:
            top = pred_series.nlargest(n_side).index
            w_raw = (inv_vol[top] * spread_penalty[top]).clip(0)
            w_sum = w_raw.sum()
            if w_sum > 0:
                for t in top:
                    weights[t] = float(w_raw[t] / w_sum)
        else:
            top = pred_series.nlargest(n_side).index.tolist()
            bot = pred_series.nsmallest(n_side).index.tolist()

            long_w  = (inv_vol[top] * spread_penalty[top]).clip(0)
            short_w = (inv_vol[bot] * spread_penalty[bot]).clip(0)

            ls = long_w.sum()
            ss = short_w.sum()
            if ls > 0:
                for t in top:
                    weights[t] = float(long_w[t] / ls) * 0.5
            if ss > 0:
                for t in bot:
                    weights[t] = -float(short_w[t] / ss) * 0.5

        return weights
