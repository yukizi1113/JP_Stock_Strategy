"""
戦略4: Q学習ポートフォリオ最適化 (Reinforcement Learning Portfolio)

【理論背景】
ML and RL in Finance Course 3 (MDP & RL) に基づく。
QLBS (Q-Learner in Black-Scholes Worlds) の考え方をポートフォリオ最適化に応用。

【MDPの定義】
状態空間 S: 市場レジーム (0=Bear, 1=Neutral, 2=Bull) × 各銘柄のモメンタム状態
行動空間 A: 各銘柄のウェイト調整 (増加/維持/減少)
報酬関数 R: リスク調整後リターン (Sharpe比の近似)
遷移確率 P: 暗黙 (モデルフリー Q-learning)
割引因子 γ: 0.95

【Q学習アルゴリズム】
Q(s,a) ← Q(s,a) + α[R + γ max_a' Q(s',a') - Q(s,a)]

パラメータ:
  α (学習率): 0.1
  γ (割引率): 0.95
  ε (探索率): 1.0 → 0.05 (ε-greedy 方策)

【状態の離散化】
- 市場レジーム: 過去20日リターンの三分位数
- 各銘柄の短期モメンタム: 正/負（2値）
- ポートフォリオの最近のパフォーマンス: 上昇/下降（2値）
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from backtest_engine import BaseStrategy


class QLearningAgent:
    """
    ε-greedy Q学習エージェント。

    状態: (market_regime, momentum_state, port_perf_state)
        - market_regime: 0=下落, 1=中立, 2=上昇  (3値)
        - momentum_state: 0=弱, 1=強  (2値)
        - port_perf_state: 0=損失継続, 1=利益  (2値)
    行動: 0=リスクオフ(等ウェイト↓), 1=維持, 2=リスクオン(等ウェイト↑)

    Q テーブル: state → action → Q値
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon_start
        self.eps_end = epsilon_end
        self.eps_decay = epsilon_decay
        self.Q: dict = defaultdict(lambda: np.zeros(3))  # 3 actions

    def encode_state(
        self,
        market_ret_20d: float,
        momentum_score: float,
        port_ret_5d: float,
        thresholds: Tuple[float, float] = (-0.02, 0.02),
    ) -> Tuple[int, int, int]:
        """状態を離散値にエンコードする"""
        low, high = thresholds

        if market_ret_20d < low:
            regime = 0
        elif market_ret_20d > high:
            regime = 2
        else:
            regime = 1

        mom_state = 1 if momentum_score > 0 else 0
        perf_state = 1 if port_ret_5d > 0 else 0

        return (regime, mom_state, perf_state)

    def select_action(self, state: tuple) -> int:
        """ε-greedy 行動選択"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 3)
        return int(np.argmax(self.Q[state]))

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
    ) -> None:
        """Q値の更新（Bellman方程式）"""
        best_next = np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next
        td_error  = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

        # ε-decay
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)


class RLPortfolioStrategy(BaseStrategy):
    """
    Q学習ベースのポートフォリオ最適化戦略。

    事前にローリングトレーニングを実施し、本番シグナル生成に使用。

    Parameters
    ----------
    universe : 対象銘柄リスト（None の場合、prices から自動決定）
    top_n : 保有銘柄数
    base_allocation : ベースのウェイト（等ウェイト vs 集中）
    train_episodes : Q学習エピソード数
    lookback : 状態計算に使うウィンドウ（日）
    """

    name = "RL ポートフォリオ(Q学習)"

    def __init__(
        self,
        top_n: int = 10,
        train_episodes: int = 500,
        lookback: int = 60,
    ):
        self.top_n   = top_n
        self.episodes = train_episodes
        self.lookback = lookback
        self.agent   = QLearningAgent()

    def _compute_market_features(
        self,
        prices: pd.DataFrame,
        i: int,
    ) -> Tuple[float, float, float]:
        """
        インデックス i 時点の市場特徴量を計算。
        Returns: (市場20日リターン, 銘柄平均モメンタムスコア, 仮想ポートリターン5日)
        """
        if i < 25:
            return 0.0, 0.0, 0.0

        mkt_20d = prices.iloc[i].mean() / prices.iloc[i-20].mean() - 1
        mom_10d = (prices.iloc[i] / prices.iloc[i-10] - 1).dropna().mean()
        port_5d = prices.iloc[i].mean() / prices.iloc[i-5].mean() - 1

        return float(mkt_20d), float(mom_10d), float(port_5d)

    def _train(self, prices: pd.DataFrame) -> None:
        """Q学習のオフライントレーニング"""
        n = len(prices)
        if n < self.lookback + 30:
            return

        for _ in range(min(self.episodes, n - 30)):
            # ランダムな開始点
            start = np.random.randint(self.lookback, n - 10)

            state_feats = self._compute_market_features(prices, start)
            state = self.agent.encode_state(*state_feats)

            for step in range(min(10, n - start - 1)):
                i = start + step
                action = self.agent.select_action(state)

                # 行動に基づくポートフォリオウェイト
                if action == 0:  # リスクオフ
                    w = 0.5
                elif action == 2:  # リスクオン
                    w = 1.0
                else:
                    w = 0.75

                # 翌期リターン（全銘柄等ウェイト × スケール因子）
                if i + 1 < n:
                    next_ret = prices.iloc[i+1] / prices.iloc[i] - 1
                    avg_ret = next_ret.dropna().mean()
                    portfolio_ret = avg_ret * w

                    # 報酬: リターン - リスクペナルティ
                    reward = portfolio_ret - 0.5 * abs(portfolio_ret) ** 2 * 100

                    next_feats = self._compute_market_features(prices, i + 1)
                    next_state = self.agent.encode_state(*next_feats)

                    self.agent.update(state, action, reward, next_state)
                    state = next_state

    def generate_signals(
        self,
        prices: pd.DataFrame,
        as_of: pd.Timestamp,
        **kwargs,
    ) -> Dict[str, float]:

        if len(prices) < self.lookback + 30:
            return {}

        # 必要に応じてトレーニング（初回のみ）
        if not self.agent.Q:
            self._train(prices.iloc[:-1])
            # ε を推論モードに固定
            self.agent.epsilon = 0.0

        # 現在の状態を評価
        i = len(prices) - 1
        state_feats = self._compute_market_features(prices, i)
        state = self.agent.encode_state(*state_feats)
        action = self.agent.select_action(state)

        # 行動 → ポートフォリオ構成
        if action == 0:
            # リスクオフ: 上位 top_n/2 銘柄
            n_hold = max(self.top_n // 2, 5)
        elif action == 2:
            # リスクオン: 上位 top_n 銘柄
            n_hold = self.top_n
        else:
            # 中立: 上位 top_n * 3/4
            n_hold = max(self.top_n * 3 // 4, 5)

        # モメンタムで銘柄を選別
        if len(prices) >= 20:
            mom = (prices.iloc[-1] / prices.iloc[-20] - 1).dropna().sort_values(ascending=False)
            top_tickers = mom.head(n_hold).index.tolist()
        else:
            top_tickers = list(prices.columns)[:n_hold]

        if not top_tickers:
            return {}

        w = 1.0 / len(top_tickers)
        return {t: w for t in top_tickers}
