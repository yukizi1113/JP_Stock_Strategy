"""
共通バックテストエンジン

すべての戦略が共有するバックテストフレームワーク。
戦略クラスは BaseStrategy を継承し generate_signals() を実装する。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Optional, Callable
from data_loader import performance_stats, fetch_benchmark


class BaseStrategy:
    """
    戦略基底クラス。サブクラスが generate_signals() を実装する。

    generate_signals() は毎月末（またはリバランス日）に呼ばれ、
    {ticker: weight} の dict を返す。weight の合計は 1.0。
    """

    name: str = "BaseStrategy"

    def generate_signals(
        self,
        prices: pd.DataFrame,
        as_of: pd.Timestamp,
        **kwargs,
    ) -> Dict[str, float]:
        raise NotImplementedError


class Backtester:
    """
    ベクトル化バックテストエンジン。

    Parameters
    ----------
    prices : DataFrame (index=date, columns=ticker)
    strategy : BaseStrategy インスタンス
    rebal_freq : str  'ME' = 月末, 'W-FRI' = 毎週金曜
    initial_cash : float
    transaction_cost : float  片道コスト (例: 0.001 = 0.1%)
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        strategy: BaseStrategy,
        rebal_freq: str = "ME",
        initial_cash: float = 10_000_000,
        transaction_cost: float = 0.001,
        benchmark_ticker: str = "^N225",
    ):
        self.prices     = prices.dropna(how="all").ffill().bfill()
        self.strategy   = strategy
        self.rebal_freq = rebal_freq
        self.initial_cash = initial_cash
        self.tc         = transaction_cost
        self.benchmark_ticker = benchmark_ticker

    def run(self) -> pd.DataFrame:
        """
        バックテストを実行し、日次ポートフォリオ価値を返す。
        Returns: DataFrame with columns [portfolio, benchmark]
        """
        prices = self.prices
        dates  = prices.index
        rebal_dates = prices.resample(self.rebal_freq).last().index

        # 日次ウェイト行列を構築
        weights = pd.DataFrame(0.0, index=dates, columns=prices.columns)
        current_w: Dict[str, float] = {}

        for rd in rebal_dates:
            # リバランス日より前の価格データを渡す
            hist = prices.loc[:rd]
            if len(hist) < 20:
                continue
            try:
                new_w = self.strategy.generate_signals(hist, as_of=rd)
            except Exception as e:
                print(f"Signal error at {rd}: {e}")
                new_w = current_w

            # 正規化
            total = sum(abs(v) for v in new_w.values())
            if total > 0:
                new_w = {k: v / total for k, v in new_w.items()}

            current_w = new_w

            # リバランス日以降に適用
            valid_tickers = [t for t in current_w if t in prices.columns]
            for t in valid_tickers:
                weights.loc[rd:, t] = current_w[t]

        # 日次リターン
        daily_ret = prices.pct_change()

        # ポートフォリオ日次リターン（ウェイト×リターン、前日ウェイト適用）
        port_ret = (weights.shift(1) * daily_ret).sum(axis=1)

        # 取引コスト（ウェイトの変化量）
        turnover = weights.diff().abs().sum(axis=1)
        port_ret -= turnover * self.tc

        # 累積価値
        portfolio = self.initial_cash * (1 + port_ret).cumprod()
        portfolio.name = self.strategy.name

        # ベンチマーク
        try:
            bench_raw = fetch_benchmark(
                prices.index[0].strftime("%Y-%m-%d"),
                prices.index[-1].strftime("%Y-%m-%d"),
                self.benchmark_ticker
            )
            bench_raw = bench_raw.reindex(dates).ffill()
            bench_val = self.initial_cash * bench_raw / bench_raw.iloc[0]
        except Exception:
            bench_val = pd.Series(np.nan, index=dates)
        bench_val.name = "benchmark"

        result = pd.concat([portfolio, bench_val], axis=1).dropna()
        self._result     = result
        self._port_ret   = port_ret.reindex(result.index)
        self._weights    = weights
        return result

    def summary(self) -> Dict:
        """パフォーマンスサマリーを返す"""
        if not hasattr(self, "_result"):
            self.run()
        bench = self._result["benchmark"].pct_change()
        stats = performance_stats(self._port_ret, benchmark=self._result["benchmark"])
        stats["strategy"] = self.strategy.name
        return stats

    def plot(self, save_path: Optional[str] = None) -> None:
        """累積リターングラフを描画する"""
        if not hasattr(self, "_result"):
            self.run()

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

        # 上段: 累積価値
        ax = axes[0]
        (self._result / self._result.iloc[0]).plot(ax=ax)
        ax.set_title(f"累積リターン: {self.strategy.name}", fontsize=14)
        ax.set_ylabel("相対価値")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 下段: ドローダウン
        ax2 = axes[1]
        cum = (1 + self._port_ret).cumprod()
        dd  = cum / cum.cummax() - 1
        dd.plot(ax=ax2, color="red", alpha=0.7)
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.3, color="red")
        ax2.set_title("ドローダウン")
        ax2.set_ylabel("DD")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close()
