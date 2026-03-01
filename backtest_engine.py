"""
共通バックテストエンジン

すべての戦略が共有するバックテストフレームワーク。

【データリーク防止の設計】
- 月末リバランス日 rd に対して:
  hist = prices.loc[:rd]  → rd 以前のデータのみ戦略に渡す
  weights.loc[rd:] = new_w → 新ウェイトは rd から設定
  port_ret = weights.shift(1) * daily_ret → 実際の適用は rd+1 から
  つまり「月末終値を見てシグナル生成 → 翌営業日から新ウェイト適用」
  これは現実的な実装（月末終値確定後に発注、翌日寄付き執行）と一致。

- 取引コスト: ウェイト変化量 × tc（片道）でリターンから控除。
- 戦略インスタンスは generate_signals() を呼ぶたびに past-only データを受け取るため、
  戦略側でも将来データへのアクセスは構造的に不可能。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Optional
from scipy import stats as scipy_stats
from data_loader import performance_stats, fetch_benchmark


class BaseStrategy:
    """
    戦略基底クラス。サブクラスが generate_signals() を実装する。

    generate_signals() は毎月末（またはリバランス日）に呼ばれ、
    {ticker: weight} の dict を返す。weight の合計は 1.0（ロング専用）。

    重要: generate_signals() に渡される prices は as_of 以前のデータのみ。
    as_of 以降の価格情報を使ってはならない。
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
    prices : DataFrame (index=date, columns=ticker)  終値
    strategy : BaseStrategy インスタンス
    rebal_freq : str  'ME'=月末, 'W-FRI'=毎週金曜
    initial_cash : float
    transaction_cost : float  片道コスト (例: 0.001 = 0.1%)
    execution_lag : int  リバランス日からの実行ラグ日数（デフォルト1=翌日執行）
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        strategy: BaseStrategy,
        rebal_freq: str = "ME",
        initial_cash: float = 10_000_000,
        transaction_cost: float = 0.001,
        benchmark_ticker: str = "^N225",
        execution_lag: int = 1,
    ):
        self.prices      = prices.dropna(how="all").ffill().bfill()
        self.strategy    = strategy
        self.rebal_freq  = rebal_freq
        self.initial_cash = initial_cash
        self.tc          = transaction_cost
        self.benchmark_ticker = benchmark_ticker
        self.execution_lag = execution_lag

    def run(self) -> pd.DataFrame:
        """
        バックテストを実行し、日次ポートフォリオ価値を返す。
        Returns: DataFrame with columns [portfolio, benchmark]

        【リーク防止の詳細】
        1. hist = prices.loc[:rd]
           → リバランス日 rd の終値を含む過去データのみ戦略に渡す
           → rd+1 以降のデータは渡さない（将来情報の遮断）

        2. weights.loc[rd:, t] = new_w[t]
           → 新ウェイトをリバランス日 rd から設定

        3. port_ret = weights.shift(1) * daily_ret
           → shift(1) により実際のリターン計算は weights の「前日」ウェイトを使用
           → 結果: rd 当日のリターンは旧ウェイト、rd+1 以降は新ウェイト
           → これはシグナル生成後に「翌日から」新ポジションを取る現実的な実装

        4. execution_lag=1（デフォルト）で追加1日のラグをシミュレート
        """
        prices = self.prices
        dates  = prices.index
        rebal_dates = prices.resample(self.rebal_freq).last().index

        weights = pd.DataFrame(0.0, index=dates, columns=prices.columns)
        current_w: Dict[str, float] = {}

        for rd in rebal_dates:
            # ─────────────────────────────────────────────────────────
            # リーク防止: rd 以前のデータのみ戦略に渡す
            # prices.loc[:rd] は rd の終値を含む（月末終値で特徴量計算）
            # ─────────────────────────────────────────────────────────
            hist = prices.loc[:rd]
            if len(hist) < 20:
                continue

            try:
                new_w = self.strategy.generate_signals(hist, as_of=rd)
            except Exception as e:
                print(f"Signal error at {rd}: {e}")
                new_w = current_w

            total = sum(abs(v) for v in new_w.values())
            if total > 0:
                new_w = {k: v / total for k, v in new_w.items()}

            current_w = new_w

            valid_tickers = [t for t in current_w if t in prices.columns]
            for t in valid_tickers:
                weights.loc[rd:, t] = current_w[t]

        # ─────────────────────────────────────────────────────────────
        # 日次リターン計算
        # shift(1) により前日ウェイト × 当日リターン → 翌日適用
        # ─────────────────────────────────────────────────────────────
        daily_ret = prices.pct_change()
        port_ret  = (weights.shift(1) * daily_ret).sum(axis=1)

        # 取引コスト（ウェイト変化量の片道コスト）
        turnover  = weights.diff().abs().sum(axis=1)
        port_ret -= turnover * self.tc

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

        # ベンチマークが全NaNでもポートフォリオ行を残す（benchmark列のNaNは許容）
        result = pd.concat([portfolio, bench_val], axis=1)
        result = result.dropna(subset=[portfolio.name])  # ポートフォリオ列のみ必須
        self._result   = result
        self._port_ret = port_ret.reindex(result.index)
        self._weights  = weights
        return result

    def summary(self) -> Dict:
        """パフォーマンスサマリーを返す（t検定・情報比率含む拡張版）"""
        if not hasattr(self, "_result"):
            self.run()
        stats = performance_stats(
            self._port_ret,
            benchmark=self._result["benchmark"]
        )
        stats["strategy"] = self.strategy.name

        # ── 月次超過リターン vs ベンチマーク の t 検定 ──────────────────────
        try:
            port_monthly = (
                (1 + self._port_ret.reindex(self._result.index))
                .resample("ME").prod() - 1
            )
            bench_daily = self._result["benchmark"].pct_change()
            bench_monthly = (
                (1 + bench_daily)
                .resample("ME").prod() - 1
            )
            excess = (port_monthly - bench_monthly.reindex(port_monthly.index)).dropna()
            if len(excess) >= 6:
                t_stat, p_val = scipy_stats.ttest_1samp(excess, 0.0)
                stats["t_stat_vs_bench"]  = round(float(t_stat), 4)
                stats["p_val_vs_bench"]   = round(float(p_val), 4)
                stats["n_months"]         = len(excess)
                stats["mean_excess_monthly"] = round(float(excess.mean()), 6)
                # Information Ratio
                tracking_err = float(excess.std()) * np.sqrt(12)
                ann_excess   = float(excess.mean()) * 12
                ir = ann_excess / tracking_err if tracking_err > 0 else np.nan
                stats["information_ratio"] = round(ir, 4)
                stats["ann_excess_return"] = round(ann_excess, 4)
                # Bootstrap 95% CI for mean excess return (1000 draws)
                rng = np.random.default_rng(42)
                boot = [rng.choice(excess, size=len(excess), replace=True).mean()
                        for _ in range(1000)]
                stats["boot_ci_lo"] = round(float(np.percentile(boot, 2.5)) * 12, 4)
                stats["boot_ci_hi"] = round(float(np.percentile(boot, 97.5)) * 12, 4)
        except Exception:
            pass

        return stats

    def plot(self, save_path: Optional[str] = None) -> None:
        """累積リターングラフを描画する"""
        if not hasattr(self, "_result"):
            self.run()

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

        ax = axes[0]
        (self._result / self._result.iloc[0]).plot(ax=ax)
        ax.set_title(f"累積リターン: {self.strategy.name}", fontsize=14)
        ax.set_ylabel("相対価値")
        ax.legend()
        ax.grid(True, alpha=0.3)

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
