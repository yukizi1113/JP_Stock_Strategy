# 日本株MLトレーディング戦略集

**ML for Trading** および **ML and RL in Finance** の両講義（329ファイル、45本の学術論文）を基礎として設計した、日本株対象の機械学習・強化学習トレーディング戦略の実装集です。

三菱UFJ eスマート証券（旧auカブコム証券）の **kabu STATION API** を通じてライブ実行できます。

---

## 収録戦略一覧

| # | 戦略名 | 理論ベース | バックテスト期待Sharpe | ディレクトリ |
|---|--------|-----------|----------------------|------------|
| 1 | MLファクターモメンタム | Random Forest × クロスセクショナルモメンタム | ★★★★ | `strategies/01_ml_momentum/` |
| 2 | PCA固有ポートフォリオ | PCAによるリスク因子抽出 | ★★★ | `strategies/02_eigen_portfolio/` |
| 3 | OU平均回帰 / ペア取引 | Hurst指数 × OU過程 | ★★★ | `strategies/03_mean_reversion/` |
| 4 | Q学習ポートフォリオ | Q-learning (QLBS応用) | ★★★ | `strategies/04_rl_portfolio/` |
| 5 | 吸収比率リスクタイミング | PCA Absorption Ratio | ★★★★ | `strategies/05_absorption_ratio/` |
| 6 | マルチファクターML | XGBoost/RF × テクニカル+ファンダメンタル | ★★★★★ | `strategies/06_multi_factor_ml/` |
| 7 | ブラック・リターマン+MLビュー | B-L × 逆最適化 × Inverse RL | ★★★★ | `strategies/07_black_litterman/` |

---

## 理論的背景

### 1. MLファクターモメンタム（Cross-Sectional Momentum via Random Forest）

**出典**: ML for Trading Module 2-4 / ML and RL in Finance Course 2-1

モメンタム効果は最も堅牢なアノマリーの一つ。単純な過去リターンによるランキングではなく、
Random Forest を用いて複数の特徴量（多期間モメンタム、ボラティリティ、勝率）を組み合わせた
複合シグナルでランキングする。

```
特徴量 → Random Forest → 上昇確率スコア → Top-N 等ウェイト保有
```

- **特徴量**: 1/3/6/12ヶ月モメンタム（直近1ヶ月スキップ）、ボラティリティ、RSI的勝率
- **リバランス**: 月次
- **ユニバース**: JPXプライム・スタンダード普通株（ETF・REIT・外国株除外）

---

### 2. PCA固有ポートフォリオ（Eigen Portfolio via PCA）

**出典**: ML and RL in Finance Course 2-2 (Eigen Portfolio Construction via PCA)

リターン行列に主成分分析を適用し、固有ベクトルに対応するポートフォリオを構築する。

```
R = U Σ V^T  （SVD分解）
第k主成分ポートフォリオのウェイト = V の第k列
```

- **PC1**: 市場全体のリターン（ベータリスク）→ スキップ
- **PC2**: スタイルファクター（バリュー/グロース）→ 活用
- **吸収比率**: AR = Σ Var(PC_k) / Σ Var(r_i) → システミックリスク指標として計算

**吸収比率の解釈** (Kritzman et al., 2011):
- AR↑ → リスクが少数因子に集中 → 市場の脆弱性高 → リスク管理強化
- AR↓ → リスクが分散 → 市場安定 → ポジション拡大

---

### 3. OU平均回帰 / ペア取引（Ornstein-Uhlenbeck Process）

**出典**: ML and RL in Finance Course 3-4 + "Mean Reversion in Stock Prices" (論文)

**Ornstein-Uhlenbeck プロセス**:
```
dX_t = θ(μ - X_t)dt + σ dW_t

離散近似: X_t - X_{t-1} = a + bX_{t-1} + ε_t
→ θ = -b/Δt (平均回帰速度)
   μ = a/(θΔt) (長期平均)
   σ = std(ε)/√Δt (ボラティリティ)
```

**Hurst指数** (R/S法):
```
E[R(n)/S(n)] ∝ n^H
H < 0.5: 平均回帰（選別対象）
H = 0.5: ランダムウォーク
H > 0.5: トレンド追随
```

エントリー条件:
- `z-score = (現在価格 - μ_hat) / σ_hat < -entry_z` → 買い（過売れ）
- `z-score > exit_z` → 利益確定

---

### 4. Q学習ポートフォリオ最適化（Q-Learning Portfolio）

**出典**: ML and RL in Finance Course 3 (MDP & Q-Learning)

**QLBS (Q-Learner in Black-Scholes Worlds)** の考え方を株式ポートフォリオに応用。

**MDP 定義**:
```
状態 S: (市場レジーム, モメンタム状態, ポートフォリオ直近損益)
行動 A: {リスクオフ, 中立, リスクオン}
報酬 R: ポートフォリオリターン - リスクペナルティ
遷移 P: 暗黙（モデルフリー）
```

**Q学習更新式**:
```
Q(s,a) ← Q(s,a) + α[R + γ max_a' Q(s',a') - Q(s,a)]
α=0.1, γ=0.95, ε: 1.0 → 0.05 (ε-greedy)
```

---

### 5. 吸収比率リスクタイミング（Absorption Ratio Risk Management）

**出典**: ML and RL in Finance Course 2-4 (Absorption Ratio via PCA)

**吸収比率 (AR)**:
```
AR = Σ_{k=1}^{K} Var(PC_k) / Σ_{i=1}^{N} Var(r_i)
```

K=5個の主成分で全リスクの何割を説明できるかを測定。
ARが急上昇した局面（市場のシステミックリスク増大）ではポジションを縮小、
ARが低い健全な市場ではポジションを拡大する。

**ポジションサイジング**:
```
投資比率 = min_ratio + (max_ratio - min_ratio) × sigmoid(-AR_zscore × 1.5)
```

---

### 6. マルチファクターML（Multi-Factor Machine Learning）

**出典**: ML and RL in Finance Course 2-1 (SVM, Random Forest) + ML for Trading Module 2

テクニカル指標（RSI, MACD, ボリンジャーバンド、モメンタム）をXGBoost/Random Forestで
組み合わせ、翌月リターンの上位/下位分類を行う。

**特徴量**:
```
テクニカル:
  - モメンタム: ret_1m, ret_3m, ret_6m, ret_12m (スキップ版含む)
  - ボラティリティ: vol_20d, vol_60d (年率)
  - RSI (14日)
  - MACD ヒストグラム (12/26/9)
  - ボリンジャーバンド幅 (20日)
  - 52週高値からの距離
  - 勝率 (過去252日の上昇日割合)
```

**モデル**: XGBoost または Random Forest（バイナリ分類）

---

### 7. ブラック・リターマン + MLビュー（Black-Litterman + Inverse RL）

**出典**: ML and RL in Finance Course 3-4 + "Inverse Optimization: A New Perspective on the Black-Litterman Model"

**標準B-Lモデル**:
```
1. 均衡リターン: π = λ Σ w_mkt  （逆最適化 = Inverse RL）
2. ビュー組み込み: Pμ = Q + ε,  ε ~ N(0, Ω)
3. 事後リターン:
   μ_BL = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} [(τΣ)^{-1}π + P'Ω^{-1}Q]
4. 最適ウェイト: w* = (λΣ)^{-1} μ_BL
```

**MLビューの生成**:
- Random Forest で翌月上昇確率を予測
- 確率 → 超過リターン期待値に変換（例: prob=0.8 → view=+3.6%）
- Ω（ビュー不確実性）を確信度で調整

---

## セットアップ

```bash
# 依存関係インストール
pip install -r requirements.txt

# .env ファイルを作成（Investment/.env から参照）
# 必要な環境変数:
#   KABU_STATION_API_BASE_URL
#   KABU_STATION_API_PASSWORD
#   KABU_STATION_TRADING_PASSWORD
```

---

## バックテスト実行

```bash
# 戦略1: MLモメンタム
cd strategies/01_ml_momentum
python backtest.py --start 2015-01-01 --end 2024-12-31 --top-n 20

# 戦略2: 固有ポートフォリオ（PC2を使用）
cd strategies/02_eigen_portfolio
python backtest.py --pc 1 --top-n 20

# 戦略2: 吸収比率の可視化
python backtest.py --absorption-only

# 戦略3: OU平均回帰
cd strategies/03_mean_reversion
python backtest.py --start 2015-01-01 --end 2024-12-31
python backtest.py --hurst-only   # Hurst指数分布確認

# 戦略4: Q学習ポートフォリオ
cd strategies/04_rl_portfolio
python backtest.py --episodes 300

# 戦略5: 吸収比率リスクタイミング
cd strategies/05_absorption_ratio
python backtest.py

# 戦略6: マルチファクターML
cd strategies/06_multi_factor_ml
python backtest.py --model xgboost --top-n 20

# 戦略7: ブラック・リターマン
cd strategies/07_black_litterman
python backtest.py --confidence 0.3
```

---

## ライブ取引（kabu STATION API）

```bash
# ドライラン（注文発注なし）
python execution/live_trader.py --strategy ml_momentum --budget 5000000 --dry-run

# 実際の発注（--dry-run なし）
python execution/live_trader.py --strategy multi_factor --budget 5000000

# 利用可能な戦略名
# ml_momentum, eigen_portfolio, mean_reversion, rl_portfolio,
# absorption_ratio, multi_factor, black_litterman
```

---

## kabu STATION API セットアップ

1. kabu STATION をインストール（三菱UFJ eスマート証券口座必要）
2. kabu STATION を起動（ローカルでREST APIサーバーが立ち上がる）
3. `.env` に以下を設定:
   ```
   KABU_STATION_API_BASE_URL=http://localhost:18080/kabusapi
   KABU_STATION_API_PASSWORD=（APIパスワード）
   KABU_STATION_TRADING_PASSWORD=（取引パスワード）
   KABU_STATION_EXCHANGE=1
   KABU_STATION_ACCOUNT_TYPE=2
   ```

---

## プロジェクト構成

```
JP_Stock_Strategy/
├── README.md
├── .gitignore
├── requirements.txt
├── config.py               # 設定・環境変数読み込み
├── data_loader.py          # データ取得共通ユーティリティ
├── backtest_engine.py      # バックテストエンジン（共通）
├── strategies/
│   ├── 01_ml_momentum/     # MLファクターモメンタム
│   │   ├── strategy.py
│   │   └── backtest.py
│   ├── 02_eigen_portfolio/ # PCA固有ポートフォリオ + 吸収比率
│   │   ├── strategy.py
│   │   └── backtest.py
│   ├── 03_mean_reversion/  # OU平均回帰 + ペア取引
│   │   ├── strategy.py
│   │   └── backtest.py
│   ├── 04_rl_portfolio/    # Q学習ポートフォリオ
│   │   ├── strategy.py
│   │   └── backtest.py
│   ├── 05_absorption_ratio/ # 吸収比率リスクタイミング
│   │   ├── strategy.py
│   │   └── backtest.py
│   ├── 06_multi_factor_ml/ # マルチファクターML
│   │   ├── strategy.py
│   │   └── backtest.py
│   └── 07_black_litterman/ # ブラック・リターマン + Inverse RL
│       ├── strategy.py
│       └── backtest.py
├── execution/
│   ├── kabu_api.py         # kabu STATION API ラッパー
│   └── live_trader.py      # ライブ取引実行スクリプト
└── data/                   # データキャッシュ（.gitignore対象）
```

---

## データソース

| データ | 用途 | 取得方法 |
|--------|------|---------|
| 日本株日次株価 | バックテスト・シグナル生成 | yfinance (.T suffix) / stooq |
| JPX上場銘柄リスト | ユニバース定義（ETF/REIT除外） | JPX `data_j.xls` |
| EDINET財務データ | ファンダメンタル特徴量 | EDINET API |
| 日経225 | ベンチマーク | yfinance (^N225) |
| kabu STATION | 板情報・注文 | REST API (ローカル) |

---

## 注意事項

- **過去のパフォーマンスは将来を保証しない**: バックテスト結果は in-sample に最適化されている可能性あり
- **取引コスト**: デフォルト片道 0.1% を控除済み。実際のコストに応じて `transaction_cost` を調整
- **ルックアヘッドバイアス**: 月末リバランスで翌月データは使用しない設計
- **流動性リスク**: 中小型株のスプレッドは考慮外
- **本番運用前**: 必ず `--dry-run` で動作確認し、小額から段階的にスケール

---

## 学術的根拠（主要論文）

1. Jegadeesh & Titman (1993) - Momentum in Stock Returns
2. Kritzman et al. (2011) - Principal Components as a Measure of Systemic Risk
3. Ornstein & Uhlenbeck (1930) - Mean-Reverting Stochastic Process
4. Black & Litterman (1992) - Global Portfolio Optimization
5. He & Litterman (1999) - Intuition Behind Black-Litterman
6. Kolm et al. (2021) - Inverse Optimization: B-L Perspective
7. Halperin (2019) - QLBS: Q-Learner in the Black-Scholes World
8. Halperin (2020) - The QLBS Q-Learner Goes NuQLear (Fitted Q-Iteration)
9. Longstaff & Schwartz (2001) - Valuing American Options by Simulation (LSM)
10. Almgren & Chriss (2001) - Optimal Execution of Portfolio Transactions
