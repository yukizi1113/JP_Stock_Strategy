# 日本株MLトレーディング戦略集

日本株対象の機械学習・強化学習トレーディング戦略の実装集です。

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
| 8 | ABCD-Forecast（マルチアセット） | データ拡張バギング × スパース逆行列アンサンブル | ★★★★★ | `strategies/08_abcd_forecast/` |

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

**改良点 (UKI Qiita記事 2024)**:

1. **3値分類 (Ternary)**: 上位30%=+1 / 下位30%=-1 / 中間を学習から除外
   → 境界近傍の曖昧サンプルを削除し分類精度を向上
2. **ファクターニュートラライゼーション**: OLS で市場共通成分を除去
   → 日次Sharpe 0.066 → 0.22 に改善（UKI実験）
3. **ランクガウス変換**: StandardScaler の代わりに NormInv(rank/(N+1)) を適用
   → 外れ値に頑健、分布の正規性を保証

**モデル**: XGBoost または Random Forest（3値分類）

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

### 8. ABCD-Forecast（マルチアセット）

**出典**: 伊藤克哉ら「ABCD-Forecast: 機密金融時系列予測のためのデータ拡張バギング手法」 JSAI 2023

クラウドソーシング型コンペを模倣した新しいアンサンブル学習枠組み。
N 種類のスパース正則行列でデータを変換し、各問題に独立なモデルを当て、
逆変換後のアンサンブルで最終シグナルを得る。

```
アルゴリズム:
  For n = 1..N:
    1. A_n: スパース可逆行列（対角=1, 下対角=-1, 行/列ランダム置換）
       → det(A_n) = ±1 → 常に正則
    2. Y^n = A_n @ Y（ボラティリティ正規化後リターン）
       → スプレッド問題 n を生成
    3. 弱学習器 p_n: Random Forest で翌日スプレッド方向を予測
    4. 逆変換: pred_n = A_n^{-1} @ p_n(X^n) → 元資産の予測スコア
  アンサンブル: ŷ = (1/N) Σ pred_n
```

**IMOS定理**: n シグナル + n² ノイズでも、平均化によりシグナルが浮き出る
（論文実験: ASR=1.516 vs Random Forest単体 0.473）

**対象アセット**:

| 資産クラス | ティッカー例 | スプレッドコスト |
|-----------|------------|---------------|
| 日本ETF | 1321.T, 1306.T, 1343.T, 1570.T | 0.05% |
| 株価指数 | ^N225, ^GSPC, ^HSI, ^FTSE | 0.03% |
| コモディティ先物 | GC=F, CL=F, SI=F, HG=F, ZW=F | 0.10% |
| 為替 | USDJPY=X, EURJPY=X, GBPJPY=X | 0.02% |
| 仮想通貨 | BTC-USD, ETH-USD | 0.30% |

> 日本以外の市場に上場する個別株は対象外（指数・ETF・FX・先物・仮想通貨は対象）

**スプレッド対応**: `1/(1 + 50 × spread_cost)` のペナルティで高スプレッド資産のウェイトを割引。
月次リバランスで回転率を抑制。

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

---

## データリーク対策（厳密な検証済み）

バックテストにおけるデータリーク（look-ahead bias）を以下の設計で完全に排除しています。

### バックテストエンジンの設計

| 処理 | リーク防止の実装 |
|------|----------------|
| シグナル生成 | `hist = prices.loc[:rd]` → リバランス日 rd までのデータのみ渡す |
| ウェイト適用 | `weights.shift(1)` → 翌日からの適用（当日リターンは旧ウェイトで計算） |
| 取引コスト | ウェイト変化量 × 片道コスト → リターンから控除 |

### 各戦略のリーク対策

| 戦略 | 対策内容 |
|------|---------|
| **1 MLモメンタム** | ラベル生成をグローバル閾値→**各月クロスセクション内中央値**に変更（修正済）|
| **2 固有ポートフォリオ** | PCAはローリングウィンドウ内の過去データのみで fitting（問題なし）|
| **3 OU平均回帰** | Hurst指数・OUパラメータとも as_of 以前のデータのみ使用（問題なし）|
| **4 Q学習** | 訓練は `prices.iloc[:-1]`（当日除外） + `retrain_months` ごとにローリング再訓練（改善済）|
| **5 吸収比率** | `_ar_history` は各呼び出しで過去ARのみ蓄積（問題なし）|
| **6 マルチファクターML** | ラベルを**各月クロスセクション内中央値**に変更 + `colsample_bytree=0.1`（修正済）|
| **7 B-L + MLビュー** | 訓練ウィンドウから最終月を除外 + 各月クロスセクションラベル（修正済）|

### 空売り規制への対応（JPX貸借銘柄リスト）

日本市場では **貸借銘柄** のみ空売りが可能。「制度信用銘柄」「非制度信用銘柄」は空売り不可。

| 信用区分 | 空売り可否 | 銘柄数（2026/02/02時点）|
|---------|----------|----------------------|
| 貸借銘柄 | **可能** | 2,700銘柄 |
| 制度信用銘柄 | **不可** | 1,558銘柄 |
| 非制度信用銘柄 | **不可** | 13銘柄 |

**実装**:
- `data_loader.build_no_short_tickers()`: JPXから貸借銘柄リストを取得し、空売り不可ティッカーのセットを返す（キャッシュ機能付き）
- **戦略2（固有ポートフォリオ）** の `long_only=False` モードでショート候補から空売り不可銘柄を除外
- 戦略1/3/4/5/6/7 はロング専用のため変更不要

> **注**: `data_loader.JPX_MARGIN_URL` のURLは定期的に更新が必要（JPXが月次更新）

---

### JQuants-Forum 知見の統合（UKI000/JQuants-Forum）

決算発表後20日間の高値・安値予測コンペ（SIGNATE）の優秀コードより以下を採用:

- **ATR 4変種**: 通常ATR / ギャップATR / 日中ATR / ヒゲATR（`data_loader.compute_atr_features()`）
- **HL Band**: 期間高値 - 安値レンジ（過去5/10/20/40/60日）
- **出来高比率**: 当日 ÷ N日移動平均出来高（流動性・注目度指標）
- **マーケットインパクト**: ATR ÷ 出来高（執行コスト指標）
- **期間別正規化**: `period_wise_normalize()` で非定常性を緩和
- **XGBoost**: `colsample_bytree=0.1`（各ツリーで特徴量の10%のみ使用）で過学習防止

---

## ライブ取引（kabu STATION API）

```bash
# ドライラン（注文発注なし）
python execution/live_trader.py --strategy ml_momentum --budget 5000000 --dry-run

# 実際の発注（--dry-run なし）
python execution/live_trader.py --strategy multi_factor --budget 5000000

# 利用可能な戦略名
# ml_momentum, eigen_portfolio, mean_reversion, rl_portfolio,
# absorption_ratio, multi_factor, black_litterman, abcd_forecast
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
│   ├── 07_black_litterman/ # ブラック・リターマン + Inverse RL
│       ├── strategy.py
│       ├── backtest.py
│   ├── 08_abcd_forecast/  # ABCD-Forecast（マルチアセット）
        └── strategy.py
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
11. 伊藤克哉ら (JSAI 2023) - ABCD-Forecast: データ拡張バギングによる時系列予測
12. UKI (JQuants-Forum) - Ternary Classification / Factor Neutralization
13. Lopez de Prado (2018) - Advances in Financial Machine Learning (Rank-Gaussian)
