# Simulation 論文 引き継ぎ

## 研究の目的

Big Five 性格特性 5 次元のみを入力とした generative agent による日本の大学入試結果シミュレーション。Park et al. (2023, 2024) のアーキテクチャ系譜を日本の教育ドメインに外挿し、個人レベル予測可能性を評価する。

**構成**

- **Part 1 (Validation)**: Big Five → generative agent → 予測偏差値分布と正解分布の照合（集団・個人両レベル）
- **Part 2 (Exploration)**: 反実仮想介入（Conscientiousness 底上げ, Neuroticism 低減）

---

## ファイル構成

```
simulation/
├── HANDOFF.md
├── data/
│   ├── raw.csv                                # N=103, Big Five + StudySapuri 指標
│   ├── raw_dataset.csv                        # 作業データ (ID, Tests, C, Hensachi)
│   ├── university_master.csv                  # 大学選抜性リファレンス (N=96)
│   ├── baseline_results.json                  # 3 種ベースラインの参照値
│   ├── Who excels in online learning in Japan.pdf   # Tokiwa (2025)
│   └── guraduate_report_{1,2}.pdf             # 2023 / 2026 卒業実績資料
├── agent/
│   ├── prompts.py                             # System + User プロンプト, Tool schema (英語)
│   ├── agent.py                               # Opus 4.7 + Extended Thinking + Tool Use
│   ├── run_pilot.py                           # 10 × 30 = 300 call pilot ランナー
│   ├── baselines.py                           # Random / OLS-C / OLS-BigFive
│   ├── reliability.py                         # ICC, within-subject spread
│   ├── load_env.py                            # .env ローダー (依存なし)
│   ├── .env.example                           # テンプレート (committed)
│   ├── .env                                   # 実キー (gitignored)
│   └── README.md                              # 実行手順
├── notes/cs222/                               # Stanford CS222 14 講義 + 25 論文ノート
├── docs/                                      # Stanford CS222 配布 PDF
└── paper/
    └── introduction.md                        # Introduction（自己引用差替え要）
```

---

## エージェント設計

| 項目 | 値 |
|---|---|
| Model | `claude-opus-4-7` |
| Extended Thinking | `adaptive`, effort `high` |
| Output | Tool Use (`submit_prediction`, JSON schema) |
| Language | English (prompts, tool schema) |
| Prompt caching | system prompt, ephemeral |
| Temperature | 1.0 |
| Samples / participant | N = 30 |
| Retry | 最大 3 回, exponential backoff |

**プロンプト方針**: Big Five スコアと cohort 文脈、偏差値スケール参照のみを提供。効果量・先行相関係数・特定理論の数値は一切プロンプトに含めない。

---

## 残タスク

### 1. Pilot 実行

```bash
cd simulation/agent
python agent.py          # smoke test (1 call, ~$0.10)
python run_pilot.py      # 10 × 30 = 300 call, ~$30-40, 15-30 min
```

Output: `data/agent_pilot_results.csv`

### 2. Pilot 品質判定

```bash
python reliability.py    # ICC, per-subject spread
```

- Baseline (`baseline_results.json`) と Pilot の Pearson r / RMSE 比較
- ICC(1,30) が 0.5 以上であれば ensemble の信頼性確保
- 必要ならプロンプト反復調整

### 3. 本番実行

103 × 30 = 3,090 calls, 約 $320–400, 1-2 h。

- `run_pilot.py` の `N_PARTICIPANTS = 103` に変更、または `run_full.py` 作成
- Message Batches API 化で 50% 削減検討

Output: `data/agent_full_results.csv`

### 4. Validation 指標実装

- **集団レベル**: Kolmogorov–Smirnov 検定, Wasserstein 距離, χ²
- **個人レベル**: Pearson r, RMSE, MAE, 95% CI
- **先行研究照合**: Hewitt et al. (2024) r = 0.85 ベンチマーク, Park (2024) 個人予測問題

### 5. Part 2 反実仮想介入

- Conscientiousness + 1 SD 条件下での再予測（103 × 30 回、同パラメータ）
- Neuroticism − 1 SD 条件下での再予測
- 介入前後の分布変化可視化、効果量推定

### 6. Paper 執筆

- **Methods**: agent architecture, Opus 4.7, Extended Thinking, Tool Use, N=30, 3 baselines
- **Results**: pilot / full 実行結果, validation 指標, reliability, baseline 比較
- **Discussion**: Park (2024) 個人予測問題への含意, Lundberg et al. (2024) 情報的上限, Wang et al. (2024) flattening バイアス

### 7. Introduction 修正

[paper/introduction.md](paper/introduction.md) の `Author, 20XX` プレースホルダーを：
- Tokiwa (2025) — Frontiers in Psychology 16, 1420996
- Tokiwa (2026) — メタ分析 (OSF DOI 10.17605/OSF.IO/E5W47)

に差替え。メタ分析の pooled r = 0.167 [0.089, 0.243] を Conscientiousness × academic achievement ベンチマークとして引用。

### 8. OSF Pre-registration (推奨)

Simulation 論文単独の pre-registration を OSF Registries に登録（メタ分析論文 [E5W47] と同手順）。本番実行前に登録し、分析手順を事前固定する。

---

## 次回再開コマンド

```bash
cd simulation/agent
python agent.py          # 接続確認
python run_pilot.py      # pilot 開始
```

---

## 主要参考文献

**手法アンカー**
- Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. *UIST '23*.
- Park, J. S., Zou, C. Q., Shaw, A., Hill, B. M., Cai, C., Morris, M. R., Willer, R., Liang, P., & Bernstein, M. S. (2024). *Generative agent simulations of 1,000 people*. arXiv:2411.10109.
- Park, J. S. (2024). *CS 222: AI agents and simulations* [Lecture notes]. Stanford University.
- Argyle, L. P., Busby, E. C., Fulda, N., Gubler, J. R., Rytting, C., & Wingate, D. (2023). Out of one, many. *Political Analysis, 31*(3), 337–351.
- Hewitt, L., Ashokkumar, A., Ghezae, I., & Willer, R. (2024). *Predicting results of social science experiments using large language models*.
- Lundberg, I., Brand, J. E., & Jeon, N. (2024). The origins of unpredictability in life outcome prediction tasks. *PNAS, 121*(24), e2322973121.
- Wang, A., Morgenstern, J., & Dickerson, J. P. (2024). *LLMs that replace human participants can harmfully misportray and flatten identity groups*. arXiv:2402.01908.

**Big Five × 学業成果**
- Tokiwa, E. (2025). Who excels in online learning in Japan? *Frontiers in Psychology, 16*, 1420996.
- Tokiwa, E. (2026). *Big Five personality traits and academic achievement in online learning environments: A systematic review and meta-analysis* [Preprint]. OSF. https://doi.org/10.17605/OSF.IO/E5W47
- Poropat, A. E. (2009). A meta-analysis of the five-factor model of personality and academic performance. *Psychological Bulletin, 135*(2), 322–338.
- Mammadov, S. (2022). Big Five personality traits and academic performance: A meta-analysis. *Journal of Personality, 90*(2), 222–255.
