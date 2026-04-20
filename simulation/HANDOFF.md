# Simulation 論文 引き継ぎ用メモ

本ドキュメントはシミュレーション論文の作業を別チャットで再開するための引き継ぎ資料。

---

## 1. 論文のゴール（現時点の計画）

Stanford CS 222（Park 2024）の generative agent 系譜と Big Five × オンライン学習の先行研究 (Tokiwa, 2025) を接続し、**Big Five 5 次元のみを入力とした generative agent による大学入試結果シミュレーション**を、実測結果と分布レベルで照合する。

**二部構成**（Park et al. 2023 の controlled + end-to-end に倣う）:

- **Part 1 (Validation)**: 120 名の Big Five → generative agent → 予測入試結果分布と実測分布の一致度検証（分布レベル）
- **Part 2 (Exploration)**: 検証済みモデルで反実仮想介入（C 底上げ / N 低減）を探索

---

## 2. データ理解

### `simulation/data/raw.csv`
- 103 unique IDs × 5 category (all, en, ja, mt, sc) = 515 rows
- StudySapuri 指標 5 列 + Big Five 主成分 5 列 + Big Five ファセット 15 列
- Big Five データは BFI-2-J (Yoshino et al., 2022) 由来、信頼性 α = 0.80–0.96

### `simulation/data/guraduate_report_2.pdf`（2026 年度実績 = 本研究対象コホート）
- 星の杜高校 1 期生
- 海外 30 大学（QS 25 位 Sydney 〜 721+ Temple 等）
- 国公立（宇都宮大、東京農工大、秋田大医学部、他）
- 有名私立（上智 ×4、立教、法政、関西学院、関西、立命館、他）
- その他私立多数
- **注意**: 合格実績であり進学先ではない（1 人複数合格が混在）

### `simulation/data/guraduate_report_1.pdf`（2023 年度実績、参考用）
- 過去 3 年卒 134 名、参考比較データ

### `simulation/data/Who excels in online learning in Japan.pdf`
- 筆者の先行論文 (Tokiwa, 2025). 同じ 103 名を対象。
- 主要結果（相関強度の序列）:
  - Conscientiousness → Tests Completed/Mastered r=0.34–0.35 (p<0.001)
  - Agreeableness → Lectures Watched r=0.23 (p<0.05)
  - Extraversion facet: Sociability 負 / Assertiveness 正
  - Neuroticism → 代償的 Viewing Time 増（ただし outcome は null）

---

## 3. 重要な制約

- **ID 紐付け不可**：Big Five（1 年時）と卒業実績（3 年時）は ID 照合できない
  → 個人レベル検証不能、**分布レベル検証のみ可能**
  → タスクは「予実相関 r ≈ 0.5 になるように合成 ID-outcome ペアを生成」

---

## 4. 提案済み方法論

### カテゴリ別分析は主解析では不要
`category == 'all'` の 103 行を主解析単位。教科別は補助解析のみ。

### 卒業実績の正規化スキーム
**A. Tier（5 段階）+ B. 連続スコア (0–100)** の併用。

| Tier | 基準 | 例 |
|------|------|---|
| T1 | QS Top 100 / 旧帝大 | Sydney, Manchester, 上智一部 |
| T2 | QS 100–300 / 難関国公立 / GMARCH | 宇都宮大, 東京農工大, 上智, 立教, 法政 |
| T3 | QS 300–700 / 中堅国公立 / 関関同立・日東駒専 | Simon Fraser, UCSI, 関西学院, 日大 |
| T4 | QS 700+ / その他私立 | Oregon State, Temple, 昭和女子, 大東文化 |
| T5 | その他 | BBT, 足利大 |

### Big Five 予測重み（暫定）

```python
weights = {
    'Conscientiousness': +0.30,  # Tokiwa 2025 r=0.35 と Mammadov アジア層 0.35 が一致
    'Openness':          +0.15,  # オンライン学習文献で Agreeableness を上回る
    'Agreeableness':     +0.10,  # outcome では減衰
    'Neuroticism':       -0.10,  # 入試本番のストレス負効果
    'Extraversion':       0.00,  # Facet が相殺
}
```

### 合成データ生成（r ≈ 0.5 を目標）

```
pred_i = Σ weights × z(Big Five)
latent_i = pred_i + ε,  ε ~ N(0, σ²)  # σ を目標 r から決定
```

Latent ソート → Marginal 分布（PDF から集計）に rank-match 割当。

---

## 5. 次のチャットで最初にやるべきこと

### Step 1: メタ分析論文が先（このリポジトリでは）
`/home/user/paper/metaanalysis/` でメタ分析論文を完遂してから simulation 論文に戻る方針。

メタ分析で抽出される **effect size がシミュレーション重みの更新に使える**ため、メタ分析完成後に simulation の重みを再検討する。

### Step 2: Simulation 論文に戻った際のタスク
1. `simulation/data/outcomes.csv` の合成スクリプト作成（`generate_outcomes.py`）
2. Tier と連続スコアを付与した大学マスタ CSV 作成（`university_master.csv`）
3. Generative agent のプロンプト設計（Park et al. 2023 アーキテクチャ参考）
4. Validation 指標計算（KS 検定、Wasserstein 距離、χ²、Tier 別分布）
5. Part 2 の反実仮想介入設計
6. Methods, Results, Discussion 執筆

---

## 6. 既に完成済み

- **Introduction**: `simulation/paper/introduction.md`
  - APA 形式の本文引用 + 参考文献リスト
  - Park et al. (2023, 2024), Argyle et al. (2023), Hewitt et al. (2024), Lundberg et al. (2024), Wang et al. (2024) 等引用済み
  - 自己引用（先行 online_learning 研究）は `Author, 20XX` のプレースホルダー → Tokiwa (2025) に差し替え必要

---

## 7. 未解決の設計判断

- Tier 境界をどう厳密に決めるか（QS 偏差値の対応表を作るか）
- Conscientiousness 重み 0.30 を経験的に決めているが、メタ分析結果が出たら再検討
- 合成データの seed 固定方針（再現性のため必須）
- Generative agent の実装（GPT-4 / Claude / ローカル LLM）

---

## 8. 重要参考文献

### 手法アンカー
- Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. *UIST '23*.
- Park, J. S., Zou, C. Q., Shaw, A., Hill, B. M., Cai, C., Morris, M. R., Willer, R., Liang, P., & Bernstein, M. S. (2024). *Generative agent simulations of 1,000 people*. arXiv:2411.10109.
- Argyle, L. P., Busby, E. C., Fulda, N., Gubler, J. R., Rytting, C., & Wingate, D. (2023). Out of one, many: Using language models to simulate human samples. *Political Analysis, 31*(3), 337–351.
- Hewitt, L., Ashokkumar, A., Ghezae, I., & Willer, R. (2024). *Predicting results of social science experiments using large language models*.

### Big Five × オンライン学習
- Tokiwa, E. (2025). Who excels in online learning in Japan? *Frontiers in Psychology, 16*, 1420996.
- 詳細は `/home/user/paper/metaanalysis/literature_review.md`

### Stanford CS 222 講義ノート
- `/home/user/paper/simulation/notes/cs222/` に日本語詳細ノート完備（14 講義 + 25 補足論文）

---

## 9. Git ブランチ

現在のブランチ: `main`（マージ済み）

新しい作業を始める際は feature branch 推奨:
```
git checkout -b claude/simulation-part1-synthesis
```
