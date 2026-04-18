# Lecture 01: A Tour of Simulations — Past, Present, and Future

- 講師: Joon Sung Park
- ソース: `docs/stanford univ AI Agents and Simulations/01 A Tour of Simulations Past Present and Future.pdf`
- 位置づけ: クォーター全体の導入講義。シミュレーションとは何か、なぜ今取り組むのか、過去・現在・未来を俯瞰。

---

## 1. シミュレーションの定義（形式化）

身近な例（The Sims、The Matrix、森林火災モデル）を挙げたうえで形式定義:

- **W(t)**: 時刻 t の世界の状態
- **E**: 環境。状態 S_E と遷移規則 R_E で定義
- **A_i**: 個別エージェント i（i = 1,…,N）、固有の行動 B(A_i) を持つ

シミュレーションは**再帰関数**:

```
W(t)   = (S_E(t), S_A1(t), …, S_AN(t))
W(t+1) = f(S_E(t), S_A1(t), …, S_AN(t))   # 環境規則と各エージェント行動の相互作用
```

ユーザー側の特徴:
- 同じ初期状態から繰り返し実行できる（決定的とは限らない）
- 途中で状態に介入できる
- 規則を自分で書いたのに結果が「驚き」をもたらす（創発）

---

## 2. なぜシミュレーションか — What-if 探索装置

核心主張:

> シミュレーションは「反実仮想（what-if）の問い」を可能性の多元宇宙として探索させる。

多くの問題は **wicked problems**（複雑な均衡と現実制約を持つ厄介問題、Rittel & Webber 1973）で、実世界で試行錯誤できない。

- 個人: どの授業／専攻を選ぶか
- 集団: 難しい会話のリハーサル、対立する価値観の調整
- 社会: 持続可能性のための集団行動、誤情報の抑制

→ シミュレーションは**「これまで答える術がなかった問い」に答える潜在力**を持つ。

---

## 3. Act 1: Past — 生成AI以前のシミュレーション

4系譜。各方法について「エージェントと環境をどう定義したか」を問う。

| 方法 | 起源 | 主要文献 |
|------|------|----------|
| Theory of Mind (ToM) | 認知哲学 | Gordon (1986) |
| セルオートマトン | 自己複製機械 | von Neumann (1966); Wolfram (2002) |
| ゲーム理論 | 経済学 | von Neumann & Morgenstern (1944) |
| エージェントベースモデル | 社会学 | Schelling (1971) |

**伝統的シミュレーションの評価**:
- 強み: シンプルで解釈可能
- 弱み: 人間の機微（contingencies）を過度に単純化

---

## 4. Act 2: Present — 生成エージェントによる転換

LLMによるパラダイムシフト:

> LLMは "[名前]は[記述]である" といったプロンプトで、多様な経験に条件づけられた人間行動を生成できる。

代表研究4つ:

1. **Social Simulacra** (UIST 2022) — Park, Popowski, Cai, Morris, Liang, Bernstein
   コミュニティ設計（目的・ルール・ペルソナ）を入力に、LLMで「populated prototype」を生成。スケール時の問題行動を事前発見。

2. **サーベイ・実験の再現**
   - Horton (2023) "Homo silicus"
   - Argyle et al. (2023) "Out of One, Many" (*Political Analysis*)

3. **処置効果の予測** — Ashokkumar, Hewitt, Ghezae, Willer (2024)
   GPT-4で社会科学実験の効果を予測（他講義で r=0.85 と言及）。

4. **Generative Agents** (UIST 2023) — Park, O'Brien, Cai, Morris, Liang, Bernstein
   25体のLLMエージェントに記憶ストリーム・リフレクション・計画を持たせ、噂伝播や自律パーティ企画を再現。

**生成エージェントベースモデルの評価**:
- 強み: Open-ended、機微を捉える
- 弱み: Complex（制御・解釈困難）

---

## 5. Act 3: Future — 80億人の世界シミュレータ

> 一つのビジョン: 80億人の世界シミュレータ。

Lecture 14で再訪されるロードマップの伏線。

---

## 6. 受講者へのメッセージ

- シミュレーションは**新興分野**
- 70% seminal論文 + 30% 実装演習（Python必須）
- 読書コメント(30%) + シミュレーション課題2本(各15%) + 最終グループプロジェクト(30%) + 出席(10%)
- Discussion-heavy

---

## 主要引用文献

- Card, Moran, Newell (1983) *The Psychology of Human-Computer Interaction*
- Weiser (1999) "The Computer for the 21st Century" *SIGMOBILE Mob. Comput. Commun. Rev.* 3(3)
- Newell (1990) *Unified Theories of Cognition*
- Rittel & Webber (1973) "Dilemmas in a general theory of planning" *Policy Sciences* 4, 155-169
- Gordon (1986) "Folk psychology as simulation" *Mind & Language* 1(2)
- von Neumann (1966) *Theory of Self-Reproducing Automata*
- Wolfram (2002) *A New Kind of Science*
- von Neumann & Morgenstern (1944) *Theory of Games and Economic Behavior*
- Schelling (1971) "Dynamic models of segregation" *J. Math. Sociol.* 1, 143-186
- Horton (2023); Argyle et al. (2023); Ashokkumar et al. (2024)
- Park et al. UIST 2022 (Social Simulacra); UIST 2023 (Generative Agents)

---

**要点**: Lecture 01はクォーターの地図。シミュレーション=`W(t+1)=f(...)`の再帰装置、目的=wicked problemsに対するwhat-if探索、パラダイム転換=LLMで"open-endedだが複雑"な生成エージェントが可能に。
