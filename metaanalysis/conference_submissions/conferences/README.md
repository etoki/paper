# 学会候補ナレッジベース

10件の学会を JSON 形式で保存。各JSONは投稿計画に必要な事実(締切・索引・preprint policy・blocking issues)の単一情報源。

## Tier の意味

- `S` = 最強推奨(SFC評価高 + 要件適合 + 締切間に合う)
- `A` = 強く推奨
- `B` = 条件付推奨
- `C` = 要慎重判断 / バックアップ

## 早見表(2026-05-09 時点)

| ID | Tier | 締切 | Online | Preprint適合 | Notes |
|---|---|---|---|---|---|
| ieee_tale_2026 | S | 2026-06-30 | ❓要照会 | ❌ Research Square不可 | IEEE Xplore |
| ecel_2026 | S | 2026-05-21 (full) | ✅ | ⚠️要照会 | Scopus + WoS CPCI |
| ascilite_2026 | B | 2026-06-29 | ❌ live必須 | - | **DROP** |
| iceel_2026 | A | 2026-06-30 | ❓ (東京開催) | ⚠️要照会 | 移動費ゼロ |
| iceri_2026 | A | 2026-07-09 | ✅ | ⚠️要照会 | WoS CPCI |
| edulearn_2026 | C | 2026-03-26 (経過) | ✅ | ⚠️ | **BACKUP_ONLY** |
| worldcte_2026 | B | 2026-07-03 | ✅ €160 | ⚠️ | mid-tier |
| ictle_2026 | B | 2026-06-12 | ✅ €160 | ⚠️ | WORLDCTEと重複 |
| aaou_2026 | A | TBD | likely✅ | TBD | CFP待ち |
| aace_elearn_2026 | A | TBD | ✅ 完全virtual | TBD | CFP待ち |

## 推奨投稿ターゲット(2件確保)

1. **第一候補: ECEL 2026** — 5/21 フルペーパー締切。延長要請を 5/9 送信済み(返信待ち)。P3 メタアナリシス(モダリティ moderator 追加分析)で投稿。
2. **第二候補: ICEEL 2026 (東京)** — 6/30 締切、移動費ゼロ。Hofstede 文化次元 moderator + 日本 synthesis を追加。
3. **保険候補: ICERI 2026** — 7/9 締切、virtual 公式対応、WoS CPCI 索引。Education-level x discipline 交互作用。
4. **追加検討: IEEE TALE 2026** — 6/30 締切、IEEE Xplore に載れば SFC 評価最高。preprint問題と virtual 可否が要確認。ML predictive layer + SHAP + fairlearn 追加。

## ファイルとの対応

- `01_ieee_tale_2026.json` … `10_aace_elearn_2026.json` — 詳細fact JSON
- `_fallback_template.json` — 新たな venue を追加する際のテンプレ(facts 確認後に番号prefix付きでリネーム)

## 更新ルール

CFP ページが変わったら該当 JSON を編集 → `verified_on` をbump → `metaanalysis/conference_submissions/dashboard/index.md` の表を更新。
