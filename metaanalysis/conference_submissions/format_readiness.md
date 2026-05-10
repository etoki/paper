# Format Readiness Checklist(投稿直前 sanity check)

各 venue の **公開 CFP / website 情報から導かれる format 要件** と現状の draft の compliance 状況。テンプレート(Word/LaTeX)が手元にない段階で確認できる項目に限定。テンプレート受領後は本ドキュメントを基準に最終調整する。

---

## ECEL 2026(ACPI)

**Submission system**: helen@academic-conferences.org / info@academic-conferences.org
**Submission portal**: ACPI submission system(URL は extension 承認返信に同梱想定)
**Public CFP**: https://www.academic-conferences.org/conferences/ecel/

### Format 要件(公開 CFP / ACPI standard より推定)

| 項目 | 要件 | 現状 | 備考 |
|---|---|---|---|
| Page count(full paper) | 8–10 pages(ACPI 標準) | ✅ 推定 8 pages 相当 | 投稿システムで上限確認 |
| Word format | `.doc / .docx`(ACPI 公式) | ✅ docx 生成済 | ACPI 専用 Word テンプレあり、未取得 |
| Reference style | **Harvard / APA-7**(ACPI 系は両用許容) | ✅ APA-7 形式 | 統一ずみ |
| Title page | 著者名 / 所属 / email / abstract / keywords | ✅ 完備 | ACPI テンプレで再配置必要 |
| Abstract | 250–300 words | ✅ 推定 280 words | 要 word 数 verify |
| Keywords | 4–6 個 | ✅ 7 個 | 上限超え注意、5 個に絞る要 |
| Page numbers | ACPI が付ける | — | 自分で付けない |
| Margins | A4 / 25 mm(ACPI 標準) | テンプレ次第 | 受領後調整 |
| Font | Arial 11 pt(ACPI 標準推定) | テンプレ次第 | 受領後調整 |
| **PRISMA flow figure** | 必須(systematic review) | ✅ 埋込済 | Figure 1 |
| Cover letter | 別添 | ✅ docx ready | — |

### Excellence Awards 用 extended abstract

| 項目 | 要件 | 現状 |
|---|---|---|
| Word count | 400–500 words(ACPI Awards standard) | ✅ ~480 words |
| Format | プレーンテキスト or short Word doc | ✅ markdown → docx 変換可 |
| Submission | `info@academic-conferences.org` メール添付想定 | — |

### テンプレート受領後の作業

1. ACPI Word テンプレに content を流し込む(コピペ + 段落書式適用)
2. References セクションを ACPI Harvard 形式に変換(現状は APA-7)
3. Page count を再 measure(テンプレでの実測必要)
4. Keywords を 5 個に絞る

---

## ICEEL 2026(zmeeting)

**Submission portal**: https://www.zmeeting.org/submission/ICEEL2026
**Contact**: iceel@academic.net
**Conference dates**: 2026-11-27〜29(Tokyo)

### Format 要件(公開 CFP より)

| 項目 | 要件 | 現状 | 備考 |
|---|---|---|---|
| Page count | **8–10 pages**(over 11 = $80/page) | ✅ 推定 8 pages 相当 | 確認要 |
| Format | `.docx`(zmeeting standard) | ✅ docx 生成済 | ICEEL 提出用テンプレあり、未取得 |
| Language | English only | ✅ 全文英語 | — |
| Reference style | **IEEE numeric**(ICEEL 標準) or APA-7 | ✅ IEEE 形式 | 確認 |
| Title page | 著者名 / 所属 / email / abstract / keywords | ✅ 完備 | — |
| Abstract | 200–300 words | ✅ 推定 280 words | — |
| Keywords | 5–8 個 | ✅ 7 個 | OK |
| **PRISMA flow figure** | 必須(systematic review) | ✅ 埋込済 | Figure 1 |
| Cover letter | 任意 | ✅ docx ready | zmeeting で別添可 |

### テンプレート受領後の作業

1. zmeeting / ICEEL 公式 Word テンプレに流し込み
2. IEEE numeric reference style 確認(現状 IEEE-like author-year mix → 純 numeric に)
3. Page count を再 measure
4. 在京者 in-person 出席意思を提出フォームで明示

---

## ICERI 2026(IATED)

**Submission portal**: https://iated.org/iceri/online_submission
**Contact**: iceri@iated.org
**Conference dates**: 2026-11-09〜11(Seville、virtual 可)

### Format 要件(公開 CFP より)

| 項目 | 要件 | 現状 | 備考 |
|---|---|---|---|
| Submission stage | Abstract first(7/9 締切)、accept 後に full paper | ✅ abstract 化可能 | full paper 締切は accept 後通知 |
| Abstract format | IATED 専用 form(online) | ✅ md/docx 両形式 ready | IATED form は短 fields の copy-paste 想定 |
| Word count(abstract) | 1500–3000 chars(IATED 標準) | 要 verify | 投稿時 truncate 可能性 |
| Reference style | IATED 標準 | ✅ IEEE/APA 混在 | full paper 化時に IATED 仕様確認 |
| Language | English only | ✅ 全文英語 | — |
| Presentation mode | Virtual / in-person 選択 | **Virtual** 申請 | cover letter で要請済 |
| **PRISMA flow figure** | full paper で必須 | ✅ 埋込済 | abstract には不要 |

### テンプレート受領後の作業

1. IATED online form に abstract fields を copy-paste
2. accept 通知後、IATED full paper template ダウンロード(7/末頃想定)
3. full paper template に流し込み(8/上旬)

---

## 共通対応事項(全 3 venue)

### 投稿前 sanity check

submit 直前にシェル一発で実行:

```bash
cd /home/user/paper
python3 metaanalysis/conference_submissions/scripts/check_numbers.py    # cross-paper consistency + Round-4 sentinels
python3 metaanalysis/conference_submissions/scripts/check_dois.py        # DOI resolution(ローカル PC で online 実行)
python3 metaanalysis/conference_submissions/inputs/derive_studies_csv.py  # studies.csv regen
python3 metaanalysis/conference_submissions/{ecel,iceel,iceri}/scripts/run_*.py  # pipeline regen
```

全て 0 failures が確認後、最新 `full_paper.docx` + `cover_letter.docx` を提出。

### Reference 整合

各 paper の References 行は `metaanalysis/reference_index.md` と完全一致しているか目視確認(`scripts/check_numbers.py` の (A2) Round-4 sentinels で機械化済)。

### COI disclosure

各 cover letter の COI block に **A-25 Tokiwa (2025), Frontiers in Psychology, 16, 1420996, https://doi.org/10.3389/fpsyg.2025.1420996** が明記されていること(機械化済)。

### Preprint disclosure

Research Square preprint DOI `10.21203/rs.3.rs-9513298/v1`(2026-04-27 deposit)を全 venue の cover letter に disclose 済(機械化済)。

---

## テンプレート別管理

このリポジトリでは現在 markdown → pandoc で docx 化しているため、venue 公式 Word テンプレ(content + style)を取得していない。テンプレートを取得した場合は:

1. `metaanalysis/conference_submissions/<venue>/template_<filename>.docx` として save
2. content を手動で template に流し込み(段落書式と表の cell-merging が pandoc 出力と互換でないことが多い)
3. final docx を `<venue>/full_paper_for_submission.docx` として保存(現 `full_paper.docx` は drafting 用、submission 用は別 file 名で区別)

最終更新: 2026-05-10
