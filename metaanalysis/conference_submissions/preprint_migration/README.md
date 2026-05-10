# Preprint Migration プラン

## 対象

**P3: Big Five Meta-Analysis**
- 現在: Research Square (DOI 10.21203/rs.3.rs-9513298/v1, 2026-04-27 公開)
- 問題: Research Square = 商業 preprint server (Springer Nature傘下) → IEEE TALE の許容リストに含まれない可能性大

## 3つのオプション

### オプションA: OSF Preprints へ移行【推奨】

| 項目 | 値 |
|---|---|
| サーバー | OSF Preprints (Center for Open Science) |
| 営利性 | 完全に非営利 |
| Endorsement | 不要 |
| 即時投稿 | 可 |
| 新DOI | 自動付与 |
| 投稿者既存 OSF | あり(`osf.io/79m5j`, `osf.io/e5w47` 等)→ エコシステム連携自然 |
| IEEE 適合性 | high(non-profit、PSPB approved の確認推奨) |

**詳細手順:** `osf_upload_steps.md`

### オプションB: arXiv へ移行

| 項目 | 値 |
|---|---|
| サーバー | arXiv |
| 営利性 | 完全に非営利 |
| Endorsement | **必要**(初回のみ、既知の研究者からの推薦) |
| 即時投稿 | 不可(endorsement待ち) |
| 新DOI | あり |
| カテゴリ | cs.CY (Computers and Society) または stat.AP (Applications) |
| IEEE 適合性 | high(IEEE policy に明記) |

**詳細手順:** `arxiv_upload_steps.md`

### オプションC: Research Square に維持

| 項目 | 値 |
|---|---|
| サーバー | Research Square (現状維持) |
| 影響 | IEEE TALE には P3 を投稿できない |
| 代替 | IEEE TALE = P5 (simulation, OSF/SocArXiv) で投稿 |
| ECEL/ICERI | preprint disclosure 付きで投稿可能(明示policy無いため) |

## 推奨意思決定フロー

```
       ┌─────────────────────────────┐
       │ IEEE TALE に P3 を出すか?  │
       └────┬────────────────────┬───┘
            │ Yes                │ No
            ▼                    ▼
    ┌───────────────┐    ┌──────────────────┐
    │ OSF Preprints │    │ オプションC       │
    │ (オプションA) │    │ Research Square   │
    │ 即時可        │    │ 維持              │
    │               │    │ → P5 を IEEE TALEへ│
    │ + RS 撤回     │    └──────────────────┘
    └───────────────┘
```

## 推奨: オプションA + P5 IEEE TALE 戦略の併用

実は **P5 (Big Five Generative Agent Simulation)** が既に OSF + SocArXiv にあり、IEEE TALE の "AI-Augmented Learning" テーマに完全合致する。

→ **戦略の最適解:**
1. **IEEE TALE → P5 で投稿**(preprint問題なし、テーマ完全一致)
2. **P3 は ECEL / ICERI / ICEEL に投稿**(preprint disclosure付き)
3. **P3 を OSF に移行するかは "保険" 判断**(必須ではない)

この戦略なら preprint migration は **必須ではない** が、長期的には OSF 移行が推奨される(エコシステム整合性)。

## ファイル

- `osf_upload_steps.md` — OSF Preprints 投稿手順
- `arxiv_upload_steps.md` — arXiv 投稿手順
- `research_square_withdrawal_steps.md` — Research Square 撤回手順
- `disclosure_letter.md` — 移行完了後の各学会への開示文面
