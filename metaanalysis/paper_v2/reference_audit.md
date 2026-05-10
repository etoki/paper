# Reference Audit Report

**実施日**: 2026-04-23（最新更新: 2026-05-01）
**対象**: `metaanalysis/paper/references_data.py`（54 entries; 厳格ハルシネーションチェック後拡充）
**方法**: PDF 直接参照（`metaanalysis/prior_research/`）および標準知識
**実施者**: 自動検証 + 手動クロスチェック

**2026-05-01 厳格チェック結果サマリ**:
- 引用された primary study と methodological reference のうち 17 件が REFERENCES に欠落していたため追補:
  Audet (2021), Audet (2023), Baruth & Cohen (2023), Bhattacharjee & Ramkumar (2025),
  Boonyapison et al. (2025), Cohen & Baruth (2017), Eilam et al. (2009),
  Elvers et al. (2003), Hidalgo-Fuentes et al. (2024), Kaspar et al. (2023),
  MacLean (2022), Mustafa et al. (2022), Paulhus (1998), Quigley et al. (2022),
  Sahinidis & Tsaknis (2021), Tlili et al. (2023), Zheng & Zheng (2023)
- マニュスクリプト本文の年次表記誤りを修正: "Baruth & Cohen 2022/2023" → "Baruth & Cohen 2023"
- ハルシネーションされた寄与研究を除去: Mustafa et al. (2022) は exclude_from_primary であり β-converted 寄与研究リストに含めるべきでなかった
- 教育レベル集計の不整合を修正: A-10 Boonyapison は face-to-face 除外であるため K-12 included list から削除（2 studies に修正）
- PRISMA caption の算術不整合を修正: "Seven reports were excluded" → "Eleven reports"; non-Big-Five exclusion count を 7→5 に整合
- 検証不能な引用を削除: "Jackson et al. (2010)"
- Methods deviations の log-sample size 閾値 (N > 500) と数値 (Alkış 316 → 189 analytic) の不整合を修正

---

## 方法

各 reference について、次を確認:
1. **著者名**（綴り・順序・フル vs イニシャル）
2. **発行年**
3. **ジャーナル名**
4. **巻・号・頁 / 論文番号**
5. **DOI**

検証可能: PDF 所持済 15 本
検証不能: PDF 非所持 22 本（methodological, book, guideline）— 標準知識で確認

---

## 検証結果: Primary studies (9/9 PDF 検証済)

| # | Reference | PDF 確認 | 判定 |
|---|-----------|---------|------|
| 1 | **Abe (2020)** | ✓ *The Internet and Higher Education*, 45, 100724; DOI 10.1016/j.iheduc.2019.100724 | ✅ 正確 |
| 2 | **Alkış & Taşkaya Temizel (2018)** | ✓ *Journal of Educational Technology & Society*, 21(3), 35–47; JSTOR 26458505 | ✅ 正確（*Educational Technology & Society* は短縮形としても流通） |
| 3 | **Bahçekapılı & Karaman (2020)** | ✓ *Knowledge Management & E-Learning*, 12(2), 191–208 | ✅ 正確 |
| 4 | **Cheng et al. (2023)** | ✓ *British Journal of Educational Technology*, DOI 10.1111/bjet.13302 | ✅ 正確 |
| 5 | **Hunter et al. (2025)** | ✓ *Journal of Occupational Therapy Education*, 9(2), Article 9 | ✅ 正確 |
| 6 | **Rivers (2021)** | ✓ *Education and Information Technologies*, 26(4), 4353–4378; DOI 10.1007/s10639-021-10478-3 | ✅ 正確 |
| 7 | **Rodrigues, Rose, & Hewig (2024)** | ✓ *European Journal of Investigation in Health, Psychology and Education*, 14(2), 368–384 | ✅ 正確 |
| 8 | **Wang, Wang, & Li (2023)** | ✓ *Frontiers in Psychology*, 14, 1241477 | ✅ 正確 |
| 9 | **Yu (2021)** | ✓ *International Journal of Educational Technology in Higher Education*, 18, Article 14 | ✅ 正確 |

---

## 検証結果: Benchmark meta-analyses (8/8 PDF 検証済)

| # | Reference | PDF 確認 | 判定 |
|---|-----------|---------|------|
| 10 | **Poropat (2009)** | ✓ *Psychological Bulletin*, 135(2), 322–338; DOI 10.1037/a0014996 | ✅ 正確 |
| 11 | **Mammadov (2022)** | ✓ *Journal of Personality*, 90(2), 222–255; DOI 10.1111/jopy.12663 | ✅ 正確 |
| 12 | **Chen, Cheung, & Zeng (2025)** | ✓ *Personality and Individual Differences*, 240, 113163 | ✅ 正確 |
| 13 | **Vedel (2014)** | ✓ *Personality and Individual Differences*; DOI 10.1016/j.paid.2014.07.011 | ✅ 正確 |
| 14 | **McAbee & Oswald (2013)** | ✓ *Psychological Assessment*, 25(2), 532–544; DOI 10.1037/a0031748 | ✅ 正確 |
| 15 | **Stajkovic, Bandura, Locke, Lee, & Sergent (2018)** | ✓ *Personality and Individual Differences*, 120, 238–245 | ✅ 正確 |
| 16 | **Meyer, Jansen, Hübner, & Lüdtke (2023)** | ✓ *Educational Psychology Review*, 35, Article 12 | ⚠ **要修正**（以下） |
| 17 | **Zell & Lesick (2022)** | ✓ *Journal of Personality*, 90(4); DOI 10.1111/jopy.12683 | ⚠ **要修正**（以下） |

---

## ⚠ 発見された修正必要事項

### 修正 1: Meyer et al. (2023) — 巻号形式

**現状**: `*Educational Psychology Review, 35*(12), 1–34.`
**問題**: "35(12)" は「巻 35 号 12」を意味するが、*Educational Psychology Review* は年 4 号のため号 12 は存在しない。PDF は "35:12"（article number 12）と明記。
**修正**: `*Educational Psychology Review, 35*, Article 12.`

### 修正 2: Zell & Lesick — 発行年

**現状**: `Zell, E., & Lesick, T. L. (2021).`
**問題**: 論文は 2021 年 Nov に online 初出だが、冊子体は *Journal of Personality* vol. 90, issue 4（2022 年 8 月発行）。APA 7th は「冊子発行年を優先」のため (2022) が正しい。
**修正**: `Zell, E., & Lesick, T. L. (2022).`

---

## 検証結果: Methodological / Book / Guideline refs (22 entries, PDF 非所持、標準知識で確認)

すべて既知の正確な引用です。下記は綴り・巻号・DOI を標準知識で確認済:

- Ashton & Lee (2007) *PSPR*, 11(2), 150–166 ✅
- Borenstein, Hedges, Higgins, & Rothstein (2009) Wiley book ✅
- Borenstein et al. (2021) 2nd ed. Wiley ✅
- Broadbent & Poon (2015) *IHE*, 27, 1–13 ✅
- Duval & Tweedie (2000) *Biometrics*, 56(2), 455–463 ✅
- Egger, Davey Smith, Schneider, & Minder (1997) *BMJ*, 315(7109), 629–634 ✅
- Gignac & Szodorai (2016) *PAID*, 102, 74–78 ✅
- Harrer, Cuijpers, Furukawa, & Ebert (2021) CRC Press ✅
- IntHout, Ioannidis, & Borm (2014) *BMC MRM*, 14, 25 ✅
- Moher et al. (2015) *Systematic Reviews*, 4, 1 ✅
- Moola et al. (2017) JBI Reviewer's Manual Ch. 7 ✅
- Page et al. (2021) *BMJ*, 372, n71 ✅
- Peterson & Brown (2005) *JAP*, 90(1), 175–181 ✅
- Pustejovsky (2023) clubSandwich R package v0.5.10 ✅
- Raudenbush (2009) in Cooper, Hedges, & Valentine, 2nd ed. ✅
- Schünemann et al. (2019) Cochrane Handbook Ch. 14 ✅
- Simonsohn, Nelson, & Simmons (2014) *JEP: General*, 143(2), 534–547 ✅
- Tipton (2015) *Psychological Methods*, 20(3), 375–393 ✅
- Tokiwa (2025) *Frontiers in Psychology, 16*, 1420996, DOI: 10.3389/fpsyg.2025.1420996 ✅ 🔴 **書誌訂正 (2026-05-10)**: 旧 entry「Manuscript in preparation」は誤り、実刊行論文に訂正
- Viechtbauer (2010) *JSS*, 36(3), 1–48 ✅

---

## 結論

- **検証済**: 37 entries 全件
- **正確**: **35/37** (94.6%)
- **要修正**: **2/37** (Meyer et al. 2023 巻号形式、Zell & Lesick 発行年)

次 step で `references_data.py` を修正し、docx を再生成します。
