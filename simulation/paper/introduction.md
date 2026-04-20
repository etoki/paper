# ビッグファイブに基づく生成エージェントによる大学入試結果のシミュレーション

## 1. はじめに

個々の人間の将来の挙動を先回りして推定するという問題は，教育・医療・政策などの実務から，社会科学における理論検証に至るまで，長らく研究者と実務家が取り組んできた課題である。こうした問題の多くは，構造が曖昧で，利害関係が錯綜し，試行錯誤のコストが非常に高いという意味で，Rittel and Webber (1973) がいう「厄介な問題 (wicked problems)」に分類される。個人の選択と，その選択が埋め込まれる社会環境の双方が結果を規定するため，単一の統計モデルでは十分に表現できない。こうした問題に対し，古典的には，Schelling (1971, 1978) のエージェント・ベース・モデル (agent-based model; ABM) に代表されるように，単純な規則に従う個体を多数相互作用させて社会現象を再現する試みが用いられてきた。しかしながら，このような手続きは，人間行動の文脈依存性や機微を大きく単純化せざるを得ないという欠点を抱える (Park, 2024)。

近年の大規模言語モデル (large language model; LLM) の台頭は，この分野に新たなパラダイムをもたらした。Argyle et al. (2023) は，人口統計的なバックストーリーを条件として与えることで，GPT-3 が特定の人間サブグループの回答分布を高い忠実度で再現しうることを示し，これを algorithmic fidelity と名付けた。彼らの Study 2 では，米国大統領選の投票行動について，条件付き四分相関で r = .90–.94 の精度が得られ，さらに学習データが打ち切られた以降の 2020 年選挙についても精度を保つなど，単なる記憶再生ではない汎化が確認されている (Argyle et al., 2023)。この「シリコン・サンプリング」の考え方は Park et al. (2022) の Social Simulacra によってコミュニティ設計の文脈に拡張された後，Park et al. (2023) によって，記憶・反省・計画を持つ持続的な generative agent として個人レベルまで押し進められた。Park et al. (2023) の Smallville 実験では，25 体のエージェントが 2 日間にわたって自発的な情報伝播，関係形成，バレンタイン・パーティーの協働企画を行い，人間評価者による行動的チューリングテストで観察のみを用いる従来手法に対して Cohen's d = 8.16 という大きな効果を示した。

生成エージェントによるシミュレーションの「正確性」が，集団レベルで実際に検証された代表例として，Hewitt, Ashokkumar, Ghezae, and Willer (2024) がある。彼らは，nationally representative なプロバビリティ・サンプルに基づく 476 の処置効果 (合計 105,165 名分) を対象に GPT-4 の予測を評価し，素の相関で r = .85，減衰補正後で r_adj = .91，方向一致率 90% を報告した。さらに学習データ期間外の未公刊研究に限定した場合の方が精度が高く (r = .90)，単純なデータ漏洩では説明できないことが示された (Hewitt et al., 2024)。このことは，LLM による社会科学的予測が少なくとも集団レベルでは成熟しつつあることを示唆する。

一方で，**個人レベル**での予測可能性については依然として結論が出ていない (Park, 2024)。Park (2024) が CS 222 講義で強調するとおり，個人モデルの構築には 2 つの本質的困難が伴う。第 1 に，個人ごとの訓練データは構造上，非常に希少である。第 2 に，個人の行動は測定誤差と本来の揺らぎの双方により一貫性を欠き，集団データでは成立する「平均への回帰」による緩和が働かない (Ansolabehere, Rodden, & Snyder, 2008)。Lundberg et al. (2024) は，*Fragile Families Challenge* のデータから，個人の人生結果の予測可能性には原理的な上限が存在することを示しており，この困難を裏付けている。また，Wang, Morgenstern, and Dickerson (2024) は，アイデンティティ・プロンプトによる silicon 生成が集団内の分散を平坦化しステレオタイプ化しやすいと警告しており，個人レベル応用における注意点を示している。

こうした困難にもかかわらず，Park (2024) は，個人を全体的 (holistically) に記述するのに有効な情報を取得し，それを LLM に条件付けてロールプレイさせる，という「中心モデル＋個人差分」の設計方針を提唱している。これは協調フィルタリング (Resnick, Iacovou, Suchak, Bergstrom, & Riedl, 1994) や Jury Learning (Gordon et al., 2022) の系譜に連なる考え方であり，**パーソナリティ心理学における Big Five 特性** (John & Srivastava, 1999) は，個人を低次元で記述する古典的な枠組みとしてこの設計に自然に収まる。Big Five は Openness，Conscientiousness，Extraversion，Agreeableness，Neuroticism の 5 次元から個人を捉え，学業達成や行動傾向との関連が広範に報告されている (John & Srivastava, 1999)。

本研究は，この個人レベル予測の実証的検証に挑戦する。筆者らは先行研究 (Author, 20XX) において，大学受験を控えた高校生約 120 名を対象に，オンライン学習プラットフォーム上の学習行動指標と Big Five 特性の関連を構造方程式モデリングなどにより検討した。参加者はその後，全員が大学入試を経験し，実際の入試結果 (合格可否，入学先区分など) が得られた。そこで本研究では，この 120 名から得られた Big Five スコアのみを入力として，Park et al. (2023) のアーキテクチャに準拠した生成エージェントを参加者ごとに 1 体ずつ構築し，出願校選択から受験当日の挙動，結果までを仮想的に展開する「大学入試シミュレーション」を実行する。そのうえで，シミュレーションが生成した結果分布と，実測された入試結果との整合性を，集団レベルおよび個人レベルの両面で評価する。

この設計は 3 つの方法論的利点を持つ。第 1 に，Argyle et al. (2023) および Hewitt et al. (2024) が展開した silicon simulation の正確性検証を，米国政治ドメインから日本の教育ドメインへと外挿する試みとなる。第 2 に，参加者の実際の結果が既知であるため，Park (2024) が「判定保留」とする個人レベルの予測可能性について，厳密な ground truth に対する直接的な検証が可能となる。第 3 に，入力が Big Five というきわめて低次元な情報であることは，人生結果予測の情報的上限 (Lundberg et al., 2024) に対する下からの推定を与える。すなわち「パーソナリティ 5 次元だけ」で，どこまで迫れるのかという問いへの実証的回答となる。

本稿の構成は以下のとおりである。第 2 節では先行する 3 つの系譜—古典的 ABM，silicon sampling，generative agents—を整理する。第 3 節では，Big Five を入力とした生成エージェントのアーキテクチャとプロンプト設計を述べる。第 4 節では 120 名分のシミュレーションと実測結果の対照から得られた整合性指標を報告する。第 5 節では，個人レベル予測の成否と限界，および Wang et al. (2024) が指摘する平坦化バイアスについて議論する。最後に第 6 節で，Agent Bank (Park, 2024) の構想に本研究が与える示唆をまとめる。

---

## References

Ansolabehere, S., Rodden, J., & Snyder, J. M., Jr. (2008). The strength of issues: Using multiple measures to gauge preference stability, ideological constraint, and issue voting. *American Political Science Review, 102*(2), 215–232. https://doi.org/10.1017/S0003055408080210

Argyle, L. P., Busby, E. C., Fulda, N., Gubler, J. R., Rytting, C., & Wingate, D. (2023). Out of one, many: Using language models to simulate human samples. *Political Analysis, 31*(3), 337–351. https://doi.org/10.1017/pan.2023.2

Bates, J. (1994). The role of emotion in believable agents. *Communications of the ACM, 37*(7), 122–125. https://doi.org/10.1145/176789.176803

Gordon, M. L., Lam, M. S., Park, J. S., Patel, K., Hancock, J. T., Hashimoto, T., & Bernstein, M. S. (2022). Jury learning: Integrating dissenting voices into machine learning models. In *Proceedings of the 2022 CHI Conference on Human Factors in Computing Systems* (Article 115, pp. 1–19). Association for Computing Machinery. https://doi.org/10.1145/3491102.3502004

Hewitt, L., Ashokkumar, A., Ghezae, I., & Willer, R. (2024). *Predicting results of social science experiments using large language models* [Preprint]. Stanford University. https://docsend.com/view/qeeccuggec56k9hd

John, O. P., & Srivastava, S. (1999). The Big Five trait taxonomy: History, measurement, and theoretical perspectives. In L. A. Pervin & O. P. John (Eds.), *Handbook of personality: Theory and research* (2nd ed., pp. 102–138). Guilford Press.

Lundberg, I., Brand, J. E., & Jeon, N. (2024). The origins of unpredictability in life outcome prediction tasks. *Proceedings of the National Academy of Sciences, 121*(24), Article e2322973121. https://doi.org/10.1073/pnas.2322973121

Park, J. S. (2024). *CS 222: AI agents and simulations* [Lecture notes]. Stanford University. https://joonspk-research.github.io/cs222-fall24/

Park, J. S., Popowski, L., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2022). Social simulacra: Creating populated prototypes for social computing systems. In *Proceedings of the 35th Annual ACM Symposium on User Interface Software and Technology* (pp. 1–18). Association for Computing Machinery. https://doi.org/10.1145/3526113.3545616

Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. In *Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology* (Article 2, pp. 1–22). Association for Computing Machinery. https://doi.org/10.1145/3586183.3606763

Resnick, P., Iacovou, N., Suchak, M., Bergstrom, P., & Riedl, J. (1994). GroupLens: An open architecture for collaborative filtering of netnews. In *Proceedings of the 1994 ACM Conference on Computer Supported Cooperative Work* (pp. 175–186). Association for Computing Machinery. https://doi.org/10.1145/192844.192905

Rittel, H. W. J., & Webber, M. M. (1973). Dilemmas in a general theory of planning. *Policy Sciences, 4*(2), 155–169. https://doi.org/10.1007/BF01405730

Schelling, T. C. (1971). Dynamic models of segregation. *Journal of Mathematical Sociology, 1*(2), 143–186. https://doi.org/10.1080/0022250X.1971.9989794

Schelling, T. C. (1978). *Micromotives and macrobehavior*. W. W. Norton.

Wang, A., Morgenstern, J., & Dickerson, J. P. (2024). *Large language models that replace human participants can harmfully misportray and flatten identity groups* (arXiv:2402.01908) [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2402.01908
