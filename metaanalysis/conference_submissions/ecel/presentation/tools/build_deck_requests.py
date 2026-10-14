import json
from slides_requests import slide
IMG="https://raw.githubusercontent.com/etoki/paper/main/metaanalysis/conference_submissions/figures/modality_interaction_ecel.png"
S=[]
S+=slide("ecel_s03",2,"Today's Three Messages",
"**1. Conscientiousness matters in online learning, too.**\n**2. Extraversion is not zero.**\n> It can be **plus or minus**, depending on the course format.\n**3. Course format is not just background.**\n> It changes which personality helps students learn.",size=22)
S+=slide("ecel_s04",3,"Key Words: The Big Five",None,
 table={"box":(25,78,670,180),"size":15,"colw":[220,450],"highlight":[2,3],"rows":[
 ["Trait","In simple words"],["Openness","Curiosity about new things"],["**Conscientiousness**","Planning and self-control"],
 ["**Extraversion**","Liking people, high energy"],["Agreeableness","Kind and helpful"],["Neuroticism","Worrying easily"]]},
 extra_text=[("Linked to **non-cognitive / social and emotional skills**\nOECD Survey on Social and Emotional Skills is based on the Big Five:\nConscientiousness = {i:task performance}, Extraversion = {i:engaging with others}",(25,275,670,80),15)],
 source="Source: OECD (2021). Beyond Academic Learning: First Results from the Survey of Social and Emotional Skills.")
S+=slide("ecel_s05",4,"Why Study Personality in Education?",
"**1. Not only genes**: about **40%** heritable\n**2. It can change**: interventions show change in a few months\n**3. Little time to train it at school**\n> → Use it as a **hint for personalized learning**\n\n**Today's question: Which course format fits which students?**",size=20,
 source="Sources: Vukasović & Bratko (2015), Psychological Bulletin; Roberts et al. (2017), Psychological Bulletin.")
S+=slide("ecel_s06",5,"Face-to-Face: Conscientiousness Is Strongest",
"**Conscientiousness × grades: r ≈ 0.20 – 0.27**\n\nOpenness: small plus\nExtraversion, Agreeableness, Neuroticism: small or near zero\nIn education research, **r ≈ 0.2 is a meaningful size**\n\n**But what about online learning?**",size=20,
 source="Sources: Poropat (2009), Psychological Bulletin; Vedel (2014), Personality and Individual Differences; Mammadov (2022), Journal of Personality.")
S+=slide("ecel_s07",6,"Why Focus on Extraversion?",
"**Reason 1: Fewer face-to-face contacts online**\n> Extraversion is the trait most connected to people → it should change the most\n**Reason 2: My first study (Japanese students)**\n> Inside Extraversion, the directions were split:",size=17,body_box=(25,78,670,130),
 table={"box":(60,212,600,90),"size":15,"colw":[220,380],"rows":[["Facet","Link with online test activity"],["Assertiveness","**plus** (r = 0.27)"],["Sociability","slightly minus (not significant)"]]},
 extra_text=[("**Hypothesis: Online, Extraversion works differently from face-to-face.**",(25,318,670,30),17)],
 source="Source: Tokiwa (2025), Frontiers in Psychology, 16, 1420996.")
S+=slide("ecel_s08",7,"My Review: Extraversion Was Almost Zero",None,
 extra_text=[("25 studies collected → 12 with usable numbers",(25,78,670,28),18),
             ("**Is it really zero? Or is something hiding inside?**",(25,318,670,30),20)],
 table={"box":(150,110,420,200),"size":16,"colw":[260,160],"highlight":[5],"rows":[["Trait","pooled r"],["Conscientiousness","**0.167**"],["Openness","0.086"],["Agreeableness","0.112"],["Neuroticism","0.018"],["**Extraversion**","**0.002**"]]})
S+=slide("ecel_s09",8,"Key Word: pooled r",
"**pooled r = one average of r values from many studies**\n\n**Bigger studies count more.**\n\npooled r = 0.002\n> → \"When we put all 12 studies together, the link is almost zero.\"",size=22)
S+=slide("ecel_s10",9,"Online Learning Has 3 Formats",None,
 table={"box":(25,85,670,190),"size":16,"colw":[170,250,250],"rows":[["Format","What it is","Example"],
 ["{b:Asynchronous}","Recorded classes, any time","MOOCs"],["Synchronous","Live classes, same time","Zoom lectures"],["{r:Mixed-online}","Recorded + live","Many university online courses"]]},
 extra_text=[("**Face-to-face classes do not have this choice.**",(25,300,670,30),20)])
S+=slide("ecel_s11",10,"Did Mixing the Formats Make It \"Zero\"?",
"{b:Asynchronous}: **minus**\n{r:Mixed-online}: **plus**\n→ **The average becomes zero**\n\n**To check: split by format, and average again.**",size=24)
S+=slide("ecel_s12",11,"Key Word: Moderator",
"**Moderator = a condition that changes the strength or direction of a link**\n\nExample: umbrella → staying dry\n> The link is strong **only on rainy days**\n> → **Weather** is the moderator\n\n**Today's question:** Is **course format** a moderator of the link between personality and grades?",size=19)
S+=slide("ecel_s13",12,"Splitting 12 Studies by Format",None,
 table={"box":(25,82,250,150),"size":16,"colw":[170,80],"rows":[["Format","Studies"],["{b:Asynchronous}","**6**"],["{r:Mixed-online}","**5**"],["Synchronous","**1**"]]},
 extra_text=[("**Why only 1 synchronous study?**\n3 found → 2 measured satisfaction or engagement, **not grades**\n(2020 lockdown Zoom classes)\n\n**Main comparison:**\n{b:Asynchronous} vs. {r:Mixed-online}",(300,82,395,190),16),
             ("{i:4 studies did not state the format → coded from the original papers; all reasons recorded.}",(25,300,670,40),13)],
 source="Sources: Audet et al. (2021), Personality and Individual Differences; Sahinidis & Tsaknis (2021), Springer Proceedings.")
S+=slide("ecel_s14",13,"Two Steps of Analysis",None,
 extra_text=[("**Step 1: Pool by format**\nFor each trait × format, combine r values\n(bigger studies count more)\nRandom-effects model\n→ **10 pooled r values**\n(5 traits × 2 formats)",(25,82,320,220),16),
             ("**Step 2: One regression model**\n42 rows (each study × each trait)\nr is explained by:\n**trait** + **format** + **trait × format**\n→ Is the **combination** significant?",(375,82,320,220),16),
             ("**If both steps give the same answer, we can trust it more.**",(25,310,670,30),18)])
S+=slide("ecel_s15",14,"Key Words for the Results",
"**Confidence interval**: the true value is probably inside this range\n**p value**: the chance of this result by luck alone\n> **p < 0.05** → probably not just luck\n**Q between**: how far apart the two pooled r values are (async vs. mixed)\n> A standard **test for subgroup differences**\n> With 2 groups: **Q > 3.84 → p < 0.05**",size=18,
 source="Sources: Borenstein et al. (2009), Introduction to Meta-Analysis; Cochrane Handbook for Systematic Reviews of Interventions.")
S+=slide("ecel_s16",15,"Only Extraversion Clearly Differs by Format",None,
 table={"box":(25,82,670,230),"size":15,"colw":[190,130,130,120,100],"highlight":[3],"rows":[
 ["Trait","{b:Async} r (k)","{r:Mixed} r (k)","Q between","p"],
 ["Openness","0.223 (3)","0.019 (5)","1.96",".161"],["Conscientiousness","0.190 (4)","0.180 (5)","0.01",".903"],
 ["**Extraversion**","**−0.121 (3)**","**+0.059 (5)**","**17.60**","**< .001**"],
 ["Agreeableness","0.283 (3)","0.032 (5)","3.28",".070"],["Neuroticism","0.076 (4)","0.028 (5)","0.42",".519"]]},
 source="k = number of studies. Random-effects pooling (REML, HKSJ 95% CI).")
S+=slide("ecel_s17",16,"Extraversion Changes Direction by Format",None,
 image=(IMG,(25,76,430,282)),
 extra_text=[("{b:Asynchronous}\n**r = −0.121**\n\n{r:Mixed-online}\n**r = +0.059**\n\n**The \"zero\" was hiding two opposite effects.**",(470,90,225,250),18)],
 source="Error bars: 95% confidence intervals (HKSJ).")
S+=slide("ecel_s18",17,"Step 2 Gives the Same Answer",None,
 table={"box":(70,90,580,120),"size":17,"colw":[430,150],"rows":[["Test","p"],["Extraversion × Mixed-online","**0.009**"],["All 4 combinations together (Wald test)","**0.0085**"]]},
 extra_text=[("**Both p < 0.05 → Format is a real moderator for Extraversion.**",(25,250,670,40),20)])
S+=slide("ecel_s19",18,"Conscientiousness Is Stable Across Formats",None,
 extra_text=[("{b:Asynchronous}: **0.190**     {r:Mixed-online}: **0.180**\nDifference = 0.010 (p = 0.903)",(25,82,670,70),22)],
 table={"box":(110,175,500,95),"size":17,"colw":[160,170,170],"rows":[["","Conscientiousness","Extraversion"],["Across formats","**Stable**","**Depends on format**"]]})
S+=slide("ecel_s20",19,"What Course Designers Can Do",
"**Why minus in asynchronous courses?** {i:(hypothesis)}\n> Extraverted students get energy from people. Studying alone for a long time may lower motivation.\n**For Conscientiousness:** support planning and self-control (works in every format)\n**For Extraversion:** if the course is only asynchronous, **add a few live sessions**\n**Not about changing personality. It is about course design.**",size=17)
S+=slide("ecel_s21",20,"Limitations",
"**1. Small number of studies**\n> (Extraversion: async **3**, mixed 5) → **A discovery, not a final answer**\n**2. Format coding**\n> 4 studies coded by the author\n**3. Only 1 synchronous study**\n> Nothing can be said about live-only classes",size=20)
S+=slide("ecel_s22",21,"Three Take-Home Messages",
"**1. Conscientiousness matters in online learning, too.**\n> It does not change with the format.\n**2. Extraversion is not zero.**\n> {b:Asynchronous}: minus / {r:Mixed-online}: plus\n**3. Course format is an important moderator.**\n> Future studies should **report results by format**.",size=21)
S+=slide("ecel_s23",22,"Thank You",
"Full paper: ECEL 2026 Proceedings\npapers.academic-conferences.org/index.php/ecel\n\nEisuke Tokiwa / SUNBLAZE Co., Ltd.\nORCID: 0009-0009-7124-6669\neisuke.tokiwa@sunblaze.jp\n\n**Questions are welcome.**",size=20)
json.dump(S,open("all.json","w"),ensure_ascii=False)
# split into batches by slide boundaries
batches=[];cur=[]
for r in S:
    if "createSlide" in r and len(cur)>120: batches.append(cur);cur=[]
    cur.append(r)
batches.append(cur)
for i,b in enumerate(batches): json.dump(b,open(f"b{i}.json","w"),ensure_ascii=False,separators=(",",":"))
print(len(S),[len(b) for b in batches],[len(open(f"b{i}.json").read()) for i in range(len(batches))])
