# Team-strength / score models vs goals-only Dixon-Coles, and fixing total-goal under-prediction

Research date: 2026-11-03. Pipeline under study: Dixon-Coles (DC), rho correction, xi = 0.0019/day (half-life of about 365 days), L2 ridge 0.5 on attack/defence, 6 seasons of PL + Championship goals. Over 90 matches it predicted 188 goals against 243 actual (2.09 vs 2.70 per match).

## Q1. Published comparisons: which models beat goals-only Dixon-Coles, and by how much?

### Takeaway
Among score-based count models (Poisson, bivariate Poisson, DC, double Poisson, Weibull count), published RPS differences are tiny, usually in the 3rd or 4th decimal place, and no family reliably beats DC. The only clear published gains come from (a) feeding rating features (pi-ratings, Berrar ratings) into gradient-boosted trees, with RPS about 0.193 against about 0.208 to 0.210 for Poisson-family models on the Open International Soccer Database, and (b) using match statistics (shots, shots on target, corners) instead of goals as the rating input, which Wheatcroft showed beats goals for outcome forecasting and betting.

### Cited Findings
**Open International Soccer Database / 2017 Soccer Prediction Challenge.** The test set was 206 matches, scored by average RPS (lower is better). Table 1.2 of Bunker, Yeung & Fujii (arXiv, March 2024):
  - CatBoost + pi-ratings: RPS 0.1925, accuracy 0.5582 (Razali et al., post-challenge) — [Bunker et al. 2024](https://arxiv.org/abs/2403.07669)
  - TabNet + pi-ratings: RPS 0.1956, accuracy 0.5582 — [Bunker et al. 2024](https://arxiv.org/abs/2403.07669)
  - Bookmaker odds: RPS 0.2020, accuracy 0.5194 — [Bunker et al. 2024](https://arxiv.org/abs/2403.07669)
  - Elo ordered logit (Robberechts & Davis): RPS 0.2035, accuracy 0.5146 — [Bunker et al. 2024](https://arxiv.org/abs/2403.07669)
  - XGBoost + Berrar ratings: RPS 0.2054, accuracy 0.5194 (obtained after the challenge closed) — [Bunker et al. 2024](https://arxiv.org/abs/2403.07669); paper: Berrar, Lopes & Dubitzky, *Machine Learning* 108:97–126 (2019) — [ORO](https://oro.open.ac.uk/96634)
  - XGBoost + pi-ratings (Hubáček, Šourek & Železný, challenge winner): RPS 0.2063, accuracy 0.5243 — [Bunker et al. 2024](https://arxiv.org/abs/2403.07669); [CTU award news](https://cs.fel.cvut.cz/en/news/detail/1302)
  - Double Poisson (Hubáček et al., 2000/01–2005/06 subset only): RPS 0.2082, accuracy 0.4888 — [Bunker et al. 2024](https://arxiv.org/abs/2403.07669)
  - Hybrid Bayesian Network + modified pi-ratings (Constantinou, "Dolores"): RPS 0.2083, accuracy 0.5146 — [Bunker et al. 2024](https://arxiv.org/abs/2403.07669)
  - Bradley–Terry / hierarchical Poisson log-linear (INLA, Tsokos et al.): RPS 0.2087, accuracy 0.5388 — [Bunker et al. 2024](https://arxiv.org/abs/2403.07669)
  - In Hubáček et al.'s experimental review, which used ratings and statistical models only, Berrar ratings scored RPS 0.2101. Bivariate Poisson, double Poisson, double Weibull and pi-ratings all scored RPS 0.2103. Most score-based models are effectively tied. — [Bunker et al. 2024](https://arxiv.org/abs/2403.07669)
  - Review conclusion: gradient-boosted trees (CatBoost) on soccer-specific ratings such as pi-ratings are currently the best performers on goals-only datasets. — [Bunker et al. 2024](https://arxiv.org/abs/2403.07669)
  - Tsokos et al. found that direct outcome modelling (Bradley–Terry extensions) and hierarchical Poisson log-linear modelling of scores show similar predictive performance. — [Turing/Machine Learning 2019](https://www.turing.ac.uk/news/publications/modeling-outcomes-soccer-matches)
- **UCL/UEL study (arXiv 2508.20075, 2025):** independent Poisson, bivariate Poisson and DC were compared on Brier, RPS and log loss. DC had the best Brier and log scores on UEL. Bivariate Poisson had the best RPS (0.4085, multi-outcome scale). The authors call the differences negligible and judge DC the most robust overall. — [arXiv 2508.20075](https://arxiv.org/pdf/2508.20075)
- **PL conference paper (2024/25):** DC reduced log loss by less than 0.1% against independent Poisson. Paired tests found no significant difference. — [even3 proceedings PDF](https://static.even3.com/anais/1688473.pdf)
- **Weibull count (Boshnakov, Kharrat & McHale, IJF 33(2):458–466, 2017):** a bivariate Weibull count model with a copula. Goodness-of-fit chi-square p-values were 0.10 (home) and 0.16 (away) for Weibull, against 0.002 and 0.0002 for Poisson. The paper is evaluated with RPS against DC on EPL data and the authors report betting profitability, but I could not retrieve the RPS numbers. — [Salford repository](https://salford-repository.worktribe.com/output/1395286/a-bivariate-weibull-count-model-for-forecasting-association-football-scores); [Countr vignette](https://mirror.linux.duke.edu/cran/web/packages/Countr/vignettes/exampleFootball.pdf)
- **Negative binomial / Sarmanov DC extension (JRSS-C 2025, women's leagues):** the "Alternative Negative Binomial Sarmanov" model outperformed classical models, including DC, on fit and predictive accuracy across four women's leagues. — [arXiv 2307.02139](https://ar5iv.labs.arxiv.org/html/2307.02139)
- **Dynamic bivariate Poisson (Koopman & Lit, JRSS-A 2015; Tinbergen DP 12-099):** intensities follow stochastic state-space processes. The out-of-sample betting strategy on EPL 2010/11–2011/12 gave a "significant positive return" over bookmaker odds. I did not retrieve exact ROI figures. — [Tinbergen](https://tinbergen.nl/discussion-paper/3773/12-099-iii-a-dynamic-bivariate-poisson-model-for-analysing-and-forecasting-match-results-in-the-english-premier-league)
- **Ley, Van de Wiele & Van Eetvelde (Statistical Modelling 19(1), 2019):** of ten maximum-likelihood models (Bradley–Terry, independent Poisson, bivariate Poisson), the bivariate and independent Poisson models with smooth time decay were best. In a follow-up on the 2018 World Cup, bivariate Poisson with a 1095-day half-period had RPS 0.17366, with neighbouring variants differing only in the 3rd decimal. — [Statistical Modelling abstract](https://statmod.org/smij/Vol19/Iss1/Ley/Abstract.html); [arXiv 1806.03208](https://arxiv.org/pdf/1806.03208)
- **Wheatcroft GAP ratings on match statistics (LSE, arXiv Jan 2020):** uses football-data.co.uk statistics (shots, shots on target, corners) for many European leagues from 2000/01, with ordinal logistic regression on GAP-predicted statistics.
  - Predicted goals is "a relatively poor predictor". Predicted shots off target is the most informative single predictor, followed by corners, shots on target and goals.
  - Predicted statistics add information beyond the bookmaker odds.
  - Level-stakes and Kelly strategies both show long-run profit, significant in several combinations, but profit falls off in recent seasons, plausibly because the market adapted.
  - The earlier Wheatcroft (2020) paper showed that shots and corners beat goals as a basis for over/under 2.5 forecasting.
  — [Wheatcroft arXiv 2001.09097](https://arxiv.org/abs/2001.09097)
- **Pi-ratings (Constantinou & Fenton 2013)** are described as a dynamic rating with superior outcome-prediction results. They are now mainly used as features for gradient-boosted trees rather than as a standalone predictor. — [piratings CRAN vignette](https://cran.r-project.org/web/packages/piratings/vignettes/README.html)

### Inferences
- Ranking by evidence for improving on goals-only DC, best first:
  1. **Better inputs.** Shot- or xG-based ratings, or a goals/xG blend (Wheatcroft; see Q2).
  2. **Gradient-boosted trees on rating features.** RPS about 0.193 against about 0.208 for Poisson-family models, roughly 7% lower RPS on one dataset. But this only predicts H/D/A outcomes, not scorelines, so it doesn't directly serve exact-score or goalscorer tasks.
  3. **Dynamic / state-space ratings.** Positive betting results, but no clean head-to-head RPS against DC was found.
  4. **Over-dispersed or alternative count families** (negative binomial, Weibull count, bivariate Poisson). These fit marginals better, but outcome-RPS gains are about 0.000–0.002.
- For a scoreline pipeline the model family matters far less than the inputs and the calibration of the goal rate.
- The 206-match test set is very small. RPS differences of about 0.001 in it are almost certainly noise.

### Gaps
- No published PL head-to-head RPS table for DC against Skellam, against Baio & Blangiardo (2010) Bayesian hierarchical, or against Rue & Salvesen (2000). Not found in this session.
- Exact RPS numbers from Boshnakov et al. (2017) and ROI figures from Koopman & Lit could not be extracted. Full texts: [Salford PDF](https://salford-repository.worktribe.com/OutputFile/1493215), [Tinbergen PDF](https://papers.tinbergen.nl/12099.pdf).
- The 2023 Soccer Prediction Challenge results were not available in the 2024 review.
- No penaltyblog benchmark with RPS numbers per model was found.

## Q2. Does using xG or shots in the rating fit improve forecasting in the PL?

### Takeaway
Evidence leans yes, especially for shots and shots-off-target/corner volume (Wheatcroft) and for xG over shorter samples (Princeton thesis). But no peer-reviewed PL study shows an xG-only model beating goals-based DC on scoreline forecasting. The safest practical recipe is a goals + xG (or goals + shots) blend.

### Cited Findings
- In Wheatcroft (2020), GAP-predicted shots and corners were more informative than predicted goals for match outcome, and gave betting profit beyond the odds (multi-league, football-data.co.uk data). — [arXiv 2001.09097](https://arxiv.org/abs/2001.09097)
- A Princeton senior thesis (Rubin, PL 2017–2022) found xG roughly unbiased for goals, though skilled teams can outperform xG over extended periods. An xG-based forecast beat an observed-goals model "with significantly better accuracy" when training on a smaller sample and when allowing time-varying strengths. This is an undergraduate thesis, not peer reviewed. — [Princeton DataSpace](https://dataspace.princeton.edu/handle/88435/dsp01xw42nc19k)
- Ijtsma's analysis found xG ratio the most predictive of rest-of-season goal ratio and points per game. — [Tony ElHabr summary](https://tonyelhabr.rbind.io/posts/xg-predictor-future-results/)
- Counterpoint (The Conversation, 2022): "no xG model has been shown to be superior to Dixon-Coles in predicting the future." — [The Conversation](https://theconversation.com/a-brief-history-of-statistics-in-football-why-actual-goals-remain-king-in-predicting-who-will-win-194314)
- Opisthokonta's `goalmodel` R package lets you fit DC-style models on xG instead of goals, so the setup is easy to test. — [goalmodel GitHub](https://github.com/opisthokonta/goalmodel); [R-bloggers intro](https://www.r-bloggers.com/2018/10/introducing-the-goalmodel-r-package/)
- Bunker et al. (2024) recommend xG or shots as an alternative target variable because they are less rare than goals. — [Bunker et al. 2024](https://arxiv.org/abs/2403.07669)

### Inferences
- With Understat unavailable, the feasible signals are:
  - **Historical shots and shots on target** from football-data.co.uk. Available for all 6 seasons, so you can backtest the full history.
  - **FPL xG**, current season only. Useful as an in-season blend, not for history.
- A cheap test is to fit DC on a weighted target, e.g. 0.5·goals + 0.5·(k·shots-on-target), or to use a non-integer Poisson-like quasi-likelihood. Then compare backtest RPS and log loss against the plan-01 baseline.

### Gaps
- Opisthokonta's and StatsBomb's own quantitative xG-vs-goals forecasting comparisons were not retrieved.
- No PL-specific peer-reviewed RPS comparison was found.
- Martin Eastwood's (pena.lt) xG-DC post was not found.

## Q3. Time-decay tuning: is xi = 0.0019/day (1-year half-life) too slow?

### Takeaway
The current xi is essentially the Dixon-Coles original: 0.0065 per half-week works out to about 0.0019 per day. Published optimal half-lives range from about 6 months to 3 years depending on dataset, and the differences are small. Decay speed is unlikely to explain a 23% goal shortfall. It should be tuned by backtest rather than assumed.

### Cited Findings
- Dixon & Coles chose xi by maximising out-of-sample outcome prediction, with time measured in half-week units. The same numeric xi means different things in different time units. — [arXiv 1706.04943](https://arxiv.org/pdf/1706.04943); [penaltyblog overview](https://penaltyblog.readthedocs.io/en/latest/models/overview.html)
- penaltyblog's example uses xi = 0.001 (per day); xi = 0 weights all matches equally. — [penaltyblog example](https://penaltyblog.readthedocs.io/en/latest/models/example.html)
- Ley et al. use weight = 0.5^(t / half-period). In later EURO 2024 work, half-periods from 0.5 to 6 years were tested and 3 years (1095.75 days) gave the best RPS for national teams. Van Eetvelde's slides report optima of about 180–240 days on other data. — [arXiv 2410.09068](https://arxiv.org/pdf/2410.09068); [Van Eetvelde slides](https://bodai.unibs.it/bdsports/wp-content/uploads/sites/2/2021/03/Slides_Napels_HVE.pdf)
- In the 2018 World Cup study, the 730, 1095 and 1460-day half-periods gave nearly identical RPS. — [arXiv 1806.03208](https://arxiv.org/pdf/1806.03208)

### Inferences
- Arithmetic: 0.0065 per half-week ÷ 3.5 days = 0.00186 per day, half-life = ln2 / 0.0019 ≈ 365 days. With a 1-year half-life, matches from 6 seasons ago carry about 1.6% weight. The 6-season window therefore costs little in recency, but the old high-weight seasons still drag the global intercept.
- More useful than changing the team-rating xi is a **separate, faster decay (or a per-season parameter) for the intercept and home advantage**, since league-wide goal rates and home advantage shift season to season (Q5, Q6).

### Gaps
- I could not confirm the specific "half-period about 390 days" optimum often attributed to Ley et al. (2019) for domestic leagues.
- No Opisthokonta xi-tuning post was retrieved.

## Q4. Including Championship data and handling promoted teams

### Takeaway
There is little formal published evidence. The main documented quantity is that promoted teams' xG falls sharply on stepping up. If Championship matches share the PL's intercept, they bias the league goal rate and promoted teams' ratings.

### Cited Findings
- Data Between the Lines' "TEAM" model: promoted teams' xG fell by an average of 33.49% (normalised) from their Championship season to their first PL season. The author built this as a promoted-team adjustment for a PL team-strength model. — [The Promotion Effect (Substack)](https://databetweenthelines.substack.com/p/the-promotion-effect)
- Opta's 2025-26 supercomputer gave promoted sides a 34.1% chance of finishing bottom, showing that promoted teams are expected to be weak. — [soccernews/Opta](https://www.soccernews.com/opta-supercomputer-backs-liverpool-for-premier-league-title-but-can-newly-promoted-teams-upset-the/388459/)
- Wheatcroft's GAP setup carries ratings across seasons and is optimised over all leagues, which is one documented way to handle cross-division movement. — [arXiv 2001.09097](https://arxiv.org/abs/2001.09097)

### Inferences
- Give each division its own intercept and its own home advantage, ideally per season (e.g. `mu[league, season]`). Without this, the single intercept averages PL and Championship scoring and the attack/defence ratings absorb the division gap. Championship matches between Championship teams then pull the shared rate.
- Alternatively, treat the Championship only as a prior for promoted teams: the Championship rating minus a fixed step-up penalty fitted on past promotions.

### Gaps
- No academic paper quantifying the RPS gain from including second-tier data in PL rating fits was found.

## Q5. Why Poisson models under-predict totals in recent PL seasons, and fixes

### Takeaway
PL scoring jumped to 3.28 goals per game in 2023-24 (a record), then fell to about 2.93 (2024-25) and about 2.79 (2025-26 to February 2026). Late and stoppage-time goals are at record shares. A model averaging 6 seasons with Championship data will lag these level shifts. However, a 2.09 per-match prediction is below every recent PL season. The cause is therefore probably more than drift: most likely summing modal scorelines instead of expected goals, a shared PL/Championship intercept, or a Jensen/shrinkage effect.

### Cited Findings
- 2023-24: about 3.28 goals per game (1,246 goals), the PL record. 112 goals were scored in the 90th minute or later. — [The Analyst](https://theanalyst.com/?p=150208); [Premier League](https://www.premierleague.com/news/4322877)
- 2024-25: 1,115 goals, 2.93 per game, 131 fewer than 2023-24. Stoppage-time goals fell to 94, and stoppage time per game was about 1 minute 51 seconds shorter than 2023-24. Sources conflict slightly: Extra Time lists 2.97. — [11v11](https://www.11v11.com/competitions/premier-league/2025/); [Premier League](https://www.premierleague.com/news/4322877); [Extra Time](https://extratime.com/stats/2129/2287/premier-league/)
- Premier League/Opta (3 Feb 2026), like-for-like matches:
  - Goals per game: 2.79 (2025-26, 666 goals in 239 matches), 3.04 (2024-25 at the same point), 3.22 (2023-24).
  - Average total match length: 100:20, 99:56 and 101:43.
  - Ball-in-play time: 55:35, 57:05 and 58:34.
  - Goals after the 90th minute: 11.4%, 9.3% and 9.5% of goals.
  - Stated cause: the anti-time-wasting directive introduced before 2023-24 increased stoppage time.
  — [Premier League/Opta](https://www.premierleague.com/en/news/4580269/why-more-late-goals-are-being-scored-in-2025-26-premier-league-season)
- 2026-27 after 50 matches (3 Oct 2026): the article gives no goals per game. Draws were 32%, the highest PL draw share ever, which suggests lower scoring early this season. — [Premier League/Opta](https://www.premierleague.com/en/news/4729254/more-comebacks-and-quicker-restarts-five-trends-from-202627-so-far)
- penaltyblog provides a negative binomial model for over-dispersion (variance > mean). — [penaltyblog overview](https://penaltyblog.readthedocs.io/en/latest/models/overview.html)
- The Weibull count model fits goal-count marginals much better than Poisson (chi-square p = 0.10–0.16 against 0.002–0.0002). — [Countr vignette](https://mirror.linux.duke.edu/cran/web/packages/Countr/vignettes/exampleFootball.pdf)

### Inferences
- **Check first: modal scoreline vs expected goals.** If "predicted goals" is the sum of each match's most likely scoreline (1-0, 1-1, 2-1…), it will systematically understate totals. The mode of a Poisson with λ ≈ 1.3–1.6 is 1, while the mean is 1.3–1.6, so modal totals of about 2.0–2.2 per match are expected even from a perfectly calibrated model.
  - Calibrate totals with Σ(λ_home + λ_away), not modal scores.
  - For exact-score content, keep the modal score but calibrate it separately, or pick the most likely score conditional on the most likely total.
- **Per-league, per-season intercept.** Add `mu[league, season]` (or an intercept with fast decay, e.g. a 90–180 day half-life). This lets the PL rate follow 3.28, then 2.93, then about 2.8.
- **Shrinkage/Jensen check.** L2 ridge pulls log-ratings toward 0. With the sum-to-zero constraint, the fitted intercept should compensate on average. Verify on the training set that mean predicted λ equals mean observed goals per league-season; if not, refit the intercept with ratings held fixed.
- **Post-hoc total calibration.** Fit a single multiplier c on recent PL matches so that Σλ matches observed goals, then refit rho. This is cheap, and the backtest decides.
- **Over-dispersion** (negative binomial / Weibull) mostly fixes the tails (more 4+ goal games). It does not fix a biased mean.
- **Context for the 90-match sample.** 243/90 = 2.70 per game matches the recent PL range of about 2.8–3.0. A 2.09 forecast is low against any recent-season prior.

### Gaps
- No full-season 2025-26 goals-per-game figure was found (only to February 2026), and no 2026-27 goals per game was found.
- No published paper evaluating a "league-season goal-rate parameter" fix specifically for DC was found.

## Q6. Home advantage trend post-COVID; does time-varying home advantage help?

### Takeaway
PL home advantage has been falling for decades and hit record lows recently: 36% home wins in 2026-27 after 50 matches, the lowest ever. A fixed home advantage fitted over 6 seasons will overstate it now. Allowing home advantage to vary (per season, or with faster decay) is well motivated, though I found no published RPS gain from doing so.

### Cited Findings
- 2020-21 (no fans): away wins (40%) exceeded home wins (38%) for the first time in PL history. Home advantage recovered with fans, then weakened again. — [Racing Post](https://racingpost.com/sport/opinion/is-the-decrease-in-home-wins-in-the-premier-league-the-new-normal-azHrZ0w3mly3)
- Fantasy Football Scout (Aug 2026) table: 2024-25 had 40.8% home wins against 34.7% away, and 2025-26 had 42.6% home against 30.0% away. Since 2018-19 the average is 44.8% home and 32.0% away. A Feb 2026 piece called 42% home wins the 5th-lowest in PL history. These are fan/betting sources, and FISO's mid-season figure (44% at GW25) differs. — [Fantasy Football Scout](https://www.fantasyfootballscout.co.uk/2026/08/19/how-important-is-home-advantage-in-fpl); [FISO](https://www.fiso.co.uk/premier-league-scoring-trends-2024-25-season-vs-2025-26-gw25-whats-changed/)
- 2026-27 after 50 matches: 36% home wins (the PL record low, below 2020-21), 32% draws and 32% away wins. — [Premier League/Opta, 3 Oct 2026](https://www.premierleague.com/en/news/4729254/more-comebacks-and-quicker-restarts-five-trends-from-202627-so-far)
- Long-run English football trend: home win share has fallen from over 60% (late 19th century) towards 40% across all four tiers, with away wins now consistently above 30%. Elvidge builds the time-varying home advantage directly into his Bayesian team-strength model. — [Sean Elvidge, Jan 2025, updated Sept 2026](https://seanelvidge.com/articles/2025/Home_advantage_in_English_football/)

### Inferences
- Use a per-season home advantage (or a faster-decayed home parameter), estimated separately for PL and Championship.
- A smaller home advantage moves probability toward draws and away wins but barely changes total goals. It is a secondary fix for outcome calibration, not for the goal shortfall.

### Gaps
- No published comparison (RPS or log loss) of fixed against time-varying home advantage in DC-type models was found.
- No home goal-difference-per-game series by season was retrieved.
