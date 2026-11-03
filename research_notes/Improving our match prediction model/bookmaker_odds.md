# Using bookmaker odds to improve football score predictions

Scope: evidence that market probabilities beat models; turning odds into a Dixon-Coles / Poisson score matrix; blending market and model; what odds data is available before the first kickoff of a gameweek. Research date: November 2026.

## (a) Do bookmaker probabilities beat statistical models? By how much?

### Takeaway
The literature consistently treats de-margined bookmaker odds as the most accurate public forecast of football 1X2 outcomes. Typical results-based models (Elo, ordered probit, Poisson) come close but fall slightly short. In the one study where I found exact numbers (Egidi et al.), a Bayesian Poisson model trails Shin-adjusted odds by about 0.003 to 0.017 in average probability on the correct outcome, with the largest gap in the Premier League. I could not get RPS or log-loss figures for Pinnacle closing odds from a primary source.

### Cited Findings
- Štrumbelj (2014), *International Journal of Forecasting* 30(4), 934–943, is cited as empirical evidence that "betting odds are the most accurate available source of probability forecasts for sports": odds-based forecasts are "better, or at least as good as, statistical models, which use sport-specific predictors and/or expert tipsters". — [Egidi, Pauli & Torelli, arXiv:1802.08848](https://arxiv.org/abs/1802.08848); IJF citation from [shin on PyPI](https://pypi.org/project/shin)
- Egidi et al. Table 3, average probability given to the observed 3-way result over a test season (2016/17), averaged over 7 bookmakers (higher is better). Premier League: model 0.4349, Shin odds 0.4516, basic-normalised odds 0.4480. Bundesliga: 0.4010 / 0.4100 / 0.4072. La Liga: 0.4553 / 0.4584 / 0.4549. Serie A: 0.4430 / 0.4554 / 0.4507. The authors say their model "does not improve the bookmakers' probabilities" on this measure. — [arXiv:1802.08848 (PDF)](https://arxiv.org/pdf/1802.08848)
- Forrest, Goddard & Simmons (2005), "Odds-setters as forecasters: the case of English football", *IJF* 21(3), 551–564. They compared published odds with a benchmark ordered-probit model built on many quantifiable variables, over nearly 10,000 English matches. They found a trend of odds-setters becoming more expert over time, which coincided with competition raising the cost of mispriced odds. I did not see the exact accuracy figures. — [Bangor research portal](https://research.bangor.ac.uk/portal/en/researchoutputs/odds-setters-as-forecasters-the-case-of-english-football(f1f76c06-f6cb-4440-91f3-71cd60705c91)/export.html); [Lancaster eprints](https://eprints.lancs.ac.uk/id/eprint/44019)
- Hvattum & Arntzen (2010), "Using ELO ratings for match result prediction in association football", *IJF* 26(3), 460–470. Elo plus ordered logit was reported as inferior to forecasts based on betting odds. This comes from secondary summaries; I did not read the paper's own abstract or numbers. — [Graham Kendall blog review](https://graham-kendall.com/blog/?cat=117); [xG Football Club Substack](https://thexgfootballclub.substack.com/p/a-new-horizon-in-football-forecasting-the-betting-odds-based-elo-system-f5032018f546)
- Wunderlich & Memmert built a "Betting Odds Rating System", an Elo variant updated from odds rather than results. They also wrote "Are betting returns a useful measure of accuracy in (sports) forecasting?" (*IJF* 36(2), 713–722). — [xG Football Club Substack](https://thexgfootballclub.substack.com/p/a-new-horizon-in-football-forecasting-the-betting-odds-based-elo-system-f5032018f546); [IDEAS/RePEc listing](https://ideas.repec.org/a/bpj/jqsprt/v17y2021i4p273-287n1.html)
- opisthokonta (2019) fitted team ratings to odds-implied expected goals for the 2015/16 EPL. Leicester's ratings came out near average from odds but much higher from results. The author reads this as consistent with the Odds-Elo finding that odds-based ratings are more stable. — [R-bloggers: Expected goals from bookmaker odds](https://www.r-bloggers.com/2019/01/expected-goals-from-bookmaker-odds/)
- Constantinou & Fenton's pi-ratings "outperform considerably" Elo and were reported profitable against bookmakers over five EPL seasons (2007/08 to 2011/12), after the margin. This is a claim about betting returns, not calibration. — [De Gruyter, JQAS 2013](https://www.degruyterbrill.com/document/doi/10.1515/jqas-2012-0036/html); [arXiv:2003.09384](https://arxiv.org/pdf/2003.09384)
- Wheatcroft (2021) argues that the Ignorance score (log loss) beats RPS and Brier as a scoring rule for football forecasts. — summary via [search result for Wheatcroft/RPS critique](https://qmro.qmul.ac.uk/xmlui/handle/123456789/10783?show=full)
- Pinnacle's de-margined closing line is widely treated as the benchmark "true probability" because it reflects high-volume sharp money. One vendor example shows Pinnacle's overround tightening from 6.84% at opening to 2.60% at kickoff. These are vendor and blog claims, not peer-reviewed. — [oddspapi CLV article](https://oddspapi.io/blog/?p=3056); [thestatsapi Pinnacle page](https://thestatsapi.com/odds-api/pinnacle)
- DataGolf (golf, not football) found Pinnacle's and DraftKings' de-margined closing odds both roughly calibrated. They note that calibration plots cannot separate two well-calibrated forecasters. — [DataGolf: analyzing betting odds](https://datagolf.com/analyzing-betting-odds)

### Inferences
- In the Egidi table, the EPL gap of 0.017 (0.435 vs 0.452) is larger than the gaps in the other leagues. Plausible reasons: EPL markets are deeper, and the odds include information the model lacks, such as lineups, injuries and motivation.
- This pipeline's Dixon-Coles model is results-only, so it is likely to trail odds by a similar margin. If our 1X2 probabilities sit 1 to 2 percentage points below the market's on the actual result, that is the expected gap, not a bug.
- Closing odds should be sharper than the Friday pre-closing odds, because the margin shrinks and late news (lineups) gets priced in. The pipeline is frozen before kickoff, so it can only use pre-closing odds. Closing odds still work as a backtest benchmark or upper bound.

### Gaps
- I found no primary RPS or log-loss numbers for Pinnacle closing odds against a statistical model on EPL data. Štrumbelj (2014) and Constantinou's thesis (which has a section on bookmaker RPS) were not retrieved in full.
- No peer-reviewed calibration study of Pinnacle soccer closing lines was found. The "Journal of Sports Sciences" claim on vendor sites could not be verified.
- No quantitative opening-vs-closing accuracy gap for football was found.

## (b) Removing the overround, and converting odds into Poisson rates or a full score matrix

### Takeaway
Use Shin's method or the power method rather than basic normalisation for 1X2. Then solve for home and away Poisson rates (λ, μ) that reproduce the market 1X2 probabilities. Adding the over/under 2.5 price pins down the total and lets you also fit a Dixon-Coles ρ. penaltyblog does this out of the box in Python.

### Cited Findings: overround removal
- penaltyblog's `implied` module offers multiplicative (the default), additive, power, Shin, differential margin weighting, odds ratio and logarithmic methods. Some return a fitted parameter (Shin's z, power's k). — [penaltyblog implied docs](https://penaltyblog.readthedocs.io/en/master/implied/implied.html)
- The R `implied` package by opisthokonta offers basic, wpo (Buchdahl's margin weights proportional to odds), or (Cheung's odds ratio), power, additive, shin, bb (balanced books) and jsd. The vignette calls basic normalisation "the least accurate" of its methods but gives no football benchmark. For 2-outcome markets such as O/U 2.5, Shin is equivalent to additive. — [implied package vignette](https://cran.uni-muenster.de/web/packages/implied/vignettes/introduction.html)
- Štrumbelj (2014) found Shin's method better than both basic normalisation and the regression approach of Forrest et al., measured by RPS. Cain et al. (2002, 2003) and Smith et al. (2009) also found Shin better than basic normalisation. — [Egidi et al., arXiv:1802.08848](https://arxiv.org/pdf/1802.08848)
- Compared with basic normalisation, Shin lowers longshot probabilities and raises favourites' probabilities. This corrects the favourite-longshot bias. — [Egidi et al.](https://arxiv.org/pdf/1802.08848)
- A La Liga study found that both normalised and Shin probabilities still show favourite bias. — [Springer, Annals of OR (s10479-022-04722-3)](https://link.springer.com/article/10.1007/s10479-022-04722-3)
- Goto, Takeishi & Yairi (arXiv, April 2026) used 90,014 matches from 5 bookmakers. Their odds-only method "OO-EPC" beat multiplicative, Shin and power for most bookmakers. Their one-parameter "FL-GLM", fitted on history, beat multinomial and logistic GLMs. I did not see the metric values. — [arXiv:2604.17194](https://arxiv.org/abs/2604.17194)

### Cited Findings: odds to (λ, μ) and a score matrix
- penaltyblog `goal_expectancy(home, draw, away, dc_adj=True, rho=0.001, remove_overround=True, objective='brier'|'cross_entropy', max_goals=15)` returns `home_exp`, `away_exp`, `error` and `success`. It optimises (log λ, log μ) with SciPy L-BFGS-B over a Poisson grid with an optional Dixon-Coles low-score correction. Home wins are the cells below the diagonal, draws the diagonal, away wins above. The docs suggest ρ around 0.001 to 0.01. — [penaltyblog goal expectancy docs](https://penaltyblog.readthedocs.io/en/master/models/goal_expectancy.html)
- penaltyblog `goal_expectancy_extended` takes 1X2 plus O/U 2.5 probabilities. It reverse-engineers μ_home, μ_away and an implied Dixon-Coles ρ together. — [penaltyblog goal expectancy docs](https://penaltyblog.readthedocs.io/en/master/models/goal_expectancy.html) (I only saw this in a search snippet, not the full API page)
- opisthokonta (2019): convert 1X2 odds with the power method, then search for the (λ, μ) pair whose Poisson outcome probabilities match. Code is in a GitHub gist. The `goalmodel` R package can then fit attack and defence ratings to these expected goals by least squares (`model="gaussian"`/`"ls"`, flagged experimental). — [R-bloggers 2019](https://www.r-bloggers.com/2019/01/expected-goals-from-bookmaker-odds/)
- opisthokonta (2020): from O/U odds alone, solve P(total ≤ 2) = Σ_{i=0}^{2} e^{-λ}λ^i/i! for the total λ with a root-finder (`uniroot`). Example: 62% under gives λ ≈ 2.21. O/U alone cannot split the total between the teams. — [R-bloggers 2020](https://www.r-bloggers.com/2020/09/expected-goals-from-over-under-odds/)
- Egidi et al. get each bookmaker's implied (θ1, θ2) by solving the equations for P(home win) and P(away win) under a Skellam (Poisson-difference) distribution for the goal difference. — [arXiv:1802.08848](https://arxiv.org/pdf/1802.08848)

### Inferences
- A practical recipe for this pipeline:
  1. De-margin the 1X2 prices (Shin or power) and the O/U 2.5 prices (power, since Shin equals additive for two outcomes).
  2. Fit (λ, μ, optionally ρ) by minimising the squared error between the market targets and the probabilities from the DC grid: P(H), P(D), P(A) and P(over 2.5).
  3. Build the score matrix with the existing Dixon-Coles code.

  That is 3 free parameters against 4 targets, with 3 of them independent: 1X2 gives 2 constraints and O/U gives 1. The fit is well-determined. Without O/U, ρ must be fixed.
- The Asian handicap line (`AHh` plus `AvgAHH`/`AvgAHA`) is mostly a goal-difference constraint. It could replace 1X2 or be added as a target, but it adds little beyond 1X2. Quarter and half lines need care because of push and half-win payoffs.
- Odds give a full score matrix only through the parametric (Poisson/DC) assumption. Exact-score markets would be needed to capture any non-Poisson shape, and football-data does not carry them.

### Gaps
- No football-specific head-to-head of power vs Shin vs odds ratio with numbers was found. Štrumbelj's full results were not retrieved.
- No published accuracy comparison was found between exact-score predictions from odds-implied λ/μ and those from a DC model.

## (c) Blending market and model

### Takeaway
The main approach in the literature is a convex combination of model rates and market-implied rates, sometimes learned per match (Egidi et al.). Feeding odds into a model as features or a prior is the other route. Egidi et al. found the market and historical data carried comparable information, but their blended model still did not beat the odds on average-probability accuracy. Its claimed betting profits should be treated with caution.

### Cited Findings
- Egidi, Pauli & Torelli: y ~ Poisson(p·θ + (1−p)·λ). Here θ are the results-based DC-style log-linear rates (home advantage, attack and defence), λ are bookmaker-implied rates, and p ~ Beta(a, b) is estimated. The training data was 9 seasons (2007/08 to 2015/16) and the test season 2016/17, across Bundesliga, EPL, La Liga and Serie A. The posterior for p suggests "the amount of information that stems from the bookmakers is comparable with that arising from historical information." — [arXiv:1802.08848 (PDF)](https://arxiv.org/pdf/1802.08848)
- Same paper: the blended model's average correct-outcome probability was still below the Shin odds in all 4 leagues (EPL 0.4349 vs 0.4516). The authors report positive expected betting profits from their posterior probabilities against all 7 bookmakers. — [arXiv:1802.08848 (PDF)](https://arxiv.org/pdf/1802.08848)
- Londono & Hassan (2015) used betting odds to elicit hyperparameters of a Dirichlet prior on 3-way outcomes, then updated it with observed results. — cited in [Egidi et al.](https://arxiv.org/pdf/1802.08848)
- opisthokonta notes a related arXiv paper that blends a score-based model with odds-derived expected goals. — [R-bloggers 2019](https://www.r-bloggers.com/2019/01/expected-goals-from-bookmaker-odds/)
- Goto et al. (2026) argue that existing GLMs "model relationships the Efficient Market Hypothesis already captures". They propose fitting only a favourite-longshot correction on top of odds. — [arXiv:2604.17194](https://arxiv.org/abs/2604.17194)

### Inferences
- The simplest version for this pipeline is to blend the rates: λ_final = w·λ_model + (1−w)·λ_market, and the same for μ. Pick w by backtest against the plan-01 baseline. Given the evidence above, expect the best w to be small, around 0 to 0.4 on the model.
- An alternative is to blend the score matrices or 1X2 probabilities with a log-linear pool. A rate blend keeps the DC structure, which matters for predicting exact scores.
- Market λ/μ can also anchor scorer and minute predictions. Scale each player's share of team expected goals by the market team λ, not the model's.
- The Egidi profits came from betting the model against odds that had already been folded into it. They do not show the blend is better calibrated, so this project should not rely on them.

### Gaps
- I found no study quantifying the gain in RPS or log loss from a model + market ensemble over the market alone for EPL.
- No evidence was found on whether such blends help exact-score points specifically.

## (d) Practical data availability before the first kickoff

### Takeaway
football-data.co.uk's `fixtures.csv` carries pre-match odds for upcoming fixtures. They are collected on Friday afternoon (by about 17:00 UK) for weekend games and by 13:00 on Tuesday for midweek games, so they are usable before most gameweeks freeze. The season CSVs add closing odds (the `C` suffix) only after the matches are played. For free goalscorer odds, The Odds API lists EPL anytime-scorer markets, but only for US bookmakers.

### Cited Findings: football-data.co.uk
- Upcoming fixtures with odds: `https://football-data.co.uk/fixtures.csv` (and `.xlsx`) give "the latest main leagues fixtures list, with the match dates, times and betting odds." The page showed "Latest fixtures uploaded: 03/11/26 10:17 UK time". Updates are announced on Twitter as @12Xpert, and the site advises clearing the browser cache if older fixtures show up. — [football-data matches.php](https://www.football-data.co.uk/matches.php)
- Timing: weekend-fixture odds "are collected ... on Fridays afternoons", generally by 17:00 UK. Midweek odds are collected on Tuesdays no later than 13:00 UK. — [football-data matches.php](https://www.football-data.co.uk/matches.php)
- notes.txt: "Betting odds for weekend games are collected Friday afternoons, and on Tuesday afternoons for midweek games." Columns without a `C` are pre-closing. Closing odds add `C` after the bookmaker, Max or Avg code (e.g. `B365CH`). — [football-data notes.txt](https://www.football-data.co.uk/notes.txt)
- Columns, 1X2: `B365H/D/A`, Pinnacle `PSH/PSD/PSA` (older seasons `PH/PD/PA`), `MaxH/D/A`, `AvgH/D/A`, Betfair Exchange `BFEH/BFED/BFEA`. O/U 2.5: `B365>2.5`/`B365<2.5`, `P>2.5`/`P<2.5`, `Max>2.5`, `Avg>2.5`, etc. Asian handicap: `AHh` (market handicap size, from 2019/20), `B365AHH/B365AHA`, `PAHH/PAHA`, `MaxAHH/MaxAHA`, `AvgAHH/AvgAHA`. Older Betbrain columns (`BbAv…`) are historical only. — [football-data notes.txt](https://www.football-data.co.uk/notes.txt)
- Terms: the site's disclaimer page only covers accuracy. It states no licence, scraping rules or attribution requirement. — [football-data disclaimer](https://www.football-data.co.uk/disclaimer.php)

### Cited Findings: other sources
- The Odds API free "Starter" plan gives 500 credits a month and covers all sports and markets (h2h, spreads, totals, outrights) and "most bookmakers". UK books listed include William Hill, Ladbrokes, Betfair, Paddy Power and others; Pinnacle is listed under EU. Paid plans: $30 for 20K, $59 for 100K, $119 for 5M, $249 for 15M credits a month. Historical snapshots go back to 2020. The page is ambiguous about whether the free tier includes historical data. — [the-odds-api.com](https://the-odds-api.com/)
- The Odds API soccer player props (EPL, La Liga, Serie A, Bundesliga, Ligue 1, MLS) include `player_goal_scorer_anytime`, `player_first_goal_scorer` and `player_last_goal_scorer`. They are fetched one event at a time via `/events/{eventId}/odds`, and "Coverage is currently limited to US bookmakers." — [The Odds API betting markets](https://the-odds-api.com/sports-odds-data/betting-markets.html)
- Betfair Exchange API: each account gets a free delayed app key (prices delayed 1–180 s; an older page says 1–60 s) and a live key (one-off £299). Keys are for personal betting use, commercial use needs Betfair's approval, and accounts must be KYC-verified. — [Betfair developer docs](https://betfair-developer-docs.atlassian.net/wiki/x/J4DS)

### Inferences
- **Freeze-time fit:** for a weekend gameweek whose first kickoff is Saturday lunchtime, Friday ~17:00 odds from `fixtures.csv` are available in time. For a Friday 20:00 opener they arrive about 3 hours before kickoff, which is tight but workable if `predict` runs on Friday evening. For midweek rounds, Tuesday 13:00 odds come before the usual 19:30/20:00 kickoffs. The pipeline should read odds from `fixtures.csv` at predict time and store them alongside the frozen prediction, since the file is overwritten each round.
- **Backtest:** for past seasons, use the non-`C` columns (Friday/Tuesday pre-closing) in the season CSVs. They mirror what `fixtures.csv` showed at freeze time, so the backtest stays honest. The `C` columns would leak information. This is an inference: the notes say the pre-closing columns are the Friday/Tuesday odds but do not explicitly say they match `fixtures.csv`.
- Prefer Pinnacle (`PSH…`, `P>2.5`) or the market average (`Avg…`) over Bet365, because they have lower margins and are sharper. Fall back to Avg when Pinnacle is missing.
- **Goalscorer odds:** the free route (The Odds API, US books, ~500 credits) could cover 10 EPL events a gameweek at a few credits each, but this is unconfirmed. Credit cost per market per region was not shown. Anytime-scorer odds would be a strong input to the scorer predictions.
- Betfair's personal-use-only terms make it a poor fit for a public content series. It is also overkill when football-data already gives Betfair Exchange prices (`BFEH…`).

### Gaps
- I could not confirm when the season CSVs (e.g. `mmz4281/2627/E0.csv`) are updated, for example twice a week after results. The site gives no fixed schedule.
- I could not confirm the exact `fixtures.csv` column set (whether it includes Pinnacle, O/U and AH, and closing columns).
- I could not confirm the credit cost of The Odds API per call (it is commonly described as markets × regions, but this is not verified here). I also could not confirm whether its EPL scorer props have usable UK or Pinnacle coverage.
- I did not retrieve the terms of use for oddsportal or The Odds API. Oddsportal is widely understood to forbid scraping, but that is not verified here.
