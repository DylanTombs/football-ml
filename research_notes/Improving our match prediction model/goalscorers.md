# Anytime-goalscorer probabilities and line-up prediction (Premier League)

Context: the pipeline currently picks N scorers per team (N = predicted team goals) greedily by FPL season-to-date 0.7·xG + 0.3·goals, with a decay for players already picked. Backtest: 67 hits from 188 predicted goals over 90 matches, which is **35.6% per pick**.

## 1. Standard anytime-scorer model structure (share × team λ × minutes), shrinkage, recency

### Takeaway
I found no published primary write-up from Opisthokonta, FPL Review, FFHub or Solio that spells out an anytime-scorer formula. The usual building blocks (Poisson team goals, a fixture-specific rate, xG as a player-rate estimate) are well supported, though. The evidence on xG stability is strong: use roughly 20–40 matches of xG rather than goals, and shrink toward a positional or team prior, because about 40% of a full-season xG/90 edge regresses away (60% if the player changes team).

### Cited Findings
- Opisthokonta's public work is match-level team-goal modelling: the R package `goalmodel` offers Poisson, negative binomial and Conway-Maxwell-Poisson models with an optional Dixon-Coles adjustment. It is not a player-scorer model. — [opisthokonta/goalmodel](https://github.com/opisthokonta/goalmodel); [R-bloggers BTTS post](https://www.r-bloggers.com/2022/02/calculating-the-probability-for-both-teams-to-score-in-r/)
- Team goals are commonly modelled as Poisson counts, and a fixture-specific rate (e.g. from expected goals) gives more realistic probabilities than a league-average rate. — [arXiv 2105.09881, Poisson modelling of EPL goals](https://arxiv.org/pdf/2105.09881)
- Scoring rate is not constant within a match: goals become more likely as the match goes on, and there are fewer than expected early in each half. So a sub who plays the last 30 minutes gets slightly more than 30/90 of the scoring opportunity. — [arXiv 2501.18606, Temporal dynamics of goal scoring](https://arxiv.org/html/2501.18606v1)
- Betting guides rank minutes as the first factor for anytime goalscorer, ahead of role, set-piece duty and matchup. A rotation-risk winger can be worse than a nailed-on central striker who takes penalties. — [Football Dream AGS guide](https://www.footballdream.in/blog/anytime-goalscorer-prediction-guide) (low-authority source)
- **FPL Review simulation** (true rate 0.30 G/90, Poisson shots, Understat 2018/19 chance quality). Error of xG/90 as an estimator by sample size: 1 match MAE 0.255; 5 matches MAE 0.116 (38% of readings within ±25%); 10 matches MAE 0.083 (54%); 20 matches MAE 0.057 (70%); 40 matches MAE 0.041 (85%); 100 matches MAE 0.026 (98%). — [FPL Review, xG accuracy](https://docs.fplreview.com/articles/xg-accuracy/)
- Same source: 40 matches of xG gives about the same confidence as 100 matches of goals, and 5 matches of goals gave 0% of readings within ±25%. Low-rate players converge more slowly: at 40 matches, 28% of 0.10-rate players were within ±10%, against 59% of 0.60-rate players. Under 10 matches is "highly distorted"; 20–40 is recommended; 100 risks stale data. The last 5 matches say more about role than about xG rate. — [FPL Review, xG accuracy](https://docs.fplreview.com/articles/xg-accuracy/)
- Year-to-year persistence of xG/90 over 30+ match windows is about 0.6: a forward 0.1 above average keeps about 0.06 next period, and about 0.04 if he changes team. Over 5–10 match windows the relationship is weak. — [Evidently, "How can we tell if a footballer will consistently score"](https://evidently.substack.com/p/how-can-we-tell-if-a-footballer-will)
- Finishing skill (goals/xG) has persistence of only about 0.1 even over 60+ shots. A goals/xG ratio of 2 regresses to about 1.1, and the average ratio is slightly below 1. — [Evidently](https://evidently.substack.com/p/how-can-we-tell-if-a-footballer-will)
- Sources disagree on the minimum minutes before per-90 stats mean anything: 270 (Franks-style replication), about 900, or 1000+. — [R-bloggers Meta-Analytics for Soccer](https://www.r-bloggers.com/2023/09/meta-analytics-for-soccer/); [Nutmeg guide](https://skills.sh/withqwerty/nutmeg/learn); [FootballGPT](https://www.footballgpt.co/scout/data-metrics-xg)
- Academic player-ability work treats finishing as a player random effect on top of xG (Bayesian hierarchical models, "Shooting Prowess"). This supports small finishing adjustments shrunk heavily toward zero. — [arXiv 1710.00001](https://arxiv.org/pdf/1710.00001); [arXiv 1702.05662](https://arxiv.org/pdf/1702.05662); [arXiv 2511.23072](https://arxiv.org/html/2511.23072v1)
- OpenFPL (an XGBoost + Random Forest ensemble per position) uses player, team and opponent features averaged over the last 1, 3, 5, 10 and 38 matches. That is a multi-window recency design rather than a single season-to-date figure. — [OpenFPL, arXiv 2508.09992](https://arxiv.org/html/2508.09992v1)

### Inferences
- A defensible model is P(player i scores) = 1 − exp(−μ_i), with μ_i = λ_team × share_i × m_i. Here λ_team is the match model's expected goals for the team. share_i is the player's shrunk share of team non-penalty xG per 90 he is on the pitch, plus his share of penalty xG if he is first on pens. m_i is expected minutes/90, perhaps nudged up slightly for late-game minutes. Each team's shares should be renormalised so that Σ share_i × m_i ≈ 1 over the expected XI plus subs, with a small own-goal and "other" bucket of about 3–4%. This is my synthesis; no source above publishes this exact form.
- Shrinkage: estimate shrunk_npxG90 = (npxG + k·prior90·mins/90) / (mins/90 + k), with the prior set by position (or position × team attack level) and k ≈ 10–20 matches' worth, matching the FPL Review finding that roughly 20 matches are needed before the sample dominates. For players who changed team, use a stronger prior (persistence 0.4 vs 0.6). k should be tuned by backtest.
- Using goals at 0.3 weight adds mostly noise, given finishing persistence of about 0.1. Use npxG (pens separately), and keep only a heavily shrunk goals − xG term if the backtest supports it.
- Recency: blend season-long shrunk npxG/90 (the rate) with last 3–5 match minutes and xG share (the role and position-in-team signal), along the lines of OpenFPL's multi-window features and FPL Review's "last 5 = role" remark.
- Sanity check on the current numbers: 35.6% hits per pick is about the anytime probability of a typical starting forward. A better model mainly helps by not wasting picks on rotated or benched players and by giving pen takers their due.

### Gaps
- I did not find primary methodology from FPL Review's xMins/AGS model, Fantasy Football Hub, Solio or Opisthokonta for player-level scorer probabilities. FPL Review's docs say only that its Massive Data Model uses history, market odds and xMins ([FPL Review MDM](https://docs.fplreview.com/the-model/projections/massive-data-model/)).
- I found no published calibration or Brier score for any public anytime-scorer model.

## 2. Penalty takers and set pieces

### Takeaway
A penalty is worth about 0.76–0.79 xG depending on provider, and recent PL conversion has run 80–90%. The first-choice taker on a team that wins about 0.1–0.15 penalties per match gets a meaningful additive boost to μ (about +0.08–0.12).

### Cited Findings
- Penalty xG by provider: Opta 0.79, StatsBomb 0.78 (since a 2022 update), Wyscout 0.76. This comes from a secondary blog, not primary StatsBomb docs. — [jobsinfootball](https://jobsinfootball.com/blog/what-is-the-xg-of-a-penalty/); 0.79 also from [Sportmonks](https://sportmonks.com/blogs/xg-explained)
- PL penalty conversion hit a record 89.6% in 2023-24. The previous best was 83.9% (2013-14), and the low was 65.8% (2001-02). 2022-23 was low, which Opta calls an anomaly. — [The Analyst (Opta)](https://theanalyst.com/articles/premier-league-penalties-like-free-goal); [premierleague.com](https://www.premierleague.com/news/3342212)
- Analysts usually strip penalties out of player/team xG (npxG) and handle them separately because of their fixed value. — [Hudl xG explainer](https://www.hudl.com/blog/expected-goals-xg-explained)

### Inferences
- Use μ_i = λ_np × npshare_i × m_i + 1[i is first on pens] × P(pen in match) × 0.78 × P(i on pitch when awarded). λ_np is team expected non-penalty goals, and P(pen in match) is team-level, estimated from pens won per match and shrunk toward the league rate. FPL `penalties_order == 1` identifies the taker. If the taker is out, pass the boost to order 2.
- Direct free kicks add little (direct FK conversion is low, typically single-digit %). `direct_freekicks_order` is at most a tiebreaker. (I found no cited figure for this; treat it as an assumption.)

### Gaps
- I did not find a sourced PL figure for penalties awarded per team-match in 2025-26, or for direct free-kick conversion. Both can be computed from FPL/ESPN data.

## 3. Predicting minutes and starts

### Takeaway
Public models do not predict minutes well from FPL data alone. OpenFPL explicitly lacks start information and is worst on the "Zeros" category. Human predicted line-ups claim about 85–96% of starters correct, all self-reported. Practical approach: start probability from recent starts (last 3–5 GWs) × availability (`chance_of_playing_next_round`), plus a small extra rotation penalty after midweek European games.

### Cited Findings
- OpenFPL has no minutes model. It uses FPL availability tags (0/25/50/75/100%) and historical minutes, and notes the FPL API does not say whether a player will start, come off the bench or be rested. That is why it trails FPL Review on Zeros (RMSE 0.818 vs 0.689) and Blanks (1.291 vs 1.189). — [OpenFPL, arXiv 2508.09992](https://arxiv.org/html/2508.09992v1)
- FPL Review's premium model uses xMins, market odds and news feeds updated hourly. The site started out on bookmaker-based projections with editable expected minutes. — [FPL Review MDM](https://docs.fplreview.com/the-model/projections/massive-data-model/); [About FPL Review](https://docs.fplreview.com/getting-started/about-fplreview/)
- Leading commercial services rely on expert minutes projections and bookmaker feeds. — [OpenFPL](https://arxiv.org/pdf/2508.09992)
- Predicted-line-up accuracy (self-reported vendor figures):
  - Flashscore 2025/26: about 1.67 wrong players per predicted XI across the top-5 leagues and UCL, i.e. about 85% correct. — [CMOtech UK](https://cmotech.uk/story/flashscore-claims-top-accuracy-in-predicted-line-ups)
  - FPL Focal: about 87–88% in 2025/26. — [fpl.page GW23](https://fpl.page/article/fpl-gw23-predicted-lineups-team-news-2526)
  - Premier Insider: 96% (early-season claim). — [LigaInsider/Premier Insider](https://www.ligainsider.de/mobile/amp/premierinsider_5990/click-crests-for-gw6-predicted-lineups-95442/)
  - AllAboutFPL 2022: weekly results from 7/11 to 11/11. — [AllAboutFPL](https://allaboutfpl.com/2022/08/predicted-gw5-lineups-of-all-20-premier-league-teams/)
  - A search snippet reported that in one GW, Fantasy Football Scout got 9/11 Crystal Palace starters right while rivals averaged 7.33/11. I could not trace the primary page, so treat it as unverified.
- Accuracy improves as kickoff approaches. — [Football Dream](https://www.footballdream.in/lineups)

### Inferences
- Baseline P(start) = (starts in last 5 GWs the player was available for) / 5, shrunk toward the season start rate, × cop/100 when `chance_of_playing_next_round` is not null. If `news` mentions injury or suspension and cop is 0, set P(start) = 0. Expected minutes ≈ P(start)·~80 + P(sub)·~20. Calibrate the 80 and 20 from FPL history.
- A naive "same XI as last match" baseline probably scores around 8.5–9/11 for PL starters (inferred from the vendor numbers above). Since even vendors are near 85–90%, the realistic gain is in zeroing out injured or suspended players and down-weighting rotation-prone ones.
- European congestion: down-weight starts for clubs that played a UCL/UEL/UECL match within about 3 days. I found no quantified effect size, so tune it in backtest.

### Gaps
- I found no independent audit of line-up prediction accuracy and no published effect size for European-fixture rotation in the PL.

## 4. Distinct players vs repeat picks (multiset matching)

### Takeaway
With +1 per matched pick and Poisson goals, the k-th pick of the same player adds exactly P(X_i ≥ k). Greedy selection on marginal gain is therefore optimal. A repeat is worth it only when P(top player scores ≥2) > P(next-best player scores ≥1), which is rare: it needs the top player's μ to be well above the next player's (e.g. μ ≈ 0.875 → P(≥2) = 0.22, beating only a next player with μ < 0.25, i.e. P(≥1) < 0.22).

### Cited Findings
- No external source on this; it is pure probability (derivation below). The Poisson player-goal assumption follows from team goals being Poisson — [arXiv 2105.09881](https://arxiv.org/pdf/2105.09881) — split by shares: thinning a Poisson process gives Poisson per player.

### Inferences
- Expected hits for a team = Σ_i Σ_{k=1..c_i} P(X_i ≥ k), where c_i is the number of picks for player i. P(X ≥ k) is decreasing in k, so the objective is separable concave and greedily taking the largest remaining marginal gain is exactly optimal. The current "decay for already picked" heuristic should be replaced with the exact P(X ≥ k).
- Formulas: P(≥1) = 1 − e^{−μ}; P(≥2) = 1 − e^{−μ}(1+μ).
- Worked numbers:
  - μ = 0.5: P(≥1) = 0.393, P(≥2) = 0.090
  - μ = 0.875 (e.g. 35% share of λ = 2.5, full 90): P(≥1) = 0.583, P(≥2) = 0.218
  - μ = 1.2: P(≥1) = 0.699, P(≥2) = 0.337
- Repeat if P_top(≥2) > P_next(≥1), i.e. μ_next < −ln(1 − P_top(≥2)). For μ_top = 0.875 that means μ_next < 0.246. For μ_top = 1.2, μ_next < 0.411.
- In practice repeats only happen for very dominant players (Haaland-type share on a high-λ team) when N is large (3+), where the 3rd–4th distinct options are low-μ midfielders or defenders. Otherwise pick distinct players.
- Picks for team T are marked against actual goals, and actual team goals are independent of our N. So use unconditional μ; there is no need to condition on the team scoring N.

### Gaps
- None needing external sources. The overdispersion of real goals (negative binomial) would raise P(≥2) slightly, but that was not quantified.

## 5. Bookmaker anytime-scorer odds: accuracy and free availability

### Takeaway
I found no academic calibration study of football anytime-scorer odds. Related work finds bookmaker probabilities broadly well calibrated, with favourite-longshot bias in niche markets such as exact scores. The Odds API does offer `player_goal_scorer_anytime` for EPL, but only from US bookmakers, one event per request.

### Cited Findings
- The Odds API soccer player-prop keys include `player_goal_scorer_anytime`, `player_first_goal_scorer`, `player_last_goal_scorer`, shots, SoT and assists for EPL, the other big-5 leagues and MLS. Coverage is "currently limited to US bookmakers", fetched via `/events/{eventId}/odds`. — [The Odds API betting markets](https://the-odds-api.com/sports-odds-data/betting-markets.html)
- The Odds API historical odds go back to mid-2020, but only for featured markets (h2h/totals). Historical player props were not stated, and free-tier credit numbers were not stated on the page. — [The Odds API EPL](https://the-odds-api.com/sports-odds-data/epl-odds.html)
- Alternative paid APIs with anytime-goalscorer keys: TheStatsAPI (`anytime_goalscorer`) and SportsGameOdds. — [TheStatsAPI player odds](https://www.thestatsapi.com/docs/api-reference/matches/player-odds.md); [SportsGameOdds EPL](https://sportsgameodds.com/blog/premier-league-betting-markets-api)
- Favourite-longshot bias is significant in exact-scoreline markets but generally absent in match-result markets. — [University of Reading football research](https://research.reading.ac.uk/football-research/?p=664)
- UFC odds-implied probabilities are mostly well calibrated, with a specific systematic bias. — [UCSAS 2021 poster, Stanhope](https://statds.org/events/ucsas2021/posters/ucsas2021_15_Stephen_Stanhope.pdf)
- Selecting models by calibration rather than accuracy yields better betting ROI (+34.69% vs −35.17%, NBA). This argues for checking calibration of our P(score). — [Walsh & Joshi, Bath](https://researchportal.bath.ac.uk/en/publications/machine-learning-for-sports-betting-should-model-selection-be-bas/)
- FPL Review's real-world xG check used "implied-odds goals" as a proxy for true potential, i.e. analysts treat bookmaker AGS prices as a near-truth benchmark. — [FPL Review xG accuracy](https://docs.fplreview.com/articles/xg-accuracy/)

### Inferences
- AGS prices carry a large overround (often 20–30%+ across a match's player list; this is not cited, so treat it as an assumption). If used, de-vig by normalising implied μ = −ln(1 − p) so that each team's sum matches the match-total market.
- Odds cannot be backtested for free, because historical props are unavailable. So odds can only be used going forward, as a feature or benchmark, and they break the "tests never hit network" setup only at fetch time.

### Gaps
- I found no published Brier or log-loss for football AGS markets and no free historical AGS odds source. The Odds API free-tier quota was not confirmed.

## 6. Avoiding leakage in backtests

### Takeaway
Season-to-date fields in `bootstrap-static` (xG, minutes, starts, penalties_order, chance_of_playing, news) are snapshots of *now* and leak future information into any past-GW backtest. Rebuild each player's pre-GW state by summing `element-summary/{id}` `history` rows with round < GW (or kickoff_time < the fixture's kickoff).

### Cited Findings
- OpenFPL avoided leakage by collecting evaluation data only after development was done, using data available the day before each GW deadline, and splitting CV folds by team. — [OpenFPL](https://arxiv.org/html/2508.09992v1)
- The vaastav FPL dataset warns ML users about lookahead bias in its xP column (sourced from the bootstrap `ep_this` field). Its per-GW merged file contains GW-by-GW player stats, including xG/xA for recent seasons. — [vaastav repo docs index](https://docsearch.algolia.com/mcp/docs/repo/vaastav/fantasy-premier-league); [Apify FPL archive description](https://apify.com/constructive_calm/fpl-intelligence/examples/fpl-historical-per-gameweek-archive.md)

### Inferences
- From `element-summary/{id}.history` (per fixture: round, kickoff_time, minutes, starts, expected_goals, goals_scored, etc.), cumulative sums with a strict `< current GW` filter give leak-free season-to-date xG, minutes, starts and last-5 starts. Double GWs need the kickoff_time filter, not round.
- `penalties_order`, `chance_of_playing_next_round` and `news` cannot be reconstructed historically from the live API. In backtests, either (a) infer the pen taker from history (who took the previous pens: FPL `penalties_missed` plus ESPN penalty-goal events), and treat availability as unknown (P(start) from recent starts only), or (b) use vaastav per-GW snapshots if they include those fields (unverified). The backtest will understate the live model's benefit from cop/news. Note this when comparing to the plan-01 baseline.
- Early GWs (1–5) have tiny samples, so lean on the prior and last season's per-90 (persistence 0.6, or 0.4 for transfers).

### Gaps
- I could not verify which vaastav seasons/columns include `chance_of_playing` or `penalties_order` snapshots, or the exact element-summary field names, against primary docs. Check the repo or a live API response.

## 7. OpenFPL and similar open-source models: scorer accuracy

### Takeaway
OpenFPL (MIT, GitHub daniegr/OpenFPL) predicts total FPL points, not goals, and reports no goalscorer accuracy. It matches FPL Review overall and beats it for high returns (e.g. FWD "Tickers" RMSE 2.694 vs 3.822), but loses on zero-minute predictions because it has no start information. I found no open-source model reporting anytime-scorer hit rates.

### Cited Findings
- Design: XGBoost + RF ensembles per position (GK/DEF/MID/FWD/AM), 50 models per position, median forecast, 122–206 features, trained on FPL + Understat 2020-21 to 2023-24 and tested prospectively on 2024-25 (GW32–38). — [OpenFPL](https://arxiv.org/html/2508.09992v1)
- 1-GW RMSE vs FPL Review: Zeros 0.818 vs 0.689; Blanks 1.291 vs 1.189; Tickers 1.517 vs 1.594; Haulers 5.142 vs 5.172. A 5–34% RMSE reduction over a last-5 baseline. — [OpenFPL](https://arxiv.org/html/2508.09992v1)
- Code: https://github.com/daniegr/OpenFPL (MIT). — [OpenFPL](https://arxiv.org/pdf/2508.09992)
- A related paper (Ramezani & Dinh 2025) does squad selection; ARIMA with a rolling window was most consistent on 2023/24. It covers no scorer prediction. — [arXiv 2505.02170](https://arxiv.org/html/2505.02170v2)

### Inferences
- OpenFPL's Understat-based rolling features (1/3/5/10/38 matches) are a cheap template for our per-player features. Its weakness on minutes confirms that minutes/availability is the main lever where FPL-only data falls short.

### Gaps
- I found no public benchmark of anytime-scorer hit rate or Brier score for any open model, so our 35.6% per-pick rate cannot be compared with published numbers.
