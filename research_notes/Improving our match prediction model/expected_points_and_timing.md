# Expected-points-optimal picks (scoreline, scorers, goal minutes) for our scoring system

Scoring system under study (from `src/football_ml/config.py` and `evaluate/points.py`): exact score 3; otherwise correct result 1; +1 per predicted scorer matched (multiset, own goals excluded, no penalty for wrong picks; number of picks = goals in the predicted score); +1 per matched scorer whose predicted minute is within ±10 of the actual one, measured on the **elapsed** scale (`45+2'` → 47, `90+4'` → 94). Baseline backtest (90 matches): 141 pts = 7 exact, 45 results, 67 scorer hits, 15 minute hits.

Methods note: published work on this exact question is thin. Several findings below come from **my own calculations**: (a) a Dixon-Coles grid in Python (rho = -0.1, independent Poisson otherwise, scorelines 0-6), and (b) goal clocks extracted from the repo's cached ESPN data (`data/cache/espn`, 2026-27 season, 90 completed matches, 243 goals, the same matches as the backtest). Scripts: `/private/tmp/claude-501/-Users-dylantombs-football-ml/4bdcdcbe-f554-41e8-98b6-e4818b4faacf/scratchpad/ep.py` and `goaltimes.py`. These are labelled "own calculation" and have no external URL.

## 1. How do papers and prediction-league experts pick a scoreline to maximise expected points?

### Takeaway
The standard approach is to enumerate every candidate scoreline, compute Σ P(actual) × points(pick, actual) from the score matrix, and take the argmax. Under 3/1 scoring this rarely differs from "most likely score". It differs more under kicktipp-style schemes that give points for goal difference, where 1-0 / 2-1 beat a modal 0-0 or 1-1. I found no peer-reviewed paper that quantifies the gain across a season.

### Cited Findings
- A 2026 World Cup prediction engine built on Dixon-Coles, scored 3 for exact and 1 for result (the same outcome rule as ours), picks "the scoreline that maximises expected points under the game's scoring rules", not the most probable one. Worked example: 1-1 was the single most likely score at 15%, but 1-0 was picked, with EP 0.713 against 0.642 for 1-1. The author reports no tournament-wide gain from the optimiser. The final pick (1-0 Spain) was also the modal score. Live result: 78/104 correct results (75%). — [sakthiish.com blog, 14 Jul 2026](https://www.sakthiish.com/blog/2026-07-14-world-cup-prediction-engine)
- WM 2014 analysis from odds, scored kicktipp-style (rules not visible, site now offline): the best tip for a Germany match was 1-0 with expected score 1.28 points, while the most likely result, 0-0, averaged only 0.9. This comes from a search snippet only; the page could not be fetched (DNS failure). — [the-quants.com](http://the-quants.com/wm-tipp-spiele-die-richtige-strategie)
- Braband (TU Braunschweig, viXra preprint, Mar 2017) relates Elo, SPI, odds and "wisdom of crowds" tips to the Poisson/Skellam model for a 2,400-entrant World Cup Tipprunde. Model-based "system tips" beat human tippers on average and usually finished in the top 20%, but "realistically cannot win a competition with very many tippers". Differences between the models were small. It does not derive an EP-optimal scoreline rule. — [Braband 2017, viXra 1703.0041](https://vixra.org/pdf/1703.0041v1.pdf)
- Kicktipp practitioner guides (non-peer-reviewed): a pure home-win tipper is right on about 50% of outcomes, and Kicktipp itself says such a tipper is hard to beat in small groups. About a quarter of Bundesliga matches end drawn, so tipping too few draws is a common mistake. Kicktipp offers alternative point rules (a "9er" rule and odds-weighted rules). — [buli-tipphilfe.de](https://buli-tipphilfe.de/kicktipp-strategie); [buli-tipphilfe.de blog](https://buli-tipphilfe.de/blog/kicktipp-bessere-punkte-erzielen)
- For probabilistic tipping (as opposed to exact-score tipping), the Monash footy-tipping work shows that reporting your true probabilities maximises expected log score: the scoring rule is proper. — [Monash probabilistic footy tipping](https://probabilistic-footy.monash.edu/~footy/about.shtml); [Dowe et al. 1996 TR](https://users.monash.edu/~dld/Publications/1996/Dowe+Farr+Hurst+Lentin1996_MonashCS_tr-cs96-297-abs.html)
- Sky Super 6 (5 pts exact, 2 pts result): the experts' pages publish picks based on xG but no EP derivation. Popularity matters only for winning a shared jackpot, not for maximising expected points (e.g. 33.4% of players picked 1-2 in one round). — [Sporting Life Super 6 tips](https://www.sportinglife.com/football/news/super-6-correct-score-tips/189399); [Sky Sports Super 6 advice](https://skysports.com/football/news/12088/10225190/jeffs-super-6-advice)
- Pullein (Racing Post, Nov 2019) on the Super 6 tie-break: for a tie-break that only ranks you against others (the time of the first goal), the best pick also depends on what other entrants choose ("beauty contest"). This does not apply to our absolute-points game. — [Racing Post, Kevin Pullein, 21 Nov 2019](https://racingpost.com/sport/opinion/kevin-pullein/how-to-pick-the-right-first-goal-time-to-go-with-your-super-6-score-selections-a7pLM0j7FW4b)

### Inferences
- The method is not in doubt: brute-force argmax over a 7×7 (or 11×11) grid is the accepted approach and costs almost nothing. Most of the gains in the literature come from schemes that reward goal difference or close scores. Our 3/1 outcome rule has no such tier, so there is less to gain from the outcome part alone (see Q3).
- Our game differs from every published one because of the free scorer picks, so the existing literature does not cover the most important decision (see Q2).

### Gaps
- I found no paper (arXiv, journal or thesis) that measures EP-optimal against modal tipping over a season of kicktipp or Superbru. No Opisthokonta or pena.lt/y post on this specific topic appeared in my searches.
- the-quants.com's full method and scoring rule could not be retrieved.

## 2. Expected-points formula for THIS system and joint optimisation of goal count vs exact-score probability

### Takeaway
E[outcome pts | pick (a,b)] = P(result(a,b)) + 2·P(a,b). The scorer and minute terms only grow with the number of predicted goals, because each extra pick is a free lottery ticket. With no penalty for wrong scorers, the unconstrained optimum is degenerate: predict the largest score allowed (e.g. 6-5). That gives about +1.3 to +1.6 EP per match over the modal pick, despite an exact-score probability of about 0. The rule should be fixed, or the search constrained to plausible scorelines.

### Cited Findings
- Scoring rule as implemented: exact 3, result 1 (`POINTS_EXACT_SCORE`, `POINTS_CORRECT_RESULT`); +1 per scorer and +1 per minute (`POINTS_PER_SCORER`, `POINTS_PER_MINUTE`); `MINUTE_TOLERANCE = 10`, compared using `GoalTime.elapsed()`. — repo files `src/football_ml/config.py` lines 123–126 and `src/football_ml/evaluate/points.py` lines 55–72
- Accepted method: sum P(outcome) × points over all outcomes for each candidate and choose the max. — [sakthiish.com, Jul 2026](https://www.sakthiish.com/blog/2026-07-14-world-cup-prediction-engine)

### Inferences (own calculation)
- **Formula.** For pick (a,b) with result r and matrix P:
  - E_out(a,b) = 3·P(a,b) + 1·[P(r) − P(a,b)] = P(r) + 2·P(a,b).
  - E_scorer(a,b) ≈ Σ_{k=1..a} q_home(k) + Σ_{k=1..b} q_away(k). Here q(k) is the probability that the k-th pick is matched. For distinct players this is roughly P(player k scores ≥1) = 1 − exp(−λ·share_k). Picking the same player twice needs him to score twice, which is much less likely, so distinct picks dominate unless one player has a very large share.
  - E_minute ≈ Σ over picks of q(k) × h, where h = P(|actual minute − predicted minute| ≤ 10 given that he scores) ≈ 0.21 random, ≈ 0.28 at the best fixed minute (see Q4).
  - Total E = E_out + (1 + h)·Σ q(k). Every q(k) > 0, so total E rises strictly with each added goal on either side.
- **Degeneracy check.** The grid used shares [0.30, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05] and h = 0.25. The argmax over 0–6 per side is 6-5 (or 6-6 / 5-6) for every matchup tried:

  | λ, μ | modal pick | total EP (modal) | unconstrained argmax | total EP | outcome EP lost |
  |---|---|---|---|---|---|
  | 1.4, 1.1 | 1-0 | 1.06 | 6-5 | 2.61 | 0.63 → 0.43 |
  | 2.0, 0.8 | 2-0 | 1.78 | 6-5 | 3.05 | 0.89 → 0.65 |
  | 1.2, 1.2 | 0-1 | 0.92 | 6-6 | 2.49 | 0.54 → 0.30 |
  | 1.8, 1.4 | 2-1 | 1.89 | 6-5 | 3.18 | 0.65 → 0.46 |

  Even the 7th-best player's ≈ 8% scoring chance is worth more than the ≈ 0.02 to 0.06 of outcome EP given up per extra goal. This is a property of the scoring rules, not a modelling insight. Predicting 6-5 would score well but look absurd in the reels.
- **Constrained search (a practical middle ground).** Keep only scorelines with P(a,b) ≥ threshold, then argmax total EP:
  - Threshold 6%: 1-0 → 2-1 (λ 1.4/μ 1.1, total EP 1.06 → 1.62); 2-0 → 3-1 (1.78 → 2.20); 0-1 → 1-2 (0.92 → 1.47); 2-1 stays 2-1. Outcome EP drops by only 0.03 to 0.11.
  - Threshold 4%: picks move to 3-1 or 2-2 (e.g. 1-0 → 3-1, total EP 1.72; 0-1 → 2-2, 1.56).
  - Threshold 2%: 3-2, 4-1, 2-2 (e.g. 1-0 → 3-2, total EP 1.87).
  - These EP gains (+0.4 to +0.8 per match) exceed anything available from the outcome term. They depend on the assumed player shares, so treat them as illustrations.
- **Better long-term fixes** (rule design, Dylan's call): (i) charge a penalty per wrong scorer pick (e.g. −0.5), so a pick is only made when q(k)·(1+h) > penalty·(1−q(k)); (ii) score scorers by precision, i.e. hits ÷ picks; (iii) fix the number of scorer picks (e.g. always 3) regardless of the predicted score. Any of these removes the degeneracy and makes "number of goals" a calibration question again.
- The model's under-prediction (188 predicted vs 243 actual goals) means the current picks are below even a calibrated goal count. Constrained EP optimisation will raise predicted goals mechanically.

### Gaps
- q(k) per player should come from the model's own scorer probabilities (xG share × team λ). I used a stylised share vector, not the repo's actual scorer model.
- Correlation between the scoreline and the scorers (in a match that ends 3-1, each player is more likely to have scored) is ignored. Conditioning on the predicted score would change the numbers but not the direction.

## 3. Typical modal vs EP-optimal picks; how often to predict draws

### Takeaway
On the outcome part of our 3/1 system, "likeliest result, then likeliest score within it" is already almost EP-optimal. Over a grid of 315 matchups, true argmax picks differ in only 17 cases and add about 0.2% of outcome EP. The differences are almost all extra 1-1 (and 0-0) draw picks in evenly matched games: draw picks rose from 8 to 24 of 315.

### Cited Findings
- About a quarter of Bundesliga matches are draws, and tipsters are warned against under-tipping them. — [buli-tipphilfe.de](https://buli-tipphilfe.de/kicktipp-strategie)
- Dixon-Coles raises low draws relative to plain Poisson. Worked example (both teams 1.3 xG): 1-1 at about 14.0% under DC vs 12.4% under Poisson; 1-0 drops from 9.5% to 7.9%. — [statsultra.com, Dixon-Coles explainer](https://statsultra.com/dixon-coles-model/)
- Under kicktipp-like scoring (points for tendency/difference), 1-0 beat the modal 0-0 (1.28 vs 0.9 EP). Under 3/1 scoring, 1-0 beat a modal 1-1 (0.713 vs 0.642) in a World Cup case. — [the-quants.com via search snippet](http://the-quants.com/wm-tipp-spiele-die-richtige-strategie); [sakthiish.com, 2026](https://www.sakthiish.com/blog/2026-07-14-world-cup-prediction-engine)

### Inferences (own calculation, DC rho = −0.1, λ 0.8–2.8, μ 0.6–2.0)
- Because E_out = P(r) + 2P(a,b), a draw pick wins when 2·P(1-1) + P(draw) > 2·P(best win score) + P(win). This happens when P(draw) is close to P(home) and P(away) and 1-1 is the single most likely cell:
  - λ = μ = 1.3: H/D/A 0.355/0.289/0.355. The baseline picks 1-0 (EP 0.523), but 1-1 has EP 0.565 (+0.042).
  - λ = 1.5, μ = 1.2: home is likeliest (0.429), and the optimal pick is 2-1 (0.611) rather than 1-1 (0.545). The baseline already gets this right.
- Typical optimal outcome picks: 1-0 / 2-1 for mild favourites (2-1 once total goals ≳ 3), 2-0 for strong favourites (λ ≳ 2), and 1-1 for near-even games. Draw picks come out at roughly 5–10% of fixtures under pure outcome EP, well below the ~23–27% real draw rate. That is correct: you predict draws only when they are the best bet, not as often as they happen.
- The sakthiish result (1-0 > modal 1-1) and our grid (1-1 > 1-0 in even games) do not conflict. Both follow from P(r) + 2P(a,b), and which pick wins depends on how much of the result probability the win side holds.
- Once scorer EP is added (Q2), draws become even rarer and higher-scoring (2-2 instead of 1-1) because each extra goal adds scorer EP.

### Gaps
- Grid results assume an independent-Poisson DC matrix. The repo's actual λ/μ distribution over PL fixtures would give the true share of differing picks.

## 4. Goal timing: distribution and best minute choice under ±10 tolerance

### Takeaway
Goals are roughly uniform in the first half and rise through the second, with a large 90+ spike that has grown since 2023-24 (11–13% of goals at 90'+). On our elapsed scale, which folds first-half stoppage into minutes 46–50, the densest ±10 window is centred on about minute 50–58. That window catches about 28% of goals, against about 21% for a random draw from the distribution (our current method). A deterministic pick therefore beats random sampling. For two picks of the same player, choose non-overlapping windows. For picks of **different** players, each should use the densest minute, since their hits don't compete.

### Cited Findings
- World Cup goals 1930–2014 (regulation time only): the first-half rate is roughly constant; the second-half rate rises linearly (goals/minute = 13.346 + 0.0708·minute, p = 0.0037, R² 0.18); about 44% of goals come in the first half. — [Mingotti, "Fooled by bursts", arXiv 1408.5442 (2014)](https://ar5iv.labs.arxiv.org/html/1408.5442)
- Multi-league studies find more second-half goals, concentrated in the last 15 minutes. Euro 2012: 57.9% of goals in the second half, most in the final 15 minutes. — [arXiv 1408.5442](https://arxiv.org/pdf/1408.5442); [Euro 2012 analysis (sport-iat)](https://lida.sport-iat.de/ta/Record/4039069?lng=en)
- Dixon & Robinson (1998), "A Birth Process Model for Association Football Matches", *The Statistician* 47(3):523–538, models time-varying scoring intensity within a match (citation confirmed; content not retrieved). — [Imperial reading list PDF](https://www.ma.imperial.ac.uk/~ejm/M3S4/Problems/football.pdf)
- Premier League through Matchweek 24 of each season: 90+ goals were 73/767 (9.5%) in 2023/24, 68/729 (9.3%) in 2024/25 and 76/666 (11.4%) in 2025/26. Average match time was 101:43, 99:56 and 100:20. — [Premier League / Opta, 3 Feb 2026](https://www.premierleague.com/en/news/4580269/why-more-late-goals-are-being-scored-in-2025-26-premier-league-season)
- 2023/24 full season: a then-record 12.8% of goals in stoppage time, after the IFAB added-time crackdown made matches about 3m13s longer than in 2022/23. Counts across PL articles differ (87 vs 93 stoppage goals), so the exact figure is uncertain. 2025-26 is reported as a new high of 13.2% of goals at 90'+. — [Premier League news](https://www.premierleague.com/news/3978045); [Premier League / Opta late goals](https://www.premierleague.com/en/news/4437338/opta-analysis-of-why-more-late-goals-are-being-scored-in-2025-26-premier-league-season); [The Analyst, time-keeping measures](https://theanalyst.com/articles/football-time-keeping-and-time-wasting-measures)
- Repo timing method: stratified quantile sampling from this season's real goal times, seeded by fixture id; minutes are compared on the elapsed scale. — `src/football_ml/predict/timing.py`, `src/football_ml/evaluate/points.py`

### Inferences (own calculation on the 243 goals in the 90 backtest matches)
- Shape: first-half share 41.6%. By base clock in 15-minute bins: 35 / 25 / 41 / 52 / 39 / 51. There were 3 first-half stoppage goals and 15 at 90'+ (6.2%, below the PL's 11–13%, so the small sample is noisy).
- P(hit) for a single predicted minute m = share of goals with |elapsed − m| ≤ 10:
  - m = 10: 0.17; 20: 0.16; 30: 0.18; 45: 0.25; **50: 0.28; 55: 0.28; 58: 0.29 (best)**; 60: 0.27; 70: 0.21; 80: 0.22; 85: 0.19; 90: 0.18; 105: 0.03.
  - A random draw from the distribution (the current method) averages **0.213**. The backtest's 15 minute hits ÷ 67 scorer hits = 22.4%, which matches this closely.
  - The window around 50–58 is boosted because first-half stoppage (45+x → 46–50) falls next to early second-half goals on the elapsed scale. Late goals are spread over 85–105 because added time varies, so 90' is a worse centre than intuition suggests.
- Several picks of the **same player** need non-overlapping windows, since the matcher pairs each actual goal with at most one prediction. Best union coverage (in-sample): two windows at 39 & 61 → 51%; three at 39, 61, 83 → 72%; four at 13, 39, 61, 83 → 90%. These are spaced about 21–22 minutes apart, i.e. tiling ±10 windows edge to edge.
- Picks of **different players**: each scorer's minute is judged only against that player's own goals, so the hits don't compete. Under a time-homogeneous player model, every pick should use the single best minute (≈ 55). Spreading them only makes the output look more natural, at a small EP cost (≈ 0.28 → ≈ 0.21 per pick).
- A deterministic choice is better than random: for a fixed density, the argmax window always beats or equals the average window. Expected gain is about +0.07 minute points per matched scorer, i.e. 67 × (0.28 − 0.213) ≈ **+4.5 points over 90 matches (~3% of 141)**. This is an in-sample estimate. To estimate it out of sample, fit the minute on prior seasons (or a smoothed parametric curve such as Mingotti's: flat first half, linear rise, plus stoppage mass).
- Refinements worth testing: the expected stoppage length per match (busy games get more added time) and score-state effects (late goals by trailing favourites). Both are second-order next to the fixed-minute gain.

### Gaps
- I found no published minute-by-minute PL goal distribution for 2023-24 onward. The PL/Opta pieces give only the 90+ shares.
- The Dixon & Robinson (1998) quantitative intensity estimates were not retrieved.
- The 243-goal sample is small. The best-minute location (50 vs 58) is within noise, and the plateau from 45 to 60 is the robust finding.

## 5. How much does expected-points optimisation add in practice?

### Takeaway
External evidence is anecdotal: single-match EP gains of ~0.07 to 0.4 points under 3/1 and kicktipp rules, and no season-level measurement. For our system, the outcome-pick gain is negligible (~0.2%), deterministic minutes are worth about +3%, and the scorer-count lever is large (tens of percent) but only because of the no-penalty rule.

### Cited Findings
- Single-match examples: 0.713 vs 0.642 EP (≈ +11%) for 1-0 over a modal 1-1 under 3/1 scoring — [sakthiish.com, Jul 2026](https://www.sakthiish.com/blog/2026-07-14-world-cup-prediction-engine). 1.28 vs 0.9 EP (≈ +42%) for 1-0 over a modal 0-0 under a kicktipp-style rule — [the-quants.com (snippet)](http://the-quants.com/wm-tipp-spiele-die-richtige-strategie)
- System tips beat the average human but rarely win big pools, and model-to-model differences were small (World Cup 2014, 2,400 entrants). — [Braband 2017](https://vixra.org/pdf/1703.0041v1.pdf)
- A practitioner reports that adding squad-quality features cut backtest points from 96 to 86, a reminder to judge changes by backtest points. — [sakthiish.com, Jul 2026](https://www.sakthiish.com/blog/2026-07-14-world-cup-prediction-engine)

### Inferences (own calculations; to be checked by backtest per CLAUDE.md rules)
- Outcome argmax vs current headline rule: +0.2% of outcome EP on the grid. That is roughly 0 to 1 points over 90 matches, so not worth prioritising except as a near-free tidy-up that mainly adds 1-1 picks in even games.
- Fixed best minute instead of random sampling: about +4.5 points over 90 matches (~+3%).
- Raising predicted goals toward the real rate (188 → ~243): about 55 extra scorer picks. At a marginal match probability of ~0.12–0.20 (3rd–5th choice scorers), that is about 7–11 extra scorer hits plus 2–3 minute hits, against an outcome-EP loss of about 0.02 to 0.06 per added goal. Net roughly **+5 to +12 points (4–8%)**.
- A constrained EP search with P(a,b) ≥ 4–6% shows +0.4 to +0.8 EP per match in the stylised grid (potentially +30% or more). That overstates things if real q(k) for depth players is low or if scorer shares are more concentrated. Run the backtest before trusting it.
- Unconstrained, the rules reward predicting 6-5 every match. Any public content series should either fix the scoring rule (penalty per wrong scorer, or a fixed number of scorer picks) or document a plausibility constraint, so the points aren't gaming the scoring.

### Gaps
- No season-level, out-of-sample study of EP-optimal vs modal tipping was found for any prediction game.
- The scorer-hit gain depends on the repo's actual per-player probabilities, which I did not run. A backtest with a `--pick-strategy` switch is the way to measure it.
