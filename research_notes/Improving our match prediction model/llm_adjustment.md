# LLMs and news/qualitative information as an adjustment layer on a Dixon-Coles model

Research date: November 2026. Most LLM-sports evidence comes from arXiv preprints posted July–August 2026 around the 2026 FIFA World Cup. I read only their abstracts or abstract pages (one full PDF could not be parsed), so detailed metrics (Brier values, exact-score hit rates) are not confirmed here. None of these preprints is peer-reviewed yet.

## 1. Published evidence (2023–2026) on LLMs forecasting football/sports, and comparison with bookmakers

### Takeaway
Every 2026 prospective benchmark found that frontier LLMs with web search do **not** beat the betting market. At best they match "back the bookmaker favourite", they rarely disagree with each other, and they under-predict draws and goals. Web access gives only a small improvement. No 2023–2026 study I found shows an LLM-adjusted statistical football model beating its unadjusted baseline.

### Cited Findings
- **WorldCup Arena** (arXiv 2608.04008, 4 Aug 2026; Wang, Bian, Li, Qi). Six frontier LLMs ran with extended thinking and native web search and predicted all 104 World Cup 2026 matches before kickoff (4,494 scored predictions, "leakage-free by construction"):
  - Mean match-outcome accuracy was 63.9%, the same as always backing the bookmaker's favourite, "which the models mostly did".
  - The models agreed with each other far more often than they were right, so a majority vote added nothing.
  - Accuracy depended on how lopsided the fixture was, not on how much information was available. It fell in the closest matches, which had the richest dossiers.
  - The models **under-commit to draws and to goals**, and their scoreline picks cluster on one prototypical result.
  - Source: [arXiv 2608.04008](https://arxiv.org/abs/2608.04008)
- **WC2026-Agents** (arXiv 2607.17765, 20 Jul 2026; Ding, Guo, Xu). Claude Opus 4.8, GPT-5.5 (high reasoning), Gemini 3.1 Pro and Grok each ran a search-act-reflect loop on all 104 matches and gave 1X2 probabilities plus a virtual bet:
  - None of the four beat the market's Brier score.
  - A flat stake on the market favourite earned more than every agent. Agent ROI ranged from -18% to +10%, and fading the market lost money for all four.
  - The agents gave the same top pick in 92% of matches.
  - The share of forecasts citing the market ranged from 12% to 100%.
  - Self-reported error rates on wrong picks ranged from 36% to 86%, so self-knowledge varies.
  - Source: [arXiv 2607.17765](https://arxiv.org/abs/2607.17765)
- **LLM-SoccerArena** (arXiv 2607.24573, Jul 2026; Schröder, Schweisthal, Müller, Weinmann, Feuerriegel). Seven LLMs (including GPT-5.5 and Claude Opus 4.8) forecast all 104 World Cup matches in a factorial design that varied model, information access, prompting strategy and horizon. Models with web access beat those without "by a small margin": a Brier improvement of 0.023. Source: [arXiv 2607.24573](https://arxiv.org/abs/2607.24573)
- **AI World Cup 2026** (arXiv 2608.03416). GPT-5.5 Thinking ranked first overall and Claude Sonnet 4.6 had the best match-level outcome accuracy. The authors recommend comparing against margin-removed bookmaker probabilities. This comes from a search snippet only, not verified in the paper. Source: [arXiv 2608.03416](https://arxiv.org/pdf/2608.03416)
- **WorldCupArena** (arXiv 2607.18084) scores result accuracy, exact-score accuracy and a partial-credit scoreline metric, and is designed to be reused for leagues. Source: [Hugging Face papers 2607.18084](https://huggingface.co/papers/2607.18084)
- **General forecasting (non-sports), ForecastBench, Oct 2025.** Superforecasters had a difficulty-adjusted Brier of 0.081 against 0.101 for the best LLM (GPT-4.5). FRI estimated LLMs improve about 0.016 per year and, by linear extrapolation, reach parity around Nov 2026 (95% CI Dec 2025–Jan 2028). Source: [FRI Substack](https://forecastingresearch.substack.com/p/ai-llm-forecasting-model-forecastbench-benchmark)
- A separate preprint claims its "AIA Forecaster" matches superforecasters on ForecastBench. This is a vendor or system claim that has not been independently checked. Source: [arXiv 2511.07678](https://arxiv.org/pdf/2511.07678v1.pdf)

### Inferences
- The World Cup biases (too few draws, too few goals, one stock scoreline) are the same direction as the pipeline's existing weakness, which is under-predicting goals. If the LLM makes picks or sets magnitudes directly, it will probably make that worse, not better.
- High agreement across models and with the market favourite suggests LLMs mostly re-derive public consensus. Their marginal information over a rating model plus odds is small.
- International tournaments are not the Premier League. National teams have sparse rating data, so LLM context could matter *more* there. That makes the null result for leagues even less encouraging.

### Gaps
- I found no peer-reviewed or preprint study of an **LLM adjustment layer on a Poisson/Dixon-Coles model** for club football, or an LLM + Elo hybrid. A search for "LLM Elo hybrid" returned nothing relevant.
- No exact-score or goals-market results for LLMs could be confirmed. Only the WorldCup Arena abstract mentions goal and draw under-commitment.
- A 2026 Porto master's dissertation reportedly used LLM-extracted features from pre-match news for Liga Portugal. It was mentioned in one search summary, but a targeted search could not find it, so treat it as unverified.

## 2. Evidence that injuries, team news and lineups improve forecasts beyond ratings

### Takeaway
Lineup and player-rating information adds value when **combined with** team ratings. Alone, player ratings perform about the same as team ratings. Bookmakers are somewhat slow to price absence news, so there is real but small signal. Simple "remove the player's xG share" logic is shaky because teams adapt tactically.

### Cited Findings
- **Arntzen & Hvattum (2021, *Statistical Modelling* 21(5):449–470).** They compared Elo team ratings with plus-minus player ratings, using the rating gap between the starting lineups. Team and player ratings predicted about equally well alone, and **using both as covariates was significantly better than either**. Source: [Statistical Modelling abstract](https://statmod.org/smij/Vol21/Iss5/Arntzen/Abstract.html)
- **Holmes & McHale (2024, *International Journal of Forecasting* 40(1):302–312).** Their player-rating-based model captures the dynamic nature of team strength, and in a Kelly-type 1X2 betting test against bookmakers it "results in significant positive returns to betting". The abstract page gives no further figures. Source: [Liverpool repository](https://livrepository.liverpool.ac.uk/id/eprint/3170651)
- **Fischer & Schmal (*Economic Inquiry*).** They used absences of elite players as a quasi-natural experiment on 117,174 odds from 32 European bookmakers. They found **initial inertia** after absence announcements, then a lagged reaction that learning cannot explain. In other words, the market under-reacts at first to team news. Source: [TU Ilmenau news page](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-wirtschaftswissenschaften-und-medien/aktuelles/news/einzelnachricht/pricing-in-response-to-new-information-the-case-of-betting-markets)
- By contrast, prices react to *goals* "swiftly and fully" (Croxson & Reade), so markets are efficient for highly visible information. Source: [Reading eprints](https://reading-clone.eprints-hosting.org/34884)
- **Teams adapt when players are absent.** A 2025 study of players away at international tournaments found teams had *higher* total shots (p = .008), shots on target (p = .025) and goals (p = .003) during the absence period, which the authors read as tactical adaptation. Source: [Universidade de Évora](https://rdpc.uevora.pt/handle/10174/39051)
- Counterfactual event-model work suggests a replacement's effect can be approximated as the incoming player's influence minus the outgoing player's influence. That is a replacement-level framing, not "the missing player's xG share is lost". Source: [arXiv 2402.06815](https://arxiv.org/pdf/2402.06815)
- A vendor's "star player out = 0.6 xG reduction" figure has no published method. **Do not use it.** Source: [Fairplay Sports Media](https://www.fairplaysportsmedia.com/insights/ai-predictive-intelligence/player-effect-ai-measures-individual-impact-markets)

### Inferences
- The earlier plan-02 experiment scaled λ down by the injured players' share of xG. That **ignored replacement value** (the substitute also produces xG) and tactical adaptation, so it over-penalised. A replacement-adjusted delta would be much smaller. Combined with the model's existing under-prediction, the net effect was predictably harmful.
- The evidence supports lineup or availability info as an **additional covariate weighted by fitting** (as Arntzen & Hvattum did), not as a hand-set multiplier.

### Gaps
- The magnitude of the gain from combining team and player ratings (for example, the RPS difference) was not available from the abstract.
- I found no Premier League study that quantifies how much λ moves per absent key player after accounting for replacements.

## 3. Design patterns for structured adjustments: bounded multipliers vs LLM feature extraction

### Takeaway
The judgemental-forecasting literature says human or expert adjustments help mainly when they are **large, rare, based on genuinely new information and downward**. Small, frequent and upward adjustments hurt. Its strongest practical fix is **statistical debiasing of the adjustments**. Together with the LLM benchmarks, the more reliable design is: the LLM extracts discrete, verifiable features; a fitted model (with shrinkage) sets the magnitudes; and the default is no adjustment.

### Cited Findings
- **Fildes, Goodwin et al. (2009, ~60,000 forecasts, 4 companies).** Adjustments improved accuracy on average in three of the four companies. **Larger adjustments gave bigger gains, while small adjustments often hurt.** Positive (upward) adjustments were much less likely to help and were more often in the wrong direction, a sign of optimism bias. About 75% of forecasts were adjusted. At one retailer, positive adjustments more than doubled MAPE, from 32% to 65%. Sources: [Lancaster eprints](https://eprints.lancs.ac.uk/id/eprint/45487/); [Foresight "Good and Bad Judgment"](https://forecasters.org/wp-content/uploads/Good-and-Bad-Judgment-in-Forecasting_Issue8.pdf)
- **Fildes, Goodwin & De Baets (2025, *IJF* 41(2):649–669, ~147,000 forecasts from 6 studies):**
  - Adjustments improved bias and accuracy for only "just over half" of SKUs.
  - Positive adjustments were confirmed as more likely to worsen performance, while negative ones usually helped, especially large ones.
  - "Evidence that forecasters made effective use of relevant information not available to the algorithm was weak". Instead, adjusters responded to irrelevant or weakly diagnostic cues.
  - **A debiasing procedure applied to the adjusted forecasts proved effective.**
  - Sources: [IDEAS/RePEc](https://ideas.repec.org/a/eee/intfor/v41y2025i2p649-669.html); [Bath research portal](https://researchportal.bath.ac.uk/en/publications/forecast-value-added-in-demand-planning/)
- The LLM benchmarks show that LLM probability judgements track consensus and are biased against draws and goals ([2608.04008](https://arxiv.org/abs/2608.04008)). Self-assessed error rates vary widely (36–86%) ([2607.17765](https://arxiv.org/abs/2607.17765)), which argues against trusting LLM-stated "confidence" fields.

### Inferences (design recommendations, not directly tested in the sources)
- **Prefer extraction over magnitudes.** Ask the LLM for schema-validated facts per team: players out or doubtful (names, cross-checked against the squad list), whether each is a regular starter, new manager within N days, midweek European or cup game, an "already priced in" flag (out for more than 3–4 weeks), and dead-rubber or relegation-battle motivation. Then fit coefficients for these features on historical data with a calibrated model, e.g. log λ offsets regularised toward 0.
- **If the LLM does output multipliers:** keep separate attack and defence values, use a hard clamp (e.g. [0.85, 1.15]), and shrink toward 1 with a weight learned in backtest (λ' = λ·m^w, fitting w). Apply them only when a trigger feature fires, which matches the "large, rare" finding. Expect w to come out small.
- **Watch for asymmetry.** The pipeline's problem is *under*-prediction of goals, so a layer that mostly lowers λ for injuries pushes the wrong way. Use symmetric triggers: the opponent's defensive absences should *raise* λ. Re-centre the layer so the mean adjustment across a gameweek is about 1, so it reshuffles goals between fixtures rather than removing them.
- The Fildes debiasing result maps directly onto this: regress realised goals on (base λ, LLM adjustment) and use the fitted combination instead of the raw multiplier.

### Gaps
- Lawrence et al. (2006, *IJF* review of judgemental forecasting) was not retrieved. I am relying on the Fildes/Goodwin line of work for the "large and rare helps" claim.
- I found no study comparing LLM-set magnitudes with LLM-extracted features plus fitted weights in sports. The preference above is inferred.

## 4. Backtesting LLM adjustments without leakage

### Takeaway
Backtests on matches before the LLM's training cutoff are **not trustworthy**. Models leak post-event knowledge even when told to pretend otherwise, and the leak shows up in the forecasts even when the reasoning text looks clean. The only clean evaluation is **forward testing**: freeze predictions before kickoff, as every 2026 World Cup benchmark did. Sample sizes need to be in the hundreds of matches to detect small Brier differences.

### Cited Findings
- **"Pitfalls in Evaluating Language Model Forecasters"** (arXiv 2506.00723, 2025). Training cutoffs "are more of a guideline than a guarantee". Backtests also suffer *logical* leakage, because choosing questions with knowledge of their outcomes constrains the answers. Source: [arXiv 2506.00723](https://arxiv.org/html/2506.00723v1)
- **"Simulated Ignorance Fails"** (arXiv 2601.13717, Jan 2026). Prompting a model to pretend it lacks pre-cutoff knowledge does not work: there is a 52% gap between simulated and true ignorance. The authors recommend against benchmarking on pre-cutoff events and measure leakage through Brier scores, because reasoning text can look compliant while the predictions still reflect leaked knowledge. Source: [arXiv 2601.13717](https://arxiv.org/html/2601.13717v1)
- **"All Leaks Count, Some Count More"** (arXiv 2602.17234, Feb 2026). Standard prompting leaks substantially even with explicit temporal instructions. Their claim-level TimeSPEC method cuts leakage by 75–99%, but that is mitigation, not elimination. Source: [arXiv 2602.17234](https://arxiv.org/html/2602.17234v1)
- On pre-cutoff data, genuine forecasting skill cannot be told apart from memorised recall. Source: [SSRN 5217505](https://astrid-online.it/static/upload/protected/ssrn/ssrn-5217505.pdf)
- The World Cup benchmarks rely on prospective, timestamped collection. LLM-SoccerArena logs schema-validated, timestamped forecasts with prompts, model versions and tool traces ([2607.24573](https://arxiv.org/abs/2607.24573)). WorldCup Arena calls itself "leakage-free by construction" ([2608.04008](https://arxiv.org/abs/2608.04008)).
- Scale reference: one tournament (104 matches, 4–7 models) produced "narrow margins throughout" and models that were statistically hard to separate ([2608.04008](https://arxiv.org/abs/2608.04008)). The web-search effect was only 0.023 Brier ([2607.24573](https://arxiv.org/abs/2607.24573)).

### Inferences
- For the Premier League pipeline, use a **feature-only historical backtest** where the LLM never sees the match. Build injury and availability features from archived snapshot data (FPL `news` / `chance_of_playing` snapshots saved before each gameweek) and fit weights with no LLM in the loop. Run LLM extraction only on gameweeks after the model's cutoff, and only on text snapshotted before kickoff.
- **Run shadow-mode forward tests.** Each gameweek, freeze both the baseline and the adjusted predictions (consistent with the "predictions are frozen" rule) and compare points over the season. With 10 matches per gameweek, a 0.02 Brier-scale effect probably needs several hundred matches (around 20–38 gameweeks) to separate from noise. This is a rough inference; I found no sports-specific power calculation.
- Pin the model version and log prompts and raw outputs so results can be reproduced. Web search results change over time, so a re-run is not the same experiment.

### Gaps
- I found no published power analysis for detecting LLM-adjustment gains in football scoring metrics. The sample-size estimate above is my inference.

## 5. Practical inputs: free team-news sources and LLM web search

### Takeaway
The FPL `bootstrap-static` API is the best free, structured, timestamped source. Each player has `news`, `news_added`, `chance_of_playing_this_round` and `chance_of_playing_next_round`, but past values are not kept, so the pipeline must snapshot it itself. Web search gives LLMs only a small edge in the benchmarks.

### Cited Findings
- The FPL endpoint `https://fantasy.premierleague.com/api/bootstrap-static/` holds player records with `news` (string, empty when there is no flag), `news_added` (timestamp or null), and `chance_of_playing_this_round` / `chance_of_playing_next_round` (integer or null). These are community-documented fields, not an official spec. Sources: [fpl.readthedocs.io](https://fpl.readthedocs.io/en/latest/classes/player.html); [ExFPL docs](https://hexdocs.pm/ex_fpl/ExFPL.Player.html); [fpl-mcp types](https://glama.ai/mcp/servers/@owen-lacey/fpl-mcp/blob/e9171d6bb4bc00c522d02752962554112a340d30/src/types.ts)
- There are aggregator injury pages built on FPL data, e.g. [OneFPL injury news](https://onefpl.com/fpl-injury-news).
- Sports Mole publishes per-match "prediction, team news, lineups" previews, which could serve as a text source to snapshot. Source: [Sports Mole](https://www.sportsmole.co.uk/football/porto/?no-amp=1)
- LLM web search improved Brier by only 0.023 ([2607.24573](https://arxiv.org/abs/2607.24573)). Richer dossiers did not translate into accuracy in close matches ([2608.04008](https://arxiv.org/abs/2608.04008)).

### Inferences
- Start with FPL flags as **structured features with no LLM**: the sum of minutes-weighted starter shares with chance under 50%, split by position (attackers mainly affect own λ, defenders and keepers the opponent's λ). Add an LLM only for things FPL does not encode, such as a new manager, rotation before a European game, or press-conference hints. Even then, use it as an extractor into fixed fields.
- Web search makes runs non-reproducible and adds cost. Snapshotted text (FPL JSON plus one preview article per fixture, saved with a timestamp) is better for fair evaluation.

### Gaps
- Not verified in this research: the terms and robots rules for the BBC, the Premier League injury page and Sports Mole; whether `news_added` is reliably updated; and how early before kickoff FPL flags change.

## 6. Known failure modes

### Takeaway
The main risks are: (a) double-counting information already in ratings and the market, (b) systematic biases (too few draws and goals, optimism or pessimism asymmetry), (c) herding toward consensus, (d) leakage in backtests, and (e) responding to non-diagnostic cues. Hallucination is a further risk for player names and injury status.

### Cited Findings
- **Herding and consensus.** The models agreed with each other in 92% of top picks ([2607.17765](https://arxiv.org/abs/2607.17765)), and agreed with each other more than they were right ([2608.04008](https://arxiv.org/abs/2608.04008)).
- **Biases.** LLMs under-commit to draws and to goals and cluster on one scoreline ([2608.04008](https://arxiv.org/abs/2608.04008)).
- **Poor self-knowledge.** Self-reported error rates ranged from 36% to 86% on wrong picks ([2607.17765](https://arxiv.org/abs/2607.17765)).
- **Non-diagnostic cues.** Human adjusters reacted to irrelevant or weakly diagnostic cues, and there was weak evidence they used private information well ([Fildes, Goodwin & De Baets 2025](https://ideas.repec.org/a/eee/intfor/v41y2025i2p649-669.html)).
- **Directional asymmetry.** Adjustments in one direction were systematically worse: upward ones in demand planning ([Fildes et al. 2025](https://ideas.repec.org/a/eee/intfor/v41y2025i2p649-669.html)). In this pipeline, the plan-02 downward λ adjustments hit the model's existing under-prediction.
- **Leakage.** On pre-cutoff matches, predictions reflect leaked knowledge even when the reasoning looks compliant ([arXiv 2601.13717](https://arxiv.org/html/2601.13717v1)).
- **Tactical adaptation.** Teams can raise shots and goals while a player is absent ([Évora 2025](https://rdpc.uevora.pt/handle/10174/39051)), so naive absence penalties are biased.
- **Market efficiency.** The market prices team news with a lag but does eventually adjust ([Fischer & Schmal](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-wirtschaftswissenschaften-und-medien/aktuelles/news/einzelnachricht/pricing-in-response-to-new-information-the-case-of-betting-markets)). No LLM agent beat the closing market ([2607.17765](https://arxiv.org/abs/2607.17765)).

### Inferences
- **Double-counting.** Dixon-Coles with time decay already absorbs long-term absences through recent results. Adjustments should fire only on *changes* since the ratings window: new injuries, returns, a new manager. Add a feature like "days out", and set the adjustment to 0 for absences longer than about 4 weeks. The fitted weight will reveal whether long absences add anything.
- **Hallucination guard.** Validate every LLM-named player against the FPL squad list. Reject outputs that fail schema validation or name unknown players, and log the rejections rather than silently dropping them.
- **Treat returns as signal too.** A key player coming back should raise λ. This also counters the downward-only bias.

### Gaps
- I found no direct measurement of LLM hallucination rates for football injury extraction.
- No study was found that quantifies double-counting between time-decayed team ratings and availability features. Arntzen & Hvattum's "combined beats either" result implies the overlap is partial, not total ([Statistical Modelling](https://statmod.org/smij/Vol21/Iss5/Arntzen/Abstract.html)).
