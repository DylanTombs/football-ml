"""All tunable constants and source URLs live here."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
PREDICTIONS_DIR = PROJECT_ROOT / "out" / "predictions"
RESULTS_DIR = PROJECT_ROOT / "out" / "results"
BACKTEST_DIR = PROJECT_ROOT / "out" / "backtest"
# Pre-deadline FPL player state, one file per gameweek; committed because the
# FPL API has no history of it.
SNAPSHOTS_DIR = PROJECT_ROOT / "data" / "snapshots"
# Copy of the upcoming odds used for each frozen prediction (fixtures.csv is overwritten).
ODDS_SNAPSHOTS_DIR = PROJECT_ROOT / "data" / "odds"
SNAPSHOT_FIELDS = (
    "id", "web_name", "first_name", "second_name", "team", "element_type",
    "expected_goals", "goals_scored", "minutes", "starts", "status",
    "chance_of_playing_next_round", "chance_of_playing_this_round", "news", "news_added",
    "penalties_order", "direct_freekicks_order", "corners_and_indirect_freekicks_order",
)

PREDICTOR_NAME = "dixon-coles"
NAIVE_FAVOURITE_NAME = "naive-favourite"

# --- Sources ---------------------------------------------------------------
FOOTBALL_DATA_URL = "https://football-data.co.uk/mmz4281/{season}/{league}.csv"
# Upcoming fixtures with pre-closing odds, all leagues; overwritten every round
# (odds appear Friday afternoon for weekend games, Tuesday for midweek).
FOOTBALL_DATA_FIXTURES_URL = "https://football-data.co.uk/fixtures.csv"
UPCOMING_ODDS_MAX_AGE_HOURS = 1
FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
FPL_EVENT_LIVE_URL = "https://fantasy.premierleague.com/api/event/{gameweek}/live/"
FPL_PLAYER_URL = "https://fantasy.premierleague.com/api/element-summary/{player_id}/"
ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard?dates={date}"
)
HTTP_TIMEOUT_SECONDS = 30
HTTP_HEADERS = {"User-Agent": "football-ml/0.1 (personal research project)"}
# Live data (FPL, current season CSV) is refetched after this many hours.
LIVE_CACHE_MAX_AGE_HOURS = 6
# Per-player FPL match history (one request per player; only changes after a round).
PLAYER_HISTORY_MAX_AGE_HOURS = 12

# --- Training data ---------------------------------------------------------
# football-data.co.uk season codes, oldest first. Last entry = current season.
# 2020-21 is a warm-up year for the plan 06 walk-forward (decayed to ~1% weight live).
TRAINING_SEASONS = ("2021", "2122", "2223", "2324", "2425", "2526", "2627")
CURRENT_SEASON = TRAINING_SEASONS[-1]
# E0 = Premier League, E1 = Championship. The Championship lets the model rate
# promoted teams; teams moving between leagues link the two on one scale.
TRAINING_LEAGUES = ("E0", "E1")

# Pre-closing market-average odds (closing "C" columns would leak late news).
# Pre-closing odds are collected Friday afternoon (Tuesday for midweek games).
ODDS_COLUMNS = {"AvgH": "odds_home", "AvgD": "odds_draw", "AvgA": "odds_away",
                "Avg>2.5": "odds_over", "Avg<2.5": "odds_under"}
ODDS_LEAGUE = "E0"
ODDS_MATCH_DAYS = 3  # an odds row matches a fixture kicking off within this many days

# --- Dixon-Coles model -----------------------------------------------------
# Per-day exponential time decay; ln(2)/365 ≈ 0.0019 → matches lose half
# their weight after a year.
TIME_DECAY_XI = 0.0019
# Ridge penalty on ratings so promoted teams with few PL games stay sane.
RATING_L2_PENALTY = 0.5
RHO_BOUNDS = (-0.2, 0.2)
MAX_GOALS = 10
TOP_SCORELINES = 3
FORM_MATCHES = 5  # recent results given to predictors as context

# --- Multi-season bench (plan 06) ------------------------------------------
BENCH_DIR = PROJECT_ROOT / "out" / "bench"
BENCH_LEAGUE = "E0"  # forecasts are scored on Premier League matches only
BENCH_TUNE_SEASONS = ("2122", "2223", "2324", "2425")
BENCH_HOLDOUT_SEASONS = ("2526",)  # tested once, after tuning
BENCH_SEASON_SETS = {"tune": BENCH_TUNE_SEASONS, "holdout": BENCH_HOLDOUT_SEASONS}
BENCH_XI_GRID = (0.0010, 0.0015, 0.0019, 0.0025, 0.0030, 0.0040)
BENCH_L2_GRID = (0.1, 0.25, 0.5, 1.0, 2.0)
MARKET_RATE_START = (1.4, 1.1)  # starting (λ, μ) when fitting rates to odds
# λ = w × Dixon-Coles + (1 − w) × market. Kept at 1.0: bookmaker odds are never
# a model input (decided 2026-11-08: the series is a statistical model). Odds
# are a benchmark only. For the record, w = 0 scored better on the plan 06 bench.
ODDS_MODEL_WEIGHT = 1.0

# --- Injuries --------------------------------------------------------------
# λ_adj = λ × (1 − INJURY_IMPACT × missing_share). Below 1 because
# replacements still score some of the missing goals. Off (0.0): every value
# swept in plan 02 (0.25 / 0.5 / 0.75) scored fewer backtest points.
INJURY_IMPACT = 0.0
# FPL statuses meaning the player is gone (u = left / loaned out, n = ineligible).
FPL_DEPARTED_STATUSES = ("u", "n")
FPL_OUT_STATUSES = ("i", "s")
# How the backtest decides who was missing in a past gameweek (FPL has no
# history of injury news):
#   lagged — 0 minutes in the team's previous match after playing before (no leak)
#   oracle — 0 minutes in that gameweek itself (uses the real line-up: ceiling only)
#   live   — today's injury list (only meaningful for the next gameweek)
AVAILABILITY_PROXIES = ("lagged", "oracle", "live")
DEFAULT_BACKTEST_PROXY = "lagged"

# --- Goalscorers -----------------------------------------------------------
MIN_CHANCE_OF_PLAYING = 75
SCORER_XG_WEIGHT = 0.7  # remainder goes to actual goals scored
SCORER_REPEAT_DECAY = 1.0  # weight / (1 + decay * goals_already_given)

# --- Scorer model (plan 05) -------------------------------------------------
# weights — share of 0.7 × xG + 0.3 × goals (plans 01-04)
# chance  — shrunk non-penalty xG per 90 × expected minutes, plus penalties
SCORER_MODELS = ("weights", "chance")
DEFAULT_SCORER_MODEL = "chance"  # plan 05: better calibrated (Brier 0.0303 vs 0.0320), +6 scorer hits on the same scorelines
PENALTY_XG = 0.78  # Opta xG of a penalty, also used as its conversion rate
PRIOR_PENALTY_ATTEMPTS_PER_MATCH = 0.13  # per team; ESPN 2023-26: 0.105 pen goals / 0.78
PENALTY_PRIOR_MATCHES = 20  # team penalty rate shrunk towards the prior with this many matches
SCORER_PRIOR_MINUTES = 900  # npxG per 90 shrunk towards the position average with this many minutes
POSITION_MIN_MINUTES = 900  # below this a position's pooled rate falls back to the table below
FALLBACK_NPXG_PER90 = {1: 0.0, 2: 0.05, 3: 0.12, 4: 0.30}  # FPL element_type: GK, DEF, MID, FWD
EXPECTED_MINUTES_MATCHES = 5  # expected minutes = mean over the team's last N matches
OWN_GOAL_SHARE = 0.03  # share of a team's goals that are opponents' own goals

# --- Picks (plan 04) -------------------------------------------------------
# modal: likeliest result, then likeliest score in it; scorers by weight decay;
#        minutes sampled from this season's goal times.
# expected-points: the plausible scoreline worth the most expected points;
#        k-th pick of a player worth P(scores ≥ k); fixed minutes.
PICK_STRATEGIES = ("modal", "expected-points")
DEFAULT_PICK_STRATEGY = "expected-points"  # beat modal by +39 pts on GW1-9 (plan 04)
# Only scorelines at least this likely may be picked (keeps picks reel-worthy;
# without it the best pick is about 6-5 every match).
MIN_SCORELINE_PROBABILITY = 0.05
# Fixed elapsed minutes for a player picked k times, with non-overlapping
# ±MINUTE_TOLERANCE windows. Tuned with `best_minutes` on 3,284 ESPN goal times
# from 2023-24 to 2025-26 (out of sample for the 2026-27 backtest).
REPEAT_PICK_MINUTES = {1: (56,), 2: (56, 86), 3: (41, 64, 86)}
# Share of goals inside a fixed ±10 window (56' covers 26%; repeat windows 23-24%).
MINUTE_HIT_RATE = 0.25

# --- Predictors ------------------------------------------------------------
# name -> (provider, model). Dixon-Coles has no provider.
PREDICTOR_SPECS = {
    "dixon-coles": None,
    "claude": ("anthropic", "claude-opus-5-5"),
    "gpt": ("openrouter", "openai/gpt-6.1-sol"),
    "gemini": ("openrouter", "google/gemini-4-argon"),
}
DEFAULT_PREDICTORS = ("dixon-coles",)
PROVIDER_API_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}

# --- LLM calls -------------------------------------------------------------
LLM_CACHE_DIR = CACHE_DIR / "llm"
LLM_MAX_ATTEMPTS = 2  # first try + one retry, then fall back to Dixon-Coles
LLM_MAX_TOKENS = 16000
LLM_TIMEOUT_SECONDS = 300
ANTHROPIC_EFFORT = "high"
# Server-side refusal fallback: the API re-runs a declined request on another model.
ANTHROPIC_FALLBACK_BETA = "server-side-fallback-2026-07-01"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Valid predicted goal clocks: regulation minute 1-90, stoppage only after 45 or 90.
MAX_STOPPAGE_MINUTES = 15
STOPPAGE_MINUTES = (45, 90)

# --- Team names: FPL name -> football-data.co.uk name (canonical) ---------
FPL_TO_CANONICAL_TEAM = {
    "Coventry City": "Coventry",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leicester City": "Leicester",
    "Luton Town": "Luton",
    "Man Utd": "Man United",
    "Sheffield Utd": "Sheffield United",
    "Spurs": "Tottenham",
    "West Brom": "West Brom",
}

# --- ESPN team name -> football-data.co.uk name (canonical) ---------------
ESPN_TO_CANONICAL_TEAM = {
    "AFC Bournemouth": "Bournemouth",
    "Brighton & Hove Albion": "Brighton",
    "Coventry City": "Coventry",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
}

# --- Points for scoring predictions ---------------------------------------
POINTS_EXACT_SCORE = 3
POINTS_CORRECT_RESULT = 1
POINTS_PER_SCORER = 1
POINTS_PER_MINUTE = 1
MINUTE_TOLERANCE = 10

# --- Comparing predictors (paired block bootstrap over gameweeks) -----------
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
POINT_COMPONENTS = ("total", "outcome", "scorers", "minutes")
