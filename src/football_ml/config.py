"""All tunable constants and source URLs live here."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
PREDICTIONS_DIR = PROJECT_ROOT / "out" / "predictions"
RESULTS_DIR = PROJECT_ROOT / "out" / "results"
BACKTEST_DIR = PROJECT_ROOT / "out" / "backtest"

PREDICTOR_NAME = "dixon-coles"

# --- Sources ---------------------------------------------------------------
FOOTBALL_DATA_URL = "https://football-data.co.uk/mmz4281/{season}/{league}.csv"
FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
FPL_EVENT_LIVE_URL = "https://fantasy.premierleague.com/api/event/{gameweek}/live/"
ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard?dates={date}"
)
HTTP_TIMEOUT_SECONDS = 30
HTTP_HEADERS = {"User-Agent": "football-ml/0.1 (personal research project)"}
# Live data (FPL, current season CSV) is refetched after this many hours.
LIVE_CACHE_MAX_AGE_HOURS = 6

# --- Training data ---------------------------------------------------------
# football-data.co.uk season codes, oldest first. Last entry = current season.
TRAINING_SEASONS = ("2122", "2223", "2324", "2425", "2526", "2627")
CURRENT_SEASON = TRAINING_SEASONS[-1]
# E0 = Premier League, E1 = Championship. The Championship lets the model rate
# promoted teams; teams moving between leagues link the two on one scale.
TRAINING_LEAGUES = ("E0", "E1")

# --- Dixon-Coles model -----------------------------------------------------
# Per-day exponential time decay; ln(2)/365 ≈ 0.0019 → matches lose half
# their weight after a year.
TIME_DECAY_XI = 0.0019
# Ridge penalty on ratings so promoted teams with few PL games stay sane.
RATING_L2_PENALTY = 0.5
RHO_BOUNDS = (-0.2, 0.2)
MAX_GOALS = 10
TOP_SCORELINES = 3

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
