"""All tunable constants and source URLs live here."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
PREDICTIONS_DIR = PROJECT_ROOT / "out" / "predictions"

# --- Sources ---------------------------------------------------------------
FOOTBALL_DATA_URL = "https://football-data.co.uk/mmz4281/{season}/E0.csv"
FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
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

# --- Dixon-Coles model -----------------------------------------------------
# Per-day exponential time decay; ln(2)/365 ≈ 0.0019 → matches lose half
# their weight after a year.
TIME_DECAY_XI = 0.0019
# Ridge penalty on ratings so promoted teams with few PL games stay sane.
RATING_L2_PENALTY = 0.5
RHO_BOUNDS = (-0.2, 0.2)
MAX_GOALS = 10
TOP_SCORELINES = 3

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
