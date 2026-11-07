# Parlay Builder

A command-line tool that pulls live odds and team stats from the web. It estimates each team's true chance of winning (the moneyline bet) and builds parlays three ways:

| Mode | What it optimizes |
|---|---|
| `safe` | Highest chance of the parlay hitting |
| `balanced` | Most expected value, while keeping the hit chance ≥ `--min-prob` (default 25%) |
| `value` | Only legs where the model thinks the book's price is too generous |

It uses only Python's standard library, so there's nothing to install. It needs Python 3.9+.

```bash
python3 parlay.py nfl                                    # today's NFL slate, all 3 modes
python3 parlay.py nfl ncaaf --legs 4 --mode balanced
python3 parlay.py nba --date 20261115 --bankroll 300     # adds a suggested stake (quarter Kelly)
ODDS_API_KEY=xxxx python3 parlay.py nfl --book fanduel
ODDS_API_KEY=xxxx python3 parlay.py nhl --best-price     # price each leg at the best book
python3 parlay.py nfl --json > slate.json
```

Leagues: `nfl ncaaf nba ncaab nhl mlb`

## How a leg's probability is estimated

The tool blends three estimates (averaged on the log-odds scale):

| Source | Default weight | Notes |
|---|---|---|
| **Market** | 60% | Without a key: DraftKings' odds via ESPN, with the book's built-in margin ("vig") removed. With `ODDS_API_KEY`: a no-vig average across 40+ books, with sharp books (Pinnacle, Circa, …) counted more heavily. |
| **ESPN predictor** | 25% | ESPN's own matchup model (FPI / BPI) |
| **Stats model** | 15% | Points scored vs. allowed this season (Pythagorean win %), a head-to-head formula (log5), and a home-field adjustment. Early-season records are pulled toward .500. |

You can change the weights with `--weights 0.7,0.2,0.1`.

For each leg:
- **Edge** = model probability − the probability implied by the price.
- **EV** (expected value) = model probability × decimal odds − 1. That's your average profit per $1 bet.

For a parlay, hit probability = the product of the leg probabilities. The tool allows at most one leg per game, so legs aren't strongly linked. The price is the product of the legs' decimal odds.

## Getting the most out of it

- **Get a free key from [The Odds API](https://the-odds-api.com)** (500 requests/month). This is the biggest upgrade. Your edge comes from a book's price being worse than the sharp consensus, and with DraftKings alone the tool can only compare DraftKings to ESPN and the stats model.
- **Use `--best-price`** if you have accounts at several books. Taking the best line on each leg is the most reliable edge in betting.
- **Be skeptical of big disagreements with the market.** If the ESPN predictor or stats model differs from the market by 10+ points, the market is usually right. This is especially true in college, because the stats model ignores schedule strength. Lower the weights if you see this often.
- **Kelly sizing** (`--bankroll`) suggests $0 when the expected value is negative. Take that seriously.

## Honest limitations

- **Moneylines only.** No spreads, totals or player props.
- **"Highest hit probability" is not the same as "a good bet."** Stacking heavy favorites raises the chance of hitting, but every leg carries the book's margin, so those parlays usually have negative expected value. The output shows this for every parlay.
- **Lines move,** and a book's actual parlay price can differ slightly from the product of the legs. Check the price on the bet slip before you place the bet.
- **ESPN's API is unofficial** and can change without notice.

Run the offline tests with `python3 -m unittest test_parlay.py`.

Bet responsibly. Problem gambling help: **1-800-GAMBLER**.
