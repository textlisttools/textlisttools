#!/usr/bin/env python3
"""
Parlay Builder - pulls live odds + stats from the web, estimates the true win
probability of every moneyline, and assembles parlays with the best hit
probability / expected value.

Data sources
  * ESPN public API (no key):   schedule, DraftKings moneylines, ESPN matchup
                                predictor, team standings (points for/against)
  * The Odds API (optional key): moneylines from 40+ books -> sharp-weighted,
                                no-vig consensus + best available price
                                (free key: https://the-odds-api.com)

Pure standard library - no pip install needed.

    python3 parlay.py nfl
    python3 parlay.py nfl ncaaf --legs 4 --mode value
    ODDS_API_KEY=xxx python3 parlay.py nba --book fanduel --bankroll 500
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict

# --------------------------------------------------------------------------- #
# League config
#   pyth   : Pythagorean exponent for points-for / points-against win%
#   home   : home-field advantage, in log-odds
#   prior  : pseudo-games of .500 added to shrink small samples
# --------------------------------------------------------------------------- #
LEAGUES = {
    "nfl":   dict(sport="football",   league="nfl",                     odds_key="americanfootball_nfl",   pyth=2.37, home=0.20, prior=4),
    "ncaaf": dict(sport="football",   league="college-football",        odds_key="americanfootball_ncaaf", pyth=2.37, home=0.45, prior=3),
    "nba":   dict(sport="basketball", league="nba",                     odds_key="basketball_nba",         pyth=14.0, home=0.25, prior=10),
    "ncaab": dict(sport="basketball", league="mens-college-basketball", odds_key="basketball_ncaab",       pyth=11.5, home=0.50, prior=6),
    "nhl":   dict(sport="hockey",     league="nhl",                     odds_key="icehockey_nhl",          pyth=2.0,  home=0.12, prior=12),
    "mlb":   dict(sport="baseball",   league="mlb",                     odds_key="baseball_mlb",           pyth=1.83, home=0.12, prior=20),
}

ESPN = "https://site.api.espn.com/apis"
ODDS_API = "https://api.the-odds-api.com/v4"
SHARP_BOOKS = {"pinnacle": 3.0, "circasports": 2.0, "betonlineag": 1.5, "lowvig": 1.5}

# How much each probability source counts toward the final model probability.
DEFAULT_WEIGHTS = {"market": 0.60, "espn": 0.25, "stats": 0.15}


# --------------------------------------------------------------------------- #
# Odds math
# --------------------------------------------------------------------------- #
def american_to_decimal(a: float) -> float:
    a = float(a)
    return 1 + a / 100 if a > 0 else 1 + 100 / abs(a)


def decimal_to_american(d: float) -> int:
    if d <= 1:
        return 0
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))


def implied(dec: float) -> float:
    return 1 / dec


def no_vig(dec_a: float, dec_b: float) -> tuple[float, float]:
    """Remove the bookmaker margin proportionally from a two-way market."""
    ia, ib = implied(dec_a), implied(dec_b)
    s = ia + ib
    return ia / s, ib / s


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def kelly(p: float, dec: float) -> float:
    b = dec - 1
    return max(0.0, (b * p - (1 - p)) / b) if b > 0 else 0.0


def fmt_american(a) -> str:
    a = int(round(a))
    return f"+{a}" if a > 0 else str(a)


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def get_json(url: str, params: dict | None = None, timeout: int = 20):
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "parlay-builder/1.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode()), r.headers


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Leg:
    league: str
    game_id: str
    game: str            # "AWAY @ HOME"
    start: str
    team: str
    team_id: str
    opponent: str
    side: str            # "home" | "away"
    american: int        # price offered at the chosen book
    book: str
    p_market: float | None = None
    p_espn: float | None = None
    p_stats: float | None = None
    p_model: float = 0.0
    link: str | None = None
    sources: list = field(default_factory=list)

    @property
    def dec(self) -> float:
        return american_to_decimal(self.american)

    @property
    def p_implied(self) -> float:
        return implied(self.dec)

    @property
    def edge(self) -> float:
        return self.p_model - self.p_implied

    @property
    def ev(self) -> float:
        return self.p_model * self.dec - 1


# --------------------------------------------------------------------------- #
# ESPN
# --------------------------------------------------------------------------- #
def espn_events(cfg: dict, date: str | None):
    params = {"dates": date} if date else {}
    if cfg["league"] in ("college-football", "mens-college-basketball"):
        params["groups"] = "80" if cfg["league"] == "college-football" else "50"  # all D-I
        params["limit"] = "400"
    data, _ = get_json(f"{ESPN}/site/v2/sports/{cfg['sport']}/{cfg['league']}/scoreboard", params)
    return data.get("events", [])


def espn_predictor(cfg: dict, event_id: str) -> dict | None:
    try:
        data, _ = get_json(f"{ESPN}/site/v2/sports/{cfg['sport']}/{cfg['league']}/summary", {"event": event_id})
    except Exception:
        return None
    pred = data.get("predictor") or {}
    try:
        h = float(pred["homeTeam"]["gameProjection"]) / 100
        a = float(pred["awayTeam"]["gameProjection"]) / 100
        s = h + a
        return {"home": h / s, "away": a / s} if s > 0 else None
    except (KeyError, TypeError, ValueError):
        return None


def espn_team_stats(cfg: dict) -> dict:
    """team_id -> {w, l, t, pf, pa, gp}"""
    try:
        data, _ = get_json(f"{ESPN}/v2/sports/{cfg['sport']}/{cfg['league']}/standings")
    except Exception:
        return {}
    out = {}

    def walk(node):
        for e in (node.get("standings") or {}).get("entries", []):
            # key by `type` (season totals); names repeat for home/road splits
            st, record = {}, None
            for s in e.get("stats", []):
                k = (s.get("type") or s.get("name") or "").lower()
                if k == "total":
                    record = s.get("summary") or s.get("displayValue")
                if k and k not in st:
                    st[k] = s.get("value")
            w, l, t = (st.get(k) or 0 for k in ("wins", "losses", "ties"))
            m = re.match(r"^(\d+)-(\d+)(?:-(\d+))?$", record or "")
            if m:   # overall record string is the most reliable source (college omits 'losses')
                w, l, t = int(m[1]), int(m[2]), int(m[3] or 0)
                if cfg["sport"] == "hockey":   # W-L-OTL: third number is losses, not ties
                    l, t, otl = l + t, 0, 0
            otl = st.get("otlosses") or st.get("overtimelosses") or 0
            pf = st.get("pointsfor") or st.get("avgpointsfor")
            pa = st.get("pointsagainst") or st.get("avgpointsagainst")
            out[str(e["team"]["id"])] = dict(w=w, l=l + otl, t=t, pf=pf, pa=pa, gp=w + l + t + otl)
        for ch in node.get("children", []):
            walk(ch)

    walk(data)
    return out


def stats_prob(cfg: dict, home: dict | None, away: dict | None, neutral: bool) -> float | None:
    """Home win probability from Pythagorean expectation + log5 + home edge."""
    def strength(s):
        if not s or not s.get("gp"):
            return None
        gp = s["gp"]
        if s.get("pf") and s.get("pa"):
            pf, pa = float(s["pf"]), float(s["pa"])
            pyth = pf ** cfg["pyth"] / (pf ** cfg["pyth"] + pa ** cfg["pyth"])
        else:
            pyth = (s["w"] + 0.5 * s["t"]) / gp
        # regress toward .500 for small samples
        return (pyth * gp + 0.5 * cfg["prior"]) / (gp + cfg["prior"])

    ph, pa = strength(home), strength(away)
    if ph is None or pa is None:
        return None
    log5 = (ph - ph * pa) / (ph + pa - 2 * ph * pa)
    return sigmoid(logit(log5) + (0 if neutral else cfg["home"]))


def dk_link(href: str | None) -> str | None:
    if not href:
        return None
    q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
    return q.get("preurl", [href])[0]


def parse_espn_moneyline(comp: dict):
    """Return (home_american, away_american, provider, home_link, away_link) or None."""
    for o in comp.get("odds") or []:
        ml = o.get("moneyline") or {}
        prov = (o.get("provider") or {}).get("name", "ESPN BET")
        try:
            h = ml["home"].get("close") or ml["home"].get("open")
            a = ml["away"].get("close") or ml["away"].get("open")
            ho, ao = int(str(h["odds"]).replace("+", "")), int(str(a["odds"]).replace("+", ""))
            return ho, ao, prov, dk_link((h.get("link") or {}).get("href")), dk_link((a.get("link") or {}).get("href"))
        except (KeyError, TypeError, ValueError, AttributeError):
            pass
        try:
            ho, ao = int(o["homeTeamOdds"]["moneyLine"]), int(o["awayTeamOdds"]["moneyLine"])
            return ho, ao, prov, None, None
        except (KeyError, TypeError, ValueError):
            pass
    return None


# --------------------------------------------------------------------------- #
# The Odds API (optional)
# --------------------------------------------------------------------------- #
def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower().replace("&", "and").replace("st.", "state"))


def odds_api_lines(cfg: dict, key: str) -> dict:
    """norm(home)+'|'+norm(away) -> {'books': {book: (home_am, away_am)}}"""
    try:
        data, headers = get_json(
            f"{ODDS_API}/sports/{cfg['odds_key']}/odds",
            {"apiKey": key, "regions": "us,us2,eu", "markets": "h2h", "oddsFormat": "american"},
        )
    except urllib.error.HTTPError as e:
        print(f"  ! The Odds API error {e.code} for {cfg['odds_key']} (check your key / quota)", file=sys.stderr)
        return {}
    rem = headers.get("x-requests-remaining")
    if rem is not None:
        print(f"  (The Odds API requests remaining: {rem})", file=sys.stderr)
    out = {}
    for g in data:
        home, away = g["home_team"], g["away_team"]
        books = {}
        for b in g.get("bookmakers", []):
            for m in b.get("markets", []):
                if m["key"] != "h2h":
                    continue
                px = {o["name"]: o["price"] for o in m["outcomes"]}
                if home in px and away in px and len(px) == 2:
                    books[b["key"]] = (px[home], px[away])
        out[norm(home) + "|" + norm(away)] = {"books": books}
    return out


def consensus(books: dict) -> tuple[float, int] | None:
    """Sharp-weighted average of no-vig home probability across books."""
    tot = wsum = 0.0
    for bk, (h, a) in books.items():
        ph, _ = no_vig(american_to_decimal(h), american_to_decimal(a))
        w = SHARP_BOOKS.get(bk, 1.0)
        tot += ph * w
        wsum += w
    return (tot / wsum, len(books)) if wsum else None


def match_odds_game(lines: dict, home_name: str, away_name: str):
    k = norm(home_name) + "|" + norm(away_name)
    if k in lines:
        return lines[k]
    nh, na = norm(home_name), norm(away_name)
    for key, v in lines.items():
        h, a = key.split("|")
        if (h.startswith(nh[:8]) or nh.startswith(h[:8])) and (a.startswith(na[:8]) or na.startswith(a[:8])):
            return v
    return None


# --------------------------------------------------------------------------- #
# Build legs
# --------------------------------------------------------------------------- #
def blend(parts: dict, weights: dict) -> float:
    """Blend probabilities in log-odds space using available sources only."""
    num = den = 0.0
    for k, p in parts.items():
        if p is None:
            continue
        num += weights[k] * logit(p)
        den += weights[k]
    return sigmoid(num / den) if den else 0.5


def build_legs(league_key: str, args, odds_key: str | None) -> list[Leg]:
    cfg = LEAGUES[league_key]
    print(f"Fetching {league_key.upper()} ...", file=sys.stderr)
    events = [e for e in espn_events(cfg, args.date) if e["competitions"][0]["status"]["type"]["state"] == "pre"]
    if not events:
        print(f"  no upcoming {league_key.upper()} games found", file=sys.stderr)
        return []

    stats = espn_team_stats(cfg)
    with ThreadPoolExecutor(max_workers=8) as ex:
        preds = dict(zip([e["id"] for e in events], ex.map(lambda e: espn_predictor(cfg, e["id"]), events)))
    lines = odds_api_lines(cfg, odds_key) if odds_key else {}

    legs = []
    for ev in events:
        comp = ev["competitions"][0]
        teams = {c["homeAway"]: c for c in comp["competitors"]}
        if "home" not in teams or "away" not in teams:
            continue
        H, A = teams["home"]["team"], teams["away"]["team"]
        game = f"{A['abbreviation']} @ {H['abbreviation']}"
        neutral = bool(comp.get("neutralSite"))

        # --- prices + market probability
        espn_ml = parse_espn_moneyline(comp)
        api = match_odds_game(lines, H["displayName"], A["displayName"]) if lines else None
        books = (api or {}).get("books", {})

        price = {}   # side -> (american, book, link)
        if books:
            if args.best_price:
                for i, side in enumerate(("home", "away")):
                    bk, px = max(books.items(), key=lambda kv: american_to_decimal(kv[1][i]))
                    price[side] = (px[i], bk, None)
            elif args.book in books:
                h, a = books[args.book]
                price = {"home": (h, args.book, None), "away": (a, args.book, None)}
        if not price and espn_ml and args.book in ("draftkings", "any"):
            price = {"home": (espn_ml[0], espn_ml[2], espn_ml[3]), "away": (espn_ml[1], espn_ml[2], espn_ml[4])}
        if not price:
            continue
        if espn_ml and args.book == "draftkings":   # keep DK bet-slip links
            price["home"] = price["home"][:2] + (espn_ml[3],)
            price["away"] = price["away"][:2] + (espn_ml[4],)

        cons = consensus(books) if len(books) >= 3 else None
        if cons:
            p_mkt_home, nbooks = cons
            mkt_src = f"consensus of {nbooks} books"
        else:
            ph_, _ = no_vig(american_to_decimal(price["home"][0]), american_to_decimal(price["away"][0]))
            p_mkt_home, mkt_src = ph_, f"no-vig {price['home'][1]}"

        pred = preds.get(ev["id"])
        p_espn_home = pred["home"] if pred else None
        p_stats_home = stats_prob(cfg, stats.get(str(H["id"])), stats.get(str(A["id"])), neutral)

        for side, me, opp in (("home", H, A), ("away", A, H)):
            flip = (lambda p: p) if side == "home" else (lambda p: None if p is None else 1 - p)
            parts = {"market": flip(p_mkt_home), "espn": flip(p_espn_home), "stats": flip(p_stats_home)}
            am, bk, link = price[side]
            leg = Leg(
                league=league_key.upper(), game_id=ev["id"], game=game, start=ev.get("date", ""),
                team=me["displayName"], team_id=str(me["id"]), opponent=opp["displayName"], side=side,
                american=int(am), book=bk, p_market=parts["market"], p_espn=parts["espn"], p_stats=parts["stats"],
                p_model=blend(parts, args.weights), link=link,
                sources=[mkt_src] + (["ESPN predictor"] if p_espn_home is not None else []) + (["stats model"] if p_stats_home is not None else []),
            )
            legs.append(leg)
    return legs


# --------------------------------------------------------------------------- #
# Parlay construction
# --------------------------------------------------------------------------- #
@dataclass
class Parlay:
    legs: list
    @property
    def dec(self): return math.prod(l.dec for l in self.legs)
    @property
    def prob(self): return math.prod(l.p_model for l in self.legs)
    @property
    def implied(self): return 1 / self.dec
    @property
    def ev(self): return self.prob * self.dec - 1


def build_parlay(legs: list[Leg], n: int, mode: str, min_prob: float, min_leg_prob: float, max_fav: int | None,
                 exclude: set) -> Parlay | None:
    pool = [l for l in legs if l.p_model >= min_leg_prob and l.game_id not in exclude
            and (max_fav is None or l.american >= max_fav)]
    if mode == "value":
        pool = [l for l in pool if l.edge > 0]

    if mode == "safe":   # most likely to hit; break ties on EV
        rank, score = (lambda l: l.p_model), (lambda pr, ev: (pr, ev))
    else:                # most expected value
        rank, score = (lambda l: l.ev), (lambda pr, ev: (ev, pr))

    # both sides of every game stay in play; cap the search space
    cands = sorted(pool, key=rank, reverse=True)[:24]
    games = {l.game_id for l in cands}
    if len(games) < n:   # not enough games - return what qualifies, one per game
        seen, picked = set(), []
        for l in cands:
            if l.game_id not in seen:
                seen.add(l.game_id)
                picked.append(l)
        return Parlay(picked) if picked else None

    best, best_score = None, None
    for combo in itertools.combinations(cands, n):
        if len({l.game_id for l in combo}) < n:      # never two legs from one game
            continue
        pr = math.prod(l.p_model for l in combo)
        if mode == "balanced" and pr < min_prob:
            continue
        ev = pr * math.prod(l.dec for l in combo) - 1
        s = score(pr, ev)
        if best_score is None or s > best_score:
            best, best_score = combo, s
    return Parlay(list(best)) if best else None


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def pct(p):
    return "  -  " if p is None else f"{p*100:5.1f}%"


def print_board(legs: list[Leg], top: int):
    legs = sorted(legs, key=lambda l: l.ev, reverse=True)[:top]
    print("\nLEG BOARD  (sorted by expected value; model = blend of market / ESPN / stats)")
    hdr = f"{'Lg':<6}{'Game':<13}{'Pick':<26}{'Price':>7}{'Book':>12}{'Impl':>8}{'Mkt':>8}{'ESPN':>8}{'Stats':>8}{'Model':>8}{'Edge':>8}{'EV':>8}"
    print(hdr)
    print("-" * len(hdr))
    for l in legs:
        print(f"{l.league:<6}{l.game:<13}{l.team[:25]:<26}{fmt_american(l.american):>7}{l.book[:11]:>12}"
              f"{pct(l.p_implied):>8}{pct(l.p_market):>8}{pct(l.p_espn):>8}{pct(l.p_stats):>8}"
              f"{pct(l.p_model):>8}{l.edge*100:>+7.1f}%{l.ev*100:>+7.1f}%")


def print_parlay(title: str, p: Parlay, stake: float, bankroll: float | None, kelly_frac: float):
    print(f"\n=== {title}: {len(p.legs)}-leg parlay ===")
    for i, l in enumerate(p.legs, 1):
        print(f"  {i}. {l.team} ML {fmt_american(l.american)} ({l.book})  vs {l.opponent}  "
              f"[{l.game}, {l.start[:16].replace('T', ' ')}Z]  model {l.p_model*100:.1f}%")
        if l.link:
            print(f"     bet slip: {l.link}")
    am = decimal_to_american(p.dec)
    print(f"  Combined price : {fmt_american(am)}  (decimal {p.dec:.2f})")
    print(f"  Hit probability: {p.prob*100:.1f}%   (book implies {p.implied*100:.1f}%)")
    print(f"  ${stake:g} stake pays: ${stake*p.dec:,.2f}   expected value: {p.ev*100:+.1f}% (${stake*p.ev:+.2f})")
    if bankroll:
        f = kelly(p.prob, p.dec) * kelly_frac
        print(f"  Suggested stake ({kelly_frac:g}x Kelly on ${bankroll:g}): "
              + (f"${bankroll*f:.2f}" if f > 0 else "$0 - no edge, don't bet this"))
    if p.ev <= 0:
        print("  NOTE: negative expected value - the book's margin outweighs the model's edge here.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Build data-driven parlays from live odds + stats.")
    ap.add_argument("leagues", nargs="*", default=["nfl"], help=f"one or more of: {', '.join(LEAGUES)}")
    ap.add_argument("--legs", type=int, default=3, help="legs per parlay (default 3)")
    ap.add_argument("--mode", choices=["safe", "value", "balanced", "all"], default="all",
                    help="safe = max hit probability, value = only +EV legs, balanced = max EV with hit prob >= --min-prob")
    ap.add_argument("--min-prob", type=float, default=0.25, help="balanced mode: min parlay hit probability (default 0.25)")
    ap.add_argument("--min-leg-prob", type=float, default=None,
                    help="ignore legs below this win prob (default: safe 0.50, balanced 0.40, value 0.20)")
    ap.add_argument("--max-favorite", type=int, default=-600,
                    help="skip legs priced shorter than this, e.g. -600 (they add risk but almost no payout)")
    ap.add_argument("--book", default="draftkings",
                    help="book to price legs at (draftkings, fanduel, betmgm, caesars, ...). Non-DK books need ODDS_API_KEY")
    ap.add_argument("--best-price", action="store_true", help="price each leg at the best book available (needs ODDS_API_KEY)")
    ap.add_argument("--odds-api-key", default=os.environ.get("ODDS_API_KEY"), help="The Odds API key (or env ODDS_API_KEY)")
    ap.add_argument("--date", help="YYYYMMDD (default: ESPN's current slate)")
    ap.add_argument("--stake", type=float, default=10, help="stake used for payout examples (default 10)")
    ap.add_argument("--bankroll", type=float, help="your bankroll, to size bets with fractional Kelly")
    ap.add_argument("--kelly", type=float, default=0.25, help="Kelly fraction (default 0.25 = quarter Kelly)")
    ap.add_argument("--weights", default=None, help="model weights market,espn,stats (default 0.60,0.25,0.15)")
    ap.add_argument("--top", type=int, default=25, help="rows to show on the leg board")
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON instead of tables")
    args = ap.parse_args(argv)

    bad = [l for l in args.leagues if l not in LEAGUES]
    if bad:
        ap.error(f"unknown league(s): {bad}. choose from {list(LEAGUES)}")
    if args.weights:
        try:
            m, e, s = (float(x) for x in args.weights.split(","))
        except ValueError:
            ap.error("--weights must look like 0.6,0.25,0.15")
        args.weights = {"market": m, "espn": e, "stats": s}
    else:
        args.weights = dict(DEFAULT_WEIGHTS)
    if args.book != "draftkings" and not args.odds_api_key:
        ap.error("--book other than draftkings needs an ODDS_API_KEY (free at the-odds-api.com)")
    if args.best_price and not args.odds_api_key:
        ap.error("--best-price needs an ODDS_API_KEY")

    legs = []
    for lg in args.leagues:
        try:
            legs += build_legs(lg, args, args.odds_api_key)
        except urllib.error.URLError as e:
            print(f"  ! could not reach data source for {lg}: {e}", file=sys.stderr)
    if not legs:
        print("No priced games found. Try another league or --date YYYYMMDD.", file=sys.stderr)
        return 1

    modes = ["safe", "balanced", "value"] if args.mode == "all" else [args.mode]
    titles = {"safe": "HIGHEST HIT PROBABILITY", "balanced": f"BEST VALUE w/ >= {args.min_prob*100:.0f}% HIT",
              "value": "+EV LEGS ONLY"}
    leg_floor = {"safe": 0.50, "balanced": 0.40, "value": 0.20}
    parlays = {m: build_parlay(legs, args.legs, m, args.min_prob,
                               args.min_leg_prob if args.min_leg_prob is not None else leg_floor[m],
                               args.max_favorite, set())
               for m in modes}

    if args.json:
        out = {"legs": [dict(asdict(l), dec=l.dec, p_implied=l.p_implied, edge=l.edge, ev=l.ev) for l in legs],
               "parlays": {m: None if p is None else {"legs": [l.team for l in p.legs], "american": decimal_to_american(p.dec),
                                                       "prob": p.prob, "ev": p.ev} for m, p in parlays.items()}}
        print(json.dumps(out, indent=2))
        return 0

    print_board(legs, args.top)
    for m in modes:
        p = parlays[m]
        if not p:
            print(f"\n=== {titles[m]}: no qualifying legs ===")
            continue
        if len(p.legs) < args.legs:
            print(f"\n(only {len(p.legs)} leg(s) qualified for {titles[m]})")
        print_parlay(titles[m], p, args.stake, args.bankroll, args.kelly)

    print("\nAssumes legs are independent (one leg per game). Lines move - re-check prices before betting.\n"
          "Parlays multiply the book's margin; even good parlays lose most of the time. Bet only what you can afford to lose.\n"
          "Problem gambling help: 1-800-GAMBLER")
    return 0


if __name__ == "__main__":
    sys.exit(main())
