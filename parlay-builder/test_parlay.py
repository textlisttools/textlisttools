"""Offline tests: python3 -m unittest test_parlay.py"""
import math
import unittest
from types import SimpleNamespace
from unittest import mock

import parlay as P


class OddsMath(unittest.TestCase):
    def test_conversions(self):
        self.assertAlmostEqual(P.american_to_decimal(-110), 1.9090909, places=6)
        self.assertAlmostEqual(P.american_to_decimal(150), 2.5)
        self.assertEqual(P.decimal_to_american(2.5), 150)
        self.assertEqual(P.decimal_to_american(1.5), -200)

    def test_no_vig_sums_to_one(self):
        a, b = P.no_vig(P.american_to_decimal(-110), P.american_to_decimal(-110))
        self.assertAlmostEqual(a, 0.5)
        a, b = P.no_vig(P.american_to_decimal(-180), P.american_to_decimal(150))
        self.assertAlmostEqual(a + b, 1)
        self.assertGreater(a, 0.6)

    def test_kelly(self):
        self.assertEqual(P.kelly(0.5, 1.9), 0)              # -EV -> no bet
        self.assertAlmostEqual(P.kelly(0.55, 2.0), 0.10)    # classic example

    def test_consensus_weights_sharps(self):
        books = {"pinnacle": (-200, 170), "draftkings": (-150, 130)}
        p, n = P.consensus(books)
        pin, _ = P.no_vig(P.american_to_decimal(-200), P.american_to_decimal(170))
        dk, _ = P.no_vig(P.american_to_decimal(-150), P.american_to_decimal(130))
        self.assertEqual(n, 2)
        self.assertAlmostEqual(p, (3 * pin + dk) / 4)

    def test_stats_prob_symmetry_and_home_edge(self):
        cfg = P.LEAGUES["nfl"]
        t = dict(w=5, l=3, t=0, pf=200, pa=180, gp=8)
        self.assertAlmostEqual(P.stats_prob(cfg, t, t, neutral=True), 0.5)
        self.assertGreater(P.stats_prob(cfg, t, t, neutral=False), 0.5)


def leg(gid, team, am, p):
    return P.Leg(league="NFL", game_id=gid, game=gid, start="", team=team, team_id=team, opponent="x",
                 side="home", american=am, book="dk", p_model=p)


class ParlayBuilder(unittest.TestCase):
    legs = [leg("g1", "A", -300, 0.78), leg("g1", "A2", 250, 0.24),
            leg("g2", "B", -250, 0.73), leg("g2", "B2", 210, 0.29),
            leg("g3", "C", 150, 0.45), leg("g3", "C2", -170, 0.57),
            leg("g4", "D", -120, 0.58), leg("g4", "D2", 100, 0.44)]

    def test_safe_picks_highest_probability_one_per_game(self):
        p = P.build_parlay(self.legs, 3, "safe", 0, 0.5, None, set())
        self.assertEqual([l.team for l in p.legs], ["A", "B", "D"])
        self.assertEqual(len({l.game_id for l in p.legs}), 3)
        self.assertAlmostEqual(p.prob, 0.78 * 0.73 * 0.58)

    def test_value_only_uses_positive_edge(self):
        p = P.build_parlay(self.legs, 3, "value", 0, 0.2, None, set())
        self.assertTrue(all(l.edge > 0 for l in p.legs))

    def test_balanced_respects_min_prob(self):
        p = P.build_parlay(self.legs, 3, "balanced", 0.30, 0.2, None, set())
        self.assertGreaterEqual(p.prob, 0.30)

    def test_max_favorite_filter(self):
        p = P.build_parlay(self.legs, 2, "safe", 0, 0.5, -260, set())
        self.assertNotIn("A", [l.team for l in p.legs])


class OddsApiPath(unittest.TestCase):
    """build_legs with a mocked ESPN + The Odds API response."""

    def fake_get_json(self, url, params=None, timeout=20):
        if "scoreboard" in url:
            return {"events": [{"id": "1", "date": "2026-11-08T18:00Z", "competitions": [{
                "status": {"type": {"state": "pre"}}, "neutralSite": False,
                "competitors": [
                    {"homeAway": "home", "team": {"id": "10", "abbreviation": "HOM", "displayName": "Home Hawks"}},
                    {"homeAway": "away", "team": {"id": "20", "abbreviation": "AWY", "displayName": "Away Owls"}}],
                "odds": [{"provider": {"name": "DraftKings"},
                          "moneyline": {"home": {"close": {"odds": "-150"}}, "away": {"close": {"odds": "+130"}}}}],
            }]}]}, {}
        if "summary" in url:
            return {"predictor": {"homeTeam": {"gameProjection": "60"}, "awayTeam": {"gameProjection": "40"}}}, {}
        if "standings" in url:
            return {"children": []}, {}
        if "the-odds-api" in url:
            return [{"home_team": "Home Hawks", "away_team": "Away Owls", "bookmakers": [
                {"key": k, "markets": [{"key": "h2h", "outcomes": [
                    {"name": "Home Hawks", "price": h}, {"name": "Away Owls", "price": a}]}]}
                for k, h, a in [("pinnacle", -140, 128), ("draftkings", -150, 130), ("fanduel", -145, 135)]
            ]}], {"x-requests-remaining": "499"}
        raise AssertionError(url)

    def test_best_price_and_consensus(self):
        args = SimpleNamespace(date=None, book="draftkings", best_price=True, weights=P.DEFAULT_WEIGHTS)
        with mock.patch.object(P, "get_json", self.fake_get_json):
            legs = P.build_legs("nfl", args, "KEY")
        by = {l.side: l for l in legs}
        self.assertEqual(by["home"].american, -140)
        self.assertEqual(by["home"].book, "pinnacle")
        self.assertEqual(by["away"].american, 135)
        self.assertEqual(by["away"].book, "fanduel")
        self.assertIn("consensus of 3 books", by["home"].sources)
        self.assertAlmostEqual(by["home"].p_model + by["away"].p_model, 1, places=6)

    def test_specific_book(self):
        args = SimpleNamespace(date=None, book="fanduel", best_price=False, weights=P.DEFAULT_WEIGHTS)
        with mock.patch.object(P, "get_json", self.fake_get_json):
            legs = P.build_legs("nfl", args, "KEY")
        self.assertEqual({l.american for l in legs}, {-145, 135})


if __name__ == "__main__":
    unittest.main()
