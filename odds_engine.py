"""
Chris's HR Value Model — vElite
backtest.py — Phase 2

Runs the 47-rule model against historical Statcast data.
Computes hit rates by score tier to calibrate the model
and produce the published accuracy table.

Usage:
  python backtest.py --year 2025
  python backtest.py --year 2026 --start 2026-03-20 --end 2026-05-10
"""

import requests
import json
import csv
import io
import datetime
import argparse
import os
from collections import defaultdict
from score_engine import (
    calculate_pitcher_pds,
    score_batter,
    safe_float,
    PARK_FACTORS,
)

TIERS = [
    ("90+",  90, 100),
    ("85-89", 85, 89),
    ("80-84", 80, 84),
    ("75-79", 75, 79),
    ("70-74", 70, 74),
    ("65-69", 65, 69),
    ("60-64", 60, 64),
    ("55-59", 55, 59),
    ("Below 55", 0, 54),
]


def fetch_statcast_game_log(start_date: str, end_date: str) -> list:
    """
    Pull all Statcast plate appearance results for a date range.
    Returns list of PA-level records including HR flag.
    """
    url = (
        "https://baseballsavant.mlb.com/statcast_search/csv"
        f"?all=true&hfPT=&hfAB=home_run%7C&hfGT=R%7C&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones=&hfPull=&hfC=&hfSea={start_date[:4]}%7C&hfSit=&player_type=batter&hfOuts=&opponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt={start_date}&game_date_lt={end_date}&hfInfield=&team=&position=&hfOutfield=&hfRO=&home_or_away=&hfFlag=&hfBBT=&metric_1=&hfInn=&min_pitches=0&min_results=0&group_by=name-date&sort_col=pitches&player_event_sort=api_p_release_speed&sort_order=desc&min_abs=0&type=details&"
    )
    print(f"  Fetching HR events {start_date} to {end_date}...")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        reader = csv.DictReader(io.StringIO(r.text))
        hrs = []
        for row in reader:
            if row.get("events") == "home_run":
                hrs.append({
                    "batter_id": safe_int(row.get("batter")),
                    "pitcher_id": safe_int(row.get("pitcher")),
                    "game_date": row.get("game_date"),
                    "home_team": row.get("home_team"),
                    "away_team": row.get("away_team"),
                    "inning": safe_int(row.get("inning")),
                    "stand": row.get("stand"),
                    "p_throws": row.get("p_throws"),
                    "launch_speed": safe_float(row.get("launch_speed")),
                    "launch_angle": safe_float(row.get("launch_angle")),
                    "hit_distance": safe_float(row.get("hit_distance_sc")),
                })
        print(f"  Found {len(hrs)} home run events.")
        return hrs
    except Exception as e:
        print(f"  WARNING: Statcast game log fetch failed: {e}")
        return []


def fetch_season_batter_stats(year: int) -> dict:
    """Pull season-long batter Statcast leaderboard for backtesting."""
    url = (
        f"https://baseballsavant.mlb.com/leaderboard/custom"
        f"?year={year}&type=batter&filter=&sort=4&sortDir=desc"
        f"&min=50&selections=xwoba,xslg,xiso,exit_velocity_avg,"
        f"barrel_batted_rate,hard_hit_percent,whiff_percent,k_percent,"
        f"bb_percent,pull_percent,flyball_percent,groundball_percent,"
        f"hr_flyball_percent,sweet_spot_percent,bat_speed&csv=true"
    )
    print(f"  Fetching {year} batter season data...")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        reader = csv.DictReader(io.StringIO(r.text))
        batters = {}
        for row in reader:
            pid = safe_int(row.get("player_id") or row.get("mlbam_id"))
            if not pid:
                continue
            batters[pid] = {
                "name": row.get("player_name", "Unknown"),
                "pa": safe_float(row.get("pa", 0)),
                "ev": safe_float(row.get("exit_velocity_avg")),
                "barrel_pct": safe_float(row.get("barrel_batted_rate")),
                "hh_pct": safe_float(row.get("hard_hit_percent")),
                "xwoba": safe_float(row.get("xwoba")),
                "xslg": safe_float(row.get("xslg")),
                "xiso": safe_float(row.get("xiso")),
                "whiff_pct": safe_float(row.get("whiff_percent")),
                "k_pct": safe_float(row.get("k_percent")),
                "bb_pct": safe_float(row.get("bb_percent")),
                "pull_pct": safe_float(row.get("pull_percent")),
                "fb_pct": safe_float(row.get("flyball_percent")),
                "gb_pct": safe_float(row.get("groundball_percent")),
                "hr_fb_pct": safe_float(row.get("hr_flyball_percent")),
                "sweet_spot_pct": safe_float(row.get("sweet_spot_percent")),
                "bat_speed": safe_float(row.get("bat_speed")),
            }
        print(f"  Loaded {len(batters)} batters for {year}.")
        return batters
    except Exception as e:
        print(f"  WARNING: Batter season fetch failed: {e}")
        return {}


def fetch_season_pitcher_stats(year: int) -> dict:
    """Pull season-long pitcher Statcast data for backtesting."""
    url = (
        f"https://baseballsavant.mlb.com/leaderboard/custom"
        f"?year={year}&type=pitcher&filter=&sort=4&sortDir=desc"
        f"&min=20&selections=xwoba,exit_velocity_avg,barrel_batted_rate,"
        f"hard_hit_percent,whiff_percent,k_percent,bb_percent,"
        f"flyball_percent,groundball_percent,hr_flyball_percent&csv=true"
    )
    url2 = (
        f"https://statsapi.mlb.com/api/v1/stats/leaders"
        f"?leaderCategories=earnedRunAverage,strikeoutsPer9Inn,"
        f"homeRunsPer9Inn,walksPer9Inn&season={year}&sportId=1&limit=500"
    )
    print(f"  Fetching {year} pitcher season data...")
    pitchers_savant = {}
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        reader = csv.DictReader(io.StringIO(r.text))
        for row in reader:
            pid = safe_int(row.get("player_id") or row.get("mlbam_id"))
            if not pid:
                continue
            pitchers_savant[pid] = {
                "barrel_pct_allowed": safe_float(row.get("barrel_batted_rate")),
                "hh_pct_allowed": safe_float(row.get("hard_hit_percent")),
                "whiff_pct": safe_float(row.get("whiff_percent")),
                "k_pct": safe_float(row.get("k_percent")),
                "bb_pct": safe_float(row.get("bb_percent")),
                "fb_pct_allowed": safe_float(row.get("flyball_percent")),
                "gb_pct_allowed": safe_float(row.get("groundball_percent")),
                "hr_fb_pct_allowed": safe_float(row.get("hr_flyball_percent")),
            }
        print(f"  Loaded {len(pitchers_savant)} pitchers for {year}.")
    except Exception as e:
        print(f"  WARNING: Pitcher Savant fetch failed: {e}")
    return pitchers_savant


def run_backtest(year: int, start_date: str = None, end_date: str = None):
    """
    Main backtesting function.
    Scores all batters who batted on each game day, compares to actual HR results.
    """
    if not start_date:
        start_date = f"{year}-03-20"
    if not end_date:
        end_date = f"{year}-11-01"

    print(f"\n{'='*60}")
    print(f"Backtesting {year} · {start_date} → {end_date}")
    print(f"{'='*60}\n")

    # Load season data
    batter_data = fetch_season_batter_stats(year)
    pitcher_data = fetch_season_pitcher_stats(year)

    # Fetch all HR events in date range
    hr_events = fetch_statcast_game_log(start_date, end_date)
    hr_set = {(e["batter_id"], e["game_date"]) for e in hr_events}

    print(f"\n  Total HR events to match: {len(hr_events)}")

    # Fetch schedule for the date range to get all PA contexts
    # We iterate by week to avoid rate limits
    tier_results = defaultdict(lambda: {"picks": 0, "hits": 0})
    score_results = []

    current = datetime.date.fromisoformat(start_date)
    end = datetime.date.fromisoformat(end_date)

    processed_days = 0
    total_scored = 0
    total_hr = 0

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")

        # Get schedule for this day
        try:
            r = requests.get(
                f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
                f"&hydrate=probablePitcher,lineups",
                timeout=15
            )
            data = r.json()
        except Exception:
            current += datetime.timedelta(days=1)
            continue

        for date_obj in data.get("dates", []):
            for game in date_obj.get("games", []):
                away = game.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation", "")
                home = game.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation", "")

                away_pid = game.get("teams", {}).get("away", {}).get("probablePitcher", {}).get("id")
                home_pid = game.get("teams", {}).get("home", {}).get("probablePitcher", {}).get("id")

                # Grade pitchers
                away_season = {}  # Would need full season stats per pitcher for perfect accuracy
                home_season = {}

                away_pds = calculate_pitcher_pds(away_pid, f"P{away_pid}", away_season, pitcher_data)
                home_pds = calculate_pitcher_pds(home_pid, f"P{home_pid}", home_season, pitcher_data)

                # Get lineups from game
                gp = game.get("gamePk")
                try:
                    lr = requests.get(
                        f"https://statsapi.mlb.com/api/v1.1/game/{gp}/feed/live",
                        timeout=10
                    )
                    ldata = lr.json()
                    for side, opp_pds, park, pitcher_hand in [
                        ("away", home_pds, home, "R"),
                        ("home", away_pds, home, "R"),
                    ]:
                        batting_team = away if side == "away" else home
                        batters_data = ldata.get("liveData", {}).get("boxscore", {}).get("teams", {}).get(side, {})
                        batting_order = batters_data.get("battingOrder", [])
                        players = batters_data.get("players", {})

                        for slot_idx, pid in enumerate(batting_order):
                            slot = slot_idx + 1
                            sd = batter_data.get(pid, {})
                            if not sd:
                                continue

                            player_info = players.get(f"ID{pid}", {})
                            hand = player_info.get("batSide", {}).get("code", "R")
                            sd["player_id"] = pid
                            sd["name"] = player_info.get("person", {}).get("fullName", f"P{pid}")

                            scored = score_batter(
                                batter=sd,
                                pitcher_pds=opp_pds,
                                slot=slot,
                                hand=hand,
                                pitcher_hand=pitcher_hand,
                                park=park,
                                il_ids=set(),
                            )

                            s = scored["score"]
                            actually_homered = (pid, date_str) in hr_set

                            # Assign to tier
                            for tier_name, lo, hi in TIERS:
                                if lo <= s <= hi:
                                    tier_results[tier_name]["picks"] += 1
                                    if actually_homered:
                                        tier_results[tier_name]["hits"] += 1
                                    break

                            score_results.append({
                                "date": date_str,
                                "player_id": pid,
                                "name": sd.get("name"),
                                "team": batting_team,
                                "score": s,
                                "homered": actually_homered,
                            })
                            total_scored += 1
                            if actually_homered:
                                total_hr += 1

                except Exception:
                    pass

        processed_days += 1
        current += datetime.timedelta(days=1)

        # Progress update every 7 days
        if processed_days % 7 == 0:
            print(f"  Processed through {date_str} · {total_scored:,} PAs · {total_hr:,} HRs")

        import time as t
        t.sleep(0.5)  # Rate limiting

    # ── COMPUTE AND DISPLAY RESULTS ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS — {year} ({start_date} to {end_date})")
    print(f"Total PAs scored: {total_scored:,} · Total HRs: {total_hr:,}")
    print(f"Baseline HR rate: {total_hr/total_scored*100:.2f}%")
    print(f"\n{'Score Tier':<12} {'Picks':>8} {'HRs':>8} {'Hit Rate':>10} {'vs Baseline':>12}")
    print("-" * 55)

    baseline = total_hr / total_scored if total_scored > 0 else 0.13
    calibration = {}

    for tier_name, lo, hi in TIERS:
        r = tier_results[tier_name]
        picks = r["picks"]
        hits = r["hits"]
        rate = hits / picks if picks > 0 else 0
        vs_baseline = rate / baseline if baseline > 0 else 0
        calibration[tier_name] = {
            "picks": picks,
            "hits": hits,
            "hit_rate": round(rate * 100, 2),
            "vs_baseline": round(vs_baseline, 2),
            "lo": lo,
            "hi": hi,
        }
        print(f"{tier_name:<12} {picks:>8,} {hits:>8,} {rate*100:>9.2f}% {vs_baseline:>10.2f}x")

    print(f"{'='*60}\n")

    # Save results
    os.makedirs("output", exist_ok=True)
    output = {
        "year": year,
        "start_date": start_date,
        "end_date": end_date,
        "total_scored": total_scored,
        "total_hr": total_hr,
        "baseline_rate": round(baseline * 100, 3),
        "calibration": calibration,
        "model_version": "vElite v4.7",
        "rules": 47,
    }

    with open(f"output/backtest_{year}.json", "w") as f:
        json.dump(output, f, indent=2)

    # Also save all scored records for deeper analysis
    with open(f"output/backtest_{year}_records.json", "w") as f:
        json.dump(score_results, f)

    print(f"Results saved to output/backtest_{year}.json")
    return output


def safe_int(val):
    try:
        return int(val) if val not in (None, "", "null") else None
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest the 47-rule HR model")
    parser.add_argument("--year", type=int, default=2025, help="Season year")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    args = parser.parse_args()
    run_backtest(args.year, args.start, args.end)
