"""FootPredict 90 — transparent pre-match probability estimators.

This is a prototype, not a calibrated or validated betting model.
Only full-time 90-minute results (FT) are used to train goal rates.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from math import exp, lgamma, log, sqrt
from statistics import mean, pvariance
from typing import Any, Optional


def num(x: Any) -> Optional[float]:
    if x is None or x == "" or x == "—":
        return None
    try:
        val = float(str(x).replace("%", "").replace(",", "."))
        return val if 0 <= val < 1000 else None
    except (ValueError, TypeError):
        return None


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return exp(-lam + k * log(lam) - lgamma(k + 1))


def poisson_cdf(k: int, lam: float) -> float:
    return clamp(sum(poisson_pmf(i, lam) for i in range(max(0, k) + 1)), 0, 1) if k >= 0 else 0.0


def count_cdf(k: int, mu: float, var: Optional[float] = None) -> float:
    """Negative-binomial method of moments, else Poisson.

    NB is a descriptive count approximation; not guaranteed calibrated.
    """
    if k < 0:
        return 0.0
    mu = clamp(mu, 0, 100)
    if not var or var <= mu * 1.005:
        return poisson_cdf(k, mu)
    r = mu * mu / (var - mu)
    p = r / (r + mu)
    q = 1 - p
    mass = exp(r * log(p))
    total = mass
    for i in range(1, k + 1):
        mass *= (i - 1 + r) / i * q
        total += mass
    return clamp(total, 0, 1)


def win_draw_loss(lh: float, la: float) -> tuple[float, float, float]:
    hg = [poisson_pmf(i, lh) for i in range(19)]
    ag = [poisson_pmf(i, la) for i in range(19)]
    h = sum(hg[i] * ag[j] for i in range(19) for j in range(i))
    d = sum(hg[i] * ag[i] for i in range(19))
    a = max(0.0, 1 - h - d)
    return h, d, a


def goal_events(lh: float, la: float, home: str, away: str) -> dict[str, float]:
    total = lh + la
    h, d, a = win_draw_loss(lh, la)
    out = {}
    for n in range(1, 6):
        out[f"Au moins {n} but{'s' if n != 1 else ''} dans le match"] = 1 - poisson_cdf(n - 1, total)
    for n in range(1, 9):
        out[f"Moins de {n + 0.5:g} buts dans le match"] = poisson_cdf(n, total)
    for who, lam in [(home, lh), (away, la)]:
        for n in range(1, 7):
            out[f"{who} marque au maximum {n} but{'s' if n != 1 else ''}"] = poisson_cdf(n, lam)
        out[f"{who} marque au moins 1 but"] = 1 - exp(-lam)
    out[f"Double chance : {home} ou nul (1X)"] = h + d
    out[f"Double chance : {away} ou nul (X2)"] = a + d
    out["Les deux équipes marquent"] = (1 - exp(-lh)) * (1 - exp(-la))
    out["Au moins une équipe ne marque pas"] = 1 - out["Les deux équipes marquent"]
    out["Match nul"] = d
    # Illustrative model: two independent equal-duration halves at lambda/2.
    p_h_half = win_draw_loss(lh / 2, la / 2)[0]
    p_a_half = win_draw_loss(lh / 2, la / 2)[2]
    out["Aucune équipe ne gagne les deux mi-temps"] = 1 - p_h_half**2 - p_a_half**2
    out["Moins de 2,5 buts en première mi-temps"] = poisson_cdf(2, total / 2)
    return out


def extract_finished_matches(fixtures: list[dict], team_id: int,
                             before: Optional[datetime] = None,
                             limit: int = 20) -> list[dict]:
    rows = []
    for f in fixtures:
        if ((f.get("fixture") or {}).get("status") or {}).get("short") != "FT":
            continue
        sides = f.get("teams") or {}
        hid, aid = (sides.get("home") or {}).get("id"), (sides.get("away") or {}).get("id")
        if team_id not in (hid, aid):
            continue
        ts = (f.get("fixture") or {}).get("timestamp")
        if not isinstance(ts, (int, float)):
            continue
        dt = datetime.fromtimestamp(ts, timezone.utc)
        if before and dt >= before:
            continue
        home_side = team_id == hid
        goals = f.get("goals") or {}
        home_goals, away_goals = num(goals.get("home")), num(goals.get("away"))
        if home_goals is None or away_goals is None:
            continue
        rows.append({
            "id": (f.get("fixture") or {}).get("id"), "date": dt,
            "home": home_side, "gf": int(home_goals if home_side else away_goals),
            "ga": int(away_goals if home_side else home_goals),
            "league_id": (f.get("league") or {}).get("id"),
            "opponent": (sides.get("away") if home_side else sides.get("home") or {}).get("name", "?"),
            "yellow": None, "fouls": None,
        })
    return sorted(rows, key=lambda x: x["date"], reverse=True)[:limit]


def enrich_stat_rows(rows: list[dict], stats_by_fixture: dict[int, list[dict]],
                     team_id: int) -> None:
    for row in rows:
        blocks = stats_by_fixture.get(row["id"], [])
        for block in blocks:
            if (block.get("team") or {}).get("id") != team_id:
                continue
            stats = {str(s.get("type") or "").strip().casefold(): s.get("value")
                     for s in block.get("statistics", [])}
            row["yellow"] = num(stats.get("yellow cards"))
            row["fouls"] = num(stats.get("fouls"))
            break


def venue_avg(rows: list[dict], key: str, at_home: bool) -> float:
    all_vals = [x[key] for x in rows if num(x.get(key)) is not None]
    if not all_vals:
        raise ValueError("Résultats historiques insuffisants pour estimer les buts.")
    same = [x[key] for x in rows if x.get("home") == at_home and num(x.get(key)) is not None]
    if len(same) >= 3:
        # venue sample deliberately shrunk toward all recent fixtures
        weight = len(same) / (len(same) + 4)
        return weight * mean(same) + (1 - weight) * mean(all_vals)
    return mean(all_vals)


def predict_goals(home_rows: list[dict], away_rows: list[dict]) -> tuple[float, float]:
    if len(home_rows) < 3 or len(away_rows) < 3:
        raise ValueError("Il faut au moins 3 résultats complets pour chaque équipe.")
    ha = venue_avg(home_rows, "gf", True)
    hd = venue_avg(home_rows, "ga", True)
    aa = venue_avg(away_rows, "gf", False)
    ad = venue_avg(away_rows, "ga", False)
    nh, na = len(home_rows), len(away_rows)
    w = min(nh, na) / (min(nh, na) + 8)
    # 1.35 and 1.10 goals/game are neutral illustrative priors, NOT live league statistics.
    lh = (1 - w) * 1.35 + w * (0.5 * ha + 0.5 * ad)
    la = (1 - w) * 1.10 + w * (0.5 * aa + 0.5 * hd)
    return clamp(lh, 0.15, 4.25), clamp(la, 0.15, 4.25)


def event_scenarios(lh: float, la: float, home: str, away: str, n: int):
    delta = clamp(0.75 / sqrt(max(n, 1)), 0.20, 0.45)
    central = goal_events(lh, la, home, away)
    low = {k: 1.0 for k in central}
    for h_mul in (1 - delta, 1, 1 + delta):
        for a_mul in (1 - delta, 1, 1 + delta):
            trial = goal_events(lh * h_mul, la * a_mul, home, away)
            for k, p in trial.items():
                low[k] = min(low[k], p)
    return central, low, delta


def describe_form(rows: list[dict]) -> dict:
    last5 = rows[:5]
    if not last5:
        return {"n": 0}
    wins = sum(x["gf"] > x["ga"] for x in last5)
    draws = sum(x["gf"] == x["ga"] for x in last5)
    rest = None
    if isinstance(last5[0].get("date"), datetime):
        rest = (datetime.now(timezone.utc) - last5[0]["date"]).days
    return {"n": len(rows), "w": wins, "d": draws,
            "l": len(last5) - wins - draws,
            "gf": round(mean(x["gf"] for x in last5), 2),
            "ga": round(mean(x["ga"] for x in last5), 2),
            "rest_days": rest}


def discrete_predictive(rows_h: list[dict], rows_a: list[dict], key: str):
    h = [float(r[key]) for r in rows_h if num(r.get(key)) is not None]
    a = [float(r[key]) for r in rows_a if num(r.get(key)) is not None]
    if len(h) < 5 or len(a) < 5:
        return None
    mu = mean(h) + mean(a)
    # Extra-Poisson variability to reduce false certainty.
    variance = max(mu * 1.20, pvariance(h) + pvariance(a))
    n = min(len(h), len(a))
    delta = clamp(0.75 / sqrt(n), 0.20, 0.45)
    return mu, variance, n, delta


def discrete_events(stat_h: list[dict], stat_a: list[dict], key: str):
    fit = discrete_predictive(stat_h, stat_a, key)
    if fit is None:
        return None
    mu, variance, n, delta = fit
    central, low = {}, {}
    if key == "yellow":
        thresholds = [3, 4, 5, 6, 7, 8, 9, 10, 11]
        prefix = "Moins de {} cartons jaunes dans le match"
    else:
        thresholds = [20, 25, 30, 35, 40, 45, 50]
        prefix = "Moins de {} fautes dans le match"
    for cut in thresholds:
        # Discrete event 'less than 7' = count <= 6.
        title = prefix.format(cut)
        p0 = count_cdf(cut - 1, mu, variance)
        trials = [count_cdf(cut - 1, mu * fac,
                            max(mu * fac * 1.2, variance * fac ** 2))
                  for fac in (1 - delta, 1, 1 + delta)]
        central[title] = p0
        low[title] = min(trials)
    return {"avg": mu, "n": n, "variation": delta,
            "central": central, "sensitivity_floor": low}


def rank_events(home: str, away: str, home_rows: list[dict],
                away_rows: list[dict]) -> dict:
    lh, la = predict_goals(home_rows, away_rows)
    n = min(len(home_rows), len(away_rows))
    central, floor, delta = event_scenarios(lh, la, home, away, n)
    details = {"Buts": {"central": central, "sensitivity_floor": floor,
                         "avg": lh + la, "n": n, "variation": delta}}
    for label, key in (("Cartons", "yellow"), ("Fautes", "fouls")):
        fit = discrete_events(home_rows, away_rows, key)
        if fit:
            details[label] = fit
    events = []
    for category, data in details.items():
        for label, p in data["central"].items():
            events.append({
                "Événement": label, "Catégorie": category,
                "Probabilité estimée": round(100 * p, 1),
                "Plancher de sensibilité": round(100 * data["sensitivity_floor"][label], 1),
                "Analyse robuste ≥90 %": data["sensitivity_floor"][label] >= 0.90,
            })
    return {"home": home, "away": away, "lambda_h": lh, "lambda_a": la,
            "n_h": len(home_rows), "n_a": len(away_rows),
            "details": details, "events": sorted(events, key=lambda x: x["Probabilité estimée"], reverse=True),
            "form_h": describe_form(home_rows), "form_a": describe_form(away_rows)}


def match_team_name(query: str, results: list[dict]) -> Optional[dict]:
    def similarity(row):
        t = (row.get("team") or {}).get("name", "")
        q = query.casefold().strip()
        return max(SequenceMatcher(None, q, t.casefold()).ratio(),
                   0.99 if q == t.casefold().strip() else 0)
    if not results:
        return None
    return max(results, key=similarity).get("team")
