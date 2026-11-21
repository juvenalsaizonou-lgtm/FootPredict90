"""FootPredict 90 — application Streamlit, version prototype.

Lancez : streamlit run app.py
Données externes : https://www.api-football.com/ (clé gratuite personnelle)
"""
from __future__ import annotations

from datetime import datetime, timezone, date
from typing import Any

import pandas as pd
import requests
import streamlit as st

from core import (
    extract_finished_matches, enrich_stat_rows, match_team_name,
    rank_events, num,
)

API_ROOT = "https://v3.football.api-sports.io/"

st.set_page_config(page_title="FootPredict 90", page_icon="⚽", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.8rem; max-width: 1190px}
h1 {font-weight: 800; letter-spacing: -.04em}
[data-testid="stMetric"] {background: rgba(100,130,140,.08); border: 1px solid rgba(128,128,128,.17);padding: .75rem;border-radius: 12px}
</style>
""", unsafe_allow_html=True)


class ApiError(RuntimeError):
    pass


@st.cache_data(ttl=3600, show_spinner=False)
def api_get(key: str, path: str, params_tuple: tuple = ()) -> list[dict]:
    url = API_ROOT + path.lstrip("/")
    try:
        resp = requests.get(url, params=dict(params_tuple),
                            headers={"x-apisports-key": key}, timeout=18)
    except requests.RequestException as exc:
        raise ApiError("Impossible de contacter API-Football. Vérifie ta connexion.") from exc
    if resp.status_code in (401, 403):
        raise ApiError("Clé API invalide ou données non autorisées par ton offre.")
    if resp.status_code == 429:
        raise ApiError("Limite de requêtes atteinte. Recommence après le renouvellement du quota.")
    try:
        obj = resp.json()
    except ValueError as exc:
        raise ApiError("Réponse illisible de la source sportive.") from exc
    if resp.status_code >= 400:
        raise ApiError(f"La source a renvoyé une erreur HTTP {resp.status_code}.")
    errors = obj.get("errors")
    if errors:
        detail = str(errors)[:230]
        raise ApiError("La source refuse cette requête : " + detail)
    return obj.get("response") or []


def fetch(key: str, path: str, **params) -> list[dict]:
    return api_get(key, path, tuple(sorted((k, v) for k, v in params.items() if v is not None)))


def date_label(f: dict) -> str:
    raw = (f.get("fixture") or {}).get("date") or ""
    return str(raw).replace("T", " ")[:16]


def discover_match(key: str, hid: int, aid: int, target_date: date):
    # Match must exist in the API. Never synthesize a fixture from team names.
    scheduled = fetch(key, "fixtures", team=hid, next=35)
    possible = []
    for f in scheduled:
        ids = f.get("teams") or {}
        if aid not in ((ids.get("home") or {}).get("id"), (ids.get("away") or {}).get("id")):
            continue
        if ((f.get("fixture") or {}).get("status") or {}).get("short") not in ("NS", "TBD"):
            continue
        raw = ((f.get("fixture") or {}).get("date") or "")[:10]
        try:
            distance = abs((date.fromisoformat(raw) - target_date).days)
        except ValueError:
            distance = 999
        possible.append((distance, f))
    return min(possible, key=lambda item: item[0])[1] if possible else None


def extract_standings(standing: list[dict], home_id: int, away_id: int):
    out = {}
    if not standing:
        return out
    try:
        table = standing[0]["league"]["standings"][0]
        for r in table:
            tid = (r.get("team") or {}).get("id")
            if tid in (home_id, away_id):
                out[tid] = {"position": r.get("rank"), "points": r.get("points"),
                            "difference": r.get("goalsDiff"), "forme": r.get("form")}
    except (IndexError, KeyError, TypeError):
        pass
    return out


def extract_players(blocks: list[dict], team_id: int) -> list[dict]:
    players = []
    for block in blocks:
        if (block.get("team") or {}).get("id") != team_id:
            continue
        for item in block.get("players") or []:
            base = item.get("player") or {}
            stats = (item.get("statistics") or [{}])[0]
            games = stats.get("games") or {}
            mins = num(games.get("minutes")) or 0
            rating = num(games.get("rating"))
            if rating is None or mins < 45:
                continue
            players.append({"Joueur": base.get("name"), "Note du dernier match": rating,
                            "Minutes": int(mins), "Poste": games.get("position"),
                            "Buts": (stats.get("goals") or {}).get("total"),
                            "Tacles": (stats.get("tackles") or {}).get("total")})
    return sorted(players, key=lambda x: x["Note du dernier match"], reverse=True)


def analyze_automatic(key: str, home_name: str, away_name: str,
                      date_match: date, n_history: int, n_detailed: int,
                      include_players: bool) -> dict:
    h_candidates = fetch(key, "teams", search=home_name)
    a_candidates = fetch(key, "teams", search=away_name)
    hteam = match_team_name(home_name, h_candidates)
    ateam = match_team_name(away_name, a_candidates)
    if not hteam or not ateam:
        raise ApiError("Équipe introuvable : essaye son nom officiel (ex. Paris Saint Germain).")
    hid, aid = hteam["id"], ateam["id"]
    if hid == aid:
        raise ApiError("Les deux recherches ont trouvé le même club. Précise les noms.")

    fixture = discover_match(key, hid, aid, date_match)
    # If reversed, use actual venue. No fabricated home/away assignments.
    if fixture:
        ids = fixture.get("teams") or {}
        hid, aid = ids["home"]["id"], ids["away"]["id"]
        hteam, ateam = ids["home"], ids["away"]
    hname, aname = hteam["name"], ateam["name"]
    cutoff = None
    if fixture:
        cutoff = datetime.fromtimestamp(fixture["fixture"]["timestamp"], timezone.utc)
    raw_h = fetch(key, "fixtures", team=hid, last=n_history + 5)
    raw_a = fetch(key, "fixtures", team=aid, last=n_history + 5)
    hrows = extract_finished_matches(raw_h, hid, before=cutoff, limit=n_history)
    arows = extract_finished_matches(raw_a, aid, before=cutoff, limit=n_history)

    # Try same-league form if enough observations, else use recent games across competitions.
    lid = ((fixture or {}).get("league") or {}).get("id")
    hleague = [r for r in hrows if r["league_id"] == lid] if lid else []
    aleague = [r for r in arows if r["league_id"] == lid] if lid else []
    filtered_to_league = len(hleague) >= 5 and len(aleague) >= 5
    if filtered_to_league:
        hrows, arows = hleague, aleague

    stat_ids = list(dict.fromkeys([r["id"] for r in hrows[:n_detailed] + arows[:n_detailed]
                                  if r.get("id") is not None]))
    stats = {}
    stat_error = None
    for fid in stat_ids:
        try:
            stats[fid] = fetch(key, "fixtures/statistics", fixture=fid)
        except ApiError as exc:
            stat_error = str(exc)
            break
    enrich_stat_rows(hrows, stats, hid)
    enrich_stat_rows(arows, stats, aid)

    context: dict[str, Any] = {"fixture": fixture, "hteam": hteam, "ateam": ateam,
                                "league_only": filtered_to_league, "stats_error": stat_error,
                                "standings": {}, "injuries": [], "player_perf": {},
                                "context_errors": []}
    if fixture:
        season = (fixture.get("league") or {}).get("season")
        if season and lid:
            try:
                standings = fetch(key, "standings", league=lid, season=season)
                context["standings"] = extract_standings(standings, hid, aid)
            except ApiError as exc:
                context["context_errors"].append("Classement : " + str(exc))
        try:
            context["injuries"] = fetch(key, "injuries", fixture=fixture["fixture"]["id"])
        except ApiError as exc:
            context["context_errors"].append("Blessures : " + str(exc))
        # Lineups only near kickoff: calling early only wastes quota.
        if cutoff and 0 <= (cutoff - datetime.now(timezone.utc)).total_seconds() <= 7200:
            try:
                context["lineups"] = fetch(key, "fixtures/lineups", fixture=fixture["fixture"]["id"])
            except ApiError:
                context["lineups"] = []
    if include_players:
        for tid, rows in ((hid, hrows), (aid, arows)):
            if rows and rows[0].get("id"):
                try:
                    data = fetch(key, "fixtures/players", fixture=rows[0]["id"])
                    context["player_perf"][tid] = extract_players(data, tid)
                except ApiError as exc:
                    context["context_errors"].append("Joueurs : " + str(exc))

    model = rank_events(hname, aname, hrows, arows)
    return {"model": model, "context": context, "rows_h": hrows,
            "rows_a": arows, "hid": hid, "aid": aid,
            "mode": "API-Football (données récupérées en direct au lancement de l'analyse)"}


def manual_editor(key: str, n: int, default_home: bool) -> list[dict]:
    # Create only empty inputs; do not populate with invented match data.
    col = {"Match": [f"Match {i+1}" for i in range(n)],
           "À domicile": [default_home] * n,
           "Buts marqués": [None] * n, "Buts encaissés": [None] * n,
           "Cartons jaunes reçus": [None] * n, "Fautes commises": [None] * n}
    df = pd.DataFrame(col)
    return st.data_editor(df, key=key, use_container_width=True, hide_index=True,
                          column_config={
                            "Match": st.column_config.TextColumn(disabled=True),
                            "À domicile": st.column_config.CheckboxColumn(help="Coche si l'équipe jouait chez elle"),
                            "Buts marqués": st.column_config.NumberColumn(min_value=0, max_value=20, step=1),
                            "Buts encaissés": st.column_config.NumberColumn(min_value=0, max_value=20, step=1),
                            "Cartons jaunes reçus": st.column_config.NumberColumn(min_value=0, max_value=20, step=1),
                            "Fautes commises": st.column_config.NumberColumn(min_value=0, max_value=70, step=1),
                          }).to_dict("records")


def to_rows(raw: list[dict]):
    rows = []
    for row in raw:
        gf, ga = num(row.get("Buts marqués")), num(row.get("Buts encaissés"))
        if gf is None or ga is None:
            continue
        rows.append({"gf": int(gf), "ga": int(ga),
                     "home": bool(row.get("À domicile")),
                     "yellow": num(row.get("Cartons jaunes reçus")),
                     "fouls": num(row.get("Fautes commises")),
                     "date": None, "id": None})
    return rows


def summary_row(rows: list[dict]):
    return pd.DataFrame([{
        "Date": r["date"].strftime("%d/%m/%Y") if r.get("date") else "Non renseignée",
        "Domicile": "Oui" if r.get("home") else "Non",
        "Buts +": r["gf"], "Buts -": r["ga"],
        "Jaunes": r.get("yellow"), "Fautes": r.get("fouls"),
        "Adversaire": r.get("opponent", "—")
    } for r in rows])


def show_output(analysis: dict, threshold: int):
    result = analysis["model"]
    events = result["events"]
    st.divider()
    st.subheader(f"{result['home']} — {result['away']}")
    st.caption(analysis["mode"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Buts attendus domicile", f"{result['lambda_h']:.2f}")
    c2.metric("Buts attendus extérieur", f"{result['lambda_a']:.2f}")
    c3.metric("Matchs étudiés — domicile", result["n_h"])
    c4.metric("Matchs étudiés — extérieur", result["n_a"])
    robust = [x for x in events if x["Plancher de sensibilité"] >= threshold]
    frag = [x for x in events if x["Probabilité estimée"] >= threshold
            and x["Plancher de sensibilité"] < threshold]
    st.markdown(f"### Événements ≥ {threshold} % dans le test de sensibilité")
    st.caption("Le plancher est le minimum du modèle sous des scénarios de taux plus hauts ou plus bas. Ce n'est PAS une borne de confiance statistique ni une garantie.")
    if robust:
        st.dataframe(pd.DataFrame(robust).drop(columns="Analyse robuste ≥90 %"),
                     use_container_width=True, hide_index=True)
    else:
        st.info("Aucun événement ne passe ce filtre conservateur avec les informations fournies.")
    with st.expander(f"Estimés ≥ {threshold} % mais sensibles aux hypothèses ({len(frag)})"):
        if frag:
            st.dataframe(pd.DataFrame(frag).drop(columns="Analyse robuste ≥90 %"),
                         use_container_width=True, hide_index=True)
        else:
            st.write("Aucun.")
    with st.expander("Voir tous les événements et probabilités estimées"):
        st.dataframe(pd.DataFrame(events).drop(columns="Analyse robuste ≥90 %"),
                     use_container_width=True, hide_index=True)
    st.download_button("Exporter les probabilités (CSV)",
                       data=pd.DataFrame(events).to_csv(index=False).encode("utf-8-sig"),
                       file_name="footpredict90_probabilites.csv", mime="text/csv")

    st.subheader("Contexte documenté")
    hform, aform = result["form_h"], result["form_a"]
    if hform.get("n") and aform.get("n"):
        left, right = st.columns(2)
        with left:
            st.write(f"**{result['home']}** — 5 derniers matchs : {hform['w']}V / {hform['d']}N / {hform['l']}D. Buts : {hform['gf']} marqués, {hform['ga']} encaissés en moyenne.")
        with right:
            st.write(f"**{result['away']}** — 5 derniers matchs : {aform['w']}V / {aform['d']}N / {aform['l']}D. Buts : {aform['gf']} marqués, {aform['ga']} encaissés en moyenne.")
    context = analysis.get("context", {})
    fixture = context.get("fixture")
    if fixture:
        st.write(f"**Match repéré :** {date_label(fixture)} · {(fixture.get('league') or {}).get('name', '?')} · {((fixture.get('league') or {}).get('round') or 'Journée inconnue')}.")
        referee = ((fixture.get("fixture") or {}).get("referee"))
        st.write("**Arbitre :** " + (referee if referee else "non renseigné"))
        standings = context.get("standings") or {}
        for team in (context.get("hteam") or {}, context.get("ateam") or {}):
            line = standings.get(team.get("id"))
            if line:
                st.write(f"**{team.get('name')} :** {line['position']}e · {line['points']} points · diff. {line['difference']} · forme {line['forme']}")
        injuries = context.get("injuries") or []
        if injuries:
            st.write("**Indisponibilités signalées pour ce match :**")
            for injury in injuries[:14]:
                player = (injury.get("player") or {}).get("name", "Joueur inconnu")
                side = (injury.get("team") or {}).get("name", "Équipe inconnue")
                st.write(f"- {side} : {player} — {injury.get('type','?')} ({injury.get('reason','non précisé')})")
        else:
            st.caption("Blessures : aucune entrée renvoyée. Cela ne signifie pas que les effectifs sont complets.")
        lineups = context.get("lineups", [])
        if lineups:
            st.write("**Compositions enregistrées :**")
            for block in lineups:
                roster = ", ".join((r.get("player") or {}).get("name", "?") for r in block.get("startXI", []))
                st.write(f"- {(block.get('team') or {}).get('name')}, {block.get('formation','?')} : {roster}")
    else:
        st.caption("Aucun match futur correspondant retrouvé dans le flux : pas de date, classement ou blessure spécifiques vérifiés pour cette confrontation.")
    for team_id, players in (context.get("player_perf") or {}).items():
        club = (context.get("hteam") or {}).get("name") if team_id == analysis.get("hid") else (context.get("ateam") or {}).get("name")
        if players:
            st.write(f"**Performances individuelles : {club} (dernier match connu uniquement)**")
            st.dataframe(pd.DataFrame(players[:7]), use_container_width=True, hide_index=True)
    for msg in context.get("context_errors", []):
        st.caption(msg)
    if context.get("stats_error"):
        st.warning("Cartons/fautes incomplets : " + context["stats_error"])
    if "Cartons" not in result["details"] or "Fautes" not in result["details"]:
        st.warning("Cartons et/ou fautes non calculés : il faut au moins 5 observations valides par équipe pour chaque catégorie.")

    with st.expander("Données historiques exploitées"):
        a, b = st.columns(2)
        with a:
            st.write(result["home"])
            st.dataframe(summary_row(analysis["rows_h"]), use_container_width=True, hide_index=True)
        with b:
            st.write(result["away"])
            st.dataframe(summary_row(analysis["rows_a"]), use_container_width=True, hide_index=True)
    st.caption("Méthode : Poisson indépendant pour les buts, moyennes récentes pondérées par domicile/extérieur, dispersion négative-binomiale pour cartons/fautes. Test de sensibilité, mais pas de calibration historique. Aucun contexte tactique ou individuel absent des données n'est inventé. Un événement très probable peut avoir une cote économiquement défavorable. Ne place aucun pari uniquement sur ce tableau.")


st.title("⚽ FootPredict 90")
st.write("Deux équipes → données disponibles → probabilités estimées → filtre prudent. **Prototype de recherche statistique, sans pari automatique.**")
mode = st.radio("Source des données", ["Automatique (API-Football)", "Saisie manuelle"], horizontal=True)
threshold = st.slider("Afficher les événements à partir de", 80, 99, 90, 1)

if mode.startswith("Automatique"):
    with st.form("auto_form"):
        key = st.text_input("Clé personnelle API-Football", type="password", help="Inscription gratuite : api-football.com. N'envoie ta clé à personne.")
        c1, c2 = st.columns(2)
        home_name = c1.text_input("Équipe 1 (domicile si connue)", placeholder="Arsenal")
        away_name = c2.text_input("Équipe 2 (extérieur si connue)", placeholder="Chelsea")
        d = st.date_input("Date estimée du match", value=date.today())
        n_history = st.select_slider("Nombre maximal de matchs récents", options=[5, 8, 12, 15], value=12)
        n_detailed = st.select_slider("Nombre de matchs pour récupérer cartons/fautes (par équipe)",
                                      options=[0, 5, 8], value=5)
        include_players = st.checkbox("Examiner les notes des joueurs du dernier match (+2 requêtes)", value=False)
        submitted = st.form_submit_button("Analyser la rencontre", use_container_width=True, type="primary")
    if submitted:
        if not key.strip() or len(home_name.strip()) < 3 or len(away_name.strip()) < 3:
            st.error("Renseigne ta clé API et deux noms d'équipes d'au moins trois caractères.")
        else:
            try:
                with st.spinner("Collecte et calcul des événements ..."):
                    st.session_state.analysis = analyze_automatic(key.strip(), home_name.strip(),
                                                                  away_name.strip(), d, n_history,
                                                                  n_detailed, include_players)
            except (ApiError, ValueError) as exc:
                st.session_state.pop("analysis", None)
                st.error(str(exc))
else:
    st.write("Renseigne les matchs récents dans l'ordre **du plus récent au plus ancien**. Les cellules vides sont ignorées, jamais remplacées par des chiffres inventés.")
    c1, c2 = st.columns(2)
    hname = c1.text_input("Équipe à domicile", value="Équipe A")
    aname = c2.text_input("Équipe à l'extérieur", value="Équipe B")
    nmanual = st.select_slider("Lignes par équipe", options=[5, 8, 12, 15], value=5)
    with st.form("manual_form"):
        st.markdown(f"#### {hname}")
        hraw = manual_editor("h_edit", nmanual, True)
        st.markdown(f"#### {aname}")
        araw = manual_editor("a_edit", nmanual, False)
        submitted = st.form_submit_button("Calculer les probabilités", use_container_width=True, type="primary")
    if submitted:
        hr, ar = to_rows(hraw), to_rows(araw)
        try:
            st.session_state.analysis = {
                "model": rank_events(hname, aname, hr, ar),
                "context": {}, "rows_h": hr, "rows_a": ar,
                "mode": "Saisie manuelle — données non vérifiées",
            }
        except ValueError as exc:
            st.session_state.pop("analysis", None)
            st.error(str(exc))

if st.session_state.get("analysis"):
    show_output(st.session_state.analysis, threshold)

st.divider()
st.caption("Source en mode automatique : API-Football, sous réserve du plan, de la couverture et de la fraîcheur des données. Les clés restent personnelles. Le modèle n'a pas été rétrospectivement calibré et ne doit pas être présenté comme fiable à 90 %.")
