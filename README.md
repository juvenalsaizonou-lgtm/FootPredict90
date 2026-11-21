# ⚽ FootPredict 90 — prototype d'analyse de football

**Ce que l'outil fait :** tu saisis deux équipes et ta clé personnelle API-Football ; l'application cherche un match à venir, récupère les résultats récents, les classements, les blessures déclarées, les fautes et les cartons disponibles, puis calcule les probabilités estimées des événements proposés.

**Autre possibilité :** tu saisis toi-même les 5 à 15 derniers matchs de chaque équipe dans deux tableaux, sans compte API.

## Utilisation gratuite en ligne, depuis un téléphone

1. Décompresse `FootPredict90.zip`.
2. Crée un dépôt GitHub sur https://github.com/new et téléverse **`app.py`, `core.py`, `requirements.txt`** à la racine (ne téléverse aucune clé API).
3. Connecte GitHub à **https://share.streamlit.io/** et sélectionne « Deploy an app ».
4. Choisis le dépôt, la branche, et le fichier principal **`app.py`**, puis publie l'application.
5. Sur https://dashboard.api-football.com/register crée un compte gratuit et récupère ta clé API personnelle.
6. Dans l'application, choisis **Automatique**, saisis la clé, les noms des équipes et la date approximative, puis appuie sur **Analyser la rencontre**.

Cette version utilise les paramètres `from` et `to` à la place de `next` et `last`, refusés par certaines offres gratuites. Si les dates ou saisons restent bloquées, utilise le mode manuel.

La version gratuite d'API-Football est limitée à 100 requêtes/jour et à certaines saisons. Elle **peut ne pas contenir les données de la saison en cours** ; dans ce cas, le mode manuel reste utilisable. Les cartes, fautes, compositions et blessures ne sont pas disponibles pour tous les matchs. Le serveur ne devine jamais les statistiques manquantes.

Guide officiel Streamlit : https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Utilisation sur ordinateur

Avec Python 3.10 ou plus :

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Une page web locale s'ouvre dans ton navigateur.

## Méthode et limites

- **Buts :** moyenne des performances offensives et défensives récentes, correction domicile/extérieur, lissage vers des références génériques *indicatives*, distribution de Poisson. Les valeurs 1,35/1,10 ne sont pas des moyennes actuelles de la ligue.
- **Cartons/fautes :** répartition par équipe basée sur des observations historiques ; approximation de Poisson ou négative binomiale en présence de surdispersion. Minimum 5 matchs complets par équipe et par métrique.
- **Mi-temps :** approximation de deux périodes indépendantes de 45 minutes.
- **Filtres :** une estimation centrale et un « plancher de sensibilité » avec scénarios de taux plus/moins élevés. **Ce plancher n'est pas un intervalle de confiance ni une probabilité garantie.**
- **Enjeux :** affiche le classement, les points, l'arbitre et les blessures quand les sources les fournissent. Ne déduit ni motivation ni duel tactique sans données spécifiques.
- **Joueurs :** mode avancé : notes du dernier match connu, éventuellement blessures et compositions publiées ; une seule note n'est pas une vraie prévision de performance contre un adversaire.
- Les événements classés « ≥ 90 % » n'ont **pas** été validés par backtesting/calibration. Ne les interprète pas comme une fréquence empirique observée.
- Ce prototype **ne compare pas** encore les probabilités aux cotes et ne détecte pas de valeur attendue positive. Un pari très probable peut être non rentable ; des résultats passés ne garantissent rien.
- Il n'effectue **aucune** mise, connexion à un bookmaker ou automatisation de paris.

## Tests du moteur

```bash
pip install pytest
pytest -q
```

### Sources

- https://www.api-football.com/documentation-v3
- https://www.api-football.com/pricing/
- https://docs.streamlit.io/deploy/streamlit-community-cloud

Prototype technique créé le 21 novembre 2026.
