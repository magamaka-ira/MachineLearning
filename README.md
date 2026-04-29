# MachineLearning
# Prédiction des Prix Immobiliers — Maine-et-Loire (Dept. 49)

> Projet de Machine Learning sur les données DVF open data — Prédiction des prix de vente en 2026

---

## L'équipe

| Membre | Rôle | Livrables |
|---|---|---|
| **Arthur** | Fusion des données, nettoyage, modèle Objectif 1 | `data_clean.csv` · `model_obj1.pkl` |
| **Mandengue** | EDA, sélection des features, modèle Objectif 2 | `features_list.py` · `model_obj2.pkl` |
| **Moneli** | Scénarios 2026, prédictions, visualisations, dashboard | `X_2026.csv` · `predictions_2026.csv` · `app.py` |

---

## Objectif

À partir de **64 000 transactions immobilières réelles** du département 49 issues du DVF (Demandes de Valeurs Foncières), on entraîne des modèles de Machine Learning pour prédire l'évolution des prix en 2026, par type de bien (Maison / Appartement) et par zone géographique.

---

## Résultats du modèle

| Modèle | R² | MAE | RMSE |
|---|---|---|---|
| Régression Linéaire | 0.386 | ~70 000 € | ~95 000 € |
| **Random Forest** | **0.589** | **~51 769 €** | **~81 403 €** |

Le Random Forest est retenu comme modèle principal.

---

## Installation

```bash
pip install pandas numpy matplotlib scikit-learn seaborn streamlit
```

---

## Ordre d'exécution

> Tous les notebooks sont à la racine du dossier `ProjetImmo/`

**1. Arthur**
```bash
# Lance arthur_objectif1.ipynb
# Produit : data_clean.csv + model_obj1.pkl
```

**2. Mandengue** *(nécessite data_clean.csv d'Arthur)*
```bash
# Lance mandengue_objectif2.ipynb
# Produit : features_list.py + model_obj2.pkl
```

**3. Moneli** *(nécessite model_obj1.pkl d'Arthur)*
```bash
# Lance moneli_scenarios_2026.ipynb  → produit X_2026.csv
# Lance moneli_predictions.ipynb     → produit predictions_2026.csv + graphiques
```

**4. Dashboard**
```bash
python3 -m streamlit run app.py
```

---

## Structure du projet

```
ProjetImmo/
│
├── Données/
│   ├── 49_2024.csv              # Transactions DVF 2024 (47 962 lignes)
│   ├── 49_2025.csv              # Transactions DVF 2025 (16 141 lignes)
│   └── transactions.csv         # Données consolidées
│
├── Notebooks/
│   ├── arthur_objectif1.ipynb   # Fusion · Nettoyage · Modèle Obj.1
│   ├── mandengue_objectif2.ipynb# EDA · Features · Modèle Obj.2
│   ├── moneli_scenarios_2026.ipynb  # Génération des scénarios 2026
│   └── moneli_predictions.ipynb # Prédictions + visualisations finales
│
├── config.py                    # Chemins partagés (créé par Moneli)
├── app.py                       # Dashboard Streamlit (créé par Moneli)
│
├── data_clean.csv               # Données nettoyées (produit par Arthur)
├── model_obj1.pkl               # Modèle Random Forest (produit par Arthur)
├── model_obj2.pkl               # Modèle par type de bien (produit par Mandengue)
├── features_list.py             # Variables sélectionnées (produit par Mandengue)
├── X_2026.csv                   # Scénarios 2026 (produit par Moneli)
├── predictions_2026.csv         # Prix prédits 2026 (produit par Moneli)
│
└── README.md
```

---

## Dashboard Streamlit

Le dashboard contient 5 pages :

| Page | Contenu |
|---|---|
| **Introduction** | Contexte, chiffres clés, répartition de l'équipe |
| **Exploration des données** | Distribution des prix, boxplot, matrice de corrélation |
| **Comparaison des modèles** | R² · MAE · RMSE · Graphes réel vs prédit · Importance des features |
| **Prédiction interactive** | Simulateur de prix en temps réel avec sliders |
| **Évolution 2024-2026** | Courbe données réelles + prédiction ML |

---

## Données

Source : **DVF (Demandes de Valeurs Foncières)** — open data gouvernemental

Colonnes principales utilisées : `valeur_fonciere` · `surface_reelle_bati` · `nombre_pieces_principales` · `surface_terrain` · `type_local` · `latitude` · `longitude` · `date_mutation`

---

## Note technique

Le fichier `49_2025.csv` n'a pas de ligne d'en-tête. Il faut le charger avec :

```python
df24 = pd.read_csv(config.CSV_2024)
colonnes = list(df24.columns)
df25 = pd.read_csv(config.CSV_2025, header=None, names=colonnes)
```

---

*Projet réalisé dans le cadre d'un cours de Machine Learning — Arthur · Mandengue · Moneli*