# =============================================================================
# app.py — Dashboard Streamlit
# Projet ML Immobilier — Maine-et-Loire (Dept. 49)
# Equipe : Arthur, Mandengue, Moneli
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
import os
import sys

# On ajoute le dossier du projet au path pour pouvoir importer config.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# Imports scikit-learn pour les modeles et metriques
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =============================================================================
# CONFIGURATION DE LA PAGE
# =============================================================================

st.set_page_config(
    page_title="Immobilier 49 — ML",
    page_icon="🏠",
    layout="wide"
)

# Style CSS personnalise pour le dashboard
st.markdown("""
<style>
    .stApp { background-color: #F8F7F4; }

    /* Sidebar foncee */
    section[data-testid="stSidebar"] { background-color: #0F2A4A; }
   section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] * { color: #0F2A4A !important; }

    /* Titres de section avec barre verte */
    .titre-section {
        font-size: 1.2rem;
        font-weight: 600;
        color: #0F2A4A;
        border-bottom: 2px solid #0D9488;
        padding-bottom: 6px;
        margin: 1.5rem 0 1rem;
    }

    /* Cartes metriques */
    div[data-testid="stMetric"] {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        border: 0.5px solid #E2E8F0;
        color: #0F2A4A !important;
    }
    div[data-testid="stMetric"] * {
        color: #0F2A4A !important;
    }

    /* Boite de prediction */
    .pred-result {
        background: #0F2A4A;
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
        color: white;
    }
    .pred-prix { font-size: 2.5rem; font-weight: 700; margin: 0.5rem 0; }
    .pred-m2   { font-size: 1.1rem; color: #5DCAA5; margin: 0; }
    .pred-sub  { font-size: 0.85rem; color: rgba(255,255,255,0.6); margin: 0.5rem 0 0; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CHARGEMENT DES DONNEES ET DU MODELE
# =============================================================================

# @st.cache_data : les donnees sont chargees une seule fois puis mises en cache
# Evite de recharger les CSV a chaque interaction de l'utilisateur
@st.cache_data
def load_data():
    """Charge et nettoie les donnees 2024 + 2025."""
    df24 = pd.read_csv(config.CSV_2024)
    df25 = pd.read_csv(config.CSV_2025)

    # Fusion des deux annees
    df = pd.concat([df24, df25], ignore_index=True)

    # On garde uniquement Maison et Appartement
    df = df[df[config.COL_TYPE].isin(config.TYPES_BIENS)].copy()

    # Suppression des lignes sans prix ou sans surface
    df = df.dropna(subset=[config.COL_PRIX, config.COL_SURFACE])

    # Extraction de l'annee et du mois depuis la date
    df['date_mutation'] = pd.to_datetime(df[config.COL_DATE])
    df['annee'] = df['date_mutation'].dt.year
    df['mois']  = df['date_mutation'].dt.month

    # Calcul du prix au m²
    df['prix_m2'] = df[config.COL_PRIX] / df[config.COL_SURFACE]

    # Suppression des valeurs aberrantes (prix/m² hors norme)
    df = df[(df['prix_m2'] > 200) & (df['prix_m2'] < 15000)]
    df = df[df[config.COL_SURFACE] < 1000]
    df = df[df[config.COL_PRIX] > 5000]

    return df

# @st.cache_resource : le modele est charge une seule fois (objet lourd)
@st.cache_resource
def load_model():
    """Charge le modele Random Forest entraine par Arthur."""
    with open(config.MODEL_OBJ1, 'rb') as f:
        return pickle.load(f)

# Chargement effectif des donnees et du modele
df    = load_data()
model = load_model()

# =============================================================================
# PREPARATION COMMUNE AUX MODELES
# =============================================================================

# Encodage du type de bien en numerique (Maison=1, Appartement=0)
le = LabelEncoder()
df['type_encode'] = le.fit_transform(df[config.COL_TYPE])

# Features utilisees par le modele (recuperees depuis le modele lui-meme)
FEATURES = list(model.feature_names_in_)

# Preparation des X et y pour evaluation
X = df[FEATURES].fillna(0)
y = df[config.COL_PRIX]

# Split train/test 80/20 avec random_state fixe pour reproductibilite
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Couleurs par type de bien (utilisees dans tous les graphes)
COULEURS = {'Maison': '#0D9488', 'Appartement': '#6366F1'}

# =============================================================================
# SIDEBAR — NAVIGATION
# =============================================================================

with st.sidebar:
    st.markdown("### Immobilier 49")
    st.markdown("---")

    # Menu de navigation entre les pages
    page = st.radio(
        "Navigation",
        [
            "Introduction",
            "Exploration des donnees",
            "Comparaison des modeles",
            "Prediction interactive",
            "Evolution 2024 - 2026"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Equipe**")
    st.markdown("Arthur · Mandengue · Moneli")
    st.markdown("---")
    st.markdown(f"**{len(df):,}** transactions")
    st.markdown("DVF open data — Dept. 49")

# =============================================================================
# PAGE 1 — INTRODUCTION
# =============================================================================

if page == "Introduction":

    st.title("Prediction des prix immobiliers")
    st.markdown("**Maine-et-Loire (Dept. 49) — Machine Learning sur donnees DVF**")
    st.markdown("---")

    # Metriques cles en haut de page
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions",        f"{len(df):,}")
    col2.metric("Annees couvertes",    "2024 + 2025")
    col3.metric("Annee predite",       "2026")
    col4.metric("Modele principal",    "Random Forest")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="titre-section">Contexte du projet</p>', unsafe_allow_html=True)
        st.markdown("""
        Le marche immobilier du Maine-et-Loire connait des evolutions constantes.
        Anticiper les prix permet aux acheteurs et investisseurs de mieux se positionner.

        Ce projet utilise les **Demandes de Valeurs Foncieres (DVF)**, open data
        gouvernemental, pour entrainer un modele capable de predire les prix de
        vente en **2026**.
        """)

        st.markdown('<p class="titre-section">Pipeline du projet</p>', unsafe_allow_html=True)
        st.markdown("""
        1. **Collecte** — DVF open data, 64 000 transactions 2024 + 2025
        2. **Nettoyage** — suppression doublons, filtres aberrants
        3. **Feature engineering** — prix/m², encodage type, annee/mois
        4. **Modelisation** — Regression Lineaire vs Random Forest
        5. **Prediction** — estimation des prix 2026 par type de bien
        """)

    with col2:
        st.markdown('<p class="titre-section">Repartition de l\'equipe</p>', unsafe_allow_html=True)

        # Tableau de repartition des taches
        equipe = pd.DataFrame({
            "Membre":   ["Arthur",     "Mandengue",            "Moneli"],
            "Role":     ["Fusion + nettoyage + modele Obj.1",
                         "EDA + features + modele Obj.2",
                         "Scenarios 2026 + predictions + visualisations"],
            "Livrable": ["data_clean.csv + model_obj1.pkl",
                         "features_list.py + model_obj2.pkl",
                         "predictions_2026.csv + graphes"]
        })
        st.dataframe(equipe, hide_index=True, use_container_width=True)

        st.markdown('<p class="titre-section">Repartition des biens</p>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        col_a.metric("Maisons",       f"{df[df[config.COL_TYPE]=='Maison'].shape[0]:,}")
        col_b.metric("Appartements",  f"{df[df[config.COL_TYPE]=='Appartement'].shape[0]:,}")

# =============================================================================
# PAGE 2 — EXPLORATION DES DONNEES (EDA)
# =============================================================================

elif page == "Exploration des donnees":

    st.title("Exploration des donnees")
    st.markdown("Analyse exploratoire avant la modelisation.")
    st.markdown("---")

    # Filtre type de bien
    filtre = st.selectbox("Filtrer par type de bien", ["Tous", "Maison", "Appartement"])
    df_eda = df if filtre == "Tous" else df[df[config.COL_TYPE] == filtre]

    # Metriques descriptives
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Prix median",     f"{df_eda[config.COL_PRIX].median():,.0f} €")
    col2.metric("Prix moyen",      f"{df_eda[config.COL_PRIX].mean():,.0f} €")
    col3.metric("Surface mediane", f"{df_eda[config.COL_SURFACE].median():.0f} m²")
    col4.metric("Prix/m² median",  f"{df_eda['prix_m2'].median():,.0f} €/m²")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="titre-section">Distribution des prix</p>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        for t in config.TYPES_BIENS:
            # Histogramme superpose par type de bien
            data = df[df[config.COL_TYPE] == t][config.COL_PRIX]
            ax.hist(data, bins=40, alpha=0.65, color=COULEURS[t],
                    label=t, edgecolor='white', linewidth=0.3)
            # Ligne verticale sur la mediane
            ax.axvline(data.median(), color=COULEURS[t],
                       linestyle='--', linewidth=1.5, alpha=0.8)

        ax.set_xlabel("Prix (€)", fontsize=11)
        ax.set_ylabel("Nombre de ventes", fontsize=11)
        ax.legend(fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#F8F7F4')
        fig.patch.set_facecolor('#F8F7F4')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<p class="titre-section">Prix au m² — boxplot</p>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        # Boxplot pour voir la dispersion du prix/m² par type
        data_box = [
            df[df[config.COL_TYPE] == 'Maison']['prix_m2'].dropna(),
            df[df[config.COL_TYPE] == 'Appartement']['prix_m2'].dropna()
        ]
        bp = ax.boxplot(data_box, labels=['Maison', 'Appartement'],
                        patch_artist=True, widths=0.5,
                        medianprops=dict(color='white', linewidth=2))
        bp['boxes'][0].set_facecolor('#0D9488')
        bp['boxes'][1].set_facecolor('#6366F1')
        for w in bp['whiskers']:
            w.set(color='#64748B', linewidth=1)

        ax.set_ylabel("Prix au m² (€/m²)", fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#F8F7F4')
        fig.patch.set_facecolor('#F8F7F4')
        st.pyplot(fig)
        plt.close()

    st.markdown('<p class="titre-section">Matrice de correlation</p>', unsafe_allow_html=True)
    st.markdown("Montre quelles variables sont le plus liees au prix.")

    # Correlation entre toutes les variables numeriques cles
    cols_corr = [config.COL_PRIX, config.COL_SURFACE, config.COL_PIECES,
                 'prix_m2', 'annee', 'mois']
    corr = df[cols_corr].corr()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, square=True, ax=ax, linewidths=0.5)
    ax.set_facecolor('#F8F7F4')
    fig.patch.set_facecolor('#F8F7F4')
    st.pyplot(fig)
    plt.close()

# =============================================================================
# PAGE 3 — COMPARAISON DES MODELES
# =============================================================================

elif page == "Comparaison des modeles":

    st.title("Comparaison des modeles")
    st.markdown("Regression Lineaire vs Random Forest — qui predit le mieux ?")
    st.markdown("---")

    # --- Regression Lineaire ---
    # Modele simple : trouve la droite qui minimise l'erreur quadratique
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # --- Random Forest ---
    # Ensemble de 100 arbres de decision — plus robuste aux non-linearites
    y_pred_rf = model.predict(X_test)

    # Calcul des metriques pour les deux modeles
    def get_metrics(y_true, y_pred):
        """Calcule R², MAE et RMSE."""
        r2   = r2_score(y_true, y_pred)
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return r2, mae, rmse

    r2_lr, mae_lr, rmse_lr = get_metrics(y_test, y_pred_lr)
    r2_rf, mae_rf, rmse_rf = get_metrics(y_test, y_pred_rf)

    # Tableau comparatif des metriques
    st.markdown('<p class="titre-section">Metriques comparatives</p>', unsafe_allow_html=True)

    comparatif = pd.DataFrame({
        "Modele":  ["Regression Lineaire", "Random Forest"],
        "R²":      [f"{r2_lr:.3f}",        f"{r2_rf:.3f}"],
        "MAE (€)": [f"{mae_lr:,.0f}",      f"{mae_rf:,.0f}"],
        "RMSE (€)":[f"{rmse_lr:,.0f}",     f"{rmse_rf:,.0f}"],
        "Conclusion": ["Score trop faible", "Modele retenu"]
    })
    st.dataframe(comparatif, hide_index=True, use_container_width=True)

    st.markdown("""
    - **R²** : part de variance expliquee (1.0 = parfait, 0 = nul)
    - **MAE** : erreur moyenne en euros (plus c'est bas, mieux c'est)
    - **RMSE** : penalise davantage les grandes erreurs
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="titre-section">Regression Lineaire — Reel vs Predit</p>',
                    unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(6, 5))
        lim = float(max(y_test.max(), y_pred_lr.max()))
        # Nuage de points : si le modele etait parfait, tous les points
        # seraient sur la droite rouge (y = x)
        ax.scatter(y_test, y_pred_lr, alpha=0.15, s=8, color='#64748B')
        ax.plot([0, lim], [0, lim], 'r--', linewidth=1.5, label='Prediction parfaite')
        ax.set_xlabel("Prix reel (€)", fontsize=10)
        ax.set_ylabel("Prix predit (€)", fontsize=10)
        ax.set_title(f"R² = {r2_lr:.3f}", fontsize=11)
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#F8F7F4')
        fig.patch.set_facecolor('#F8F7F4')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<p class="titre-section">Random Forest — Reel vs Predit</p>',
                    unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(6, 5))
        lim = float(max(y_test.max(), y_pred_rf.max()))
        ax.scatter(y_test, y_pred_rf, alpha=0.15, s=8, color='#0D9488')
        ax.plot([0, lim], [0, lim], 'r--', linewidth=1.5, label='Prediction parfaite')
        ax.set_xlabel("Prix reel (€)", fontsize=10)
        ax.set_ylabel("Prix predit (€)", fontsize=10)
        ax.set_title(f"R² = {r2_rf:.3f}", fontsize=11)
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#F8F7F4')
        fig.patch.set_facecolor('#F8F7F4')
        st.pyplot(fig)
        plt.close()

    # Importance des features : quelle variable le Random Forest utilise-t-il
    # le plus pour prendre ses decisions ?
    st.markdown('<p class="titre-section">Importance des variables — Random Forest</p>',
                unsafe_allow_html=True)
    st.markdown("Plus la barre est longue, plus la variable influence la prediction du prix.")

    importances = pd.Series(
        model.feature_importances_, index=FEATURES
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    importances.plot(kind='barh', ax=ax, color='#0D9488')
    ax.set_xlabel("Importance relative", fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#F8F7F4')
    fig.patch.set_facecolor('#F8F7F4')
    st.pyplot(fig)
    plt.close()

# =============================================================================
# PAGE 4 — PREDICTION INTERACTIVE
# =============================================================================

elif page == "Prediction interactive":

    st.title("Simulateur de prix 2026")
    st.markdown("Entre les caracteristiques d'un bien pour obtenir une estimation.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        # Inputs utilisateur
        type_bien = st.selectbox("Type de bien", ["Maison", "Appartement"])
        surface   = st.slider("Surface habitable (m²)", 20, 300, 90)
        pieces    = st.slider("Nombre de pieces",        1,  10,   4)

        # Le terrain n'est pertinent que pour une maison
        terrain = 0
        if type_bien == "Maison":
            terrain = st.slider("Surface terrain (m²)", 0, 2000, 300)

        latitude  = st.number_input("Latitude",  value=47.478419, format="%.6f")
        longitude = st.number_input("Longitude", value=-0.563166, format="%.6f")

        st.caption("Coordonnees par defaut : centre d'Angers")

    with col2:
        # Construction du vecteur de features pour la prediction
        X_input = pd.DataFrame([{
            'surface_reelle_bati':       surface,
            'nombre_pieces_principales': pieces,
            'surface_terrain':           terrain,
            'latitude':                  latitude,
            'longitude':                 longitude,
            'type_encode':               1 if type_bien == "Maison" else 0,
            'annee':                     2026,  # On predit pour 2026
            'mois':                      6,     # Milieu de l'annee
        }])[FEATURES]  # On reordonne dans l'ordre exact du modele

        # Prediction du prix
        prix_predit = model.predict(X_input)[0]
        prix_m2_predit = prix_predit / surface

        # Affichage du resultat
        st.markdown(f"""
        <div class="pred-result">
            <p class="pred-sub">Estimation pour 2026</p>
            <p class="pred-prix">{prix_predit:,.0f} €</p>
            <p class="pred-m2">{prix_m2_predit:,.0f} €/m²</p>
            <p class="pred-sub">{type_bien} · {surface} m² · {pieces} pieces</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # Comparaison avec le prix moyen actuel (toutes annees confondues)
        prix_moyen_actuel = df[df[config.COL_TYPE] == type_bien][config.COL_PRIX].mean()
        prix_m2_actuel    = df[df[config.COL_TYPE] == type_bien]['prix_m2'].mean()

        diff_prix = ((prix_predit - prix_moyen_actuel) / prix_moyen_actuel) * 100
        diff_m2   = ((prix_m2_predit - prix_m2_actuel) / prix_m2_actuel) * 100

        col_a, col_b = st.columns(2)
        # La fleche verte/rouge indique si le prix 2026 est au-dessus ou
        # en-dessous de la moyenne actuelle
        col_a.metric("vs prix moyen actuel",
                     f"{prix_moyen_actuel:,.0f} €", f"{diff_prix:+.1f}%")
        col_b.metric("vs prix/m² actuel",
                     f"{prix_m2_actuel:,.0f} €/m²", f"{diff_m2:+.1f}%")

# =============================================================================
# PAGE 5 — EVOLUTION 2024 → 2025 → 2026
# =============================================================================

elif page == "Evolution 2024 - 2026":

    st.title("Evolution des prix 2024 → 2025 → 2026")
    st.markdown("Donnees reelles 2024-2025 + prediction ML pour 2026.")
    st.markdown("---")

    # Calcul des stats reelles par type de bien et par annee
    stats = df.groupby([config.COL_TYPE, 'annee']).agg(
        prix_moyen = (config.COL_PRIX, 'mean'),
        prix_m2    = ('prix_m2',       'mean'),
        nb_ventes  = (config.COL_PRIX, 'count')
    ).round(0).reset_index()

    # Fonction utilitaire pour recuperer une valeur sans planter si absente
    def safe_get(type_bien, annee, col='prix_moyen'):
        row = stats[
            (stats[config.COL_TYPE] == type_bien) &
            (stats['annee'] == annee)
        ]
        return float(row[col].values[0]) if len(row) > 0 else None

    # Chargement des predictions 2026 (generees par moneli_predictions.ipynb)
    try:
        preds       = pd.read_csv(config.PREDICTIONS)
        pred_maison = preds[preds[config.COL_TYPE] == 'Maison']['prix_predit'].mean()
        pred_appart = preds[preds[config.COL_TYPE] == 'Appartement']['prix_predit'].mean()
        pred_ok     = True
    except Exception:
        # Si le fichier n'existe pas encore, on calcule une projection simple
        pred_ok     = False
        pred_maison = None
        pred_appart = None

    col1, col2 = st.columns(2)

    # ── Graphe prix moyen ──────────────────────────────────────────────────
    with col1:
        st.markdown('<p class="titre-section">Prix moyen par type de bien</p>',
                    unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(7, 5))
        patches = []

        for type_bien in config.TYPES_BIENS:
            c    = COULEURS[type_bien]
            v24  = safe_get(type_bien, 2024)
            v25  = safe_get(type_bien, 2025)
            v26  = pred_maison if type_bien == 'Maison' else pred_appart

            # Construction des listes annees/valeurs en ignorant les None
            ann_reels = [a for a, v in zip([2024, 2025], [v24, v25]) if v is not None]
            val_reels = [v for v in [v24, v25] if v is not None]

            if ann_reels:
                # Trait plein = donnees reelles
                ax.plot(ann_reels, val_reels, color=c, linewidth=2.5,
                        marker='o', markersize=9, label=type_bien)
                for x, y in zip(ann_reels, val_reels):
                    ax.annotate(f'{y:,.0f} €', (x, y),
                                textcoords='offset points', xytext=(0, 12),
                                ha='center', fontsize=9, color=c)

            if v26 and ann_reels:
                # Pointilles = prediction 2026
                ax.plot([ann_reels[-1], 2026], [val_reels[-1], v26],
                        color=c, linewidth=2.5, linestyle='--',
                        marker='o', markersize=9,
                        markerfacecolor='white', markeredgewidth=2)
                ax.annotate(f'{v26:,.0f} €\n(ML)', (2026, v26),
                            textcoords='offset points', xytext=(0, 12),
                            ha='center', fontsize=9, color=c, fontweight='bold')

            patches.append(mpatches.Patch(color=c, label=type_bien))

        # Zone grisee pour la partie prediction
        ax.axvspan(2025.5, 2026.4, alpha=0.07, color='gray')
        ax.text(2026, ax.get_ylim()[0], 'prediction', ha='center',
                fontsize=8, color='gray', style='italic')

        ligne_plein   = plt.Line2D([0],[0], color='gray', lw=2,
                                   label='Donnees reelles')
        ligne_pointil = plt.Line2D([0],[0], color='gray', lw=2,
                                   linestyle='--', label='Prediction ML')
        ax.legend(handles=patches + [ligne_plein, ligne_pointil],
                  loc='upper left', fontsize=9)
        ax.set_xticks([2024, 2025, 2026])
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#F8F7F4')
        fig.patch.set_facecolor('#F8F7F4')
        st.pyplot(fig)
        plt.close()

    # ── Graphe prix au m² ─────────────────────────────────────────────────
    with col2:
        st.markdown('<p class="titre-section">Prix au m² par type de bien</p>',
                    unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(7, 5))

        for type_bien in config.TYPES_BIENS:
            c   = COULEURS[type_bien]
            v24 = safe_get(type_bien, 2024, 'prix_m2')
            v25 = safe_get(type_bien, 2025, 'prix_m2')

            ann_reels = [a for a, v in zip([2024, 2025], [v24, v25]) if v is not None]
            val_reels = [v for v in [v24, v25] if v is not None]

            if ann_reels:
                ax.plot(ann_reels, val_reels, color=c, linewidth=2.5,
                        marker='o', markersize=9, label=type_bien)
                for x, y in zip(ann_reels, val_reels):
                    ax.annotate(f'{y:,.0f} €/m²', (x, y),
                                textcoords='offset points', xytext=(0, 12),
                                ha='center', fontsize=9, color=c)

        ax.legend(fontsize=9)
        ax.set_xticks([2024, 2025])
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#F8F7F4')
        fig.patch.set_facecolor('#F8F7F4')
        st.pyplot(fig)
        plt.close()

    # ── Tableau recapitulatif ─────────────────────────────────────────────
    st.markdown('<p class="titre-section">Tableau recapitulatif</p>',
                unsafe_allow_html=True)

    st.dataframe(
        stats.rename(columns={
            config.COL_TYPE: 'Type',
            'annee':         'Annee',
            'prix_moyen':    'Prix moyen (€)',
            'prix_m2':       'Prix/m² (€)',
            'nb_ventes':     'Nb ventes'
        }),
        hide_index=True,
        use_container_width=True
    )