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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

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

st.markdown("""
<style>
    .stApp { background-color: #F8F7F4; }

    /* Sidebar foncee — on cible uniquement les elements de la sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0F2A4A;
    }
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown strong {
        color: white !important;
    }

    /* Titres de section */
    .titre-section {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0F2A4A;
        border-bottom: 2px solid #0D9488;
        padding-bottom: 6px;
        margin: 1.5rem 0 1rem;
    }

    /* Boite de prediction */
    .pred-result {
        background: #0F2A4A;
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
    }
    .pred-titre { font-size: 0.9rem; color: rgba(255,255,255,0.6); margin: 0; }
    .pred-prix  { font-size: 2.5rem; font-weight: 700; color: white; margin: 0.4rem 0; }
    .pred-m2    { font-size: 1.1rem; color: #5DCAA5; margin: 0; }
    .pred-sub   { font-size: 0.85rem; color: rgba(255,255,255,0.5); margin: 0.5rem 0 0; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CHARGEMENT DES DONNEES ET DU MODELE
# =============================================================================

@st.cache_data
def load_data():
    df24 = pd.read_csv(config.CSV_2024)
    df25 = pd.read_csv(config.CSV_2025)
    df = pd.concat([df24, df25], ignore_index=True)
    df = df[df[config.COL_TYPE].isin(config.TYPES_BIENS)].copy()
    df = df.dropna(subset=[config.COL_PRIX, config.COL_SURFACE])
    df['date_mutation'] = pd.to_datetime(df[config.COL_DATE])
    df['annee'] = df['date_mutation'].dt.year
    df['mois']  = df['date_mutation'].dt.month
    df['prix_m2'] = df[config.COL_PRIX] / df[config.COL_SURFACE]
    df = df[(df['prix_m2'] > 200) & (df['prix_m2'] < 15000)]
    df = df[df[config.COL_SURFACE] < 1000]
    df = df[df[config.COL_PRIX] > 5000]
    return df

@st.cache_resource
def load_model():
    with open(config.MODEL_OBJ1, 'rb') as f:
        return pickle.load(f)

df    = load_data()
model = load_model()

le = LabelEncoder()
df['type_encode'] = le.fit_transform(df[config.COL_TYPE])

FEATURES = list(model.feature_names_in_)
X = df[FEATURES].fillna(0)
y = df[config.COL_PRIX]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

COULEURS = {'Maison': '#0D9488', 'Appartement': '#6366F1'}

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("### Immobilier 49")
    st.markdown("---")
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

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions",     f"{len(df):,}")
    col2.metric("Annees couvertes", "2024 + 2025")
    col3.metric("Annee predite",    "2026")
    col4.metric("Modele principal", "Random Forest")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="titre-section">Contexte</p>', unsafe_allow_html=True)
        st.markdown("""
        Le marche immobilier du Maine-et-Loire connait des evolutions constantes.
        Anticiper les prix permet aux acheteurs et investisseurs de mieux se positionner.

        Ce projet utilise les **Demandes de Valeurs Foncieres (DVF)**, open data
        gouvernemental, pour entrainer un modele capable de predire les prix de vente en 2026.
        """)

        st.markdown('<p class="titre-section">Pipeline</p>', unsafe_allow_html=True)
        st.markdown("""
        1. **Collecte** — DVF open data, 64 000 transactions 2024 + 2025
        2. **Nettoyage** — suppression doublons, filtres aberrants
        3. **Feature engineering** — prix/m², encodage type, annee/mois
        4. **Modelisation** — Regression Lineaire vs Random Forest
        5. **Prediction** — estimation des prix 2026 par type de bien
        """)

    with col2:
        st.markdown('<p class="titre-section">Repartition de l\'equipe</p>', unsafe_allow_html=True)
        equipe = pd.DataFrame({
            "Membre":   ["Arthur", "Mandengue", "Moneli"],
            "Role":     [
                "Fusion + nettoyage + modele Obj.1",
                "EDA + features + modele Obj.2",
                "Scenarios 2026 + predictions + visualisations"
            ],
            "Livrable": [
                "data_clean.csv + model_obj1.pkl",
                "features_list.py + model_obj2.pkl",
                "predictions_2026.csv + graphes"
            ]
        })
        st.dataframe(equipe, hide_index=True, use_container_width=True)

        st.markdown('<p class="titre-section">Repartition des biens</p>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        col_a.metric("Maisons",      f"{df[df[config.COL_TYPE]=='Maison'].shape[0]:,}")
        col_b.metric("Appartements", f"{df[df[config.COL_TYPE]=='Appartement'].shape[0]:,}")

# =============================================================================
# PAGE 2 — EDA
# =============================================================================

elif page == "Exploration des donnees":

    st.title("Exploration des donnees")
    st.markdown("---")

    filtre = st.selectbox("Filtrer par type de bien", ["Tous", "Maison", "Appartement"])
    df_eda = df if filtre == "Tous" else df[df[config.COL_TYPE] == filtre]

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
            data = df[df[config.COL_TYPE] == t][config.COL_PRIX]
            ax.hist(data, bins=40, alpha=0.65, color=COULEURS[t], label=t,
                    edgecolor='white', linewidth=0.3)
            ax.axvline(data.median(), color=COULEURS[t], linestyle='--',
                       linewidth=1.5, alpha=0.8)
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
    st.markdown("Regression Lineaire vs Random Forest")
    st.markdown("---")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_pred_rf = model.predict(X_test)

    def get_metrics(y_true, y_pred):
        return (
            r2_score(y_true, y_pred),
            mean_absolute_error(y_true, y_pred),
            np.sqrt(mean_squared_error(y_true, y_pred))
        )

    r2_lr, mae_lr, rmse_lr = get_metrics(y_test, y_pred_lr)
    r2_rf, mae_rf, rmse_rf = get_metrics(y_test, y_pred_rf)

    st.markdown('<p class="titre-section">Metriques comparatives</p>', unsafe_allow_html=True)
    comparatif = pd.DataFrame({
        "Modele":   ["Regression Lineaire", "Random Forest"],
        "R²":       [f"{r2_lr:.3f}",        f"{r2_rf:.3f}"],
        "MAE (€)":  [f"{mae_lr:,.0f}",      f"{mae_rf:,.0f}"],
        "RMSE (€)": [f"{rmse_lr:,.0f}",     f"{rmse_rf:,.0f}"],
        "Verdict":  ["Score insuffisant",    "Modele retenu"]
    })
    st.dataframe(comparatif, hide_index=True, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="titre-section">Regression Lineaire — Reel vs Predit</p>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        lim = float(max(y_test.max(), y_pred_lr.max()))
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

    st.markdown('<p class="titre-section">Importance des variables — Random Forest</p>',
                unsafe_allow_html=True)
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)
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
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        type_bien = st.selectbox("Type de bien", ["Maison", "Appartement"])
        surface   = st.slider("Surface habitable (m²)", 20, 300, 90)
        pieces    = st.slider("Nombre de pieces",        1,  10,   4)
        terrain   = st.slider("Surface terrain (m²)", 0, 2000, 300) if type_bien == "Maison" else 0
        latitude  = st.number_input("Latitude",  value=47.478419, format="%.6f")
        longitude = st.number_input("Longitude", value=-0.563166, format="%.6f")
        st.caption("Coordonnees par defaut : centre d'Angers")

    with col2:
        X_input = pd.DataFrame([{
            'surface_reelle_bati':       surface,
            'nombre_pieces_principales': pieces,
            'surface_terrain':           terrain,
            'latitude':                  latitude,
            'longitude':                 longitude,
            'type_encode':               1 if type_bien == "Maison" else 0,
            'annee':                     2026,
            'mois':                      6,
        }])[FEATURES]

        prix_predit    = model.predict(X_input)[0]
        prix_m2_predit = prix_predit / surface

        st.markdown(f"""
        <div class="pred-result">
            <p class="pred-titre">Estimation pour 2026</p>
            <p class="pred-prix">{prix_predit:,.0f} €</p>
            <p class="pred-m2">{prix_m2_predit:,.0f} €/m²</p>
            <p class="pred-sub">{type_bien} · {surface} m² · {pieces} pieces</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        prix_moyen = df[df[config.COL_TYPE] == type_bien][config.COL_PRIX].mean()
        prix_m2_moy = df[df[config.COL_TYPE] == type_bien]['prix_m2'].mean()
        diff      = ((prix_predit    - prix_moyen)   / prix_moyen)   * 100
        diff_m2   = ((prix_m2_predit - prix_m2_moy)  / prix_m2_moy)  * 100

        col_a, col_b = st.columns(2)
        col_a.metric("vs prix moyen actuel", f"{prix_moyen:,.0f} €",    f"{diff:+.1f}%")
        col_b.metric("vs prix/m² actuel",    f"{prix_m2_moy:,.0f} €/m²", f"{diff_m2:+.1f}%")

# =============================================================================
# PAGE 5 — EVOLUTION 2024 → 2026
# =============================================================================

elif page == "Evolution 2024 - 2026":

    st.title("Evolution des prix 2024 → 2025 → 2026")
    st.markdown("---")

    stats = df.groupby([config.COL_TYPE, 'annee']).agg(
        prix_moyen = (config.COL_PRIX, 'mean'),
        prix_m2    = ('prix_m2',       'mean'),
        nb_ventes  = (config.COL_PRIX, 'count')
    ).round(0).reset_index()

    def safe_get(type_bien, annee, col='prix_moyen'):
        row = stats[(stats[config.COL_TYPE] == type_bien) & (stats['annee'] == annee)]
        return float(row[col].values[0]) if len(row) > 0 else None

    try:
        preds       = pd.read_csv(config.PREDICTIONS)
        pred_maison = preds[preds[config.COL_TYPE] == 'Maison']['prix_predit'].mean()
        pred_appart = preds[preds[config.COL_TYPE] == 'Appartement']['prix_predit'].mean()
    except Exception:
        pred_maison = None
        pred_appart = None

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="titre-section">Prix moyen</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        patches = []
        for type_bien in config.TYPES_BIENS:
            c   = COULEURS[type_bien]
            v24 = safe_get(type_bien, 2024)
            v25 = safe_get(type_bien, 2025)
            v26 = pred_maison if type_bien == 'Maison' else pred_appart

            ann_r = [a for a, v in zip([2024, 2025], [v24, v25]) if v is not None]
            val_r = [v for v in [v24, v25] if v is not None]

            if ann_r:
                ax.plot(ann_r, val_r, color=c, linewidth=2.5, marker='o', markersize=9)
                for x, y in zip(ann_r, val_r):
                    ax.annotate(f'{y:,.0f} €', (x, y),
                                textcoords='offset points', xytext=(0, 12),
                                ha='center', fontsize=9, color=c)
            if v26 and ann_r:
                ax.plot([ann_r[-1], 2026], [val_r[-1], v26], color=c,
                        linewidth=2.5, linestyle='--', marker='o', markersize=9,
                        markerfacecolor='white', markeredgewidth=2)
                ax.annotate(f'{v26:,.0f} €\n(ML)', (2026, v26),
                            textcoords='offset points', xytext=(0, 12),
                            ha='center', fontsize=9, color=c, fontweight='bold')
            patches.append(mpatches.Patch(color=c, label=type_bien))

        ax.axvspan(2025.5, 2026.4, alpha=0.07, color='gray')
        ligne_plein   = plt.Line2D([0],[0], color='gray', lw=2, label='Donnees reelles')
        ligne_pointil = plt.Line2D([0],[0], color='gray', lw=2, linestyle='--', label='Prediction ML')
        ax.legend(handles=patches + [ligne_plein, ligne_pointil], loc='upper left', fontsize=9)
        ax.set_xticks([2024, 2025, 2026])
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#F8F7F4')
        fig.patch.set_facecolor('#F8F7F4')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<p class="titre-section">Prix au m²</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        for type_bien in config.TYPES_BIENS:
            c   = COULEURS[type_bien]
            v24 = safe_get(type_bien, 2024, 'prix_m2')
            v25 = safe_get(type_bien, 2025, 'prix_m2')
            ann_r = [a for a, v in zip([2024, 2025], [v24, v25]) if v is not None]
            val_r = [v for v in [v24, v25] if v is not None]
            if ann_r:
                ax.plot(ann_r, val_r, color=c, linewidth=2.5, marker='o', markersize=9, label=type_bien)
                for x, y in zip(ann_r, val_r):
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

    st.markdown('<p class="titre-section">Tableau recapitulatif</p>', unsafe_allow_html=True)
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