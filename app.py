import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

st.title("Prediction immobilier — Dept. 49")
st.markdown("---")

@st.cache_data
def load_data():
    df24 = pd.read_csv(config.CSV_2024)
    df25 = pd.read_csv(config.CSV_2025)
    df = pd.concat([df24, df25], ignore_index=True)
    df = df[df[config.COL_TYPE].isin(config.TYPES_BIENS)].copy()
    df = df.dropna(subset=[config.COL_PRIX, config.COL_SURFACE])
    df['prix_m2'] = df[config.COL_PRIX] / df[config.COL_SURFACE]
    df = df[(df['prix_m2'] > 200) & (df['prix_m2'] < 15000)]
    return df

@st.cache_resource
def load_model():
    with open(config.MODEL_OBJ1, 'rb') as f:
        return pickle.load(f)

df = load_data()
model = load_model()

# ── STATS ─────────────────────────────────────────────────────────────────────

col1, col2, col3 = st.columns(3)
col1.metric("Transactions", f"{len(df):,}")
col2.metric("Prix median Maison", f"{df[df[config.COL_TYPE]=='Maison'][config.COL_PRIX].median():,.0f} €")
col3.metric("Prix median Appart", f"{df[df[config.COL_TYPE]=='Appartement'][config.COL_PRIX].median():,.0f} €")

st.markdown("---")

# ── SIMULATEUR ────────────────────────────────────────────────────────────────

st.subheader("Simulateur de prix 2026")

col1, col2 = st.columns(2)

with col1:
    type_bien = st.selectbox("Type de bien", ["Maison", "Appartement"])
    surface   = st.slider("Surface (m²)", 20, 300, 90)
    pieces    = st.slider("Nombre de pieces", 1, 10, 4)
    terrain   = st.slider("Surface terrain (m²)", 0, 2000, 300) if type_bien == "Maison" else 0

with col2:
    X_input = pd.DataFrame([{
        'surface_reelle_bati':       surface,
        'nombre_pieces_principales': pieces,
        'surface_terrain':           terrain,
        'latitude':                  47.478419,
        'longitude':                 -0.563166,
        'type_encode':               1 if type_bien == "Maison" else 0,
        'annee':                     2026,
        'mois':                      6,
    }])[list(model.feature_names_in_)]

    prix = model.predict(X_input)[0]
    prix_moyen = df[df[config.COL_TYPE]==type_bien][config.COL_PRIX].mean()
    diff = ((prix - prix_moyen) / prix_moyen) * 100

    st.markdown("### Estimation 2026")
    st.markdown(f"## {prix:,.0f} €")
    st.markdown(f"**{prix/surface:,.0f} €/m²**")
    st.metric("vs prix moyen actuel", f"{prix_moyen:,.0f} €", f"{diff:+.1f}%")

st.markdown("---")

# ── DISTRIBUTION ──────────────────────────────────────────────────────────────

st.subheader("Distribution des prix")

fig, ax = plt.subplots(figsize=(9, 4))
for t, c in [("Maison", "#0D9488"), ("Appartement", "#6366F1")]:
    ax.hist(df[df[config.COL_TYPE]==t][config.COL_PRIX], bins=40, alpha=0.6, color=c, label=t)
ax.set_xlabel("Prix (€)")
ax.set_ylabel("Nombre de ventes")
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
st.pyplot(fig)
plt.close()