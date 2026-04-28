import os

# ──────────────────────────────────────────────
#  Racine du projet — à ne modifier que si tu
#  déplaces le dossier MAJ ailleurs
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Données brutes ─────────────────────────────
CSV_2024      = os.path.join(BASE_DIR, "49_2024.csv")
CSV_2025      = os.path.join(BASE_DIR, "49_2025.csv")
CSV_TRANS     = os.path.join(BASE_DIR, "transactions.csv")

# ── Fichiers produits par Arthur ───────────────
DATA_CLEAN    = os.path.join(BASE_DIR, "data_clean.csv")      # export Arthur étape 1
MODEL_OBJ1    = os.path.join(BASE_DIR, "model_obj1.pkl")      # modèle Arthur étape 2

# ── Fichiers produits par Mandengue ───────────
FEATURES_LIST = os.path.join(BASE_DIR, "features_list.py")   # variables à utiliser
MODEL_OBJ2    = os.path.join(BASE_DIR, "model_obj2.pkl")      # modèle Mandengue étape 2

# ── Fichiers produits par Moneli (toi) ─────────
X_2026        = os.path.join(BASE_DIR, "X_2026.csv")          # scénarios futurs
PREDICTIONS   = os.path.join(BASE_DIR, "predictions_2026.csv")# résultats prédiction

# ── Colonnes clés communes ─────────────────────
COL_PRIX      = "valeur_fonciere"
COL_SURFACE   = "surface_reelle_bati"
COL_PIECES    = "nombre_pieces_principales"
COL_TERRAIN   = "surface_terrain"
COL_TYPE      = "type_local"
COL_DATE      = "date_mutation"
COL_LAT       = "latitude"
COL_LON       = "longitude"

# ── Types de biens à garder ────────────────────
TYPES_BIENS   = ["Maison", "Appartement"]

# ── Année à prédire ────────────────────────────
ANNEE_PRED    = 2026
