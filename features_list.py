# features_list.py — produit par Mandengue
# Variables sélectionnées après analyse EDA

FEATURES = ['surface_reelle_bati', 'nombre_pieces_principales', 'surface_terrain', 'latitude', 'longitude', 'annee', 'mois']

# Justification :
# - surface_reelle_bati : corrélation la plus forte avec le prix
# - nombre_pieces_principales : lié à la surface, bon complément
# - surface_terrain : important pour les maisons
# - latitude / longitude : capture l'effet zone géographique
# - annee / mois : capture la tendance temporelle
