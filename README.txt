PROJET ML IMMOBILIER - Maine-et-Loire (Dept. 49)

ORDRE D'EXECUTION :

1. ARTHUR (arthur_objectif1.ipynb)
   - Fusion 49_2024.csv + 49_2025.csv
   - Nettoyage + feature engineering
   - Regression lineaire vs Random Forest
   - PRODUIT : data_clean.csv + model_obj1.pkl

2. MANDENGUE (mandengue_objectif2.ipynb)
   - Charge data_clean.csv (d'Arthur)
   - EDA + analyse des correlations
   - Model Random Forest par type de bien
   - PRODUIT : features_list.py + model_obj2.pkl

3. MONELI
   a. moneli_scenarios_2026.ipynb
      - Genere les scenarios futurs
      - PRODUIT : X_2026.csv
      
   b. moneli_predictions.ipynb
      - Charge model_obj1.pkl (d'Arthur)
      - Predict les prix 2026
      - PRODUIT : predictions_2026.csv + graphes PNG

FICHIERS GENERES AUTOMATIQUEMENT (ne pas creer manuellement) :

- model_obj1.pkl : modele scikit-learn d'Arthur (fichier binaire)
- model_obj2.pkl : modele scikit-learn de Mandengue (fichier binaire)
- *.png : graphes matplotlib generes par les notebooks

FICHIERS TEMPLATES FOURNIS (remplacés lors de l'execution) :

- data_clean.csv
- features_list.py
- X_2026.csv
- predictions_2026.csv

Ces fichiers contiennent un exemple de structure. Ils seront remplaces avec les vraies donnees quand les notebooks s'executent.

STRUCTURE FINALE (apres execution complete) :

PROJETIMMO/
  Donnees/
    49_2024.csv
    49_2025.csv
    transactions.csv
  Notebooks/
    arthur_objectif1.ipynb
    mandengue_objectif2.ipynb
    moneli_scenarios_2026.ipynb
    moneli_predictions.ipynb
  Fichiers generes/
    data_clean.csv
    model_obj1.pkl
    model_obj2.pkl
    features_list.py
    X_2026.csv
    predictions_2026.csv
    moneli_courbe_finale.png
    moneli_courbe_m2.png
    arthur_reel_vs_predit.png
    arthur_feature_importance.png
    mandengue_distribution_prix.png
    mandengue_heatmap.png
    mandengue_feature_importance.png
  config.py
