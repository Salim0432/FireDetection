# 🔥 Surveillance & Confinement Autonome des Feux de Forêt — Module Vision par Ordinateur (CNN / U-Net)

**Projet 5 — Data Mining & Deep Learning (4ème Année Ingéniorat)**  
👤 **Responsable du Module :** **Salim Chablaoui** (*Responsable Vision par Ordinateur — CNN & Support Data*)

---

## 📌 Présentation du Projet

Ce dépôt contient l'implémentation complète du **Module 1 : Vision par Ordinateur (CNN / U-Net)** ainsi que le **Support Data** pour l'architecture multi-agents de surveillance et de confinement autonome des feux de forêt (**Projet 5**) :

1. **👤 Agent 1 — Salim (Vision par Ordinateur - CNN / U-Net)** : Détection binaire de la présence du feu (`Oui`/`Non`) et segmentation spatiale pixel par pixel de la zone touchée sur imagerie satellite/aérienne.
2. **👤 Agent 2 — Muondir (Modélisation Temporelle & Météo - RNN/LSTM)** : Prédiction de la vitesse et de la direction de propagation selon la localisation extraite par le CNN, le vent et l'humidité.
3. **👤 Agent 3 — Système d'Alerte & Intégration (Arbre de Décision & Streamlit)** : Fusion des prédictions CNN + LSTM + météo pour déclencher les paliers d'alerte (`Vert`, `Orange`, `Rouge`, `Évacuation`).

---

## 🏗️ Architecture et Missions Réalisées (Salim)

### ✅ Tâche 1 : Prétraitement des Images Satellites (`preprocessing.py`)
- **Nettoyage automatique (`clean_dataset_directory`, `is_image_valid`)** : Vérification de l'intégrité des images et filtrage des fichiers corrompus ou uniformes.
- **Correction Atmosphérique (CLAHE)** : Égalisation adaptative d'histogramme dans l'espace $L^*a^*b^*$ pour réduire le voile atmosphérique satellite.
- **Redimensionnement & Normalisation (`preprocess_image`)** : Standardisation en `128×128` RGB normalisé dans $[0, 1]$.
- **Extraction Spectrale des Masques & Data Augmentation (`extract_fire_segmentation_mask`, `augment_image_and_mask`)** :
  - Combinaison des règles chromatiques et thermiques dans les espaces **RGB**, **HSV** et **YCbCr** ($Y \ge Cb$, $Cr > Cb$).
  - Augmentation synchronisée image + masque : retournements horizontaux/verticaux, rotations ($90^\circ, 180^\circ, 270^\circ$), variations photométriques (luminosité/contraste) et bruit gaussien capteur.

### ✅ Tâche 2 : Entraînement du Modèle CNN Multi-Tâches (`model_cnn.py`)
Architecture **`FireUNet`** (Encodeur-Décodeur U-Net avec *Skip Connections* + Tête de Classification Globale sur le *Bottleneck*) entraînée avec une fonction de perte hybride **BCE + Dice Loss** sur **100% du dataset Kaggle** (`1 179` images au total : `966` Train / `213` Validation) :

| Métrique de Validation | Résultat Obtenu |
| :--- | :---: |
| **Exactitude de Classification (Validation Accuracy)** | **99.1%** |
| **Coefficient de Dice (Segmentation Spatiale)** | **84.19%** (`0.8419`) |
| **Intersection-over-Union (IoU)** | **79.54%** (`0.7954`) |

### ✅ Tâche 3 : Sauvegarde du Modèle (`.pkl` & `.h5`) et Fonction `predict(image_path)` (`predict.py`)
- Modèles finaux sauvegardés simultanément aux deux formats standards :
  - `models/fire_cnn_unet.pkl`
  - `models/fire_cnn_unet.h5`
- Fonction `predict(image_path)` retournant la détection (`Oui`/`Non`), la confiance, le masque de segmentation et la localisation complète du foyer.

### ✅ Mission Secondaire : Support Data (`data_support.py`)
- Préparation et exportation du dataset global synchronisé `dataset/global_fire_weather_dataset.csv` (`517` enregistrements complets) fusionnant les sorties du CNN de Salim avec les relevés météorologiques réels du dataset Kaggle `forestfires_kaggle.csv` (température, humidité `RH`, vitesse et direction du vent, vitesse de propagation et paliers d'alerte).

---

## 📂 Structure du Dépôt

```text
FireDetection/
├── preprocessing.py                     # Tâche 1 : Nettoyage, CLAHE, redimensionnement, masques & augmentation
├── model_cnn.py                         # Tâche 2 : Architecture CNN U-Net (FireUNet), entraînement & sauvegarde .pkl/.h5
├── predict.py                           # Tâche 3 : Fonction officielle predict(image_path) -> Oui/Non + Localisation
├── data_support.py                      # Mission Secondaire : Téléchargement Kaggle & génération du dataset global CSV
├── train_and_verify.py                  # Script complet d'entraînement et de validation de bout en bout
├── Salim_CNN_FireDetection_Notebook.ipynb # Notebook Jupyter/Kaggle documenté
├── Rapport_Technique_Salim_CNN.md       # Partie du rapport technique rédigée par Salim
├── requirements.txt                     # Dépendances Python
├── models/
│   ├── fire_cnn_unet.pkl                # Modèle CNN U-Net entraîné (format Pickle)
│   └── fire_cnn_unet.h5                 # Modèle CNN U-Net entraîné (format HDF5)
├── dataset/
│   ├── forestfires_kaggle.csv           # Dataset météo officiel Kaggle (517 lignes)
│   ├── global_fire_weather_dataset.csv  # Dataset global unifié (CNN + Météo + Propagation + Alerte)
│   ├── kaggle_cleaned/                  # Images Kaggle nettoyées (128x128) + masques
│   └── satellite_samples/               # Échantillons satellites forestiers + masques vérité terrain
└── outputs/
    ├── demo_fire_segmented.png          # Exemple de segmentation sur scène satellite avec feu
    ├── demo_kaggle_fire_segmented.png   # Exemple de segmentation sur image Kaggle avec feu
    ├── demo_nofire_segmented.png        # Exemple de prédiction sur zone forestière saine
    └── training_metrics.json            # Historique des courbes d'entraînement (Loss, Acc, Dice, IoU)
```

---

## 🚀 Installation & Utilisation Rapide

### 1. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. Utiliser la fonction `predict(image_path)` (Intégration dans `main.py` / Streamlit)
```python
from predict import predict

resultat = predict(
    "dataset/satellite_samples/images/fire/sat_fire_0027.png",
    save_visualization="outputs/prediction_vis.png"
)

print("Feu détecté :", resultat["fire_detected"])   # "Oui" ou "Non"
print("Confiance   :", resultat["confidence"])      # ex: 1.0 (100%)
print("Localisation:", resultat["location"])
# {
#   'zone': 'Centre',
#   'center_xy': (47, 56),
#   'normalized_coords': (0.3701, 0.4409),
#   'gps_coords': (36.75591, 5.03701),
#   'bbox': [16, 9, 127, 112],
#   'affected_area_pct': 6.8
# }
```

### 3. Relancer l'entraînement complet
```bash
python train_and_verify.py
```

---

## 🔗 Références des Datasets Kaggle (Projet 5)
1. **Imagerie Feu & Forêt** : [FIRE Dataset (`phylake1337/fire-dataset`)](https://www.kaggle.com/datasets/phylake1337/fire-dataset) & [Wildfire Prediction Dataset (Satellite Images)](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset)
2. **Données Météorologiques & Feux de Forêt** : [Forest Fires Data Set (`elikplim/forest-fires-data-set`)](https://www.kaggle.com/datasets/elikplim/forest-fires-data-set)
