# 🔥 Surveillance & Confinement Autonome des Feux de Forêt — Modules CNN & RNN/LSTM

**Projet 5 — Data Mining & Deep Learning (4ème Année Ingéniorat)**  
👥 **Équipe :**
- 👤 **Salim Chablaoui** (*Responsable Vision par Ordinateur — CNN / U-Net & Support Data*)
- 👤 **Moundir** (*Responsable Modélisation Temporelle & Météo — RNN / LSTM & Support Data*)

---

## 📌 Présentation du Projet Multi-Agents

Ce dépôt contient l'implémentation complète des **deux premiers modules clés** pour l'architecture multi-agents de surveillance et de confinement autonome des feux de forêt (**Projet 5**) :

1. **👤 Agent 1 — Salim (Vision par Ordinateur - CNN / U-Net)** : Détection binaire de la présence du feu (`Oui`/`Non`) et segmentation spatiale pixel par pixel de la zone touchée sur imagerie satellite/aérienne (`predict(image_path)`).
2. **👤 Agent 2 — Moundir (Modélisation Temporelle & Météo - RNN/LSTM)** : Prédiction de la vitesse et de la direction de propagation selon la localisation extraite par le CNN, le vent et l'humidité (`predict_propagation(location, wind, humidity)`).
3. **👤 Agent 3 — Système d'Alerte & Intégration (Arbre de Décision & Streamlit)** : Fusion des prédictions CNN + LSTM + météo pour déclencher les paliers d'alerte (`Vert`, `Orange`, `Rouge`, `Évacuation`).

---

## 🏗️ Architecture et Missions Réalisées

### 👤 Module 1 — Vision par Ordinateur (Salim)

- **Tâche 1 : Prétraitement des Images Satellites (`preprocessing.py`)** : Nettoyage automatique, égalisation adaptative CLAHE (réduction du voile atmosphérique), standardisation `128×128`, extraction spectrale des masques et data augmentation synchronisée.
- **Tâche 2 : Entraînement du Modèle CNN Multi-Tâches (`model_cnn.py`)** : Architecture `FireUNet` (Encodeur-Décodeur U-Net avec Skip Connections + classification globale) avec fonction hybride BCE + Dice Loss (**Accuracy : 99.1%**, **Dice : 84.2%**, **IoU : 79.5%**).
- **Tâche 3 : Sauvegarde & Inférence (`predict.py`)** : Modèle sauvegardé dans `models/fire_cnn_unet.pkl` et `models/fire_cnn_unet.h5`. Fonction `predict(image_path)` retournant `fire_detected`, `confidence`, `location` (secteur, barycentre, GPS, surface touchée %) et masque binaire.
- **Mission Secondaire (Support Data - `data_support.py`)** : Téléchargement Kaggle et création du dataset global unifié `dataset/global_fire_weather_dataset.csv`.

---

### 👤 Module 2 — Modélisation Temporelle & Météo (Moundir)

- **Tâche 1 : Traitement Météorologique & Séries Temporelles (`temporal_preprocessing.py`)** :
  - Traitement et feature engineering des variables atmosphériques (température, humidité relative `RH`, vent : vitesse et composantes vectorielles).
  - Encodage cyclique trigonométrique des directions $(\sin \theta, \cos \theta)$ pour éliminer la discontinuité critique à $0^\circ \equiv 360^\circ$ dans les modèles récurrents.
  - Calcul de l'indice météorologique de propagation initiale (FWI / ISI).
- **Tâche 2 : Entraînement du Modèle RNN/LSTM (`model_lstm.py`)** :
  - Architecture récurrente bidirectionnelle `FirePropagationLSTM` avec mécanisme de pooling par attention temporelle (`MultiHeadAttentionPooling`).
  - Têtes de prédiction multi-tâches découplées :
    1. Vitesse de propagation instantanée ($v \ge 0$ km/h).
    2. Vecteur de direction unitaire garantissant une cohérence géométrique à $360^\circ$.
    3. Expansion horaire multi-horizons ($t+1\text{h}$, $t+2\text{h}$, $t+3\text{h}$).
  - Entraînement optimisé avec fonction de perte multi-objectifs (**MAE Vitesse : 0.28 km/h**, **Erreur Angulaire Moyenne : 5.7°**).
- **Tâche 3 : Sauvegarde & Inférence (`predict_propagation.py`)** :
  - Modèle final sauvegardé sous `models/fire_propagation_lstm.pkl` (et `models/fire_propagation_lstm.pt`).
  - Fonction officielle **`predict_propagation(location, wind, humidity)`** qui retourne directement la direction et la vitesse de propagation du feu.
- **Mission Secondaire : Support Data** :
  - Outils de préparation et découpage en fenêtres temporelles glissantes (`create_sliding_sequences`, `generate_synthetic_propagation_sequences`) pour alimenter le Système d'Alerte et l'Arbre de Décision.

---

## 📂 Structure du Dépôt

```text
FireDetection/
├── preprocessing.py                        # Salim - Tâche 1 : Nettoyage, CLAHE, masques & augmentation images
├── model_cnn.py                            # Salim - Tâche 2 : Architecture CNN U-Net (FireUNet) & entraînement
├── predict.py                              # Salim - Tâche 3 : Fonction predict(image_path) -> Oui/Non + Localisation
├── data_support.py                         # Salim - Support Data : Téléchargement Kaggle & dataset global
├── Salim_CNN_FireDetection_Notebook.ipynb    # Salim - Notebook Jupyter documenté
├── Rapport_Technique_Salim_CNN.md          # Salim - Rapport technique détaillé
├── train_and_verify.py                     # Salim - Script de vérification de bout en bout
│
├── temporal_preprocessing.py               # Moundir - Tâche 1 & Support Data : Prétraitement météo & séquences
├── model_lstm.py                           # Moundir - Tâche 2 : Architecture RNN/LSTM & entraînement multi-tâches
├── predict_propagation.py                  # Moundir - Tâche 3 : Fonction officielle predict_propagation()
├── Moundir_LSTM_FirePropagation_Notebook.ipynb # Moundir - Notebook Jupyter documenté
├── Rapport_Technique_Moundir_LSTM.md        # Moundir - Rapport technique détaillé
├── train_and_verify_lstm.py                # Moundir - Script de vérification de bout en bout
│
├── requirements.txt                        # Dépendances Python
├── models/
│   ├── fire_cnn_unet.pkl                   # Modèle CNN U-Net entraîné (Pickle)
│   ├── fire_cnn_unet.h5                    # Modèle CNN U-Net entraîné (HDF5)
│   ├── fire_propagation_lstm.pkl           # Modèle LSTM Propagation entraîné (Pickle)
│   └── fire_propagation_lstm.pt           # Modèle LSTM Propagation entraîné (PyTorch)
├── dataset/
│   ├── forestfires_kaggle.csv              # Dataset météo officiel Kaggle (517 lignes)
│   ├── global_fire_weather_dataset.csv     # Dataset unifié (CNN + Météo + Propagation + Alerte)
│   ├── kaggle_cleaned/                     # Images Kaggle nettoyées (128x128) + masques
│   └── satellite_samples/                  # Échantillons satellites forestiers + masques
└── outputs/
    ├── demo_fire_segmented.png             # Démonstration segmentation satellite
    ├── demo_kaggle_fire_segmented.png      # Démonstration segmentation Kaggle
    ├── demo_nofire_segmented.png           # Démonstration non-feu
    ├── training_metrics.json               # Historique d'entraînement CNN
    └── lstm_training_metrics.json          # Historique d'entraînement LSTM
```

---

## 🚀 Installation & Utilisation

### 1. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. Détection par le CNN (Salim)
```python
from predict import predict

res_cnn = predict("dataset/satellite_samples/images/fire/sat_fire_0001.png")
print("Feu détecté :", res_cnn["fire_detected"])   # "Oui" / "Non"
print("Localisation :", res_cnn["location"]["zone"]) # "Nord-Est", "Centre", etc.
```

### 3. Prédiction de la Propagation par le LSTM (Moundir)
```python
from predict_propagation import predict_propagation

# Utilisation directe avec la localisation du CNN, le vent et l'humidité
res_lstm = predict_propagation(
    location=res_cnn["location"],
    wind={"speed": 38.0, "direction": "Sud-Ouest"},
    humidity=22.0,
    temperature=34.0,
)

print("Direction de propagation :", res_lstm["propagation_direction"]) # ex: "Sud-Ouest"
print("Vitesse de propagation   :", res_lstm["propagation_speed"], "km/h") # ex: 7.82 km/h
print("Expansion horaire        :", res_lstm["expansion_next_hours"]) # {'+1h': 8.1, '+2h': 8.4, '+3h': 8.9}
print("Synthèse :", res_lstm["summary"])
```

### 4. Entraîner et vérifier les modèles
```bash
# Vérification du module CNN (Salim)
python train_and_verify.py

# Vérification du module LSTM (Moundir)
python train_and_verify_lstm.py
```

---

## 🔗 Références des Datasets Kaggle (Projet 5)
1. **Imagerie Feu & Forêt** : [FIRE Dataset (`phylake1337/fire-dataset`)](https://www.kaggle.com/datasets/phylake1337/fire-dataset)
2. **Données Météorologiques & Feux de Forêt** : [Forest Fires Data Set (`elikplim/forest-fires-data-set`)](https://www.kaggle.com/datasets/elikplim/forest-fires-data-set)
