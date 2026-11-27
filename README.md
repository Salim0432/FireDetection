# 🔥 Surveillance & Confinement Autonome des Feux de Forêt

**Projet 5 — Data Mining & Deep Learning (Architecture Multi-Agents)**

👥 **Équipe :**
- 👤 **Salim Chablaoui** — *Agent 1 : Vision par Ordinateur (CNN / U-Net) & Support Data*
- 👤 **Moundir Boucheriha** — *Agent 2 : Modélisation Temporelle & Météo (RNN / LSTM) & Support Data*
- 👤 **Asma Dadou** — *Agent 3 : Système d'Alerte (Arbre de Décision), Orchestration & Démonstration Streamlit*

---

## 📌 Objectif du Projet

Coordonner la détection précoce d'un incendie sur imagerie satellite et modéliser son expansion temporelle selon les conditions météorologiques afin de déclencher automatiquement les paliers d'alerte et d'évacuation :

1. **Agent 1 — Salim (CNN / U-Net)** : Détection binaire de la présence du feu (`Oui`/`Non`) et segmentation spatiale pixel par pixel de la zone touchée sur imagerie satellite (`predict(image_path)`).
2. **Agent 2 — Moundir (RNN / LSTM)** : Prédiction temporelle de la vitesse (`km/h`) et de la direction de propagation selon la localisation extraite par le CNN, le vent et l'humidité (`predict_propagation(location, wind, humidity)`).
3. **Agent 3 — Asma (Arbre de Décision & Intégration)** : Classification explicable des paliers d'alerte (`🟢 VERT`, `🟠 ORANGE`, `🔴 ROUGE`, `🚨 ÉVACUATION`) via un `DecisionTreeClassifier` (`donner_alerte(...)`), pipeline unifié (`main.py` + `contracts.py`) et interface de démonstration (`app.py`).

---

## 🏗️ Architecture et Modules Réalisés

### 👤 Agent 1 — Vision par Ordinateur (Salim)
- **Prétraitement (`preprocessing.py`)** : Nettoyage automatique, égalisation adaptative CLAHE, standardisation `128×128`, extraction spectrale des masques (RGB / HSV / YCbCr) et data augmentation synchronisée.
- **Modèle CNN Multi-Tâches (`model_cnn.py`)** : Architecture `FireUNet` (Encodeur-Décodeur U-Net avec Skip Connections + tête de classification globale) entraînée avec BCE + Dice Loss.
- **Inférence (`predict.py`)** : Sauvegarde dans `models/fire_cnn_unet.pkl` et fonction `predict(image_path)` retournant `fire_detected`, `confidence`, `location` (secteur cardinal, barycentre, boîte englobante, `%` de surface touchée) et masque binaire.

### 👤 Agent 2 — Modélisation Temporelle & Météo (Moundir)
- **Séries Temporelles & Météo (`temporal_preprocessing.py`)** : Prétraitement de la vitesse/direction du vent, de l'humidité relative (`RH %`) et de la température, encodage cyclique $(\sin \theta, \cos \theta)$ des directions et découpage en fenêtres temporelles glissantes.
- **Modèle RNN/LSTM (`model_lstm.py`)** : Architecture récurrente bidirectionnelle `FirePropagationLSTM` avec pooling temporel (`TemporalAttentionPooling`) prédisant la vitesse de propagation et le vecteur directionnel unitaire.
- **Inférence (`predict_propagation.py`)** : Sauvegarde dans `models/fire_propagation_lstm.pkl` et fonction `predict_propagation(location, wind, humidity)`.

### 👤 Agent 3 — Aide à la Décision & Orchestration (Asma)
- **Arbre de Décision (`train_tree.py`, `predict_alert.py`)** : Classifieur CART (`DecisionTreeClassifier`, critère de Gini, validation croisée stratifiée 5-plis) sauvegardé dans `models/mon_arbre_alerte.pkl` pour déclencher les 4 paliers d'alerte à partir de `[feu, vent, humidite, vitesse_propagation]`.
- **Contrat & Pipeline (`contracts.py`, `main.py`)** : Validation formelle des échanges inter-agents et orchestration séquentielle de bout en bout (avec court-circuit immédiat en palier `🟢 VERT` lorsqu'aucun feu n'est détecté).
- **Démonstration Interactive (`app.py`)** : Tableau de bord web Streamlit permettant de charger une image satellite, régler le vent et l'humidité en temps réel, et visualiser la segmentation, la propagation et la décision d'évacuation.

---

## 📂 Structure du Dépôt

```text
FireDetection/
├── contracts.py                                # Contrat d'interface et validation inter-agents
├── main.py                                     # Pipeline multi-agents complet (Salim -> Moundir -> Asma)
├── app.py                                      # Application web de démonstration (Streamlit)
│
├── preprocessing.py                            # Salim - Prétraitement, CLAHE, masques & augmentation
├── model_cnn.py                                # Salim - Architecture CNN U-Net (FireUNet) & entraînement
├── predict.py                                  # Salim - Inférence predict(image_path)
├── data_support.py                             # Salim - Acquisition Kaggle & dataset global d'équipe
├── train_and_verify.py                         # Salim - Script d'entraînement et vérification CNN
├── Salim_CNN_FireDetection_Notebook.ipynb      # Salim - Notebook documenté
├── Rapport_Technique_Salim_CNN.md              # Salim - Rapport technique (Vision CNN)
│
├── temporal_preprocessing.py                   # Moundir - Prétraitement météo & fenêtres glissantes
├── model_lstm.py                               # Moundir - Architecture RNN/LSTM & entraînement
├── predict_propagation.py                      # Moundir - Inférence predict_propagation()
├── train_and_verify_lstm.py                    # Moundir - Script d'entraînement et vérification LSTM
├── Moundir_LSTM_FirePropagation_Notebook.ipynb # Moundir - Notebook documenté
├── Rapport_Technique_Moundir_LSTM.md           # Moundir - Rapport technique (Propagation LSTM)
│
├── train_tree.py                               # Asma - Entraînement & validation de l'Arbre de Décision
├── predict_alert.py                            # Asma - Inférence donner_alerte()
├── Asma_Integration_Notebook.ipynb             # Asma - Notebook documenté
├── Rapport_Technique_Asma_Integration.md       # Asma - Rapport technique (Décision & Intégration)
│
├── requirements.txt                            # Dépendances Python
├── models/
│   ├── fire_cnn_unet.pkl                       # Modèle CNN U-Net entraîné
│   ├── fire_propagation_lstm.pkl               # Modèle RNN/LSTM entraîné
│   └── mon_arbre_alerte.pkl                    # Arbre de Décision entraîné
├── dataset/
│   ├── forestfires_kaggle.csv                  # Forest Fires Dataset officiel (Kaggle, 517 lignes)
│   ├── global_fire_weather_dataset.csv         # Dataset unifié (CNN + Météo + Propagation + Alerte)
│   ├── kaggle_cleaned/                         # Images Kaggle nettoyées (128x128) + masques
│   └── satellite_samples/                      # Échantillons satellites forestiers + masques
└── outputs/                                    # Métriques JSON d'entraînement et démos segmentées
```

---

## 🚀 Installation & Utilisation

### 1. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. Lancer la démonstration interactive (Streamlit)
```bash
streamlit run app.py
```

### 3. Exécuter le pipeline multi-agents complet en ligne de commande
```bash
python main.py
```

Exemple d'utilisation programmatique des 3 agents :
```python
from main import run_pipeline

rapport = run_pipeline(
    image_path="dataset/satellite_samples/images/fire/sat_fire_0001.png",
    wind_speed=38.0,
    wind_dir="Sud-Ouest",
    humidity=22.0,
    temperature=34.0,
    return_details=True,
)
print("Alerte :", rapport["alert"])
print("Recommandation :", rapport["recommendation"])
```

### 4. Ré-entraîner et vérifier chaque modèle individuellement
```bash
# Agent 1 — CNN / U-Net (Salim)
python train_and_verify.py

# Agent 2 — RNN / LSTM (Moundir)
python train_and_verify_lstm.py

# Agent 3 — Arbre de Décision (Asma)
python train_tree.py
```

---

## 🔗 Références des Datasets Kaggle (Projet 5)
1. **Satellite / Fire Imagery** : [FIRE Dataset (`phylake1337/fire-dataset`)](https://www.kaggle.com/datasets/phylake1337/fire-dataset)
2. **Forest Fires Dataset** : [Forest Fires Data Set (`elikplim/forest-fires-data-set`)](https://www.kaggle.com/datasets/elikplim/forest-fires-data-set)
