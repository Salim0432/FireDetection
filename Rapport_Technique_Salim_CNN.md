# Rapport Technique — Partie 1 : Vision par Ordinateur (CNN / U-Net)
**Auteur :** Salim — *Responsable Vision par Ordinateur (CNN) & Support Data*

---

## 1. Introduction et Choix Architectural
Dans le cadre de notre système intelligent de détection précoce et de suivi des feux de forêt par imagerie satellite, le module de **Vision par Ordinateur** constitue le premier maillon critique de la chaîne décisionnelle. Ses deux objectifs principaux sont :
1. **Détecter avec fiabilité la présence d'un incendie** (`Oui` / `Non`) sur une scène satellite ou aérienne.
2. **Segmenter spatialement la zone touchée** pixel par pixel afin d'en extraire la **localisation exacte** (secteur géographique, coordonnées du barycentre, boîte englobante et pourcentage de surface forestière impactée).

### Pourquoi une architecture CNN Multi-Tâches de type U-Net (`FireUNet`) ?
Plutôt qu'un simple classifieur CNN (qui ne donne qu'une étiquette globale sans contour spatial) ou un détecteur par boîtes rectangulaires seules, nous avons conçu une architecture **U-Net Multi-Tâches (`FireUNet`)** :
- **Encodeur Convolutif** : 4 niveaux de doubles blocs convolutifs (`Conv2d 3×3` + `BatchNorm2d` + `ReLU` + `MaxPool2d`) qui extraient les caractéristiques spectrales et texturales multi-échelles (flammes actives, panaches de fumée, végétation saine).
- **Tête de Classification Globale ( sur le Bottleneck )** : Un `AdaptiveAvgPool2d` suivi de couches denses qui prédit la probabilité globale de présence de feu $p_{\text{cls}} \in [0, 1]$.
- **Décodeur U-Net avec Connexions Résiduelles (*Skip Connections*)** : Reconstruit une carte de probabilité haute résolution $M \in [0, 1]^{H \times W}$ permettant de délimiter la forme organique exacte du foyer d'incendie.

---

## 2. Tâche 1 : Prétraitement des Images Satellites (`preprocessing.py`)
Le pipeline de prétraitement mis en place dans `preprocessing.py` comprend quatre étapes systématiques :
1. **Nettoyage automatique (`is_image_valid`, `clean_dataset_directory`)** : Vérification de l'intégrité des fichiers d'entrée, rejet des images tronquées, corrompues ou uniformes ($\sigma < 2.0$).
2. **Correction du contraste atmosphérique (CLAHE)** : Application d'une égalisation adaptative d'histogramme limitée en contraste sur le canal de luminance $L^*$ (espace $L^*a^*b^*$) afin d'atténuer le voile atmosphérique fréquent sur les acquisitions satellites.
3. **Redimensionnement et Normalisation (`preprocess_image`)** : Redimensionnement bilinéaire standardisé en `128×128` pixels et normalisation des intensités RGB dans $[0, 1]$.
4. **Extraction Spectrale des Masques & Augmentation des Données (`augment_image_and_mask`)** :
   - Fusion de règles chromatiques et thermiques dans les espaces **RGB**, **HSV** et **YCbCr** ($Y \ge Cb$, $Cr > Cb$) suivie d'opérations morphologiques d'ouverture/fermeture pour isoler les pixels de combustion.
   - Augmentation aléatoire synchronisée entre l'image et son masque : retournements horizontaux/verticaux, rotations par multiples de $90^\circ$, variations photométriques de luminosité/contraste ($\alpha \in [0.85, 1.18]$, $\beta \in [-0.08, 0.08]$) et bruit gaussien capteur.

---

## 3. Tâche 2 : Entraînement du Modèle CNN (`model_cnn.py`)
Le modèle `FireUNet` est entraîné de bout en bout avec l'optimiseur `AdamW` ($\text{lr} = 1.2 \times 10^{-3}$, régularisation $L_2 = 10^{-4}$) et un ordonnanceur `CosineAnnealingLR`.

### Fonction de Perte Hybride (Multi-Task Loss)
Pour gérer le déséquilibre spatial (le feu n'occupe souvent qu'une fraction de l'image satellite), nous combinons la perte d'entropie croisée binaire ($\mathcal{L}_{\text{BCE}}$) et la perte de Dice ($\mathcal{L}_{\text{Dice}}$) :

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}^{\text{cls}}(y, \hat{y}) + 1.2 \times \left( 0.5\,\mathcal{L}_{\text{BCE}}^{\text{seg}}(M, \hat{M}) + 0.5\,\mathcal{L}_{\text{Dice}}(M, \hat{M}) \right)$$

où le coefficient de Dice mesure le recouvrement spatial entre la zone prédite $\hat{M}$ et la vérité terrain $M$ :

$$\text{Dice}(M, \hat{M}) = \frac{2 \sum_i M_i \hat{M}_i + \epsilon}{\sum_i M_i + \sum_i \hat{M}_i + \epsilon}$$

---

## 4. Tâche 3 : Sauvegarde du Modèle et Fonction `predict(image_path)` (`predict.py`)
Conformément au cahier des charges, le modèle entraîné est sauvegardé sous deux formats standards dans le dossier `models/` :
- **`models/fire_cnn_unet.pkl`** : Archive sérialisée Python (`pickle`) contenant l'architecture, les poids appris et les seuils de décision.
- **`models/fire_cnn_unet.h5`** : Conteneur hiérarchique **HDF5** (`h5py`) compressé contenant les tenseurs de poids et les métadonnées JSON.

### Interface d'intégration pour l'équipe
La fonction `predict(image_path)` dans `predict.py` prend en entrée le chemin d'une image satellite et retourne un dictionnaire directement exploitable par le modèle **LSTM de Mondir** et l'**Arbre de Décision** dans `main.py` :

```python
from predict import predict

resultat = predict("chemin/vers/image_satellite.png")
# Exemple de retour :
# {
#   "fire_detected": "Oui",
#   "has_fire": True,
#   "confidence": 0.9962,
#   "location": {
#       "zone": "Nord-Est",
#       "center_xy": (84, 39),
#       "normalized_coords": (0.6614, 0.3071),
#       "gps_coords": (36.76929, 5.06614),
#       "bbox": [66, 22, 102, 57],
#       "affected_area_pct": 6.45
#   },
#   "summary": "Feu détecté : OUI (Confiance : 99.6%) | Localisation : Secteur Nord-Est...",
#   "segmentation_mask": np.ndarray (H, W)
# }
```

---

## 5. Mission Secondaire : Support Data (`data_support.py`)
Afin de faciliter le travail de **Mondir (RNN/LSTM)** et du **Responsable Système d'Alerte (Arbre de Décision)**, le script `data_support.py` génère le dataset unifié `dataset/global_fire_weather_dataset.csv` (200 échantillons synchronisés) associant :
- Les sorties du CNN (`fire_detected`, `confidence`, `location_zone`, `latitude`, `longitude`, `affected_area_pct`),
- Les relevés météorologiques (`temperature_c`, `humidity_pct`, `wind_speed_kmh`, `wind_direction`),
- Les cibles de propagation (`propagation_direction`, `propagation_speed_kmh`),
- Les indicateurs de risque et paliers d'alerte (`towards_populated_zone`, `alert_level` $\in \{\text{Vert}, \text{Orange}, \text{Rouge}, \text{Évacuation}\}$).
