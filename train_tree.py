"""
Module d'Entraînement de l'Arbre de Décision — Système d'Alerte (Tâche 1 - Asma)
================================================================================
Responsable : Asma Dadou (Système d'Alerte, Aide à la Décision & Intégration)

Entraîne un classifieur CART (`DecisionTreeClassifier`) conforme au Rapport Général
(Figure 1 & Table 2) fusionnant :
  - `feu`                 : Détection binaire issue du CNN de Salim (0 = Non, 1 = Oui)
  - `vent`                : Vitesse du vent en km/h
  - `humidite`            : Humidité relative de l'air en %
  - `vitesse_propagation` : Vitesse de propagation prédite par le Bi-LSTM de Moundir (km/h)

Paliers d'alerte opérationnelle produits (Table 2 du rapport) :
  - 0 : 🟢 VERT       (Aucun feu détecté)
  - 1 : 🟠 ORANGE     (Feu détecté, vent modéré < 20 km/h, humidité élevée > 40%)
  - 2 : 🔴 ROUGE      (Feu actif, vent moyen à fort, combustible sec RH <= 40%)
  - 3 : 🚨 ÉVACUATION (Feu actif, vent violent >= 38-42 km/h ou conditions arides extrêmes)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from contracts import (
    ALERT_CODE_TO_LABEL,
    ALERT_LABEL_TO_CODE,
    DECISION_TREE_FEATURES,
    compute_reference_alert_code,
)


def _estimate_speed_from_weather(vent: float, humidite: float) -> float:
    """Estime physiquement la vitesse de propagation (km/h) à partir du vent et de l'humidité."""
    wind_term = 0.075 * (max(0.0, vent) ** 1.15)
    moisture_term = max(0.4, (100.0 - humidite) / 45.0)
    return round(float(np.clip(wind_term * moisture_term * 0.95, 0.2, 15.0)), 2)


def build_decision_tree_dataset(
    global_csv: str | Path = "dataset/global_fire_weather_dataset.csv",
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Construit le jeu d'entraînement complet combinant :
    1. Les observations réelles du dataset unifié `dataset/global_fire_weather_dataset.csv`
    2. Une grille opérationnelle complète couvrant tout le domaine météo [0..100 km/h, 5..95% RH]
       afin de garantir une monotonie physique stricte (ex: feu=0 -> toujours Vert quel que soit le vent).
    """
    rng = np.random.default_rng(seed)
    rows = []
    labels = []

    csv_path = Path(global_csv)
    if csv_path.is_file():
        df_csv = pd.read_csv(csv_path)
        for _, r in df_csv.iterrows():
            feu = 1 if str(r["fire_detected"]).strip() == "Oui" else 0
            vent = float(r["wind_speed_kmh"])
            hum = float(r["humidity_pct"])
            vit = 0.0 if feu == 0 else float(r["propagation_speed_kmh"])
            lbl_str = str(r["alert_level"]).strip()
            if lbl_str in ALERT_LABEL_TO_CODE:
                code = ALERT_LABEL_TO_CODE[lbl_str]
            else:
                code = compute_reference_alert_code(feu=feu, vent_kmh=vent, humidite_pct=hum, vitesse_kmh=vit)
            rows.append([feu, vent, hum, vit])
            labels.append(code)

    # Couverture systématique des cas sans feu (y compris tempêtes sèches -> toujours VERT)
    for vent in np.linspace(0.0, 95.0, 20):
        for hum in np.linspace(8.0, 95.0, 10):
            rows.append([0, round(float(vent), 1), round(float(hum), 1), 0.0])
            labels.append(0)

    # Couverture systématique des régimes d'incendie actif (Orange, Rouge, Évacuation)
    for vent in np.linspace(4.0, 85.0, 28):
        for hum in np.linspace(8.0, 88.0, 20):
            base_vit = _estimate_speed_from_weather(float(vent), float(hum))
            vit = round(max(0.2, base_vit + float(rng.normal(0.0, 0.25))), 2)
            code = compute_reference_alert_code(
                feu=1,
                vent_kmh=float(vent),
                humidite_pct=float(hum),
                vitesse_kmh=vit,
            )
            rows.append([1, round(float(vent), 1), round(float(hum), 1), vit])
            labels.append(code)

    df_X = pd.DataFrame(rows, columns=DECISION_TREE_FEATURES)
    y_arr = np.array(labels, dtype=int)
    return df_X, y_arr


def train_alert_decision_tree(
    save_path: str | Path = "models/mon_arbre_alerte.pkl",
    global_csv: str | Path = "dataset/global_fire_weather_dataset.csv",
    seed: int = 42,
) -> DecisionTreeClassifier:
    """Entraîne, évalue par validation croisée et sauvegarde l'Arbre de Décision."""
    df_X, y = build_decision_tree_dataset(global_csv=global_csv, seed=seed)

    X_train, X_val, y_train, y_val = train_test_split(
        df_X, y, test_size=0.2, random_state=seed, stratify=y
    )

    arbre = DecisionTreeClassifier(
        criterion="gini",
        max_depth=6,
        min_samples_leaf=2,
        random_state=seed,
    )
    arbre.fit(X_train, y_train)

    y_pred = arbre.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(arbre, df_X, y, cv=cv, scoring="accuracy")

    # Ré-entraînement final sur 100% du corpus validé pour déploiement optimal
    arbre.fit(df_X, y)

    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(arbre, out_file)

    print("=" * 72)
    print("🌳 ENTRAÎNEMENT DE L'ARBRE DE DÉCISION — SYSTÈME D'ALERTE (ASMA)")
    print("=" * 72)
    print(f"Échantillons totaux     : {len(df_X)} | Variables : {DECISION_TREE_FEATURES}")
    print(f"Accuracy Validation     : {val_acc * 100:.2f}%")
    print(f"Accuracy Cross-Val (5x) : {cv_scores.mean() * 100:.2f}% (±{cv_scores.std() * 100:.2f}%)")
    print("\nRapport de classification sur le jeu de validation :")
    target_names = [ALERT_CODE_TO_LABEL[i] for i in range(4)]
    print(classification_report(y_val, y_pred, target_names=target_names))
    print("Règles apprises par l'arbre :")
    print(export_text(arbre, feature_names=DECISION_TREE_FEATURES))
    print(f"✅ Modèle entraîné et sauvegardé dans : {out_file.resolve()}")
    return arbre


if __name__ == "__main__":
    train_alert_decision_tree()
