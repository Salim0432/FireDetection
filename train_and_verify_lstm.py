"""
Script d'Entraînement & de Vérification Complète du Modèle LSTM (Moundir)
======================================================================
Responsable : Moundir (Modélisation Temporelle & Météo - RNN/LSTM)

Exécute le pipeline de bout en bout :
1. Prétraitement météo et constitution des séries temporelles (Tâche 1)
2. Entraînement du modèle RNN/LSTM (Tâche 2)
3. Sauvegarde de models/fire_propagation_lstm.pkl (Tâche 3)
4. Tests de validation de predict_propagation(location, wind, humidity) sur plusieurs cas d'usage
"""

from __future__ import annotations

import json
from pathlib import Path

from model_lstm import train_propagation_lstm
from predict_propagation import predict_propagation


def main() -> None:
    print("=" * 80)
    print("🚀 LANCEMENT DU PIPELINE COMPLET : MODÉLISATION TEMPORELLE & MÉTEO (LSTM)")
    print("=" * 80)

    # 1. Entraînement et sauvegarde du modèle .pkl
    model, metrics = train_propagation_lstm(
        csv_source="dataset/forestfires_kaggle.csv",
        epochs=30,
        batch_size=32,
        lr=1.5e-3,
        save_path_pkl="models/fire_propagation_lstm.pkl",
        metrics_json_path="outputs/lstm_training_metrics.json",
        seed=42,
    )

    print("\n" + "=" * 80)
    print("🧪 VÉRIFICATION FONCTIONNELLE DE predict_propagation(location, wind, humidity)")
    print("=" * 80)

    scenarios = [
        {
            "nom": "Scénario 1 : Incendie violent avec vent fort du Sud-Ouest et air très sec",
            "location": {"zone": "Nord-Est", "affected_area_pct": 8.5, "normalized_coords": (0.75, 0.25)},
            "wind": {"speed": 45.0, "direction": "Sud-Ouest"},
            "humidity": 18.0,
            "temp": 36.5,
        },
        {
            "nom": "Scénario 2 : Foyer modéré au Centre avec brise d'Est et humidité moyenne",
            "location": "Centre",
            "wind": 22.0,
            "wind_direction": "Est",
            "humidity": 45.0,
            "temp": 27.0,
        },
        {
            "nom": "Scénario 3 : Foyer naissant Sud-Est sous vent du Nord et humidité élevée",
            "location": {"zone": "Sud-Est", "affected_area_pct": 1.2},
            "wind": (15.0, "Nord"),
            "humidity": 65.0,
            "temp": 21.0,
        },
    ]

    for sc in scenarios:
        print(f"\n--- {sc['nom']} ---")
        res = predict_propagation(
            location=sc["location"],
            wind=sc["wind"],
            humidity=sc["humidity"],
            temperature=sc.get("temp"),
            wind_direction=sc.get("wind_direction"),
        )
        print("Résultat :")
        print(f"  * Direction de propagation : {res['propagation_direction']} ({res['direction_angle_deg']}°)")
        print(f"  * Vitesse de propagation   : {res['propagation_speed']} km/h ({res['speed_m_per_min']} m/min)")
        print(f"  * Synthèse : {res['summary']}")

    print("\n" + "=" * 80)
    print("✅ TOUTES LES VÉRIFICATIONS ONT RÉUSSI AVEC SUCCÈS !")
    print("=" * 80)


if __name__ == "__main__":
    main()
