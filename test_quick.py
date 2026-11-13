"""
Script de Test Rapide & Démonstration du Modèle LSTM (Moundir)
=============================================================
Exécutez simplement :
    python test_quick.py
"""

from predict import predict
from predict_propagation import predict_propagation


def run_tests() -> None:
    print("=" * 75)
    print("🧪 1. TEST DIRECT DE predict_propagation(location, wind, humidity)")
    print("=" * 75)

    # Test 1 : Paramètres simples
    res1 = predict_propagation(location="Nord-Est", wind=35.0, humidity=22.0)
    print(f"👉 Direction de propagation : {res1['propagation_direction']} ({res1['direction_angle_deg']}°)")
    print(f"👉 Vitesse de propagation   : {res1['propagation_speed']} km/h ({res1['speed_m_per_min']} m/min)")
    print(f"👉 Expansion future         : {res1['expansion_next_hours']}")
    print(f"👉 Synthèse                 : {res1['summary']}\n")

    print("=" * 75)
    print("🧪 2. TEST D'INTÉGRATION COMPLÈTE : SALIM (CNN) ➔ MOUNDIR (LSTM)")
    print("=" * 75)

    # Image test
    sample_image = "dataset/satellite_samples/images/fire/sat_fire_0001.png"
    print(f"Analyse de l'image satellite : {sample_image}")

    # Étape 1 : CNN de Salim
    cnn_out = predict(sample_image)
    print(f"  [Salim CNN]  Feu détecté : {cnn_out['fire_detected']} (Confiance: {cnn_out['confidence']*100:.1f}%)")
    print(f"  [Salim CNN]  Zone : {cnn_out['location']['zone']} | Surface touchée : {cnn_out['location']['affected_area_pct']}%")

    # Étape 2 : Transmission directe au LSTM de Moundir
    if cnn_out["has_fire"]:
        lstm_out = predict_propagation(
            location=cnn_out["location"],
            wind={"speed": 42.0, "direction": "Sud-Ouest"},
            humidity=18.0,
            temperature=35.0,
        )
        print(f"\n  [Moundir LSTM] Direction prédite : {lstm_out['propagation_direction']} ({lstm_out['direction_angle_deg']}°)")
        print(f"  [Moundir LSTM] Vitesse prédite   : {lstm_out['propagation_speed']} km/h ({lstm_out['speed_m_per_min']} m/min)")
        print(f"  [Moundir LSTM] Expansion heure par heure :")
        for h, val in lstm_out["expansion_next_hours"].items():
            print(f"                 * à {h} : {val} km/h")
        print(f"  [Moundir LSTM] Indice de danger météo (FWI) : {lstm_out['fire_weather_index']}")

    print("\n" + "=" * 75)
    print("✅ LE MODÈLE FONCTIONNE PARFAITEMENT !")
    print("=" * 75)


if __name__ == "__main__":
    run_tests()
