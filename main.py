"""
Pipeline d'Intégration Multi-Agents Unifié (`main.py` — Tâche 2 Asma)
=====================================================================
Orchestre la chaîne complète de détection, prévision temporelle et décision :
  1. Salim   (CNN / U-Net)        : `predict(image_path)`
  2. Moundir (RNN / Bi-LSTM)      : `predict_propagation(location, wind, humidity, temperature)`
  3. Asma    (Arbre de Décision)  : `donner_alerte(feu, vent, humidite, vitesse_propagation)`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from contracts import (
    ALERT_RECOMMENDATIONS,
    validate_cnn_output,
    validate_lstm_output,
)
from predict import predict
from predict_alert import donner_alerte
from predict_propagation import predict_propagation


def run_pipeline(
    image_path: str | Path,
    wind_speed: float,
    wind_dir: str,
    humidity: float,
    temperature: float = 28.0,
    save_visualization: Optional[str | Path] = None,
    return_details: bool = False,
) -> Union[str, Dict[str, Any]]:
    """
    Exécute le pipeline multi-agents complet.

    Args:
        image_path: Chemin de l'image satellite / aérienne.
        wind_speed: Vitesse du vent (km/h).
        wind_dir: Direction cardinale du vent (ex: "Sud-Ouest").
        humidity: Humidité relative (%).
        temperature: Température ambiante (°C).
        save_visualization: Chemin optionnel pour sauvegarder l'image segmentée annotée.
        return_details: Si True, retourne le rapport structuré complet des 3 agents
            (utilisé par le tableau de bord Streamlit `app.py`). Si False (défaut),
            retourne le palier d'alerte textuel officiel.
    """
    print("--- 1. Analyse de l'image (Salim) ---")
    res_cnn = validate_cnn_output(
        predict(image_path, save_visualization=save_visualization)
    )

    # Si Salim détecte qu'il n'y a pas de feu, arrêt immédiat en palier VERT
    if res_cnn["fire_detected"] == "Non":
        alert_str = "🟢 VERT (Aucun feu)"
        if return_details:
            decision_info = donner_alerte(
                feu=0,
                vent=wind_speed,
                humidite=humidity,
                vitesse_propagation=0.0,
                return_contract=True,
            )
            return {
                "alert": alert_str,
                "alert_badge": decision_info["alert_badge"],
                "alert_label": decision_info["alert_label"],
                "alert_code": decision_info["alert_code"],
                "recommendation": ALERT_RECOMMENDATIONS[0],
                "cnn": res_cnn,
                "lstm": None,
                "decision": decision_info,
                "contract_valid": True,
            }
        return alert_str

    print("--- 2. Analyse de la propagation (Moundir) ---")
    res_lstm = validate_lstm_output(
        predict_propagation(
            location=res_cnn["location"],
            wind={"speed": wind_speed, "direction": wind_dir},
            humidity=humidity,
            temperature=temperature,
        )
    )

    print("--- 3. Décision (Asma) ---")
    feu_binaire = 1 if res_cnn["fire_detected"] == "Oui" else 0
    decision_info = donner_alerte(
        feu=feu_binaire,
        vent=wind_speed,
        humidite=humidity,
        vitesse_propagation=res_lstm["propagation_speed"],
        return_contract=True,
    )
    alerte = decision_info["alert_badge"]

    if return_details:
        return {
            "alert": alerte,
            "alert_badge": alerte,
            "alert_label": decision_info["alert_label"],
            "alert_code": decision_info["alert_code"],
            "recommendation": decision_info["recommendation"],
            "cnn": res_cnn,
            "lstm": res_lstm,
            "decision": decision_info,
            "contract_valid": True,
        }

    return alerte


if __name__ == "__main__":
    print("Démarrage du test...")
    image = "dataset/satellite_samples/images/fire/sat_fire_0001.png"
    resultat = run_pipeline(image, 38.0, "Sud-Ouest", 22.0, 34.0)
    print("RÉSULTAT :", resultat)
