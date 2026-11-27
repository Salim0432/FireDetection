"""
Contrat d'Interface Multi-Agents (Projet 5 — Surveillance & Confinement des Feux de Forêt)
==========================================================================================
Définit et valide formellement les structures d'échange de données entre les 3 agents :
  1. Agent 1 (Salim   — CNN / U-Net)         : Image -> Détection & Localisation spatiale
  2. Agent 2 (Moundir — RNN / Bi-LSTM)       : Localisation + Météo -> Vitesse & Direction
  3. Agent 3 (Asma    — Arbre de Décision)   : Feu + Météo + Vitesse -> Palier d'Alerte
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Référentiels Normalisés du Projet
# ---------------------------------------------------------------------------
CARDINAL_DIRECTIONS_8: Tuple[str, ...] = (
    "Nord",
    "Nord-Est",
    "Est",
    "Sud-Est",
    "Sud",
    "Sud-Ouest",
    "Ouest",
    "Nord-Ouest",
)

ALERT_CODE_TO_BADGE: Dict[int, str] = {
    0: "🟢 VERT",
    1: "🟠 ORANGE",
    2: "🔴 ROUGE",
    3: "🚨 ÉVACUATION",
}

ALERT_CODE_TO_LABEL: Dict[int, str] = {
    0: "Vert",
    1: "Orange",
    2: "Rouge",
    3: "Évacuation",
}

ALERT_LABEL_TO_CODE: Dict[str, int] = {
    "Vert": 0,
    "VERT": 0,
    "🟢 VERT": 0,
    "Orange": 1,
    "ORANGE": 1,
    "🟠 ORANGE": 1,
    "Rouge": 2,
    "ROUGE": 2,
    "🔴 ROUGE": 2,
    "Évacuation": 3,
    "Evacuation": 3,
    "ÉVACUATION": 3,
    "EVACUATION": 3,
    "🚨 ÉVACUATION": 3,
}

ALERT_RECOMMENDATIONS: Dict[int, str] = {
    0: "Surveillance normale des patrouilles et veille satellite de routine.",
    1: "Mobilisation des vigies locales, surveillance renforcée sans alarme publique.",
    2: "Pré-positionnement des bombardiers d'eau et interdiction d'accès aux massifs.",
    3: "Déclenchement de la sirène, fermeture des axes routiers et évacuation d'urgence.",
}

DECISION_TREE_FEATURES: List[str] = [
    "feu",
    "vent",
    "humidite",
    "vitesse_propagation",
]


# ---------------------------------------------------------------------------
# Schéma de Décision Typé (Dataclass)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AlertDecisionContract:
    alert_code: int
    alert_label: str
    alert_badge: str
    recommendation: str
    inputs_used: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Fonctions de Validation de Contrat Inter-Agents
# ---------------------------------------------------------------------------
def validate_cnn_output(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Valide le contrat de sortie de l'Agent 1 (Salim - predict.py)."""
    required_keys = {"fire_detected", "has_fire", "confidence", "location", "summary", "segmentation_mask"}
    missing = required_keys - set(payload.keys())
    if missing:
        raise ValueError(f"[Contrat CNN invalide] Clés manquantes : {sorted(missing)}")

    if payload["fire_detected"] not in {"Oui", "Non"}:
        raise ValueError(f"[Contrat CNN invalide] fire_detected={payload['fire_detected']!r} attendu dans {{'Oui', 'Non'}}")

    expected_bool = payload["fire_detected"] == "Oui"
    if bool(payload["has_fire"]) != expected_bool:
        raise ValueError("[Contrat CNN invalide] Incohérence entre fire_detected et has_fire.")

    conf = float(payload["confidence"])
    if not (0.0 <= conf <= 1.0):
        raise ValueError(f"[Contrat CNN invalide] confidence={conf} hors de [0, 1].")

    loc = payload["location"]
    loc_keys = {"zone", "center_xy", "normalized_coords", "bbox", "affected_area_pct"}
    missing_loc = loc_keys - set(loc.keys())
    if missing_loc:
        raise ValueError(f"[Contrat CNN invalide] Clés location manquantes : {sorted(missing_loc)}")

    if float(loc["affected_area_pct"]) < 0.0 or float(loc["affected_area_pct"]) > 100.0:
        raise ValueError(f"[Contrat CNN invalide] affected_area_pct={loc['affected_area_pct']} hors de [0, 100].")

    return payload


def validate_lstm_output(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Valide le contrat de sortie de l'Agent 2 (Moundir - predict_propagation.py)."""
    required_keys = {
        "direction",
        "propagation_direction",
        "speed",
        "propagation_speed",
        "direction_angle_deg",
        "speed_kmh",
        "speed_m_per_min",
        "summary",
    }
    missing = required_keys - set(payload.keys())
    if missing:
        raise ValueError(f"[Contrat LSTM invalide] Clés manquantes : {sorted(missing)}")

    if payload["propagation_direction"] not in CARDINAL_DIRECTIONS_8:
        raise ValueError(
            f"[Contrat LSTM invalide] propagation_direction={payload['propagation_direction']!r} "
            f"attendu dans {CARDINAL_DIRECTIONS_8}"
        )

    if float(payload["propagation_speed"]) < 0.0:
        raise ValueError(f"[Contrat LSTM invalide] propagation_speed négative : {payload['propagation_speed']}")

    angle = float(payload["direction_angle_deg"])
    if not (0.0 <= angle < 360.0):
        raise ValueError(f"[Contrat LSTM invalide] direction_angle_deg={angle} hors de [0, 360[.")

    return payload


def validate_alert_inputs(
    feu: int,
    vent: float,
    humidite: float,
    vitesse_propagation: float,
) -> Tuple[int, float, float, float]:
    """Normalise et valide les entrées injectées dans l'Agent 3 (Asma - Arbre de Décision)."""
    feu_clean = 1 if int(round(float(feu))) >= 1 else 0
    vent_clean = max(0.0, float(vent))
    hum_clean = min(100.0, max(0.0, float(humidite)))
    vit_clean = 0.0 if feu_clean == 0 else max(0.0, float(vitesse_propagation))
    return feu_clean, vent_clean, hum_clean, vit_clean


def compute_reference_alert_code(
    feu: int,
    vent_kmh: float,
    humidite_pct: float,
    vitesse_kmh: float,
) -> int:
    """
    Règle opérationnelle de référence pour les paliers d'alerte :
      - 0 (Vert)       : Aucun feu détecté (feu == 0)
      - 3 (Évacuation) : Feu actif + vent violent (> 40 km/h, ou >= 38 km/h avec RH <= 30% / vitesse >= 6.0 km/h,
                         ou sécheresse aride RH <= 20% avec vent >= 30 km/h, ou vitesse >= 7.2 km/h)
      - 2 (Rouge)      : Feu actif + vent moyen à fort (>= 20 km/h) ou combustible sec (RH <= 40%)
                         ou vitesse de propagation soutenue (>= 2.8 km/h)
      - 1 (Orange)     : Feu détecté + vent modéré (< 20 km/h), humidité élevée (> 40%) et propagation lente
    """
    if int(feu) == 0:
        return 0

    if (
        vent_kmh > 40.0
        or (vent_kmh >= 38.0 and (humidite_pct <= 30.0 or vitesse_kmh >= 6.0))
        or (humidite_pct <= 20.0 and vent_kmh >= 30.0)
        or vitesse_kmh >= 7.2
    ):
        return 3

    if (
        vent_kmh >= 20.0
        or humidite_pct <= 40.0
        or vitesse_kmh >= 2.8
    ):
        return 2

    return 1
