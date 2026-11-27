"""
Module d'Inférence du Système d'Alerte par Arbre de Décision (Tâche 2 - Asma)
=============================================================================
Responsable : Asma Dadou (Système d'Alerte, Aide à la Décision & Intégration)

Fournit la fonction officielle `donner_alerte(...)` traduisant les sorties du CNN
(Salim), du Bi-LSTM (Moundir) et des capteurs météo en 4 paliers normalisés :
  - 0 : 🟢 VERT
  - 1 : 🟠 ORANGE
  - 2 : 🔴 ROUGE
  - 3 : 🚨 ÉVACUATION
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd

from contracts import (
    ALERT_CODE_TO_BADGE,
    ALERT_CODE_TO_LABEL,
    ALERT_RECOMMENDATIONS,
    DECISION_TREE_FEATURES,
    AlertDecisionContract,
    validate_alert_inputs,
)

_CACHED_TREE = None
_CACHED_TREE_PATH: Optional[str] = None


def _load_alert_tree(model_path: str | Path = "models/mon_arbre_alerte.pkl"):
    """Charge l'arbre de décision en mémoire (avec entraînement automatique si absent)."""
    global _CACHED_TREE, _CACHED_TREE_PATH

    base_dir = Path(__file__).resolve().parent
    resolved = Path(model_path) if Path(model_path).is_absolute() else (base_dir / model_path)

    if not resolved.is_file():
        from train_tree import train_alert_decision_tree

        train_alert_decision_tree(save_path=resolved)

    resolved_str = str(resolved.resolve())
    if _CACHED_TREE is None or _CACHED_TREE_PATH != resolved_str:
        _CACHED_TREE = joblib.load(resolved)
        _CACHED_TREE_PATH = resolved_str

    return _CACHED_TREE


def _estimate_fallback_speed(feu: int, vent: float, humidite: float) -> float:
    """Estime la vitesse de propagation (km/h) si l'appelant ne fournit que 3 paramètres."""
    if feu == 0:
        return 0.0
    wind_term = 0.075 * (max(0.0, vent) ** 1.15)
    moisture_term = max(0.4, (100.0 - humidite) / 45.0)
    return round(float(np.clip(wind_term * moisture_term * 0.95, 0.2, 15.0)), 2)


def donner_alerte(
    feu: int,
    vent: float,
    humidite: float,
    vitesse_propagation: Optional[float] = None,
    model_path: str | Path = "models/mon_arbre_alerte.pkl",
    return_contract: bool = False,
) -> Union[str, Dict[str, Any]]:
    """
    Évalue le niveau d'alerte opérationnel via l'Arbre de Décision entraîné.

    Compatible avec :
      - L'appel complet 4 entrées (Figure 1 du rapport) :
        `donner_alerte(feu, vent, humidite, vitesse_propagation=res_lstm["propagation_speed"])`
      - L'appel historique 3 entrées :
        `donner_alerte(feu, vent, humidite)`
    """
    arbre = _load_alert_tree(model_path=model_path)

    feu_pre = 1 if int(round(float(feu))) >= 1 else 0
    if vitesse_propagation is None:
        vit_in = _estimate_fallback_speed(feu_pre, float(vent), float(humidite))
    else:
        vit_in = float(vitesse_propagation)

    feu_c, vent_c, hum_c, vit_c = validate_alert_inputs(
        feu=feu_pre,
        vent=float(vent),
        humidite=float(humidite),
        vitesse_propagation=vit_in,
    )

    expected_cols = list(getattr(arbre, "feature_names_in_", DECISION_TREE_FEATURES))
    if len(expected_cols) == 4:
        X_df = pd.DataFrame([[feu_c, vent_c, hum_c, vit_c]], columns=expected_cols)
    else:
        X_df = pd.DataFrame([[feu_c, vent_c, hum_c]], columns=expected_cols)

    prediction = int(arbre.predict(X_df)[0])
    badge = ALERT_CODE_TO_BADGE[prediction]

    if return_contract:
        contract = AlertDecisionContract(
            alert_code=prediction,
            alert_label=ALERT_CODE_TO_LABEL[prediction],
            alert_badge=badge,
            recommendation=ALERT_RECOMMENDATIONS[prediction],
            inputs_used={
                "feu": float(feu_c),
                "vent": round(vent_c, 2),
                "humidite": round(hum_c, 2),
                "vitesse_propagation": round(vit_c, 2),
            },
        )
        return contract.to_dict()

    return badge
