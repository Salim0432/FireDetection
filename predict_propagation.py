"""
Module d'Inférence de la Propagation Temporelle (Tâche 3 - Moundir)
=================================================================
Responsable : Moundir (Modélisation Temporelle & Météo - RNN/LSTM)

Fournit la fonction officielle exigée par le cahier des charges :
    predict_propagation(location, wind, humidity, ...)

Cette fonction est conçue pour s'interfacer directement avec :
- Les sorties de Salim (`predict(image_path)` -> `location`)
- Les stations météo en temps réel (vent, humidité, température)
- Le Système d'Alerte & Confinement (Arbre de Décision / `main.py` / Streamlit)

Retourne :
- La direction de propagation (Secteur cardinal, ex: "Nord-Est", et angle géométrique)
- La vitesse de propagation (km/h et m/min)
- L'expansion temporelle heure par heure pour les prochaines heures
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from contracts import validate_lstm_output
from model_lstm import FirePropagationLSTM, load_lstm_model
from temporal_preprocessing import (
    CARDINAL_8_SECTORS,
    FEATURE_COLUMNS,
    TabularStandardScaler,
    angle_to_cardinal,
    angle_to_sin_cos,
    compute_fire_weather_index,
    direction_to_angle,
    sin_cos_to_angle,
)


# Cache mémoire du modèle pour inférence instantanée répétée
_CACHED_LSTM_MODEL: Optional[FirePropagationLSTM] = None
_CACHED_X_SCALER: Optional[TabularStandardScaler] = None
_CACHED_Y_SCALER: Optional[TabularStandardScaler] = None
_CACHED_LSTM_PAYLOAD: Optional[Dict[str, Any]] = None
_CACHED_LSTM_PATH: Optional[str] = None


def _parse_location(location: Any) -> Tuple[str, float, float, float, Optional[Tuple[float, float]]]:
    """
    Extrait les coordonnées spatiales à partir de formats hétérogènes :
    - Dictionnaire de Salim (`predict.py`)
    - Chaîne cardinale ("Nord-Est", "Centre")
    - Tuple GPS ou pixel
    """
    zone = "Centre"
    norm_x, norm_y = 0.5, 0.5
    affected_area_pct = 3.5
    gps_coords = (36.75, 5.05)

    if isinstance(location, dict):
        zone = str(location.get("zone", "Centre"))
        affected_area_pct = float(location.get("affected_area_pct", 3.5))

        if "normalized_coords" in location and location["normalized_coords"] is not None:
            norm_x, norm_y = float(location["normalized_coords"][0]), float(location["normalized_coords"][1])
        elif "center_xy" in location and location["center_xy"] is not None:
            norm_x = float(location["center_xy"][0]) / 128.0
            norm_y = float(location["center_xy"][1]) / 128.0

        if "gps_coords" in location and location["gps_coords"] is not None:
            gps_coords = (float(location["gps_coords"][0]), float(location["gps_coords"][1]))

    elif isinstance(location, (list, tuple)) and len(location) >= 2:
        val1, val2 = float(location[0]), float(location[1])
        if 0.0 <= val1 <= 1.0 and 0.0 <= val2 <= 1.0:
            norm_x, norm_y = val1, val2
        else:
            gps_coords = (val1, val2)
            norm_x, norm_y = 0.5, 0.5

    elif isinstance(location, str):
        zone = location.strip()
        # Estimation approximative des coordonnées normalisées depuis le secteur
        zone_map = {
            "Nord": (0.5, 0.2),
            "Nord-Est": (0.8, 0.2),
            "Est": (0.8, 0.5),
            "Sud-Est": (0.8, 0.8),
            "Sud": (0.5, 0.8),
            "Sud-Ouest": (0.2, 0.8),
            "Ouest": (0.2, 0.5),
            "Nord-Ouest": (0.2, 0.2),
            "Centre": (0.5, 0.5),
        }
        for k, coords in zone_map.items():
            if k.lower() in zone.lower():
                norm_x, norm_y = coords
                break

    return zone, norm_x, norm_y, affected_area_pct, gps_coords


def _parse_wind(wind: Any, default_direction: Optional[str] = None) -> Tuple[float, float, str]:
    """
    Extrait la vitesse du vent (km/h) et sa direction (degrés et texte)
    depuis un nombre, un dictionnaire ou un tuple.
    """
    speed_kmh = 20.0
    direction_deg = 45.0
    direction_str = default_direction or "Nord-Est"

    if isinstance(wind, (int, float)):
        speed_kmh = float(wind)
        if default_direction is not None:
            direction_deg = direction_to_angle(default_direction)
            direction_str = angle_to_cardinal(direction_deg)
        else:
            direction_deg = 45.0
            direction_str = "Nord-Est"

    elif isinstance(wind, dict):
        speed_kmh = float(wind.get("speed", wind.get("wind_speed_kmh", 20.0)))
        raw_dir = wind.get("direction", wind.get("wind_direction", default_direction or "Nord-Est"))
        direction_deg = direction_to_angle(raw_dir)
        direction_str = angle_to_cardinal(direction_deg)

    elif isinstance(wind, (list, tuple)) and len(wind) >= 2:
        speed_kmh = float(wind[0])
        direction_deg = direction_to_angle(wind[1])
        direction_str = angle_to_cardinal(direction_deg)

    return max(0.0, speed_kmh), direction_deg, direction_str


def predict_propagation(
    location: Any,
    wind: Any,
    humidity: Union[float, int],
    temperature: Optional[float] = None,
    wind_direction: Optional[str] = None,
    model_path: str | Path = "models/fire_propagation_lstm.pkl",
    historical_observations: Optional[List[Dict[str, float]]] = None,
    time_horizon_hours: int = 3,
) -> Dict[str, Any]:
    """
    Fonction officielle Tâche 3 - Moundir :
    Prédit la vitesse et la direction de propagation du feu à partir de la localisation,
    des conditions de vent et de l'humidité.

    Paramètres :
    ------------
    location : Dict (sortie Salim), str ("Nord-Est"), ou Tuple (lat, lon)
    wind : float (vitesse km/h), Dict {'speed': ..., 'direction': ...}, ou Tuple
    humidity : float (humidité relative en %)
    temperature : float optionnel (°C, défaut = 28.0°C)
    wind_direction : str optionnel (ex: "Sud-Ouest") si absent du paramètre wind
    model_path : chemin vers models/fire_propagation_lstm.pkl
    time_horizon_hours : horizon d'anticipation temporelle (défaut = 3h)

    Retourne :
    ----------
    Dictionnaire complet contenant :
    - 'propagation_direction' : Secteur cardinal prédit (ex: "Nord-Est")
    - 'propagation_speed'     : Vitesse de propagation en km/h (ex: 3.42)
    - 'direction'             : Alias direct de propagation_direction
    - 'speed'                 : Alias direct de propagation_speed
    - 'direction_angle_deg'   : Angle continu en degrés [0, 360[
    - 'speed_kmh'             : Vitesse en km/h
    - 'speed_m_per_min'       : Vitesse en mètres par minute
    - 'expansion_hourly'      : Liste des vitesses/expansions sur t+1h, t+2h, t+3h
    - 'expansion_next_hours'  : Dictionnaire détaillé heure par heure
    - 'fire_weather_index'    : Indice de danger météo calculé (FWI/ISI)
    - 'location_summary'      : Localisation spatiale prise en compte
    - 'summary'               : Synthèse textuelle formatée
    """
    global _CACHED_LSTM_MODEL, _CACHED_X_SCALER, _CACHED_Y_SCALER, _CACHED_LSTM_PAYLOAD, _CACHED_LSTM_PATH

    model_p = Path(model_path)
    if not model_p.is_file():
        # Fallback automatique vers entraînement rapide si le modèle n'existe pas encore
        from model_lstm import train_propagation_lstm

        print(f"[Info] Modèle {model_p} non trouvé. Lancement automatique de l'entraînement...")
        train_propagation_lstm(save_path_pkl=model_p, epochs=20)

    # Chargement en cache
    resolved_str = str(model_p.resolve())
    if _CACHED_LSTM_MODEL is None or _CACHED_LSTM_PATH != resolved_str:
        _CACHED_LSTM_MODEL, _CACHED_X_SCALER, _CACHED_Y_SCALER, _CACHED_LSTM_PAYLOAD = load_lstm_model(model_p)
        _CACHED_LSTM_PATH = resolved_str

    model = _CACHED_LSTM_MODEL
    x_scaler = _CACHED_X_SCALER
    y_scaler = _CACHED_Y_SCALER

    # 1. Parsing robuste des entrées
    zone_name, norm_x, norm_y, affected_area, gps = _parse_location(location)
    wind_speed, wind_deg, wind_cardinal = _parse_wind(wind, default_direction=wind_direction)
    rh_pct = float(np.clip(humidity, 5.0, 100.0))
    temp_c = float(temperature if temperature is not None else 28.0)

    # 2. Construction de la séquence temporelle pour le LSTM (longueur = 4 pas)
    seq_len = int(_CACHED_LSTM_PAYLOAD["config"].get("sequence_length", 4))
    sequence_features: List[List[float]] = []

    # Si un historique réel est fourni, on l'utilise (complété à gauche si < seq_len),
    # sinon on modélise l'inertie physique sur les pas temporels t-3, t-2, t-1, t
    if historical_observations:
        obs_list = list(historical_observations)[-seq_len:]
        while len(obs_list) < seq_len:
            obs_list.insert(0, obs_list[0])
        for obs in obs_list:
            obs_temp = float(obs.get("temperature_c", obs.get("temperature", temp_c)))
            obs_rh = float(np.clip(obs.get("humidity_pct", obs.get("humidity", rh_pct)), 5.0, 100.0))
            obs_wind = max(0.0, float(obs.get("wind_speed_kmh", obs.get("wind_speed", wind_speed))))
            obs_dir = direction_to_angle(obs.get("wind_direction", obs.get("direction", wind_deg)))
            w_sin, w_cos = angle_to_sin_cos(obs_dir)
            fwi = compute_fire_weather_index(obs_temp, obs_rh, obs_wind)
            obs_area = max(0.1, float(obs.get("affected_area_pct", affected_area)))
            default_hist_speed = max(0.2, (obs_wind * 0.08) * (1.0 + (50.0 - obs_rh) / 60.0))
            obs_speed = max(0.1, float(obs.get("historical_speed_kmh", obs.get("speed", default_hist_speed))))
            sequence_features.append(
                [obs_temp, obs_rh, obs_wind, w_sin, w_cos, fwi, obs_area, norm_x, norm_y, obs_speed]
            )
    else:
        for step_back in range(seq_len - 1, -1, -1):
            dt_factor = 1.0 - (step_back * 0.08)
            past_temp = temp_c - (step_back * 0.4)
            past_rh = min(100.0, rh_pct + (step_back * 1.5))
            past_wind = max(5.0, wind_speed * dt_factor)
            past_dir_deg = (wind_deg + (step_back * 2.0)) % 360.0

            w_sin, w_cos = angle_to_sin_cos(past_dir_deg)
            fwi = compute_fire_weather_index(past_temp, past_rh, past_wind)
            hist_area = max(0.5, affected_area * (0.85 ** step_back))
            hist_speed = max(0.2, (past_wind * 0.08) * (1.0 + (50.0 - past_rh) / 60.0))

            feat_vector = [
                past_temp,
                past_rh,
                past_wind,
                w_sin,
                w_cos,
                fwi,
                hist_area,
                norm_x,
                norm_y,
                hist_speed,
            ]
            sequence_features.append(feat_vector)

    # 3. Standardisation et inférence PyTorch
    x_raw = np.array([sequence_features], dtype=np.float32)  # shape (1, seq_len, n_features)
    n_feat = x_raw.shape[2]
    x_scaled = x_scaler.transform(x_raw.reshape(-1, n_feat)).reshape(1, seq_len, n_feat)

    x_tensor = torch.from_numpy(x_scaled)
    device = next(model.parameters()).device
    x_tensor = x_tensor.to(device)

    with torch.no_grad():
        preds_scaled = model(x_tensor).cpu().numpy()

    # 4. Dénormalisation des sorties
    preds_denorm = y_scaler.inverse_transform(preds_scaled)[0]

    # Extraction des variables physiques
    raw_speed = float(max(0.1, preds_denorm[0]))
    sin_dir = float(preds_denorm[1])
    cos_dir = float(preds_denorm[2])
    exp_1h = float(max(0.1, preds_denorm[3]))
    exp_2h = float(max(exp_1h * 1.02, preds_denorm[4]))
    exp_3h = float(max(exp_2h * 1.02, preds_denorm[5]))

    # Calcul de l'angle et du secteur cardinal
    pred_angle = sin_cos_to_angle(sin_dir, cos_dir)
    pred_cardinal = angle_to_cardinal(pred_angle)
    speed_m_min = round(raw_speed * 1000.0 / 60.0, 1)

    # Multi-horizon dict
    expansions_hourly = [round(raw_speed, 2), round(exp_1h, 2), round(exp_2h, 2), round(exp_3h, 2)]

    fwi_current = round(compute_fire_weather_index(temp_c, rh_pct, wind_speed), 2)

    summary_text = (
        f"Direction: {pred_cardinal} ({pred_angle:.1f}°) | "
        f"Vitesse: {raw_speed:.2f} km/h ({speed_m_min} m/min) | "
        f"Localisation: {zone_name} | "
        f"Météo: Vent={wind_speed:.1f} km/h ({wind_cardinal}), Humidité={rh_pct:.1f}%, Temp={temp_c:.1f}°C"
    )

    result = {
        # Champs officiels exigés par le cahier des charges
        "direction": pred_cardinal,
        "propagation_direction": pred_cardinal,
        "speed": round(raw_speed, 2),
        "propagation_speed": round(raw_speed, 2),

        # Informations physiques enrichies
        "direction_angle_deg": round(pred_angle, 1),
        "speed_kmh": round(raw_speed, 2),
        "speed_m_per_min": speed_m_min,
        "expansion_hourly": expansions_hourly[: time_horizon_hours + 1],
        "expansion_next_hours": {
            "+1h": round(exp_1h, 2),
            "+2h": round(exp_2h, 2),
            "+3h": round(exp_3h, 2),
        },
        "fire_weather_index": fwi_current,
        "input_conditions": {
            "location_zone": zone_name,
            "normalized_coords": (round(norm_x, 4), round(norm_y, 4)),
            "gps_coords": gps,
            "wind_speed_kmh": wind_speed,
            "wind_direction": wind_cardinal,
            "wind_angle_deg": wind_deg,
            "humidity_pct": rh_pct,
            "temperature_c": temp_c,
        },
        "summary": summary_text,
    }

    return validate_lstm_output(result)


if __name__ == "__main__":
    print("\n--- Test Direct de predict_propagation() ---")
    test_loc = {"zone": "Nord-Est", "affected_area_pct": 6.45, "gps_coords": (36.76, 5.06)}
    res = predict_propagation(location=test_loc, wind=35.0, humidity=25.0, wind_direction="Sud-Ouest")
    print(res["summary"])
    print(f"Direction : {res['propagation_direction']}")
    print(f"Vitesse   : {res['propagation_speed']} km/h")
