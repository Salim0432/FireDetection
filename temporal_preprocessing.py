"""
Module de Traitement Météorologique & Préparation des Séries Temporelles (Tâche 1 & Support Data - Moundir)
=========================================================================================================
Responsable : Moundir (Modélisation Temporelle & Météo - RNN/LSTM)

Ce module assure :
1. Le prétraitement et le feature engineering des données météorologiques :
   - Température (°C)
   - Humidité relative (RH %)
   - Vent : vitesse (km/h) et direction angulaire / cardinale
2. La transformation cyclique des directions angulaires en composantes sin/cos
   pour éliminer les discontinuités à 0°/360° dans les réseaux récurrents.
3. La génération et le séquençage glissant (sliding window) des séries temporelles
   d'incendies simulant la dynamique temporelle de propagation selon le vent et l'humidité.
4. La normalisation robuste (StandardScaler) avec sérialisation des statistiques
   pour garantir une inférence déterministe sans data leakage.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constantes & Référentiel Géographique / Angulaire
# ---------------------------------------------------------------------------
CARDINAL_TO_ANGLE: Dict[str, float] = {
    "N": 0.0,
    "NORD": 0.0,
    "NORTH": 0.0,
    "NNE": 22.5,
    "NE": 45.0,
    "NORD-EST": 45.0,
    "NORTH-EAST": 45.0,
    "ENE": 67.5,
    "E": 90.0,
    "EST": 90.0,
    "EAST": 90.0,
    "ESE": 112.5,
    "SE": 135.0,
    "SUD-EST": 135.0,
    "SOUTH-EAST": 135.0,
    "SSE": 157.5,
    "S": 180.0,
    "SUD": 180.0,
    "SOUTH": 180.0,
    "SSW": 202.5,
    "SW": 225.0,
    "SUD-OUEST": 225.0,
    "SOUTH-WEST": 225.0,
    "WSW": 247.5,
    "W": 270.0,
    "O": 270.0,
    "OUEST": 270.0,
    "WEST": 270.0,
    "WNW": 292.5,
    "NW": 315.0,
    "NORD-OUEST": 315.0,
    "NORTH-WEST": 315.0,
    "NNW": 337.5,
    "CENTRE": 0.0,
    "AUCUNE": 0.0,
}

CARDINAL_8_SECTORS: List[Tuple[str, float, float]] = [
    ("Nord", 337.5, 22.5),
    ("Nord-Est", 22.5, 67.5),
    ("Est", 67.5, 112.5),
    ("Sud-Est", 112.5, 157.5),
    ("Sud", 157.5, 202.5),
    ("Sud-Ouest", 202.5, 247.5),
    ("Ouest", 247.5, 292.5),
    ("Nord-Ouest", 292.5, 337.5),
]

FEATURE_COLUMNS: List[str] = [
    "temperature_c",
    "humidity_pct",
    "wind_speed_kmh",
    "wind_sin",
    "wind_cos",
    "affected_area_pct",
    "loc_x_norm",
    "loc_y_norm",
    "historical_speed_kmh",
]

TARGET_COLUMNS: List[str] = [
    "target_speed_kmh",
    "target_dir_sin",
    "target_dir_cos",
]


# ---------------------------------------------------------------------------
# Utilitaires de Conversion Angulaire & Vectorielle
# ---------------------------------------------------------------------------
def direction_to_angle(direction: Union[str, float, int]) -> float:
    """Convertit une direction (chaîne textuelle ou valeur numérique) en angle en degrés [0, 360[."""
    if isinstance(direction, (int, float)):
        return float(direction) % 360.0

    raw_str = str(direction).strip()
    try:
        return float(raw_str) % 360.0
    except ValueError:
        pass

    clean_str = re.sub(r"[\s_/]+", "-", raw_str.upper()).strip("-")
    if clean_str in CARDINAL_TO_ANGLE:
        return CARDINAL_TO_ANGLE[clean_str]

    tokens = [t for t in re.split(r"[^A-Z]+", clean_str) if t]
    compound = "-".join(tokens)
    if compound in CARDINAL_TO_ANGLE:
        return CARDINAL_TO_ANGLE[compound]

    for key, angle in sorted(CARDINAL_TO_ANGLE.items(), key=lambda kv: len(kv[0]), reverse=True):
        if len(key) >= 2 and key in clean_str:
            return angle
        if len(key) == 1 and key in tokens:
            return angle
    return 0.0


def angle_to_cardinal(angle_deg: float) -> str:
    """Convertit un angle géométrique en degrés [0, 360[ vers le secteur cardinal le plus proche."""
    angle_mod = angle_deg % 360.0
    for sector_name, lower, upper in CARDINAL_8_SECTORS:
        if lower > upper:
            if angle_mod >= lower or angle_mod < upper:
                return sector_name
        else:
            if lower <= angle_mod < upper:
                return sector_name
    return "Nord"


def angle_to_sin_cos(angle_deg: float) -> Tuple[float, float]:
    """Calcule les composantes cycliques trigonométriques (sin, cos) d'un angle en degrés."""
    rad = math.radians(angle_deg)
    return math.sin(rad), math.cos(rad)


def sin_cos_to_angle(sin_val: float, cos_val: float) -> float:
    """Reconstruit l'angle en degrés [0, 360[ à partir de ses composantes (sin, cos)."""
    rad = math.atan2(sin_val, cos_val)
    deg = math.degrees(rad)
    return deg % 360.0


# ---------------------------------------------------------------------------
# Normaliseur Robuste Embarqué
# ---------------------------------------------------------------------------
class TabularStandardScaler:
    """StandardScaler portable sérialisable dans le modèle .pkl."""

    def __init__(self, feature_names: List[str]) -> None:
        self.feature_names = feature_names
        self.mean: np.ndarray = np.zeros(len(feature_names), dtype=np.float32)
        self.scale: np.ndarray = np.ones(len(feature_names), dtype=np.float32)

    def fit(self, X: np.ndarray) -> TabularStandardScaler:
        self.mean = np.mean(X, axis=0).astype(np.float32)
        std = np.std(X, axis=0).astype(np.float32)
        self.scale = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.scale

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        return (X_scaled * self.scale) + self.mean

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TabularStandardScaler:
        scaler = cls(feature_names=data["feature_names"])
        scaler.mean = np.array(data["mean"], dtype=np.float32)
        scaler.scale = np.array(data["scale"], dtype=np.float32)
        return scaler


# ---------------------------------------------------------------------------
# Génération & Séquençage de Séries Temporelles de Propagation
# ---------------------------------------------------------------------------
def simulate_fire_propagation_step(
    temp_c: float,
    humidity_pct: float,
    wind_speed_kmh: float,
    wind_dir_deg: float,
    affected_area_pct: float,
    prev_speed_kmh: float,
    dt_hours: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """
    Modèle physique semi-empirique de propagation (Rothermel simplifié) pour calculer :
    - La vitesse de propagation instantanée (km/h)
    - La direction de propagation résultante (en degrés)
    - L'accroissement de surface brûlée (%)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if affected_area_pct <= 0.01:
        return 0.0, 0.0, 0.0

    wind_term = 0.075 * (wind_speed_kmh ** 1.15)
    moisture_term = max(0.4, (100.0 - humidity_pct) / 45.0)
    heat_term = max(0.5, (temp_c / 25.0) ** 0.8)
    inertia_term = 0.25 * prev_speed_kmh

    base_speed = (wind_term * moisture_term * heat_term * 0.75) + inertia_term
    noise_speed = rng.normal(0.0, max(0.05, base_speed * 0.08))
    speed = float(np.clip(base_speed + noise_speed, 0.1, 15.0))

    dir_drift = rng.normal(0.0, max(2.0, 20.0 / (1.0 + wind_speed_kmh * 0.1)))
    prop_direction = (wind_dir_deg + dir_drift) % 360.0

    expansion_factor = (speed * 0.65) * (1.0 + affected_area_pct * 0.08)
    delta_area = float(np.clip(expansion_factor * dt_hours, 0.05, 12.0))

    return round(speed, 3), round(prop_direction, 2), round(delta_area, 3)


def generate_synthetic_propagation_sequences(
    base_dataset_csv: str | Path = "dataset/forestfires_kaggle.csv",
    num_episodes: int = 150,
    episode_length: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Génère des séries temporelles horaires réalistes de propagation de feux
    en combinant les distributions météo réelles de Kaggle (temp, RH, wind)
    avec une dynamique temporelle physique d'incendie.
    """
    rng = np.random.default_rng(seed)
    csv_path = Path(base_dataset_csv)

    meteo_samples: List[Dict[str, float]] = []
    if csv_path.is_file():
        df_raw = pd.read_csv(csv_path)
        for _, row in df_raw.iterrows():
            meteo_samples.append({
                "temp": float(row.get("temp", 24.0)),
                "rh": float(row.get("RH", 45.0)),
                "wind": float(row.get("wind", 4.0)) * 6.0,
            })

    if not meteo_samples:
        for _ in range(200):
            meteo_samples.append({
                "temp": float(rng.uniform(18.0, 39.0)),
                "rh": float(rng.uniform(15.0, 70.0)),
                "wind": float(rng.uniform(10.0, 55.0)),
            })

    records: List[Dict[str, Any]] = []

    for ep_id in range(num_episodes):
        base_meteo = rng.choice(meteo_samples)
        curr_temp = base_meteo["temp"] + rng.uniform(-3.0, 3.0)
        curr_rh = base_meteo["rh"] + rng.uniform(-5.0, 5.0)
        curr_wind = max(8.0, base_meteo["wind"] + rng.uniform(-5.0, 10.0))

        base_wind_dir = rng.uniform(0.0, 360.0)
        curr_wind_dir = base_wind_dir

        pos_x = rng.uniform(0.15, 0.85)
        pos_y = rng.uniform(0.15, 0.85)
        curr_area = rng.uniform(0.8, 4.5)
        curr_speed = 0.5

        for step in range(episode_length):
            curr_temp = np.clip(curr_temp + rng.normal(0.2, 0.8), 12.0, 44.0)
            curr_rh = np.clip(curr_rh + rng.normal(-0.3, 1.2), 10.0, 90.0)
            curr_wind = np.clip(curr_wind + rng.normal(0.1, 1.5), 5.0, 75.0)
            curr_wind_dir = (curr_wind_dir + rng.normal(0.0, 6.0)) % 360.0

            speed, prop_dir, delta_area = simulate_fire_propagation_step(
                temp_c=curr_temp,
                humidity_pct=curr_rh,
                wind_speed_kmh=curr_wind,
                wind_dir_deg=curr_wind_dir,
                affected_area_pct=curr_area,
                prev_speed_kmh=curr_speed,
                dt_hours=1.0,
                rng=rng,
            )

            rad = math.radians(prop_dir)
            step_disp = speed * 0.008
            pos_x = np.clip(pos_x + step_disp * math.sin(rad), 0.05, 0.95)
            pos_y = np.clip(pos_y - step_disp * math.cos(rad), 0.05, 0.95)
            curr_area = min(65.0, curr_area + delta_area)
            curr_speed = speed

            w_sin, w_cos = angle_to_sin_cos(curr_wind_dir)
            p_sin, p_cos = angle_to_sin_cos(prop_dir)

            records.append({
                "episode_id": ep_id,
                "hour_step": step,
                "temperature_c": round(float(curr_temp), 2),
                "humidity_pct": round(float(curr_rh), 2),
                "wind_speed_kmh": round(float(curr_wind), 2),
                "wind_direction_deg": round(float(curr_wind_dir), 2),
                "wind_sin": round(float(w_sin), 4),
                "wind_cos": round(float(w_cos), 4),
                "affected_area_pct": round(float(curr_area), 3),
                "loc_x_norm": round(float(pos_x), 4),
                "loc_y_norm": round(float(pos_y), 4),
                "historical_speed_kmh": round(float(curr_speed), 3),
                "propagation_direction_deg": round(float(prop_dir), 2),
                "target_speed_kmh": round(float(curr_speed), 3),
                "target_dir_sin": round(float(p_sin), 4),
                "target_dir_cos": round(float(p_cos), 4),
            })

    return pd.DataFrame(records)


def create_sliding_sequences(
    df: pd.DataFrame,
    sequence_length: int = 4,
    feature_cols: Optional[List[str]] = None,
    target_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Découpe les séries temporelles en fenêtres glissantes (sequences) pour l'entraînement RNN/LSTM :
    - X : Tenseur 3D de forme (N_samples, sequence_length, N_features)
    - Y : Tenseur 2D de forme (N_samples, N_targets)
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS
    if target_cols is None:
        target_cols = TARGET_COLUMNS

    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for _, group in df.groupby("episode_id"):
        feat_vals = group[feature_cols].to_numpy(dtype=np.float32)
        tgt_vals = group[target_cols].to_numpy(dtype=np.float32)

        if len(feat_vals) < sequence_length:
            continue

        for i in range(len(feat_vals) - sequence_length + 1):
            x_seq = feat_vals[i : i + sequence_length]
            y_tgt = tgt_vals[i + sequence_length - 1]
            x_list.append(x_seq)
            y_list.append(y_tgt)

    X = np.array(x_list, dtype=np.float32)
    Y = np.array(y_list, dtype=np.float32)
    return X, Y


def prepare_datasets_for_training(
    csv_source: str | Path = "dataset/forestfires_kaggle.csv",
    sequence_length: int = 4,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TabularStandardScaler, TabularStandardScaler]:
    """
    Prépare l'ensemble des données d'entraînement et de validation avec standardisation
    complète des entrées et cibles.
    """
    df_series = generate_synthetic_propagation_sequences(
        base_dataset_csv=csv_source,
        num_episodes=200,
        episode_length=12,
        seed=seed,
    )

    unique_episodes = df_series["episode_id"].unique()
    rng = np.random.default_rng(seed)
    shuffled_episodes = rng.permutation(unique_episodes)
    split_ep_idx = int(len(shuffled_episodes) * (1.0 - val_split))

    train_episodes = set(shuffled_episodes[:split_ep_idx].tolist())
    val_episodes = set(shuffled_episodes[split_ep_idx:].tolist())

    df_train = df_series[df_series["episode_id"].isin(train_episodes)].reset_index(drop=True)
    df_val = df_series[df_series["episode_id"].isin(val_episodes)].reset_index(drop=True)

    X_train_raw, Y_train_raw = create_sliding_sequences(df_train, sequence_length=sequence_length)
    X_val_raw, Y_val_raw = create_sliding_sequences(df_val, sequence_length=sequence_length)

    n_train, seq_len, n_feat = X_train_raw.shape
    x_scaler = TabularStandardScaler(feature_names=FEATURE_COLUMNS)
    x_scaler.fit(X_train_raw.reshape(-1, n_feat))

    y_scaler = TabularStandardScaler(feature_names=TARGET_COLUMNS)
    y_scaler.fit(Y_train_raw)
    # Préservation géométrique exacte du cercle unitaire pour (target_dir_sin, target_dir_cos)
    y_scaler.mean[1:3] = np.array([0.0, 0.0], dtype=np.float32)
    y_scaler.scale[1:3] = np.array([1.0, 1.0], dtype=np.float32)

    X_train = x_scaler.transform(X_train_raw.reshape(-1, n_feat)).reshape(n_train, seq_len, n_feat)
    X_val = x_scaler.transform(X_val_raw.reshape(-1, n_feat)).reshape(len(X_val_raw), seq_len, n_feat)

    Y_train = y_scaler.transform(Y_train_raw)
    Y_val = y_scaler.transform(Y_val_raw)

    return X_train, Y_train, X_val, Y_val, x_scaler, y_scaler
