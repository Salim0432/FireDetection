"""
Module de Traitement Météorologique & Préparation des Séries Temporelles (Tâche 1 & Support Data - Moundir)
=========================================================================================================
Responsable : Moundir (Modélisation Temporelle & Météo - RNN/LSTM)

Ce module assure :
1. Le prétraitement et le feature engineering des données météorologiques :
   - Température (°C)
   - Humidité relative (RH %)
   - Vent : vitesse (km/h) et direction angulaire / cardinale
   - Estimation de l'Indice de Propagation Initiale (ISI / FWI)
2. La transformation cyclique des directions angulaires en composantes sin/cos
   pour éliminer les discontinuités à 0°/360° dans les réseaux récurrents.
3. La génération et le séquençage glissant (sliding window) des séries temporelles
   d'incendies simulant la dynamique d'expansion heure par heure (Support Data).
4. La normalisation robuste (StandardScaler / MinMax) avec sérialisation des statistiques
   pour garantir une inférence déterministe sans data leakage.
"""

from __future__ import annotations

import csv
import math
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
    "fire_weather_index",
    "affected_area_pct",
    "loc_x_norm",
    "loc_y_norm",
    "historical_speed_kmh",
]

TARGET_COLUMNS: List[str] = [
    "target_speed_kmh",
    "target_dir_sin",
    "target_dir_cos",
    "expansion_next_1h",
    "expansion_next_2h",
    "expansion_next_3h",
]


# ---------------------------------------------------------------------------
# Utilitaires de Conversion Angulaire & Vectorielle
# ---------------------------------------------------------------------------
def direction_to_angle(direction: Union[str, float, int]) -> float:
    """Convertit une direction (chaîne textuelle ou valeur numérique) en angle en degrés [0, 360[."""
    if isinstance(direction, (int, float)):
        return float(direction) % 360.0

    clean_str = str(direction).strip().upper()
    if clean_str in CARDINAL_TO_ANGLE:
        return CARDINAL_TO_ANGLE[clean_str]

    for key, angle in CARDINAL_TO_ANGLE.items():
        if key in clean_str:
            return angle
    return 0.0


def angle_to_cardinal(angle_deg: float) -> str:
    """Convertit un angle géométrique en degrés [0, 360[ vers le secteur cardinal le plus proche."""
    angle_mod = angle_deg % 360.0
    for sector_name, lower, upper in CARDINAL_8_SECTORS:
        if lower > upper:  # Cas du Nord qui traverse 0° (337.5° -> 22.5°)
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


def compute_fire_weather_index(temp_c: float, rh_pct: float, wind_kmh: float) -> float:
    """
    Estime l'indice météorologique de propagation (inspiré de la formulation canadienne FWI / ISI).
    Un FWI élevé indique une propagation fulgurante due à la chaleur, la sécheresse et le vent.
    """
    temp_factor = max(0.1, (temp_c - 10.0) * 0.06 + 0.8)
    humidity_factor = max(0.2, (100.0 - rh_pct) / 50.0)
    wind_factor = (wind_kmh / 15.0) ** 1.35
    isi = 0.208 * wind_factor * humidity_factor * temp_factor
    return float(np.clip(isi, 0.0, 50.0))


# ---------------------------------------------------------------------------
# Normaliseur Robuste Embarqué (sans dépendance externe lourde à l'inférence)
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
# Mission Secondaire : Support Data & Génération de Séries Temporelles
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

    # Effet du vent (moteur majeur)
    wind_term = 0.075 * (wind_speed_kmh ** 1.15)
    # Effet de la sécheresse relative
    moisture_term = max(0.4, (100.0 - humidity_pct) / 45.0)
    # Effet de la température
    heat_term = max(0.5, (temp_c / 25.0) ** 0.8)
    # Inertie du feu
    inertia_term = 0.25 * prev_speed_kmh

    # Vitesse de propagation
    base_speed = (wind_term * moisture_term * heat_term * 0.75) + inertia_term
    # Légère fluctuation turbulente
    noise_speed = rng.normal(0.0, max(0.05, base_speed * 0.08))
    speed = float(np.clip(base_speed + noise_speed, 0.1, 15.0))

    # Direction de propagation : alignée avec le vent avec dérive turbulente
    dir_drift = rng.normal(0.0, max(2.0, 20.0 / (1.0 + wind_speed_kmh * 0.1)))
    prop_direction = (wind_dir_deg + dir_drift) % 360.0

    # Expansion horaire de surface
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
    avec une dynamique temporelle physique d'incendie (Support Data).
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
                "wind": float(row.get("wind", 4.0)) * 6.0,  # km/h
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
        # Conditions initiales du feu
        base_meteo = rng.choice(meteo_samples)
        curr_temp = base_meteo["temp"] + rng.uniform(-3.0, 3.0)
        curr_rh = base_meteo["rh"] + rng.uniform(-5.0, 5.0)
        curr_wind = max(8.0, base_meteo["wind"] + rng.uniform(-5.0, 10.0))

        # Direction dominante du vent pour cet épisode
        base_wind_dir = rng.uniform(0.0, 360.0)
        curr_wind_dir = base_wind_dir

        # Position spatiale initiale du feu
        pos_x = rng.uniform(0.15, 0.85)
        pos_y = rng.uniform(0.15, 0.85)
        curr_area = rng.uniform(0.8, 4.5)
        curr_speed = 0.5

        for step in range(episode_length):
            # Évolution diurne de la météo au fil des heures
            curr_temp = np.clip(curr_temp + rng.normal(0.2, 0.8), 12.0, 44.0)
            curr_rh = np.clip(curr_rh + rng.normal(-0.3, 1.2), 10.0, 90.0)
            curr_wind = np.clip(curr_wind + rng.normal(0.1, 1.5), 5.0, 75.0)
            curr_wind_dir = (curr_wind_dir + rng.normal(0.0, 6.0)) % 360.0

            # Propagation dynamique
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

            # Déplacement du barycentre sous l'effet du vent
            rad = math.radians(prop_dir)
            step_disp = (speed * 0.008)  # déplacement normalisé
            pos_x = np.clip(pos_x + step_disp * math.sin(rad), 0.05, 0.95)
            pos_y = np.clip(pos_y - step_disp * math.cos(rad), 0.05, 0.95)
            curr_area = min(65.0, curr_area + delta_area)
            curr_speed = speed

            fwi = compute_fire_weather_index(curr_temp, curr_rh, curr_wind)
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
                "fire_weather_index": round(float(fwi), 3),
                "affected_area_pct": round(float(curr_area), 3),
                "loc_x_norm": round(float(pos_x), 4),
                "loc_y_norm": round(float(pos_y), 4),
                "historical_speed_kmh": round(float(curr_speed), 3),
                "propagation_direction_deg": round(float(prop_dir), 2),
                "target_speed_kmh": round(float(curr_speed), 3),
                "target_dir_sin": round(float(p_sin), 4),
                "target_dir_cos": round(float(p_cos), 4),
            })

    df = pd.DataFrame(records)

    # Calcul multi-horizon (expansion aux prochaines 1h, 2h, 3h)
    df["expansion_next_1h"] = df.groupby("episode_id")["historical_speed_kmh"].shift(-1)
    df["expansion_next_2h"] = df.groupby("episode_id")["historical_speed_kmh"].shift(-2)
    df["expansion_next_3h"] = df.groupby("episode_id")["historical_speed_kmh"].shift(-3)

    # Complétion des horizons finaux avec extrapolation physique
    df["expansion_next_1h"] = df["expansion_next_1h"].fillna(df["target_speed_kmh"] * 1.05)
    df["expansion_next_2h"] = df["expansion_next_2h"].fillna(df["expansion_next_1h"] * 1.08)
    df["expansion_next_3h"] = df["expansion_next_3h"].fillna(df["expansion_next_2h"] * 1.10)

    return df


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

    X_raw, Y_raw = create_sliding_sequences(df_series, sequence_length=sequence_length)

    # Séparation train / validation par épisodes complets pour éviter tout data leakage
    num_samples = len(X_raw)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    split_idx = int(num_samples * (1.0 - val_split))

    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    X_train_raw = X_raw[train_idx]
    Y_train_raw = Y_raw[train_idx]
    X_val_raw = X_raw[val_idx]
    Y_val_raw = Y_raw[val_idx]

    # Ajustement des scalers sur Train uniquement
    n_train, seq_len, n_feat = X_train_raw.shape
    x_scaler = TabularStandardScaler(feature_names=FEATURE_COLUMNS)
    x_scaler.fit(X_train_raw.reshape(-1, n_feat))

    y_scaler = TabularStandardScaler(feature_names=TARGET_COLUMNS)
    y_scaler.fit(Y_train_raw)

    # Transformation
    X_train = x_scaler.transform(X_train_raw.reshape(-1, n_feat)).reshape(n_train, seq_len, n_feat)
    X_val = x_scaler.transform(X_val_raw.reshape(-1, n_feat)).reshape(len(X_val_raw), seq_len, n_feat)

    Y_train = y_scaler.transform(Y_train_raw)
    Y_val = y_scaler.transform(Y_val_raw)

    return X_train, Y_train, X_val, Y_val, x_scaler, y_scaler
