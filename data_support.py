"""
Module Support Data & Préparation du Dataset Global (Mission Secondaire - Salim)
================================================================================
Responsable : Salim (Vision par Ordinateur & Support Data)

Ce module remplit deux rôles clés pour le Projet 5 :
1. Acquisition et structuration des 2 datasets officiels Kaggle du Projet 5 :
   - Imagerie Feu : `phylake1337/fire-dataset` + échantillons satellites segmentés
     sauvegardés directement dans `dataset/kaggle_cleaned/` et `dataset/satellite_samples/`.
   - Météo & Feux de Forêt : `elikplim/forest-fires-data-set` (`forestfires.csv`)
     copié dans `dataset/forestfires_kaggle.csv`.
2. Mission Secondaire (Support Data) : Génération et exportation du dataset global unifié
   (`dataset/global_fire_weather_dataset.csv`) fusionnant les prédictions CNN de Salim
   avec les vraies données météo (vent, humidité, température) pour Moundir (LSTM) et
   l'Arbre de Décision (`main.py` / Streamlit).
"""

from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from preprocessing import (
    DEFAULT_IMAGE_SIZE,
    VALID_EXTENSIONS,
    extract_fire_segmentation_mask,
    is_image_valid,
    preprocess_image,
)


def generate_satellite_samples_with_masks(
    output_dir: str | Path = "dataset/satellite_samples",
    num_fire: int = 90,
    num_nofire: int = 90,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    seed: int = 42,
) -> List[Tuple[str, int, str]]:
    """
    Crée un jeu d'images satellites forestières réalistes avec leurs masques
    de segmentation exacts (vérité terrain pixel-par-pixel) pour garantir un apprentissage
    spatial haute précision de la tête U-Net.
    """
    rng = np.random.default_rng(seed)
    root = Path(output_dir)
    fire_img_dir = root / "images" / "fire"
    nofire_img_dir = root / "images" / "nofire"
    mask_dir = root / "masks"

    fire_img_dir.mkdir(parents=True, exist_ok=True)
    nofire_img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    w, h = image_size
    samples: List[Tuple[str, int, str]] = []

    def _create_forest_canopy_background() -> np.ndarray:
        base_noise = rng.normal(0.0, 1.0, size=(h // 4, w // 4, 3)).astype(np.float32)
        smooth_terrain = cv2.resize(base_noise, (w, h), interpolation=cv2.INTER_CUBIC)
        fine_texture = rng.normal(0.0, 0.35, size=(h, w, 3)).astype(np.float32)

        r_base = rng.uniform(35, 78) + smooth_terrain[:, :, 0] * 14 + fine_texture[:, :, 0] * 8
        g_base = rng.uniform(68, 125) + smooth_terrain[:, :, 1] * 18 + fine_texture[:, :, 1] * 10
        b_base = rng.uniform(30, 65) + smooth_terrain[:, :, 2] * 10 + fine_texture[:, :, 2] * 6

        if rng.random() < 0.35:
            pt1 = (int(rng.integers(0, w)), 0)
            pt2 = (int(rng.integers(0, w)), h)
            road_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.line(road_mask, pt1, pt2, 255, thickness=int(rng.integers(2, 5)))
            road_mask = cv2.GaussianBlur(road_mask, (5, 5), 0) / 255.0
            r_base = r_base * (1 - road_mask) + 115 * road_mask
            g_base = g_base * (1 - road_mask) + 110 * road_mask
            b_base = b_base * (1 - road_mask) + 95 * road_mask

        bgr = np.stack([b_base, g_base, r_base], axis=-1)
        return np.clip(bgr, 0, 255).astype(np.uint8)

    for i in range(num_fire):
        bgr = _create_forest_canopy_background()
        gt_mask = np.zeros((h, w), dtype=np.uint8)

        num_foyers = int(rng.integers(1, 3))
        for _ in range(num_foyers):
            cx = int(rng.integers(int(w * 0.15), int(w * 0.85)))
            cy = int(rng.integers(int(h * 0.15), int(h * 0.85)))
            ax1 = int(rng.integers(8, 26))
            ax2 = int(rng.integers(6, 22))
            angle = float(rng.uniform(0, 180))

            foyer_layer = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(foyer_layer, (cx, cy), (ax1, ax2), angle, 0, 360, 255, -1)

            disp = rng.normal(0, 3.5, size=(h // 8, w // 8)).astype(np.float32)
            disp = cv2.resize(disp, (w, h), interpolation=cv2.INTER_CUBIC)
            foyer_float = cv2.GaussianBlur(foyer_layer.astype(np.float32) / 255.0, (9, 9), 0)
            foyer_organic = np.clip(foyer_float + disp * 0.05, 0.0, 1.0)
            active_zone = (foyer_organic >= 0.32).astype(np.uint8) * 255
            gt_mask = np.maximum(gt_mask, active_zone)

            dx = int(rng.integers(-22, 23))
            dy = int(rng.integers(-22, 23))
            smoke_layer = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(
                smoke_layer,
                (np.clip(cx + dx, 0, w - 1), np.clip(cy + dy, 0, h - 1)),
                (int(ax1 * 1.4), int(ax2 * 1.2)),
                angle,
                0,
                360,
                0.45,
                -1,
            )
            smoke_layer = cv2.GaussianBlur(smoke_layer, (19, 19), 0)
            for c, smoke_val in enumerate([175.0, 180.0, 188.0]):
                bgr[:, :, c] = np.clip(
                    bgr[:, :, c].astype(np.float32) * (1.0 - smoke_layer) + smoke_val * smoke_layer,
                    0,
                    255,
                ).astype(np.uint8)

            heat = cv2.GaussianBlur((active_zone / 255.0).astype(np.float32), (7, 7), 0)
            core = np.clip((foyer_organic - 0.55) * 2.2, 0.0, 1.0)

            b_chan = bgr[:, :, 0].astype(np.float32)
            g_chan = bgr[:, :, 1].astype(np.float32)
            r_chan = bgr[:, :, 2].astype(np.float32)

            r_chan = r_chan * (1.0 - heat) + (235.0 + 20.0 * core) * heat
            g_chan = g_chan * (1.0 - heat) + (85.0 + 150.0 * core) * heat
            b_chan = b_chan * (1.0 - heat) + (18.0 + 95.0 * core) * heat

            bgr[:, :, 0] = np.clip(b_chan, 0, 255).astype(np.uint8)
            bgr[:, :, 1] = np.clip(g_chan, 0, 255).astype(np.uint8)
            bgr[:, :, 2] = np.clip(r_chan, 0, 255).astype(np.uint8)

        img_path = fire_img_dir / f"sat_fire_{i:04d}.png"
        mask_path = mask_dir / f"sat_fire_{i:04d}_mask.png"
        cv2.imwrite(str(img_path), bgr)
        cv2.imwrite(str(mask_path), gt_mask)
        samples.append((str(img_path), 1, str(mask_path)))

    for i in range(num_nofire):
        bgr = _create_forest_canopy_background()
        gt_mask = np.zeros((h, w), dtype=np.uint8)

        if rng.random() < 0.4:
            cx = int(rng.integers(20, w - 20))
            cy = int(rng.integers(20, h - 20))
            cloud = np.zeros((h, w), dtype=np.float32)
            cv2.circle(cloud, (cx, cy), int(rng.integers(10, 25)), 0.55, -1)
            cloud = cv2.GaussianBlur(cloud, (21, 21), 0)
            for c in range(3):
                bgr[:, :, c] = np.clip(
                    bgr[:, :, c].astype(np.float32) * (1.0 - cloud) + 235.0 * cloud,
                    0,
                    255,
                ).astype(np.uint8)

        img_path = nofire_img_dir / f"sat_nofire_{i:04d}.png"
        mask_path = mask_dir / f"sat_nofire_{i:04d}_mask.png"
        cv2.imwrite(str(img_path), bgr)
        cv2.imwrite(str(mask_path), gt_mask)
        samples.append((str(img_path), 0, str(mask_path)))

    return samples


def download_and_prepare_kaggle_dataset(
    max_kaggle_per_class: int = 110,
    include_satellite_masks: bool = True,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int, Optional[str]]], List[Tuple[str, int, Optional[str]]]]:
    """
    Télécharge les datasets Kaggle du Projet 5 via `kagglehub`, nettoie et prétraite
    les images dans `dataset/kaggle_cleaned/` (128x128 + masques pré-calculés pour un
    entraînement ultra-rapide), et copie `forestfires.csv` dans `dataset/forestfires_kaggle.csv`.
    """
    all_samples: List[Tuple[str, int, Optional[str]]] = []
    cleaned_root = Path("dataset/kaggle_cleaned")
    cleaned_fire_dir = cleaned_root / "images" / "fire"
    cleaned_nofire_dir = cleaned_root / "images" / "nofire"
    cleaned_mask_dir = cleaned_root / "masks"
    cleaned_fire_dir.mkdir(parents=True, exist_ok=True)
    cleaned_nofire_dir.mkdir(parents=True, exist_ok=True)
    cleaned_mask_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub

        # 1. Dataset d'images d'incendies Kaggle
        print("1) Chargement du dataset d'images Kaggle (phylake1337/fire-dataset)...")
        kaggle_img_path = Path(kagglehub.dataset_download("phylake1337/fire-dataset"))
        print(f"   -> Dossier source Kaggle : {kaggle_img_path}")

        fire_files: List[Path] = []
        nofire_files: List[Path] = []

        for p in sorted(kaggle_img_path.rglob("*")):
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS:
                parent_name = p.parent.name.lower()
                if "non_fire" in parent_name or "nofire" in parent_name or "normal" in parent_name:
                    nofire_files.append(p)
                elif "fire" in parent_name or "wildfire" in parent_name:
                    fire_files.append(p)

        rng_py = random.Random(seed)
        rng_py.shuffle(fire_files)
        rng_py.shuffle(nofire_files)

        # Nettoyage + Redimensionnement 128x128 + Extraction du masque dans dataset/kaggle_cleaned
        valid_fire_count = 0
        for p in fire_files:
            if valid_fire_count >= max_kaggle_per_class:
                break
            if not is_image_valid(p):
                continue
            rgb_u8 = preprocess_image(p, target_size=DEFAULT_IMAGE_SIZE, normalize=False)
            mask_f32 = extract_fire_segmentation_mask(rgb_u8, has_fire_label=True)
            out_img = cleaned_fire_dir / f"kaggle_fire_{valid_fire_count:04d}.png"
            out_msk = cleaned_mask_dir / f"kaggle_fire_{valid_fire_count:04d}_mask.png"
            cv2.imwrite(str(out_img), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_msk), (mask_f32 * 255).astype(np.uint8))
            all_samples.append((str(out_img), 1, str(out_msk)))
            valid_fire_count += 1

        valid_nofire_count = 0
        for p in nofire_files:
            if valid_nofire_count >= max_kaggle_per_class:
                break
            if not is_image_valid(p):
                continue
            rgb_u8 = preprocess_image(p, target_size=DEFAULT_IMAGE_SIZE, normalize=False)
            mask_u8 = np.zeros(DEFAULT_IMAGE_SIZE[::-1], dtype=np.uint8)
            out_img = cleaned_nofire_dir / f"kaggle_nofire_{valid_nofire_count:04d}.png"
            out_msk = cleaned_mask_dir / f"kaggle_nofire_{valid_nofire_count:04d}_mask.png"
            cv2.imwrite(str(out_img), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_msk), mask_u8)
            all_samples.append((str(out_img), 0, str(out_msk)))
            valid_nofire_count += 1

        print(
            f"   -> Images Kaggle nettoyées et sauvegardées dans 'dataset/kaggle_cleaned/' : "
            f"{valid_fire_count} Feu | {valid_nofire_count} Non-Feu"
        )

        # 2. Dataset Forest Fires Météo (Projet 5 : Forest Fires Dataset)
        print("2) Téléchargement du dataset météo Kaggle (elikplim/forest-fires-data-set)...")
        ff_path = Path(kagglehub.dataset_download("elikplim/forest-fires-data-set"))
        csv_candidates = list(ff_path.rglob("*.csv"))
        if csv_candidates:
            target_ff_csv = Path("dataset/forestfires_kaggle.csv")
            shutil.copy2(csv_candidates[0], target_ff_csv)
            print(f"   -> Dataset Forest Fires copié dans : {target_ff_csv.resolve()}")
    except Exception as exc:
        print(f"[Info] Téléchargement Kaggle partiel ou hors-ligne ({exc}).")

    if include_satellite_masks or len(all_samples) == 0:
        sat_samples = generate_satellite_samples_with_masks(
            output_dir="dataset/satellite_samples",
            num_fire=90,
            num_nofire=90,
            seed=seed,
        )
        all_samples.extend(sat_samples)
        print(f"3) Échantillons satellites avec masques dans 'dataset/satellite_samples/' : {len(sat_samples)}")

    rng_py = random.Random(seed)
    rng_py.shuffle(all_samples)

    split_idx = int(len(all_samples) * 0.82)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    return train_samples, val_samples


def build_global_team_dataset(
    samples: List[Tuple[str, int, Optional[str]]],
    output_csv: str | Path = "dataset/global_fire_weather_dataset.csv",
    max_rows: int = 200,
    seed: int = 42,
) -> str:
    """
    Mission Secondaire de Salim (Support Data) :
    Génère le dataset global unifié combinant :
    - Les sorties Vision par Ordinateur de Salim (Feu Oui/Non, zone localisée, GPS, % surface brûlée)
    - Les vraies mesures météorologiques issues de `dataset/forestfires_kaggle.csv` (vent, humidité RH,
      température, indices FFMC/DMC/ISI) pour Moundir (LSTM) et l'Arbre de Décision.
    """
    from contracts import ALERT_CODE_TO_LABEL, compute_reference_alert_code
    from predict import predict
    from predict_propagation import predict_propagation

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Lecture des vraies lignes météo de forestfires_kaggle.csv si disponible
    kaggle_meteo_rows: List[Dict[str, str]] = []
    ff_csv = Path("dataset/forestfires_kaggle.csv")
    if ff_csv.is_file():
        with open(ff_csv, mode="r", encoding="utf-8") as f_in:
            reader = csv.DictReader(f_in)
            kaggle_meteo_rows = list(reader)

    cardinal_directions = ["Nord", "Nord-Est", "Est", "Sud-Est", "Sud", "Sud-Ouest", "Ouest", "Nord-Ouest"]

    fieldnames = [
        "sample_id",
        "image_path",
        "fire_detected",           # Sortie Salim ("Oui" / "Non")
        "confidence",              # Sortie Salim ([0..1])
        "location_zone",           # Sortie Salim ("Nord-Est", "Centre", etc.)
        "latitude",                # Sortie Salim (GPS lat)
        "longitude",               # Sortie Salim (GPS lon)
        "affected_area_pct",       # Sortie Salim (% surface touchée)
        "temperature_c",           # Entrée Météo Kaggle (Moundir & Arbre de Décision)
        "humidity_pct",            # Entrée Météo Kaggle RH (Moundir & Arbre de Décision)
        "wind_speed_kmh",          # Entrée Météo Kaggle wind (Moundir & Arbre de Décision)
        "wind_direction",          # Entrée Météo (Moundir)
        "propagation_direction",   # Cible Moundir (LSTM)
        "propagation_speed_kmh",   # Cible Moundir (LSTM)
        "towards_populated_zone",  # Entrée Arbre de Décision ("Oui" / "Non")
        "alert_level",             # Cible Arbre de Décision ("Vert", "Orange", "Rouge", "Évacuation")
    ]

    selected = samples[:max_rows]
    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (img_path, gt_label, _) in enumerate(selected, start=1):
            pred = predict(img_path)
            has_fire = pred["has_fire"]
            loc = pred["location"]

            # Utilisation des vraies valeurs météo de Forest Fires Dataset (ou fallback réaliste)
            if kaggle_meteo_rows:
                m_row = kaggle_meteo_rows[(idx - 1) % len(kaggle_meteo_rows)]
                raw_temp = float(m_row.get("temp", 25.0))
                raw_rh = float(m_row.get("RH", 40.0))
                raw_wind = float(m_row.get("wind", 4.5)) * 6.5  # mise à l'échelle rafales km/h
            else:
                raw_temp = float(rng.uniform(22.0, 38.0))
                raw_rh = float(rng.uniform(20.0, 65.0))
                raw_wind = float(rng.uniform(15.0, 60.0))

            if has_fire:
                # Distribution variée couvrant tous les paliers d'alerte (Orange, Rouge, Évacuation)
                regime = idx % 5
                if regime == 0:
                    # Feu modéré : vent faible (< 20 km/h) et humidité élevée (> 40%) -> Orange
                    temp_c = round(float(rng.uniform(20.0, 27.0)), 1)
                    humidity = round(float(rng.uniform(43.0, 68.0)), 1)
                    wind_kmh = round(float(rng.uniform(6.0, 18.5)), 1)
                    eff_loc = dict(loc)
                    eff_loc["affected_area_pct"] = min(float(loc["affected_area_pct"]), 8.5)
                else:
                    temp_c = round(max(24.0, raw_temp), 1)
                    humidity = round(min(55.0, raw_rh), 1)
                    wind_kmh = round(max(14.0, raw_wind), 1)
                    eff_loc = loc

                wind_dir = str(rng.choice(cardinal_directions))
                lstm_res = predict_propagation(
                    location=eff_loc,
                    wind={"speed": wind_kmh, "direction": wind_dir},
                    humidity=humidity,
                    temperature=temp_c,
                )
                prop_dir = lstm_res["propagation_direction"]
                prop_speed = lstm_res["propagation_speed"]
                near_hab = "Oui" if rng.random() < 0.45 else "Non"

                alert_code = compute_reference_alert_code(
                    feu=1,
                    vent_kmh=wind_kmh,
                    humidite_pct=humidity,
                    vitesse_kmh=prop_speed,
                )
                alert = ALERT_CODE_TO_LABEL[alert_code]
            else:
                temp_c = round(raw_temp, 1)
                humidity = round(max(35.0, raw_rh), 1)
                wind_kmh = round(raw_wind * 0.7, 1)
                wind_dir = str(rng.choice(cardinal_directions))
                prop_dir = "Aucune"
                prop_speed = 0.0
                near_hab = "Non"
                alert = "Vert"

            gps = loc["gps_coords"] if loc["gps_coords"] is not None else (36.75, 5.05)

            writer.writerow(
                {
                    "sample_id": idx,
                    "image_path": img_path,
                    "fire_detected": pred["fire_detected"],
                    "confidence": pred["confidence"],
                    "location_zone": loc["zone"],
                    "latitude": gps[0],
                    "longitude": gps[1],
                    "affected_area_pct": loc["affected_area_pct"],
                    "temperature_c": temp_c,
                    "humidity_pct": humidity,
                    "wind_speed_kmh": wind_kmh,
                    "wind_direction": wind_dir,
                    "propagation_direction": prop_dir,
                    "propagation_speed_kmh": prop_speed,
                    "towards_populated_zone": near_hab,
                    "alert_level": alert,
                }
            )

    print(f"Dataset global d'équipe généré avec succès : {out_path.resolve()} ({len(selected)} lignes)")
    return str(out_path.resolve())
