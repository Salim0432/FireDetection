"""
Module d'Inférence et de Localisation du Feu (Tâche 3 - Salim)
==============================================================
Responsable : Salim (Vision par Ordinateur - CNN / U-Net)

Ce module fournit la fonction officielle `predict(image_path)` exigée par le cahier
des charges pour l'intégration avec :
- Mondir (Modèle RNN/LSTM `predict_propagation(location, wind, humidity)`)
- Responsable Système d'Alerte (Arbre de Décision + `main.py` + Streamlit)

Sortie de `predict(image_path)` :
- `fire_detected` : "Oui" ou "Non"
- `has_fire`      : bool (True / False)
- `confidence`    : float (probabilité dans [0, 1])
- `location`      : Dictionnaire détaillé de la localisation du foyer :
    * `zone`              : Secteur cardinal sur l'image satellite (ex: "Nord-Est", "Centre")
    * `center_xy`         : Coordonnées pixels (x, y) du barycentre du feu
    * `normalized_coords` : Coordonnées normalisées (nx, ny) dans [0, 1]
    * `gps_coords`        : Coordonnées géographiques estimées (latitude, longitude)
    * `bbox`              : Boîte englobante [x_min, y_min, x_max, y_max]
    * `affected_area_pct` : Pourcentage de la surface satellite touchée par le feu (%)
- `segmentation_mask` : Masque binaire 2D (H, W) de la zone touchée
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from model_cnn import FireUNet, load_trained_model
from preprocessing import DEFAULT_IMAGE_SIZE, preprocess_image


# Modèle mis en cache en mémoire pour accélérer les appels successifs depuis main.py / Streamlit
_CACHED_MODEL: Optional[FireUNet] = None
_CACHED_CONFIG: Optional[Dict[str, Any]] = None
_CACHED_PATH: Optional[str] = None


def _compute_cardinal_zone(nx: float, ny: float) -> str:
    """Détermine le secteur cardinal (Nord-Ouest, Centre, Sud-Est, etc.) à partir de (nx, ny) dans [0, 1]."""
    if 0.35 <= nx <= 0.65 and 0.35 <= ny <= 0.65:
        return "Centre"

    vert = "Nord" if ny < 0.40 else ("Sud" if ny > 0.60 else "")
    horiz = "Ouest" if nx < 0.40 else ("Est" if nx > 0.60 else "")

    if vert and horiz:
        return f"{vert}-{horiz}"
    return vert or horiz or "Centre"


def _extract_localization_from_mask(
    prob_mask: np.ndarray,
    orig_width: int,
    orig_height: int,
    seg_threshold: float = 0.5,
    base_lat: float = 36.75,
    base_lon: float = 5.05,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Calcule la boîte englobante, le barycentre, le secteur cardinal et la surface touchée
    à partir de la carte de probabilité de segmentation redimensionnée à la taille d'origine.
    """
    mask_resized = cv2.resize(prob_mask, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)
    binary_mask = (mask_resized >= seg_threshold).astype(np.uint8)

    total_pixels = float(orig_width * orig_height)
    fire_pixels = int(np.count_nonzero(binary_mask))
    affected_area_pct = round((fire_pixels / max(total_pixels, 1.0)) * 100.0, 2)

    if fire_pixels == 0:
        return (
            {
                "zone": "Aucune",
                "center_xy": None,
                "normalized_coords": None,
                "gps_coords": None,
                "bbox": None,
                "affected_area_pct": 0.0,
            },
            binary_mask,
        )

    ys, xs = np.where(binary_mask > 0)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    # Barycentre pondéré par la probabilité du feu
    weights = mask_resized[ys, xs]
    w_sum = float(weights.sum())
    if w_sum > 1e-6:
        cx = int(round(float((xs * weights).sum()) / w_sum))
        cy = int(round(float((ys * weights).sum()) / w_sum))
    else:
        cx = int(round(float(xs.mean())))
        cy = int(round(float(ys.mean())))

    nx = round(cx / max(float(orig_width - 1), 1.0), 4)
    ny = round(cy / max(float(orig_height - 1), 1.0), 4)
    zone = _compute_cardinal_zone(nx, ny)

    # Projection GPS locale indicative (ex: région forestière méditerranéenne / Béjaïa-Kabylie)
    lat = round(base_lat + (0.5 - ny) * 0.10, 5)
    lon = round(base_lon + (nx - 0.5) * 0.10, 5)

    location_info = {
        "zone": zone,
        "center_xy": (cx, cy),
        "normalized_coords": (nx, ny),
        "gps_coords": (lat, lon),
        "bbox": [x_min, y_min, x_max, y_max],
        "affected_area_pct": affected_area_pct,
    }
    return location_info, binary_mask


def predict(
    image_path: str | Path,
    model_path: Optional[str | Path] = None,
    save_visualization: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Fonction principale de prédiction (Tâche 3 - Salim).
    Analyse une image satellite pour détecter la présence d'un incendie et localiser
    précisément la zone touchée.

    Args:
        image_path (str | Path): Chemin vers l'image satellite à analyser.
        model_path (Optional[str | Path]): Chemin vers le modèle `.pkl` ou `.h5`.
            Par défaut, cherche `models/fire_cnn_unet.pkl` puis `models/fire_cnn_unet.h5`.
        save_visualization (Optional[str | Path]): Si fourni, sauvegarde une image
            annotée (masque rouge semi-transparent + boîte englobante + barycentre).

    Returns:
        Dict[str, Any]:
            - "fire_detected" (str) : "Oui" ou "Non"
            - "has_fire" (bool) : True si feu détecté, False sinon
            - "confidence" (float) : Niveau de confiance du modèle dans [0, 1]
            - "location" (dict) : Secteur, coordonnées (x, y), GPS, bbox et % surface touchée
            - "summary" (str) : Texte explicatif prêt pour l'interface Streamlit
            - "segmentation_mask" (np.ndarray) : Masque binaire 2D (H, W)
    """
    global _CACHED_MODEL, _CACHED_CONFIG, _CACHED_PATH

    img_file = Path(image_path)
    if not img_file.is_file():
        raise FileNotFoundError(f"Image introuvable : {img_file}")

    # Résolution automatique du chemin du modèle (.pkl ou .h5)
    base_dir = Path(__file__).resolve().parent
    if model_path is None:
        default_pkl = base_dir / "models" / "fire_cnn_unet.pkl"
        default_h5 = base_dir / "models" / "fire_cnn_unet.h5"
        if default_pkl.is_file():
            resolved_model_path = default_pkl
        elif default_h5.is_file():
            resolved_model_path = default_h5
        else:
            raise FileNotFoundError(
                "Aucun modèle sauvegardé trouvé dans 'models/fire_cnn_unet.pkl' ou '.h5'."
            )
    else:
        resolved_model_path = Path(model_path)

    # Chargement avec mise en cache
    if _CACHED_MODEL is None or _CACHED_PATH != str(resolved_model_path.resolve()):
        _CACHED_MODEL, _CACHED_CONFIG = load_trained_model(resolved_model_path)
        _CACHED_PATH = str(resolved_model_path.resolve())

    model = _CACHED_MODEL
    config = _CACHED_CONFIG or {}
    target_size = tuple(config.get("image_size", DEFAULT_IMAGE_SIZE))
    cls_threshold = float(config.get("cls_threshold", 0.5))
    seg_threshold = float(config.get("seg_threshold", 0.5))

    # Lecture de l'image originale pour ses dimensions exactes
    orig_bgr = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
    if orig_bgr is None:
        raise ValueError(f"Impossible de décoder l'image : {img_file}")
    orig_h, orig_w = orig_bgr.shape[:2]

    # Prétraitement (Tâche 1)
    rgb_norm = preprocess_image(img_file, target_size=target_size, normalize=True)
    device = next(model.parameters()).device
    input_tensor = (
        torch.from_numpy(np.transpose(rgb_norm, (2, 0, 1)))
        .unsqueeze(0)
        .float()
        .to(device)
    )

    # Inférence CNN Multi-Tâches (Tâche 2)
    with torch.no_grad():
        cls_logits, seg_logits = model(input_tensor)
        cls_prob = float(torch.sigmoid(cls_logits).item())
        seg_prob_map = torch.sigmoid(seg_logits).squeeze().cpu().numpy()

    # Décision de détection + extraction de la localisation
    location_info, binary_mask = _extract_localization_from_mask(
        prob_mask=seg_prob_map,
        orig_width=orig_w,
        orig_height=orig_h,
        seg_threshold=seg_threshold,
    )

    has_fire = (cls_prob >= cls_threshold) or (location_info["affected_area_pct"] >= 1.5)
    if not has_fire:
        binary_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        location_info = {
            "zone": "Aucune",
            "center_xy": None,
            "normalized_coords": None,
            "gps_coords": None,
            "bbox": None,
            "affected_area_pct": 0.0,
        }
    elif location_info["affected_area_pct"] == 0.0:
        # Si la tête de classification détecte le feu avec un seuil de masque légèrement prudent,
        # on abaisse le seuil de segmentation à 0.30 pour localiser le point chaud principal.
        location_info, binary_mask = _extract_localization_from_mask(
            prob_mask=seg_prob_map,
            orig_width=orig_w,
            orig_height=orig_h,
            seg_threshold=max(0.25, float(seg_prob_map.max()) * 0.65),
        )

    fire_str = "Oui" if has_fire else "Non"
    confidence = round(cls_prob if has_fire else (1.0 - cls_prob), 4)

    if has_fire:
        summary = (
            f"Feu détecté : OUI (Confiance : {confidence*100:.1f}%) | "
            f"Localisation : Secteur {location_info['zone']}, "
            f"Centre={location_info['center_xy']}, "
            f"Surface touchée={location_info['affected_area_pct']:.2f}%"
        )
    else:
        summary = f"Feu détecté : NON (Confiance : {confidence*100:.1f}%) | Aucune anomalie thermique détectée."

    # Sauvegarde optionnelle d'une visualisation superposée (pour Streamlit / Rapport)
    if save_visualization is not None:
        vis_bgr = orig_bgr.copy()
        if has_fire and location_info["bbox"] is not None:
            overlay = vis_bgr.copy()
            overlay[binary_mask > 0] = (0, 0, 255)  # Rouge BGR sur la zone segmentée
            vis_bgr = cv2.addWeighted(overlay, 0.42, vis_bgr, 0.58, 0)

            x1, y1, x2, y2 = location_info["bbox"]
            cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cx, cy = location_info["center_xy"]
            cv2.circle(vis_bgr, (cx, cy), 4, (0, 255, 255), -1)
            label_txt = f"FEU ({location_info['zone']}) - {location_info['affected_area_pct']}%"
            cv2.putText(
                vis_bgr,
                label_txt,
                (max(5, x1), max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
        else:
            cv2.putText(
                vis_bgr,
                "NON-FEU (Zone Saine)",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 200, 0),
                2,
            )

        out_vis_path = Path(save_visualization)
        out_vis_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_vis_path), vis_bgr)

    return {
        "fire_detected": fire_str,
        "has_fire": has_fire,
        "confidence": confidence,
        "location": location_info,
        "summary": summary,
        "segmentation_mask": binary_mask,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        res = predict(sys.argv[1], save_visualization="outputs/prediction_demo.png")
        print(res["summary"])
        print("Détails Localisation :", res["location"])
    else:
        print("Usage : python predict.py <chemin_image_satellite>")
