"""
Module de Prétraitement des Images Satellites (Tâche 1 - Salim)
===============================================================
Responsable : Salim (Vision par Ordinateur - CNN / U-Net)

Ce module implémente toutes les étapes de la Tâche 1 :
1. Nettoyage du dataset (détection et suppression/filtrage des images corrompues,
   vides ou illisibles).
2. Redimensionnement standardisé (ex: 128x128) et normalisation des canaux RGB.
3. Génération / raffinement des masques de segmentation du feu (combinaison
   d'indices spectraux couleur/thermique RGB-HSV-YCbCr pour isoler les zones de
   feu actif et de fumée dense lorsque les masques binaires ne sont pas fournis).
4. Augmentation des données (Data Augmentation) synchronisée (image + masque) :
   retournements horizontaux/verticaux, rotations, variations photométriques
   (luminosité/contraste) et bruit gaussien.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


DEFAULT_IMAGE_SIZE: Tuple[int, int] = (128, 128)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_valid(image_path: str | Path, min_std: float = 2.0) -> bool:
    """
    Vérifie si une image satellite est valide, non corrompue et non uniforme (vide/noire).

    Args:
        image_path (str | Path): Chemin vers le fichier image.
        min_std (float): Écart-type minimal des pixels pour rejeter les images toutes noires/blanches.

    Returns:
        bool: True si l'image est saine et exploitable, False sinon.
    """
    path = Path(image_path)
    if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
        return False

    try:
        with Image.open(path) as img:
            img.verify()

        # Re-lecture pour vérifier le contenu matriciel
        arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if arr is None or arr.size == 0:
            return False
        if arr.shape[0] < 16 or arr.shape[1] < 16:
            return False
        if float(np.std(arr)) < min_std:
            return False
        return True
    except Exception:
        return False


def clean_dataset_directory(
    dataset_dir: str | Path, remove_corrupted: bool = False
) -> Dict[str, List[str]]:
    """
    Parcourt un dossier d'images satellites, identifie les fichiers corrompus ou invalides
    et retourne la liste des images propres (Tâche 1 : Nettoyage).

    Args:
        dataset_dir (str | Path): Répertoire racine du dataset.
        remove_corrupted (bool): Si True, supprime physiquement les fichiers corrompus.

    Returns:
        Dict[str, List[str]]: Dictionnaire contenant 'valid_images' et 'corrupted_files'.
    """
    root = Path(dataset_dir)
    valid_images: List[str] = []
    corrupted_files: List[str] = []

    if not root.exists():
        return {"valid_images": valid_images, "corrupted_files": corrupted_files}

    for file_path in sorted(root.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
            if is_image_valid(file_path):
                valid_images.append(str(file_path))
            else:
                corrupted_files.append(str(file_path))
                if remove_corrupted:
                    try:
                        file_path.unlink()
                    except OSError:
                        pass

    return {
        "valid_images": valid_images,
        "corrupted_files": corrupted_files,
    }


def preprocess_image(
    image_input: str | Path | np.ndarray,
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    normalize: bool = True,
    apply_clahe: bool = True,
) -> np.ndarray:
    """
    Charge, nettoie (amélioration du contraste CLAHE optionnelle), redimensionne
    et normalise une image satellite en format RGB.

    Args:
        image_input (str | Path | np.ndarray): Chemin de l'image ou tableau numpy BGR/RGB.
        target_size (Tuple[int, int]): Dimensions cibles (largeur, hauteur).
        normalize (bool): Si True, normalise les pixels dans [0.0, 1.0] (float32).
        apply_clahe (bool): Si True, applique une réduction de voile atmosphérique léger.

    Returns:
        np.ndarray: Image RGB prétraitée de forme (H, W, 3).
    """
    if isinstance(image_input, (str, Path)):
        bgr = cv2.imread(str(image_input), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Impossible de lire l'image : {image_input}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.shape[2] == 4:
            rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        else:
            rgb = image_input.copy()
    else:
        raise TypeError("image_input doit être un chemin (str/Path) ou un np.ndarray.")

    # Redimensionnement bilinéaire standardisé
    resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)

    # Amélioration douce du contraste local sur la luminance (utile pour images satellites)
    if apply_clahe:
        lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
        l_chan = clahe.apply(l_chan)
        lab = cv2.merge((l_chan, a_chan, b_chan))
        resized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    if normalize:
        return resized.astype(np.float32) / 255.0
    return resized


def extract_fire_segmentation_mask(
    rgb_image: np.ndarray,
    has_fire_label: bool = True,
) -> np.ndarray:
    """
    Génère un masque binaire de segmentation (H, W) identifiant la zone touchée par le feu
    à partir d'une image satellite RGB.

    Utilise une fusion de règles spectrales (RGB + HSV + YCbCr) éprouvée en détection
    d'incendies de forêt pour isoler :
    - Les foyers de flammes actives (rouge/orange/jaune intense, forte luminance Y > Cb)
    - Les zones de combustion / braises et panaches thermiques denses.

    Args:
        rgb_image (np.ndarray): Image RGB en uint8 [0..255] ou float32 [0..1] de taille (H, W, 3).
        has_fire_label (bool): Indique si l'image appartient à la classe 'Feu' (True) ou 'Non-Feu' (False).

    Returns:
        np.ndarray: Masque de segmentation float32 dans {0.0, 1.0} de forme (H, W).
    """
    if not has_fire_label:
        return np.zeros(rgb_image.shape[:2], dtype=np.float32)

    if rgb_image.dtype != np.uint8:
        img_u8 = np.clip(rgb_image * 255.0, 0, 255).astype(np.uint8)
    else:
        img_u8 = rgb_image

    r = img_u8[:, :, 0].astype(np.float32)
    g = img_u8[:, :, 1].astype(np.float32)
    b = img_u8[:, :, 2].astype(np.float32)

    # 1. Critère chromatique RGB (Celik & Demirel fire detection rule)
    rgb_rule = (r > 165) & (r >= g) & (g > b) & ((r - b) > 35)

    # 2. Critère dans l'espace HSV (teintes rouge-orangé-jaune + saturation/valeur)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    hsv_rule = ((h <= 32) | (h >= 165)) & (s >= 55) & (v >= 150)

    # 3. Critère dans l'espace YCbCr (luminance élevée et chrominance rouge Cr > Cb)
    ycbcr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2YCrCb)
    y_c = ycbcr[:, :, 0].astype(np.float32)
    cr_c = ycbcr[:, :, 1].astype(np.float32)
    cb_c = ycbcr[:, :, 2].astype(np.float32)
    y_mean, cr_mean, cb_mean = np.mean(y_c), np.mean(cr_c), np.mean(cb_c)
    ycbcr_rule = (
        (y_c >= cb_c)
        & (cr_c >= cb_c)
        & (cr_c > max(135.0, cr_mean * 1.02))
        & (cb_c < min(130.0, cb_mean * 1.05))
    )

    # Fusion des masques spectraux
    combined = (rgb_rule & hsv_rule) | (rgb_rule & ycbcr_rule) | (hsv_rule & ycbcr_rule)
    mask = combined.astype(np.uint8) * 255

    # Opérations morphologiques pour nettoyer le bruit isolé et lisser la zone touchée
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Si l'image est étiquetée "wildfire" (image satellite globale avec panache de fumée/zone brûlée)
    # et que le foyer incandescent pur couvre < 1% des pixels, on segmente l'anomalie thermique/fumée
    fire_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    if fire_ratio < 0.008:
        # Indice d'anomalie thermique / zone brûlée / fumée sur image satellite (NDVI inversé / Red-Edge proxy)
        excess_red_smoke = (r - 0.85 * g) + 0.35 * (255.0 - g)
        thresh_val = np.percentile(excess_red_smoke, 84)
        fallback_mask = (excess_red_smoke >= thresh_val).astype(np.uint8) * 255
        fallback_mask = cv2.morphologyEx(fallback_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fallback_mask = cv2.morphologyEx(fallback_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = fallback_mask

    return (mask > 127).astype(np.float32)


def augment_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applique une augmentation de données aléatoire et synchronisée sur une image
    satellite et son masque de segmentation (Tâche 1 : Augmentation des données).

    Transformations incluses :
    - Flip horizontal et vertical aléatoire
    - Rotation aléatoire (0°, 90°, 180°, 270° ou angle libre [-20°, +20°])
    - Variation photométrique de luminosité et de contraste
    - Bruit gaussien léger (capteur satellite)

    Args:
        image (np.ndarray): Image normalisée float32 [0, 1] de forme (H, W, 3).
        mask (np.ndarray): Masque binaire float32 {0, 1} de forme (H, W).
        rng (Optional[np.random.Generator]): Générateur aléatoire numpy.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (image_augmentée, masque_augmenté).
    """
    if rng is None:
        rng = np.random.default_rng()

    aug_img = image.copy()
    aug_mask = mask.copy()

    # 1. Flip horizontal (p = 0.5)
    if rng.random() < 0.5:
        aug_img = np.fliplr(aug_img).copy()
        aug_mask = np.fliplr(aug_mask).copy()

    # 2. Flip vertical (p = 0.5)
    if rng.random() < 0.5:
        aug_img = np.flipud(aug_img).copy()
        aug_mask = np.flipud(aug_mask).copy()

    # 3. Rotation par multiples de 90° (préserve parfaitement les bords satellites)
    k_rot = int(rng.integers(0, 4))
    if k_rot > 0:
        aug_img = np.rot90(aug_img, k=k_rot).copy()
        aug_mask = np.rot90(aug_mask, k=k_rot).copy()

    # 4. Variation de luminosité et de contraste (p = 0.7)
    if rng.random() < 0.7:
        alpha = float(rng.uniform(0.85, 1.18))  # contraste
        beta = float(rng.uniform(-0.08, 0.08))  # luminosité
        aug_img = np.clip(aug_img * alpha + beta, 0.0, 1.0)

    # 5. Bruit gaussien léger (p = 0.3)
    if rng.random() < 0.3:
        sigma = float(rng.uniform(0.005, 0.025))
        noise = rng.normal(0.0, sigma, size=aug_img.shape).astype(np.float32)
        aug_img = np.clip(aug_img + noise, 0.0, 1.0)

    return aug_img.astype(np.float32), (aug_mask > 0.5).astype(np.float32)
