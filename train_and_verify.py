"""
Script Principal d'Entraînement, Sauvegarde et Validation (Salim - CNN / U-Net)
===============================================================================
Exécute de bout en bout les Tâches 1, 2, 3 et la Mission Secondaire de Salim :
1. Tâche 1 : Téléchargement Kaggle, nettoyage, redimensionnement et augmentation.
2. Tâche 2 : Entraînement du modèle CNN U-Net (`FireUNet`) pour la détection et
   la segmentation de la zone touchée.
3. Tâche 3 : Sauvegarde du modèle final (`.pkl` et `.h5`) et test de `predict(image_path)`.
4. Mission Secondaire : Génération du dataset global synchronisé pour Mondir (LSTM)
   et le Système d'Alerte (Arbre de Décision).
"""

from __future__ import annotations

import json
from pathlib import Path

from data_support import build_global_team_dataset, download_and_prepare_kaggle_dataset
from model_cnn import train_fire_cnn
from predict import predict


def main() -> None:
    print("=" * 72)
    print("  MISSION SALIM : VISION PAR ORDINATEUR (CNN / U-NET) - FIRE DETECTION")
    print("=" * 72)

    # 1. Préparation et nettoyage de 100% du dataset Kaggle (Tâche 1)
    print("\n[Étape 1/4] Préparation, nettoyage et augmentation de 100% du dataset Kaggle...")
    train_samples, val_samples = download_and_prepare_kaggle_dataset(
        max_kaggle_per_class=1000,
        include_satellite_masks=True,
        seed=42,
    )
    print(f"Total Train : {len(train_samples)} images | Total Validation : {len(val_samples)} images")

    # 2. Entraînement du modèle CNN U-Net et sauvegarde .pkl + .h5 (Tâche 2 & Tâche 3)
    print("\n[Étape 2/4] Entraînement du modèle CNN Multi-Tâches (FireUNet)...")
    model, history = train_fire_cnn(
        train_samples=train_samples,
        val_samples=val_samples,
        epochs=10,
        batch_size=16,
        lr=1.2e-3,
        base_filters=16,
        pkl_save_path="models/fire_cnn_unet.pkl",
        h5_save_path="models/fire_cnn_unet.h5",
    )

    # Sauvegarde des métriques d'entraînement pour le rapport
    Path("outputs").mkdir(parents=True, exist_ok=True)
    with open("outputs/training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # 3. Test de la fonction predict(image_path) sur des images Feu et Non-Feu (Tâche 3)
    print("\n[Étape 3/4] Validation de la fonction predict(image_path) (.pkl & .h5)...")
    fire_test_img = next(p for p, lbl, _ in val_samples if lbl == 1)
    nofire_test_img = next(p for p, lbl, _ in val_samples if lbl == 0)

    res_fire_pkl = predict(
        fire_test_img,
        model_path="models/fire_cnn_unet.pkl",
        save_visualization="outputs/demo_fire_segmented.png",
    )
    res_fire_h5 = predict(
        fire_test_img,
        model_path="models/fire_cnn_unet.h5",
    )
    res_nofire = predict(
        nofire_test_img,
        model_path="models/fire_cnn_unet.pkl",
        save_visualization="outputs/demo_nofire_segmented.png",
    )

    print("\n--- Test 1 : Image avec Feu (via .pkl) ---")
    print("  Chemin      :", fire_test_img)
    print("  Feu détecté :", res_fire_pkl["fire_detected"])
    print("  Confiance   :", res_fire_pkl["confidence"])
    print("  Localisation:", res_fire_pkl["location"])

    print("\n--- Test 2 : Vérification identique via modèle .h5 ---")
    print("  Feu détecté :", res_fire_h5["fire_detected"], "| Confiance :", res_fire_h5["confidence"])

    print("\n--- Test 3 : Image sans Feu ---")
    print("  Chemin      :", nofire_test_img)
    print("  Feu détecté :", res_nofire["fire_detected"])
    print("  Confiance   :", res_nofire["confidence"])
    print("  Localisation:", res_nofire["location"])

    # 4. Mission Secondaire : Support Data pour Mondir & Arbre de Décision
    print("\n[Étape 4/4] Génération du dataset global pour l'équipe (Support Data)...")
    all_samples = val_samples + train_samples
    build_global_team_dataset(
        samples=all_samples,
        output_csv="dataset/global_fire_weather_dataset.csv",
        max_rows=517,
    )

    print("\n" + "=" * 72)
    print("  TOUTES LES TÂCHES DE SALIM ONT ÉTÉ EXÉCUTÉES AVEC SUCCÈS !")
    print("=" * 72)


if __name__ == "__main__":
    main()
