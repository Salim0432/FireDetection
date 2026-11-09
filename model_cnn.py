"""
Module d'Architecture et d'Entraînement du Modèle CNN U-Net (Tâche 2 - Salim)
=============================================================================
Responsable : Salim (Vision par Ordinateur - CNN / U-Net)

Ce module implémente :
- L'architecture Multi-Tâches `FireUNet` (CNN Encodeur-Décodeur de type U-Net avec
  connexions résiduelles/skip connections) qui effectue simultanément :
  1. La détection binaire de la présence de feu (Tête de Classification Globale).
  2. La segmentation pixel par pixel de la zone touchée par l'incendie (Tête Décodeur U-Net).
- La fonction de perte hybride combinant Binary Cross-Entropy (BCE) et Dice Loss.
- La sauvegarde et le chargement du modèle final aux formats `.pkl` et `.h5` (Tâche 3).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from preprocessing import (
    DEFAULT_IMAGE_SIZE,
    augment_image_and_mask,
    extract_fire_segmentation_mask,
    preprocess_image,
)


class ConvBlock(nn.Module):
    """Double bloc de convolution (Conv2d -> BatchNorm -> ReLU) x 2."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FireUNet(nn.Module):
    """
    Architecture CNN Multi-Tâches (U-Net + Classification Head) pour :
    1. Détecter la présence de feu dans une image satellite (probabilité scalaire [0, 1]).
    2. Segmenter précisément la zone touchée par le feu (carte de probabilité H x W).
    """

    def __init__(self, in_channels: int = 3, base_filters: int = 16) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_filters = base_filters

        f1 = base_filters          # 16
        f2 = base_filters * 2      # 32
        f3 = base_filters * 4      # 64
        f4 = base_filters * 8      # 128

        # Encodeur (Extraction de caractéristiques multi-échelles)
        self.enc1 = ConvBlock(in_channels, f1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(f1, f2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(f2, f3, dropout=0.1)
        self.pool3 = nn.MaxPool2d(2)

        # Goulot d'étranglement (Bottleneck)
        self.bottleneck = ConvBlock(f3, f4, dropout=0.2)

        # Tête 1 : Classification Globale (Présence de feu : Oui / Non)
        self.classifier_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(f4, f2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(f2, 1),
        )

        # Tête 2 : Décodeur U-Net (Segmentation spatiale de la zone touchée)
        self.up3 = nn.ConvTranspose2d(f4, f3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(f3 + f3, f3)

        self.up2 = nn.ConvTranspose2d(f3, f2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(f2 + f2, f2)

        self.up1 = nn.ConvTranspose2d(f2, f1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(f1 + f1, f1)

        self.seg_out = nn.Conv2d(f1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Passe avant (Forward pass).

        Args:
            x (torch.Tensor): Tensor d'images satellites de forme (B, 3, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - cls_logits : Logits de classification (B, 1)
                - seg_logits : Logits du masque de segmentation (B, 1, H, W)
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        # Classification globale
        cls_logits = self.classifier_head(b)

        # Décodage avec skip connections
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        seg_logits = self.seg_out(d1)
        return cls_logits, seg_logits


class SatelliteFireDataset(Dataset):
    """Dataset PyTorch avec mise en cache RAM pour un entraînement rapide du modèle FireUNet."""

    def __init__(
        self,
        samples: List[Tuple[str, int, Optional[str]]],
        image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        augment: bool = False,
    ) -> None:
        """
        Args:
            samples: Liste de tuples (image_path, label_int, mask_path_optionnel).
            image_size: Dimensions (W, H) de redimensionnement.
            augment: Activer l'augmentation de données (Tâche 1).
        """
        import cv2

        self.samples = samples
        self.image_size = image_size
        self.augment = augment
        self.rng = np.random.default_rng()
        self.cached_items: List[Tuple[np.ndarray, int, np.ndarray]] = []

        for img_path, label, mask_path in samples:
            image = preprocess_image(img_path, target_size=self.image_size, normalize=True, apply_clahe=False)
            if mask_path and Path(mask_path).is_file():
                raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if raw_mask.shape[:2] != (self.image_size[1], self.image_size[0]):
                    raw_mask = cv2.resize(raw_mask, self.image_size, interpolation=cv2.INTER_NEAREST)
                mask = (raw_mask > 127).astype(np.float32)
            else:
                mask = extract_fire_segmentation_mask(image, has_fire_label=bool(label == 1))
            self.cached_items.append((image, int(label), mask))

    def __len__(self) -> int:
        return len(self.cached_items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, label, mask = self.cached_items[idx]
        if self.augment:
            image, mask = augment_image_and_mask(image, mask, rng=self.rng)

        img_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        mask_tensor = torch.from_numpy(mask[np.newaxis, :, :]).float()
        label_tensor = torch.tensor([float(label)], dtype=torch.float32)
        return img_tensor, label_tensor, mask_tensor


def dice_loss_with_logits(seg_logits: torch.Tensor, seg_targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Calcule la perte de Dice (1 - Dice Coefficient) pour la segmentation."""
    probs = torch.sigmoid(seg_logits)
    probs_flat = probs.view(probs.size(0), -1)
    targets_flat = seg_targets.view(seg_targets.size(0), -1)
    intersection = (probs_flat * targets_flat).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + smooth)
    return 1.0 - dice.mean()


def compute_metrics(
    cls_logits: torch.Tensor,
    cls_targets: torch.Tensor,
    seg_logits: torch.Tensor,
    seg_targets: torch.Tensor,
     smooth: float = 1e-6,
) -> Dict[str, float]:
    """Calcule l'exactitude de classification (Accuracy), le score Dice et l'IoU de segmentation."""
    with torch.no_grad():
        cls_preds = (torch.sigmoid(cls_logits) >= 0.5).float()
        acc = (cls_preds == cls_targets).float().mean().item()

        seg_preds = (torch.sigmoid(seg_logits) >= 0.5).float()
        inter = (seg_preds * seg_targets).sum(dim=(1, 2, 3))
        union = (seg_preds + seg_targets).clamp(0, 1).sum(dim=(1, 2, 3))
        dice = ((2.0 * inter + smooth) / (seg_preds.sum(dim=(1, 2, 3)) + seg_targets.sum(dim=(1, 2, 3)) + smooth)).mean().item()
        iou = ((inter + smooth) / (union + smooth)).mean().item()

    return {"accuracy": acc, "dice": dice, "iou": iou}


def save_model_artifacts(
    model: FireUNet,
    pkl_path: str | Path = "models/fire_cnn_unet.pkl",
    h5_path: str | Path = "models/fire_cnn_unet.h5",
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    history: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Sauvegarde le modèle final simultanément aux formats `.pkl` et `.h5` (Tâche 3).

    Args:
        model (FireUNet): Modèle entraîné.
        pkl_path (str | Path): Chemin du fichier `.pkl`.
        h5_path (str | Path): Chemin du fichier `.h5`.
        image_size (Tuple[int, int]): Résolution d'entrée du modèle.
        history (Optional[Dict[str, Any]]): Historique des métriques d'entraînement.

    Returns:
        Dict[str, str]: Chemins absolus des modèles sauvegardés.
    """
    pkl_file = Path(pkl_path)
    h5_file = Path(h5_path)
    pkl_file.parent.mkdir(parents=True, exist_ok=True)
    h5_file.parent.mkdir(parents=True, exist_ok=True)

    # Extraction des poids en numpy (portable entre CPU/GPU et sérialisable en .pkl et .h5)
    state_dict_np = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    config = {
        "architecture": "FireUNet",
        "in_channels": model.in_channels,
        "base_filters": model.base_filters,
        "image_size": list(image_size),
        "cls_threshold": 0.5,
        "seg_threshold": 0.5,
    }

    # 1. Sauvegarde au format .pkl
    payload = {
        "config": config,
        "state_dict_np": state_dict_np,
        "history": history or {},
    }
    with open(pkl_file, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 2. Sauvegarde au format .h5 (HDF5)
    with h5py.File(h5_file, "w") as hf:
        hf.attrs["config_json"] = json.dumps(config)
        if history:
            hf.attrs["history_json"] = json.dumps(history)
        weights_grp = hf.create_group("weights")
        for key, arr in state_dict_np.items():
            if arr.ndim > 0:
                weights_grp.create_dataset(key, data=arr, compression="gzip")
            else:
                weights_grp.create_dataset(key, data=arr)

    return {"pkl_path": str(pkl_file.resolve()), "h5_path": str(h5_file.resolve())}


def load_trained_model(
    model_path: str | Path = "models/fire_cnn_unet.pkl",
    device: Optional[str] = None,
) -> Tuple[FireUNet, Dict[str, Any]]:
    """
    Charge le modèle CNN U-Net depuis un fichier `.pkl` ou `.h5`.

    Args:
        model_path (str | Path): Chemin vers le fichier `.pkl` ou `.h5`.
        device (Optional[str]): 'cpu' ou 'cuda'.

    Returns:
        Tuple[FireUNet, Dict[str, Any]]: (Modèle en mode eval, dictionnaire de configuration).
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Modèle introuvable : {path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if path.suffix.lower() == ".pkl":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        config = payload["config"]
        state_dict_np = payload["state_dict_np"]
    elif path.suffix.lower() in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as hf:
            config = json.loads(hf.attrs["config_json"])
            weights_grp = hf["weights"]
            state_dict_np = {k: np.array(weights_grp[k]) for k in weights_grp.keys()}
    else:
        raise ValueError(f"Format de modèle non supporté ({path.suffix}). Utilisez .pkl ou .h5.")

    model = FireUNet(
        in_channels=int(config.get("in_channels", 3)),
        base_filters=int(config.get("base_filters", 16)),
    )
    state_dict_torch = {k: torch.from_numpy(v) for k, v in state_dict_np.items()}
    model.load_state_dict(state_dict_torch)
    model.to(device)
    model.eval()
    return model, config


def train_fire_cnn(
    train_samples: List[Tuple[str, int, Optional[str]]],
    val_samples: List[Tuple[str, int, Optional[str]]],
    epochs: int = 12,
    batch_size: int = 16,
    lr: float = 1e-3,
    base_filters: int = 16,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    pkl_save_path: str | Path = "models/fire_cnn_unet.pkl",
    h5_save_path: str | Path = "models/fire_cnn_unet.h5",
) -> Tuple[FireUNet, Dict[str, List[float]]]:
    """
    Entraîne le modèle CNN Multi-Tâches (FireUNet) sur le dataset d'images satellites
    et sauvegarde automatiquement le meilleur modèle aux formats `.pkl` et `.h5`.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = SatelliteFireDataset(train_samples, image_size=image_size, augment=True)
    val_ds = SatelliteFireDataset(val_samples, image_size=image_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = FireUNet(in_channels=3, base_filters=base_filters).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    bce_loss = nn.BCEWithLogitsLoss()

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_dice": [],
        "val_iou": [],
    }

    best_score = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n_batches = 0

        for imgs, cls_t, seg_t in train_loader:
            imgs = imgs.to(device)
            cls_t = cls_t.to(device)
            seg_t = seg_t.to(device)

            optimizer.zero_grad()
            cls_logits, seg_logits = model(imgs)

            loss_cls = bce_loss(cls_logits, cls_t)
            loss_seg = 0.5 * bce_loss(seg_logits, seg_t) + 0.5 * dice_loss_with_logits(seg_logits, seg_t)
            loss = loss_cls + 1.2 * loss_seg

            loss.backward()
            optimizer.step()

            metrics = compute_metrics(cls_logits, cls_t, seg_logits, seg_t)
            running_loss += float(loss.item())
            running_acc += metrics["accuracy"]
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_dice = 0.0
        val_iou = 0.0
        v_batches = 0

        with torch.no_grad():
            for imgs, cls_t, seg_t in val_loader:
                imgs = imgs.to(device)
                cls_t = cls_t.to(device)
                seg_t = seg_t.to(device)

                cls_logits, seg_logits = model(imgs)
                loss_cls = bce_loss(cls_logits, cls_t)
                loss_seg = 0.5 * bce_loss(seg_logits, seg_t) + 0.5 * dice_loss_with_logits(seg_logits, seg_t)
                loss = loss_cls + 1.2 * loss_seg

                metrics = compute_metrics(cls_logits, cls_t, seg_logits, seg_t)
                val_loss += float(loss.item())
                val_acc += metrics["accuracy"]
                val_dice += metrics["dice"]
                val_iou += metrics["iou"]
                v_batches += 1

        t_loss = running_loss / max(n_batches, 1)
        t_acc = running_acc / max(n_batches, 1)
        v_loss = val_loss / max(v_batches, 1)
        v_acc = val_acc / max(v_batches, 1)
        v_dice = val_dice / max(v_batches, 1)
        v_iou = val_iou / max(v_batches, 1)

        history["train_loss"].append(round(t_loss, 4))
        history["val_loss"].append(round(v_loss, 4))
        history["train_acc"].append(round(t_acc, 4))
        history["val_acc"].append(round(v_acc, 4))
        history["val_dice"].append(round(v_dice, 4))
        history["val_iou"].append(round(v_iou, 4))

        composite_score = 0.6 * v_acc + 0.4 * v_dice
        if composite_score >= best_score:
            best_score = composite_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[Epoch {epoch:02d}/{epochs:02d}] "
            f"Train Loss: {t_loss:.4f} | Train Acc: {t_acc*100:.1f}% || "
            f"Val Loss: {v_loss:.4f} | Val Acc: {v_acc*100:.1f}% | "
            f"Val Dice: {v_dice:.4f} | Val IoU: {v_iou:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    saved_paths = save_model_artifacts(
        model=model,
        pkl_path=pkl_save_path,
        h5_path=h5_save_path,
        image_size=image_size,
        history=history,
    )
    print(f"Modèle sauvegardé avec succès :\n  - PKL : {saved_paths['pkl_path']}\n  - H5  : {saved_paths['h5_path']}")
    return model, history
