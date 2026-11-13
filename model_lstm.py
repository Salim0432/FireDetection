"""
Module d'Architecture et d'Entraînement du Modèle RNN/LSTM (Tâche 2 - Moundir)
=============================================================================
Responsable : Moundir (Modélisation Temporelle & Météo - RNN/LSTM)

Ce module implémente :
1. L'architecture récurrente avancée `FirePropagationLSTM` pour séries temporelles météorologiques
   et historiques d'incendies.
2. Une tête de prédiction multi-tâches :
   - Vitesse de propagation instantanée (km/h)
   - Vecteur directionnel unitaire (sin θ, cos θ) garantissant une cohérence angulaire 360°
   - Horizons futurs d'expansion (t+1h, t+2h, t+3h) pour anticiper le front de flamme
3. L'entraînement optimisé avec fonction de perte multi-objectifs (MSE + Perte Angulaire Vectorielle)
4. La sérialisation complète du modèle entraîné au format `.pkl` (Tâche 3).
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from temporal_preprocessing import (
    CARDINAL_8_SECTORS,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    TabularStandardScaler,
    angle_to_cardinal,
    angle_to_sin_cos,
    prepare_datasets_for_training,
    sin_cos_to_angle,
)


class MultiHeadAttentionPooling(nn.Module):
    """Mécanisme d'attention temporelle pour pondérer les pas de temps critiques de la séquence."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, rnn_out: torch.Tensor) -> torch.Tensor:
        # rnn_out: (batch_size, seq_len, hidden_dim)
        scores = self.attn(rnn_out)  # (batch_size, seq_len, 1)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * rnn_out, dim=1)  # (batch_size, hidden_dim)
        return context


class FirePropagationLSTM(nn.Module):
    """
    Réseau Récurrent LSTM pour la prédiction de la propagation des feux de forêt.
    Intègre la dynamique temporelle météo (vent, humidité, température) et l'inertie du feu.
    """

    def __init__(
        self,
        input_dim: int = len(FEATURE_COLUMNS),
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        output_dim: int = len(TARGET_COLUMNS),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.output_dim = output_dim

        effective_hidden = hidden_dim * (2 if bidirectional else 1)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.attn_pool = MultiHeadAttentionPooling(effective_hidden)

        # Tronc commun partagé
        self.shared_fc = nn.Sequential(
            nn.Linear(effective_hidden, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Têtes spécialisées multi-tâches
        # 1. Tête Vitesse (km/h)
        self.speed_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # 2. Tête Direction (vecteur sin, cos)
        self.direction_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        # 3. Tête Expansion Future multi-horizons (+1h, +2h, +3h)
        self.expansion_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tenseur d'entrée (batch_size, seq_len, input_dim)
        Retourne les prédictions concaténées: [speed, sin, cos, exp_1h, exp_2h, exp_3h]
        """
        proj = self.input_proj(x)
        lstm_out, _ = self.lstm(proj)
        context = self.attn_pool(lstm_out)
        shared = self.shared_fc(context)

        speed_out = self.speed_head(shared)
        dir_raw = self.direction_head(shared)
        dir_norm = F.normalize(dir_raw, p=2, dim=-1)  # vecteur unitaire (sin, cos)
        exp_out = self.expansion_head(shared)

        return torch.cat([speed_out, dir_norm, exp_out], dim=-1)


class PropagationMultiTaskLoss(nn.Module):
    """Fonction de perte multi-tâches combinant régression de vitesse, cohérence angulaire et expansion."""

    def __init__(self, speed_weight: float = 1.0, dir_weight: float = 1.5, exp_weight: float = 0.8) -> None:
        super().__init__()
        self.speed_weight = speed_weight
        self.dir_weight = dir_weight
        self.exp_weight = exp_weight
        self.mse = nn.MSELoss()
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Séparation des composantes
        pred_speed = preds[:, 0:1]
        pred_dir = preds[:, 1:3]
        pred_exp = preds[:, 3:6]

        target_speed = targets[:, 0:1]
        target_dir = targets[:, 1:3]
        target_exp = targets[:, 3:6]

        # Perte de vitesse (Smooth L1 robuste aux rafales extrêmes)
        loss_speed = self.smooth_l1(pred_speed, target_speed)

        # Perte vectorielle directionnelle (Cosine Similarity Loss: 1 - cos(pred, tgt))
        target_dir_norm = F.normalize(target_dir, p=2, dim=-1)
        cos_sim = torch.sum(pred_dir * target_dir_norm, dim=-1)
        loss_dir = torch.mean(1.0 - cos_sim)

        # Perte d'expansion multi-horizons
        loss_exp = self.mse(pred_exp, target_exp)

        total_loss = (
            self.speed_weight * loss_speed
            + self.dir_weight * loss_dir
            + self.exp_weight * loss_exp
        )

        metrics = {
            "loss_speed": float(loss_speed.item()),
            "loss_dir": float(loss_dir.item()),
            "loss_exp": float(loss_exp.item()),
            "total_loss": float(total_loss.item()),
        }
        return total_loss, metrics


def save_lstm_model(
    model: FirePropagationLSTM,
    x_scaler: TabularStandardScaler,
    y_scaler: TabularStandardScaler,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    save_path_pkl: str | Path = "models/fire_propagation_lstm.pkl",
    save_path_pt: Optional[str | Path] = "models/fire_propagation_lstm.pt",
) -> str:
    """Sauvegarde le modèle complet, scalers, et métadonnées au format .pkl."""
    path_pkl = Path(save_path_pkl)
    path_pkl.parent.mkdir(parents=True, exist_ok=True)

    state_dict_np = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    payload = {
        "model_type": "FirePropagationLSTM",
        "author": "Moundir - Responsable Modélisation Temporelle & Météo (RNN/LSTM)",
        "config": config,
        "x_scaler": x_scaler.to_dict(),
        "y_scaler": y_scaler.to_dict(),
        "metrics": metrics,
        "cardinal_mapping": {name: (low + high) / 2.0 for name, low, high in CARDINAL_8_SECTORS},
        "state_dict_np": state_dict_np,
    }

    with open(path_pkl, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    if save_path_pt:
        path_pt = Path(save_path_pt)
        path_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": config,
                "x_scaler": x_scaler.to_dict(),
                "y_scaler": y_scaler.to_dict(),
                "metrics": metrics,
            },
            path_pt,
        )

    print(f"Modèle LSTM sauvegardé avec succès dans : {path_pkl.resolve()}")
    return str(path_pkl.resolve())


def load_lstm_model(
    model_path: str | Path = "models/fire_propagation_lstm.pkl",
    device: Optional[str] = None,
) -> Tuple[FirePropagationLSTM, TabularStandardScaler, TabularStandardScaler, Dict[str, Any]]:
    """Charge le modèle sérialisé .pkl et restaure son état d'inférence déterministe."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Modèle introuvable : {path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(path, "rb") as f:
        payload = pickle.load(f)

    config = payload["config"]
    x_scaler = TabularStandardScaler.from_dict(payload["x_scaler"])
    y_scaler = TabularStandardScaler.from_dict(payload["y_scaler"])

    model = FirePropagationLSTM(
        input_dim=config.get("input_dim", len(FEATURE_COLUMNS)),
        hidden_dim=config.get("hidden_dim", 64),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.0),
        bidirectional=config.get("bidirectional", True),
        output_dim=config.get("output_dim", len(TARGET_COLUMNS)),
    )

    state_dict_torch = {k: torch.from_numpy(v) for k, v in payload["state_dict_np"].items()}
    model.load_state_dict(state_dict_torch)
    model.to(device)
    model.eval()

    return model, x_scaler, y_scaler, payload


def train_propagation_lstm(
    csv_source: str | Path = "dataset/forestfires_kaggle.csv",
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 1.5e-3,
    hidden_dim: int = 64,
    num_layers: int = 2,
    sequence_length: int = 4,
    save_path_pkl: str | Path = "models/fire_propagation_lstm.pkl",
    metrics_json_path: str | Path = "outputs/lstm_training_metrics.json",
    seed: int = 42,
) -> Tuple[FirePropagationLSTM, Dict[str, Any]]:
    """
    Entraîne le modèle LSTM de bout en bout sur les séries temporelles météorologiques
    et historiques de propagation (Tâche 2).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("🔥 ENTRAÎNEMENT DU MODÈLE RNN/LSTM - PROPAGATION DU FEU (MOUNDIR)")
    print("=" * 70)
    print(f"Périphérique d'exécution : {device}")

    # 1. Préparation des séries temporelles
    X_train, Y_train, X_val, Y_val, x_scaler, y_scaler = prepare_datasets_for_training(
        csv_source=csv_source,
        sequence_length=sequence_length,
        val_split=0.2,
        seed=seed,
    )

    print(f"Données d'entraînement : {X_train.shape[0]} séquences (seq_len={sequence_length}, features={X_train.shape[2]})")
    print(f"Données de validation   : {X_val.shape[0]} séquences")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = FirePropagationLSTM(
        input_dim=X_train.shape[2],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.2,
        bidirectional=True,
        output_dim=Y_train.shape[1],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = PropagationMultiTaskLoss()

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_speed_mae_kmh": [],
        "val_dir_angular_error_deg": [],
    }

    best_val_loss = float("inf")
    best_state_dict: Optional[Dict[str, Any]] = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_acc = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss, _ = criterion(preds, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            optimizer.step()

            train_loss_acc += loss.item() * len(batch_x)

        scheduler.step()
        train_loss = train_loss_acc / len(train_ds)

        # Validation
        model.eval()
        val_loss_acc = 0.0
        val_preds_list: List[np.ndarray] = []
        val_targets_list: List[np.ndarray] = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                preds = model(batch_x)
                loss, _ = criterion(preds, batch_y)
                val_loss_acc += loss.item() * len(batch_x)

                val_preds_list.append(preds.cpu().numpy())
                val_targets_list.append(batch_y.cpu().numpy())

        val_loss = val_loss_acc / len(val_ds)
        val_preds_all = np.concatenate(val_preds_list, axis=0)
        val_targets_all = np.concatenate(val_targets_list, axis=0)

        # Dénormalisation des prédictions pour calcul des métriques réelles (km/h et degrés)
        pred_denorm = y_scaler.inverse_transform(val_preds_all)
        target_denorm = y_scaler.inverse_transform(val_targets_all)

        pred_speed = np.maximum(0.0, pred_denorm[:, 0])
        true_speed = target_denorm[:, 0]
        speed_mae = float(np.mean(np.abs(pred_speed - true_speed)))

        # Erreur angulaire absolue en degrés
        pred_angles = np.array([sin_cos_to_angle(s, c) for s, c in pred_denorm[:, 1:3]])
        true_angles = np.array([sin_cos_to_angle(s, c) for s, c in target_denorm[:, 1:3]])
        diff = np.abs(pred_angles - true_angles)
        ang_errors = np.minimum(diff, 360.0 - diff)
        mean_ang_error = float(np.mean(ang_errors))

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_speed_mae_kmh"].append(round(speed_mae, 3))
        history["val_dir_angular_error_deg"].append(round(mean_ang_error, 2))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"Époque {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"MAE Vitesse: {speed_mae:.2f} km/h | "
                f"Erreur Angle: {mean_ang_error:.1f}°"
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    final_metrics = {
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "best_val_loss": round(best_val_loss, 4),
        "val_speed_mae_kmh": history["val_speed_mae_kmh"][-1],
        "val_dir_angular_error_deg": history["val_dir_angular_error_deg"][-1],
        "history": history,
    }

    # Sauvegarde du fichier métriques JSON
    metrics_path = Path(metrics_json_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    config = {
        "input_dim": X_train.shape[2],
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": 0.2,
        "bidirectional": True,
        "output_dim": Y_train.shape[1],
        "sequence_length": sequence_length,
        "feature_names": FEATURE_COLUMNS,
        "target_names": TARGET_COLUMNS,
    }

    save_lstm_model(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        config=config,
        metrics=final_metrics,
        save_path_pkl=save_path_pkl,
    )

    print("=" * 70)
    print(f"✅ Entraînement terminé ! Meilleure perte de validation : {best_val_loss:.4f}")
    print(f"MAE Vitesse finale : {final_metrics['val_speed_mae_kmh']} km/h")
    print(f"Erreur directionnelle moyenne : {final_metrics['val_dir_angular_error_deg']}°")
    print("=" * 70)

    return model, final_metrics


if __name__ == "__main__":
    train_propagation_lstm()
