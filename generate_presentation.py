"""
Générateur de la Présentation PowerPoint de Soutenance — Projet 5
=================================================================
Système Multi-Agents de Surveillance et de Confinement Autonome des Feux de Forêt
Équipe :
  - Salim Chablaoui (Agent 1 : Vision par Ordinateur — CNN / U-Net & Support Data)
  - Moundir Boucheriha (Agent 2 : Modélisation Temporelle — Bi-LSTM + Attention)
  - Asma Dadou (Agent 3 : Système d'Alerte & Intégration — Arbre de Décision, Contrat & Streamlit)

Structure narrative orientée « Problème -> Rôle de chaque Agent -> Comment nous l'avons résolu » :
  - Slide 1  : Page de garde & Équipe
  - Slide 2  : Le Problème Opérationnel & Décomposition en 3 Défis Techniques
  - Slide 3  : Notre Solution Globale : Architecture Multi-Agents & Contrat (`contracts.py`)
  - Slide 4  : Comment nous avons construit & unifié les données (`data_support.py`)
  - Slide 5  : Agent 1 (Salim) — Son rôle & Comment nous avons prétraité les images (`preprocessing.py`)
  - Slide 6  : Agent 1 (Salim) — Comment nous avons conçu `FireUNet` & résolu la localisation (`model_cnn.py`, `predict.py`)
  - Slide 7  : Agent 1 (Salim) — Résultats expérimentaux & Segmentation visuelle
  - Slide 8  : Agent 2 (Moundir) — Son rôle & Comment nous avons modélisé la physique météo (`temporal_preprocessing.py`)
  - Slide 9  : Agent 2 (Moundir) — Comment fonctionne `FirePropagationLSTM` & Résultats (`model_lstm.py`)
  - Slide 10 : Agent 3 (Asma) — Son rôle & Comment l'Arbre de Décision prend une décision explicable (`train_tree.py`)
  - Slide 11 : Agent 3 (Asma) — Comment nous avons connecté le Pipeline (`main.py`) & l'App (`app.py`) + Performances CART
  - Slide 12 : Démonstration Pas-à-Pas : Comment une alerte est générée de bout en bout
  - Slide 13 : Bilan Comparatif des 3 Agents & Validation Méthodologique (Anti-Overfitting)
  - Slide 14 : Perspectives d'Évolution & Conclusion
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt
from sklearn.metrics import confusion_matrix


# Palette de couleurs harmonisée avec le Rapport Général PDF
NAVY = RGBColor(27, 54, 93)          # #1B365D - Bleu marine académique
DARK_TEXT = RGBColor(30, 41, 59)     # #1E293B - Gris anthracite foncé
MUTED_TEXT = RGBColor(100, 116, 139) # #64748B - Gris ardoise secondaire
FIRE_ORANGE = RGBColor(217, 83, 47)  # #D9532F - Terre de feu / accent Agent 1
TEAL = RGBColor(20, 130, 115)        # #148273 - Vert émeraude / accent Agent 2
PURPLE = RGBColor(109, 40, 217)      # #6D28D9 - Violet / Contrat & Ingénierie
BG_LIGHT = RGBColor(248, 250, 252)   # #F8FAFC - Fond clair de diapositive
CARD_BG = RGBColor(255, 255, 255)    # #FFFFFF - Fond de carte
CARD_ALT = RGBColor(241, 245, 249)   # #F1F5F9 - Fond de carte secondaire
BORDER_GRAY = RGBColor(203, 213, 225)# #CBD5E1 - Bordure subtile
WHITE = RGBColor(255, 255, 255)

# Couleurs des paliers d'alerte
GREEN_ALERT = RGBColor(22, 163, 74)
GREEN_BG = RGBColor(220, 252, 231)
ORANGE_ALERT = RGBColor(217, 119, 6)
ORANGE_BG = RGBColor(254, 243, 199)
RED_ALERT = RGBColor(220, 38, 38)
RED_BG = RGBColor(254, 226, 226)
EVAC_ALERT = RGBColor(153, 27, 27)
EVAC_BG = RGBColor(254, 205, 211)


def generate_visual_assets() -> None:
    """Génère les graphiques haute résolution des 3 agents (CNN, Bi-LSTM, Arbre de Décision)."""
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "axes.edgecolor": "#CBD5E1"})

    # ------------------------------------------------------------------
    # 1. Courbes d'évaluation CNN (cnn_evaluation_plots.png)
    # ------------------------------------------------------------------
    with open(out_dir / "training_metrics.json", "r", encoding="utf-8") as f:
        m_cnn = json.load(f)

    epochs_cnn = np.arange(1, len(m_cnn["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7), dpi=220)
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.plot(epochs_cnn, m_cnn["train_loss"], color="#D9532F", lw=2.4, marker="o", markersize=4, label="Train Loss")
    ax.plot(epochs_cnn, m_cnn["val_loss"], color="#1B365D", lw=2.4, ls="--", marker="s", markersize=4, label="Val Loss")
    ax.set_title("Convergence de la Perte Hybride", fontsize=11, fontweight="bold", color="#1B365D", pad=8)
    ax.set_xlabel("Époques", fontsize=9.5)
    ax.set_ylabel("BCE + Dice Loss", fontsize=9.5)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=8.5)

    ax = axes[1]
    train_acc_pct = [v * 100 for v in m_cnn["train_acc"]]
    val_acc_pct = [v * 100 for v in m_cnn["val_acc"]]
    ax.plot(epochs_cnn, train_acc_pct, color="#64748B", lw=2.0, ls=":", marker="o", markersize=4, label="Train Acc (%)")
    ax.plot(epochs_cnn, val_acc_pct, color="#148273", lw=2.5, marker="o", markersize=4, label="Val Acc (%)")
    ax.axhline(99.11, color="#D9532F", ls="--", lw=1.5, label="Pic : 99,11 %")
    ax.set_title("Classification Feu Oui/Non", fontsize=11, fontweight="bold", color="#1B365D", pad=8)
    ax.set_xlabel("Époques", fontsize=9.5)
    ax.set_ylabel("Accuracy (%)", fontsize=9.5)
    ax.set_ylim(65, 101)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=8.5, loc="lower right")

    ax = axes[2]
    dice_pct = [v * 100 for v in m_cnn["val_dice"]]
    iou_pct = [v * 100 for v in m_cnn["val_iou"]]
    ax.plot(epochs_cnn, dice_pct, color="#D9532F", lw=2.5, marker="s", markersize=4, label="Dice (84,2 %)")
    ax.plot(epochs_cnn, iou_pct, color="#1B365D", lw=2.2, marker="^", markersize=4, label="IoU (79,5 %)")
    ax.set_title("Segmentation Spatiale U-Net", fontsize=11, fontweight="bold", color="#1B365D", pad=8)
    ax.set_xlabel("Époques", fontsize=9.5)
    ax.set_ylabel("Score (%)", fontsize=9.5)
    ax.set_ylim(55, 90)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=8.5, loc="lower right")

    plt.tight_layout()
    fig.savefig(out_dir / "cnn_evaluation_plots.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Panneau visuel de segmentation 3 étapes (cnn_comparison_panel.png)
    # ------------------------------------------------------------------
    from predict import predict

    sample_img = Path("dataset/kaggle_cleaned/images/fire/kaggle_fire_0002.png")
    if not sample_img.exists():
        sample_img = next(Path("dataset/satellite_samples/images/fire").glob("*.png"))

    res = predict(sample_img)
    bgr = cv2.imread(str(sample_img))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask = res["segmentation_mask"]

    overlay = rgb.copy()
    overlay[mask > 0] = [235, 45, 35]
    blended = cv2.addWeighted(overlay, 0.45, rgb, 0.55, 0)

    fig2, ax2 = plt.subplots(1, 3, figsize=(10.2, 3.2), dpi=220)
    fig2.patch.set_facecolor("white")

    ax2[0].imshow(rgb)
    ax2[0].set_title("1. Image d'Entrée (128×128)", fontsize=10, fontweight="bold", color="#1B365D")
    ax2[0].axis("off")

    ax2[1].imshow(mask, cmap="inferno")
    ax2[1].set_title("2. Masque U-Net Prédit", fontsize=10, fontweight="bold", color="#D9532F")
    ax2[1].axis("off")

    ax2[2].imshow(blended)
    if res["location"]["bbox"]:
        x1, y1, x2, y2 = res["location"]["bbox"]
        cx, cy = res["location"]["center_xy"]
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#FACC15", linewidth=2.0)
        ax2[2].add_patch(rect)
        ax2[2].scatter([cx], [cy], c="#FACC15", s=55, edgecolors="black", zorder=5)
    zone_lbl = res["location"]["zone"]
    area_lbl = res["location"]["affected_area_pct"]
    ax2[2].set_title(f"3. Localisation ({zone_lbl}, {area_lbl}%)", fontsize=10, fontweight="bold", color="#148273")
    ax2[2].axis("off")

    plt.tight_layout()
    fig2.savefig(out_dir / "cnn_comparison_panel.png", dpi=220, bbox_inches="tight")
    plt.close(fig2)

    # ------------------------------------------------------------------
    # 3. Courbes d'évaluation Bi-LSTM à jour (lstm_evaluation_plots.png)
    # ------------------------------------------------------------------
    with open(out_dir / "lstm_training_metrics.json", "r", encoding="utf-8") as f:
        m_lstm = json.load(f)

    hist_l = m_lstm["history"]
    ep_l = np.arange(1, len(hist_l["train_loss"]) + 1)
    fig3, ax3 = plt.subplots(1, 3, figsize=(13.5, 3.9), dpi=220)
    fig3.patch.set_facecolor("white")

    ax3[0].plot(ep_l, hist_l["train_loss"], color="#D9532F", lw=2.3, label="Train Loss")
    ax3[0].plot(ep_l, hist_l["val_loss"], color="#1B365D", lw=2.3, ls="--", label="Val Loss (Épisodes isolés)")
    ax3[0].set_title("Convergence Sans Fuite (Loss)", fontsize=11, fontweight="bold", color="#1B365D", pad=8)
    ax3[0].set_xlabel("Époques", fontsize=9.5)
    ax3[0].set_ylabel("Perte Multi-Tâches", fontsize=9.5)
    ax3[0].grid(True, alpha=0.25)
    ax3[0].legend(frameon=True, fontsize=8.5)

    ax3[1].plot(ep_l, hist_l["val_speed_mae_kmh"], color="#148273", lw=2.3, marker="o", markersize=3.5, label="MAE Vitesse")
    ax3[1].axhline(m_lstm["val_speed_mae_kmh"], color="#D9532F", ls=":", lw=1.8, label=f"Finale : {m_lstm['val_speed_mae_kmh']:.2f} km/h")
    ax3[1].set_title("Erreur Vitesse de Propagation", fontsize=11, fontweight="bold", color="#1B365D", pad=8)
    ax3[1].set_xlabel("Époques", fontsize=9.5)
    ax3[1].set_ylabel("MAE (km/h)", fontsize=9.5)
    ax3[1].grid(True, alpha=0.25)
    ax3[1].legend(frameon=True, fontsize=8.5)

    ax3[2].plot(ep_l, hist_l["val_dir_angular_error_deg"], color="#1B365D", lw=2.3, marker="s", markersize=3.5, label="Erreur Angulaire")
    ax3[2].axhline(m_lstm["val_dir_angular_error_deg"], color="#D9532F", ls=":", lw=1.8, label=f"Finale : {m_lstm['val_dir_angular_error_deg']:.1f}°")
    ax3[2].set_title("Précision Angulaire (Cercle Unité)", fontsize=11, fontweight="bold", color="#1B365D", pad=8)
    ax3[2].set_xlabel("Époques", fontsize=9.5)
    ax3[2].set_ylabel("Erreur Moyenne (°)", fontsize=9.5)
    ax3[2].grid(True, alpha=0.25)
    ax3[2].legend(frameon=True, fontsize=8.5)

    plt.tight_layout()
    fig3.savefig(out_dir / "lstm_evaluation_plots.png", dpi=220, bbox_inches="tight")
    plt.close(fig3)

    # ------------------------------------------------------------------
    # 4. Visuel Arbre de Décision — Importance des variables & Matrice de Confusion
    # ------------------------------------------------------------------
    from contracts import ALERT_LABEL_TO_CODE

    tree = joblib.load("models/mon_arbre_alerte.pkl")
    df_g = pd.read_csv("dataset/global_fire_weather_dataset.csv")
    X_g = pd.DataFrame({
        "feu": (df_g["fire_detected"] == "Oui").astype(int),
        "vent": df_g["wind_speed_kmh"],
        "humidite": df_g["humidity_pct"],
        "vitesse_propagation": df_g["propagation_speed_kmh"],
    })
    y_true = df_g["alert_level"].map(ALERT_LABEL_TO_CODE).to_numpy()
    y_pred = tree.predict(X_g)
    cm = confusion_matrix(y_true, y_pred)

    fig4, (ax_imp, ax_cm) = plt.subplots(1, 2, figsize=(9.4, 3.5), dpi=220)
    fig4.patch.set_facecolor("white")

    feat_labels = ["Feu (CNN)", "Vent (km/h)", "Humidité (%)", "Vitesse (LSTM)"]
    importances = tree.feature_importances_ * 100.0
    bar_colors = ["#D9532F", "#1B365D", "#6D28D9", "#148273"]
    y_pos = np.arange(len(feat_labels))
    bars = ax_imp.barh(y_pos, importances, color=bar_colors, height=0.58)
    ax_imp.set_yticks(y_pos)
    ax_imp.set_yticklabels(feat_labels, fontsize=9.5, fontweight="bold")
    ax_imp.invert_yaxis()
    ax_imp.set_xlabel("Importance Gini (%)", fontsize=9.5)
    ax_imp.set_title("Poids Décisionnel des Entrées", fontsize=10.5, fontweight="bold", color="#1B365D")
    ax_imp.set_xlim(0, max(importances) * 1.25)
    ax_imp.grid(True, axis="x", alpha=0.25)
    for bar, val in zip(bars, importances):
        ax_imp.text(val + 1.0, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=8.8, fontweight="bold", color="#1E293B")

    im = ax_cm.imshow(cm, cmap="Blues")
    cls_names = ["Vert", "Orange", "Rouge", "Évac."]
    ax_cm.set_xticks(range(4))
    ax_cm.set_yticks(range(4))
    ax_cm.set_xticklabels(cls_names, fontsize=9, fontweight="bold")
    ax_cm.set_yticklabels(cls_names, fontsize=9, fontweight="bold")
    ax_cm.set_xlabel("Palier Prédit", fontsize=9.5)
    ax_cm.set_ylabel("Palier Réel (517 cas)", fontsize=9.5)
    ax_cm.set_title("Matrice de Confusion (98,45 %)", fontsize=10.5, fontweight="bold", color="#1B365D")
    thresh = cm.max() / 2.0
    for i in range(4):
        for j in range(4):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9.5, fontweight="bold", color="white" if cm[i, j] > thresh else "#1E293B")

    plt.tight_layout()
    fig4.savefig(out_dir / "decision_tree_visual.png", dpi=220, bbox_inches="tight")
    plt.close(fig4)


# ==============================================================================
# Fonctions utilitaires de mise en forme PowerPoint
# ==============================================================================
def set_slide_background(slide, color: RGBColor = BG_LIGHT) -> None:
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_header_and_footer(
    slide,
    title_text: str,
    section_tag: str,
    slide_num: int,
    total_slides: int = 14,
) -> None:
    set_slide_background(slide, BG_LIGHT)

    top_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.08))
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = FIRE_ORANGE
    top_bar.line.fill.background()

    tag_box = slide.shapes.add_textbox(Inches(0.6), Inches(0.20), Inches(10.0), Inches(0.28))
    tf_tag = tag_box.text_frame
    tf_tag.word_wrap = True
    tf_tag.margin_left = tf_tag.margin_top = tf_tag.margin_bottom = 0
    p_tag = tf_tag.paragraphs[0]
    p_tag.text = section_tag.upper()
    p_tag.font.name = "Calibri"
    p_tag.font.size = Pt(10.5)
    p_tag.font.bold = True
    p_tag.font.color.rgb = FIRE_ORANGE

    title_box = slide.shapes.add_textbox(Inches(0.6), Inches(0.44), Inches(12.1), Inches(0.56))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    tf_title.margin_left = tf_title.margin_top = tf_title.margin_bottom = 0
    p_title = tf_title.paragraphs[0]
    p_title.text = title_text
    p_title.font.name = "Calibri"
    p_title.font.size = Pt(22.5)
    p_title.font.bold = True
    p_title.font.color.rgb = NAVY

    sep = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(1.04), Inches(12.133), Inches(0.02))
    sep.fill.solid()
    sep.fill.fore_color.rgb = BORDER_GRAY
    sep.line.fill.background()

    footer_sep = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(7.08), Inches(12.133), Inches(0.015))
    footer_sep.fill.solid()
    footer_sep.fill.fore_color.rgb = BORDER_GRAY
    footer_sep.line.fill.background()

    footer_left = slide.shapes.add_textbox(Inches(0.6), Inches(7.13), Inches(9.8), Inches(0.28))
    tf_fl = footer_left.text_frame
    tf_fl.margin_left = tf_fl.margin_top = 0
    p_fl = tf_fl.paragraphs[0]
    p_fl.text = "Projet 5 — Surveillance & Confinement Autonome des Feux de Forêt  |  Salim Chablaoui · Moundir Boucheriha · Asma Dadou"
    p_fl.font.name = "Calibri"
    p_fl.font.size = Pt(9.5)
    p_fl.font.color.rgb = MUTED_TEXT

    footer_right = slide.shapes.add_textbox(Inches(11.3), Inches(7.13), Inches(1.433), Inches(0.28))
    tf_fr = footer_right.text_frame
    tf_fr.margin_right = tf_fr.margin_top = 0
    p_fr = tf_fr.paragraphs[0]
    p_fr.alignment = PP_ALIGN.RIGHT
    p_fr.text = f"{slide_num} / {total_slides}"
    p_fr.font.name = "Calibri"
    p_fr.font.size = Pt(10)
    p_fr.font.bold = True
    p_fr.font.color.rgb = NAVY


def add_card(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    bg_color: RGBColor = CARD_BG,
    border_color: RGBColor = BORDER_GRAY,
    accent_top_color: RGBColor | None = None,
):
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left),
        Inches(top),
        Inches(width),
        Inches(height),
    )
    card.adjustments[0] = 0.03
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = border_color
    card.line.width = Pt(1.0)

    if accent_top_color is not None:
        bar = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            Inches(left + 0.02),
            Inches(top),
            Inches(width - 0.04),
            Inches(0.07),
        )
        bar.fill.solid()
        bar.fill.fore_color.rgb = accent_top_color
        bar.line.fill.background()
    return card


def add_bullets_box(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    heading: str | None,
    bullets: list[tuple[str, str]],
    heading_color: RGBColor = NAVY,
    font_size: float = 12.5,
) -> None:
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.06)
    tf.margin_right = Inches(0.06)
    tf.margin_top = Inches(0.04)

    first = True
    if heading:
        p = tf.paragraphs[0]
        p.text = heading
        p.font.name = "Calibri"
        p.font.size = Pt(font_size + 2.2)
        p.font.bold = True
        p.font.color.rgb = heading_color
        p.space_after = Pt(5)
        first = False

    for bold_prefix, body in bullets:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(4.5)
        p.line_spacing = 1.13

        r1 = p.add_run()
        r1.text = f"▪ {bold_prefix}"
        r1.font.name = "Calibri"
        r1.font.size = Pt(font_size)
        r1.font.bold = True
        r1.font.color.rgb = DARK_TEXT

        if body:
            r2 = p.add_run()
            r2.text = f" {body}"
            r2.font.name = "Calibri"
            r2.font.size = Pt(font_size)
            r2.font.bold = False
            r2.font.color.rgb = DARK_TEXT


def add_mission_banner(
    slide,
    role_label: str,
    problem_and_solution_text: str,
    accent_color: RGBColor,
) -> None:
    """Ajoute un bandeau synthétique en haut de diapositive expliquant le rôle de l'agent et comment il résout son défi."""
    add_card(slide, 0.6, 1.18, 12.13, 0.92, bg_color=CARD_BG, border_color=accent_color, accent_top_color=accent_color)
    tb = slide.shapes.add_textbox(Inches(0.8), Inches(1.28), Inches(11.75), Inches(0.74))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.line_spacing = 1.15
    r1 = p.add_run()
    r1.text = f"{role_label} : "
    r1.font.name = "Calibri"
    r1.font.size = Pt(12.2)
    r1.font.bold = True
    r1.font.color.rgb = accent_color

    r2 = p.add_run()
    r2.text = problem_and_solution_text
    r2.font.name = "Calibri"
    r2.font.size = Pt(12.0)
    r2.font.color.rgb = DARK_TEXT


def add_kpi_badge(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    value_str: str,
    label_str: str,
    accent_color: RGBColor,
) -> None:
    add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=BORDER_GRAY, accent_top_color=accent_color)
    tb = slide.shapes.add_textbox(Inches(left + 0.1), Inches(top + 0.10), Inches(width - 0.2), Inches(height - 0.14))
    tf = tb.text_frame
    tf.word_wrap = True

    p1 = tf.paragraphs[0]
    p1.alignment = PP_ALIGN.CENTER
    p1.text = value_str
    p1.font.name = "Calibri"
    p1.font.size = Pt(22)
    p1.font.bold = True
    p1.font.color.rgb = accent_color

    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    p2.text = label_str
    p2.font.name = "Calibri"
    p2.font.size = Pt(10.5)
    p2.font.bold = True
    p2.font.color.rgb = MUTED_TEXT


def add_speaker_notes(slide, text: str) -> None:
    notes_slide = slide.notes_slide
    notes_slide.notes_text_frame.text = text


# ==============================================================================
# Construction des 14 diapositives (Problème -> Agents -> Comment nous l'avons résolu)
# ==============================================================================
def build_presentation() -> Path:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    total_slides = 14

    # --------------------------------------------------------------------------
    # SLIDE 1 : PAGE DE TITRE & PRÉSENTATION DES 3 AGENTS
    # --------------------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1, NAVY)

    accent_side = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.28), Inches(7.5))
    accent_side.fill.solid()
    accent_side.fill.fore_color.rgb = FIRE_ORANGE
    accent_side.line.fill.background()

    tb_sub = s1.shapes.add_textbox(Inches(0.9), Inches(0.80), Inches(11.5), Inches(0.45))
    p = tb_sub.text_frame.paragraphs[0]
    p.text = "PROJET 5  —  DATA MINING & DEEP LEARNING  |  4ème ANNÉE INGÉNIORAT"
    p.font.name = "Calibri"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = RGBColor(253, 186, 116)

    tb_main = s1.shapes.add_textbox(Inches(0.9), Inches(1.40), Inches(11.6), Inches(2.0))
    tf_m = tb_main.text_frame
    tf_m.word_wrap = True
    p1 = tf_m.paragraphs[0]
    p1.text = "Système Multi-Agents de Surveillance\net de Confinement Autonome des Feux de Forêt"
    p1.font.name = "Calibri"
    p1.font.size = Pt(35)
    p1.font.bold = True
    p1.font.color.rgb = WHITE
    p1.line_spacing = 1.15

    tb_desc = s1.shapes.add_textbox(Inches(0.9), Inches(3.32), Inches(11.4), Inches(0.85))
    tf_d = tb_desc.text_frame
    tf_d.word_wrap = True
    pd_p = tf_d.paragraphs[0]
    pd_p.text = "Du Problème Terrain à la Solution IA Complète : Comment 3 Agents Spécialisés (Vision U-Net • Propagation Bi-LSTM • Décision CART) collaborent pour détecter, anticiper et confiner les incendies."
    pd_p.font.name = "Calibri"
    pd_p.font.size = Pt(14.5)
    pd_p.font.color.rgb = RGBColor(203, 213, 225)

    members = [
        ("Salim Chablaoui", "Agent 1 — Vision Spatiale", "Détecter & localiser le feu sur satellite\nCNN Multi-Tâches (FireUNet) & Data", FIRE_ORANGE),
        ("Moundir Boucheriha", "Agent 2 — Dynamique & Météo", "Prédire vitesse, direction & expansion\nBi-LSTM + Attention & Physique du feu", TEAL),
        ("Asma Dadou", "Agent 3 — Décision & Intégration", "Déclencher l'alerte explicable & UI\nArbre CART, Contrat & App Streamlit", RGBColor(234, 179, 8)),
    ]
    for idx, (name, role, tech, col) in enumerate(members):
        x = 0.9 + idx * 3.95
        card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(4.48), Inches(3.65), Inches(2.05))
        card.adjustments[0] = 0.04
        card.fill.solid()
        card.fill.fore_color.rgb = RGBColor(38, 69, 115)
        card.line.color.rgb = col
        card.line.width = Pt(1.8)

        tb_c = s1.shapes.add_textbox(Inches(x + 0.18), Inches(4.64), Inches(3.3), Inches(1.75))
        tf_c = tb_c.text_frame
        tf_c.word_wrap = True
        pc1 = tf_c.paragraphs[0]
        pc1.text = name
        pc1.font.name = "Calibri"
        pc1.font.size = Pt(17.5)
        pc1.font.bold = True
        pc1.font.color.rgb = WHITE

        pc2 = tf_c.add_paragraph()
        pc2.text = role
        pc2.font.name = "Calibri"
        pc2.font.size = Pt(12)
        pc2.font.bold = True
        pc2.font.color.rgb = RGBColor(253, 186, 116)
        pc2.space_before = Pt(4)

        pc3 = tf_c.add_paragraph()
        pc3.text = tech
        pc3.font.name = "Calibri"
        pc3.font.size = Pt(11.2)
        pc3.font.color.rgb = RGBColor(226, 232, 240)
        pc3.space_before = Pt(6)

    tb_date = s1.shapes.add_textbox(Inches(0.9), Inches(6.85), Inches(11.5), Inches(0.4))
    p_dt = tb_date.text_frame.paragraphs[0]
    p_dt.text = "Dépôt du projet : FireDetection/   |   Architecture Multi-Agents Validée de Bout en Bout   |   Novembre 2026"
    p_dt.font.name = "Calibri"
    p_dt.font.size = Pt(11)
    p_dt.font.color.rgb = RGBColor(148, 163, 184)

    add_speaker_notes(
        s1,
        "Bonjour à toutes et à tous. Nous allons vous présenter le problème opérationnel des feux de forêt, le rôle précis de chacun de nos 3 agents intelligents (Salim, Moundir et Asma), et surtout comment nous avons conçu, entraîné et connecté notre solution étape par étape."
    )

    # --------------------------------------------------------------------------
    # SLIDE 2 : LE PROBLÈME OPÉRATIONNEL & POURQUOI 3 AGENTS SPÉCIALISÉS ?
    # --------------------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    add_header_and_footer(s2, "Quel est Notre Problème ? Du Constat Terrain aux 3 Défis IA", "1. Définition du Problème", 2, total_slides)

    add_card(s2, 0.6, 1.25, 5.75, 5.6, bg_color=CARD_BG, accent_top_color=RED_ALERT)
    add_bullets_box(
        s2, 0.8, 1.45, 5.35, 5.25,
        "Le Problème Réel : Course Contre la Montre",
        [
            ("1. Détection humaine trop tardive :", "les tours de guet et appels témoins repèrent souvent l'incendie lorsqu'il a déjà échappé au contrôle initial."),
            ("2. Emballement météo imprévisible :", "sous chaleur (> 30 °C), air sec (RH < 25 %) et vent fort (> 35 km/h), un petit foyer triple d'ampleur en moins d'une heure."),
            ("3. Limite d'un modèle IA unique :", "une image seule ne connaît pas le vent futur ; un capteur météo seul ignore s'il y a un feu ; une boîte noire ne peut pas justifier un ordre d'évacuation."),
            ("Notre Idée Clé :", "diviser ce problème complexe en 3 sous-problèmes complémentaires résolus par 3 agents qui coopèrent."),
        ],
        heading_color=RED_ALERT,
        font_size=12.6,
    )

    sub_problems = [
        ("Sous-Problème 1 → Confié à l'Agent 1 (Salim)",
         "« Y a-t-il un feu sur l'image satellite et OÙ se situe exactement la zone brûlée ? »\n→ Résolu par Vision par Ordinateur & Segmentation pixel-par-pixel (CNN FireUNet).",
         FIRE_ORANGE),
        ("Sous-Problème 2 → Confié à l'Agent 2 (Moundir)",
         "« Vers quelle DIRECTION et à quelle VITESSE le feu va-t-il avancer à +1h, +2h, +3h ? »\n→ Résolu par Modélisation Séquentielle Météo & Physique du Feu (Bi-LSTM + Attention).",
         TEAL),
        ("Sous-Problème 3 → Confié à l'Agent 3 (Asma)",
         "« Quel PALIER D'ALERTE déclencher immédiatement et comment l'expliquer aux secours ? »\n→ Résolu par un Arbre de Décision explicable (CART), un Contrat strict et Streamlit.",
         NAVY),
    ]
    for i, (h_txt, b_txt, col) in enumerate(sub_problems):
        y = 1.25 + i * 1.92
        add_card(s2, 6.65, y, 6.08, 1.75, bg_color=CARD_BG, accent_top_color=col)
        add_bullets_box(s2, 6.85, y + 0.15, 5.7, 1.5, h_txt, [("Question & Approche :", b_txt)], heading_color=col, font_size=11.8)

    add_speaker_notes(
        s2,
        "Cette diapositive pose clairement notre problématique : un incendie est un phénomène à la fois visuel, météorologique et décisionnel. Aucun modèle isolé ne peut tout faire. Nous avons donc découpé le problème en 3 questions précises confiées à 3 agents complémentaires."
    )

    # --------------------------------------------------------------------------
    # SLIDE 3 : COMMENT NOTRE SOLUTION FONCTIONNE — ARCHITECTURE & CONTRAT
    # --------------------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    add_header_and_footer(s3, "Comment Nous Avons Architecturé la Solution : Pipeline & Contrat", "2. Vue d'Ensemble de la Solution", 3, total_slides)

    agents_spec = [
        ("AGENT 1 : VISION (Salim)", FIRE_ORANGE, 0.6, [
            "1. Reçoit l'image satellite (128×128)",
            "2. Nettoie & normalise les canaux RGB",
            "3. Exécute FireUNet (Double Tête)",
            "4. Détecte Feu : Oui / Non + Confiance",
            "5. Extrait Zone, Barycentre, GPS & % Brûlé",
        ]),
        ("AGENT 2 : PROPAGATION (Moundir)", TEAL, 4.75, [
            "1. Reçoit Localisation CNN + Météo",
            "2. Encode le vent en (sin θ, cos θ) + FWI",
            "3. Construit la séquence horaire (4 pas)",
            "4. Exécute FirePropagationLSTM + Attention",
            "5. Prédit Vitesse (km/h), Angle & +1h..+3h",
        ]),
        ("AGENT 3 : DÉCISION & UI (Asma)", NAVY, 8.9, [
            "1. Fusionne [Feu, Vent, Humidité, Vitesse]",
            "2. Évalue DecisionTreeClassifier (CART)",
            "3. Produit Vert / Orange / Rouge / Évacuation",
            "4. Associe la recommandation opérationnelle",
            "5. Affiche tout en direct sur Streamlit (app.py)",
        ]),
    ]

    for title_a, col_a, x_pos, steps in agents_spec:
        add_card(s3, x_pos, 1.28, 3.83, 4.30, bg_color=CARD_BG, accent_top_color=col_a)
        tb_h = s3.shapes.add_textbox(Inches(x_pos + 0.12), Inches(1.42), Inches(3.59), Inches(0.42))
        p_h = tb_h.text_frame.paragraphs[0]
        p_h.alignment = PP_ALIGN.CENTER
        p_h.text = title_a
        p_h.font.name = "Calibri"
        p_h.font.size = Pt(13.5)
        p_h.font.bold = True
        p_h.font.color.rgb = col_a

        for s_idx, step_txt in enumerate(steps):
            y_box = 1.96 + s_idx * 0.68
            step_shape = s3.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE,
                Inches(x_pos + 0.20),
                Inches(y_box),
                Inches(3.43),
                Inches(0.54),
            )
            step_shape.adjustments[0] = 0.08
            step_shape.fill.solid()
            step_shape.fill.fore_color.rgb = CARD_ALT
            step_shape.line.color.rgb = BORDER_GRAY
            tf_s = step_shape.text_frame
            tf_s.word_wrap = True
            ps = tf_s.paragraphs[0]
            ps.alignment = PP_ALIGN.CENTER
            ps.text = step_txt
            ps.font.name = "Calibri"
            ps.font.size = Pt(10.8)
            ps.font.bold = (s_idx in (2, 4))
            ps.font.color.rgb = DARK_TEXT

    for arrow_x in [4.46, 8.61]:
        arr = s3.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(arrow_x), Inches(3.15), Inches(0.26), Inches(0.42))
        arr.fill.solid()
        arr.fill.fore_color.rgb = FIRE_ORANGE
        arr.line.fill.background()

    # Deux bandeaux explicatifs en bas : Contrat + Court-circuit
    add_card(s3, 0.6, 5.75, 6.0, 1.15, bg_color=CARD_BG, border_color=PURPLE, accent_top_color=PURPLE)
    add_bullets_box(
        s3, 0.78, 5.88, 5.65, 0.95,
        None,
        [("Contrat Inter-Agents (contracts.py) :", "garantit formellement que les sorties de Salim et Moundir respectent les types, bornes et secteurs attendus par Asma.")],
        font_size=11.5,
    )

    add_card(s3, 6.75, 5.75, 5.98, 1.15, bg_color=GREEN_BG, border_color=GREEN_ALERT, accent_top_color=GREEN_ALERT)
    add_bullets_box(
        s3, 6.93, 5.88, 5.65, 0.95,
        None,
        [("Court-Circuit Temps Réel (main.py) :", "si Salim détecte fire_detected == 'Non', le pipeline renvoie immédiatement 🟢 VERT sans exécuter le Bi-LSTM.")],
        font_size=11.5,
    )

    add_speaker_notes(
        s3,
        "Comment avons-nous relié trois modèles codés par trois membres différents sans erreur d'intégration ? Nous avons mis en place un module contracts.py qui valide chaque échange de données entre Salim, Moundir et Asma, ainsi qu'un court-circuit intelligent dans main.py."
    )

    # --------------------------------------------------------------------------
    # SLIDE 4 : COMMENT NOUS AVONS CONSTRUIT & UNIFIÉ LES DONNÉES
    # --------------------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    add_header_and_footer(s4, "Comment Nous Avons Résolu le Défi des Données Hétérogènes", "3. Ingénierie des Données (Support Data)", 4, total_slides)

    add_kpi_badge(s4, 0.6, 1.22, 3.85, 1.12, "1 279 Images", "999 Kaggle FIRE + 280 Scènes Satellites", FIRE_ORANGE)
    add_kpi_badge(s4, 4.74, 1.22, 3.85, 1.12, "517 Relevés", "Dataset Kaggle Forest Fires (Météo Réelle)", TEAL)
    add_kpi_badge(s4, 8.88, 1.22, 3.85, 1.12, "1 Dataset Unifié", "global_fire_weather_dataset.csv (3 Agents)", PURPLE)

    add_card(s4, 0.6, 2.55, 5.9, 4.30, bg_color=CARD_BG, accent_top_color=FIRE_ORANGE)
    add_bullets_box(
        s4, 0.8, 2.72, 5.5, 4.0,
        "Défi 1 : Préparer les Images & Créer les Masques",
        [
            ("Problème initial :", "le dataset Kaggle FIRE (phylake1337) contient des photos brutes de tailles variées sans masques pixel-par-pixel."),
            ("Comment nous l'avons résolu :", "nettoyage automatique de 999 images Kaggle (755 Feu / 244 Non-Feu) en 128×128 + extraction spectrale des masques."),
            ("Complément satellite :", "génération de 280 scènes satellites forestières réalistes avec vérité terrain exacte (dataset/satellite_samples/)."),
        ],
        heading_color=FIRE_ORANGE,
        font_size=12.3,
    )

    add_card(s4, 6.8, 2.55, 5.93, 4.30, bg_color=CARD_BG, accent_top_color=TEAL)
    add_bullets_box(
        s4, 7.0, 2.72, 5.5, 4.0,
        "Défi 2 : Relier Images, Météo & Paliers d'Alerte",
        [
            ("Problème initial :", "les images satellites et le dataset météo Kaggle Forest Fires (517 relevés) provenaient de sources séparées."),
            ("Comment nous l'avons résolu (data_support.py) :", "création d'un pipeline de fusion couplant les prédictions spatiales de Salim aux relevés météo réels (temp, RH, vent) et aux vitesses Bi-LSTM de Moundir."),
            ("Résultat :", "un référentiel commun de 517 scénarios complets prêt à entraîner et valider l'Arbre de Décision d'Asma."),
        ],
        heading_color=TEAL,
        font_size=12.3,
    )

    add_speaker_notes(
        s4,
        "Un grand défi pratique était que nos sources Kaggle étaient séparées : d'un côté des images d'incendies sans masques, de l'autre un fichier CSV météo sans images. Voici comment nous avons nettoyé, généré les masques de segmentation et fusionné l'ensemble dans global_fire_weather_dataset.csv."
    )

    # --------------------------------------------------------------------------
    # SLIDE 5 : AGENT 1 (SALIM) — SON RÔLE & COMMENT PRÉTRAITER LES IMAGES
    # --------------------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    add_header_and_footer(s5, "Agent 1 (Salim) : Son Rôle & Comment Prétraiter les Images", "4. Agent 1 — Vision par Ordinateur (preprocessing.py)", 5, total_slides)

    add_mission_banner(
        s5,
        "Rôle de l'Agent 1 (Salim)",
        "Analyser toute image aérienne/satellite brute pour répondre « Y a-t-il un feu ? » et isoler précisément les pixels brûlés malgré la fumée et le bruit capteur.",
        FIRE_ORANGE,
    )

    steps_salim = [
        ("Obstacle 1 : Fichiers Corrompus ou Vides",
         "Solution : is_image_valid() vérifie l'intégrité matricielle, rejette les images noires/uniformes (σ < 2,0) et standardise la taille en 128×128 RGB [0, 1].",
         NAVY),
        ("Obstacle 2 : Voile Atmosphérique & Fumée",
         "Solution : Application d'une égalisation adaptative CLAHE (clipLimit=1.5) sur la luminance L* (espace L*a*b*) lors de la préparation pour rehausser les points chauds.",
         FIRE_ORANGE),
        ("Obstacle 3 : Absence de Masques Vérité Terrain",
         "Solution : Fusion multi-spectrale RGB + HSV + YCbCr (R > G > B, Y ≥ Cb, Cr > Cb) nettoyée par morphologie mathématique (fermeture & ouverture 5×5).",
         TEAL),
        ("Obstacle 4 : Sur-Apprentissage Visuel",
         "Solution : Data Augmentation synchronisée Image + Masque (flips H/V, rotations 90°/180°/270°, variations photométriques α, β et bruit gaussien).",
         PURPLE),
    ]

    for idx, (st_title, st_desc, col) in enumerate(steps_salim):
        col_i = idx % 2
        row_i = idx // 2
        x = 0.6 + col_i * 6.2
        y = 2.28 + row_i * 2.32
        add_card(s5, x, y, 5.93, 2.12, bg_color=CARD_BG, accent_top_color=col)
        add_bullets_box(
            s5, x + 0.18, y + 0.14, 5.55, 1.88,
            st_title,
            [("Comment nous l'avons résolu :", st_desc.replace("Solution : ", ""))],
            heading_color=col,
            font_size=12.2,
        )

    add_speaker_notes(
        s5,
        "Salim : Mon agent est la porte d'entrée du système. Avant même d'entraîner le réseau de neurones, j'ai dû résoudre 4 obstacles concrets dans preprocessing.py : filtrer les images invalides, corriger le voile atmosphérique avec CLAHE, extraire des masques spectraux fiables et augmenter les paires image-masque."
    )

    # --------------------------------------------------------------------------
    # SLIDE 6 : AGENT 1 (SALIM) — COMMENT NOUS AVONS CONÇU & CODÉ FIREUNET
    # --------------------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    add_header_and_footer(s6, "Agent 1 (Salim) : Comment FireUNet Détecte & Localise le Feu", "4. Agent 1 — Architecture & Localisation (model_cnn.py, predict.py)", 6, total_slides)

    add_card(s6, 0.6, 1.25, 6.15, 5.6, bg_color=CARD_BG, accent_top_color=FIRE_ORANGE)
    add_bullets_box(
        s6, 0.8, 1.42, 5.75, 5.3,
        "Pourquoi & Comment Nous Avons Conçu FireUNet",
        [
            ("Pourquoi pas un CNN simple ?", "Un classifieur classique dit seulement « Oui/Non » sans dire OÙ est le feu pour l'agent de Moundir."),
            ("Notre Architecture à Double Tête :", "un encodeur partagé (4 niveaux : 16 → 32 → 64 → 128 filtres, BatchNorm + ReLU + Dropout2d) alimente simultanément 2 têtes :"),
            ("1. Tête Classification (Bottleneck) :", "AdaptiveAvgPool2d + MLP → probabilité globale p_cls ∈ [0, 1] de présence de feu."),
            ("2. Tête Segmentation (Décodeur U-Net) :", "ConvTranspose2d + Skip Connections → carte pixel-par-pixel M ∈ [0, 1]^(128×128) de la zone touchée."),
        ],
        heading_color=FIRE_ORANGE,
        font_size=12.2,
    )

    add_card(s6, 7.0, 1.25, 5.73, 2.75, bg_color=CARD_BG, accent_top_color=NAVY)
    add_bullets_box(
        s6, 7.2, 1.42, 5.35, 2.45,
        "Comment Résoudre le Déséquilibre des Pixels Feu",
        [
            ("Défi :", "un départ de feu ne couvre parfois que 2 % à 10 % de l'image ; la BCE seule ignorerait ces petits foyers."),
            ("Solution (Perte Hybride) :", "L = L_BCE(cls) + 1,2 × [0,5 · L_BCE(seg) + 0,5 · L_Dice(M, M̂)] optimisée avec AdamW + CosineAnnealingLR."),
        ],
        heading_color=NAVY,
        font_size=12.0,
    )

    add_card(s6, 7.0, 4.18, 5.73, 2.67, bg_color=CARD_ALT, border_color=TEAL, accent_top_color=TEAL)
    add_bullets_box(
        s6, 7.2, 4.35, 5.35, 2.38,
        "Du Masque aux Coordonnées GPS (predict.py)",
        [
            ("Calcul du Barycentre :", "moyenne des coordonnées (x, y) pondérée par la probabilité thermique de chaque pixel actif."),
            ("Géolocalisation :", "conversion en secteur cardinal (Centre, Nord-Est…), projection GPS locale, % de surface brûlée et validation via contracts.py."),
            ("Livrables :", "models/fire_cnn_unet.pkl & models/fire_cnn_unet.h5."),
        ],
        heading_color=TEAL,
        font_size=11.8,
    )

    add_speaker_notes(
        s6,
        "Salim : Voici comment nous avons résolu le double besoin de détection et de localisation : l'architecture FireUNet partage un encodeur convolutif entre une tête de classification globale et un décodeur U-Net entraîné avec une perte hybride BCE + Dice, puis predict.py convertit le masque binaire en coordonnées GPS et secteur géographique."
    )

    # --------------------------------------------------------------------------
    # SLIDE 7 : AGENT 1 (SALIM) — RÉSULTATS EXPÉRIMENTAUX & PREUVE VISUELLE
    # --------------------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    add_header_and_footer(s7, "Agent 1 (Salim) : Résultats Obtenus & Segmentation en Action", "4. Agent 1 — Validation Expérimentale", 7, total_slides)

    add_kpi_badge(s7, 0.6, 1.20, 3.85, 1.05, "99,11 % / 98,2 %", "Pic Val Acc  |  Global (1 279 images)", FIRE_ORANGE)
    add_kpi_badge(s7, 4.74, 1.20, 3.85, 1.05, "84,2 % / 91,1 %", "Dice Global  |  Dice sur Feux Actifs", TEAL)
    add_kpi_badge(s7, 8.88, 1.20, 3.85, 1.05, "100,0 %", "Exactitude sur Scènes Satellites (280/280)", NAVY)

    add_card(s7, 0.6, 2.40, 7.15, 4.50, bg_color=CARD_BG)
    s7.shapes.add_picture("outputs/cnn_evaluation_plots.png", Inches(0.72), Inches(2.58), width=Inches(6.9))
    tb_cap1 = s7.shapes.add_textbox(Inches(0.75), Inches(4.78), Inches(6.85), Inches(1.98))
    tf_cap1 = tb_cap1.text_frame
    tf_cap1.word_wrap = True
    pc = tf_cap1.paragraphs[0]
    pc.text = "Ce que montrent nos tests sur les 1 279 images :"
    pc.font.name = "Calibri"
    pc.font.size = Pt(12.2)
    pc.font.bold = True
    pc.font.color.rgb = NAVY
    for line in [
        "• Zéro sur-apprentissage : Train Acc (98,19 %) ≈ Validation Acc (98,27 %).",
        "• Alignement exact Prétraitement/Inférence : 0 anomalie de confiance (< 50 %).",
        "• Rapidité temps réel : ~31 ms par image sur CPU grâce à la mise en cache mémoire.",
    ]:
        pl = tf_cap1.add_paragraph()
        pl.text = line
        pl.font.name = "Calibri"
        pl.font.size = Pt(11.3)
        pl.font.color.rgb = DARK_TEXT
        pl.space_before = Pt(3)

    add_card(s7, 7.95, 2.40, 4.78, 4.50, bg_color=CARD_BG)
    s7.shapes.add_picture("outputs/cnn_comparison_panel.png", Inches(8.07), Inches(2.60), width=Inches(4.54))
    tb_cap2 = s7.shapes.add_textbox(Inches(8.12), Inches(4.32), Inches(4.45), Inches(2.45))
    tf_cap2 = tb_cap2.text_frame
    tf_cap2.word_wrap = True
    p_out = tf_cap2.paragraphs[0]
    p_out.text = "Contrat JSON transmis à l'Agent 2 (Moundir) :"
    p_out.font.name = "Calibri"
    p_out.font.size = Pt(11.8)
    p_out.font.bold = True
    p_out.font.color.rgb = FIRE_ORANGE

    code_txt = (
        '{\n'
        '  "fire_detected": "Oui", "confidence": 1.0,\n'
        '  "location": {\n'
        '    "zone": "Centre", "center_xy": (66, 58),\n'
        '    "gps_coords": (36.7543, 5.0520),\n'
        '    "affected_area_pct": 10.27\n'
        '  }\n'
        '}'
    )
    p_code = tf_cap2.add_paragraph()
    p_code.text = code_txt
    p_code.font.name = "Consolas"
    p_code.font.size = Pt(9.8)
    p_code.font.color.rgb = DARK_TEXT
    p_code.space_before = Pt(4)

    add_speaker_notes(
        s7,
        "Salim : Les résultats confirment l'efficacité de FireUNet : 99,11 % au pic de validation, 98,20 % sur l'intégralité des 1 279 images (avec un écart Train/Val nul prouvant l'absence d'overfitting) et 100 % sur les scènes satellites. Le dictionnaire JSON ci-contre est immédiatement envoyé à Moundir."
    )

    # --------------------------------------------------------------------------
    # SLIDE 8 : AGENT 2 (MOUNDIR) — SON RÔLE & COMMENT MODÉLISER LA MÉTÉO
    # --------------------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    add_header_and_footer(s8, "Agent 2 (Moundir) : Son Rôle & Comment Modéliser la Physique du Feu", "5. Agent 2 — Ingénierie Temporelle & Météo (temporal_preprocessing.py)", 8, total_slides)

    add_mission_banner(
        s8,
        "Rôle de l'Agent 2 (Moundir)",
        "À partir de la position du feu trouvée par Salim et des relevés météo, prédire VERS OÙ et À QUELLE VITESSE le front de flamme va progresser sur +1h, +2h et +3h.",
        TEAL,
    )

    add_card(s8, 0.6, 2.25, 6.0, 4.60, bg_color=CARD_BG, accent_top_color=TEAL)
    add_bullets_box(
        s8, 0.8, 2.42, 5.6, 4.30,
        "Comment Nous Avons Construit les Séquences Temporelles",
        [
            ("1. Variables physiques (10 descripteurs) :", "température (°C), humidité RH (%), vitesse du vent (km/h), sin/cos du vent, indice canadien FWI/ISI, % surface, position (x, y) et vitesse passée."),
            ("2. Dynamique horaire (Rothermel) :", "couplage des distributions météo réelles de Kaggle (forestfires.csv) avec l'inertie thermique du feu sur des fenêtres glissantes de 4 heures."),
            ("3. Zéro Data Leakage :", "séparation stricte par épisodes d'incendie complets (160 épisodes Train / 40 épisodes Validation isolés)."),
        ],
        heading_color=TEAL,
        font_size=11.9,
    )

    add_card(s8, 6.85, 2.25, 5.88, 4.60, bg_color=CARD_BG, accent_top_color=NAVY)
    add_bullets_box(
        s8, 7.05, 2.42, 5.48, 2.65,
        "Comment Nous Avons Résolu le Piège 359° ↔ 1°",
        [
            ("Le problème angulaire :", "pour un réseau classique, 359° et 1° ont un écart énorme de 358°, alors qu'ils désignent tous deux le Nord (2° d'écart réel) !"),
            ("Notre solution géométrique :", "projection cyclique sur le cercle unité x_sin = sin(2πθ/360), x_cos = cos(2πθ/360), puis restitution exacte via atan2(ŷ_sin, ŷ_cos) mod 360°."),
        ],
        heading_color=NAVY,
        font_size=11.9,
    )

    add_card(s8, 7.08, 5.15, 5.42, 1.50, bg_color=CARD_ALT, border_color=FIRE_ORANGE)
    add_bullets_box(
        s8, 7.24, 5.25, 5.1, 1.30,
        "Fonction de Perte Multi-Objectifs sur Mesure",
        [
            ("L_total =", "1,0·SmoothL1(vitesse) + 1,5·(1 − cos Δθ) + 0,8·MSE(+1h..+3h)"),
        ],
        heading_color=FIRE_ORANGE,
        font_size=11.6,
    )

    add_speaker_notes(
        s8,
        "Moundir : Une fois le feu localisé par Salim, mon agent doit prédire sa dynamique future. Deux choix techniques ont été décisifs : d'abord projeter les directions de vent sur le cercle trigonométrique (sin, cos) pour éliminer la discontinuité 0°/360°, ensuite découper les données par épisodes complets sans aucune fuite entre Train et Validation."
    )

    # --------------------------------------------------------------------------
    # SLIDE 9 : AGENT 2 (MOUNDIR) — ARCHITECTURE BI-LSTM & RÉSULTATS
    # --------------------------------------------------------------------------
    s9 = prs.slides.add_slide(blank_layout)
    add_header_and_footer(s9, "Agent 2 (Moundir) : Architecture FirePropagationLSTM & Résultats", "5. Agent 2 — Modèle Récurrent & Validation (model_lstm.py)", 9, total_slides)

    add_kpi_badge(s9, 0.6, 1.20, 3.85, 1.05, "0,31 km/h", "MAE Vitesse (R² = 0,963 sans fuite)", TEAL)
    add_kpi_badge(s9, 4.74, 1.20, 3.85, 1.05, "6,27°  (90 %)", "Erreur Angulaire | Exactitude 8 Secteurs", FIRE_ORANGE)
    add_kpi_badge(s9, 8.88, 1.20, 3.85, 1.05, "< 3 ms", "Temps d'Inférence Temps Réel", NAVY)

    add_card(s9, 0.6, 2.42, 4.9, 4.48, bg_color=CARD_BG, accent_top_color=TEAL)
    add_bullets_box(
        s9, 0.78, 2.58, 4.55, 4.20,
        "Comment Fonctionne FirePropagationLSTM",
        [
            ("1. Projection & LayerNorm :", "projette les 10 variables météo dans un espace latent de dimension 64."),
            ("2. Bi-LSTM Empilé (2 couches, 128 unités) :", "modélise à la fois l'inertie thermique accumulée et les accélérations soudaines de vent."),
            ("3. Multi-Head Attention Pooling :", "focalise automatiquement le réseau sur l'heure où les rafales et la sécheresse ont culminé."),
            ("4. Trois Têtes Découplées :", "• Vitesse instantanée (km/h)\n• Vecteur directionnel unitaire L2\n• Expansion future à +1h, +2h et +3h."),
        ],
        heading_color=TEAL,
        font_size=11.6,
    )

    add_card(s9, 5.7, 2.42, 7.03, 4.48, bg_color=CARD_BG)
    s9.shapes.add_picture("outputs/lstm_evaluation_plots.png", Inches(5.82), Inches(2.62), width=Inches(6.78))
    tb_l = s9.shapes.add_textbox(Inches(5.88), Inches(5.08), Inches(6.65), Inches(1.68))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True
    pl1 = tf_l.paragraphs[0]
    pl1.text = "Sortie transmise à l'Agent 3 (Asma) via predict_propagation(...) :"
    pl1.font.name = "Calibri"
    pl1.font.size = Pt(11.8)
    pl1.font.bold = True
    pl1.font.color.rgb = NAVY

    pl2 = tf_l.add_paragraph()
    pl2.text = (
        "Entrée : Zone='Centre' (10,27 %), Vent=38 km/h (Sud-Ouest), RH=22 %, T=34 °C\n"
        "→ Direction = Sud-Ouest (226,5°)  |  Vitesse = 9,30 km/h  |  FWI = 3,15\n"
        "→ Expansion horaire prévue : +1h = 9,69 km/h | +2h = 10,06 km/h | +3h = 10,76 km/h"
    )
    pl2.font.name = "Consolas"
    pl2.font.size = Pt(9.8)
    pl2.font.color.rgb = DARK_TEXT
    pl2.space_before = Pt(4)

    add_speaker_notes(
        s9,
        "Moundir : Grâce au couplage Bi-LSTM + Attention Temporelle, notre modèle atteint une MAE de 0,31 km/h (R² = 0,963 sur 40 épisodes jamais vus) et une erreur angulaire moyenne de 6,27°. Cette vitesse prédite est immédiatement transmise à l'Arbre de Décision d'Asma."
    )

    # --------------------------------------------------------------------------
    # SLIDE 10 : AGENT 3 (ASMA) — SON RÔLE & COMMENT DÉCIDER EXPLICABLEMENT
    # --------------------------------------------------------------------------
    s10 = prs.slides.add_slide(blank_layout)
    add_header_and_footer(s10, "Agent 3 (Asma) : Son Rôle & Comment Décider de Façon Explicable", "6. Agent 3 — Système d'Alerte par Arbre de Décision (train_tree.py)", 10, total_slides)

    add_mission_banner(
        s10,
        "Rôle de l'Agent 3 (Asma)",
        "Fusionner la détection visuelle de Salim, la vitesse de propagation de Moundir et la météo dans un Arbre CART 100 % transparent pour émettre le niveau d'alerte officiel.",
        NAVY,
    )

    # Explication de comment l'arbre a été entraîné
    add_card(s10, 0.6, 2.22, 12.13, 0.96, bg_color=CARD_ALT, border_color=NAVY)
    add_bullets_box(
        s10, 0.8, 2.30, 11.75, 0.82,
        None,
        [
            ("Comment nous avons entraîné l'Arbre (mon_arbre_alerte.pkl) :",
             "entraîné sur 1 277 scénarios (517 cas réels unifiés + grille physique complète) avec les 4 entrées [feu, vent, humidite, vitesse_propagation], garantissant la monotonie physique (feu=0 → toujours Vert)."),
        ],
        font_size=11.7,
    )

    tiers = [
        ("🟢 PALIER 0 : VERT", GREEN_ALERT, GREEN_BG,
         "Aucun feu détecté par Salim\n(feu == 0, quel que soit le vent)",
         "Surveillance normale des patrouilles forestières et veille satellite de routine."),
        ("🟠 PALIER 1 : ORANGE", ORANGE_ALERT, ORANGE_BG,
         "Feu détecté + Vent modéré (< 20 km/h)\nHumidité élevée (> 40 %) & Vitesse lente",
         "Mobilisation préventive des vigies locales et surveillance renforcée sans alarme publique."),
        ("🔴 PALIER 2 : ROUGE", RED_ALERT, RED_BG,
         "Feu actif + Vent moyen/fort (≥ 20 km/h)\nou Air sec (RH ≤ 40 %) / Vitesse ≥ 2,8 km/h",
         "Pré-positionnement des bombardiers d'eau et interdiction d'accès aux massifs."),
        ("🚨 PALIER 3 : ÉVACUATION", EVAC_ALERT, EVAC_BG,
         "Feu actif + Vent violent (> 40 km/h)\nou Vitesse critique (> 6,9 km/h) & Air aride",
         "Déclenchement immédiat de la sirène, fermeture des axes routiers et évacuation d'urgence."),
    ]

    for idx, (t_name, border_c, bg_c, cond_txt, action_txt) in enumerate(tiers):
        x = 0.6 + idx * 3.08
        add_card(s10, x, 3.32, 2.9, 3.55, bg_color=bg_c, border_color=border_c, accent_top_color=border_c)

        tb_t = s10.shapes.add_textbox(Inches(x + 0.13), Inches(3.48), Inches(2.64), Inches(3.25))
        tf_t = tb_t.text_frame
        tf_t.word_wrap = True

        pt1 = tf_t.paragraphs[0]
        pt1.text = t_name
        pt1.font.name = "Calibri"
        pt1.font.size = Pt(14)
        pt1.font.bold = True
        pt1.font.color.rgb = border_c

        pt2 = tf_t.add_paragraph()
        pt2.text = "Règle apprise par l'Arbre :"
        pt2.font.name = "Calibri"
        pt2.font.size = Pt(11)
        pt2.font.bold = True
        pt2.font.color.rgb = DARK_TEXT
        pt2.space_before = Pt(7)

        pt3 = tf_t.add_paragraph()
        pt3.text = cond_txt
        pt3.font.name = "Calibri"
        pt3.font.size = Pt(10.8)
        pt3.font.color.rgb = DARK_TEXT
        pt3.space_before = Pt(2)

        pt4 = tf_t.add_paragraph()
        pt4.text = "Recommandation secours :"
        pt4.font.name = "Calibri"
        pt4.font.size = Pt(11)
        pt4.font.bold = True
        pt4.font.color.rgb = DARK_TEXT
        pt4.space_before = Pt(9)

        pt5 = tf_t.add_paragraph()
        pt5.text = action_txt
        pt5.font.name = "Calibri"
        pt5.font.size = Pt(10.8)
        pt5.font.color.rgb = DARK_TEXT
        pt5.space_before = Pt(2)

    add_speaker_notes(
        s10,
        "Asma : Pourquoi avoir choisi un Arbre de Décision CART plutôt qu'un réseau de neurones boîte noire pour le 3ème agent ? Parce qu'en gestion de crise, les autorités doivent pouvoir lire et auditer chaque seuil qui déclenche une évacuation. Voici les 4 paliers officiels appris par notre arbre."
    )

    # --------------------------------------------------------------------------
    # SLIDE 11 : AGENT 3 (ASMA) — PERFORMANCES CART, MAIN.PY & APP STREAMLIT
    # --------------------------------------------------------------------------
    s11 = prs.slides.add_slide(blank_layout)
    add_header_and_footer(s11, "Agent 3 (Asma) : Intégration (main.py), Validation CART & Streamlit", "6. Agent 3 — Orchestration & Interface Web (app.py)", 11, total_slides)

    # Gauche : Graphique de validation de l'Arbre + Pipeline main.py
    add_card(s11, 0.6, 1.22, 6.15, 5.65, bg_color=CARD_BG, accent_top_color=NAVY)
    s11.shapes.add_picture("outputs/decision_tree_visual.png", Inches(0.75), Inches(1.42), width=Inches(5.85))

    add_bullets_box(
        s11, 0.78, 3.92, 5.8, 2.85,
        "Comment main.py Orchestre les 3 Agents (98,98 % CV Acc)",
        [
            ("1. Validation Contrat CNN :", "appelle predict(image_path) ; arrêt immédiat 🟢 VERT si 'Non'."),
            ("2. Validation Contrat LSTM :", "passe res_cnn['location'] + météo (8 directions) à predict_propagation()."),
            ("3. Verdict Décisionnel :", "injecte [feu, vent, humidite, vitesse_lstm] dans donner_alerte() → 98,45 % d'exactitude sur les 517 cas réels (0 warning)."),
        ],
        heading_color=NAVY,
        font_size=11.5,
    )

    # Droite : Tableau de bord Streamlit enrichi
    add_card(s11, 6.95, 1.22, 5.78, 5.65, bg_color=CARD_BG, accent_top_color=FIRE_ORANGE)
    tb_st = s11.shapes.add_textbox(Inches(7.15), Inches(1.38), Inches(5.4), Inches(0.72))
    tf_st = tb_st.text_frame
    tf_st.word_wrap = True
    pst1 = tf_st.paragraphs[0]
    pst1.text = "Tableau de Bord Web — streamlit run app.py"
    pst1.font.name = "Calibri"
    pst1.font.size = Pt(15)
    pst1.font.bold = True
    pst1.font.color.rgb = FIRE_ORANGE
    pst2 = tf_st.add_paragraph()
    pst2.text = "Interface unifiée affichant en direct les résultats des 3 agents :"
    pst2.font.name = "Calibri"
    pst2.font.size = Pt(11.5)
    pst2.font.color.rgb = MUTED_TEXT

    add_card(s11, 7.15, 2.18, 5.38, 4.48, bg_color=CARD_ALT, border_color=BORDER_GRAY)

    # Vignette CNN + Météo dans la maquette Streamlit
    add_card(s11, 7.32, 2.34, 2.45, 2.10, bg_color=WHITE, border_color=BORDER_GRAY)
    s11.shapes.add_picture("outputs/demo_kaggle_fire_segmented.png", Inches(7.92), Inches(2.72), width=Inches(1.25))
    tb_u1 = s11.shapes.add_textbox(Inches(7.40), Inches(2.38), Inches(2.3), Inches(0.32))
    p_u1 = tb_u1.text_frame.paragraphs[0]
    p_u1.text = "🛰️ Carte U-Net (Salim)"
    p_u1.font.name = "Calibri"
    p_u1.font.size = Pt(10)
    p_u1.font.bold = True
    p_u1.font.color.rgb = NAVY

    add_card(s11, 9.92, 2.34, 2.45, 2.10, bg_color=WHITE, border_color=BORDER_GRAY)
    tb_u2 = s11.shapes.add_textbox(Inches(10.00), Inches(2.38), Inches(2.3), Inches(2.0))
    tf_u2 = tb_u2.text_frame
    tf_u2.word_wrap = True
    pu2 = tf_u2.paragraphs[0]
    pu2.text = "🌪️ Bi-LSTM (Moundir)"
    pu2.font.name = "Calibri"
    pu2.font.size = Pt(10)
    pu2.font.bold = True
    pu2.font.color.rgb = TEAL
    for sl_txt in [
        "• Dir : Sud-Ouest (226,5°)",
        "• Vitesse : 9,30 km/h",
        "• +1h/+3h : 9,69 → 10,76",
        "• Indice FWI : 3,15",
    ]:
        psl = tf_u2.add_paragraph()
        psl.text = sl_txt
        psl.font.name = "Calibri"
        psl.font.size = Pt(9.8)
        psl.font.color.rgb = DARK_TEXT
        psl.space_before = Pt(4)

    btn = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.32), Inches(4.58), Inches(5.05), Inches(0.45))
    btn.adjustments[0] = 0.15
    btn.fill.solid()
    btn.fill.fore_color.rgb = FIRE_ORANGE
    btn.line.fill.background()
    p_btn = btn.text_frame.paragraphs[0]
    p_btn.alignment = PP_ALIGN.CENTER
    p_btn.text = "🚀 LANCER L'ANALYSE MULTI-AGENTS"
    p_btn.font.name = "Calibri"
    p_btn.font.size = Pt(10.8)
    p_btn.font.bold = True
    p_btn.font.color.rgb = WHITE

    banner = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.32), Inches(5.16), Inches(5.05), Inches(1.32))
    banner.adjustments[0] = 0.08
    banner.fill.solid()
    banner.fill.fore_color.rgb = EVAC_BG
    banner.line.color.rgb = EVAC_ALERT
    p_ban = banner.text_frame.paragraphs[0]
    p_ban.alignment = PP_ALIGN.CENTER
    p_ban.text = (
        "PALIER D'ALERTE : 🚨 ÉVACUATION\n"
        "Vecteur fusionné : [Feu=1, Vent=38 km/h, RH=22%, Vitesse=9,30 km/h]\n"
        "Action : Sirène civile, fermeture routière & évacuation d'urgence"
    )
    p_ban.font.name = "Calibri"
    p_ban.font.size = Pt(10.2)
    p_ban.font.bold = True
    p_ban.font.color.rgb = EVAC_ALERT

    add_speaker_notes(
        s11,
        "Asma : À gauche, on voit l'importance des 4 variables dans notre Arbre de Décision ainsi que sa matrice de confusion à 98,45 % sur les 517 cas réels. À droite, notre tableau de bord Streamlit réunit visuellement la carte segmentée de Salim, les projections horaires de Moundir et le bandeau d'alerte opérationnel."
    )

    # --------------------------------------------------------------------------
    # SLIDE 12 : EXEMPLE CONCRET DE BOUT EN BOUT (PAS À PAS)
    # --------------------------------------------------------------------------
    s12 = prs.slides.add_slide(blank_layout)
    add_header_and_footer(s12, "Exemple Concret : Comment Notre Solution Traite une Crise de A à Z", "7. Démonstration Pas-à-Pas du Pipeline", 12, total_slides)

    walkthrough_steps = [
        ("ÉTAPE 1 : Acquisition Terrain", PURPLE,
         "Données reçues par le système :\n\n"
         "• Image : sat_fire_0001.png\n"
         "• Vent : 38,0 km/h (Sud-Ouest)\n"
         "• Humidité : 22,0 % (air très sec)\n"
         "• Température : 34,0 °C\n\n"
         "➔ Lancement de run_pipeline(...)"),
        ("ÉTAPE 2 : Agent 1 (Salim)", FIRE_ORANGE,
         "Analyse spatiale FireUNet :\n\n"
         "• Feu détecté : OUI (Confiance 100 %)\n"
         "• Secteur : Centre\n"
         "• Barycentre : (66, 58) px\n"
         "• GPS : (36.7543, 5.0520)\n"
         "• Surface brûlée : 10,27 %\n\n"
         "➔ Contrat CNN validé ✓"),
        ("ÉTAPE 3 : Agent 2 (Moundir)", TEAL,
         "Prévision Bi-LSTM + Attention :\n\n"
         "• Direction : Sud-Ouest (226,5°)\n"
         "• Vitesse actuelle : 9,30 km/h\n"
         "• +1h : 9,69 km/h | +2h : 10,06 km/h\n"
         "• +3h : 10,76 km/h\n"
         "• Indice météo FWI : 3,15\n\n"
         "➔ Contrat LSTM validé ✓"),
        ("ÉTAPE 4 : Agent 3 (Asma)", EVAC_ALERT,
         "Décision CART & Action :\n\n"
         "• Entrées : [1, 38.0, 22.0, 9.30]\n"
         "• Règle activée : Feu=1 & Vitesse > 6,94 km/h\n"
         "• Verdict : 🚨 ÉVACUATION\n\n"
         "📋 Ordre émis sur Streamlit :\n"
         "Sirène, fermeture des axes routiers et évacuation immédiate."),
    ]

    for idx, (st_head, col_w, body_w) in enumerate(walkthrough_steps):
        x = 0.6 + idx * 3.08
        add_card(s12, x, 1.32, 2.90, 4.35, bg_color=CARD_BG, border_color=col_w, accent_top_color=col_w)
        tb_w = s12.shapes.add_textbox(Inches(x + 0.14), Inches(1.50), Inches(2.62), Inches(4.0))
        tf_w = tb_w.text_frame
        tf_w.word_wrap = True

        pw1 = tf_w.paragraphs[0]
        pw1.text = st_head
        pw1.font.name = "Calibri"
        pw1.font.size = Pt(13.5)
        pw1.font.bold = True
        pw1.font.color.rgb = col_w

        pw2 = tf_w.add_paragraph()
        pw2.text = body_w
        pw2.font.name = "Calibri"
        pw2.font.size = Pt(11.0)
        pw2.font.color.rgb = DARK_TEXT
        pw2.space_before = Pt(8)
        pw2.line_spacing = 1.15

        if idx < 3:
            arr = s12.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + 2.93), Inches(3.25), Inches(0.13), Inches(0.32))
            arr.fill.solid()
            arr.fill.fore_color.rgb = FIRE_ORANGE
            arr.line.fill.background()

    # Terminal récapitulatif en bas
    term = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(5.85), Inches(12.13), Inches(1.02))
    term.adjustments[0] = 0.05
    term.fill.solid()
    term.fill.fore_color.rgb = RGBColor(15, 23, 42)
    term.line.color.rgb = RGBColor(51, 65, 85)
    tf_term = term.text_frame
    tf_term.word_wrap = True
    tf_term.margin_left = Inches(0.20)
    tf_term.margin_top = Inches(0.12)
    p_t = tf_term.paragraphs[0]
    p_t.text = (
        "$ python main.py   ➔   1. Salim (CNN): Feu=Oui (100%, Centre, 10.27%)  |  "
        "2. Moundir (Bi-LSTM): Sud-Ouest (226.5°), 9.30 km/h  |  "
        "3. Asma (CART): 🚨 ÉVACUATION  (Temps total < 40 ms)"
    )
    p_t.font.name = "Consolas"
    p_t.font.size = Pt(10.8)
    p_t.font.bold = True
    p_t.font.color.rgb = RGBColor(253, 224, 71)

    add_speaker_notes(
        s12,
        "Pour illustrer concrètement comment nos 3 agents coopèrent, voici la trace réelle d'exécution sur sat_fire_0001.png en moins de 40 millisecondes : acquisition météo et image, segmentation GPS par Salim, projection de propagation à 9,30 km/h vers le Sud-Ouest par Moundir, et déclenchement de l'ordre d'évacuation par Asma."
    )

    # --------------------------------------------------------------------------
    # SLIDE 13 : BILAN COMPARATIF & ANALYSE MÉTHODOLOGIQUE (POURQUOI ~98-99 % ?)
    # --------------------------------------------------------------------------
    s13 = prs.slides.add_slide(blank_layout)
    add_header_and_footer(s13, "Synthèse des 3 Agents & Analyse Critique des Performances", "8. Bilan Global & Rigueur Scientifique", 13, total_slides)

    rows_data = [
        ["Critère", "Agent 1 : Salim Chablaoui", "Agent 2 : Moundir Boucheriha", "Agent 3 : Asma Dadou"],
        ["Problème Résolu", "Détecter & localiser le feu sur image", "Prédire direction, vitesse & expansion", "Décider le palier d'urgence & intégrer"],
        ["Comment (Méthode)", "FireUNet (CNN + U-Net) + BCE/Dice Loss", "Bi-LSTM + Attention + Cercle (sin, cos)", "Arbre CART 4 entrées + Contrat + Streamlit"],
        ["Entrées → Sorties", "Image 128×128 → Oui/Non, GPS, % Brûlé", "Zone + Météo (4h) → km/h, Angle, +1..3h", "[Feu, Vent, RH, Vitesse] → 4 Paliers"],
        ["Résultats Vérifiés", "98,20 % Global (Pic Val 99,11 %) | Dice 84,2 %", "MAE : 0,31 km/h (R²=0,963) | Angle : 6,27°", "CV 5x : 98,98 % | Global CSV : 98,45 %"],
        ["Livrables Clés", "preprocessing, model_cnn, predict (.pkl/.h5)", "temporal_prep, model_lstm, predict_prop", "contracts, train_tree, main.py, app.py"],
    ]

    table_shape = s13.shapes.add_table(len(rows_data), 4, Inches(0.6), Inches(1.22), Inches(12.13), Inches(4.15))
    table = table_shape.table
    col_widths = [Inches(2.03), Inches(3.38), Inches(3.42), Inches(3.30)]
    for idx_w, w_val in enumerate(col_widths):
        table.columns[idx_w].width = w_val

    for r_idx, row in enumerate(rows_data):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if r_idx == 0:
                cell.fill.fore_color.rgb = NAVY
            elif r_idx == 4:
                cell.fill.fore_color.rgb = RGBColor(254, 243, 199)
            elif r_idx % 2 == 1:
                cell.fill.fore_color.rgb = WHITE
            else:
                cell.fill.fore_color.rgb = CARD_ALT

            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = "Calibri"
            p.font.size = Pt(11.0 if r_idx > 0 else 12.0)
            p.font.bold = (r_idx == 0 or c_idx == 0 or r_idx == 4)
            if r_idx == 0:
                p.font.color.rgb = WHITE
            elif c_idx == 0:
                p.font.color.rgb = NAVY
            elif r_idx == 4:
                p.font.color.rgb = FIRE_ORANGE
            else:
                p.font.color.rgb = DARK_TEXT

    # Encadré méthodologique en bas : Pourquoi pas de sur-apprentissage & limites honnêtes
    add_card(s13, 0.6, 5.52, 12.13, 1.38, bg_color=CARD_BG, border_color=TEAL, accent_top_color=TEAL)
    add_bullets_box(
        s13, 0.8, 5.64, 11.75, 1.20,
        "Analyse Critique — Pourquoi nos scores atteignent ~98 % sans Overfitting ?",
        [
            ("Absence de sur-apprentissage vérifiée :", "Train ≈ Validation sur les 3 modules (CNN : 98,19 % vs 98,27 % ; Bi-LSTM : épisodes séparés ; CART : CV 98,98 %)."),
            ("Explication scientifique & Limites :", "fort contraste chromatique flamme/forêt en RGB (sensible aux couchers de soleil sans canal infrarouge SWIR), simulation physique régulière de Rothermel, et règles opérationnelles déterministes (87,2 % sous fort bruit capteur météo)."),
        ],
        heading_color=TEAL,
        font_size=11.0,
    )

    add_speaker_notes(
        s13,
        "Ce tableau récapitule comment chaque agent résout sa part du problème. Nous ajoutons également un recul scientifique important : nos tests prouvent qu'il n'y a pas d'overfitting (Train = Validation), et expliquent honnêtement les limites actuelles en conditions réelles bruitées."
    )

    # --------------------------------------------------------------------------
    # SLIDE 14 : PERSPECTIVES D'ÉVOLUTION & CONCLUSION
    # --------------------------------------------------------------------------
    s14 = prs.slides.add_slide(blank_layout)
    add_header_and_footer(s14, "Perspectives d'Amélioration Terrain & Conclusion", "9. Conclusion & Ouverture", 14, total_slides)

    perspectives = [
        ("1. Capteurs Infrarouges (SWIR / NIR Sentinel-2)",
         "Coupler le RGB aux bandes thermiques Sentinel-2/Landsat-8 pour éliminer les faux positifs orangés (couchers de soleil, feuillages d'automne) et percer la fumée épaisse.",
         FIRE_ORANGE),
        ("2. Topographie & Modèle Numérique de Terrain",
         "Intégrer la pente et l'exposition des versants dans le Bi-LSTM de Moundir, car une pente montante accélère exponentiellement la convection des flammes.",
         TEAL),
        ("3. Drones Thermiques Autonomes (Edge AI)",
         "Déployer FireUNet quantifié sur NVIDIA Jetson à bord de drones forestiers pour confirmer localement les alertes Orange et Rouge sous canopée.",
         NAVY),
        ("4. Micro-Services Cloud & Connexion CODIS",
         "Conteneuriser l'architecture (Docker / API REST autour de contracts.py) avec envoi automatique d'alertes SMS/SIG vers les centres de secours.",
         PURPLE),
    ]

    for idx, (p_title, p_desc, col) in enumerate(perspectives):
        x = 0.6 + (idx % 2) * 6.2
        y = 1.30 + (idx // 2) * 2.05
        add_card(s14, x, y, 5.93, 1.84, bg_color=CARD_BG, accent_top_color=col)
        add_bullets_box(s14, x + 0.18, y + 0.14, 5.55, 1.56, p_title, [("Comment aller plus loin :", p_desc)], heading_color=col, font_size=12.0)

    closing = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(5.52), Inches(12.13), Inches(1.32))
    closing.adjustments[0] = 0.06
    closing.fill.solid()
    closing.fill.fore_color.rgb = NAVY
    closing.line.color.rgb = FIRE_ORANGE
    closing.line.width = Pt(1.5)

    tf_cl = closing.text_frame
    tf_cl.word_wrap = True
    pcl1 = tf_cl.paragraphs[0]
    pcl1.alignment = PP_ALIGN.CENTER
    pcl1.text = "Merci de votre attention !  —  Place à la démonstration live Streamlit & vos questions"
    pcl1.font.name = "Calibri"
    pcl1.font.size = Pt(19.5)
    pcl1.font.bold = True
    pcl1.font.color.rgb = WHITE

    pcl2 = tf_cl.add_paragraph()
    pcl2.alignment = PP_ALIGN.CENTER
    pcl2.text = "Salim Chablaoui  •  Moundir Boucheriha  •  Asma Dadou   |   Projet 5 — Data Mining & Deep Learning"
    pcl2.font.name = "Calibri"
    pcl2.font.size = Pt(12.5)
    pcl2.font.color.rgb = RGBColor(253, 186, 116)
    pcl2.space_before = Pt(4)

    add_speaker_notes(
        s14,
        "En conclusion, nous avons résolu une problématique critique en faisant coopérer Vision profonde (U-Net), modélisation physique séquentielle (Bi-LSTM) et décision explicable (Arbre CART) autour d'un contrat logiciel robuste. Merci de votre attention, nous passons maintenant à la démonstration live sur Streamlit !"
    )

    out_pptx = Path("Presentation_Projet5_FireDetection.pptx")
    prs.save(str(out_pptx))
    return out_pptx.resolve()


if __name__ == "__main__":
    print("1. Génération des visuels haute définition (CNN, Bi-LSTM, Arbre de Décision)...")
    generate_visual_assets()
    print("2. Création du fichier PowerPoint (.pptx)...")
    pptx_path = build_presentation()
    print(f"Présentation générée avec succès : {pptx_path}")
