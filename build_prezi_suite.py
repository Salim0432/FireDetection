"""
Générateur Prezi-Style Complet — Projet 5 (FireDetection)
=========================================================
Produit deux livrables complémentaires au style Prezi :
1. `Presentation_Prezi_Style_FireDetection.pptx` :
   Présentation PowerPoint (15 vues) utilisant un canevas spatial orbital, des
   nœuds circulaires interconnectés et la transition native **PowerPoint Morph**
   (`!!ObjectName` XML matching) qui reproduit l'effet de caméra Prezi (Vue Globale
   -> Zoom dans un Nœud -> Sous-nœuds -> Dézoom vers la Carte).
2. `Presentation_Prezi_Interactive.html` :
   Véritable canevas spatial Prezi 2D interactif 100 % hors-ligne (images encodées
   en Base64, transitions caméra fluides 60 fps zoom/pan/rotation, navigation au
   clavier ← / → / Espace, touche 'O' pour Vue Globale, touche 'N' pour Notes).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


# Palette Aerospace / Observation de la Terre (Haute lisibilité sur projecteur)
BG_SPACE = RGBColor(11, 19, 32)       # #0B1320 - Fond orbital profond
BG_PANEL = RGBColor(20, 31, 51)       # #141F33 - Surface de hub / carte sombre
BG_NODE = RGBColor(28, 43, 69)        # #1C2B45 - Intérieur de nœud circulaire
RING_MUTED = RGBColor(51, 72, 104)    # #334868 - Anneaux d'orbite
TEXT_WHITE = RGBColor(248, 250, 252)  # #F8FAFC - Blanc pur lisible
TEXT_SOFT = RGBColor(186, 200, 220)   # #BAC8DC - Texte secondaire clair
TEXT_DIM = RGBColor(118, 138, 168)    # #768AA8 - Légendes discrètes

# Accents thématiques des 3 hubs agents
EMBER = RGBColor(239, 91, 42)         # #EF5B2A - Hub 1 : Salim (Vision Feu)
EMBER_GLOW = RGBColor(253, 164, 114)  # #FDA472
CYAN_TEAL = RGBColor(20, 184, 166)    # #14B8A6 - Hub 2 : Moundir (Météo / LSTM)
CYAN_GLOW = RGBColor(110, 231, 216)   # #6EE7D8
AMBER_GOLD = RGBColor(245, 158, 11)   # #F59E0B - Hub 3 : Asma (Décision / Alerte)
AMBER_GLOW = RGBColor(252, 211, 77)   # #FCD34D

# Paliers d'alerte
C_GREEN = RGBColor(34, 197, 94)
C_ORANGE = RGBColor(249, 115, 22)
C_RED = RGBColor(239, 68, 68)
C_EVAC = RGBColor(225, 29, 72)


# ==============================================================================
# 1. Génération des graphiques sombres intégrés au canevas Prezi (sans fond blanc)
# ==============================================================================
def build_dark_prezi_charts() -> dict[str, Path]:
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "training_metrics.json", "r", encoding="utf-8") as f:
        cnn_m = json.load(f)
    with open(out / "lstm_training_metrics.json", "r", encoding="utf-8") as f:
        lstm_m = json.load(f)["history"]

    plt.rcParams.update({
        "font.sans-serif": "DejaVu Sans",
        "text.color": "#F8FAFC",
        "axes.labelcolor": "#BAC8DC",
        "xtick.color": "#94A3B8",
        "ytick.color": "#94A3B8",
        "axes.edgecolor": "#334868",
        "axes.facecolor": "#141F33",
        "figure.facecolor": "#141F33",
    })

    # A. Graphique CNN (prezi_cnn_charts.png)
    ep_c = np.arange(1, len(cnn_m["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), dpi=220)
    fig.patch.set_facecolor("#141F33")

    axes[0].plot(ep_c, [v * 100 for v in cnn_m["train_acc"]], color="#94A3B8", ls="--", lw=1.8, marker="o", markersize=4, label="Train Acc")
    axes[0].plot(ep_c, [v * 100 for v in cnn_m["val_acc"]], color="#EF5B2A", lw=2.8, marker="o", markersize=5, label="Val Acc (Pic 99,11 %)")
    axes[0].axhline(99.11, color="#F59E0B", ls=":", lw=1.5)
    axes[0].set_title("Classification Feu Oui/Non (%)", fontsize=11, fontweight="bold", color="#F8FAFC", pad=8)
    axes[0].set_xlabel("Époques")
    axes[0].set_ylim(68, 101)
    axes[0].grid(True, color="#263854", alpha=0.6)
    axes[0].legend(facecolor="#0B1320", edgecolor="#334868", fontsize=8.5, loc="lower right")

    axes[1].plot(ep_c, [v * 100 for v in cnn_m["val_dice"]], color="#14B8A6", lw=2.8, marker="s", markersize=5, label="Dice (84,2 %)")
    axes[1].plot(ep_c, [v * 100 for v in cnn_m["val_iou"]], color="#60A5FA", lw=2.2, marker="^", markersize=5, label="IoU (79,5 %)")
    axes[1].set_title("Segmentation Spatiale U-Net (%)", fontsize=11, fontweight="bold", color="#F8FAFC", pad=8)
    axes[1].set_xlabel("Époques")
    axes[1].set_ylim(58, 90)
    axes[1].grid(True, color="#263854", alpha=0.6)
    axes[1].legend(facecolor="#0B1320", edgecolor="#334868", fontsize=8.5, loc="lower right")

    plt.tight_layout()
    cnn_chart_p = out / "prezi_cnn_charts.png"
    fig.savefig(cnn_chart_p, dpi=220, facecolor="#141F33", bbox_inches="tight")
    plt.close(fig)

    # B. Panneau Segmentation Visuelle (prezi_cnn_seg.png)
    from predict import predict

    sample_img = Path("dataset/kaggle_cleaned/images/fire/kaggle_fire_0002.png")
    if not sample_img.exists():
        sample_img = next(Path("dataset/satellite_samples/images/fire").glob("*.png"))

    res = predict(sample_img)
    bgr = cv2.imread(str(sample_img))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask = res["segmentation_mask"]
    overlay = rgb.copy()
    overlay[mask > 0] = [239, 68, 52]
    blended = cv2.addWeighted(overlay, 0.46, rgb, 0.54, 0)

    fig2, ax2 = plt.subplots(1, 3, figsize=(9.2, 3.1), dpi=220)
    fig2.patch.set_facecolor("#141F33")
    ax2[0].imshow(rgb)
    ax2[0].set_title("Acquisition (128×128)", fontsize=10, fontweight="bold", color="#BAC8DC")
    ax2[0].axis("off")

    ax2[1].imshow(mask, cmap="magma")
    ax2[1].set_title("Masque Dense FireUNet", fontsize=10, fontweight="bold", color="#FDA472")
    ax2[1].axis("off")

    ax2[2].imshow(blended)
    if res["location"]["bbox"]:
        x1, y1, x2, y2 = res["location"]["bbox"]
        cx, cy = res["location"]["center_xy"]
        ax2[2].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#FACC15", lw=2.0))
        ax2[2].scatter([cx], [cy], c="#FACC15", s=60, edgecolors="#0B1320", zorder=5)
    ax2[2].set_title(f"Cible : {res['location']['zone']} ({res['location']['affected_area_pct']}%)", fontsize=10, fontweight="bold", color="#6EE7D8")
    ax2[2].axis("off")

    plt.tight_layout()
    cnn_seg_p = out / "prezi_cnn_seg.png"
    fig2.savefig(cnn_seg_p, dpi=220, facecolor="#141F33", bbox_inches="tight")
    plt.close(fig2)

    # C. Graphique Bi-LSTM (prezi_lstm_charts.png)
    ep_l = np.arange(1, len(lstm_m["val_speed_mae_kmh"]) + 1)
    fig3, ax3 = plt.subplots(1, 2, figsize=(9.2, 3.5), dpi=220)
    fig3.patch.set_facecolor("#141F33")

    ax3[0].plot(ep_l, lstm_m["val_speed_mae_kmh"], color="#14B8A6", lw=2.6, marker="o", markersize=3.5, label="MAE Vitesse (km/h)")
    ax3[0].axhline(0.283, color="#F59E0B", ls="--", lw=1.6, label="Convergence : 0,28 km/h")
    ax3[0].set_title("Erreur Vitesse de Propagation", fontsize=11, fontweight="bold", color="#F8FAFC", pad=8)
    ax3[0].set_xlabel("Époques")
    ax3[0].set_ylabel("MAE (km/h)")
    ax3[0].grid(True, color="#263854", alpha=0.6)
    ax3[0].legend(facecolor="#0B1320", edgecolor="#334868", fontsize=8.5)

    ax3[1].plot(ep_l, lstm_m["val_dir_angular_error_deg"], color="#60A5FA", lw=2.6, marker="s", markersize=3.5, label="Erreur Angulaire (°)")
    ax3[1].axhline(5.72, color="#EF5B2A", ls="--", lw=1.6, label="Convergence : 5,72°")
    ax3[1].set_title("Précision Géodésique Directionnelle", fontsize=11, fontweight="bold", color="#F8FAFC", pad=8)
    ax3[1].set_xlabel("Époques")
    ax3[1].set_ylabel("Degrés (°)")
    ax3[1].grid(True, color="#263854", alpha=0.6)
    ax3[1].legend(facecolor="#0B1320", edgecolor="#334868", fontsize=8.5)

    plt.tight_layout()
    lstm_chart_p = out / "prezi_lstm_charts.png"
    fig3.savefig(lstm_chart_p, dpi=220, facecolor="#141F33", bbox_inches="tight")
    plt.close(fig3)

    return {
        "cnn_charts": cnn_chart_p,
        "cnn_seg": cnn_seg_p,
        "lstm_charts": lstm_chart_p,
    }


# ==============================================================================
# 2. Utilitaires PowerPoint Prezi + Injection Transition MORPH (Caméra Zoom/Pan)
# ==============================================================================
def inject_morph_transition(slide, duration_ms: int = 950) -> None:
    """
    Injecte la transition native PowerPoint 'Morph' (Morphose par objet `!!`) dans le XML
    de la diapositive afin que PowerPoint anime les zooms et déplacements spatiaux comme Prezi.
    """
    sld = slide._element
    # Supprime une éventuelle transition existante
    for old in sld.findall("{http://schemas.openxmlformats.org/presentationml/2006/main}transition"):
        sld.remove(old)

    morph_xml = f"""
    <p:transition xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
                  xmlns:p14="http://schemas.microsoft.com/office/powerpoint/2010/main"
                  xmlns:p159="http://schemas.microsoft.com/office/powerpoint/2015/09/main"
                  spd="med" p14:dur="{duration_ms}">
      <p:extLst>
        <p:ext uri="{{EC167BDD-8182-4AB7-AECC-EB403E3ABB37}}">
          <p159:morph option="byObject"/>
        </p:ext>
      </p:extLst>
    </p:transition>
    """
    trans_el = etree.fromstring(morph_xml.strip())
    # Dans le schéma OpenXML, <p:transition> se place juste avant <p:timing> ou en fin de <p:sld>
    timing = sld.find("{http://schemas.openxmlformats.org/presentationml/2006/main}timing")
    if timing is not None:
        sld.insert(list(sld).index(timing), trans_el)
    else:
        sld.append(trans_el)


def set_morph_name(shape, morph_id: str) -> None:
    """Assigne un nom `!!MorphID` à une forme pour que PowerPoint Morph l'anime entre deux vues."""
    shape.name = f"!!{morph_id}"


def style_space_canvas(slide, morph_suffix: str = "") -> None:
    """Crée le fond spatial Prezi avec ses anneaux d'orbite subtils nommés pour Morph."""
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG_SPACE

    # Grands anneaux orbitaux décoratifs (qui bougent/zooment avec Morph !)
    outer_ring = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(-1.8), Inches(-1.6), Inches(16.9), Inches(10.8))
    outer_ring.fill.background()
    outer_ring.line.color.rgb = RGBColor(24, 38, 62)
    outer_ring.line.width = Pt(1.25)
    set_morph_name(outer_ring, "OrbitOuter")

    inner_ring = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(1.4), Inches(0.4), Inches(10.5), Inches(6.7))
    inner_ring.fill.background()
    inner_ring.line.color.rgb = RGBColor(30, 48, 78)
    inner_ring.line.width = Pt(1.0)
    set_morph_name(inner_ring, "OrbitInner")


def add_circle_node(
    slide,
    cx: float,
    cy: float,
    diameter: float,
    fill_col: RGBColor,
    ring_col: RGBColor,
    ring_width: float = 2.5,
    morph_id: str | None = None,
    title: str = "",
    subtitle: str = "",
    title_pt: float = 15,
    sub_pt: float = 10.5,
):
    left = cx - diameter / 2.0
    top = cy - diameter / 2.0
    oval = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(left), Inches(top), Inches(diameter), Inches(diameter))
    oval.fill.solid()
    oval.fill.fore_color.rgb = fill_col
    oval.line.color.rgb = ring_col
    oval.line.width = Pt(ring_width)
    if morph_id:
        set_morph_name(oval, morph_id)

    if title or subtitle:
        tf = oval.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = Inches(0.16)
        tf.margin_right = Inches(0.16)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        p.text = title
        p.font.name = "Calibri"
        p.font.size = Pt(title_pt)
        p.font.bold = True
        p.font.color.rgb = TEXT_WHITE

        if subtitle:
            p2 = tf.add_paragraph()
            p2.alignment = PP_ALIGN.CENTER
            p2.text = subtitle
            p2.font.name = "Calibri"
            p2.font.size = Pt(sub_pt)
            p2.font.color.rgb = TEXT_SOFT
            p2.space_before = Pt(4)
    return oval


def add_glass_panel(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    border_col: RGBColor,
    fill_col: RGBColor = BG_PANEL,
    morph_id: str | None = None,
):
    panel = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    panel.adjustments[0] = 0.035
    panel.fill.solid()
    panel.fill.fore_color.rgb = fill_col
    panel.line.color.rgb = border_col
    panel.line.width = Pt(1.5)
    if morph_id:
        set_morph_name(panel, morph_id)
    return panel


def add_connector_line(slide, x1: float, y1: float, x2: float, y2: float, col: RGBColor, morph_id: str | None = None):
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    line.line.color.rgb = col
    line.line.width = Pt(2.0)
    if morph_id:
        set_morph_name(line, morph_id)
    return line


def add_breadcrumb(slide, path_text: str, step_num: int, total_steps: int, accent: RGBColor) -> None:
    tb = slide.shapes.add_textbox(Inches(0.55), Inches(0.28), Inches(9.5), Inches(0.36))
    p = tb.text_frame.paragraphs[0]
    p.text = path_text
    p.font.name = "Calibri"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = accent

    tb_r = slide.shapes.add_textbox(Inches(11.2), Inches(0.28), Inches(1.6), Inches(0.36))
    pr = tb_r.text_frame.paragraphs[0]
    pr.alignment = PP_ALIGN.RIGHT
    pr.text = f"VUE {step_num:02d} / {total_steps:02d}"
    pr.font.name = "Calibri"
    pr.font.size = Pt(10.5)
    pr.font.bold = True
    pr.font.color.rgb = TEXT_DIM


def draw_overview_constellation(slide, active_hub: int | None = None) -> None:
    """
    Dessine la Carte Spatiale Prezi (Vue Globale) avec le Noyau Central et les 3 Hubs Agents.
    Utilisée à l'ouverture ET comme vue de transition avant chaque plongée (Zoom-In) !
    """
    # Lignes orbitales reliant le centre aux 3 hubs et entre eux
    add_connector_line(slide, 6.66, 3.85, 2.75, 2.55, EMBER if active_hub == 1 else RING_MUTED, "Link1")
    add_connector_line(slide, 6.66, 3.85, 10.58, 2.55, CYAN_TEAL if active_hub == 2 else RING_MUTED, "Link2")
    add_connector_line(slide, 6.66, 3.85, 6.66, 6.05, AMBER_GOLD if active_hub == 3 else RING_MUTED, "Link3")
    add_connector_line(slide, 2.75, 2.55, 10.58, 2.55, RING_MUTED, "Arc12")
    add_connector_line(slide, 10.58, 2.55, 6.66, 6.05, RING_MUTED, "Arc23")

    # Noyau central du projet
    add_circle_node(
        slide, 6.66, 3.65, 2.35,
        fill_col=RGBColor(17, 27, 46),
        ring_col=TEXT_WHITE if active_hub is None else RING_MUTED,
        ring_width=2.5,
        morph_id="CoreNode",
        title="PROJET 5\nFEUX DE FORÊT",
        subtitle="Architecture Multi-Agents\nAutonome",
        title_pt=14.5,
        sub_pt=10,
    )

    # Hub 1 : Salim (Gauche-Haut)
    d1 = 2.45 if active_hub == 1 else 2.10
    add_circle_node(
        slide, 2.75, 2.55, d1,
        fill_col=RGBColor(48, 24, 24) if active_hub == 1 else BG_NODE,
        ring_col=EMBER,
        ring_width=4.0 if active_hub == 1 else 2.5,
        morph_id="HubVision",
        title="01 • VISION CNN",
        subtitle="Salim Chablaoui\nFireUNet (99,11 %)",
        title_pt=13.5,
        sub_pt=10,
    )

    # Hub 2 : Moundir (Droite-Haut)
    d2 = 2.45 if active_hub == 2 else 2.10
    add_circle_node(
        slide, 10.58, 2.55, d2,
        fill_col=RGBColor(15, 46, 48) if active_hub == 2 else BG_NODE,
        ring_col=CYAN_TEAL,
        ring_width=4.0 if active_hub == 2 else 2.5,
        morph_id="HubLSTM",
        title="02 • BI-LSTM MÉTÉO",
        subtitle="Moundir\nPropagation (0,28 km/h)",
        title_pt=13.5,
        sub_pt=10,
    )

    # Hub 3 : Asma (Bas-Centre)
    d3 = 2.45 if active_hub == 3 else 2.10
    add_circle_node(
        slide, 6.66, 5.92, d3,
        fill_col=RGBColor(52, 38, 16) if active_hub == 3 else BG_NODE,
        ring_col=AMBER_GOLD,
        ring_width=4.0 if active_hub == 3 else 2.5,
        morph_id="HubAlert",
        title="03 • ALERTE & UI",
        subtitle="Asma\nArbre CART & Streamlit",
        title_pt=13.5,
        sub_pt=10,
    )

    # Petits satellites enfants autour des 3 hubs (donnent l'aspect galaxie Prezi !)
    add_circle_node(slide, 1.05, 1.45, 0.78, BG_PANEL, EMBER_GLOW, 1.5, "SubV1", "CLAHE", "", 9.5)
    add_circle_node(slide, 1.05, 3.70, 0.78, BG_PANEL, EMBER_GLOW, 1.5, "SubV2", "U-Net", "", 9.5)
    add_circle_node(slide, 12.28, 1.45, 0.78, BG_PANEL, CYAN_GLOW, 1.5, "SubL1", "sin/cos", "", 9.5)
    add_circle_node(slide, 12.28, 3.70, 0.78, BG_PANEL, CYAN_GLOW, 1.5, "SubL2", "+1h..3h", "", 9.0)
    add_circle_node(slide, 4.55, 6.35, 0.78, BG_PANEL, AMBER_GLOW, 1.5, "SubA1", "4 Paliers", "", 8.8)
    add_circle_node(slide, 8.77, 6.35, 0.78, BG_PANEL, AMBER_GLOW, 1.5, "SubA2", "Web App", "", 8.8)


def add_dark_bullets(slide, left: float, top: float, w: float, h: float, title: str, items: list[tuple[str, str]], title_col: RGBColor, pt_size: float = 13.0):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = title
    p0.font.name = "Calibri"
    p0.font.size = Pt(pt_size + 3)
    p0.font.bold = True
    p0.font.color.rgb = title_col
    p0.space_after = Pt(8)

    for head, body in items:
        p = tf.add_paragraph()
        p.space_after = Pt(6)
        p.line_spacing = 1.16
        r1 = p.add_run()
        r1.text = f"▸ {head} "
        r1.font.name = "Calibri"
        r1.font.size = Pt(pt_size)
        r1.font.bold = True
        r1.font.color.rgb = TEXT_WHITE

        r2 = p.add_run()
        r2.text = body
        r2.font.name = "Calibri"
        r2.font.size = Pt(pt_size)
        r2.font.color.rgb = TEXT_SOFT


def set_notes(slide, note_txt: str) -> None:
    slide.notes_slide.notes_text_frame.text = note_txt


# ==============================================================================
# 3. Assemblage des 15 Vues Prezi-Morph dans PowerPoint
# ==============================================================================
def build_prezi_pptx(charts: dict[str, Path]) -> Path:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]
    total = 15

    # --------------------------------------------------------------------------
    # VUE 1 : OUVERTURE SPATIALE — CARTE GLOBALE DU SYSTÈME (PREZI OVERVIEW)
    # --------------------------------------------------------------------------
    s1 = prs.slides.add_slide(blank)
    style_space_canvas(s1)
    tb_top = s1.shapes.add_textbox(Inches(0.6), Inches(0.28), Inches(12.1), Inches(0.85))
    tf_t = tb_top.text_frame
    p = tf_t.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.text = "SYSTÈME MULTI-AGENTS DE SURVEILLANCE & CONFINEMENT DES FEUX DE FORÊT"
    p.font.name = "Calibri"
    p.font.size = Pt(21)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE
    p2 = tf_t.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    p2.text = "Projet 5 — Data Mining & Deep Learning (4ème Année Ingéniorat)   |   Salim Chablaoui  •  Moundir  •  Asma"
    p2.font.name = "Calibri"
    p2.font.size = Pt(12.5)
    p2.font.color.rgb = EMBER_GLOW

    draw_overview_constellation(s1, active_hub=None)
    set_notes(s1, "Vue Globale d'ouverture (Style Prezi) : Voici la carte complète de notre écosystème multi-agents. Tout au long de la présentation, nous allons zoomer dans chacun des 3 pôles orbitaux : la Vision Satellite (Salim), la Dynamique Temporelle (Moundir) et le Moteur d'Alerte (Asma).")

    # --------------------------------------------------------------------------
    # VUE 2 : ZOOM AU CŒUR DU NOYAU CENTRAL — PROBLÉMATIQUE & DATASETS KAGGLE
    # (`!!CoreNode` s'agrandit en grand cercle d'ancrage à gauche grâce à Morph !)
    # --------------------------------------------------------------------------
    s2 = prs.slides.add_slide(blank)
    style_space_canvas(s2)
    inject_morph_transition(s2, 1000)
    add_breadcrumb(s2, "🌍 VUE GLOBALE  ›  NOYAU CENTRAL : PROBLÉMATIQUE & DONNÉES KAGGLE", 2, total, TEXT_WHITE)

    # Le CoreNode de la Vue 1 zoome et glisse à gauche !
    add_circle_node(
        s2, 2.35, 3.95, 3.35,
        fill_col=RGBColor(17, 27, 46),
        ring_col=EMBER,
        ring_width=3.5,
        morph_id="CoreNode",
        title="URGENCE\nCLIMATIQUE",
        subtitle="Détecter tôt\nModéliser vite\nÉvacuer à temps",
        title_pt=19,
        sub_pt=12.5,
    )

    add_glass_panel(s2, 4.45, 0.95, 8.3, 2.85, border_col=EMBER)
    add_dark_bullets(
        s2, 4.7, 1.12, 7.8, 2.55,
        "Pourquoi une Architecture Multi-Agents Autonome ?",
        [
            ("Délai critique des tours de guet :", "un départ de feu isolé sous vent sec (> 35 km/h, RH < 25 %) devient incontrôlable en moins de 30 minutes."),
            ("Synergie de 3 IA complémentaires :", "CNN spatial (Où est le feu ?) → Bi-LSTM physique (Où va-t-il ?) → Arbre CART (Quelle décision civile ?)."),
            ("Court-circuit temps réel :", "si l'image satellite est saine (Feu = Non), retour immédiat 🟢 VERT sans calcul superflu."),
        ],
        EMBER_GLOW, 12.8,
    )

    # 3 Bulles de métriques Data Kaggle en bas à droite
    kpis_data = [
        ("1 179", "Images Totales\n999 Kaggle + 180 Sat.", EMBER, 5.55),
        ("517", "Relevés Météo\nForest Fires Kaggle", CYAN_TEAL, 8.55),
        ("82 / 18 %", "Split Validation\n966 Train / 213 Val", AMBER_GOLD, 11.45),
    ]
    for val, lbl, col, cx in kpis_data:
        add_circle_node(s2, cx, 5.55, 2.25, BG_NODE, col, 2.5, None, val, lbl, 20, 10.5)

    set_notes(s2, "Zoom dans le Noyau Central : Face à la vitesse d'emballement des incendies, nous avons réuni deux datasets Kaggle de référence : 1 179 images (dont 100 % du Kaggle FIRE Dataset + 180 scènes satellites segmentées) et les 517 relevés météo Forest Fires.")

    # --------------------------------------------------------------------------
    # VUE 3 : RETOUR CARTE GLOBALE — FOCUS SUR HUB 01 (SALIM : VISION CNN)
    # --------------------------------------------------------------------------
    s3 = prs.slides.add_slide(blank)
    style_space_canvas(s3)
    inject_morph_transition(s3, 900)
    add_breadcrumb(s3, "🌍 CARTE MULTI-AGENTS  ›  CIBLAGE DU PÔLE 01 : VISION PAR ORDINATEUR (SALIM)", 3, total, EMBER_GLOW)
    draw_overview_constellation(s3, active_hub=1)
    set_notes(s3, "Transition Prezi vers Salim : Nous revenons un instant à la carte globale pour plonger dans le premier pôle en haut à gauche : la Vision par Ordinateur dirigée par Salim Chablaoui.")

    # --------------------------------------------------------------------------
    # VUE 4 : ZOOM HUB 01 (SALIM) — PRÉTRAITEMENT SATELLITE & MASQUES SPECTRAUX
    # (`!!HubVision` grandit en hub principal, `!!SubV1` devient nœud focus !)
    # --------------------------------------------------------------------------
    s4 = prs.slides.add_slide(blank)
    style_space_canvas(s4)
    inject_morph_transition(s4, 1000)
    add_breadcrumb(s4, "🌍 CARTE  ›  01 • VISION CNN (SALIM)  ›  PRÉTRAITEMENT SATELLITE", 4, total, EMBER_GLOW)

    add_circle_node(
        s4, 2.25, 3.95, 3.10,
        fill_col=RGBColor(48, 24, 24),
        ring_col=EMBER,
        ring_width=4.0,
        morph_id="HubVision",
        title="01 • SALIM\nPRÉTRAITEMENT",
        subtitle="preprocessing.py\nImages 128×128 RGB",
        title_pt=16.5,
        sub_pt=11.5,
    )

    sat_steps = [
        (5.75, 2.35, "SubV1", "1. CLAHE L*a*b*", "Rehaussement adaptatif du canal L* pour percer la fumée et le voile atmosphérique."),
        (10.35, 2.35, None, "2. Filtrage σ ≥ 2,0", "Élimination automatique des acquisitions satellites corrompues, tronquées ou noires."),
        (5.75, 5.45, "SubV2", "3. Fusion Spectrale", "Règles multi-espaces RGB + HSV + YCbCr (Y ≥ Cb, Cr > Cb) + morphologie 5×5."),
        (10.35, 5.45, None, "4. Augmentation", "Flips, rotations 90°, contraste α ∈ [0,85; 1,18] et bruit capteur synchronisés au masque."),
    ]
    for x_c, y_c, m_id, st_t, st_d in sat_steps:
        add_connector_line(s4, 3.80, 3.95, x_c - 1.85, y_c, EMBER)
        p_box = add_glass_panel(s4, x_c - 1.85, y_c - 1.12, 4.15, 2.24, border_col=EMBER, morph_id=m_id)
        add_dark_bullets(s4, x_c - 1.70, y_c - 0.98, 3.85, 1.98, st_t, [("Action :", st_d)], EMBER_GLOW, 12.2)

    set_notes(s4, "Salim : Dans ce premier sous-module, chaque image satellite subit 4 transformations : vérification d'intégrité, correction atmosphérique CLAHE dans l'espace L*a*b*, extraction spectrale multi-espaces RGB/HSV/YCbCr et augmentation géométrique synchronisée avec le masque.")

    # --------------------------------------------------------------------------
    # VUE 5 : ZOOM SOUS-NŒUD 01B (SALIM) — ARCHITECTURE MULTI-TÂCHES FIREUNET
    # --------------------------------------------------------------------------
    s5 = prs.slides.add_slide(blank)
    style_space_canvas(s5)
    inject_morph_transition(s5, 950)
    add_breadcrumb(s5, "🌍 CARTE  ›  01 • VISION CNN (SALIM)  ›  ARCHITECTURE FIREUNET", 5, total, EMBER_GLOW)

    # Schéma en U visuel Prezi (Encodeur descendant -> Bottleneck -> Décodeur montant)
    enc_nodes = [
        (1.6, 2.1, "Enc 1\n16 filtres"),
        (2.7, 3.5, "Enc 2\n32 filtres"),
        (3.8, 4.9, "Enc 3\n64 filtres"),
    ]
    dec_nodes = [
        (7.2, 4.9, "Déc 3\n64 + Skip"),
        (8.3, 3.5, "Déc 2\n32 + Skip"),
        (9.4, 2.1, "Déc 1\nMasque 128²"),
    ]
    for i in range(len(enc_nodes) - 1):
        add_connector_line(s5, enc_nodes[i][0], enc_nodes[i][1], enc_nodes[i + 1][0], enc_nodes[i + 1][1], EMBER)
        add_connector_line(s5, dec_nodes[i][0], dec_nodes[i][1], dec_nodes[i + 1][0], dec_nodes[i + 1][1], CYAN_TEAL)
    # Skip connections horizontales !
    for (ex, ey, _), (dx, dy, _) in zip(enc_nodes, reversed(dec_nodes)):
        add_connector_line(s5, ex, ey, dx, dy, RING_MUTED)
    add_connector_line(s5, 3.8, 4.9, 5.5, 5.85, EMBER)
    add_connector_line(s5, 5.5, 5.85, 7.2, 4.9, CYAN_TEAL)

    for cx, cy, txt in enc_nodes:
        add_circle_node(s5, cx, cy, 1.32, BG_NODE, EMBER, 2.2, None, txt, "", 10.5)
    for cx, cy, txt in dec_nodes:
        add_circle_node(s5, cx, cy, 1.38, BG_NODE, CYAN_TEAL, 2.2, None, txt, "", 10.2)

    # Bottleneck central + Tête Classification
    add_circle_node(
        s5, 5.5, 5.85, 1.78,
        RGBColor(55, 25, 25), AMBER_GOLD, 3.0, "SubV2",
        "Bottleneck\n128 filtres", "Tête 1 : Feu Oui/Non", 11, 9.2,
    )

    # Panneau de droite : Équation de perte hybride & Sérialisation
    add_glass_panel(s5, 10.35, 1.15, 2.55, 5.75, border_col=EMBER)
    add_dark_bullets(
        s5, 10.48, 1.32, 2.30, 5.45,
        "Double Tête & Loss",
        [
            ("Perte Hybride :", "L_BCE(cls) + 1,2 × (0,5·BCE + 0,5·Dice)."),
            ("Pourquoi Dice ?", "Le feu ne couvre parfois que 2 % de l'image."),
            ("Sauvegarde :", "fire_cnn_unet.pkl & .h5 (HDF5)."),
        ],
        EMBER_GLOW, 11.2,
    )

    tb_u_lbl = s5.shapes.add_textbox(Inches(3.6), Inches(1.35), Inches(3.8), Inches(0.6))
    pu = tb_u_lbl.text_frame.paragraphs[0]
    pu.alignment = PP_ALIGN.CENTER
    pu.text = "⟵  Skip Connections Spatiales  ⟶"
    pu.font.name = "Calibri"
    pu.font.size = Pt(12)
    pu.font.bold = True
    pu.font.color.rgb = TEXT_SOFT

    set_notes(s5, "Salim : Ce schéma en U illustre notre architecture FireUNet. La branche descendante encode la scène jusqu'au Bottleneck qui décide s'il y a un feu (Tête 1), tandis que la branche montante reconstruit le masque pixel par pixel grâce aux Skip Connections (Tête 2).")

    # --------------------------------------------------------------------------
    # VUE 6 : ZOOM RÉSULTATS 01C (SALIM) — MÉTRIQUES & SEGMENTATION VISUELLE
    # --------------------------------------------------------------------------
    s6 = prs.slides.add_slide(blank)
    style_space_canvas(s6)
    inject_morph_transition(s6, 950)
    add_breadcrumb(s6, "🌍 CARTE  ›  01 • VISION CNN (SALIM)  ›  PERFORMANCES & LOCALISATION GPS", 6, total, EMBER_GLOW)

    # 3 Anneaux KPI à gauche
    add_circle_node(s6, 1.75, 2.05, 1.85, BG_NODE, EMBER, 3.0, None, "99,11 %", "Val Accuracy", 19, 10.5)
    add_circle_node(s6, 1.75, 4.15, 1.85, BG_NODE, CYAN_TEAL, 3.0, None, "84,2 %", "Score Dice", 19, 10.5)
    add_circle_node(s6, 1.75, 6.15, 1.75, BG_NODE, AMBER_GOLD, 2.5, None, "79,5 %", "Score IoU", 18, 10.5)

    # Panneau Courbes + Panneau Segmentation sombre
    add_glass_panel(s6, 3.15, 0.95, 9.65, 3.05, border_col=RING_MUTED)
    s6.shapes.add_picture(str(charts["cnn_charts"]), Inches(3.35), Inches(1.05), width=Inches(7.35))
    tb_api = s6.shapes.add_textbox(Inches(10.78), Inches(1.18), Inches(1.9), Inches(2.6))
    tf_a = tb_api.text_frame
    tf_a.word_wrap = True
    pa = tf_a.paragraphs[0]
    pa.text = "Inférence Rapide\n\n• < 35 ms / img\n• Cache RAM\n• Zéro faux positif sur forêts saines"
    pa.font.name = "Calibri"
    pa.font.size = Pt(11.5)
    pa.font.bold = True
    pa.font.color.rgb = EMBER_GLOW

    add_glass_panel(s6, 3.15, 4.20, 9.65, 2.95, border_col=EMBER)
    s6.shapes.add_picture(str(charts["cnn_seg"]), Inches(3.35), Inches(4.30), width=Inches(7.25))
    tb_loc = s6.shapes.add_textbox(Inches(10.72), Inches(4.38), Inches(1.98), Inches(2.6))
    tf_lc = tb_loc.text_frame
    tf_lc.word_wrap = True
    plc = tf_lc.paragraphs[0]
    plc.text = "Sortie predict() :\n• Oui (99,9 %)\n• Zone : Centre\n• GPS : (36.74, 5.06)\n• Surface : 54,2 %"
    plc.font.name = "Consolas"
    plc.font.size = Pt(10.2)
    plc.font.color.rgb = TEXT_WHITE

    set_notes(s6, "Salim : Sur les 1 179 images, FireUNet culmine à 99,11 % d'exactitude et 84,2 % de Dice. En bas, on visualise le masque prédit et le barycentre jaune qui génère automatiquement les coordonnées GPS transmises à Moundir.")

    # --------------------------------------------------------------------------
    # VUE 7 : RETOUR CARTE GLOBALE — FOCUS SUR HUB 02 (MOUNDIR : BI-LSTM MÉTÉO)
    # --------------------------------------------------------------------------
    s7 = prs.slides.add_slide(blank)
    style_space_canvas(s7)
    inject_morph_transition(s7, 900)
    add_breadcrumb(s7, "🌍 CARTE MULTI-AGENTS  ›  CIBLAGE DU PÔLE 02 : DYNAMIQUE & MÉTÉO (MOUNDIR)", 7, total, CYAN_GLOW)
    draw_overview_constellation(s7, active_hub=2)
    set_notes(s7, "Transition Prezi vers Moundir : Dézoomons vers notre carte orbitale. Une fois la position du foyer connue, le flux traverse vers le pôle supérieur droit : la modélisation spatio-temporelle de Moundir.")

    # --------------------------------------------------------------------------
    # VUE 8 : ZOOM HUB 02 (MOUNDIR) — CERCLE TRIGONOMÉTRIQUE & PHYSIQUE DU FEU
    # --------------------------------------------------------------------------
    s8 = prs.slides.add_slide(blank)
    style_space_canvas(s8)
    inject_morph_transition(s8, 1000)
    add_breadcrumb(s8, "🌍 CARTE  ›  02 • BI-LSTM MÉTÉO (MOUNDIR)  ›  ENCODAGE CYCLIQUE & PHYSIQUE", 8, total, CYAN_GLOW)

    # Cercle Unité Visuel (359° <-> 1°) à droite (Morph depuis `!!HubLSTM` !)
    add_circle_node(
        s8, 9.95, 4.05, 3.85,
        fill_col=RGBColor(15, 46, 48),
        ring_col=CYAN_TEAL,
        ring_width=4.0,
        morph_id="HubLSTM",
        title="CERCLE UNITÉ\n(sin θ, cos θ)",
        subtitle="359° et 1° deviennent\nvoisins continus !\nθ = atan2(ŷ_sin, ŷ_cos)",
        title_pt=17,
        sub_pt=12,
    )
    # 4 Repères cardinaux autour du cercle unité
    add_circle_node(s8, 9.95, 1.65, 0.72, BG_PANEL, AMBER_GOLD, 2.0, "SubL1", "0° N", "", 10)
    add_circle_node(s8, 12.35, 4.05, 0.72, BG_PANEL, CYAN_GLOW, 1.5, None, "90° E", "", 10)
    add_circle_node(s8, 9.95, 6.45, 0.72, BG_PANEL, CYAN_GLOW, 1.5, None, "180° S", "", 9.5)
    add_circle_node(s8, 7.55, 4.05, 0.72, BG_PANEL, CYAN_GLOW, 1.5, None, "270° O", "", 9.5)

    add_glass_panel(s8, 0.65, 1.15, 6.35, 5.75, border_col=CYAN_TEAL)
    add_dark_bullets(
        s8, 0.90, 1.38, 5.85, 5.3,
        "Physique Atmosphérique & Feature Engineering",
        [
            ("Vecteur Vent (W, θ) :", "moteur principal du transfert thermique convectif et de la projection d'escarbilles."),
            ("Humidité (RH) & Température (T) :", "gouvernent la sécheresse du combustible végétal mort et l'inflammabilité."),
            ("Indice FWI / ISI (Rothermel) :", "descripteur physique non-linéaire intégré aux séquences temporelles."),
            ("Continuité Angulaire :", "la perte cosinus 1,5·(1 − u·û) élimine la singularité artificielle entre 359° et 1°."),
        ],
        CYAN_GLOW, 13.0,
    )

    set_notes(s8, "Moundir : En physique des incendies, l'angle du vent souffre d'une discontinuité numérique à 360°. En projetant θ sur le cercle unité (à droite) via sinus et cosinus, notre réseau comprend naturellement que 359° et 1° représentent quasiment le même cap Nord.")

    # --------------------------------------------------------------------------
    # VUE 9 : ZOOM SOUS-NŒUD 02B (MOUNDIR) — ARCHITECTURE BI-LSTM & RÉSULTATS
    # --------------------------------------------------------------------------
    s9 = prs.slides.add_slide(blank)
    style_space_canvas(s9)
    inject_morph_transition(s9, 950)
    add_breadcrumb(s9, "🌍 CARTE  ›  02 • BI-LSTM MÉTÉO (MOUNDIR)  ›  ARCHITECTURE & CONVERGENCE", 9, total, CYAN_GLOW)

    # 3 Bulles KPI à gauche
    add_circle_node(s9, 1.80, 2.05, 1.92, BG_NODE, CYAN_TEAL, 3.2, None, "0,28 km/h", "MAE Vitesse", 17.5, 10.5)
    add_circle_node(s9, 1.80, 4.18, 1.92, BG_NODE, EMBER, 3.0, None, "5,72°", "Erreur Angle", 19, 10.5)
    add_circle_node(s9, 1.80, 6.18, 1.75, BG_NODE, AMBER_GOLD, 2.5, "SubL2", "< 3 ms", "Temps Inférence", 18, 10)

    # Architecture en flux horizontal en haut à droite
    arch_steps = [
        (4.35, 1.85, "Bi-LSTM ×2\n128 unités"),
        (6.75, 1.85, "Multi-Head\nAttention"),
        (9.15, 1.85, "Vitesse v ≥ 0\n& Vecteur L2"),
        (11.55, 1.85, "Expansion\nt+1h, +2h, +3h"),
    ]
    for i in range(len(arch_steps) - 1):
        add_connector_line(s9, arch_steps[i][0], arch_steps[i][1], arch_steps[i + 1][0], arch_steps[i + 1][1], CYAN_TEAL)
    for cx, cy, lbl in arch_steps:
        add_circle_node(s9, cx, cy, 1.65, BG_PANEL, CYAN_GLOW, 2.0, None, lbl, "", 11)

    # Courbes sombres Bi-LSTM en bas à droite
    add_glass_panel(s9, 3.25, 3.05, 9.55, 4.05, border_col=CYAN_TEAL)
    s9.shapes.add_picture(str(charts["lstm_charts"]), Inches(3.45), Inches(3.18), width=Inches(9.15))

    set_notes(s9, "Moundir : FirePropagationLSTM empile 2 couches Bi-LSTM suivies d'un Multi-Head Attention Pooling qui cible les rafales critiques. Après 30 époques, l'erreur tombe à 0,28 km/h en vitesse et 5,72° en direction.")

    # --------------------------------------------------------------------------
    # VUE 10 : RETOUR CARTE GLOBALE — FOCUS SUR HUB 03 (ASMA : ALERTE & UI)
    # --------------------------------------------------------------------------
    s10 = prs.slides.add_slide(blank)
    style_space_canvas(s10)
    inject_morph_transition(s10, 900)
    add_breadcrumb(s10, "🌍 CARTE MULTI-AGENTS  ›  CIBLAGE DU PÔLE 03 : DÉCISION & STREAMLIT (ASMA)", 10, total, AMBER_GLOW)
    draw_overview_constellation(s10, active_hub=3)
    set_notes(s10, "Transition Prezi vers Asma : Retour sur la carte globale avant notre troisième plongée vers le pôle inférieur : le système d'alerte par Arbre de Décision et l'interface opérationnelle conçus par Asma.")

    # --------------------------------------------------------------------------
    # VUE 11 : ZOOM HUB 03 (ASMA) — ARBRE DE DÉCISION RADIAL & 4 PALIERS
    # (`!!HubAlert` devient le nœud racine qui irradie vers les 4 paliers !)
    # --------------------------------------------------------------------------
    s11 = prs.slides.add_slide(blank)
    style_space_canvas(s11)
    inject_morph_transition(s11, 1000)
    add_breadcrumb(s11, "🌍 CARTE  ›  03 • ALERTE & UI (ASMA)  ›  ARBRE DE DÉCISION & 4 PALIERS CIVILS", 11, total, AMBER_GLOW)

    # Racine de l'arbre à gauche
    add_circle_node(
        s11, 2.30, 4.05, 2.85,
        fill_col=RGBColor(52, 38, 16),
        ring_col=AMBER_GOLD,
        ring_width=4.0,
        morph_id="HubAlert",
        title="ARBRE CART\nEXPLICABLE",
        subtitle="[Feu, Vent, Humidité]\n100 % Déterministe",
        title_pt=16,
        sub_pt=11,
    )

    tiers_prezi = [
        (1.65, C_GREEN, "🟢 PALIER 0 — VERT", "Feu == Non  |  Patrouilles normales & veille satellite de routine."),
        (3.25, C_ORANGE, "🟠 PALIER 1 — ORANGE", "Feu actif, Vent < 20 km/h, RH > 40 %  |  Mise en alerte des vigies locales."),
        (4.85, C_RED, "🔴 PALIER 2 — ROUGE", "Vent moyen/fort, RH < 30 %  |  Pré-positionnement des bombardiers d'eau & fermeture massifs."),
        (6.45, C_EVAC, "🚨 PALIER 3 — ÉVACUATION", "Vent violent > 40 km/h, RH < 20 %  |  Sirènes civiles, blocage routier & évacuation immédiate."),
    ]
    for y_t, col_t, head_t, desc_t in tiers_prezi:
        add_connector_line(s11, 3.72, 4.05, 4.75, y_t, col_t)
        add_glass_panel(s11, 4.75, y_t - 0.62, 7.95, 1.25, border_col=col_t)
        tb_tr = s11.shapes.add_textbox(Inches(5.0), Inches(y_t - 0.48), Inches(7.5), Inches(1.0))
        tf_tr = tb_tr.text_frame
        tf_tr.word_wrap = True
        p_h = tf_tr.paragraphs[0]
        p_h.text = head_t
        p_h.font.name = "Calibri"
        p_h.font.size = Pt(14.5)
        p_h.font.bold = True
        p_h.font.color.rgb = col_t
        p_d = tf_tr.add_paragraph()
        p_d.text = desc_t
        p_d.font.name = "Calibri"
        p_d.font.size = Pt(12)
        p_d.font.color.rgb = TEXT_WHITE

    set_notes(s11, "Asma : En protection civile, chaque ordre d'évacuation doit être justifiable. Notre DecisionTreeClassifier traduit instantanément le vecteur [Feu, Vent, Humidité] en 4 paliers opérationnels clairs.")

    # --------------------------------------------------------------------------
    # VUE 12 : ZOOM SOUS-NŒUD 03B (ASMA) — COCKPIT STREAMLIT & PIPELINE MAIN.PY
    # --------------------------------------------------------------------------
    s12 = prs.slides.add_slide(blank)
    style_space_canvas(s12)
    inject_morph_transition(s12, 950)
    add_breadcrumb(s12, "🌍 CARTE  ›  03 • ALERTE & UI (ASMA)  ›  ORCHESTRATION MAIN.PY & STREAMLIT", 12, total, AMBER_GLOW)

    # Colonne Gauche : Terminal CLI main.py
    add_glass_panel(s12, 0.60, 1.05, 5.35, 5.95, border_col=AMBER_GOLD, morph_id="SubA1")
    add_dark_bullets(
        s12, 0.82, 1.25, 4.9, 2.6,
        "Orchestrateur Unifié (main.py)",
        [
            ("1. Vision (Salim) :", "predict(image) → stop immédiat en 🟢 VERT si 'Non'."),
            ("2. Météo (Moundir) :", "predict_propagation(location, wind, RH, T)."),
            ("3. Verdict (Asma) :", "donner_alerte(feu, vent, humidite)."),
        ],
        AMBER_GLOW, 12.2,
    )
    term_box = add_glass_panel(s12, 0.85, 4.10, 4.85, 2.55, border_col=RING_MUTED, fill_col=RGBColor(8, 13, 23))
    tf_tm = term_box.text_frame
    tf_tm.word_wrap = True
    tf_tm.margin_left = Inches(0.18)
    tf_tm.margin_top = Inches(0.16)
    ptm = tf_tm.paragraphs[0]
    ptm.text = (
        "$ python main.py\n"
        "Démarrage du test...\n"
        "--- 1. Analyse de l'image (Salim) ---\n"
        "--- 2. Analyse propagation (Moundir) ---\n"
        "--- 3. Décision (Asma) ---\n"
        "RÉSULTAT : 🚨 ÉVACUATION"
    )
    ptm.font.name = "Consolas"
    ptm.font.size = Pt(11)
    ptm.font.color.rgb = CYAN_GLOW
    ptm.line_spacing = 1.25

    # Colonne Droite : Cockpit Web Streamlit (app.py)
    add_glass_panel(s12, 6.20, 1.05, 6.55, 5.95, border_col=EMBER, morph_id="SubA2")
    tb_ui = s12.shapes.add_textbox(Inches(6.45), Inches(1.22), Inches(6.05), Inches(0.8))
    pui = tb_ui.text_frame.paragraphs[0]
    pui.text = "🔥 Tableau de Bord Web Temps Réel — streamlit run app.py"
    pui.font.name = "Calibri"
    pui.font.size = Pt(15.5)
    pui.font.bold = True
    pui.font.color.rgb = TEXT_WHITE

    add_glass_panel(s12, 6.48, 1.95, 2.85, 2.55, border_col=RING_MUTED, fill_col=BG_NODE)
    s12.shapes.add_picture("outputs/demo_kaggle_fire_segmented.png", Inches(7.18), Inches(2.42), width=Inches(1.45))
    tb_l1 = s12.shapes.add_textbox(Inches(6.58), Inches(2.02), Inches(2.65), Inches(0.35))
    pl1 = tb_l1.text_frame.paragraphs[0]
    pl1.text = "1. Image Satellite Uploadée"
    pl1.font.name = "Calibri"
    pl1.font.size = Pt(11)
    pl1.font.bold = True
    pl1.font.color.rgb = EMBER_GLOW

    add_glass_panel(s12, 9.55, 1.95, 2.92, 2.55, border_col=RING_MUTED, fill_col=BG_NODE)
    add_dark_bullets(
        s12, 9.68, 2.05, 2.68, 2.35,
        "2. Curseurs Météo",
        [
            ("Vent :", "38 km/h (Sud-Ouest)"),
            ("Humidité :", "22 %"),
            ("Température :", "34 °C"),
        ],
        CYAN_GLOW, 11.2,
    )

    btn_st = add_glass_panel(s12, 6.48, 4.72, 5.99, 0.55, border_col=EMBER, fill_col=EMBER)
    pb = btn_st.text_frame.paragraphs[0]
    pb.alignment = PP_ALIGN.CENTER
    pb.text = "🚀 LANCER L'ANALYSE MULTI-AGENTS (localhost:8501)"
    pb.font.name = "Calibri"
    pb.font.size = Pt(12)
    pb.font.bold = True
    pb.font.color.rgb = TEXT_WHITE

    al_st = add_glass_panel(s12, 6.48, 5.45, 5.99, 1.25, border_col=C_EVAC, fill_col=RGBColor(69, 15, 28))
    pal = al_st.text_frame.paragraphs[0]
    pal.alignment = PP_ALIGN.CENTER
    pal.text = "NIVEAU D'ALERTE : 🚨 ÉVACUATION IMMÉDIATE\nFoyer détecté au Centre + Vent fort (38 km/h) + Air sec (22 %)"
    pal.font.name = "Calibri"
    pal.font.size = Pt(12.5)
    pal.font.bold = True
    pal.font.color.rgb = TEXT_WHITE

    set_notes(s12, "Asma : Voici l'orchestration complète dans main.py à gauche et notre cockpit web Streamlit à droite. En téléversant une image et en réglant les curseurs météo, les autorités visualisent immédiatement la recommandation d'urgence.")

    # --------------------------------------------------------------------------
    # VUE 13 : DÉZOOM PANORAMIQUE — MATRICE COMPARATIVE DES 3 AGENTS
    # --------------------------------------------------------------------------
    s13 = prs.slides.add_slide(blank)
    style_space_canvas(s13)
    inject_morph_transition(s13, 950)
    add_breadcrumb(s13, "🌍 SYNTHÈSE GLOBALE  ›  MATRICE DES PERFORMANCES DES 3 MODULES", 13, total, TEXT_WHITE)

    cols_cards = [
        ("MODULE 1 — SALIM", EMBER, "HubVision", [
            ("Modèle :", "FireUNet (CNN U-Net double tête)"),
            ("Entrée :", "Image RGB 3 × 128 × 128"),
            ("Sorties :", "Oui/Non, Masque, GPS, % Surface"),
            ("Précision :", "99,11 % Classification"),
            ("Segmentation :", "Dice 84,2 %  |  IoU 79,5 %"),
            ("Artefacts :", "fire_cnn_unet.pkl & .h5"),
        ]),
        ("MODULE 2 — MOUNDIR", CYAN_TEAL, "HubLSTM", [
            ("Modèle :", "Bi-LSTM (2×128) + Attention"),
            ("Entrées :", "Séquence météo + GPS du CNN"),
            ("Sorties :", "Vitesse, Angle (°), Horizons +1..3h"),
            ("Erreur Vitesse :", "MAE = 0,28 km/h"),
            ("Erreur Angle :", "5,72°  |  Latence < 3 ms"),
            ("Artefacts :", "fire_propagation_lstm.pkl & .pt"),
        ]),
        ("MODULE 3 — ASMA", AMBER_GOLD, "HubAlert", [
            ("Modèle :", "DecisionTreeClassifier (CART)"),
            ("Entrées :", "Vecteur [Feu, Vent, Humidité]"),
            ("Sorties :", "Vert, Orange, Rouge, Évacuation"),
            ("Fiabilité :", "100 % Explicable & Déterministe"),
            ("Intégration :", "Pipeline CLI main.py"),
            ("Interface :", "Dashboard Web Streamlit app.py"),
        ]),
    ]
    for idx, (c_title, col, m_id, bullets) in enumerate(cols_cards):
        x = 0.62 + idx * 4.12
        add_glass_panel(s13, x, 1.15, 3.85, 5.85, border_col=col, morph_id=m_id)
        add_dark_bullets(s13, x + 0.18, 1.38, 3.5, 5.4, c_title, bullets, col, 12.2)

    set_notes(s13, "Équipe : En prenant du recul, nos trois agents forment une chaîne complète, rapide et rigoureusement validée : 99,11 % en vision, 0,28 km/h en prévision dynamique et une décision explicable temps réel.")

    # --------------------------------------------------------------------------
    # VUE 14 : ORBITE FUTURE — 4 PERSPECTIVES D'ÉVOLUTION INDUSTRIELLE
    # --------------------------------------------------------------------------
    s14 = prs.slides.add_slide(blank)
    style_space_canvas(s14)
    inject_morph_transition(s14, 950)
    add_breadcrumb(s14, "🌍 TRAJECTOIRE FUTURE  ›  4 PERSPECTIVES D'ÉVOLUTION OPÉRATIONNELLE", 14, total, CYAN_GLOW)

    persp_nodes = [
        (3.4, 2.45, EMBER, "1. API Satellites Live", "Ingestion automatique Sentinel-2 (Copernicus) & Landsat-8 toutes les 6 à 12h."),
        (9.9, 2.45, CYAN_TEAL, "2. Topographie MNT", "Couplage du Bi-LSTM avec la pente et l'exposition des versants montagneux."),
        (3.4, 5.45, AMBER_GOLD, "3. Drones Edge AI", "Déploiement embarqué sur NVIDIA Jetson pour inspecter sous canopée dense."),
        (9.9, 5.45, RGBColor(168, 85, 247), "4. Cloud & CODIS", "Micro-services Docker/Kubernetes et alertes SMS/Push vers les pompiers."),
    ]
    add_connector_line(s14, 3.4, 2.45, 9.9, 2.45, RING_MUTED)
    add_connector_line(s14, 9.9, 2.45, 9.9, 5.45, RING_MUTED)
    add_connector_line(s14, 3.4, 5.45, 9.9, 5.45, RING_MUTED)
    add_connector_line(s14, 3.4, 2.45, 3.4, 5.45, RING_MUTED)

    for cx, cy, col, t_txt, b_txt in persp_nodes:
        add_glass_panel(s14, cx - 2.65, cy - 1.05, 5.30, 2.10, border_col=col)
        add_dark_bullets(s14, cx - 2.45, cy - 0.88, 4.90, 1.80, t_txt, [("Perspective :", b_txt)], col, 12.8)

    set_notes(s14, "Équipe : Pour passer du prototype académique au déploiement national, nous identifions 4 axes : les flux live Copernicus Sentinel-2, les Modèles Numériques de Terrain, l'Edge AI sur drones thermiques et la conteneurisation Cloud connectée aux centres CODIS.")

    # --------------------------------------------------------------------------
    # VUE 15 : DÉZOOM FINAL SUR LA CONSTELLATION COMPLÈTE — MERCI & DÉMO LIVE
    # --------------------------------------------------------------------------
    s15 = prs.slides.add_slide(blank)
    style_space_canvas(s15)
    inject_morph_transition(s15, 1100)
    draw_overview_constellation(s15, active_hub=None)

    banner_end = add_glass_panel(s15, 2.15, 0.25, 9.03, 0.92, border_col=EMBER, fill_col=RGBColor(24, 36, 60))
    tf_e = banner_end.text_frame
    pe1 = tf_e.paragraphs[0]
    pe1.alignment = PP_ALIGN.CENTER
    pe1.text = "MERCI DE VOTRE ATTENTION !  —  PLACE À LA DÉMONSTRATION LIVE STREAMLIT"
    pe1.font.name = "Calibri"
    pe1.font.size = Pt(16.5)
    pe1.font.bold = True
    pe1.font.color.rgb = TEXT_WHITE
    pe2 = tf_e.add_paragraph()
    pe2.alignment = PP_ALIGN.CENTER
    pe2.text = "Salim Chablaoui  •  Moundir  •  Asma   |   Des questions ?"
    pe2.font.name = "Calibri"
    pe2.font.size = Pt(12)
    pe2.font.color.rgb = EMBER_GLOW

    set_notes(s15, "Conclusion : Nous terminons sur la vue globale de notre système multi-agents. Merci pour votre écoute ! Nous allons maintenant lancer `streamlit run app.py` pour la démonstration interactive et répondre à vos questions.")

    out_pptx = Path("Presentation_Prezi_Style_FireDetection.pptx")
    prs.save(str(out_pptx))
    return out_pptx.resolve()


# ==============================================================================
# 4. Génération du Vrai Canevas Prezi Interactif HTML5 100 % Hors-Ligne
# ==============================================================================
def _img_to_b64(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def build_interactive_prezi_html(charts: dict[str, Path]) -> Path:
    b64_cnn_charts = _img_to_b64(charts["cnn_charts"])
    b64_cnn_seg = _img_to_b64(charts["cnn_seg"])
    b64_lstm_charts = _img_to_b64(charts["lstm_charts"])
    b64_demo_fire = _img_to_b64("outputs/demo_kaggle_fire_segmented.png")

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Prezi Interactive — Projet 5 : Surveillance Autonome des Feux de Forêt</title>
<style>
  :root {{
    --bg: #080f1d;
    --panel: rgba(20, 31, 51, 0.92);
    --node: rgba(25, 40, 68, 0.95);
    --ember: #ef5b2a;
    --ember-soft: #fda472;
    --teal: #14b8a6;
    --teal-soft: #6ee7d8;
    --amber: #f59e0b;
    --amber-soft: #fcd34d;
    --text: #f8fafc;
    --muted: #bac8dc;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: 100vw; height: 100vh; overflow: hidden;
    background: radial-gradient(circle at 50% 45%, #13223d 0%, #080f1d 70%, #040811 100%);
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    color: var(--text);
    user-select: none;
  }}

  /* Grille spatiale subtile */
  #viewport {{
    position: relative; width: 100vw; height: 100vh; overflow: hidden;
    perspective: 1400px;
  }}

  #world {{
    position: absolute; left: 50%; top: 50%;
    width: 6000px; height: 4000px;
    transform-origin: 0 0;
    transition: transform 1.15s cubic-bezier(0.22, 1, 0.36, 1);
    will-change: transform;
  }}

  /* Lignes orbitales SVG sur le canevas géant */
  svg.orbit-svg {{
    position: absolute; left: -3000px; top: -2000px;
    width: 6000px; height: 4000px; pointer-events: none; overflow: visible;
  }}

  /* Nœuds Prezi sur le canevas */
  .p-frame {{
    position: absolute;
    transform: translate(-50%, -50%);
    cursor: pointer;
    transition: box-shadow 0.35s ease, border-color 0.35s ease;
  }}
  .circle-hub {{
    border-radius: 50%;
    background: var(--node);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    text-align: center; padding: 36px;
    box-shadow: 0 24px 70px rgba(0,0,0,0.55), inset 0 0 40px rgba(255,255,255,0.03);
  }}
  .card-hub {{
    border-radius: 26px;
    background: var(--panel);
    padding: 34px 40px;
    box-shadow: 0 24px 65px rgba(0,0,0,0.6);
    border: 2px solid rgba(148, 163, 184, 0.25);
  }}
  .b-ember {{ border: 4px solid var(--ember); }}
  .b-teal  {{ border: 4px solid var(--teal); }}
  .b-amber {{ border: 4px solid var(--amber); }}
  .b-white {{ border: 4px solid rgba(248, 250, 252, 0.75); }}

  .tag {{
    font-size: 13px; font-weight: 700; letter-spacing: 1.6px;
    text-transform: uppercase; margin-bottom: 8px; display: inline-block;
  }}
  h1 {{ font-size: 38px; line-height: 1.15; margin-bottom: 12px; font-weight: 800; }}
  h2 {{ font-size: 27px; line-height: 1.2; margin-bottom: 14px; font-weight: 700; }}
  h3 {{ font-size: 20px; line-height: 1.25; margin-bottom: 10px; font-weight: 700; }}
  p, li {{ font-size: 16.5px; line-height: 1.5; color: var(--muted); }}
  ul {{ list-style: none; }}
  ul li {{ margin-bottom: 10px; padding-left: 18px; position: relative; }}
  ul li::before {{
    content: "▸"; position: absolute; left: 0; color: var(--ember-soft); font-weight: bold;
  }}
  ul.teal-list li::before {{ color: var(--teal-soft); }}
  ul.amber-list li::before {{ color: var(--amber-soft); }}
  strong {{ color: var(--text); }}

  .kpi-row {{ display: flex; gap: 16px; margin: 16px 0; }}
  .kpi-pill {{
    flex: 1; background: rgba(8, 15, 29, 0.75); border-radius: 16px;
    padding: 14px 12px; text-align: center; border: 1px solid rgba(255,255,255,0.12);
  }}
  .kpi-pill .val {{ font-size: 28px; font-weight: 800; display: block; }}
  .kpi-pill .lbl {{ font-size: 12.5px; color: var(--muted); }}

  .chart-img {{
    width: 100%; border-radius: 14px; border: 1px solid rgba(255,255,255,0.12);
    margin-top: 10px; display: block;
  }}

  .tier-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 12px; }}
  .tier-box {{
    padding: 15px 18px; border-radius: 14px; background: rgba(8, 15, 29, 0.78);
    border-left: 6px solid;
  }}

  .code-block {{
    font-family: 'Cascadia Code', Consolas, monospace;
    background: #060b14; color: #6ee7d8; padding: 16px 18px;
    border-radius: 12px; font-size: 13.5px; line-height: 1.45;
    border: 1px solid rgba(110, 231, 216, 0.25); margin-top: 12px; white-space: pre-wrap;
  }}

  /* Barre de contrôle Prezi flottante en bas */
  #hud {{
    position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
    background: rgba(11, 19, 32, 0.88); backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.16); border-radius: 999px;
    padding: 10px 22px; display: flex; align-items: center; gap: 14px;
    z-index: 100; box-shadow: 0 12px 35px rgba(0,0,0,0.6);
  }}
  .hud-btn {{
    background: rgba(255,255,255,0.08); color: var(--text); border: none;
    padding: 8px 16px; border-radius: 999px; font-size: 13.5px; font-weight: 600;
    cursor: pointer; transition: background 0.2s;
  }}
  .hud-btn:hover {{ background: var(--ember); }}
  #step-label {{ font-size: 13.5px; font-weight: 700; min-width: 240px; text-align: center; color: var(--muted); }}

  /* Panneau Notes de l'Orateur (Touche N) */
  #notes-drawer {{
    position: fixed; right: 22px; bottom: 82px; width: 410px;
    background: rgba(11, 19, 32, 0.94); backdrop-filter: blur(14px);
    border: 1px solid var(--amber); border-radius: 18px; padding: 18px 20px;
    z-index: 99; display: none; box-shadow: 0 16px 45px rgba(0,0,0,0.65);
  }}
  #notes-drawer.visible {{ display: block; }}
  #notes-drawer h4 {{ color: var(--amber-soft); font-size: 13px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }}
  #notes-content {{ font-size: 14px; line-height: 1.5; color: var(--text); }}

  /*Barre de progression supérieure */
  #progress-bar {{
    position: fixed; top: 0; left: 0; height: 4px; background: linear-gradient(90deg, var(--ember), var(--teal), var(--amber));
    width: 0%; transition: width 0.5s ease; z-index: 101;
  }}
</style>
</head>
<body>

<div id="progress-bar"></div>

<div id="viewport">
  <div id="world">
    <!-- Anneaux orbitaux et connecteurs spatiaux -->
    <svg class="orbit-svg" viewBox="-3000 -2000 6000 4000">
      <circle cx="0" cy="0" r="1420" fill="none" stroke="rgba(148,163,184,0.12)" stroke-width="4" stroke-dasharray="18 14"/>
      <circle cx="0" cy="0" r="880" fill="none" stroke="rgba(148,163,184,0.16)" stroke-width="3"/>
      <!-- Lignes vers les 3 hubs principaux -->
      <line x1="0" y1="0" x2="-1220" y2="-650" stroke="#ef5b2a" stroke-width="5" stroke-opacity="0.55"/>
      <line x1="0" y1="0" x2="1220" y2="-650" stroke="#14b8a6" stroke-width="5" stroke-opacity="0.55"/>
      <line x1="0" y1="0" x2="0" y2="1080" stroke="#f59e0b" stroke-width="5" stroke-opacity="0.55"/>
      <!-- Sous-connexions Hub 1 (Salim) -->
      <line x1="-1220" y1="-650" x2="-2180" y2="-1120" stroke="#ef5b2a" stroke-width="3" stroke-opacity="0.45"/>
      <line x1="-1220" y1="-650" x2="-2240" y2="-320" stroke="#ef5b2a" stroke-width="3" stroke-opacity="0.45"/>
      <!-- Sous-connexions Hub 2 (Moundir) -->
      <line x1="1220" y1="-650" x2="2200" y2="-1080" stroke="#14b8a6" stroke-width="3" stroke-opacity="0.45"/>
      <line x1="1220" y1="-650" x2="2240" y2="-250" stroke="#14b8a6" stroke-width="3" stroke-opacity="0.45"/>
      <!-- Sous-connexions Hub 3 (Asma) -->
      <line x1="0" y1="1080" x2="-980" y2="1520" stroke="#f59e0b" stroke-width="3" stroke-opacity="0.45"/>
      <line x1="0" y1="1080" x2="980" y2="1520" stroke="#f59e0b" stroke-width="3" stroke-opacity="0.45"/>
    </svg>

    <!-- NOYAU CENTRAL (0, 0) -->
    <div class="p-frame circle-hub b-white" style="left:0px; top:0px; width:680px; height:680px;" onclick="goToStep(1)">
      <span class="tag" style="color:var(--ember-soft);">Projet 5 • Data Mining & Deep Learning</span>
      <h1 style="font-size:34px;">Surveillance & Confinement Autonome des Feux de Forêt</h1>
      <p style="margin-bottom:18px;">Architecture Multi-Agents Intelligente — 4ème Année Ingéniorat</p>
      <div class="kpi-row" style="width:100%;">
        <div class="kpi-pill"><span class="val" style="color:var(--ember); font-size:18px;">Salim</span><span class="lbl">CNN / U-Net</span></div>
        <div class="kpi-pill"><span class="val" style="color:var(--teal); font-size:18px;">Moundir</span><span class="lbl">Bi-LSTM Météo</span></div>
        <div class="kpi-pill"><span class="val" style="color:var(--amber); font-size:18px;">Asma</span><span class="lbl">CART & Streamlit</span></div>
      </div>
      <p style="font-size:13.5px; margin-top:10px; color:#94a3b8;">1 179 Images Satellites/Feu • 517 Relevés Météo Kaggle</p>
    </div>

    <!-- ==================== PÔLE 1 : SALIM (-1220, -650) ==================== -->
    <div class="p-frame circle-hub b-ember" style="left:-1220px; top:-650px; width:590px; height:590px;" onclick="goToStep(3)">
      <span class="tag" style="color:var(--ember-soft);">Module 01 • Salim Chablaoui</span>
      <h2>Vision par Ordinateur & Segmentation</h2>
      <ul style="text-align:left; margin-top:8px;">
        <li><strong>CLAHE (L*a*b*) :</strong> percée du voile atmosphérique.</li>
        <li><strong>Masques Spectraux :</strong> fusion RGB + HSV + YCbCr.</li>
        <li><strong>FireUNet Double Tête :</strong> Classification + Masque 128².</li>
        <li><strong>Loss Hybride :</strong> BCE + Coefficient de Dice.</li>
      </ul>
    </div>

    <!-- Sous-nœud 1A : Architecture & Métriques CNN (-2180, -1120) -->
    <div class="p-frame card-hub b-ember" style="left:-2180px; top:-1120px; width:760px;" onclick="goToStep(4)">
      <span class="tag" style="color:var(--ember-soft);">Salim • Convergence & Performances (1 179 Images)</span>
      <h3>Modèle FireUNet (.pkl & .h5)</h3>
      <div class="kpi-row">
        <div class="kpi-pill"><span class="val" style="color:var(--ember);">99,11 %</span><span class="lbl">Val Accuracy</span></div>
        <div class="kpi-pill"><span class="val" style="color:var(--teal);">84,2 %</span><span class="lbl">Score Dice</span></div>
        <div class="kpi-pill"><span class="val" style="color:var(--amber);">79,5 %</span><span class="lbl">Score IoU</span></div>
      </div>
      <img class="chart-img" src="{b64_cnn_charts}" alt="Courbes CNN">
    </div>

    <!-- Sous-nœud 1B : Segmentation Visuelle & GPS (-2240, -320) -->
    <div class="p-frame card-hub b-ember" style="left:-2240px; top:-320px; width:760px;" onclick="goToStep(5)">
      <span class="tag" style="color:var(--ember-soft);">Salim • API Officielle predict(image_path)</span>
      <h3>Délimitation du Foyer & Extraction GPS</h3>
      <img class="chart-img" src="{b64_cnn_seg}" alt="Segmentation U-Net">
      <div class="code-block">res = predict("sat_fire.png")
# → {{"fire_detected": "Oui", "confidence": 0.999,
#    "location": {{"zone": "Centre", "gps_coords": (36.741, 5.064), "affected_area_pct": 54.21}}}}</div>
    </div>

    <!-- ==================== PÔLE 2 : MOUNDIR (1220, -650) ==================== -->
    <div class="p-frame circle-hub b-teal" style="left:1220px; top:-650px; width:590px; height:590px;" onclick="goToStep(7)">
      <span class="tag" style="color:var(--teal-soft);">Module 02 • Moundir</span>
      <h2>Modélisation Temporelle & Météo</h2>
      <ul class="teal-list" style="text-align:left; margin-top:8px;">
        <li><strong>Physique Rothermel :</strong> Vent (W, θ), RH, Température, FWI.</li>
        <li><strong>Cercle Unité :</strong> (sin θ, cos θ) résout 359° ↔ 1°.</li>
        <li><strong>Bi-LSTM (2×128) :</strong> + Multi-Head Attention Pooling.</li>
        <li><strong>Sorties :</strong> Vitesse (km/h), Cap (°), Horizons +1h..+3h.</li>
      </ul>
    </div>

    <!-- Sous-nœud 2A : Mathématiques Cycliques (2200, -1080) -->
    <div class="p-frame card-hub b-teal" style="left:2200px; top:-1080px; width:720px;" onclick="goToStep(8)">
      <span class="tag" style="color:var(--teal-soft);">Moundir • Continuité Angulaire & Perte Physique</span>
      <h3>Projection Géodésique sur le Cercle Unité</h3>
      <ul class="teal-list">
        <li><strong>Encodage :</strong> x_sin = sin(2πθ / 360)  et  x_cos = cos(2πθ / 360).</li>
        <li><strong>Restitution :</strong> θ = atan2(ŷ_sin, ŷ_cos) mod 360° sans effet de bord au Nord.</li>
        <li><strong>Loss Multi-Tâches :</strong> L_SmoothL1(vitesse) + 1,5·(1 − u·û) + 0,8·L_MSE(expansion).</li>
      </ul>
      <div class="code-block">res_lstm = predict_propagation(location=res_cnn["location"],
    wind={{"speed": 38.0, "direction": "Sud-Ouest"}}, humidity=22.0, temperature=34.0)
# → Vitesse : 7,41 km/h | Direction : Sud-Ouest | Expansion t+1h, t+2h, t+3h</div>
    </div>

    <!-- Sous-nœud 2B : Courbes Bi-LSTM (2240, -250) -->
    <div class="p-frame card-hub b-teal" style="left:2240px; top:-250px; width:760px;" onclick="goToStep(9)">
      <span class="tag" style="color:var(--teal-soft);">Moundir • Validation Expérimentale (30 Époques)</span>
      <h3>Performances FirePropagationLSTM</h3>
      <div class="kpi-row">
        <div class="kpi-pill"><span class="val" style="color:var(--teal);">0,28 km/h</span><span class="lbl">MAE Vitesse</span></div>
        <div class="kpi-pill"><span class="val" style="color:var(--ember);">5,72°</span><span class="lbl">Erreur Angulaire</span></div>
        <div class="kpi-pill"><span class="val" style="color:var(--amber);">&lt; 3 ms</span><span class="lbl">Inférence Temps Réel</span></div>
      </div>
      <img class="chart-img" src="{b64_lstm_charts}" alt="Courbes Bi-LSTM">
    </div>

    <!-- ==================== PÔLE 3 : ASMA (0, 1080) ==================== -->
    <div class="p-frame circle-hub b-amber" style="left:0px; top:1080px; width:590px; height:590px;" onclick="goToStep(11)">
      <span class="tag" style="color:var(--amber-soft);">Module 03 • Asma</span>
      <h2>Système d'Alerte & Intégration</h2>
      <ul class="amber-list" style="text-align:left; margin-top:8px;">
        <li><strong>DecisionTreeClassifier :</strong> IA explicable & auditée.</li>
        <li><strong>Court-Circuit Vert :</strong> arrêt immédiat si Feu == Non.</li>
        <li><strong>Pipeline Unifié :</strong> orchestration dans main.py.</li>
        <li><strong>Cockpit Web :</strong> application interactive Streamlit.</li>
      </ul>
    </div>

    <!-- Sous-nœud 3A : 4 Paliers d'Alerte (-980, 1520) -->
    <div class="p-frame card-hub b-amber" style="left:-980px; top:1520px; width:760px;" onclick="goToStep(12)">
      <span class="tag" style="color:var(--amber-soft);">Asma • Matrice Opérationnelle (mon_arbre_alerte.pkl)</span>
      <h3>Les 4 Paliers Normalisés de Sécurité Civile</h3>
      <div class="tier-grid">
        <div class="tier-box" style="border-color:#22c55e;"><strong style="color:#4ade80;">🟢 VERT</strong><p style="font-size:14px;">Aucun feu (feu=0). Veille satellite normale.</p></div>
        <div class="tier-box" style="border-color:#f97316;"><strong style="color:#fb923c;">🟠 ORANGE</strong><p style="font-size:14px;">Vent &lt; 20 km/h, RH &gt; 40 %. Vigies locales mobilisées.</p></div>
        <div class="tier-box" style="border-color:#ef4444;"><strong style="color:#f87171;">🔴 ROUGE</strong><p style="font-size:14px;">Vent fort, RH &lt; 30 %. Bombardiers d'eau pré-positionnés.</p></div>
        <div class="tier-box" style="border-color:#e11d48;"><strong style="color:#fb7185;">🚨 ÉVACUATION</strong><p style="font-size:14px;">Vent &gt; 40 km/h, RH &lt; 20 %. Sirènes & évacuation civile.</p></div>
      </div>
    </div>

    <!-- Sous-nœud 3B : Démo Streamlit & Perspectives (980, 1520) -->
    <div class="p-frame card-hub b-amber" style="left:980px; top:1520px; width:760px;" onclick="goToStep(13)">
      <span class="tag" style="color:var(--amber-soft);">Asma & Équipe • Interface Web & Perspectives</span>
      <h3>Tableau de Bord Streamlit (app.py) & Évolutions</h3>
      <div style="display:flex; gap:18px; align-items:center; margin:10px 0;">
        <img src="{b64_demo_fire}" style="width:135px; height:135px; border-radius:14px; border:2px solid var(--ember);" alt="Demo">
        <div>
          <div class="code-block" style="margin-top:0;">$ streamlit run app.py
→ Upload Image + Vent 38 km/h + RH 22 %
→ VERDICT : 🚨 ÉVACUATION IMMÉDIATE</div>
          <p style="font-size:14px; margin-top:8px;"><strong>Perspectives :</strong> Flux live Sentinel-2, Topographie MNT, Drones Edge AI (Jetson) & Cloud CODIS.</p>
        </div>
      </div>
    </div>

  </div>
</div>

<!-- Panneau Notes Orateur -->
<div id="notes-drawer">
  <h4>🎤 Notes de l'Orateur (Touche N)</h4>
  <div id="notes-content"></div>
</div>

<!-- Barre de contrôle Prezi -->
<div id="hud">
  <button class="hud-btn" onclick="prevStep()">◀ Préc.</button>
  <span id="step-label">01 / 15 — Vue Globale</span>
  <button class="hud-btn" onclick="nextStep()">Suiv. ▶</button>
  <button class="hud-btn" onclick="goToStep(0)" title="Vue Globale (Touche O)">🌍 Carte (O)</button>
  <button class="hud-btn" onclick="toggleNotes()" title="Notes Orateur (Touche N)">🎤 Notes (N)</button>
  <button class="hud-btn" onclick="toggleFull()" title="Plein Écran (Touche F)">⛶ Plein Écran</button>
</div>

<script>
  // Trajectoire caméra Prezi : x, y (cible sur le canevas), scale (niveau de zoom), rot (légère inclinaison Prezi)
  const steps = [
    {{ x: 0,     y: 120,   scale: 0.25, rot: 0,  title: "Vue Globale de la Constellation", note: "Ouverture (Équipe) : Voici l'architecture complète de notre Projet 5 répartie en 3 pôles multi-agents autour du noyau central." }},
    {{ x: 0,     y: 0,     scale: 1.02, rot: 0,  title: "Noyau Central — Problématique & Données", note: "Asma / Salim : Pourquoi ce système ? Pour coupler détection précoce sur 1 179 images et modélisation météo sur 517 relevés Forest Fires." }},
    {{ x: -450,  y: -220,  scale: 0.34, rot: 0,  title: "Cap vers Pôle 01 (Salim — Vision)", note: "Transition vers le premier agent : l'analyse d'images satellites par Salim Chablaoui." }},
    {{ x: -1220, y: -650,  scale: 1.12, rot: -2, title: "Module 01 — Vision & Prétraitement (Salim)", note: "Salim : Prétraitement CLAHE sur le canal L*, masques spectraux RGB/HSV/YCbCr et architecture U-Net double tête." }},
    {{ x: -2180, y: -1120, scale: 1.08, rot: 0,  title: "Salim — Performances FireUNet (99,11 %)", note: "Salim : Convergence dès l'époque 6 avec 99,11 % d'exactitude globale, 84,2 % de score Dice et sauvegarde en .pkl et .h5." }},
    {{ x: -2240, y: -320,  scale: 1.08, rot: 0,  title: "Salim — Segmentation & Extraction GPS", note: "Salim : Voici le résultat de predict(image_path) qui extrait le barycentre jaune, la surface touchée et les coordonnées GPS." }},
    {{ x: 0,     y: -350,  scale: 0.30, rot: 0,  title: "Flux Spatial : Vision → Météo Bi-LSTM", note: "Transition : Les coordonnées GPS détectées par Salim sont transmises instantanément au pôle météo de Moundir." }},
    {{ x: 1220,  y: -650,  scale: 1.12, rot: 2,  title: "Module 02 — Dynamique & Météo (Moundir)", note: "Moundir : Modélisation temporelle intégrant vent, humidité, température et indice FWI via un Bi-LSTM à attention." }},
    {{ x: 2200,  y: -1080, scale: 1.12, rot: 0,  title: "Moundir — Cercle Unité & Continuité 360°", note: "Moundir : Résolution mathématique de la discontinuité entre 359° et 1° par projection trigonométrique (sin θ, cos θ)." }},
    {{ x: 2240,  y: -250,  scale: 1.08, rot: 0,  title: "Moundir — Résultats Bi-LSTM (0,28 km/h)", note: "Moundir : Erreur vitesse réduite à 0,28 km/h et erreur angulaire à 5,72°, en moins de 3 ms par requête." }},
    {{ x: 400,   y: 250,   scale: 0.30, rot: 0,  title: "Flux Décisionnel : Propagation → Alerte", note: "Transition vers le troisième pôle : l'aide à la décision et l'interface web d'Asma." }},
    {{ x: 0,     y: 1080,  scale: 1.12, rot: 0,  title: "Module 03 — Arbre de Décision (Asma)", note: "Asma : Utilisation d'un arbre CART explicable avec court-circuit VERT immédiat lorsqu'aucun feu n'est détecté." }},
    {{ x: -980,  y: 1520,  scale: 1.10, rot: 0,  title: "Asma — Les 4 Paliers de Sécurité Civile", note: "Asma : Détail des 4 niveaux gradués : Vert, Orange, Rouge et Évacuation d'urgence." }},
    {{ x: 980,   y: 1520,  scale: 1.10, rot: 0,  title: "Asma & Équipe — Cockpit Streamlit & Futur", note: "Asma : Aperçu de l'interface Streamlit localhost:8501 et de nos 4 perspectives (Sentinel-2, MNT, Drones Jetson, Cloud CODIS)." }},
    {{ x: 0,     y: 140,   scale: 0.25, rot: 0,  title: "Conclusion — Merci & Démo Live !", note: "Conclusion (Équipe) : Merci de votre attention ! Passons maintenant à la démonstration en direct sur Streamlit." }}
  ];

  let current = 0;
  const world = document.getElementById('world');
  const label = document.getElementById('step-label');
  const pBar = document.getElementById('progress-bar');
  const notesContent = document.getElementById('notes-content');
  const notesDrawer = document.getElementById('notes-drawer');

  function renderCamera() {{
    const s = steps[current];
    // Adaptation responsive à la taille de l'écran/projecteur
    const fitFactor = Math.min(window.innerWidth / 1440, window.innerHeight / 860);
    const finalScale = s.scale * fitFactor * 1.12;
    world.style.transform = `rotate(${{s.rot}}deg) scale(${{finalScale}}) translate(${{-s.x}}px, ${{-s.y}}px)`;
    label.textContent = `${{String(current + 1).padStart(2, '0')}} / ${{steps.length}} — ${{s.title}}`;
    pBar.style.width = `${{((current + 1) / steps.length) * 100}}%`;
    notesContent.textContent = s.note;
  }}

  function nextStep() {{ current = (current + 1) % steps.length; renderCamera(); }}
  function prevStep() {{ current = (current - 1 + steps.length) % steps.length; renderCamera(); }}
  function goToStep(i) {{ current = i; renderCamera(); }}
  function toggleNotes() {{ notesDrawer.classList.toggle('visible'); }}
  function toggleFull() {{
    if (!document.fullscreenElement) document.documentElement.requestFullscreen();
    else document.exitFullscreen();
  }}

  window.addEventListener('resize', renderCamera);
  window.addEventListener('keydown', (e) => {{
    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {{ e.preventDefault(); nextStep(); }}
    else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {{ e.preventDefault(); prevStep(); }}
    else if (e.key === 'o' || e.key === 'O') goToStep(0);
    else if (e.key === 'n' || e.key === 'N') toggleNotes();
    else if (e.key === 'f' || e.key === 'F') toggleFull();
  }});

  renderCamera();
</script>
</body>
</html>
"""
    out_html = Path("Presentation_Prezi_Interactive.html")
    out_html.write_text(html, encoding="utf-8")
    return out_html.resolve()


if __name__ == "__main__":
    print("1/3 - Génération des graphiques haute résolution adaptés au thème sombre Prezi...")
    dark_charts = build_dark_prezi_charts()
    print("2/3 - Génération du PowerPoint Prezi-Morph (.pptx)...")
    pptx_file = build_prezi_pptx(dark_charts)
    print(f"      -> Sauvegardé : {pptx_file}")
    print("3/3 - Génération du Canevas Spatial Prezi Interactif HTML5 (.html)...")
    html_file = build_interactive_prezi_html(dark_charts)
    print(f"      -> Sauvegardé : {html_file}")
