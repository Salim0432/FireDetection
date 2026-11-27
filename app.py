import os
from pathlib import Path

import streamlit as st

from contracts import CARDINAL_DIRECTIONS_8
from main import run_pipeline

# Configuration de la page
st.set_page_config(page_title="Alerte Incendie - Projet 5", page_icon="🔥", layout="wide")

st.title("🔥 Système Multi-Agents de Surveillance et d'Alerte des Feux de Forêt")
st.markdown(
    "**Projet 5 — Vision CNN/U-Net (Salim) • Propagation Bi-LSTM (Moundir) • Arbre de Décision (Asma)**"
)

col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Acquisition Image Satellite")
    uploaded_file = st.file_uploader(
        "Chargez une image aérienne ou satellite (.jpg, .jpeg, .png)",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Image satellite chargée", use_container_width=True)

with col2:
    st.header("2. Relevés Météorologiques Temps Réel")
    wind_speed = st.slider("Vitesse du vent (km/h)", 0, 100, 38)
    wind_dir = st.selectbox(
        "Direction du vent",
        list(CARDINAL_DIRECTIONS_8),
        index=list(CARDINAL_DIRECTIONS_8).index("Sud-Ouest"),
    )
    humidity = st.slider("Humidité relative (%)", 0, 100, 22)
    temperature = st.slider("Température ambiante (°C)", 0, 50, 34)

st.markdown("---")

if st.button("🚀 LANCER L'ANALYSE MULTI-AGENTS", use_container_width=True):
    if uploaded_file is not None:
        chemin_temp = "temp_image_upload.png"
        chemin_vis = "temp_image_segmented.png"
        with open(chemin_temp, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Exécution du pipeline : CNN (Salim) ➔ Bi-LSTM (Moundir) ➔ Arbre de Décision (Asma)..."):
            try:
                rapport = run_pipeline(
                    image_path=chemin_temp,
                    wind_speed=float(wind_speed),
                    wind_dir=wind_dir,
                    humidity=float(humidity),
                    temperature=float(temperature),
                    save_visualization=chemin_vis,
                    return_details=True,
                )

                resultat = rapport["alert"]
                recommandation = rapport["recommendation"]
                res_cnn = rapport["cnn"]
                res_lstm = rapport["lstm"]
                res_dec = rapport["decision"]

                st.subheader("3. Verdict du Système d'Alerte (Agent 3 — Asma)")
                if "VERT" in resultat:
                    st.success(f"**PALIER D'ALERTE : {resultat}**\n\n📋 **Action recommandée :** {recommandation}")
                elif "ORANGE" in resultat:
                    st.warning(f"**PALIER D'ALERTE : {resultat}**\n\n📋 **Action recommandée :** {recommandation}")
                else:
                    st.error(f"**PALIER D'ALERTE : {resultat}**\n\n📋 **Action recommandée :** {recommandation}")

                pan_cnn, pan_lstm = st.columns([1, 1])

                with pan_cnn:
                    st.markdown("### 🛰️ Agent 1 — Vision & Segmentation (Salim)")
                    if Path(chemin_vis).is_file():
                        st.image(chemin_vis, caption="Segmentation spatiale U-Net", use_container_width=True)
                    loc = res_cnn["location"]
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Feu détecté", res_cnn["fire_detected"])
                    m2.metric("Confiance CNN", f"{res_cnn['confidence'] * 100:.1f}%")
                    m3.metric("Surface touchée", f"{loc['affected_area_pct']:.2f}%")
                    st.caption(
                        f"**Secteur :** `{loc['zone']}` | **Barycentre :** `{loc['center_xy']}` | "
                        f"**GPS estimé :** `{loc['gps_coords']}`"
                    )

                with pan_lstm:
                    st.markdown("### 🌪️ Agent 2 — Dynamique & Propagation (Moundir)")
                    if res_lstm is not None:
                        l1, l2, l3 = st.columns(3)
                        l1.metric("Direction prédite", res_lstm["propagation_direction"])
                        l2.metric("Vitesse (RoS)", f"{res_lstm['propagation_speed']:.2f} km/h")
                        l3.metric("Indice FWI", f"{res_lstm['fire_weather_index']:.2f}")

                        exp = res_lstm["expansion_next_hours"]
                        st.markdown("**Projection d'expansion horaire (Bi-LSTM + Attention) :**")
                        h1, h2, h3 = st.columns(3)
                        h1.metric("Horizon +1h", f"{exp['+1h']:.2f} km/h")
                        h2.metric("Horizon +2h", f"{exp['+2h']:.2f} km/h")
                        h3.metric("Horizon +3h", f"{exp['+3h']:.2f} km/h")
                        st.caption(
                            f"**Angle continu :** `{res_lstm['direction_angle_deg']}°` | "
                            f"**Vitesse linéaire :** `{res_lstm['speed_m_per_min']} m/min`"
                        )
                    else:
                        st.info("Aucun foyer actif détecté par le CNN — modélisation de propagation non requise.")

                st.caption(
                    f"✅ Contrat multi-agents validé | Vecteur décisionnel transmis à l'Arbre : `{res_dec['inputs_used']}`"
                )

            except Exception as e:
                st.error(f"Erreur lors de l'analyse : {e}")
            finally:
                for p_tmp in (chemin_temp, chemin_vis):
                    if os.path.exists(p_tmp):
                        os.remove(p_tmp)
    else:
        st.warning("⚠️ Veuillez d'abord charger une image satellite ou aérienne.")
