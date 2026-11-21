import streamlit as st
from main import run_pipeline
import os

# Configuration de la page
st.set_page_config(page_title="Alerte Incendie - Démo", page_icon="🔥", layout="wide")

st.title("🔥 Système Global de Détection et d'Alerte des Incendies")
st.markdown("**Projet 5 : Surveillance & Confinement Autonome des Feux de Forêt**")

# Création de deux colonnes : gauche pour l'image, droite pour la météo
col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Image Satellite")
    uploaded_file = st.file_uploader("Chargez une image aérienne ou satellite", type=["jpg", "jpeg", "png"])

with col2:
    st.header("2. Paramètres Météo")
    # On met la saisie météo bien en évidence sur la page
    wind_speed = st.slider("Vitesse du vent (km/h)", 0, 100, 38)
    wind_dir = st.selectbox("Direction du vent", ["Nord", "Sud", "Est", "Ouest", "Sud-Ouest", "Nord-Est"])
    humidity = st.slider("Humidité (%)", 0, 100, 22)
    temperature = st.slider("Température (°C)", 0, 50, 34)

st.markdown("---")

# Bouton d'analyse centré
if st.button("🚀 LANCER L'ANALYSE", use_container_width=True):
    if uploaded_file is not None:
        # Sauvegarde temporaire de l'image uploadée pour que Salim puisse l'analyser
        chemin_temp = "temp_image_upload.png"
        with open(chemin_temp, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Analyse de l'image par le CNN, calcul de la propagation par le LSTM et prise de décision..."):
            try:
                # APPEL DE TON PIPELINE COMPLET
                resultat = run_pipeline(
                    image_path=chemin_temp,
                    wind_speed=wind_speed,
                    wind_dir=wind_dir,
                    humidity=humidity,
                    temperature=temperature
                )
                
                # Affichage du résultat avec la bonne couleur
                st.subheader("Résultat du Système d'Alerte")
                
                if "VERT" in resultat:
                    st.success(f"**NIVEAU D'ALERTE : {resultat}**")
                elif "ORANGE" in resultat:
                    st.warning(f"**NIVEAU D'ALERTE : {resultat}**")
                else:
                    st.error(f"**NIVEAU D'ALERTE : {resultat}**")
                
                # Afficher un petit message d'explication
                st.info("Le niveau d'alerte a été calculé par l'Arbre de Décision en combinant la détection visuelle, la vitesse de propagation et la météo.")
                
            except Exception as e:
                st.error(f"Erreur lors de l'analyse : {e}")
            finally:
                # Nettoyage du fichier temporaire
                if os.path.exists(chemin_temp):
                    os.remove(chemin_temp)
    else:
        st.warning("⚠️ Veuillez d'abord uploader une image.")