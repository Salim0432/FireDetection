# main.py
from predict import predict
from predict_propagation import predict_propagation
from predict_alert import donner_alerte

def run_pipeline(image_path, wind_speed, wind_dir, humidity, temperature):
    print("--- 1. Analyse de l'image (Salim) ---")
    res_cnn = predict(image_path)
    
    # Si Salim dit qu'il n'y a pas de feu, on s'arrête là.
    if res_cnn["fire_detected"] == "Non":
        return "🟢 VERT (Aucun feu)"

    print("--- 2. Analyse de la propagation (Mondir) ---")
    res_lstm = predict_propagation(
        location=res_cnn["location"],
        wind={"speed": wind_speed, "direction": wind_dir},
        humidity=humidity,
        temperature=temperature
    )
    
    print("--- 3. Décision (asma) ---")
      # 1. On transforme la détection de Salim en 1 (Oui) ou 0 (Non)
    feu_binaire = 1 if res_cnn["fire_detected"] == "Oui" else 0
    
    # 2. ON APPELLE TON ARBRE DE DÉCISION ICI
    alerte = donner_alerte(feu_binaire, wind_speed, humidity)
    
    return alerte

# TEST
if __name__ == "__main__":
    print("Démarrage du test...")
    # On utilise une image du dossier
    image = "dataset/satellite_samples/images/fire/sat_fire_0001.png" 
    
    resultat = run_pipeline(image, 38.0, "Sud-Ouest", 22.0, 34.0)
    print("RÉSULTAT :", resultat)