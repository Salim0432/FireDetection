import joblib

# On charge mon modèle sauvegardé
arbre = joblib.load("models/mon_arbre_alerte.pkl")

def donner_alerte(feu, vent, humidite):
    # On demande à l'arbre de faire une prédiction
    prediction = arbre.predict([[feu, vent, humidite]])[0]
    
    # On traduit le chiffre (0, 1, 2 ou 3) en texte
    noms_alertes = {0: "🟢 VERT", 1: "🟠 ORANGE", 2: "🔴 ROUGE", 3: "🚨 ÉVACUATION"}
    return noms_alertes[prediction]