import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# 1. On crée des données d'exemple pour que le modèle apprenne
# [Feu (1=oui, 0=non), Vent (km/h), Humidité (%)]
data = [
    [0, 10, 80], # Pas de feu -> Vert
    [0, 20, 50], # Pas de feu -> Vert
    [1, 10, 30], # Feu, vent faible -> Orange
    [1, 20, 40], # Feu, vent moyen -> Rouge
    [1, 50, 15], # Feu, vent fort, sec -> Evacuation
    [1, 80, 10], # Feu, tempête -> Evacuation
]
df = pd.DataFrame(data, columns=['feu', 'vent', 'humidite'])
# 0=Vert, 1=Orange, 2=Rouge, 3=Evacuation
labels = [0, 0, 1, 2, 3, 3] 

# 2. On entraîne l'arbre de décision
arbre = DecisionTreeClassifier()
arbre.fit(df, labels)

# 3. On sauvegarde le modèle 
joblib.dump(arbre, "models/mon_arbre_alerte.pkl")
print(" Modèle entraîné et sauvegardé !")