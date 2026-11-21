# Rapport Technique — Module Système d'Alerte & Intégration

**Responsable** : Asma
**Rôle** : Responsable Système d'Alerte & Intégration (Arbre de Décision & Pipeline)

---

## 1. Introduction
La surveillance des feux de forêt est un enjeu environnemental et sécuritaire majeur. Ce projet propose une architecture multi-agents basée sur le Deep Learning pour automatiser la détection et la prédiction de la propagation des incendies. Notre système se divise en trois modules complémentaires : l'analyse visuelle par Vision par Ordinateur (CNN), la modélisation temporelle et météorologique (LSTM), et un système expert d'aide à la décision. L'objectif est de fournir une alerte précoce et fiable pour faciliter le confinement du feu et l'évacuation des zones à risque.

## 2. Système d'Alerte & Intégration
Le système d'alerte représente le cerveau décisionnel de notre architecture. Il repose sur un pipeline d'intégration développé en Python (`main.py`). 

Dans un premier temps, il récupère les informations spatiales (présence du feu, zone touchée) fournies par le modèle CNN de Salim. Si un feu est détecté, il interroge le modèle LSTM de Mondir pour obtenir la vitesse et la direction de propagation en fonction des données météorologiques en temps réel (vent, humidité, température). 

Ces données sont ensuite injectées dans un **Arbre de Décision** (`DecisionTreeClassifier` de scikit-learn) préalablement entraîné. Ce modèle interprète les risques et déclenche l'un des quatre paliers d'alerte :
*   🟢 **Vert** : Aucun danger détecté.
*   🟠 **Orange** : Vigilance, feu présent mais risque de propagation faible.
*   🔴 **Rouge** : Danger élevé, propagation rapide vers des zones sensibles.
*   🚨 **Évacuation** : Danger imminent, conditions météorologiques extrêmes.

Enfin, pour faciliter la prise de décision des autorités, ce pipeline est relié à une interface web développée avec **Streamlit**, permettant d'uploader une image satellite en temps réel et de visualiser instantanément le niveau d'alerte.

## 3. Conclusion
Ce projet a permis de mettre en place un système complet et modulaire pour la gestion des incendies de forêt. L'intégration des modèles CNN et LSTM a démontré de très bonnes performances. Notre Arbre de Décision a prouvé son efficacité pour transformer ces prédictions complexes en actions claires. L'interface Streamlit rend ce système directement exploitable. À l'avenir, ce travail pourrait être enrichi par l'utilisation de flux satellites en direct et un déploiement sur le cloud.