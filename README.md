## SOMMAIRE

- [DESCRIPTION DU PROJET](#1.-DESCRIPTION)
- [ETAPES DE REALISATION](#2.-ETAPES-DE-REALISATION)
- [LIBRAIRIES UTILISEES](#3.-LIBRAIRIES-UTILISEES)

## 1. DESCRIPTION

Cette étude se concentre sur l'attrition des employés chez IBM, visant à identifier les variables qui influencent la décision d'un employé de quitter l'entreprise. En utilisant les données fournies, nous chercherons à construire un modèle prédictif capable d'estimer avec précision la probabilité qu'un employé quitte son poste.

## 2. ETAPES DE REALISATION

### 2.1 Importation des librairies
Les packages de base ont été importés dans un premier temps, notamment Pandas, numpy, Matplolib, Seaborn.

### 2.2 Structure globale du travail
Pour une meilleure lisibilité du code principal, le fichier "attrition_utils.py" a été créé pour héberger toutes les fonctions nécessaires pour ce projet. Ces dernières sont y sont d'ailleurs décrites. Ainsi, lesdites fonctions sont appelées dans le fichier principal "Attrition_employees.ipynb" pour exécution.

### 2.3 Importation et structure du dataset
#### 2.3.1 Importation et aperçu
Le jeu de données est importé à travers la fonction implémentée à cet effet; l'importation est faite pour s'assurer de la réussite de l'opération.

#### 2.3.2 Structure du dataset
Cette partie nous permet d'en savoir un peu plus sur le contenu de la base de données, tout en sachant que la variable cible (Attrition) est connue d'avance par le dictionnaire des données.

#### 2.3.3 Résumé statistique
Calculs statistiques de base sur toutes les variables de type numérique du jeu de données.

### 2.4 EDA
A cette étape, plusieurs démarches ont été sollicitées pour faciliter l'analyse :
- Les statistiques univariées: Recueillir le nombre d'occurrences par variables de type catégoriel;
- Les statistiques bivariées : Nombre d'occurrences de chaque variable catégorielle en fonction des modalités de la variable cible;
- Les statistiques bivariées pour les variables  de type numérique : cela s'est fait par la matrice des corrélations et les boites à moustache.

### 2.5 Machine Learning
- Séparation des variables (explicatives et expliquée);
- Séparation en base d'entrainement, validation et test;
- Modélisation, évaluation de l'algorithme et choix du meilleur modèle.

### 2.6 Résultats obtenus
Le processus de modélisation a revélé, après détermination du meilleur modèle de Machine Learning et recherche des meilleurs hyperparamètres à adopter pour ledit modèle, que le SVM est meilleur que la Régression Logistique dans ce contexte et les hyperparamètres associés audit modèle sont  :
  - 'model__C': 0.1;
  - 'model__gamma': 0.01;
  - 'model__kernel': 'rbf'
La courbe ROC du meilleur modèle est ci-dessous illustrée :

![Results](https://github.com/guymartial80/Attrition_employes/blob/main/best_model_svm_optimized.png)

## 3. LIBRAIRIES UTILISEES
![Static Badge](https://img.shields.io/badge/Pandas-black?style=for-the-badge&logo=Pandas) ![Static Badge](https://img.shields.io/badge/Scikit-learn-black?style=for-the-badge&logo=Scikit-learn) ![Static Badge](https://img.shields.io/badge/Numpy-black?style=for-the-badge&logo=Numpy) ![Static Badge](https://img.shields.io/badge/Matplotlib-black?style=for-the-badge&logo=Matplotlib) ![Static Badge](https://img.shields.io/badge/Seaborn-black?style=for-the-badge&logo=Seaborn)


