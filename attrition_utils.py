import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import time

def importer_affichage_dataset(chemin_fichier):
    """
    Cette fonction importe un fichier Excel ou CSV et affiche les 5 premières lignes du DataFrame.
    
    Args:
    - chemin_fichier : Chemin vers le fichier à importer (Excel ou CSV).
    
    Returns:
    - df : DataFrame contenant les données du fichier.
    """
    # Vérifier l'extension du fichier pour déterminer le type de fichier
    if chemin_fichier.endswith('.xlsx'):
        # Importer un fichier Excel
        df = pd.read_excel(chemin_fichier)
    elif chemin_fichier.endswith('.csv'):
        # Importer un fichier CSV
        df = pd.read_csv(chemin_fichier)
    else:
        raise ValueError("Le fichier doit être au format Excel (.xlsx) ou CSV (.csv)")
    
    
    return df



def plot_categorical_column(df, column):
        """
        Fonction interne pour tracer le graphique de la variable catégorielle.
        """
        plt.figure(figsize=(5, 3))
        # Créer un countplot avec la variable catégorielle en x et Attrition en hue
        ax = sns.countplot(y=column, hue=column, data=df)

        # Ajouter les libellés sur l'axe des abscisses dans le sens vertical
        plt.xlabel('Effectif des employés par modalités')
        plt.ylabel(column)
        plt.title(f'Comptage des valeurs de {column}')
        plt.show()



# analyses univariés
def univariate_statistics(df):
    """
    Fonction pour afficher les comptages, pourcentages et graphiques pour chaque variable catégorielle d'un DataFrame.
    
    Args:
    - df: DataFrame pandas à analyser.
    """

    for column in df.columns:
        if df[column].dtype == object:
            # Compter les occurrences de chaque catégorie
            value_counts = df[column].value_counts()
            
            # Calculer les pourcentages
            percentages = (df[column].value_counts(normalize=True) * 100).round(2)
            
            # Créer un DataFrame à partir des comptages et pourcentages
            result = pd.concat([value_counts, percentages], axis=1)
            result.columns = ['Comptage', 'Pourcentage (%)']
            
            # Afficher les résultats
            print(f"Tableau de comptage des valeurs avec pourcentages pour la variable '{column}':\n")
            print(result)
            print("\n")
            
            # Tracer le graphique de la variable catégorielle
            plot_categorical_column(df, column)




def bivariate_statistics(df):
    """
    Fonction pour afficher des countplots bivariés pour chaque colonne catégorielle en fonction de 'Attrition'.
    
    Args:
    - df: DataFrame pandas contenant les données.
    """
    # Récupérer les noms des colonnes catégorielles
    categorical_columns = df.select_dtypes(include=['object']).columns
    # Exclure la variable 'Attrition' de la liste des colonnes catégorielles
    categorical_columns_except_attrition = categorical_columns.drop('Attrition') 
    
    # Pour chaque colonne catégorielle
    for column_name in categorical_columns_except_attrition:
        # ============================
        # Compter les occurrences de chaque catégorie pour Attrition = "Yes"
        value_counts_yes = df[df['Attrition'] == 'Yes'][column_name].value_counts()
        # Calculer les pourcentages pour Attrition = "Yes"
        percentages_yes = (df[df['Attrition'] == 'Yes'][column_name].value_counts(normalize=True) * 100).round(2)
        
        # Compter les occurrences de chaque catégorie pour Attrition = "No"
        value_counts_no = df[df['Attrition'] == 'No'][column_name].value_counts()
        # Calculer les pourcentages pour Attrition = "No"
        percentages_no = (df[df['Attrition'] == 'No'][column_name].value_counts(normalize=True) * 100).round(2)
        
        # Créer un DataFrame à partir des comptages et pourcentages pour Attrition = "Yes"
        result_yes = pd.concat([value_counts_yes, percentages_yes], axis=1)
        result_yes.columns = ['Attrition_Yes', 'Ratio_Yes']
        
        # Créer un DataFrame à partir des comptages et pourcentages pour Attrition = "No"
        result_no = pd.concat([value_counts_no, percentages_no], axis=1)
        result_no.columns = ['Attrition_No', 'Ration_No']
        
        # Fusionner les deux DataFrames sur les index (valeurs de catégories)
        result = pd.concat([result_yes, result_no], axis=1, sort=False)
        
        # Afficher les résultats
        print(f"Tableau de comptage des valeurs avec pourcentages pour la variable '{column_name}':\n")
        print(result)
        print("\n")
        
        # Tracer le graphique de la variable catégorielle en fonction de l'attrition
        plt.figure(figsize=(6, 4))
        # Créer un countplot avec la variable catégorielle en y et Attrition en hue
        ax = sns.countplot(y=column_name, hue="Attrition", data=df)

        # Afficher le titre et les étiquettes des axes
        plt.title('Histogramme des modalités de {}'.format(column_name))
        plt.xlabel('Effectif des employés')
        plt.ylabel(column_name)

        # Afficher la légende
        plt.legend(title='Attrition', loc='lower right')

        # Afficher le plot
        plt.show()



def categorical_variables(df):
    """
    Fonction pour sélectionner les variables de type catégoriel dans un DataFrame.
    
    Args:
    - df: DataFrame pandas contenant les variables à afficher.

    Returns:
    - Liste des libellés des variables de type catégoriel.
    """
    # Sélectionner les colonnes de type 'object'
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    return categorical_cols



def numerical_variables(df):
    """
    Fonction pour sélectionner les variables de type catégoriel dans un DataFrame.
    
    Args:
    - df: DataFrame pandas contenant les variables à afficher.

    Returns:
    - Liste des libellés des variables de type numérique.
    """
    # Sélectionner les colonnes de type 'object'
    numerical_cols = df.select_dtypes(include=['int64']).columns.tolist()
    return numerical_cols



def select_numeric_columns_corr(df):
    """
    Fonction pour sélectionner les variables de type numérique dans un DataFrame.
    
    Args:
    - df: DataFrame pandas contenant les variables à afficher.
    """
    numeric_columns = df.select_dtypes(include=['number']).columns
    return df[numeric_columns]



def plot_correlation_matrix(df):
    """
    Fonction pour afficher la matrice des corrélations, dans le but de déceler les liens entre les variables.
    
    Args:
    - df: DataFrame pandas contenant les variables à afficher.
    """
    plt.figure(figsize=(20,20))
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    sns.heatmap(df.corr(), mask=mask, center=0, cmap='RdBu', annot=True, fmt=".2f", vmin=-1, vmax=1)
    plt.title('Matrice des corrélations', fontsize = 18, fontweight = 'bold')
    plt.show()



def boxplot_numeric_variables(df):
    """
    Fonction pour afficher les boxplots des variables de type int64 dans un DataFrame.
    
    Args:
    - df: DataFrame pandas contenant les variables à afficher.
    """
    # Récupérer les variables avec le dtype 'int64'
    int64_columns = df.select_dtypes(include=['int64']).columns
    
    # Diviser les variables en blocs de 3 sur une ligne
    num_plots = len(int64_columns)
    num_rows = (num_plots + 2) // 3
    num_cols = min(3, num_plots)
    
    # Créer une nouvelle figure
    plt.figure(figsize=(15, 5*num_rows))
    
    # Afficher les boxplots pour chaque variable
    for i, column in enumerate(int64_columns):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.boxplot(data=df, x="Attrition", y=df[column], hue='Attrition')
        plt.title(column)
    
    # Ajuster l'espacement entre les sous-graphiques
    plt.tight_layout()
    
    # Afficher les graphiques
    plt.show()



def remove_outliers(df):
    """
    Fonction pour éliminer les outliers d'un DataFrame en les remplaçant par les limites inférieures (lower)
    et supérieures (upper) définies par la méthode de la zone interquartile (IQR).
    
    Args:
    - df: DataFrame pandas contenant les données.
    
    Returns:
    - DataFrame pandas avec les outliers remplacés par les limites inférieures et supérieures.
    """
    int64_columns = df.select_dtypes(include=['int64']).columns  # Sélectionner les colonnes de type int64
    
    # Pour chaque variable numérique
    for var in int64_columns:
        IQR = df[var].quantile(0.75) - df[var].quantile(0.25)
        lower = df[var].quantile(0.25) - (1.5 * IQR)
        upper = df[var].quantile(0.75) + (1.5 * IQR)
        
        # Remplacer les valeurs atypiques par les limites inférieures et supérieures
        df[var] = df[var].apply(lambda x: min(upper, max(x, lower)))
    
    boxplot_numeric_variables(df)



def modelling(X_train, y_train, X_val, y_val, X_test, y_test, numerical_variables, cat_vars):
    """
    Cette fonction affichera les métriques, la courbe ROC et la matrice de confusion pour chaque modèle, 
    puis sélectionnera le meilleur modèle en fonction de l'exactitude de la validation et du temps d'exécution.
    
    Args:
    - X_train et y_train: Base d'entrainement.
    - X_val et y_val: Base de validation.
    - X_test et y_test: Base de test.
    - numerical_variables: Variables de type numérique du dataset.
    - cat_vars: Variables de type catégoriel en dehors de la variable cible "Attritioon".
    
    Returns:
    - best_model: Nom du meilleur modèle pandas avec les outliers remplacés par s. best_model, model_acc, model_time
    - model_acc: Meilleur score
    - best_model: Meilleur temps d'exécution
    """
    models = [
        ('La Forêt Aléatoire', RandomForestClassifier()),
        ('La Regression Logistique', LogisticRegression())
    ]
    
    best_model = None
    best_accuracy = 0
    best_runtime = float('inf')
    model_acc = []
    model_time = []
    
    for model_name, model in models:
        start = time.time()
        
        # Création du pipeline avec le préprocesseur et le modèle
        pipeline = Pipeline(steps=[
            ('preprocessor', ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_variables),
                    ('cat', OneHotEncoder(), cat_vars)
                ]
            )),
            ('model', model)
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Prédictions
        y_pred = pipeline.predict(X_test)
        
        # Calcul et affichage des métriques
        print(f"Metriques pour {model_name}:")
        print(classification_report(y_test, y_pred))
        
        # Affichage de la matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(f'Matrice de Confusion pour {model_name}')
        plt.show()
        
        # Convertir les étiquettes en valeurs binaires
        y_test_binary = y_test.replace({"No": 0, "Yes": 1})
        # Récupération des scores liés aux prédictions
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        # Calcul et affichage de la courbe ROC
        fpr, tpr, _ = roc_curve(y_test_binary, y_prob)
        roc_auc = roc_auc_score(y_test_binary, y_prob)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic for {model_name}')
        plt.legend(loc="lower right")
        plt.show()
        
        # Mise à jour du meilleur modèle
        accuracy = accuracy_score(y_val, pipeline.predict(X_val))
        stop = time.time()
        runtime = stop - start
        if accuracy > best_accuracy or (accuracy == best_accuracy and runtime < best_runtime):
            best_model = pipeline
            best_accuracy = accuracy
            best_runtime = runtime
        
        # Stockage des performances
        model_acc.append(accuracy)
        model_time.append(runtime)
    
    # Affichage du meilleur modèle
    print("====================================")
    print("Best Model:", model_name)
    print("------------------------------------")
    print("Best Accuracy:", best_accuracy)
    print("------------------------------------")
    print("Best Runtime:", best_runtime)
    print("====================================")

    return best_model, model_acc, model_time

