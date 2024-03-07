import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import time


# analyses univariés
def univariate_statistics(df):
    """
    Fonction pour afficher les comptages, pourcentages et graphiques pour chaque variable catégorielle d'un DataFrame.
    
    Args:
    - df: DataFrame pandas à analyser.
    """
    def plot_categorical_column(df, column):
        """
        Fonction interne pour tracer le graphique de la variable catégorielle.
        """
        plt.figure(figsize=(5, 3))
        # Créer un countplot avec la variable catégorielle en x et Attrition en hue
        ax = sns.countplot(x=column, hue=column, data=df)

        # Ajouter les libellés sur l'axe des abscisses dans le sens vertical
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
        plt.xlabel(column)
        plt.ylabel('Nombre d\'occurrences')
        plt.title(f'Comptage des valeurs de {column}')
        plt.show()

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
    
    # Pour chaque colonne catégorielle
    for column_name in categorical_columns:
        plt.figure(figsize=(5, 3))
        # Créer un countplot avec la variable catégorielle en x et Attrition en hue
        ax = sns.countplot(x=column_name, hue="Attrition", data=df)

        # Ajouter les libellés sur l'axe des abscisses dans le sens vertical
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
        
        # Ajouter des annotations à chaque barre de l'histogramme
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 5), 
                        textcoords = 'offset points')

        # Afficher le titre et les étiquettes des axes
        plt.title('Histogramme des modalités de {}'.format(column_name))
        plt.xlabel(column_name)
        plt.ylabel('Nombre d\'employés')

        # Afficher la légende
        plt.legend(title='Attrition', loc='upper right')

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



def select_numeric_columns(df):
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
        sns.boxplot(data=df, y="Attrition", x=df[column], hue='Attrition')
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
    
    return df



def modelling(X_train, y_train, X_val, y_val, X_test, y_test, numerical_variables, cat_vars):
    models = [
        ('La Regression Logistique', LogisticRegression()),
        ('La Forêt Aléatoire', RandomForestClassifier())
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
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        # Calcul et affichage des métriques
        print(f"Metriques pour {model_name}:")
        print(classification_report(y_test, y_pred))
        
        # Affichage de la matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(f'Matrice de Confusion pour {model_name}')
        plt.show()
        
        # Convertir les étiquettes en valeurs binaires
        y_test_binary = y_test.replace({"No": 0, "Yes": 1})
        
        # Calcul et affichage de la courbe ROC
        fpr, tpr, _ = roc_curve(y_test_binary, y_prob)
        roc_auc = roc_auc_score(y_test_binary, y_prob)
        plt.figure()
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
    print("================================")
    print("Best Model:", model_name)
    print("================================")
    
    return best_model, model_acc, model_time

