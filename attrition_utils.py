import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def count_and_percentage(df, column_name):
    # Comptage des valeurs
    value_counts = df[column_name].value_counts()

    # Calcul des pourcentages
    percentages = (df[column_name].value_counts(normalize=True) * 100).round(2)

    # Création d'un DataFrame à partir des séries
    result = pd.concat([value_counts, percentages], axis=1)

    # Renommer les colonnes
    result.columns = ['Comptage', 'Pourcentage (%)']

    # Affichage du résultat
    print("PROPORTION DES DONNEES DE LA VARIABLE '{}' :".format(column_name))
    print('======================================================')
    print(result)
    print('======================================================\n\n')


def plot_histogram_by_category(df, column_name):
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
    # Sélectionner les colonnes de type 'object'
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Afficher la liste des variables catégorielles
    print("Liste des variables catégorielles :")
    for col in categorical_cols:
        print("- {}".format(col))



def select_numeric_columns(df):
    numeric_columns = df.select_dtypes(include=['number']).columns
    return df[numeric_columns]



def plot_correlation_matrix(df):
    plt.figure(figsize=(20,20))
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    sns.heatmap(df.corr(), mask=mask, center=0, cmap='RdBu', annot=True, fmt=".2f", vmin=-1, vmax=1)
    plt.title('Matrice des corrélations', fontsize = 18, fontweight = 'bold')
    plt.show()