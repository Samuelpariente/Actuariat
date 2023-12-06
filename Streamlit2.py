import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import random
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from scipy.stats import poisson

st.set_page_config(layout="wide")

def load_and_concat_csv(file_path1, file_path2):
    """
    Load two CSV files and concatenate them into a single DataFrame.

    Parameters:
    file_path1 (str): The path to the first CSV file.
    file_path2 (str): The path to the second CSV file.

    Returns:
    DataFrame: A pandas DataFrame containing the concatenated data from the two CSV files.
    """
    try:
        # Load the CSV files into DataFrames
        df1 = pd.read_csv(file_path1)
        df2 = pd.read_csv(file_path2)

        # Concatenate the DataFrames
        concatenated_df = pd.concat([df1, df2], ignore_index=True)

        print("Files successfully loaded and concatenated.")
        return concatenated_df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None




# Streamlit app layout
df_2022 = load_and_concat_csv("df2022_1.csv","df2022_2.csv")
catnat_gaspar =  pd.read_csv("catnat_gaspar_clean.csv")
catnat_gaspar['dat_deb'] = pd.to_datetime(catnat_gaspar['dat_deb'])
catnat_gaspar['dat_deb'] = pd.to_datetime(catnat_gaspar['dat_deb'])
data_dep = pd.read_csv("data_dep.csv")


def get_color_scale_by_risk_type(type_risque):
    # Définir des échelles de couleur pour chaque type de risque
    color_scales = {
        'Inondations': 'teal',
        'Mouvements de Terrain': 'Brwnyl',
        'Climatique': 'Redor',
        'Autre': 'Purp',
        # Ajoutez d'autres types de risque avec leurs échelles de couleur respectives
    }
    return color_scales[type_risque]


def plot_catastrophes_by_commune(type_risque, start_date=None):
   # Filtrer les données pour le type de risque spécifié et la date de début si elle est fournie

    # Convert 'start_date' from python date to pandas Timestamp
    if start_date:
        start_date = pd.Timestamp(start_date)

    # Now you can filter the data
    filtered_data = catnat_gaspar[catnat_gaspar['type_risque'] == type_risque]
    
    if start_date:
        filtered_data = filtered_data[filtered_data['dat_deb'] >= start_date]

    # Agréger les données par commune
    df_aggregated = filtered_data.groupby('cod_dep').size().reset_index(name='Nombre de Catastrophes')

    # Obtenir l'échelle de couleur en fonction du type de risque
    couleur_carte = get_color_scale_by_risk_type(type_risque)

    # Créer un DataFrame avec tous les départements
    all_departements = pd.DataFrame({'cod_dep': catnat_gaspar['cod_dep'].unique()})

    # Fusionner les données agrégées avec le DataFrame de tous les départements
    df_aggregated = all_departements.merge(df_aggregated, on='cod_dep', how='left')

    # Remplacer les valeurs manquantes par 0
    df_aggregated['Nombre de Catastrophes'] = df_aggregated['Nombre de Catastrophes'].fillna(0)


    # Créer la carte avec Plotly Express
    fig = px.choropleth(
        df_aggregated,
        geojson='https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson',
        featureidkey="properties.code",
        locations='cod_dep',
        color='Nombre de Catastrophes',

        color_continuous_scale=couleur_carte,  # Utiliser l'échelle de couleur spécifique
        range_color=(0, df_aggregated['Nombre de Catastrophes'].max()),
        template='ggplot2',
        title=f'<b>Nombre de catastrophes de type {type_risque}</b>'
    )

    # Centrer la carte sur la France
    fig.update_geos(
        center={"lat": 46.603354, "lon": 1.888334},
        projection_scale=20,
        visible=False, 
    )

    # Ajuster les dimensions de la carte
    fig.update_layout(
        width=1500,
        height=800,
        geo=dict(
            showframe=False,
            showcoastlines=False,
        )
    )

    # Afficher la carte
    return fig

def proba_cat_nat_annee_prochaine(cod_dep, catnat_gaspar):
    # Filtrer les catastrophes naturelles pour le département spécifié
    catnat_dep = catnat_gaspar[catnat_gaspar['cod_dep'] == float(cod_dep)]

    # Supprimer les doublons basés sur les colonnes 'dat_deb' et 'cod_dep'
    catnat_dep = catnat_dep.drop_duplicates(subset=['dat_deb', 'cod_dep']).reset_index(drop=True)


    # Convertir la colonne 'dat_deb' en type datetime
    catnat_dep['dat_deb'] = pd.to_datetime(catnat_dep['dat_deb'], errors='coerce')

    # Supprimer les lignes avec des dates invalides
    catnat_obs = catnat_dep.dropna(subset=['dat_deb'])

    # Calculer le nombre d'événements dans la période d'observation
    nb_evenements_obs = len(catnat_obs)
   
    if nb_evenements_obs == 0:
        moyenne_historique = 0
    else:
        moyenne_historique = nb_evenements_obs / 35 # durée en année d'observation

    # Modéliser la distribution de Poisson avec la moyenne historique
    distribution_poisson = poisson(moyenne_historique)

    # Calculer la probabilité d'observer au moins un événement dans l'année suivante
    proba_annee_prochaine = 1 - distribution_poisson.pmf(0)

    # Calculer l'intervalle de confiance de la probabilité
    intervalle_confiance = distribution_poisson.interval(0.95)

    return proba_annee_prochaine, intervalle_confiance

def boxplot_proba_cat_nat_annee_prochaine(cod_dep, catnat_gaspar):
    # Calculer la probabilité et l'intervalle de confiance pour le département spécifié
    proba_annee_prochaine, intervalle_confiance = proba_cat_nat_annee_prochaine(cod_dep, catnat_gaspar)

    # Créer un objet bar pour le graphique à barres
    bar = go.Figure()

    # Ajouter la barre pour la probabilité avec une barre d'erreur pour l'intervalle de confiance
    bar.add_trace(go.Bar(
        x=[proba_annee_prochaine],
        orientation='h',  # Orientation horizontale pour la boîte à moustaches
        name='Probabilité',
        error_x=dict(type='data', array=[proba_annee_prochaine - intervalle_confiance[0], intervalle_confiance[1] - proba_annee_prochaine], color='black'),
        width=0.4,  # Largeur des barres
        marker_color='black'  # Couleur de la boîte à moustaches
    ))

    # Mettre en forme le layout
    bar.update_layout(
        title=f'<b>Nombre de catastrophe naturelle dans le département {cod_dep} (1 an) avec Intervalle de Confiance</b>',
        xaxis=dict(title='Nombre de catastrophe naturelle'),  # Aucun titre pour l'axe x
        yaxis=dict(showline=False, showgrid=False, showticklabels=False),  # Axe y invisible
        height=400,  # Hauteur de la figure
        width=1000,  # Largeur de la figure
        plot_bgcolor='white'  # Couleur de fond
    )

    # Afficher le graphique à barres
    return bar


def plot_sales_and_disasters(start_date, end_date, cod_dep, catnat_gaspar):
    # Filtrer les données pour la région spécifiée
    filtered_df_2022 = df_2022[(df_2022['cod_dep'] == float(cod_dep))]
    # Créer le DataFrame avec toutes les dates de la période spécifiée
    dates_period = pd.date_range(start=start_date, end=end_date, freq='D')

    # Exclure tous les dimanches
    non_sundays = dates_period[dates_period.dayofweek != 6]

    graphique = pd.DataFrame({'Date': non_sundays})

    # Ajouter une colonne pour compter le nombre de transactions à chaque jour
    graphique['Ventes'] = 0

    filtered_df_2022['Date'] = pd.to_datetime(filtered_df_2022['Date'])
    
    # Créer la colonne 'Montant total' dans graphique en utilisant la colonne 'Date' convertie
    graphique['Montant total'] = (
        filtered_df_2022.groupby('Date')['Valeur fonciere'].sum().reindex(graphique['Date']).fillna(0).values
    )

    # Créer la colonne 'Surface terrain' dans graphique en utilisant la colonne 'Date' convertie
    graphique['Surface terrain'] = (
        filtered_df_2022.groupby('Date')['Surface terrain'].sum().reindex(graphique['Date']).fillna(0).values
    )

    graphique['Prix_m2_Moyenne_Mobile'] = (graphique['Montant total']/graphique['Surface terrain']).rolling(window=30, min_periods=1).mean()
    
    # Créer une base de données filtrée pour les catastrophes naturelles
    catnat_filtered = catnat_gaspar[
    (catnat_gaspar['cod_dep'] == float(cod_dep)) &
    (catnat_gaspar['dat_deb'] >= pd.to_datetime(start_date)) &
    (catnat_gaspar['dat_deb'] <= pd.to_datetime(end_date))]

    # Créer le graphique avec Plotly Express
    fig = px.line(graphique, x='Date', y='Prix_m2_Moyenne_Mobile', title=f'<b>Evolution du prix au m² - Département {cod_dep}</b>')

    # Ajouter une légende pour les traits noirs et la ligne bleue
    fig.add_trace(go.Scatter(x=[None], y=[None], line=dict(color='blue', width=2), name='Moyenne mobile 1 mois'))
    
    # Ajouter des traits verticaux noirs pour chaque catastrophe naturelle
    for _, row in catnat_filtered.iterrows():
        # Obtenez la valeur de graphique['Prix_m2_Moyenne_Mobile'] à la date 'dat_deb'


        fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=row['dat_deb'],
                x1=row['dat_deb'],
                y0=0,
                y1=graphique['Prix_m2_Moyenne_Mobile'].max(),
                line=dict(color="black", width=1),
            )
        )
     # Ajouter une annotation à la légende
    fig.add_trace(go.Scatter(x=[None], y=[None], line=dict(color='black', width=1), name='Catastrophe Naturelle'))
    
    # Mise en forme du thème seaborn
    fig.update_layout(
        height=600,
        width=1200,
        plot_bgcolor='white',  # Fond blanc
        xaxis=dict(title='Date', showline=True, linecolor='black'),  # Ajouter un titre pour l'axe x et une ligne noire
        yaxis=dict(title='Prix au m² (€)', showline=True, linecolor='black')  # Ajouter un titre pour l'axe y et une ligne noire
    )

    # Afficher le graphique
    return fig

def plot_pie_chart_by_catastrophe_type(cod_dep, data_dep):
    dep_data = data_dep[data_dep['cod_dep'] == cod_dep].copy()  # Utilisez .copy() pour éviter le SettingWithCopyWarning
    total = dep_data['Inondations'].sum() + dep_data['Mouvements de Terrain'].sum() + dep_data['Climatique'].sum() + dep_data['Autre'].sum()

    # Normaliser les montants en fonction du total
    dep_data.loc[:, 'Inondations'] = dep_data['Inondations'] / total * dep_data['montant']
    dep_data.loc[:, 'Mouvements de Terrain'] = dep_data['Mouvements de Terrain'] / total * dep_data['montant']
    dep_data.loc[:, 'Climatique'] = dep_data['Climatique'] / total * dep_data['montant']
    dep_data.loc[:, 'Autre'] = dep_data['Autre'] / total * dep_data['montant']


    # Créer un DataFrame pour le pie chart
    pie_data = dep_data[['Inondations', 'Mouvements de Terrain', 'Climatique', 'Autre']].sum().reset_index()
    pie_data.columns = ['Type de Catastrophe', 'Montant Accordé']

    # Définir des échelles de couleur pour chaque type de catastrophe
    color_scales = {
        'Inondations': 'teal',
        'Mouvements de Terrain': 'Brwnyl',
        'Climatique': 'Redor',
        'Autre': 'Purp',
    }

    # Créer le pie chart avec Plotly Express
    fig = px.pie(pie_data, names='Type de Catastrophe', values='Montant Accordé',
                 title=f"<b>Répartition du montant sous risque pour le Département {cod_dep}</b>",
                 color='Type de Catastrophe', color_discrete_map=color_scales)

    # Ajuster les dimensions de la figure
    fig.update_layout(width=600, height=400)

    # Afficher le pie chart
    return fig


# Définir une échelle de couleur pour chaque type de risque
couleurs_risque = {
    'Inondations': 'blue',
    'Mouvements de Terrain': 'brown',
    'Climatique': 'green',
    'Autre': 'purple',
}

def affichage_info_dep(code_dep, data_dep, catnat_gaspar, nb_m2):
    # Filtrer les données de catnat_gaspar pour le code de département spécifié
    filtered_catnat = catnat_gaspar[catnat_gaspar['cod_dep'] == float(code_dep)]

    # Filtrer les données de data_dep pour le code de département spécifié
    dep_data = data_dep[data_dep['cod_dep'] == float(code_dep)]

    # Nombre de catastrophes naturelles recensées pour chaque type
    nombre_catastrophes = dep_data[['Inondations', 'Mouvements de Terrain', 'Climatique', 'Autre']].sum().reset_index()
    nombre_catastrophes.columns = ['type_risque', 'Nombre de Catastrophes']

    # Montant à risque pour le département spécifié
    montant_risque_calculé = nb_m2 * dep_data['prix_moyen_m2'].values[0]

    print(f"Code Département: {code_dep}")
    print(f"Montant à Risque (calculé): {montant_risque_calculé} euros")

    # Frise chronologique des catastrophes naturelles
    fig_timeline = px.timeline(filtered_catnat, x_start='dat_deb', x_end='dat_fin', y='type_risque',
                               category_orders={'type_risque': ['Inondations', 'Mouvements de Terrain', 'Climatique', 'Autre']},
                               labels={'type_risque': 'Type de Catastrophe Naturelle'},
                               color='type_risque', color_discrete_map=couleurs_risque)
    fig_timeline.update_layout(width=1000, height=500, plot_bgcolor='white')

    # Barplot des sommes des colonnes 'Inondations', 'Mouvements de Terrain', 'Climatique', 'Autre'
    fig_barplot = px.bar(nombre_catastrophes, x='type_risque', y='Nombre de Catastrophes',
                        title=f"<b>Somme des Catastrophes par Type - Département {code_dep}</b>",
                        color='type_risque', color_discrete_map=couleurs_risque)
    fig_barplot.update_layout(width=800, height=500, plot_bgcolor='white')

    # Afficher la figure du barplot
    return fig_timeline, fig_barplot,code_dep,montant_risque_calculé




# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ('Page One', 'Page Two', 'Page Three'))

if page == 'Page One':
    # Streamlit app layout
    st.title('Visualisation des Catastrophes Naturelles')
    col1, col2 = st.columns(2)
    with col2:
        # Sidebar for user inputs
        st.markdown("##### Il existait une dizaine de type de catastrophes naturelles, nous avons choisist de les regrouper sous 4 catégories pour faciliter leur visualisation.   On propose également un approche simplifiée pour représenter le risque à venir d'une potentielle catastrophe naturelle du département concerné.  Notons que note approche reste sommaire (simple loi de poisson), compte tenu du sujet 'visualisation' (et non prédiction)")
        type_risque = st.selectbox('Choisir le type de risque', ['Inondations', 'Mouvements de Terrain', 'Climatique', 'Autre'])
        start_date = st.date_input('Choisir la date de début', value=pd.to_datetime('2022-01-01'))
        # Utilisation de la fonction
        box = boxplot_proba_cat_nat_annee_prochaine('75', catnat_gaspar)
        st.plotly_chart(box, use_container_width=True)
        
    with col1:
        fig = plot_catastrophes_by_commune(type_risque, start_date)
        st.plotly_chart(fig, use_container_width=True)
        # Display the list of catastrophes for the chosen risk type
    categories_risque = {
        'Inondations': ['Inondations et/ou Coulées de Boue', 'Inondations Remontée Nappe', 'Inondations par choc mécanique des vagues'],
        'Mouvements de Terrain': ['Mouvement de Terrain', 'Mouvements de terrain différentiels consécutifs à la sécheresse et à la réhydratation des sols', 'Mouvements de terrains (hors sécheresse géotechnique)', 'Glissement et Effondrement de Terrain', 'Glissement et Eboulement Rocheux'],
        'Climatique': ['Sécheresse', 'Chocs Mécaniques liés à l\'action des Vagues', 'Vents Cycloniques', 'Secousse Sismique', 'Tempête', 'Grêle', 'Effondrement et/ou Affaisement', 'Glissement de Terrain', 'Eboulement et/ou Chute de Blocs', 'Poids de la Neige', 'Lave Torrentielle', 'Coulée de Boue', 'Raz de Marée', 'Eruption Volcanique'],
        'Autre': ['Divers', 'Avalanche', 'Séismes']
    }
    # Construct a string with HTML for better formatting
    html_string = f"<h3>Liste des Catastrophes pour le Risque: {type_risque}</h3><ul>"
    for cat in categories_risque[type_risque]:
        html_string += f"<li>{cat}</li>"
    html_string += "</ul>"

    # Use markdown to display the HTML string
    st.markdown(html_string, unsafe_allow_html=True)


    # Call your function and display the plot
if page == 'Page Two':
    col1, col2 = st.columns(2)
    with col2:
        st.header("Configuration des filtres")
        col11, col12 = st.columns(2)
        with col11:
            start_date = st.date_input("Date de début", value=pd.to_datetime('2022-01-01'))
        with col12:
            end_date = st.date_input("Date de fin", value=pd.to_datetime('2022-12-31'))

        # Example list of region codes
        region_codes = list(catnat_gaspar['cod_dep'].unique())

        # Define the default region code
        default_region_code = 45.0  

        # Find the index of the default region code in the list
        default_index = region_codes.index(default_region_code)

        # Create the selectbox with the default selection
        cod_dep = st.selectbox("Code de région", region_codes, index=default_index)

        # Utilisation de la fonction avec une date de début, une date de fin et un code de région spécifiques
        fig = plot_sales_and_disasters(start_date,end_date, cod_dep, catnat_gaspar)
        st.plotly_chart(fig, use_container_width=True)
    with col1:
        # Déterminer les trois valeurs maximales à exclure
        max_values_to_exclude = data_dep['mesure1'].nlargest(10).values

        # Créer la carte avec Plotly Express pour les montants à assurer
        fig_data_dep = px.choropleth(
            data_dep,
            geojson='https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson',
            featureidkey="properties.code",
            locations='cod_dep',
            color='mesure1',  # Utiliser la colonne 'mesure1'
            
            color_continuous_scale='Peach',  # Utiliser l'échelle de couleur rouge
            range_color=(0, max_values_to_exclude.min()),  # Exclure les trois valeurs maximales
            template='ggplot2',
            title=f'<b>Montants à Assurer par Département</b>'
        )

        # Centrer la carte sur la France
        fig_data_dep.update_geos(
            center={"lat": 46.603354, "lon": 1.888334},
            projection_scale=20,
            visible=False
        )

        # Ajuster les dimensions de la carte
        fig_data_dep.update_layout(
            width=1500,
            height=800,
            geo=dict(
                showframe=False,
                showcoastlines=False,
            ),
            coloraxis_colorbar=dict(title='Montant')
        )

        # Afficher la carte
        st.plotly_chart(fig_data_dep, use_container_width=True)
        # Utilisation de la fonction avec un numéro de département spécifié
        pie = plot_pie_chart_by_catastrophe_type(cod_dep, data_dep)
        st.plotly_chart(pie, use_container_width=True)
        
if page == 'Page Three':
    col1, col2 = st.columns(2)
    with col1:
        # Example list of region codes
        region_codes = list(catnat_gaspar['cod_dep'].unique())

        # Define the default region code
        default_region_code = 45.0  

        # Find the index of the default region code in the list
        default_index = region_codes.index(default_region_code)
        nb_m2 = st.slider("Choisir m2", min_value=100, max_value=5000, value=1000, step=100)

        # Create the selectbox with the default selection
        cod_dep = st.selectbox("Code de région", region_codes, index=default_index)
        time,bar,c,m = affichage_info_dep(cod_dep, data_dep, catnat_gaspar, nb_m2)
        st.markdown(f"""
            <style>
                @keyframes shake {{
                    0% {{ transform: translate(1px, 1px) rotate(0deg); }}
                    10% {{ transform: translate(-1px, -2px) rotate(-1deg); }}
                    20% {{ transform: translate(-3px, 0px) rotate(1deg); }}
                    30% {{ transform: translate(3px, 2px) rotate(0deg); }}
                    40% {{ transform: translate(1px, -1px) rotate(1deg); }}
                    50% {{ transform: translate(-1px, 2px) rotate(-1deg); }}
                    60% {{ transform: translate(-3px, 1px) rotate(0deg); }}
                    70% {{ transform: translate(3px, 1px) rotate(-1deg); }}
                    80% {{ transform: translate(-1px, -1px) rotate(1deg); }}
                    90% {{ transform: translate(1px, 2px) rotate(0deg); }}
                    100% {{ transform: translate(1px, -2px) rotate(-1deg); }}
                }}

                @keyframes fadeIn {{
                    0% {{ opacity: 0; transform: scale(0.5); }}
                    100% {{ opacity: 1; transform: scale(1); }}
                }}

                .bottom-center-container {{
                    display: flex;
                    justify-content: center;
                    align-items: end;
                    height: 18vh;
                }}

                .animated-text {{
                    animation: fadeIn 2s ease, shake 3s ease;
                    font-size: 30px;
                    margin-bottom: 5px;
                    text-align: center;
                }}

                .info-label {{
                    color: #000000;
                    font-weight: bold;
                }}

                .info-value {{
                    color: #FF5722;
                }}
            </style>
            <div class="bottom-center-container">
                <div class="animated-text">
                    <span class="info-label">Montant à Risque (calculé):</span>
                    <span class="info-value">{int(m)} €</span>
                </div>
            </div>
        """, unsafe_allow_html=True)


    with col2:
        
        st.plotly_chart(bar, use_container_width=True)
    
    st.plotly_chart(time, use_container_width=True)
    
    