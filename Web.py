import streamlit as st
import sys
import os

# Tilføj sti så vi kan importere fra stats-mappen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import af statistikfunktioner
from stats import correlation
from stats import dispersion
from stats import central_tendency
from stats import visualization

# Forsidefunktion
def show_homepage():
    st.header(" Uddannelse Data Analysis")
    st.write("Velkommen til vores BI-analyseværktøj for uddannelse og frafald.")
    st.write("Brug menuen til venstre for at se grafer eller prøve en forudsigelsesmodel.")
    st.write("Problemformulering: "+
             " Hvordan kan machine learning og geografisk dataanalyse anvendes til at identificere "+
            " og forudsige frafaldsmønstre blandt studerende på videregående uddannelser i Danmark " +
             "– med fokus på fagretninger, institutionstyper og geografi " +
            " – og hvordan kan denne indsigt bruges til at understøtte beslutningstagning og målrettede indsatser?")

# Hovedfunktion
def main():
    st.set_page_config(page_title="Uddannelse BI", layout="wide")
    st.sidebar.title("Navigation")

    page = st.sidebar.selectbox("Vælg en side", ["Homepage", "Visualization", "Prediction", "Institutioner", "Kortvisning", "Konklusion"])

    try:
        if page == "Homepage":
            show_homepage()
        elif page == "Visualization":
            visualization.show_graphs()
        elif page == "Prediction":
            visualization.show_prediction_model()
            visualization.show_uddannelse_prediction_model()
        elif page == "Institutioner":
            visualization.show_graphsInstitutionerSelvValgt()
            visualization.show_institution_clustering()
            visualization.show_feature_importance()

        elif page == "Kortvisning":
            visualization.show_map_institution()
        elif page == "Konklusion":
            visualization.render_conclusion_page()
    except Exception as e:
        st.error(f" Fejl under visning af siden: {e}")

if __name__ == "__main__":
    main()
