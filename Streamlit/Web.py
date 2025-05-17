# Web.py
import streamlit as st
import sys
import os
import wikipediaapi

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stats import correlation
from stats import dispersion
from stats import central_tendency
from stats import visualization

# Function to display the homepage content
def show_homepage():
    st.header("Uddannelse Data Analysis")

# Main function that runs the app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Homepage", "Visualization"])

    if page == "Homepage":
        show_homepage()
    elif page == "Visualization":
        visualization.show_graphs()

if __name__ == "__main__":
    main()
