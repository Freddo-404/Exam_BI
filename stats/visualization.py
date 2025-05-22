import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# Funktion: Histogram
def histogram(data, column_name, title="Histogram"):
    fig, ax = plt.subplots()
    ax.hist(data[column_name], bins=10, edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel(column_name)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# Funktion: Boxplot
def boxplot(data, column_name, title="Box Plot"):
    fig, ax = plt.subplots()
    sns.boxplot(y=data[column_name], ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

# Funktion: Scatter plot (2D)
def scatter_plot(data, x_column, y_column, title="Scatter Plot"):
    fig, ax = plt.subplots()
    ax.scatter(data[x_column], data[y_column])
    ax.set_title(title)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    st.pyplot(fig)

# Funktion: Scatter plot (3D)
def scatter_plot_3d(data, x_column, y_column, z_column, title="3D Scatter Plot", color_column=None):
    st.subheader(title)
    fig = px.scatter_3d(
        data,
        x=x_column,
        y=y_column,
        z=z_column,
        color=color_column,
        opacity=0.7
    )
    fig.update_layout(title=title)
    st.plotly_chart(fig)
    
    


def show_graphsInstitutioner():
    st.header("Visualisering af data")

    # Læs data
    data = pd.read_excel("Streamlit/Data/Afbrudte_og_fuldførte_institution.xlsx")

    # Rens data (fjerner 'Hovedinstitution' m.m.)
    data = data[~data["Institution"].isin(["Institution", "HovedInstitutionTx", "Hovedinstitution"])]
    data = data.sort_values("Fuldførte", ascending=False)
    

   
    st.subheader("Antal fuldførte pr. institution")
    fig1 = px.bar(
        data,
        x="InstitutionType",
        y="Fuldførte",
        title="Fuldførte pr. InstitutionType",
        hover_name="InstitutionType"
    )
    fig1.update_layout(xaxis={'visible': False}, width=1200, height=500)  # Skjul labels, bred grafik
    st.plotly_chart(fig1)

    data_sorted_afbrudte = data.sort_values("Afbrudte", ascending=False)
    st.subheader("Antal afbrudte pr. institution")
    fig2 = px.bar(
        data_sorted_afbrudte,
        x="InstitutionType",
        y="Afbrudte",
        title="Afbrudte pr. InstitutionType",
        hover_name="InstitutionType"
    )
    fig2.update_layout(xaxis={'visible': False}, width=1200, height=500)
    st.plotly_chart(fig2)






# VISUALISERING: Faglinje og grafer
def show_graphs():
    # Find sti til data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    excel_path = os.path.join(base_dir, 'Streamlit', 'Data', 'Uddannelse_combined.xlsx')

    st.header("Uddannelse Data Visualization")

    # Indlæs data
    data = pd.read_excel(excel_path)
    data.columns = data.columns.map(str)  # alle kolonnenavne som strenge

    # Konverter årskolonner til numerisk
    year_cols = [str(y) for y in range(2015, 2025)]
    data[year_cols] = data[year_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    # ✅ FagLinje dropdown + afbrudt/fuldført bar chart
    faglinje_valg = st.selectbox("Vælg en FagLinje", sorted(data["FagLinjer"].dropna().unique()))
    filtered = data[data["FagLinjer"] == faglinje_valg]

    if filtered.empty:
        st.warning("Ingen data fundet for den valgte faglinje.")
    else:
        grouped = filtered.groupby("Type")[year_cols].sum().transpose()
        st.subheader(f"Fuldført vs. Afbrudt for: {faglinje_valg}")
        st.bar_chart(grouped)
        st.dataframe(grouped)

    st.divider()
    st.subheader("Andre visualiseringer")

    # ✅ Tidligere udkommenterede blokke er bevaret herunder:

    # if all(col in data.columns for col in ['alcohol', 'chlorides', 'quality']):
    #     scatter_plot_3d(data, 'alcohol', 'chlorides', 'quality', title="Alcohol vs Chlorides vs Quality")

    # if 'Type' in data.columns:
    #     histogram(data, 'Type', title="Alcohol Content Distribution")

    # if 'alcohol' in data.columns and 'quality' in data.columns:
    #     scatter_plot(data, 'alcohol', 'quality', title="Alcohol vs Quality")

    # 📊 Faktiske plots baseret på dine data:
    if all(col in data.columns for col in ['Type', '2015']):
        histogram(data, 'Type', title="Antallet af afbrudte og fuldførte")
        boxplot(data, '2015', title="Education Levels in 2015")

    if all(col in data.columns for col in ['Type', '2020']):
        boxplot(data, '2020', title="Education Levels in 2020")

    if all(col in data.columns for col in ['Type', '2023']):
        boxplot(data, '2023', title="Education Levels in 2023")

    if all(col in data.columns for col in ['2015', '2020', '2023', 'Type']):
        scatter_plot_3d(data, '2015', '2020', '2023', title="2015-2023 Trends Colored by Type", color_column='Type')




