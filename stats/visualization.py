import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def histogram(data, column_name, title="Histogram"):
    fig, ax = plt.subplots()
    ax.hist(data[column_name], bins=10, edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel(column_name)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

def boxplot(data, column_name, title="Box Plot"):
    fig, ax = plt.subplots()
    sns.boxplot(y=data[column_name], ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

def scatter_plot(data, x_column, y_column, title="Scatter Plot"):
    fig, ax = plt.subplots()
    ax.scatter(data[x_column], data[y_column])
    ax.set_title(title)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    st.pyplot(fig)

def scatter_plot_3d(data, x_column, y_column, z_column, title="3D Scatter Plot", color_column=None):
    st.subheader(title)
    fig = px.scatter_3d(data, x=x_column, y=y_column, z=z_column, color=color_column, opacity=0.7)
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
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    excel_path = os.path.join(base_dir, 'Streamlit', 'Data', 'Uddannelse_combined.xlsx')

    st.header("Uddannelse Data Visualization")

    data = pd.read_excel(excel_path)
    data.columns = data.columns.map(str)
    year_cols = [str(y) for y in range(2015, 2025)]
    data[year_cols] = data[year_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

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

    if all(col in data.columns for col in ['Type', '2015']):
        histogram(data, 'Type', title="Antallet af afbrudte og fuldførte")
        boxplot(data, '2015', title="Uddannelsesniveau i 2015")

    if all(col in data.columns for col in ['Type', '2020']):
        boxplot(data, '2020', title="Uddannelsesniveau i 2020")

    if all(col in data.columns for col in ['Type', '2023']):
        boxplot(data, '2023', title="Uddannelsesniveau i 2023")

    if all(col in data.columns for col in ['2015', '2020', '2023', 'Type']):
        scatter_plot_3d(data, '2015', '2020', '2023', title="2015-2023 Trends Colored by Type", color_column='Type')

def show_prediction_model():
    st.header("Forudsig frafald i 2024 og 2025 med lineær regression")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    excel_path = os.path.join(base_dir, 'Streamlit', 'Data', 'Uddannelse_combined.xlsx')
    df = pd.read_excel(excel_path)
    df.columns = df.columns.map(str)

    all_years = [str(y) for y in range(2015, 2025)]
    df[all_years] = df[all_years].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Filtrér kun "Afbrudt" rækker til modellen
    df_afbrudt = df[df["Type"] == "Afbrudt"].copy()

    # Model for 2024
    feature_cols_2024 = [str(y) for y in range(2015, 2024)]
    X_2024 = df_afbrudt[feature_cols_2024]
    y_2024 = df_afbrudt["2024"]

    model_2024 = LinearRegression()
    model_2024.fit(X_2024, y_2024)
    y_pred_2024 = model_2024.predict(X_2024)

    # Model for 2025 (forudsigelse baseret på tidligere år)
    feature_cols_2025 = [str(y) for y in range(2016, 2025)]
    X_2025 = df_afbrudt[feature_cols_2025]
    model_2025 = LinearRegression()
    model_2025.fit(X_2025, df_afbrudt["2024"])
    y_pred_2025 = model_2025.predict(X_2025)

    # Saml resultater
    df_vis = df_afbrudt[["Uddannelse", "FagLinjer", "FagRetning"]].copy()
    df_vis["2024_forudsagt"] = y_pred_2024
    df_vis["2024_faktisk"] = y_2024
    df_vis["Forskel"] = df_vis["2024_forudsagt"] - df_vis["2024_faktisk"]
    df_vis["2025_forudsagt"] = y_pred_2025

    # Total antal studerende fra både Afbrudt og Fuldført
    fuldfoert_2024 = df[df["Type"] == "Fuldført"]
    afbrudt_2024 = df[df["Type"] == "Afbrudt"]

    sum_fuldført = fuldfoert_2024.groupby(["Uddannelse", "FagLinjer", "FagRetning"])["2024"].sum().reset_index()
    sum_afbrudt = afbrudt_2024.groupby(["Uddannelse", "FagLinjer", "FagRetning"])["2024"].sum().reset_index()

    merged = pd.merge(sum_fuldført, sum_afbrudt, on=["Uddannelse", "FagLinjer", "FagRetning"], suffixes=("_fuldført", "_afbrudt"))
    merged["total_2024"] = merged["2024_fuldført"] + merged["2024_afbrudt"]

    # Slå totals sammen med forudsigelser
    df_vis = pd.merge(df_vis, merged[["Uddannelse", "FagLinjer", "FagRetning", "total_2024"]], on=["Uddannelse", "FagLinjer", "FagRetning"], how="left")
    df_vis["Frafaldsprocent_2025"] = df_vis["2025_forudsagt"] / df_vis["total_2024"]

    # Vis resultater
    st.subheader("Tabel med forudsagte og faktiske værdier (kun afbrudt)")
    st.dataframe(df_vis)

    st.subheader("Top 20 fagretninger – forudsagt frafald i 2025")
    top20_antal = df_vis.sort_values("2025_forudsagt", ascending=False).head(20)
    st.bar_chart(top20_antal.set_index("FagRetning")["2025_forudsagt"])

    st.subheader("Top 20 fagretninger – forudsagt frafaldsprocent i 2025")
    top20_procent = df_vis.sort_values("Frafaldsprocent_2025", ascending=False).head(20)
    st.bar_chart(top20_procent.set_index("FagRetning")["Frafaldsprocent_2025"])
