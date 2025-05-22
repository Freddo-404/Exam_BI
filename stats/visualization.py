import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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

# Funktion: Hovedvisualisering
def show_graphs():
    st.title("Analyse af Frafald på Videregående Uddannelser")

    # Indlæs data
    file_path = "Data/Uddannelse_combined.xlsx"
    df = pd.read_excel(file_path)
    years = list(range(2015, 2025))

    # Split data
    fuldfort = df[df['Type'] == 'Fuldført']
    afbrudt = df[df['Type'] == 'Afbrudt']

    # Aggreger på FagLinje og FagRetning
    agg_fuldfort = fuldfort.groupby(['FagLinjer','FagRetning'])[years].sum().reset_index()
    agg_afbrudt = afbrudt.groupby(['FagLinjer','FagRetning'])[years].sum().reset_index()

    # Merge og beregn frafaldsrate
    ret_merged = pd.merge(agg_fuldfort, agg_afbrudt, on=['FagLinjer','FagRetning'], suffixes=('_fuldfort', '_afbrudt'))
    ret_merged['Total_fuldfort'] = ret_merged[[f"{y}_fuldfort" for y in years]].sum(axis=1)
    ret_merged['Total_afbrudt'] = ret_merged[[f"{y}_afbrudt" for y in years]].sum(axis=1)
    ret_merged['Frafaldsrate'] = 100 * ret_merged['Total_afbrudt'] / (ret_merged['Total_fuldfort'] + ret_merged['Total_afbrudt'])

    # Første valg: FagLinje
    st.header("Trin 1: Vælg en FagLinje")
    alle_linjer = sorted(ret_merged['FagLinjer'].unique())
    valgt_linje = st.selectbox("Vælg en FagLinje", alle_linjer)

    linje_data = ret_merged[ret_merged['FagLinjer'] == valgt_linje]

    if linje_data.empty:
        st.warning("Ingen data fundet for den valgte FagLinje.")
        return

    # Tilføj bar chart (Fuldført vs. Afbrudt for valgt FagLinje)
    grouped_linje = df[df['FagLinjer'] == valgt_linje].groupby("Type")[years].sum().transpose()
    st.subheader(f"Fuldført vs. Afbrudt for: {valgt_linje}")
    st.bar_chart(grouped_linje)
    st.dataframe(grouped_linje)

    st.divider()

    # Vis FagRetninger med frafaldsrate
    st.subheader(f"Frafaldsrate for FagRetninger under {valgt_linje}")
    st.dataframe(linje_data[['FagRetning', 'Frafaldsrate']].sort_values(by="Frafaldsrate", ascending=False))

    # Andet valg: FagRetning
    st.header("Trin 2: Vælg en FagRetning under den valgte FagLinje")
    retninger = linje_data['FagRetning'].unique()
    valgt_retning = st.selectbox("Vælg en FagRetning", retninger)

    valgte_data = linje_data[linje_data['FagRetning'] == valgt_retning]

    if valgte_data.empty:
        st.warning("Ingen data for den valgte kombination.")
        return

    row = valgte_data.iloc[0]
    fuldførte = row[[f"{y}_fuldfort" for y in years]].values
    afbrudte = row[[f"{y}_afbrudt" for y in years]].values
    total = fuldførte + afbrudte
    frafaldsrate = 100 * afbrudte / total

    st.subheader(f"Tidsserie for {valgt_retning} under {valgt_linje}")
    st.line_chart(pd.DataFrame({
        "Fuldført": fuldførte,
        "Afbrudt": afbrudte
    }, index=years))

    st.line_chart(pd.DataFrame({
        "Frafaldsrate (%)": frafaldsrate
    }, index=years))
    
    # Ekstra plots
    st.subheader("Fordelingsanalyser")
    if 'Type' in df.columns:
        histogram(df, 'Type', title="Antallet af afbrudte og fuldførte")

    if '2015' in df.columns:
        boxplot(df, '2015', title="Niveau i 2015")
    if '2020' in df.columns:
        boxplot(df, '2020', title="Niveau i 2020")
    if '2023' in df.columns:
        boxplot(df, '2023', title="Niveau i 2023")

    if all(col in df.columns for col in ['2015', '2020', '2023', 'Type']):
        scatter_plot_3d(df, '2015', '2020', '2023', title="Tendenser 2015–2023", color_column='Type')
