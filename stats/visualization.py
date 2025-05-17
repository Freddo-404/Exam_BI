import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os


# Function to create a histogram
def histogram(data, column_name, title="Histogram"):
    fig, ax = plt.subplots()
    ax.hist(data[column_name], bins=10, edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel(column_name)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# Function to create a boxplot
def boxplot(data, column_name, title="Box Plot"):
    fig, ax = plt.subplots()
    sns.boxplot(y=data[column_name], ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

# Function to create a scatter plot
def scatter_plot(data, x_column, y_column, title="Scatter Plot"):
    fig, ax = plt.subplots()
    ax.scatter(data[x_column], data[y_column])
    ax.set_title(title)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    st.pyplot(fig)

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

# def scatter_plot_3d(data, x_column, y_column, z_column, title="3D Scatter Plot"):
#    st.subheader(title)
#
#    fig = px.scatter_3d(
#        data,
#        x=x_column,
#        y=y_column,
#        z=z_column,
#        color=z_column,
#        opacity=0.7
#    ) 

#    fig.update_layout(title=title)
#    st.plotly_chart(fig)

def show_graphs():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    excel_path = os.path.join(base_dir, 'Streamlit', 'Data', 'Uddannelse_combined.xlsx')

    st.header("Uddannelse Data Visualization")
    
    # Load wine data from Excel
    data = pd.read_excel(excel_path)
    data.columns = data.columns.map(str)  # Convert all column names to strings

    
    # Show the first few rows to verify data
    st.write(data.head())

    # if all(col in data.columns for col in ['alcohol', 'chlorides', 'quality']):
    #    scatter_plot_3d(data, 'alcohol', 'chlorides', 'quality', title="Alcohol vs Chlorides vs Quality")

    # Show a histogram for alcohol content
    # if 'Type' in data.columns:
    #    histogram(data, 'Type', title="Alcohol Content Distribution")
    
    # Show a scatter plot for alcohol vs. type
    # if 'alcohol' in data.columns and 'quality' in data.columns:
    #    scatter_plot(data, 'alcohol', 'quality', title="Alcohol vs Quality")

    if all(col in data.columns for col in ['Type', '2015']):
        histogram(data, 'Type', title="Antallet af afbrudte og fuldførte")
        boxplot(data, '2015', title="Education Levels in 2015")

    if all(col in data.columns for col in ['Type', '2020']):
        boxplot(data, '2020', title="Education Levels in 2020")

    if all(col in data.columns for col in ['Type', '2023']):
        boxplot(data, '2023', title="Education Levels in 2023")

    if all(col in data.columns for col in ['2015', '2020', '2023', 'Type']):
        scatter_plot_3d(data, '2015', '2020', '2023', title="2015-2023 Trends Colored by Type", color_column='Type')


    
