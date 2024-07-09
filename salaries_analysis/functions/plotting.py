import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt

def scatter_plot(df:pd.DataFrame, labels:list, color_col:str, 
                 title="Scatter Plot", 
                 save_loc = "./",
                 format="png"):

    # Create a scatter plot using Plotly Express
    fig = px.scatter(df, x=labels[0], y=labels[1], color=color_col, title=title)

    # Update layout (optional)
    fig.update_layout(
        xaxis_title=labels[0],
        yaxis_title=labels[1],
        font=dict(
            family='Arial',
            size=12,
            color='black'
        )
    )

    # Show and saves the plot
    fig.show()

    if save_loc != "./":
        pio.write_image(fig, f"{save_loc}/{title}.{format}", format=f'{format}')
    

def bar_plot(df:pd.DataFrame, labels:list, 
                 title="Bar Plot", 
                 save_loc = "./",
                 format="png"):

    # Create a scatter plot using Plotly Express
    fig = px.bar(df, x=labels[0], y=labels[1], title=title)

    # Update layout (optional)
    fig.update_layout(
        xaxis_title=labels[0],
        yaxis_title=labels[1],
        font=dict(
            family='Arial',
            size=12,
            color='black'
        )
    )

    # Show and saves the plot
    fig.show()

    if save_loc != "./":
        pio.write_image(fig, f"{save_loc}/{title}.{format}", format=f'{format}')

#Correlation matrix 
def corr_mx(pandas_df:pd.DataFrame):
    #Calculates correlation scores
    correlation_matrix = pandas_df.corr()
    # Create correlation matrix plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', fmt=".2f", linewidths=.5)
    plt.show()


def summary_plot(df:pd.DataFrame):
    pass