import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def scatter_plot(df: pd.DataFrame, labels: list, color_col: str, 
                 title="Scatter Plot", 
                 save_loc = "./",
                 format="png"):

    # Create a scatter plot using Seaborn
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(data=df, x=labels[0], y=labels[1], hue=color_col)

    # Update layout (optional)
    scatter.set_title(title, fontsize=16)
    scatter.set_xlabel(labels[0], fontsize=14)
    scatter.set_ylabel(labels[1], fontsize=14)
    scatter.legend(title=color_col)

    # Show and save the plot
    plt.show()

    if save_loc != "./":
        plt.savefig(f"{save_loc}/{title}.{format}", format=format)

    plt.close()




def bar_plot(df:pd.DataFrame, labels:list, 
                 title="Bar Plot", 
                 save_loc = "./",
                 format="png"):

   # Create a scatter plot using Seaborn
    plt.figure(figsize=(10, 6))
    scatter = sns.barplot(data=df, x=labels[0], y=labels[1])

    # Update layout (optional)
    scatter.set_title(title, fontsize=16)
    scatter.set_xlabel(labels[0], fontsize=14)
    scatter.set_ylabel(labels[1], fontsize=14)

    # Show and save the plot
    plt.show()

    if save_loc != "./":
        plt.savefig(f"{save_loc}/{title}.{format}", format=format)

    plt.close()

#Correlation matrix 
def corr_mx(pandas_df:pd.DataFrame):
    #Calculates correlation scores
    correlation_matrix = pandas_df.corr()
    # Create correlation matrix plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', fmt=".2f", linewidths=.5)
    plt.show()


def summary_plot(df:pd.DataFrame):
    #Calculates correlation scores
    pairplot = sns.pairplot(df)
    # Create correlation matrix plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(pairplot, annot=True, cmap='YlGnBu', fmt=".2f", linewidths=.5)
    plt.show()