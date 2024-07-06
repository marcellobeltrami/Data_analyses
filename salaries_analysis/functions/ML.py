import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans

# Carries out tsne dim reduction
def TSNE_reduction (df: pd.DataFrame, rng=42 ): 

    tsne = TSNE(n_components=2, random_state=rng)

    # Apply t-SNE transformation to the DataFrame
    tsne_results = tsne.fit_transform(df)
    df_tsne = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
    
    # Plots and ouputs figures 
    fig = px.scatter(df_tsne, x='tsne1', y='tsne2', title='t-SNE Visualization')
    fig.show()

    return df_tsne 

# Clusters using DBSCAN algorithm
def DBSCAN_cluster(df: pd.DataFrame):
    dbscan = DBSCAN()

    dbscan_results = dbscan.fit(df)

    df['cluster'] = dbscan_results.labels_

    return df

# Clusters using KMEANS algorithm
def KMEANS_cluster(df: pd.DataFrame, clusters= 2 ,rng=42):
    kmeans = KMeans(n_clusters=clusters,
                    random_state=rng)

    kmeans_results = kmeans.fit(df)

    df['cluster'] = kmeans_results.labels_

    return df
