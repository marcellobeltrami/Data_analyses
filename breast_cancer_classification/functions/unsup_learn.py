import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering



# Carries out tsne dim reduction
def TSNE_reduction (df: pd.DataFrame, rng=42 ): 

    tsne = TSNE(n_components=2, random_state=rng)

    # Apply t-SNE transformation to the DataFrame
    tsne_results = tsne.fit_transform(df)
    df_tsne = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])

    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(data=df_tsne, x='tsne1', y='tsne2',cmap="flare", edgecolor='black', linewidth=0.5)

    # Update layout (optional)
    scatter.set_title('t-SNE Visualization', fontsize=16)
    scatter.set_xlabel('tsne1', fontsize=14)
    scatter.set_ylabel('tsne2', fontsize=14)

    # Show and save the plot
    plt.show()

    return df_tsne 

def UMAP_reduction (df: pd.DataFrame, rng=42 ): 

    umap = UMAP(n_components=2, random_state=rng)

    # Apply t-SNE transformation to the DataFrame
    umap_results = umap.fit_transform(df)
    df_umap = pd.DataFrame(umap_results, columns=['umap1', 'umap2'])
    
    # Plots and ouputs figures 
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(data=df_umap, x='umap1', y='umap2',palette="flare",edgecolor='black', linewidth=0.5)

    # Update layout (optional)
    scatter.set_title('UMAP Visualization', fontsize=16)
    scatter.set_xlabel('umap1', fontsize=14)
    scatter.set_ylabel('umap2', fontsize=14)

    # Show and save the plot
    plt.show()

    return df_umap

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

# Clusters using Agglomerative algorithm
def AGGL_cluster (df:pd.DataFrame, clusters=8 ):
    agg = AgglomerativeClustering(n_clusters=clusters )
    agg_results = agg.fit(df)

    df['cluster'] = agg_results.labels_

    return df


