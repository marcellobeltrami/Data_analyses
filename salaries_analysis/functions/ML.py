import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

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

def UMAP_reduction (df: pd.DataFrame, rng=42 ): 

    umap = UMAP(n_components=2, random_state=rng)

    # Apply t-SNE transformation to the DataFrame
    umap_results = umap.fit_transform(df)
    df_umap = pd.DataFrame(umap_results, columns=['umap1', 'umap2'])
    
    # Plots and ouputs figures 
    fig = px.scatter(df_umap, x='umap1', y='umap2', title='UMAP Visualization')
    fig.show()

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




# Carries out random forest regressor after splitting for training and testing
def RF_train_test (preprocessed_df: pd.DataFrame, prediction_col:str, randomizer:str = None):
        
    X_train, X_test, y_train, y_test = train_test_split(preprocessed_df.drop(columns=prediction_col), 
                                                    preprocessed_df[prediction_col], 
                                                    test_size=0.2, 
                                                    random_state=randomizer)
    
    rgrs = RandomForestRegressor()

    fitted = rgrs.fit(X_train,y_train)

    # Feature importances
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': fitted.feature_importances_
    })

    return fitted.score(X_test,y_test), importance_df.sort_values(by="Importance"), fitted



# Implement another supervised algorithm
def STR_train_test (preprocessed_df: pd.DataFrame, prediction_col:str, randomizer:str = None):
        
    X_train, X_test, y_train, y_test = train_test_split(preprocessed_df.drop(columns=prediction_col), 
                                                    preprocessed_df[prediction_col], 
                                                    test_size=0.2, 
                                                    random_state=randomizer)
    
    '''
    rgrs = ...insert algorthm

    fitted = rgrs.fit(X_train,y_train)

    # Feature importances
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': fitted.feature_importances_
    })

    return (fitted.score(X_test,y_test), importance_df)
    '''

