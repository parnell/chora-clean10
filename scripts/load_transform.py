import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chora import DATA_DIR
from chora.models.clustering import ClusteringModel
from chora.types.song_data import SongDataSchema

f = f"{DATA_DIR}/merged_music_data.csv"
pf = f"{DATA_DIR}/merged_music_data.parquet"
transformed_file = f"{DATA_DIR}/transformed_music_data.csv"


def transform():
    df = pd.read_csv(f)
    df.to_csv(f, index=False)
    vdf = SongDataSchema.validate_df(df)
    vdf.to_parquet(pf)
    tdf = SongDataSchema.transform_for_ml(vdf)
    print(vdf.shape)
    print(tdf.shape)
    tdf.to_parquet(f"{DATA_DIR}/transformed_music_data.parquet")


def fit_cluster():
    nclusters = 20
    df = pd.read_parquet(pf)
    ## drop the genres column
    # df = df.drop(columns=["genres"])
    cluster_pipeline = Pipeline(
        [("scaler", StandardScaler()), ("kmeans", KMeans(n_clusters=nclusters))]
    )
    cluster = ClusteringModel(n_clusters=nclusters, pipeline=cluster_pipeline)

    X = df.select_dtypes(np.number)
    cluster.fit(X)  ### put songs into one of the nclusters clusters
    cluster.save("cluster_model.pkl")

def load_cluster():
    cluster = ClusteringModel.load("cluster_model.pkl")
    print(cluster.clusters)
    print(cluster.pipeline)
    print(cluster.tsne_pipeline)
    print(cluster.tsne_embeddings)
    print(cluster.n_clusters)

if __name__ == "__main__":
    # transform()
    # fit_cluster()
    load_cluster()
