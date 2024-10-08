import pandas as pd
from flask import Flask
from pi_conf import Config
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chora.models.abstract_model import AbstractModel
from chora.models.clustering import ClusteringModel
from chora.models.online_models import BiasedMFModel
from chora.recommender import RecommenderService
from chora.types.song_data import SongDataSchema


class ChoraFlask(Flask):
    def __init__(self, import_name: str, cfg: Config, *args, **kwargs):
        super().__init__(import_name, *args, **kwargs)
        self.data_df = pd.read_parquet(cfg.data_file)
        transformed_df = pd.read_parquet(cfg.transformed_data_file)
        likes_df = pd.read_csv(cfg.likes_file)

        ## Clustering
        nclusters = 20
        cluster_pipeline = Pipeline(
            [("scaler", StandardScaler()), ("kmeans", KMeans(n_clusters=nclusters))]
        )

        cmodel = ClusteringModel(
            pipeline=cluster_pipeline,
            feature_columns=SongDataSchema._relevant_features,
            data_df=self.data_df,
        )

        ## Online Models
        om = BiasedMFModel(data_df=transformed_df)

        ## Create Service
        models: dict[str, AbstractModel] = {
            "biased_mf": om,
            "clustering": cmodel,
        }
        self.service = RecommenderService(
            models=models,
            data_df=self.data_df,
            tdata_df=transformed_df,
            likes_df=likes_df,
        )

        self._register_routes()

    def _register_routes(self):
        """
        Register the blueprints (routes)
        """
        from chora.app.api.user_routes import bp as user_bp
        from chora.app.api.system_routes import bp as system_bp

        self.register_blueprint(user_bp)
        self.register_blueprint(system_bp)

        self.print_registered_routes()

    def print_registered_routes(self):
        """
        Print all registered routes
        """
        print("Registered Routes:")
        for rule in self.url_map.iter_rules():
            methods = ", ".join(rule.methods or [])
            print(f"{rule.endpoint}: {rule.rule} [{methods}]")


def create_app(_cfg: Config) -> ChoraFlask:
    return ChoraFlask(__name__, cfg=_cfg)
