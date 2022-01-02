import os

import mlflow
from mlflow.tracking import MlflowClient
from utils.exception import raise_exception
from utils.logger import App_Logger
from utils.read_params import read_params
from wafer.s3_bucket_operations.s3_operations import S3_Operations


class load_prod_model:
    """
    Description :   This class shall be used for loading the production model
    Written by  :   iNeuron Intelligence
    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self, num_clusters):
        self.log_writer = App_Logger()

        self.config = read_params()

        self.class_name = self.__class__.__name__

        self.model_bucket = self.config["s3_bucket"]["wafer_model_bucket"]

        self.trained_models_dir = self.config["models_dir"]["trained"]

        self.staged_models_dir = self.config["models_dir"]["staged"]

        self.prod_models_dir = self.config["models_dir"]["prod"]

        self.model_save_format = self.config["model_params"]["save_format"]

        self.num_clusters = num_clusters

        self.s3_obj = S3_Operations()

        self.load_prod_model_log = self.config["train_db_log"]["load_prod_model"]

        self.db_name = self.config["db_log"]["db_train_log"]

    def load_production_model(self):
        """
        Method Name :   load_production_model
        Description :   This method is responsible for moving the models from the trained models dir to
                        prod models dir and stag models dir based on the metrics
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """

        method_name = self.load_production_model.__name__

        self.log_writer.log(
            db_name=self.db_name,
            collection_name=self.load_prod_model_log,
            log_message="Started transitioning of models based on metrics",
        )

        try:
            remote_server_uri = self.config["mlflow_config"]["remote_server_uri"]

            mlflow.set_tracking_uri(remote_server_uri)

            client = MlflowClient(tracking_uri=remote_server_uri)

            exp = client.get_experiment_by_name(
                name=self.config["mlflow_config"]["experiment_name"]
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.load_prod_model_log,
                log_message="Set remote server uri",
            )

            runs = mlflow.search_runs(experiment_ids=exp.experiment_id)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.load_prod_model_log,
                log_message=f"Completed searchiing for runs in mlflow with experiment ids as {exp.experiment_id}",
            )

            """
            Code Explaination: 
            num_clusters - Dynamically allocated based on the number of clusters created using elbow plot

            Here, we are trying to iterate over the number of clusters and then dynamically create the cols 
            where in the best model names can be found, and then copied to production or staging depending on
            the condition

            Eg- metrics.XGBoost1-best_score
            """

            # for i in range(0, self.num_clusters):
            #     for model in self.config["model_names"].values():
            #         if model != self.config["model_names"]["kmeans_model_name"]:
            #             temp = "metrics." + str(model) + str(i) + "-best_score"

            #             cols.append(temp)

            #         else:
            #             pass

            # cols = [
            #     "metrics." + str(model) + str(i) + "-best_score"
            #     for i in range(0, self.num_clusters)
            #     for model in self.config["model_names"].values()
            #     if model != self.config["model_names"]["kmeans_model_name"]
            # ]

            reg_model_names = [
                dict(rm.names).values() for rm in client.list_registered_models()
            ]

            cols = [
                f"metrics." + str(model) + str(i) + "-best_score"
                for i in range(0, self.num_clusters)
                for model in reg_model_names
                if not model.startswith("KMeans")
            ]

            """ 
            Eg-output: For 3 clusters, 
            
            [
                metrics.XGBoost0-best_score,
                metrics.XGBoost1-best_score,
                metrics.XGBoost2-best_score,
                metrics.RandomForest0-best_score,
                metrics.RandomForest1-best_score,
                metrics.RandomForest2-best_score
            ] 
            """
            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.load_prod_model_log,
                log_message="Got all registered models based on metrics",
            )

            ## sort the metrics in descending order and extract the first 3 metrics
            ## this will return a series object

            """
            Eg- runs_dataframe: I am only showing for 3 cols,actual runs dataframe will be different
                                based on the number of clusters
                
                since for every run cluster values changes, rest two cols will be left as blank,
                so only we are taking the max value of each col, which is nothing but the value of the metric
                

run_number  metrics.XGBoost0-best_score metrics.RandomForest1-best_score metrics.XGBoost1-best_score
    0                   1                       0.5
    1                                                                                   1                 
    2                                                                           
            """
            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.load_prod_model_log,
                log_message="Searching for best 3 models based on the metrics",
            )

            best_metrics = (
                runs[cols]
                .max()
                .sort_values(ascending=False)[
                    : self.config["mlflow_config"]["num_of_prod_models"]
                ]
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.load_prod_model_log,
                log_message=f"Got top {self.config['mlflow_config']['num_of_prod_models']} model names based on the metrics",
            )

            ## best_metrics will store the value of metrics, but we want the names of the models,
            ## so best_metrics.index will return the name of the metric as registered in mlflow

            ## Eg. metrics.XGBoost1-best_score

            ## top_mn_lst - will store the top 3 model names

            # for mn in best_metrics.index:
            #     top_mn = mn.split("-")[0].split(".")[1]

            #     top_mn_lst.append(top_mn)

            top_mn_lst = [mn.split("-")[0].split(".")[1] for mn in best_metrics.index]

            ## Searching registered models in mlflow in descending order
            results = client.search_registered_models(order_by=["name DESC"])

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.load_prod_model_log,
                log_message=f"Got the top {self.config['mlflow_config']['num_of_prod_models']} model names",
            )

            ## results - This will store all the registered models in mlflow
            ## Here we are iterating through all the registered model and for every latest registered model
            ## we are checking if the model name is in the top 3 model list, if present we are putting that
            ## model into production.

            for res in results:
                for mv in res.latest_versions:
                    if mv.name in top_mn_lst:
                        current_version = mv.version

                        client.transition_model_version_stage(
                            name=mv.name, version=current_version, stage="Production"
                        )

                        self.log_writer.log(
                            db_name=self.db_name,
                            collection_name=self.load_prod_model_log,
                            log_message="Transitioned "
                            + mv.name
                            + " with version "
                            + current_version
                            + " into production",
                        )

                        self.log_writer.log(
                            db_name=self.db_name,
                            collection_name=self.load_prod_model_log,
                            log_message="Started copying "
                            + mv.name
                            + " from trained models folder to prod models folder",
                        )

                        self.trained_model_file = os.path.join(
                            self.trained_models_dir, mv.name + self.model_save_format
                        )

                        self.prod_model_file = os.path.join(
                            self.prod_models_dir, mv.name + self.model_save_format
                        )

                        self.s3_obj.copy_data_to_other_bucket(
                            src_bucket=self.model_bucket,
                            src_file=self.trained_model_file,
                            dest_bucket=self.model_bucket,
                            dest_file=self.prod_model_file,
                        )

                    ## In the registered models, even kmeans model is present, so during prediction,
                    ## this model also needs to be in present in production, the code logic is present below

                    elif mv.name == "KMeans":
                        current_version = mv.version

                        client.transition_model_version_stage(
                            name=mv.name, version=current_version, stage="Production"
                        )

                        self.log_writer.log(
                            db_name=self.db_name,
                            collection_name=self.load_prod_model_log,
                            log_message="Transitioned "
                            + mv.name
                            + "to production in mlflow",
                        )

                        self.trained_kmeans_model_file = os.path.join(
                            self.trained_models_dir, mv.name + self.model_save_format
                        )

                        self.prod_kmeans_model_file = os.path.join(
                            self.prod_models_dir, mv.name + self.model_save_format
                        )

                        self.s3_obj.copy_data_to_other_bucket(
                            src_bucket=self.model_bucket,
                            src_file=self.trained_kmeans_model_file,
                            dest_bucket=self.model_bucket,
                            dest_file=self.prod_kmeans_model_file,
                        )

                    else:
                        current_version = mv.version

                        client.transition_model_version_stage(
                            name=mv.name, version=current_version, stage="Staging"
                        )

                        self.log_writer.log(
                            db_name=self.db_name,
                            collection_name=self.load_prod_model_log,
                            log_message="Transitioned "
                            + mv.name
                            + "to staging in mlflow",
                        )

                        self.trained_model_file = os.path.join(
                            self.trained_models_dir, mv.name + self.model_save_format
                        )

                        self.stag_model_file = os.path.join(
                            self.trained_models_dir, mv.name + self.model_save_format
                        )

                        self.s3_obj.copy_data_to_other_bucket(
                            src_bucket=self.model_bucket,
                            src_file=self.trained_model_file,
                            dest_bucket=self.model_bucket,
                            dest_file=self.stag_model_file,
                            db_name=self.db_name,
                            collection_name=self.load_prod_model_log,
                        )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.load_prod_model_log,
                log_message="Transitioning of models based on scores successfully done",
            )

        except Exception as e:
            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.load_prod_model_log,
                log_message="Transitioning of models failed",
            )

            raise_exception(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.load_prod_model_log,
            )
