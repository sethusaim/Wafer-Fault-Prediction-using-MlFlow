import os
import shutil

import mlflow
from mlflow.tracking import MlflowClient
from utils.application_logging import logger
from utils.read_params import read_params


class LoadProdModel:
    """
    Description :   This class shall be used for loading the production model
    Written by  :   iNeuron Intelligence
    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self):
        self.log_writer = logger.App_Logger()

        self.config = read_params()

        self.load_prod_model_log = os.path.join(
            self.config["log_dir"]["train_log_dir"], "loadProdModelLog.txt"
        )

        self.file_object = open(self.load_prod_model_log, "a+")

    def load_production_model(self, num_clusters):
        """
        Method Name :   load_production_model
        Description :   This method is responsible for moving the models from the trained models dir to
                        prod models dir and stag models dir based on the metrics
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """

        self.log_writer.log(
            self.file_object, "Started transitioning of models based on metrics"
        )

        try:
            self.log_writer.log(
                self.file_object, "Started setting the remote server uri for mlflow"
            )

            remote_server_uri = self.config["mlflow_config"]["remote_server_uri"]

            mlflow.set_tracking_uri(remote_server_uri)

            self.log_writer.log(self.file_object, "Set remote server uri")

            self.log_writer.log(
                self.file_object,
                "Started searching for runs in mlflow with experiment ids as 1",
            )

            runs = mlflow.search_runs(experiment_ids=1)

            self.log_writer.log(
                self.file_object,
                "Completed searchiing for runs in mlflow with experiment ids as 1",
            )

            cols, top_mn_lst = [], []

            """
            Code Explaination: 
            num_clusters - Dynamically allocated based on the number of clusters created using elbow plot

            Here, we are trying to iterate over the number of clusters and then dynamically create the cols 
            where in the best model names can be found, and then copied to production or staging depending on
            the condition

            Eg- metrics.XGBoost1-best_score
            """

            self.log_writer.log(
                self.file_object,
                "Started finding all the registered models based on the metrics",
            )

            for i in range(0, num_clusters):
                for model in self.config["model_names"].values():
                    if model != self.config["model_names"]["kmeans_model_name"]:
                        temp = "metrics." + str(model) + str(i) + "-best_score"

                        cols.append(temp)

                    else:
                        pass

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
                self.file_object, "Got all registered models based on metrics"
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
                self.file_object, "Searching for best 3 models based on the metrics"
            )

            best_metrics = runs[cols].max().sort_values(ascending=False)[:3]

            self.log_writer.log(
                self.file_object, "Got top 3 model names based on the metrics"
            )

            client = MlflowClient()

            ## best_metrics will store the value of metrics, but we want the names of the models,
            ## so best_metrics.index will return the name of the metric as registered in mlflow

            ## Eg. metrics.XGBoost1-best_score

            ## top_mn_lst - will store the top 3 model names

            self.log_writer.log(self.file_object, "Getting the top 3 model names")

            for mn in best_metrics.index:
                top_mn = mn.split("-")[0].split(".")[1]

                top_mn_lst.append(top_mn)

            ## Searching registered models in mlflow in descending order
            results = client.search_registered_models(order_by=["name DESC"])

            self.log_writer.log(
                self.file_object, "Got top 3 models based on the model names"
            )

            self.log_writer.log(
                self.file_object,
                "Started transitioning the models based on the results obtained",
            )

            ## results - This will store all the registered models in mlflow
            ## Here we are iterating through all the registered model and for every latest registered model
            ## we are checking if the model name is in the top 3 model list, if present we are putting that
            ## model into production.

            for res in results:
                for mv in res.latest_versions:
                    if mv.name in top_mn_lst:
                        current_version = mv.version

                        self.log_writer.log(
                            self.file_object,
                            "Started transitioning "
                            + mv.name
                            + " with version "
                            + current_version
                            + " into production",
                        )

                        client.transition_model_version_stage(
                            name=mv.name, version=current_version, stage="Production"
                        )

                        self.log_writer.log(
                            self.file_object,
                            "Transitioned "
                            + mv.name
                            + " with version "
                            + current_version
                            + " into production",
                        )

                        self.log_writer.log(
                            self.file_object,
                            "Started copying "
                            + mv.name
                            + " from trained models folder to prod models folder",
                        )

                        shutil.copy(
                            self.config["models_dir"]["trained_models_dir"]
                            + "/"
                            + mv.name
                            + self.config["model_save_format"],
                            self.config["models_dir"]["prod_models_dir"],
                        )

                        self.log_writer.log(
                            self.file_object,
                            "Copied " + mv.name + " to production model folder",
                        )

                    ## In the registered models, even kmeans model is present, so during prediction,
                    ## this model also needs to be in present in production, the code logic is present below

                    elif mv.name == self.config["model_names"]["kmeans_model_name"]:
                        current_version = mv.version

                        self.log_writer.log(
                            self.file_object,
                            "Started transitioning "
                            + mv.name
                            + " with version "
                            + current_version
                            + " into production",
                        )

                        client.transition_model_version_stage(
                            name=mv.name, version=current_version, stage="Production"
                        )

                        self.log_writer.log(
                            self.file_object,
                            "Transitioned " + mv.name + "to production in mlflow",
                        )

                        self.log_writer.log(
                            self.file_object,
                            "Started copying "
                            + mv.name
                            + " from trained model folder to prod model folder",
                        )

                        shutil.copy(
                            self.config["models_dir"]["trained_models_dir"]
                            + "/"
                            + mv.name
                            + self.config["model_save_format"],
                            self.config["models_dir"]["prod_models_dir"],
                        )

                        self.log_writer.log(
                            self.file_object,
                            "Copied " + mv.name + " to production model folder",
                        )

                    else:
                        current_version = mv.version

                        self.log_writer.log(
                            self.file_object,
                            "Started transitioning "
                            + mv.name
                            + " with version "
                            + current_version
                            + " into staging",
                        )

                        client.transition_model_version_stage(
                            name=mv.name, version=current_version, stage="Staging"
                        )

                        self.log_writer.log(
                            self.file_object,
                            "Transitioned " + mv.name + "to staging in mlflow",
                        )

                        self.log_writer.log(
                            self.file_object,
                            "Started copying "
                            + mv.name
                            + " from trained model folder to stag model folder",
                        )

                        shutil.copy(
                            self.config["models_dir"]["trained_models_dir"]
                            + "/"
                            + mv.name
                            + self.config["model_save_format"],
                            self.config["models_dir"]["stag_models_dir"],
                        )

                        self.log_writer.log(
                            self.file_object,
                            f"Copied {mv.name} to staging model folder",
                        )

            self.log_writer.log(
                self.file_object,
                "Transitioning of models based on scores successfully done",
            )

        except Exception as e:
            self.log_writer.log(self.file_object, str(e))

            self.log_writer.log(self.file_object, "Transitioning of models failed")

            raise e
