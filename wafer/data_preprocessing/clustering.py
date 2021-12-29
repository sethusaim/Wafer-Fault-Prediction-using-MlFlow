import os

from kneed import KneeLocator
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from utils.logger import App_Logger
from utils.main_utils import raise_exception
from utils.read_params import read_params
from wafer.file_operations.file_methods import File_Operation
from wafer.s3_bucket_operations.s3_operations import S3_Operations


class KMeansClustering:
    """
    Description :   This class shall be used to divide the data into clusters before training
    Written by  :   iNeuron Intelligence
    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self, db_name, collection_name):
        self.db_name = db_name

        self.collection_name = collection_name

        self.config = read_params()

        self.input_files_bucket = self.config["s3_bucket"]["input_files_bucket"]

        self.s3_obj = S3_Operations()

        self.log_writter = App_Logger()

        self.class_name = self.__class__.__name__

    def elbow_plot(self, data):
        """
        Method Name :   elbow plot
        Description :   This method saves the plot to decide the optimum number of clusters to the file
        Output      :   A picture saved to the directory
        On failure  :   Raise Exception
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """

        self.log_writter.log(
            db_name=self.db_name,
            collection_name=self.collection_name,
            log_message="Entered the elbow_plot method of the KMeansClustering class",
        )

        wcss = []

        method_name = self.elbow_plot.__name__

        try:
            for i in range(1, self.config["kmeans_cluster"]["max_clusters"]):
                kmeans = KMeans(
                    n_clusters=i,
                    init=self.config["kmeans_cluster"]["init"],
                    random_state=self.config["base"]["random_state"],
                )

                kmeans.fit(data)

                wcss.append(kmeans.inertia_)

            plt.plot(range(1, self.config["kmeans_cluster"]["max_clusters"]), wcss)

            plt.title("The Elbow Method")

            plt.xlabel("Number of clusters")

            plt.ylabel("WCSS")

            plt.savefig(self.config["elbow_plot_fig"])

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Saved elbow_plot fig and local copy is created",
            )

            self.s3_obj.upload_to_s3(
                src_file=self.config["elbow_plot_fig"],
                bucket=self.input_files_bucket,
                dest_file=self.config["elbow_plot_fig"],
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Uploaded elbow plot figure in bucket {self.input_files_bucket}",
            )

            os.remove(self.config["elbow_plot_fig"])

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Removed local copy of the elbow plot figure",
            )

            self.kn = KneeLocator(
                range(1, self.config["kmeans_cluster"]["max_clusters"]),
                wcss,
                curve=self.config["kmeans_cluster"]["knee_locator"]["curve"],
                direction=self.config["kmeans_cluster"]["knee_locator"]["direction"],
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="The optimum number of clusters is: "
                + str(self.kn.knee)
                + " . Exited the elbow_plot method of the KMeansClustering class",
            )

            return self.kn.knee

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Finding the number of clusters failed. \
                    Exited the elbow_plot method of the KMeansClustering class",
            )

            raise_exception(
                class_name=self.class_name,
                method_name=method_name,
                exception=str(e),
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def create_clusters(self, data, number_of_clusters):
        """
        Method Name :   create_clusters
        Description :   create a new dataframe consisting of cluster information
        Output      :   a dataframe with cluster column
        On failure  :   Raise Exception
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on params.yaml file
        """
        self.log_writter.log(
            db_name=self.db_name,
            collection_name=self.collection_name,
            log_message="Entered the create_clusters method of the KMeansClustering class",
        )

        self.data = data

        method_name = self.create_clusters.__name__

        try:
            self.kmeans = KMeans(
                n_clusters=number_of_clusters,
                init=self.config["kmeans_cluster"]["init"],
                random_state=self.config["base"]["random_state"],
            )

            self.y_kmeans = self.kmeans.fit_predict(data)

            self.file_op = File_Operation(self.db_name, self.collection_name)

            self.file_op.save_model(
                self.kmeans, self.config["model_names"]["kmeans_model_name"]
            )

            self.data["Cluster"] = self.y_kmeans

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="succesfully created "
                + str(self.kn.knee)
                + " clusters. Exited the create_clusters method of the KMeansClustering class",
            )

            return self.data, self.kmeans

        except Exception as e:
            raise_exception(
                class_name=self.class_name,
                method_name=method_name,
                exception=str(e),
                db_name=self.db_name,
                collection_name=self.collection_name,
            )
