from kneed import KneeLocator
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from src.file_operations import file_methods
from utils.read_params import read_params


class KMeansClustering:
    """
    Description :   This class shall be used to divide the data into clusters before training
    Written by  :   iNeuron Intelligence
    Version     :   1.0
    Revisions   :   None
    """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object

        self.logger_object = logger_object

        self.config = read_params()

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
        self.logger_object.log(
            self.file_object,
            "Entered the elbow_plot method of the KMeansClustering class",
        )

        wcss = []

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

            self.kn = KneeLocator(
                range(1, self.config["kmeans_cluster"]["max_clusters"]),
                wcss,
                curve=self.config["kmeans_cluster"]["knee_locator"]["curve"],
                direction=self.config["kmeans_cluster"]["knee_locator"]["direction"],
            )

            self.logger_object.log(
                self.file_object,
                "The optimum number of clusters is: "
                + str(self.kn.knee)
                + " . Exited the elbow_plot method of the KMeansClustering class",
            )

            return self.kn.knee

        except Exception as e:
            self.logger_object.log(
                self.file_object,
                "Exception occured in elbow_plot method of the KMeansClustering class. Exception message:  "
                + str(e),
            )

            self.logger_object.log(
                self.file_object,
                "Finding the number of clusters failed. Exited the elbow_plot method of the KMeansClustering class",
            )

            raise e

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
        self.logger_object.log(
            self.file_object,
            "Entered the create_clusters method of the KMeansClustering class",
        )

        self.data = data

        try:
            self.kmeans = KMeans(
                n_clusters=number_of_clusters,
                init=self.config["kmeans_cluster"]["init"],
                random_state=self.config["base"]["random_state"],
            )

            self.y_kmeans = self.kmeans.fit_predict(data)

            self.file_op = file_methods.File_Operation(
                self.file_object, self.logger_object
            )

            self.file_op.save_model(
                self.kmeans, self.config["model_names"]["kmeans_model_name"]
            )

            self.data["Cluster"] = self.y_kmeans

            self.logger_object.log(
                self.file_object,
                "succesfully created "
                + str(self.kn.knee)
                + " clusters. Exited the create_clusters method of the KMeansClustering class",
            )

            return self.data, self.kmeans

        except Exception as e:
            self.logger_object.log(
                self.file_object,
                "Exception occured in create_clusters method of the KMeansClustering class Exception message: "
                + str(e),
            )

            self.logger_object.log(
                self.file_object,
                "Fitting the data to clusters failed. Exited the create_clusters method of the KMeansClustering class",
            )

            raise e
