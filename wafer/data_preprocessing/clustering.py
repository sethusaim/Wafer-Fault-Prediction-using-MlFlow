from kneed import KneeLocator
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from utils.logger import App_Logger
from utils.read_params import read_params
from wafer.s3_bucket_operations.s3_operations import S3_Operation


class KMeans_Clustering:
    """
    Description :   This class shall be used to divide the data into clusters before training.
    
    Version     :   1.2
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self, log_file):
        self.log_file = log_file

        self.config = read_params()

        self.input_files_bucket = self.config["s3_bucket"]["input_files"]

        self.model_bucket_name = self.config["s3_bucket"]["wafer_model"]

        self.random_state = self.config["base"]["random_state"]

        self.kmeans_init = self.config["kmeans_cluster"]["init"]

        self.max_clusters = self.config["kmeans_cluster"]["max_clusters"]

        self.kmeans_curve = self.config["kmeans_cluster"]["knee"]["curve"]

        self.kmeans_direction = self.config["kmeans_cluster"]["knee"]["direction"]

        self.trained_model_dir = self.config["model_dir"]["trained"]

        self.s3 = S3_Operation()

        self.elbow_plot_file = self.config["elbow_plot_fig"]

        self.log_writer = App_Logger()

        self.class_name = self.__class__.__name__

    def elbow_plot(self, data):
        """
        Method Name :   elbow_plot
        Description :   This method saves the plot to s3 bucket and decides the optimum number of clusters to the file.
        
        Output      :   An elbow plot figure saved to input files bucket
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   Moved to setup to cloud 
        """
        method_name = self.elbow_plot.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            log_file=self.log_file,
        )

        try:
            wcss = []

            for i in range(1, self.max_clusters):
                kmeans = KMeans(
                    n_clusters=i, init=self.kmeans_init, random_state=self.random_state
                )

                kmeans.fit(data)

                wcss.append(kmeans.inertia_)

            plt.plot(range(1, self.max_clusters), wcss)

            plt.title("The Elbow Method")

            plt.xlabel("Number of clusters")

            plt.ylabel("WCSS")

            plt.savefig(self.elbow_plot_file)

            self.log_writer.log(
                log_file=self.log_file,
                log_info="Saved elbow_plot fig and local copy is created",
            )

            self.s3.upload_file(
                from_fname=self.elbow_plot_file,
                to_fname=self.elbow_plot_file,
                bucket=self.input_files_bucket,
                log_file=self.log_file,
            )

            self.kn = KneeLocator(
                range(1, self.max_clusters),
                wcss,
                curve=self.kmeans_curve,
                direction=self.kmeans_direction,
            )

            self.log_writer.log(
                log_file=self.log_file,
                log_info=f"The optimum number of clusters is {str(self.kn.knee)}.",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                log_file=self.log_file,
            )

            return self.kn.knee

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                log_file=self.log_file,
            )

    def create_clusters(self, data, number_of_clusters):
        """
        Method Name :   create_clusters
        Description :   Create a new dataframe consisting of the cluster information.
        
        Output      :   A dataframe with cluster column
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   Moved to setup to cloud 
        """
        method_name = self.create_clusters.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            log_file=self.log_file,
        )

        try:
            self.data = data

            self.kmeans = KMeans(
                n_clusters=number_of_clusters,
                init=self.kmeans_init,
                random_state=self.random_state,
            )

            self.y_kmeans = self.kmeans.fit_predict(data)

            self.s3.save_model(
                model=self.kmeans,
                model_dir=self.trained_model_dir,
                model_bucket_name=self.model_bucket_name,
                log_file=self.log_file,
            )

            self.data["Cluster"] = self.y_kmeans

            self.log_writer.log(
                log_file=self.log_file,
                log_info=f"Successfully created {str(self.kn.knee)} clusters",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                log_file=self.log_file,
            )

            return self.data, self.kmeans

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                log_file=self.log_file,
            )
