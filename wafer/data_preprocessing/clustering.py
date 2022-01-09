from kneed import KneeLocator
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from utils.exception import raise_exception
from utils.logger import App_Logger
from utils.read_params import read_params
from wafer.s3_bucket_operations.s3_operations import S3_Operations


class KMeansClustering:
    def __init__(self, db_name, collection_name):
        self.db_name = db_name

        self.collection_name = collection_name

        self.config = read_params()

        self.input_files_bucket = self.config["s3_bucket"]["input_files_bucket"]

        self.model_bucket = self.config["s3_bucket"]["wafer_model_bucket"]

        self.random_state = self.config["base"]["random_state"]

        self.kmeans_init = self.config["kmeans_cluster"]["init"]

        self.max_clusters = self.config["kmeans_cluster"]["max_clusters"]

        self.kmeans_curve = self.config["kmeans_cluster"]["knee_locator"]["curve"]

        self.kmeans_direction = self.config["kmeans_cluster"]["knee_locator"][
            "direction"
        ]

        self.s3_obj = S3_Operations()

        self.elbow_plot_file = self.config["elbow_plot_fig"]

        self.log_writer = App_Logger()

        self.class_name = self.__class__.__name__

    def elbow_plot(self, data):
        method_name = self.elbow_plot.__name__

        self.log_writer.log(
            db_name=self.db_name,
            collection_name=self.collection_name,
            log_message=f"Entered the {method_name} method of the {self.class_name} class",
        )

        wcss = []

        try:
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
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Saved elbow_plot fig and local copy is created",
            )

            self.s3_obj.upload_to_s3(
                src_file=self.elbow_plot_file,
                bucket=self.input_files_bucket,
                dest_file=self.elbow_plot_file,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            self.kn = KneeLocator(
                range(1, self.max_clusters),
                wcss,
                curve=self.kmeans_curve,
                direction=self.kmeans_direction,
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"The optimum number of clusters is {str(self.kn.knee)}. Exited the {method_name} method of the {self.class_name} class",
            )

            return self.kn.knee

        except Exception as e:
            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Finding the number of clusters failed. Exited the {method_name} method of the {self.class_name} class",
            )

            raise_exception(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def create_clusters(self, data, number_of_clusters):
        method_name = self.create_clusters.__name__

        self.log_writer.log(
            db_name=self.db_name,
            collection_name=self.collection_name,
            log_message=f"Entered the {method_name} method of the {self.class_name} class",
        )

        self.data = data

        try:
            self.kmeans = KMeans(
                n_clusters=number_of_clusters,
                init=self.kmeans_init,
                random_state=self.random_state,
            )

            self.y_kmeans = self.kmeans.fit_predict(data)

            self.s3_obj.save_model_to_s3(
                idx=None,
                model=self.kmeans,
                model_bucket=self.model_bucket,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            self.data["Cluster"] = self.y_kmeans

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="succesfully created "
                + str(self.kn.knee)
                + f" clusters. Exited the {method_name} method of the {self.class_name} class",
            )

            return self.data, self.kmeans

        except Exception as e:
            raise_exception(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )
