from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from utils.logger import App_Logger
from utils.model_utils import get_best_score_for_model, get_model_param_grid
from utils.read_params import read_params


class Model_Finder:
    """
    Description :   This method is used for hyperparameter tuning of selected models  
                    some preprocessing steps and then train the models and register them in mlflow

    Version     :   1.2
    Revisions   :   moved to setup to cloud
    """

    def __init__(self, db_name, collection_name):
        self.db_name = db_name

        self.collection_name = collection_name

        self.class_name = self.__class__.__name__

        self.config = read_params()

        self.log_writer = App_Logger()

        self.cv = self.config["model_utils"]["cv"]

        self.verbose = self.config["model_utils"]["verbose"]

        self.n_jobs = self.config["model_utils"]["n_jobs"]

        self.svm = SVC()

        self.knn = KNeighborsClassifier()

        self.lr = LogisticRegression()

    def get_best_params_for_lr(self, train_x, train_y):
        """
        Method Name :   get_best_params_for_lr
        Description :   This method is used for getting the best params for logistic regression model

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_best_params_for_lr.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            db_name=self.db_name,
            collection_name=self.collection_name,
        )

        try:
            self.lr_param_grid = get_model_param_grid(
                model_key_name="lr_model",
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            self.lr_grid = RandomizedSearchCV(
                estimator=self.lr,
                param_grid=self.svm_param_grid,
                cv=self.cv,
                verbose=self.verbose,
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Initialized {self.lr_param_grid.__class__.__name__} model with {self.lr_param_grid} as params",
            )

            self.lr_grid.fit(train_x, train_y)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Found the best params for {self.lr.__class__.__name__} model based on {self.lr_param_grid} as params",
            )

            self.C = self.lr_grid.best_params_["C"]

            self.penalty = self.lr_grid.best_params_["penalty"]

            self.max_iter = self.lr_grid.best_params_["max_iter"]

            self.solver = self.lr_grid.best_params_["solver"]

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"{self.lr.__class__.__name__} model best params are {self.lr_grid.best_params_}",
            )

            self.lr = LogisticRegression(
                penalty=self.penalty,
                C=self.C,
                max_iter=self.max_iter,
                solver=self.solver,
                n_jobs=self.n_jobs,
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Initialized {self.lr.__class__,__name__} with {self.lr_grid.best_params_} as params",
            )

            self.lr.fit(train_x, train_y)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Created {self.lr.__class__.__name__} model with {self.lr_grid.best_params_} as params",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            return self.lr

        except Exception as e:
            self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def get_best_params_for_svm(self, train_x, train_y):
        """
        Method Name :   get_best_params_for_svm
        Description :   This method is used for getting the best params for svm model

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_best_params_for_svm.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            db_name=self.db_name,
            collection_name=self.collection_name,
        )

        try:
            self.svm_param_grid = get_model_param_grid(
                model_key_name="svm_model",
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            self.svm_grid = RandomizedSearchCV(
                estimator=self.svm,
                param_grid=self.svm_param_grid,
                cv=self.cv,
                verbose=self.verbose,
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Initialized {self.svm_param_grid.__class__.__name__} model with {self.svm_param_grid} as params",
            )

            self.svm_grid.fit(train_x, train_y)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Found the best params {self.svm.__class__.__name__} based on {self.svm_param_grid} as params",
            )

            self.kernel = self.svm_grid.best_params_["kernel"]

            self.C = self.svm_grid.best_params_["C"]

            self.random_state = self.svm_grid.best_params_["random_state"]

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"The best params for {self.svm.__class__.__name__} are {self.svm_grid.best_params_}",
            )

            self.svm = SVC(kernel=self.kernel, C=self.C, random_state=self.random_state)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Initialized {self.svm.__class__.__name__} model with the {self.svm_grid.best_params_}",
            )

            self.svm.fit(train_x, train_y)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Created {self.svm.__class__.__name__} model with {self.svm_grid.best_params_}",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            return self.svm

        except Exception as e:
            self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def get_best_params_for_knn(self, train_x, train_y):
        """
        Method Name :   get_best_params_for_knn
        Description :   This method is used for getting the best params for knn model

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_best_params_for_knn.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            db_name=self.db_name,
            collection_name=self.collection_name,
        )

        try:
            self.knn_param_grid = get_model_param_grid(
                model_key_name="knn_model",
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            self.knn_grid = RandomizedSearchCV(
                self.knn, self.knn_param_grid, verbose=self.verbose, cv=self.cv
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Initialized {self.knn_grid.__class__.__name__} model  with {self.knn_grid} as params",
            )

            self.knn_grid.fit(train_x, train_y)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Found the best params {self.knn.__class__.__name__} model based on {self.knn_param_grid} as params",
            )

            self.algorithm = self.knn_grid.best_params_["algorithm"]

            self.leaf_size = self.knn_grid.best_params_["leaf_size"]

            self.n_neighbors = self.knn_grid.best_params_["n_neighbors"]

            self.p = self.knn_grid.best_params_["p"]

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"The best params for {self.knn.__class__.__name__} are {self.knn_grid.best_params_}",
            )

            self.knn = KNeighborsClassifier(
                algorithm=self.algorithm,
                leaf_size=self.leaf_size,
                n_neighbors=self.n_neighbors,
                p=self.p,
                n_jobs=self.n_jobs,
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Initialized {self.knn.__class__.__name__} with the {self.knn_grid.best_params_}",
            )

            self.knn.fit(train_x, train_y)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Created {self.knn.__class__.__name__} with the {self.knn_grid.best_params_}",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"{self.knn.__class__.__name__} best params are {str(self.knn_grid.best_params_)}",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            return self.knn

        except Exception as e:
            self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def get_trained_models(self, train_x, train_y, test_x, test_y):
        """
        Method Name :   get_trained_models
        Description :   this method traines all the given models, and returns the best score
        Output      :   trained models with thier best scores
        """
        method_name = self.get_trained_models.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            db_name=self.db_name,
            collection_name=self.collection_name,
        )

        try:
            self.knn = self.get_best_params_for_knn(train_x, train_y)

            self.knn_score = get_best_score_for_model(
                model=self.knn,
                test_x=test_x,
                test_y=test_y,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            self.svm = self.get_best_params_for_svm(train_x=train_x, train_y=train_y)

            self.svm_score = get_best_score_for_model(
                model=self.svm,
                test_x=test_x,
                test_y=test_y,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            self.lr = self.get_best_params_for_lr(train_x=train_x, train_y=train_y)

            self.lr_score = get_best_score_for_model(
                model=self.lr,
                test_x=test_x,
                test_y=test_y,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            return (
                self.knn,
                self.knn_score,
                self.lr,
                self.lr_score,
                self.svm,
                self.svm_score,
            )

        except Exception as e:
            self.log_writer.raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )
