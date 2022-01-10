from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from utils.exception import raise_exception_log
from utils.logger import App_Logger
from utils.main_utils import get_model_param_grid
from utils.read_params import read_params
from xgboost import XGBClassifier


class Model_Finder:
    """
    Description :   This class shall be to train all the selected models with metrics as accuracy and
                    roc auc score
    Written By  :   iNeuron Intelligence
    Version     :   1.1
    Revisions   :   code changed from returning best models to returning all the trained models
                    modified code based on params.yaml file
    """

    def __init__(self, db_name, collection_name):
        self.db_name = db_name

        self.collection_name = collection_name

        self.config = read_params()

        self.class_name = self.__class__.__name__

        self.log_writer = App_Logger()

        self.clf = RandomForestClassifier()

        self.xgb = XGBClassifier(objective="binary:logistic")

    def get_best_params_for_random_forest(self, train_x, train_y):
        """
        Method name :   get_best_params_for_random_forest
        Description :   get the parameters for Random Forest Algorithm which gives the best results
                        Used Hyperparameter tuning, Grid Search CV
        Output      :   The model with best parameters
        On Failure  :   Raise Exception
        Written By  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on config file
        """

        method_name = self.get_best_params_for_random_forest.__name__

        self.log_writer.log(
            db_name=self.db_name,
            collection_name=self.collection_name,
            log_message=f"Entered the {method_name} method of {self.class_name} class",
        )

        try:
            self.param_grid_rf = get_model_param_grid(
                model_key_name="rf_model",
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            self.grid = GridSearchCV(
                estimator=self.clf,
                param_grid=self.param_grid_rf,
                cv=self.config["model_utils"]["cv"],
                verbose=self.config["model_utils"]["verbose"],
                n_jobs=self.config["model_utils"]["n_jobs"],
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Initialized grid search cv as hyperparameter tuning method to find best params",
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Started searching for best params using grid search cv",
            )

            self.grid.fit(train_x, train_y)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Finised seraching for best params",
            )

            self.criterion = self.grid.best_params_["criterion"]

            self.max_depth = self.grid.best_params_["max_depth"]

            self.max_features = self.grid.best_params_["max_features"]

            self.n_estimators = self.grid.best_params_["n_estimators"]

            self.clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                criterion=self.criterion,
                max_depth=self.max_depth,
                max_features=self.max_features,
                n_jobs=self.config["model_utils"]["n_jobs"],
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Initialized and starting training xgb_model with best params",
            )

            self.clf.fit(train_x, train_y)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Random Forest best params: "
                + str(self.grid.best_params_)
                + f". Exited the {method_name} method of the {self.class_name} class",
            )

            return self.clf

        except Exception as e:
            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Random forest parameter tuning failed.Exited the {method_name} method of the {self.class_name} class",
            )

            raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def get_best_params_for_xgboost(self, train_x, train_y):
        """
        Method Name :   get_best_params_for_xgboost
        Description :   get the parameters for XGBoost Algorithm which give the best metric
                        Used Hyper parameter tuning - Grid Search CV
        Output      :   The model with best parameters
        On Failure  :   Raise Exception
        Written by  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on the config file
        """

        method_name = self.get_best_params_for_xgboost.__name__

        self.log_writer.log(
            db_name=self.db_name,
            collection_name=self.collection_name,
            log_message=f"Entered the {method_name} method of the {self.class_name} class",
        )

        try:
            self.param_grid_xgb = get_model_param_grid(
                model_key_name="xgb_model",
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            self.grid = GridSearchCV(
                XGBClassifier(objective="binary:logistic"),
                self.param_grid_xgb,
                verbose=self.config["model_utils"]["verbose"],
                cv=self.config["model_utils"]["cv"],
                n_jobs=self.config["model_utils"]["n_jobs"],
            )

            self.grid.fit(train_x, train_y)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Finised seraching for best params",
            )

            self.learning_rate = self.grid.best_params_["learning_rate"]

            self.max_depth = self.grid.best_params_["max_depth"]

            self.n_estimators = self.grid.best_params_["n_estimators"]

            self.xgb = XGBClassifier(
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                n_estimators=self.n_estimators,
                n_jobs=self.config["model_utils"]["n_jobs"],
            )

            self.xgb.fit(train_x, train_y)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="XGBoost best params: "
                + str(self.grid.best_params_)
                + f". exited the {method_name} method of the {self.class_name} class",
            )

            return self.xgb

        except Exception as e:
            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"XGBoost parameter tuning failed. Exited the {method_name} method of the {self.class_name} class",
            )

            raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def get_trained_models(self, train_x, train_y, test_x, test_y):
        """
        Method Name :   get_trained_models
        Description :   Return all the trained models with thier corresponding metric
        Output      :   The trained model and model score
        On Failure  :   Raise Exception
        Written By  :   iNeuron Intelligence
        Version     :   1.1
        Revisions   :   modified code based on the config file
        """

        method_name = self.get_trained_models.__name__

        self.log_writer.log(
            db_name=self.db_name,
            collection_name=self.collection_name,
            log_message=f"Entered the {method_name} method of the {self.class_name} class",
        )

        try:
            ## creating the xgboost model
            self.xgboost = self.get_best_params_for_xgboost(train_x, train_y)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Started prediction using xgboost model",
            )

            self.pred_xgboost = self.xgboost.predict(test_x)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Prediction done using xgboost model",
            )

            if len(test_y.unique()) == 1:
                self.log_writer.log(
                    db_name=self.db_name,
                    collection_name=self.collection_name,
                    log_message="Started calculating the accuracy score of xgb_model",
                )

                self.xgboost_score = accuracy_score(test_y, self.pred_xgboost)

                self.log_writer.log(
                    db_name=self.db_name,
                    collection_name=self.collection_name,
                    log_message="Accuracy for XGBoost: " + str(self.xgboost_score),
                )

            else:
                self.log_writer.log(
                    db_name=self.db_name,
                    collection_name=self.collection_name,
                    log_message="Started calculating the roc auc score of xgb_model",
                )

                self.xgboost_score = roc_auc_score(test_y, self.pred_xgboost)

                self.log_writer.log(
                    db_name=self.db_name,
                    collection_name=self.collection_name,
                    log_message="AUC score for XGBoost: " + str(self.xgboost_score),
                )

            ## creating the random forest model
            self.random_forest = self.get_best_params_for_random_forest(
                train_x, train_y
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Started prediction using random forest model",
            )

            self.pred_rf = self.random_forest.predict(test_x)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message="Prediction done using random forest model",
            )

            if len(test_y.unique()) == 1:
                self.log_writer.log(
                    db_name=self.db_name,
                    collection_name=self.collection_name,
                    log_message="Started calculating accuracy score of random forest model",
                )

                self.rf_score = accuracy_score(test_y, self.pred_rf)

                self.log_writer.log(
                    db_name=self.db_name,
                    collection_name=self.collection_name,
                    log_message="Accuracy score for RF: " + str(self.rf_score),
                )

            else:
                self.log_writer.log(
                    db_name=self.db_name,
                    collection_name=self.collection_name,
                    log_message="Started calculating roc auc score of random forest model",
                )

                self.rf_score = roc_auc_score(test_y, self.pred_rf)

                self.log_writer.log(
                    db_name=self.db_name,
                    collection_name=self.collection_name,
                    log_message="AUC score for XGB: " + str(self.rf_score),
                )

            return self.random_forest, self.rf_score, self.xgboost, self.xgboost_score

        except Exception as e:
            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_message=f"Model training failed. Exited the {method_name} method of the {self.class_name} class",
            )

            raise_exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )
