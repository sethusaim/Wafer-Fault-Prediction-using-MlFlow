from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from utils.logger import App_Logger
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

    def __init__(self, db_name, logger_object):
        self.db_name = db_name

        self.logger_object = logger_object

        self.config = read_params()

        self.log_writter = App_Logger()

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
        self.log_writter.log(
            db_name=self.db_name,
            collection_name=self.logger_object,
            log_message="Entered the get_best_params_for_random_forest method of Model_Finder class",
        )

        try:
            self.param_grid_rf = {
                "n_estimators": self.config["model_params"]["rf_model"]["n_estimators"],
                "criterion": self.config["model_params"]["rf_model"]["criterion"],
                "max_depth": self.config["model_params"]["rf_model"]["max_depth"],
                "max_features": self.config["model_params"]["rf_model"]["max_features"],
            }

            self.grid = GridSearchCV(
                estimator=self.clf,
                param_grid=self.param_grid_rf,
                cv=self.config["model_params"]["cv"],
                verbose=self.config["model_params"]["verbose"],
                n_jobs=self.config["model_params"]["n_jobs"],
            )

            self.grid.fit(train_x, train_y)

            self.criterion = self.grid.best_params_["criterion"]

            self.max_depth = self.grid.best_params_["max_depth"]

            self.max_features = self.grid.best_params_["max_features"]

            self.n_estimators = self.grid.best_params_["n_estimators"]

            self.clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                criterion=self.criterion,
                max_depth=self.max_depth,
                max_features=self.max_features,
                n_jobs=self.config["model_params"]["n_jobs"],
            )

            self.clf.fit(train_x, train_y)

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.logger_object,
                log_message="Random Forest best params: "
                + str(self.grid.best_params_)
                + ". Exited the get_best_params_for_random_forest method of the Model Finder class",
            )

            return self.clf

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.logger_object,
                log_message=f"Exception occured in Class : Model_Finder. \
                    Method : get_best_params_for_random_forest, Error : {str(e)}",
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.logger_object,
                log_message="Random forest parameter tuning failed. \
                    Exited the get_best_params_for_random_forest method of the Model Finder class",
            )

            raise Exception(
                "Exception occured in Class : Model_Finder. \
                    Method : get_best_params_for_random_forest, Error : ",
                str(e),
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

        self.log_writter.log(
            db_name=self.db_name,
            collection_name=self.logger_object,
            log_message="Entered the get_best_params_for_xgboost method of the Model Finder class",
        )

        try:
            self.param_grid_xgb = {
                "learning_rate": self.config["model_params"]["xgb_model"][
                    "learning_rate"
                ],
                "max_depth": self.config["model_params"]["xgb_model"]["max_depth"],
                "n_estimators": self.config["model_params"]["xgb_model"][
                    "n_estimators"
                ],
            }

            self.grid = GridSearchCV(
                XGBClassifier(objective="binary:logistic"),
                self.param_grid_xgb,
                verbose=self.config["model_params"]["verbose"],
                cv=self.config["model_params"]["cv"],
                n_jobs=self.config["model_params"]["n_jobs"],
            )

            self.grid.fit(train_x, train_y)

            self.learning_rate = self.grid.best_params_["learning_rate"]

            self.max_depth = self.grid.best_params_["max_depth"]

            self.n_estimators = self.grid.best_params_["n_estimators"]

            self.xgb = XGBClassifier(
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                n_estimators=self.n_estimators,
                n_jobs=self.config["model_params"]["n_jobs"],
            )

            self.xgb.fit(train_x, train_y)

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.logger_object,
                log_message="XGBoost best params: "
                + str(self.grid.best_params_)
                + ". exited the get_best_params_for_xgboost method of the Model Finder class",
            )

            return self.xgb

        except Exception as e:
            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.logger_object,
                log_message=f"Exception occured in Class : Model_Finder. \
                    Method : get_best_params_for_xgboost, Error : {str(e)} ",
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.logger_object,
                log_message="XGBoost parameter tuning failed. Exited the get_best_params_for_xgboost method of the Model Finder class",
            )

            raise Exception(
                "Exception occured in Class : Model_Finder. \
                    Method : get_best_params_for_xgboost, Error : ",
                str(e),
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
        self.log_writter.log(
            db_name=self.db_name,
            collection_name=self.logger_object,
            log_message="Entered the get_trained_models method of the Model Finder class",
        )

        try:
            ## creating the xgboost model
            self.xgboost = self.get_best_params_for_xgboost(train_x, train_y)

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.logger_object,
                log_message="Started prediction using xgboost model",
            )

            self.pred_xgboost = self.xgboost.predict(test_x)

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.logger_object,
                log_message="Prediction done using xgboost model",
            )

            if len(test_y.unique()) == 1:
                self.log_writter.log(
                    db_name=self.db_name,
                    collection_name=self.logger_object,
                    log_message="Started calculating the accuracy score of xgb_model",
                )

                self.xgboost_score = accuracy_score(test_y, self.pred_xgboost)

                self.log_writter.log(
                    db_name=self.db_name,
                    collection_name=self.logger_object,
                    log_message="Accuracy for XGBoost: " + str(self.xgboost_score),
                )

            else:
                self.log_writter.log(
                    db_name=self.db_name,
                    collection_name=self.logger_object,
                    log_message="Started calculating the roc auc score of xgb_model",
                )

                self.xgboost_score = roc_auc_score(test_y, self.pred_xgboost)

                self.log_writter.log(
                    db_name=self.db_name,
                    collection_name=self.logger_object,
                    log_message="AUC score for XGBoost: " + str(self.xgboost_score),
                )

            ## creating the random forest model
            self.random_forest = self.get_best_params_for_random_forest(
                train_x, train_y
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.logger_object,
                log_message="Started prediction using random forest model",
            )

            self.pred_rf = self.random_forest.predict(test_x)

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.logger_object,
                log_message="Prediction done using random forest model",
            )

            if len(test_y.unique()) == 1:
                self.log_writter.log(
                    db_name=self.db_name,
                    collection_name=self.logger_object,
                    log_message="Started calculating accuracy score of random forest model",
                )

                self.rf_score = accuracy_score(test_y, self.pred_rf)

                self.log_writter.log(
                    db_name=self.db_name,
                    collection_name=self.logger_object,
                    log_message="Accuracy score for RF: " + str(self.rf_score),
                )

            else:
                self.log_writter.log(
                    db_name=self.db_name,
                    collection_name=self.logger_object,
                    log_message="Started calculating roc auc score of random forest model",
                )

                self.rf_score = roc_auc_score(test_y, self.pred_rf)

                self.log_writter.log(
                    db_name=self.db_name,
                    collection_name=self.logger_object,
                    log_message="AUC score for XGB: " + str(self.rf_score),
                )

            return self.random_forest, self.rf_score, self.xgboost, self.xgboost_score

        except Exception as e:
            self.logger_object.log(
                self.file_object,
                "Exception occurred in get_trained_models method of the Model Finder class "
                + str(e),
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.logger_object,
                log_message=f"Exception occured in Class : Model Finder, Method : get_trained_models, Error : {str(e)}",
            )

            self.log_writter.log(
                db_name=self.db_name,
                collection_name=self.logger_object,
                log_message="Model training failed. Exited the get_trained_models method of the Model Finder class",
            )

            raise Exception(
                "Exception occured in Class : Model Finder, Method : get_trained_models, Error : ",
                str(e),
            )
