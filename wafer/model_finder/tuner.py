from sklearn.ensemble import RandomForestClassifier
from utils.logger import App_Logger
from utils.model_utils import Model_Utils
from utils.read_params import read_params
from xgboost import XGBClassifier


class Model_Finder:
    """
    Description :   This class shall  be used to find the model with best accuracy and AUC score.
    Written by  :   iNeuron Intelligence
    
    Version     :   1.2
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self, log_file):
        self.log_file = log_file

        self.class_name = self.__class__.__name__

        self.config = read_params()

        self.model_utils = Model_Utils()

        self.log_writer = App_Logger()

        self.rf_model = RandomForestClassifier()

        self.xgb_model = XGBClassifier(objective="binary:logistic")

    def get_rf_model(self, train_x, train_y):
        """
        Method Name :   get_rf_model
        Description :   get the parameters for Random Forest Algorithm which give the best accuracy.
                        Use Hyper Parameter Tuning.
        
        Output      :   The model with the best parameters
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Written by  :   iNeuron Intelligence
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_rf_model.__name__

        self.log_writer.start_log("start", self.class_name, method_name, self.log_file)

        try:
            self.rf_model_name = self.rf_model.__class__.__name__

            self.rf_best_params = self.model_utils.get_model_params(
                self.rf_model, train_x, train_y, self.log_file
            )

            self.log_writer.log(
                self.log_file,
                f"{self.rf_model_name} model best params are {self.rf_best_params}",
            )

            self.rf_model.set_params(**self.rf_best_params)

            self.log_writer.log(
                self.log_file,
                f"Initialized {self.rf_model_name} with {self.rf_best_params} as params",
            )

            self.rf_model.fit(train_x, train_y)

            self.log_writer.log(
                self.log_file,
                f"Created {self.rf_model_name} based on the {self.rf_best_params} as params",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file
            )

            return self.rf_model

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file
            )

    def get_xgboost_model(self, train_x, train_y):
        """
        Method Name :   get_xgboost_model
        Description :   get the parameters for XGBoost Algorithm which give the best accuracy.
                        Use Hyper Parameter Tuning.
        
        Output      :   The model with the best parameters
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Written by  :   iNeuron Intelligence
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_xgboost_model.__name__

        self.log_writer.start_log("start", self.class_name, method_name, self.log_file)

        try:
            self.xgb_model_name = self.xgb_model.__class__.__name__

            self.xgb_best_params = self.model_utils.get_model_params(
                self.xgb_model, train_x, train_y, self.log_file
            )

            self.log_writer.log(
                self.log_file,
                f"{self.rf_model_name} model best params are {self.rf_best_params}",
            )

            self.xgb_model.set_params(**self.xgb_best_params)

            self.log_writer.log(
                self.log_file,
                f"Initialized {self.xgb_model_name} model with best params as {self.xgb_best_params}",
            )

            self.xgb_model.fit(train_x, train_y)

            self.log_writer.log(
                self.log_file,
                f"Created {self.xgb_model_name} model with best params as {self.xgb_best_params}",
            )

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file
            )

            return self.xgb_model

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file
            )

    def get_trained_models(self, train_x, train_y, test_x, test_y):
        """
        Method Name :   get_trained_models
        Description :   Find out the Model which has the best score.
        
        Output      :   The best model name and the model object
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Written by  :   iNeuron Intelligence
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_trained_models.__name__

        self.log_writer.start_log("start", self.class_name, method_name, self.log_file)

        try:
            self.xgb_model = self.get_xgboost_model(train_x, train_y)

            self.xgb_model_score = self.model_utils.get_model_score(
                self.xgb_model, test_x, test_y, self.log_file
            )

            self.rf_model = self.get_rf_model(train_x, train_y)

            self.rf_model_score = self.model_utils.get_model_score(
                self.rf_model, test_x, test_y, self.log_file
            )

            lst = [
                (self.xgb_model, self.xgb_model_score),
                (self.rf_model, self.rf_model_score),
            ]

            self.log_writer.start_log(
                "exit", self.class_name, method_name, self.log_file
            )

            return lst

        except Exception as e:
            self.log_writer.exception_log(
                e, self.class_name, method_name, self.log_file
            )
