import os
from datetime import datetime


class App_Logger:
    """
    Description :   This class is used for logging the info
    
    Version     :   1.2
    Revisions   :   Moved to setup to cloud 
    """

    def __init__(self):
        self.class_name = self.__class__.__name__

    def log(self, log_file, log_info):
        try:
            self.now = datetime.now()

            self.date = self.now.strftime("%d:%m:%Y")

            self.current_time = self.now.strftime("%H:%M:%S")

            log_file_path = os.path.join("logs", log_file)

            with open(file=log_file_path, mode="a+") as f:
                f.write(
                    str(self.date)
                    + "\t"
                    + str(self.current_time)
                    + "\t"
                    + log_info
                    + "\n"
                )

                f.close()

        except Exception as e:
            raise e

    def start_log(self, key, class_name, method_name, log_file):
        """
        Method Name :   start_log
        Description :   This method creates an entry point log in DynamoDB

        Output      :   An entry point is created in DynamoDB
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """

        start_method_name = self.start_log.__name__

        try:
            func = lambda: "Entered" if key == "start" else "Exited"

            log_msg = f"{func()} {method_name} method of class {class_name}"

            self.log(log_file, log_msg)

        except Exception as e:
            error_msg = f"Exception occured in Class : {self.class_name}, Method : {start_method_name}, Error : {str(e)}"

            raise Exception(error_msg)

    def exception_log(self, error, class_name, method_name, log_file):
        """
        Method Name :   exception_log
        Description :   This method creates an exception log in DynamoDB and raises Exception

        Output      :   A exception log is created in DynamoDB and expection is raised
        On Failure  :   Write an exception log and then raise an exception

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """

        self.start_log("exit", class_name, method_name, log_file)

        exception_msg = f"Exception occured in Class : {class_name}, Method : {method_name}, Error : {str(error)}"

        self.log(log_file, exception_msg)

        raise Exception(exception_msg)
