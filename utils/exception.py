from utils.logger import App_Logger

log_writer = App_Logger()


def raise_exception(error, class_name, method_name, db_name, collection_name):
    exception_msg = f"Exception occured in Class : {class_name}, Method : {method_name}, Error : {str(error)}"

    log_writer.log(
        db_name=db_name, collection_name=collection_name, log_message=exception_msg,
    )

    raise Exception(exception_msg)
