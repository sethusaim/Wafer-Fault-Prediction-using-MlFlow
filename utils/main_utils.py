import json
import pickle
from io import StringIO

import pandas as pd
from utils.logger import App_Logger

log_writter = App_Logger()

class_name = "main_utils.py"


def make_readable(data, db_name, collection_name):
    try:
        f = StringIO(data)

        log_writter.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message="Converted bytes content to string content using StringIO",
        )

        return f

    except Exception as e:
        raise_exception(
            class_name=class_name,
            method_name="make_readable",
            exception=str(e),
            db_name=db_name,
            collection_name=collection_name,
        )


def convert_object_to_dataframe(obj, db_name, collection_name):
    try:
        content = convert_object_to_bytes(
            obj, db_name=db_name, collection_name=collection_name
        )

        data = make_readable(
            data=content, db_name=db_name, collection_name=collection_name
        )

        df = pd.read_csv(data)

        log_writter.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message=f"Converted {obj} to dataframe",
        )

        return df

    except Exception as e:
        raise_exception(
            class_name=class_name,
            method_name="convert_object_to_dataframe",
            exception=str(e),
            db_name=db_name,
            collection_name=collection_name,
        )


def read_s3_obj(obj, db_name, collection_name, decode=True):
    try:
        if decode:
            content = obj.get()["Body"].read().decode()

            log_writter.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Read the object with decode as {decode}",
            )

            return content

        else:
            content = obj.get()["Body"].read()

            log_writter.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Read the object with decode as {decode}",
            )
            return content

    except Exception as e:
        raise_exception(
            class_name=class_name,
            method_name="read_s3_obj",
            exception=str(e),
            db_name=db_name,
            collection_name=collection_name,
        )


def convert_object_to_pickle(obj, db_name, collection_name):
    try:
        model_content = read_s3_obj(
            obj, decode=False, db_name=db_name, collection_name=collection_name
        )

        model = pickle.loads(model_content)

        log_writter.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message=f"Loaded {obj} as pickle model",
        )

        return model

    except Exception as e:
        raise_exception(
            class_name=class_name,
            method_name="convert_object_to_pickle",
            error=str(e),
            db_name=db_name,
            collection_name=collection_name,
        )


def convert_object_to_bytes(obj, db_name, collection_name):
    try:
        content = read_s3_obj(
            obj, decode=True, db_name=db_name, collection_name=collection_name
        )

        log_writter.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message="Converted s3 object to bytes",
        )

        return content

    except Exception as e:
        raise_exception(
            class_name=class_name,
            method_name="convert_object_to_bytes",
            exception=str(e),
            db_name=db_name,
            collection_name=collection_name,
        )


def get_model_name(model, db_name, collection_name):
    try:
        model_name = model.__class__.__name__

        log_writter.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message=f"Got the {model} model_name",
        )

        return model_name

    except Exception as e:
        raise_exception(
            class_name=class_name,
            method_name="get_model_name",
            exception=str(e),
            db_name=db_name,
            collection_name=collection_name,
        )


def convert_obj_to_json(obj, db_name, collection_name):
    try:
        res = convert_object_to_bytes(
            obj=obj, db_name=db_name, collection_name=collection_name
        )

        dic = json.loads(res)

        log_writter.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message="Converted s3 object to json",
        )

        return dic

    except Exception as e:
        raise_exception(
            class_name=class_name,
            method_name="convert_obj_to_json",
            exception=str(e),
            db_name=db_name,
            collection_name=collection_name,
        )


def raise_exception(
    class_name, method_name, exception, db_name=None, collection_name=None
):
    try:
        exception_msg = f"Exception occured in Class : {class_name}, Method : {method_name}, Error : {exception}"

        if db_name == None and collection_name == None:
            raise Exception(exception_msg)

        else:
            log_writter.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=exception_msg,
            )

            raise Exception(exception_msg)

    except Exception as e:
        raise e
