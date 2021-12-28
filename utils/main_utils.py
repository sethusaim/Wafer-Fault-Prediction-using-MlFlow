import json
import pickle
from io import StringIO

import pandas as pd
import yaml

from utils.logger import App_Logger

class_name = "main_utils.py"

log_writter = App_Logger()


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
        raise_error(
            class_name=class_name,
            method_name="make_readable",
            error=str(e),
        )


def convert_object_to_dataframe(obj):
    try:
        content = convert_object_to_bytes(obj)

        data = make_readable(content)

        df = pd.read_csv(data)

        return df

    except Exception as e:
        raise_error(
            class_name=class_name,
            method_name="convert_object_to_dataframe",
            error=str(e),
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
        raise_error(class_name=class_name, method_name="read_s3_obj", error=str(e))


def convert_object_to_pickle(obj):
    try:
        model_content = read_s3_obj(obj, decode=False)

        model = pickle.loads(model_content)

        return model

    except Exception as e:
        raise_error(
            class_name=class_name,
            method_name="convert_object_to_pickle",
            error=str(e),
        )


def convert_object_to_bytes(obj, db_name, collection_name):
    try:
        content = read_s3_obj(obj, decode=True)

        log_writter.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message="Converted s3 object to bytes",
        )

        return content

    except Exception as e:
        raise_error(
            class_name=class_name,
            method_name="convert_obj_to_bytes",
            error=str(e),
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
        raise_error(class_name=class_name, method_name="get_model_name", error=str(e))


def convert_obj_to_json(obj, db_name, collection_name):
    try:
        res = convert_object_to_bytes(obj)

        dic = json.loads(res)

        log_writter.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message="Converted s3 object to json",
        )

        return dic

    except Exception as e:
        raise_error(
            class_name=class_name,
            method_name="convert_obj_to_json",
            error=str(e),
        )


def read_params(config_path="params.yaml"):
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        return config

    except Exception as e:
        raise_error(class_name=class_name, method_name="read_params", error=str(e))


def raise_error(class_name, method_name, error):
    error_msg = f"Exception occured in Class : {class_name}, Method : {method_name}, Error : {error}"

    raise Exception(error_msg)


def raise_error_with_log(class_name, method_name, error, db_name, collection_name):
    error_msg = f"Exception occured in Class : {class_name}, Method : {method_name}, Error : {error}"

    log_writter.log(
        db_name=db_name, collection_name=collection_name, log_message=error_msg
    )
