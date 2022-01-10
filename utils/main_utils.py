import json
import pickle
from io import StringIO

import pandas as pd

from utils.exception import raise_exception_log
from utils.logger import App_Logger
from utils.read_params import read_params

log_writer = App_Logger()

config = read_params()

class_name = "main_utils.py"


def make_readable(data, db_name, collection_name):
    try:
        f = StringIO(data)

        log_writer.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message="Converted bytes content to string content using StringIO",
        )

        return f

    except Exception as e:
        raise_exception_log(
            error=e,
            class_name=class_name,
            method_name="make_readable",
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

        log_writer.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message=f"Converted {obj} to dataframe",
        )

        return df

    except Exception as e:
        raise_exception_log(
            error=e,
            class_name=class_name,
            method_name="convert_object_to_dataframe",
            db_name=db_name,
            collection_name=collection_name,
        )


def read_s3_obj(obj, db_name, collection_name, decode=True):
    try:
        if decode:
            content = obj.get()["Body"].read().decode()

            log_writer.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Read the object with decode as {decode}",
            )

            return content

        else:
            content = obj.get()["Body"].read()

            log_writer.log(
                db_name=db_name,
                collection_name=collection_name,
                log_message=f"Read the object with decode as {decode}",
            )
            return content

    except Exception as e:
        raise_exception_log(
            error=e,
            class_name=class_name,
            method_name="read_s3_obj",
            db_name=db_name,
            collection_name=collection_name,
        )


def convert_object_to_pickle(obj, db_name, collection_name):
    try:
        model_content = read_s3_obj(
            obj, decode=False, db_name=db_name, collection_name=collection_name
        )

        model = pickle.loads(model_content)

        log_writer.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message=f"Loaded {obj} as pickle model",
        )

        return model

    except Exception as e:
        raise_exception_log(
            error=e,
            class_name=class_name,
            method_name="convert_object_to_pickle",
            db_name=db_name,
            collection_name=collection_name,
        )


def convert_object_to_bytes(obj, db_name, collection_name):
    try:
        content = read_s3_obj(
            obj, decode=True, db_name=db_name, collection_name=collection_name
        )

        log_writer.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message="Converted s3 object to bytes",
        )

        return content

    except Exception as e:
        raise_exception_log(
            error=e,
            class_name=class_name,
            method_name="convert_object_to_bytes",
            db_name=db_name,
            collection_name=collection_name,
        )


def get_model_name(model, db_name, collection_name):
    method_name = get_model_name.__name__
    try:
        model_name = model.__class__.__name__

        log_writer.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message=f"Got the {model} model_name",
        )

        return model_name

    except Exception as e:
        raise_exception_log(
            error=e,
            class_name=class_name,
            method_name=method_name,
            db_name=db_name,
            collection_name=collection_name,
        )


def get_model_param_grid(model_key_name, db_name, collection_name):
    method_name = get_model_param_grid.__name__
    try:
        model_grid = {}

        model_param_name = config["model_params"][model_key_name]

        params_names = list(model_param_name.keys())

        for param in params_names:
            model_grid[param] = model_param_name[param]

        log_writer.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message=f"Inserted {model_key_name} params as to model_grid",
        )

        return model_grid

    except Exception as e:
        raise_exception_log(
            error=e,
            class_name=class_name,
            method_name=method_name,
            db_name=db_name,
            collection_name=collection_name,
        )


def convert_obj_to_json(obj, db_name, collection_name):
    try:
        res = convert_object_to_bytes(
            obj=obj, db_name=db_name, collection_name=collection_name
        )

        dic = json.loads(res)

        log_writer.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message="Converted s3 object to json",
        )

        return dic

    except Exception as e:
        raise_exception_log(
            error=e,
            class_name=class_name,
            method_name="convert_obj_to_json",
            db_name=db_name,
            collection_name=collection_name,
        )
