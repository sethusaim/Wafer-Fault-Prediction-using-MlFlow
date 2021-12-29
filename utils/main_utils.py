import json
import pickle
from io import StringIO

import pandas as pd
import yaml

from utils.logger import App_Logger

log_writer = App_Logger()


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
        error_msg = f"Exception occured in main_utils.py, Method : make_readable, Error : {str(e)}"

        log_writer.log(
            db_name=db_name, collection_name=collection_name, log_message=error_msg
        )

        raise Exception(error_msg)


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
        exception_msg = f"Exception occured in main_utils.py,, Method : convert_object_to_dataframe, Error : {str(e)}"

        log_writer.log(
            db_name=db_name, collection_name=collection_name, log_message=exception_msg
        )

        raise Exception(exception_msg)


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
        exception_msg = f"Exception occured in main_utils.py,, Method : read_s3_obj, Error : {str(e)}"

        log_writer.log(
            db_name=db_name, collection_name=collection_name, log_message=exception_msg
        )

        raise Exception(exception_msg)


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
        exception_msg = f"Exception occured in main_utils.py,, Method : convert_object_to_pickle, Error : {str(e)}"

        log_writer.log(
            db_name=db_name, collection_name=collection_name, log_message=exception_msg
        )

        raise Exception(exception_msg)


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
        exception_msg = (
            f"Exception occured in, Method : convert_object_to_bytes, Error : {str(e)}"
        )

        log_writer.log(
            db_name=db_name, collection_name=collection_name, log_message=exception_msg
        )

        raise Exception(exception_msg)


def get_model_name(model, db_name, collection_name):
    try:
        model_name = model.__class__.__name__

        log_writer.log(
            db_name=db_name,
            collection_name=collection_name,
            log_message=f"Got the {model} model_name",
        )

        return model_name

    except Exception as e:
        exception_msg = (
            f"Exception occured in, Method : get_model_name, Error : {str(e)}"
        )

        log_writer.log(
            db_name=db_name, collection_name=collection_name, log_message=exception_msg
        )

        raise Exception(exception_msg)


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
        exception_msg = (
            f"Exception occured in, Method : convert_obj_to_json, Error : {str(e)}"
        )

        log_writer.log(
            db_name=db_name, collection_name=collection_name, log_message=exception_msg
        )

        raise Exception(exception_msg)
