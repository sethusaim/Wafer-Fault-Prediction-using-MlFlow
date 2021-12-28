import json
from io import StringIO

import pandas as pd
import yaml


def make_readable(data):
    try:
        f = StringIO(data)

        return f

    except Exception as e:
        raise Exception(
            f"Exception occured in main_utils.py, Method : make_readable, Error : {str(e)}"
        )


def convert_object_to_dataframe(obj):
    try:
        content = convert_object_to_bytes(obj)

        data = make_readable(content)

        df = pd.read_csv(data)

        return df

    except Exception as e:
        raise Exception(
            f"Exception occured in main_utils.py, Method : convert_object_to_dataframe, Error : {str(e)}"
        )


def convert_object_to_bytes(obj):
    try:
        content = obj.get()["Body"].read().decode()

        return content

    except Exception as e:
        raise Exception(
            f"Exception occured in main_utils.py, Method : convert_object_to_bytes, Error : {str(e)}"
        )


def get_model_name(model):
    try:
        model_name = model.__class__.__name__

        return model_name

    except Exception as e:
        raise Exception(
            f"Exception occured in main_utils.py, Method : get_model_name, Error : {str(e)}"
        )


def convert_obj_to_json(obj):
    try:
        res = convert_object_to_bytes(obj)

        dic = json.loads(res)

        return dic

    except Exception as e:
        raise Exception(
            f"Exception occured in main_utils.py, Method : convert_obj_to_json, Error : {str(e)}"
        )


def read_params(config_path="params.yaml"):
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        return config

    except Exception as e:
        raise Exception(
            f"Exception occured in main_utils.py, Method : read_params, Error : {str(e)}"
        )
