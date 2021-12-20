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


def read_params(config_path="params.yaml"):
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        return config

    except Exception as e:
        raise Exception(
            f"Exception occured in main_utils.py, Method : read_params, Error : {str(e)}"
        )


def get_dataframe_from_bytes(content):
    try:
        data = make_readable(content)

        df = pd.read_csv(data)

        return df

    except Exception as e:
        raise Exception(
            f"Exception occured in main_utils.py, Method : read_params, Error : {str(e)}"
        )


def convert_object_to_bytes(obj):
    try:
        content = obj.get()["Body"].read().decode()

        return content

    except Exception as e:
        raise Exception(
            f"Exception occured in Class : S3_Operations, Method : convert_object_to_bytes, Error : {str(e)}"
        )
