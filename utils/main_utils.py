from io import StringIO
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
        raise Exception(f"Exception occured in method : read_params, Error : {str(e)}")
