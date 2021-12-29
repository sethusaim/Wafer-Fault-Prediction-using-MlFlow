import yaml
from utils.main_utils import raise_exception


def read_params(config_path="params.yaml"):
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        return config

    except Exception as e:
        raise_exception(
            class_name="read_params.py", method_name="read_params", exception=str(e)
        )
