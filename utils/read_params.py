import yaml


def read_params(config_path="params.yaml"):
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        return config

    except Exception as e:
        raise Exception(f"Exception occured in method : read_params, Error : {str(e)}")
