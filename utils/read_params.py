import yaml


def read_params(config_path="params.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config
