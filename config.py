import yaml


def load_config(config_file_path: str):
    with open(config_file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
