import yaml

def read_config_file(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config