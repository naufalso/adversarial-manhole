import yaml

def load_yaml(config_path:str):
    # Loading YAML files
    with open(config_path, 'r') as f:
        data_loaded = yaml.safe_load(f)
        
    return data_loaded