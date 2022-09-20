import os

def get_configuration(keys):
    configuration = {}
    
    for this_key in keys:
        this_value = os.environ.get(this_key, '')
        configuration[this_key] = this_value

    return configuration
