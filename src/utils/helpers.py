import yaml
import numpy as np

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def define_system_parameters(config_path='config/parameters.yaml'):
    config = load_config(config_path)
    
    params = {
        'A0': float(config['system']['A0']),
        'theta_incidence': np.deg2rad(float(config['system']['theta_incidence'])),
        'k_magnitude': float(config['system']['k_magnitude']),
        'f': float(config['system']['f']),
        'lambda_laser': float(config['system']['lambda_laser']),
        'beta': (4 * np.pi * float(config['system']['A0'])) / float(config['system']['lambda_laser']),
        'r': float(config['reconstruction']['r']),
        'x0': float(config['reconstruction']['x0']),
        'y0': float(config['reconstruction']['y0']),
        'x_range': [float(x) for x in config['grid']['x_range']],
        'y_range': [float(y) for y in config['grid']['y_range']],
        't_range': [float(t) for t in config['grid']['t_range']],
        'x_points': int(config['grid']['x_points']),
        'y_points': int(config['grid']['y_points']),
        't_points': int(config['grid']['t_points']),
        'theta_range': [np.deg2rad(float(t)) for t in config['optimization']['theta_range']],
        'theta_points': int(config['optimization']['theta_points'])
    }

    return params