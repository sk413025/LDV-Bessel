import yaml
import numpy as np

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def define_system_parameters():
    A0 = 1e-7  # Vibration amplitude (m)
    theta_incidence = np.deg2rad(30)  # Incidence angle (rad)
    k_magnitude = 2 * np.pi / 0.02  # Wave number magnitude (rad/m)
    f = 1000  # Frequency (Hz)
    lambda_laser = 1e-6  # Laser wavelength (m)
    beta = (4 * np.pi * A0) / lambda_laser  # Phase modulation index (dimensionless)

    params = {
        'A0': A0,
        'theta_incidence': theta_incidence,
        'k_magnitude': k_magnitude,
        'f': f,
        'lambda_laser': lambda_laser,
        'beta': beta
    }

    return params