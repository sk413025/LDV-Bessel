import pytest
from src.utils.helpers import load_config, define_system_parameters

def test_load_config():
    config = load_config('config/parameters.yaml')
    assert 'system' in config
    assert 'reconstruction' in config
    assert 'grid' in config
    assert 'optimization' in config

def test_define_system_parameters():
    params = define_system_parameters()
    assert 'A0' in params
    assert 'theta_incidence' in params
    assert 'k_magnitude' in params
    assert 'f' in params
    assert 'lambda_laser' in params
    assert 'beta' in params