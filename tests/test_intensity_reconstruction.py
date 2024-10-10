import numpy as np
import pytest
from src.models.intensity_reconstruction import IntensityReconstructor
from src.utils.helpers import define_system_parameters

@pytest.fixture
def reconstructor():
    params = define_system_parameters()
    return IntensityReconstructor(params)

def test_define_space_time_grid(reconstructor):
    x, y, t, X, Y, T = reconstructor.define_space_time_grid()
    assert x.shape == (reconstructor.params['x_points'],)
    assert y.shape == (reconstructor.params['y_points'],)
    assert t.shape == (reconstructor.params['t_points'],)
    assert X.shape == (reconstructor.params['x_points'], reconstructor.params['y_points'], reconstructor.params['t_points'])
    assert Y.shape == (reconstructor.params['x_points'], reconstructor.params['y_points'], reconstructor.params['t_points'])
    assert T.shape == (reconstructor.params['x_points'], reconstructor.params['y_points'], reconstructor.params['t_points'])

def test_define_measurement_region(reconstructor):
    x, y, t, X, Y, T = reconstructor.define_space_time_grid()
    measurement_mask, measurement_indices = reconstructor.define_measurement_region(X, Y)
    assert measurement_mask.shape == (reconstructor.params['x_points'], reconstructor.params['y_points'])
    assert len(measurement_indices) == 2
    assert measurement_mask.sum() > 0

def test_calculate_phase_modulation(reconstructor):
    x, y, t, X, Y, T = reconstructor.define_space_time_grid()
    phase = reconstructor.calculate_phase_modulation(X, Y, T)
    assert phase.shape == (reconstructor.params['x_points'], reconstructor.params['y_points'], reconstructor.params['t_points'])
    assert np.all(np.abs(phase) <= reconstructor.params['beta'])

def test_generate_interference_intensity(reconstructor):
    x, y, t, X, Y, T = reconstructor.define_space_time_grid()
    phase = reconstructor.calculate_phase_modulation(X, Y, T)
    intensity = reconstructor.generate_interference_intensity(phase)
    assert intensity.shape == (reconstructor.params['x_points'], reconstructor.params['y_points'], reconstructor.params['t_points'])
    assert np.all(intensity >= 0)

def test_reconstruct(reconstructor):
    x, y, t, X, Y, T = reconstructor.define_space_time_grid()
    phase = reconstructor.calculate_phase_modulation(X, Y, T)
    intensity = reconstructor.generate_interference_intensity(phase)
    measurement_mask, measurement_indices = reconstructor.define_measurement_region(X, Y)
    y_measurements = reconstructor.get_intensity_measurements(intensity, measurement_indices)
    theta_list = np.linspace(reconstructor.params['theta_range'][0], reconstructor.params['theta_range'][1], reconstructor.params['theta_points'])
    
    reconstructed_intensity, s_est, reconstruction_error, reconstructed_theta = reconstructor.reconstruct(y_measurements, theta_list)
    
    assert reconstructed_intensity.shape == intensity.shape
    assert s_est.shape == (reconstructor.params['theta_points'],)
    assert isinstance(reconstruction_error, float)
    assert reconstructor.params['theta_range'][0] <= reconstructed_theta <= reconstructor.params['theta_range'][1]