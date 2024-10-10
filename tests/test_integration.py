import numpy as np
from src.models.intensity_reconstruction import IntensityReconstructor
from src.data.data_loader import DataLoader
from src.utils.helpers import define_system_parameters

def test_full_reconstruction_process():
    params = define_system_parameters()
    data_loader = DataLoader(params)
    reconstructor = IntensityReconstructor(params)

    data = data_loader.get_data(use_synthetic=True)
    reconstructed_intensity, s_est, reconstruction_error, reconstructed_theta = reconstructor.reconstruct(
        data['y_measurements'], 
        np.linspace(params['theta_range'][0], params['theta_range'][1], params['theta_points'])
    )

    assert reconstructed_intensity.shape == data['intensity'].shape
    assert reconstruction_error > 0
    assert params['theta_range'][0] <= reconstructed_theta <= params['theta_range'][1]