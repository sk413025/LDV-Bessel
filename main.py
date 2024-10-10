from src.models.intensity_reconstruction import IntensityReconstructor
from src.visualization.visualizer import Visualizer
from src.data.data_loader import DataLoader
from src.utils.helpers import define_system_parameters

def main():
    # Load configuration
    params = define_system_parameters()

    # Initialize data loader
    data_loader = DataLoader(params)

    # Get data (synthetic for now)
    data = data_loader.get_data(use_synthetic=True)

    # Initialize and run reconstruction
    reconstructor = IntensityReconstructor(params)
    reconstructed_intensity, s_est, reconstruction_error, reconstructed_theta = reconstructor.reconstruct(
        data['y_measurements'], 
        np.linspace(params['theta_range'][0], params['theta_range'][1], params['theta_points'])
    )

    # Visualize results
    visualizer = Visualizer()
    visualizer.visualize_intensity_fields(data['intensity'], reconstructed_intensity)
    visualizer.plot_objective_function(reconstructor.history)
    visualizer.plot_reconstruction_spectrum(
        np.linspace(params['theta_range'][0], params['theta_range'][1], params['theta_points']), 
        s_est
    )
    visualizer.analyze_peaks(
        np.linspace(params['theta_range'][0], params['theta_range'][1], params['theta_points']), 
        s_est
    )

if __name__ == "__main__":
    main()