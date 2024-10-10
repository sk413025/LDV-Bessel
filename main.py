from src.models.intensity_reconstruction import IntensityReconstructor
from src.visualization.visualizer import Visualizer
from src.data.data_loader import DataLoader
from src.utils.helpers import load_config, define_system_parameters

def main():
    # Load configuration
    config = load_config('config/parameters.yaml')

    # Define system parameters
    params = define_system_parameters()

    # Load or generate data
    data_loader = DataLoader()
    # For synthetic data:
    X, Y, T, intensity, measurement_indices = data_loader.generate_synthetic_data(params)
    # For experimental data:
    # data = data_loader.load_experimental_data('path/to/data.csv')

    # Initialize and run reconstruction
    reconstructor = IntensityReconstructor(params)
    reconstructed_intensity, s_est, reconstruction_error, reconstructed_theta = reconstructor.reconstruct(
        intensity[measurement_indices], np.linspace(0, np.pi/2, 181)
    )

    # Visualize results
    visualizer = Visualizer()
    visualizer.visualize_intensity_fields(intensity, reconstructed_intensity)
    visualizer.plot_objective_function(reconstructor.history)
    visualizer.plot_reconstruction_spectrum(np.linspace(0, np.pi/2, 181), s_est)
    visualizer.analyze_peaks(np.linspace(0, np.pi/2, 181), s_est)

if __name__ == "__main__":
    main()