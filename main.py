from src.models.intensity_reconstruction import IntensityReconstructor
from src.visualization.visualizer import Visualizer
from src.data.data_loader import DataLoader
from src.utils.helpers import define_system_parameters
import matplotlib.pyplot as plt
import numpy as np

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
    
    fig1 = visualizer.visualize_intensity_fields(data['intensity'], reconstructed_intensity)
    fig2 = visualizer.plot_objective_function(reconstructor.history)
    fig3 = visualizer.plot_reconstruction_spectrum(
        np.linspace(params['theta_range'][0], params['theta_range'][1], params['theta_points']), 
        s_est
    )
    fig4, peak_angles, peak_values = visualizer.analyze_peaks(
        np.linspace(params['theta_range'][0], params['theta_range'][1], params['theta_points']), 
        s_est
    )

    # Display results
    print(f"Reconstruction Error: {reconstruction_error}")
    print(f"Reconstructed Theta: {np.degrees(reconstructed_theta):.2f} degrees")
    print(f"Peak Angles: {peak_angles}")
    print(f"Peak Values: {peak_values}")

    plt.show()

if __name__ == "__main__":
    main()