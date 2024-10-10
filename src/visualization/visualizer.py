import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

class Visualizer:
    @staticmethod
    def visualize_intensity_fields(original_intensity, reconstructed_intensity):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        mid_time = original_intensity.shape[2] // 2
        
        im1 = ax1.imshow(original_intensity[:, :, mid_time], cmap='viridis')
        ax1.set_title('Original Intensity')
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.ax.set_label('Colorbar 1')  # 给颜色条添加标签
        
        im2 = ax2.imshow(reconstructed_intensity[:, :, mid_time], cmap='viridis')
        ax2.set_title('Reconstructed Intensity')
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.ax.set_label('Colorbar 2')  # 给颜色条添加标签
        
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_objective_function(history):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history['obj_values'])
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Function Value')
        ax.set_title('Optimization Progress')
        return fig

    @staticmethod
    def plot_reconstruction_spectrum(theta_list, s_est):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(np.degrees(theta_list), s_est)
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Reconstruction Coefficient')
        ax.set_title('Reconstruction Spectrum')
        return fig

    @staticmethod
    def plot_data_comparison(time_array, y_measurements, y_predicted):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(time_array, y_measurements, label='Measured')
        ax.plot(time_array, y_predicted, label='Predicted')
        ax.set_xlabel('Time')
        ax.set_ylabel('Intensity')
        ax.set_title('Data Comparison')
        ax.legend()
        return fig

    @staticmethod
    def analyze_peaks(theta_list, s_est):
        peaks, _ = find_peaks(s_est, height=0)
        peak_angles = np.degrees(theta_list[peaks])
        peak_values = s_est[peaks]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(np.degrees(theta_list), s_est)
        ax.plot(peak_angles, peak_values, 'ro')
        
        for angle, value in zip(peak_angles, peak_values):
            ax.annotate(f'{angle:.2f}°', (angle, value), textcoords="offset points", xytext=(0,10), ha='center')
        
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Reconstruction Coefficient')
        ax.set_title('Peak Analysis')
        return fig, peak_angles, peak_values