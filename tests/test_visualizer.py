import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from src.visualization.visualizer import Visualizer

@pytest.fixture
def sample_data():
    original_intensity = np.random.rand(64, 64, 10)
    reconstructed_intensity = np.random.rand(64, 64, 10)
    history = {'obj_values': np.random.rand(100)}
    theta_list = np.linspace(0, np.pi/2, 181)
    s_est = np.random.rand(181)
    time_array = np.linspace(0, 1, 100)
    y_measurements = np.random.rand(100)
    y_predicted = np.random.rand(100)
    return (original_intensity, reconstructed_intensity, history, 
            theta_list, s_est, time_array, y_measurements, y_predicted)

def test_visualize_intensity_fields(sample_data):
    original_intensity, reconstructed_intensity, _, _, _, _, _, _ = sample_data
    fig = Visualizer.visualize_intensity_fields(original_intensity, reconstructed_intensity)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 4  # 包括两个主轴和两个颜色条
    main_axes = [ax for ax in fig.axes if not ax.get_label().startswith('Colorbar')]
    assert len(main_axes) == 2  # 检查主轴的数量
    colorbar_axes = [ax for ax in fig.axes if ax.get_label().startswith('Colorbar')]
    assert len(colorbar_axes) == 2  # 检查颜色条的数量
    plt.close(fig)

def test_plot_objective_function(sample_data):
    _, _, history, _, _, _, _, _ = sample_data
    fig = Visualizer.plot_objective_function(history)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    plt.close(fig)

def test_plot_reconstruction_spectrum(sample_data):
    _, _, _, theta_list, s_est, _, _, _ = sample_data
    fig = Visualizer.plot_reconstruction_spectrum(theta_list, s_est)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    plt.close(fig)

def test_plot_data_comparison(sample_data):
    _, _, _, _, _, time_array, y_measurements, y_predicted = sample_data
    fig = Visualizer.plot_data_comparison(time_array, y_measurements, y_predicted)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    plt.close(fig)

def test_analyze_peaks(sample_data):
    _, _, _, theta_list, s_est, _, _, _ = sample_data
    fig, peak_angles, peak_values = Visualizer.analyze_peaks(theta_list, s_est)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    assert isinstance(peak_angles, np.ndarray)
    assert isinstance(peak_values, np.ndarray)
    plt.close(fig)