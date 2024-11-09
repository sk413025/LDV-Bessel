import numpy as np
from typing import Optional, Tuple, List
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import matplotlib.pyplot as plt

class PlottingUtils:
    """繪圖工具類，提供各種視覺化功能"""
    
    @staticmethod
    def create_figure(figsize: Tuple[float, float] = (10, 6)) -> Tuple[Figure, Axes]:
        """創建新的圖表"""
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax

    @staticmethod
    def plot_frequency_response(
        frequencies: np.ndarray,
        amplitude: np.ndarray,
        title: str = "頻率響應",
        xlabel: str = "頻率 (Hz)",
        ylabel: str = "振幅",
        ax: Optional[Axes] = None
    ) -> Axes:
        """繪製頻率響應圖"""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(frequencies, amplitude)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        return ax

    @staticmethod
    def plot_time_series(
        time: np.ndarray,
        signal: np.ndarray,
        title: str = "時域響應",
        xlabel: str = "時間 (s)",
        ylabel: str = "振幅",
        ax: Optional[Axes] = None
    ) -> Axes:
        """繪製時域響應圖"""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
            
        ax.plot(time, signal)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        return ax

    @staticmethod
    def plot_mode_shape(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        title: str = "模態形狀",
        ax: Optional[Axes] = None
    ) -> Axes:
        """繪製模態形狀"""
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
        surf = ax.plot_surface(x, y, z, cmap='viridis')
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('位移')
        plt.colorbar(surf, ax=ax, label='振幅')
        return ax

    @staticmethod
    def save_figure(
        fig: Figure,
        filename: str,
        dpi: int = 300,
        bbox_inches: str = 'tight'
    ) -> None:
        """儲存圖表"""
        fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)