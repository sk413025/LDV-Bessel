from typing import List, Dict, Optional, Tuple, Callable
import numpy as np
from ..models.material import MaterialProperties
from ..models.system import SystemParameters
from ..models.modal import ClassicalModalAnalysis, BesselModalAnalysis

class SurfaceVibrationModel:
    """表面振動模型類，用於模擬和分析表面振動行為"""
    
    def __init__(self, 
                 system_params: SystemParameters,
                 analysis_type: str = "classical"):
        """
        初始化表面振動模型
        
        Args:
            system_params: 系統參數物件
            analysis_type: 分析類型 ("classical" 或 "bessel")
        """
        self.params = system_params
        self.analysis_type = analysis_type
        self.modal_analyzer = None
        self.displacement_history = []
        self.velocity_history = []
        
    def setup_modal_analysis(self, box_dimensions: Dict[str, float]) -> None:
        """
        設置模態分析器
        
        Args:
            box_dimensions: 結構尺寸參數 {"length": float, "width": float, "thickness": float}
        """
        if self.analysis_type == "classical":
            self.modal_analyzer = ClassicalModalAnalysis(self.params, box_dimensions)
        elif self.analysis_type == "bessel":
            self.modal_analyzer = BesselModalAnalysis(self.params, box_dimensions)
        else:
            raise ValueError("不支援的分析類型")
            
    def calculate_surface_response(self, 
                                 x: float, 
                                 y: float, 
                                 time_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        計算表面振動響應
        
        Args:
            x: x座標位置
            y: y座標位置
            time_points: 時間點陣列
            
        Returns:
            displacement: 位移時間序列
            velocity: 速度時間序列
        """
        if self.modal_analyzer is None:
            raise RuntimeError("請先設置模態分析器")
            
        # 計算模態頻率和形狀
        frequencies = self.modal_analyzer.calculate_modal_frequencies()
        mode_shapes = self.modal_analyzer.calculate_modal_shapes()
        
        # 計算位移和速度響應
        displacement = np.zeros_like(time_points)
        velocity = np.zeros_like(time_points)
        
        for t_idx, t in enumerate(time_points):
            # 計算位移
            displacement[t_idx] = self.modal_analyzer.calculate_modal_response(x, y, t)
            
            # 計算速度（位移的時間導數）
            if t_idx > 0:
                dt = time_points[t_idx] - time_points[t_idx-1]
                velocity[t_idx] = (displacement[t_idx] - displacement[t_idx-1]) / dt
                
        self.displacement_history = displacement
        self.velocity_history = velocity
        
        return displacement, velocity
    
    def get_frequency_response(self, 
                             sampling_rate: float, 
                             duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        獲取頻率響應
        
        Args:
            sampling_rate: 採樣率 (Hz)
            duration: 訊號持續時間 (s)
            
        Returns:
            frequencies: 頻率陣列
            amplitude: 振幅陣列
        """
        if len(self.velocity_history) == 0:
            raise RuntimeError("請先計算表面響應")
            
        # 計算FFT
        n_samples = int(sampling_rate * duration)
        frequencies = np.fft.fftfreq(n_samples, 1/sampling_rate)
        fft_result = np.fft.fft(self.velocity_history)
        amplitude = np.abs(fft_result)
        
        # 只返回正頻率部分
        positive_freq_idx = frequencies > 0
        return frequencies[positive_freq_idx], amplitude[positive_freq_idx]
    
    def calculate_rms_velocity(self, window_size: int = 1000) -> float:
        """
        計算RMS速度值
        
        Args:
            window_size: 計算窗口大小
            
        Returns:
            rms_velocity: RMS速度值
        """
        if len(self.velocity_history) == 0:
            raise RuntimeError("請先計算表面響應")
            
        velocity_squared = np.square(self.velocity_history)
        rms_velocity = np.sqrt(np.mean(velocity_squared[-window_size:]))
        return rms_velocity
    
    def get_peak_frequencies(self, 
                           n_peaks: int = 3) -> List[Tuple[float, float]]:
        """
        獲取主要振動頻率
        
        Args:
            n_peaks: 返回的峰值數量
            
        Returns:
            peak_freqs: [(頻率, 振幅), ...]的列表
        """
        if len(self.velocity_history) == 0:
            raise RuntimeError("請先計算表面響應")
            
        freqs, amp = self.get_frequency_response(
            self.params.sampling_rate, 
            len(self.velocity_history)/self.params.sampling_rate
        )
        
        # 找出最大的n個峰值
        peak_indices = (-amp).argsort()[:n_peaks]
        peak_freqs = [(freqs[i], amp[i]) for i in peak_indices]
        return sorted(peak_freqs, key=lambda x: x[0])