from typing import List, Dict, Optional, Tuple, Callable
import numpy as np
from ..models.material import MaterialProperties
from ..models.system import SystemParameters
from ..models.modal import ClassicalModalAnalysis, BesselModalAnalysis
from ..utils.logger import logger

class SurfaceVibrationModel:
    """表面振動分析模型"""
    
    def __init__(self, 
                 system_params: SystemParameters,
                 analysis_type: str = "classical",
                 modal_analyzer: type = None):
        """
        初始化振動分析模型
        
        Args:
            system_params: 系統參數
            analysis_type: 分析類型 ("classical" 或 "bessel")
            modal_analyzer: 模態分析器類別 (ClassicalModalAnalysis 或 BesselModalAnalysis)
        """
        self.system_params = system_params
        self.analysis_type = analysis_type
        self.modal_analyzer = modal_analyzer
        self.modal_analysis = None
        self.box_dimensions = None  # 初始化為None
        
    def setup_modal_analysis(self, box_dimensions: Dict[str, float]):
        """設置模態分析"""
        self.box_dimensions = box_dimensions  # 保存box_dimensions

        if self.modal_analyzer is None:
            raise ValueError("No modal analyzer specified")
            
        self.modal_analysis = self.modal_analyzer(
            self.system_params,
            box_dimensions
        )
        
        # 計算並存儲模態頻率和形狀
        self.frequencies = self.modal_analysis.calculate_modal_frequencies()
        self.mode_shapes = self.modal_analysis.calculate_modal_shapes()
        
        # 驗證計算結果
        if not self.frequencies or not self.mode_shapes:
            raise RuntimeError("模態計算失敗")
        
        # 打印模態分析結果
        logger.info("\n模態分析結果：")
        logger.info(f"計算得到 {len(self.frequencies)} 個模態")
        logger.info(f"基頻: {self.frequencies[0]:.1f} Hz")
        logger.info(f"最高頻: {self.frequencies[-1]:.1f} Hz")

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
        if not hasattr(self, 'frequencies') or not hasattr(self, 'mode_shapes'):
            raise RuntimeError("請先執行setup_modal_analysis")
            
        if self.box_dimensions is None:
            raise RuntimeError("請先設置box_dimensions")
        
        # 改用板中心點而不是邊界點
        x = self.box_dimensions['length'] / 2 if x == 0 else x
        y = self.box_dimensions['width'] / 2 if y == 0 else y
        
        logger.debug("\n表面響應計算驗證：")
        logger.debug(f"實際計算位置: x={x:.3f}m, y={y:.3f}m")
        logger.debug(f"時間點數: {len(time_points)}")
        logger.debug(f"時間範圍: {time_points[0]:.3f}s - {time_points[-1]:.3f}s")
        
        # 計算位移和速度響應
        displacement = np.zeros_like(time_points)
        velocity = np.zeros_like(time_points)
        
        # 逐點計算響應
        for t_idx, t in enumerate(time_points):
            disp = self.modal_analysis.calculate_modal_response(x, y, t)
            displacement[t_idx] = disp
            
            # 計算速度（使用中心差分）
            if t_idx > 0 and t_idx < len(time_points)-1:
                dt = time_points[t_idx+1] - time_points[t_idx-1]
                velocity[t_idx] = (displacement[t_idx+1] - displacement[t_idx-1]) / dt
        
        # 增加模態響應分析的記錄
        logger.debug("\n模態分析統計:")
        mode_contributions = []
        for idx, freq in enumerate(self.frequencies):
            response = np.array([self.modal_analysis.calculate_modal_response(x, y, t) for t in time_points])
            contribution = np.sqrt(np.mean(response**2))
            mode_contributions.append((idx+1, freq, contribution))
            logger.debug(f"模態 {idx+1} (頻率: {freq:.1f} Hz):")
            logger.debug(f"  RMS貢獻: {contribution*1e9:.2f} nm")
            
        # 排序找出主要貢獻模態
        main_modes = sorted(mode_contributions, key=lambda x: x[2], reverse=True)[:3]
        logger.debug("\n主要貢獻模態:")
        for mode, freq, contrib in main_modes:
            logger.debug(f"模態 {mode} (頻率: {freq:.1f} Hz): {contrib*1e9:.2f} nm")
        
        # 儲存歷史數據
        self.displacement_history = displacement
        self.velocity_history = velocity
        
        # 輸出統計信息
        logger.debug("\n響應統計:")
        logger.debug(f"最大位移: {np.max(np.abs(displacement))*1e9:.2f} nm")
        logger.debug(f"最大速度: {np.max(np.abs(velocity))*1e3:.2f} mm/s")
        
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
            self.system_params.sampling_rate, 
            len(self.velocity_history)/self.system_params.sampling_rate
        )
        
        # 找出最大的n個峰值
        peak_indices = (-amp).argsort()[:n_peaks]
        peak_freqs = [(freqs[i], amp[i]) for i in peak_indices]
        return sorted(peak_freqs, key=lambda x: x[0])