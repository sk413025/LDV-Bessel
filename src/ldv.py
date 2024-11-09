from typing import Dict, Tuple, Optional, List
import numpy as np
from .models.system import SystemParameters, OpticalSystem
from .models.material import MaterialProperties
from .analysis.vibration import SurfaceVibrationModel

class LaserDopplerVibrometer:
    """雷射都卜勒振動儀主類"""

    def __init__(self,
                 system_params: SystemParameters,
                 material: MaterialProperties,
                 analysis_type: str = "classical"):
        """
        初始化LDV系統
        
        Args:
            system_params: 系統參數
            material: 待測物材料特性
            analysis_type: 分析類型 ("classical" 或 "bessel")
        """
        self.params = system_params
        self.material = material
        self.optical_system = OpticalSystem(
            wavelength=system_params.wavelength,
            beam_diameter=2*system_params.beam_radius,
            focal_length=system_params.focal_length,
            working_distance=system_params.working_distance,
            beam_divergence=system_params.scan_angle,
            optical_power=system_params.power
        )
        self.vibration_model = SurfaceVibrationModel(
            system_params=system_params,
            analysis_type=analysis_type
        )
        
    def setup_measurement(self, box_dimensions: Dict[str, float]) -> None:
        """
        設置測量參數
        
        Args:
            box_dimensions: 結構尺寸參數 {"length": float, "width": float, "thickness": float}
        """
        self.vibration_model.setup_modal_analysis(box_dimensions)
        
    def measure_point(self, 
                     x: float, 
                     y: float, 
                     duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        測量單點振動
        
        Args:
            x: x座標位置
            y: y座標位置
            duration: 測量時間
            
        Returns:
            time_points: 時間陣列
            velocities: 速度陣列
        """
        time_points = np.linspace(0, duration, int(self.params.sampling_rate * duration))
        _, velocities = self.vibration_model.calculate_surface_response(x, y, time_points)
        
        return time_points, velocities
    
    def get_frequency_spectrum(self, 
                             velocities: np.ndarray, 
                             duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        計算頻譜
        
        Args:
            velocities: 速度時間序列
            duration: 測量時間
            
        Returns:
            frequencies: 頻率陣列
            amplitudes: 振幅陣列
        """
        return self.vibration_model.get_frequency_response(
            self.params.sampling_rate,
            duration
        )
    
    def get_measurement_quality(self, distance: float) -> Dict[str, float]:
        """
        評估測量品質
        
        Args:
            distance: 實際測量距離
            
        Returns:
            Dict: 包含SNR等測量品質指標
        """
        snr = self.optical_system.estimate_snr(self.material, distance)
        spot_size = self.optical_system.calculate_spot_size()
        
        return {
            "snr": snr,
            "spot_size": spot_size,
            "max_velocity": self.params.wavelength * self.params.sampling_rate / 4,
            "spatial_resolution": spot_size
        }
    
    def get_system_status(self) -> Dict[str, Dict]:
        """
        獲取系統狀態摘要
        
        Returns:
            Dict: 系統狀態資訊
        """
        return {
            "optical_parameters": self.optical_system.get_system_parameters(),
            "system_parameters": self.params.get_parameters_dict(),
            "material_properties": self.material.__dict__
        }