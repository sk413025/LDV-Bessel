from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
from .material import MaterialProperties

@dataclass
class OpticalSystem:
    """雷射都卜勒振動儀的光學系統類"""
    
    wavelength: float          # 雷射波長 [m]
    beam_diameter: float       # 光束直徑 [m] 
    focal_length: float        # 焦距 [m]
    working_distance: float    # 工作距離 [m]
    beam_divergence: float     # 光束發散角 [rad]
    optical_power: float       # 光功率 [W]
    
    def __init__(self,
                 wavelength: float = 632.8e-9,    # He-Ne雷射
                 beam_diameter: float = 1e-3,     # 1mm
                 focal_length: float = 0.2,       # 20cm
                 working_distance: float = 0.5,   # 50cm
                 beam_divergence: float = 1e-3,   # 1mrad
                 optical_power: float = 1e-3):    # 1mW
        """初始化光學系統參數"""
        self.wavelength = wavelength
        self.beam_diameter = beam_diameter
        self.focal_length = focal_length
        self.working_distance = working_distance
        self.beam_divergence = beam_divergence
        self.optical_power = optical_power
        
    def calculate_spot_size(self) -> float:
        """計算聚焦點大小
        
        Returns:
            float: 聚焦點直徑 [m]
        """
        return (4 * self.wavelength * self.focal_length) / (np.pi * self.beam_diameter)
    
    def calculate_rayleigh_range(self) -> float:
        """計算瑞利範圍
        
        Returns:
            float: 瑞利範圍 [m]
        """
        return (np.pi * self.beam_diameter**2) / (4 * self.wavelength)
    
    def calculate_intensity_distribution(self, 
                                      r: float, 
                                      z: float) -> float:
        """計算高斯光束強度分佈
        
        Args:
            r: 徑向距離 [m]
            z: 軸向距離 [m]
            
        Returns:
            float: 光強度 [W/m²]
        """
        w0 = self.calculate_spot_size() / 2  # 束腰半徑
        zR = self.calculate_rayleigh_range()  # 瑞利範圍
        
        w = w0 * np.sqrt(1 + (z/zR)**2)  # 光束半徑
        I0 = (2 * self.optical_power) / (np.pi * w0**2)  # 中心強度
        
        return I0 * (w0/w)**2 * np.exp(-2*(r/w)**2)
    
    def calculate_doppler_shift(self, velocity: float) -> float:
        """計算都卜勒頻移
        
        Args:
            velocity: 表面速度 [m/s]
            
        Returns:
            float: 頻移量 [Hz]
        """
        return 2 * velocity / self.wavelength
    
    def estimate_snr(self, 
                    surface: MaterialProperties,
                    distance: float) -> float:
        """估算訊噪比
        
        Args:
            surface: 材料特性
            distance: 測量距離 [m]
            
        Returns:
            float: 訊噪比 [dB]
        """
        # 考慮表面反射率和粗糙度的影響
        reflection_loss = -10 * np.log10(surface.reflectivity)
        roughness_loss = -20 * np.log10(self.wavelength / surface.surface_roughness)
        
        # 考慮距離衰減
        distance_loss = -20 * np.log10(distance / self.working_distance)
        
        # 基本訊噪比（假設在工作距離處為60dB）
        base_snr = 60
        
        return base_snr + reflection_loss + roughness_loss + distance_loss
    
    def get_system_parameters(self) -> Dict[str, float]:
        """獲取系統參數摘要
        
        Returns:
            Dict[str, float]: 系統參數字典
        """
        return {
            "spot_size": self.calculate_spot_size(),
            "rayleigh_range": self.calculate_rayleigh_range(),
            "max_frequency": self.calculate_doppler_shift(1.0),  # 1 m/s的頻移
            "beam_waist": self.calculate_spot_size() / 2
        }