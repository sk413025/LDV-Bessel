from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from ..utils.verification import PhysicsVerification
from .material import MaterialProperties
import numpy as np

@dataclass
class SystemParameters:
    """系統參數設置"""
    material: MaterialProperties
    f_acoustic: float = 1000.0      # 聲波頻率 [Hz]
    Q_factor: float = 10.0          # 品質因子
    boundary_factor: float = 0.8    # 邊界條件修正因子
    working_distance: float = 0.5    # 工作距離 [m]
    wavelength: float = 632.8e-9    # He-Ne雷射
    power: float = 1e-3            # 1mW
    beam_radius: float = 0.5e-3    # 0.5mm
    focal_length: float = 0.2      # 20cm
    scan_angle: float = 1e-3       # 1mrad
    sampling_rate: float = 1e4     # 10kHz
    measurement_time: float = 1.0   # 1s

    def __post_init__(self):
        """初始化後的額外設置"""
        if not isinstance(self.material, MaterialProperties):
            raise TypeError("material must be an instance of MaterialProperties")
        self.validate_parameters()
        
        # 光學參數
        self.wavelength = 632.8e-9  # HeNe雷射波長 [m]
        self.beam_diameter = 1e-3   # 光束直徑 [m]
        self.optical_power = 1e-3   # 光功率 [W]
        
        # 計算衍生參數
        self.w_0 = self.beam_diameter / 2  # 光束腰半徑 [m]
        self.z_R = (np.pi * self.w_0**2) / self.wavelength  # 瑞利長度 [m]
    
    def validate_parameters(self) -> Dict[str, PhysicsVerification]:
        """驗證所有系統參數"""
        validations = {
            'wavelength': PhysicsVerification(
                is_valid=0.1e-6 <= self.wavelength <= 10e-6,
                actual_value=self.wavelength,
                valid_range=(0.1e-6, 10e-6),
                description="雷射波長"
            ),
            'power': PhysicsVerification(
                is_valid=0 < self.power <= 100,
                actual_value=self.power,
                valid_range=(0, 100),
                description="雷射功率"
            ),
            'beam_radius': PhysicsVerification(
                is_valid=0 < self.beam_radius <= 0.1,
                actual_value=self.beam_radius,
                valid_range=(0, 0.1),
                description="光束半徑"
            ),
            'sampling_rate': PhysicsVerification(
                is_valid=1 <= self.sampling_rate <= 1e6,
                actual_value=self.sampling_rate,
                valid_range=(1, 1e6),
                description="採樣率"
            )
        }
        
        return validations
    
    def get_parameters_dict(self) -> Dict[str, Any]:
        """返回所有參數的字典形式"""
        return {
            'wavelength': self.wavelength,
            'power': self.power,
            'beam_radius': self.beam_radius,
            'focal_length': self.focal_length,
            'working_distance': self.working_distance,
            'scan_angle': self.scan_angle,
            'sampling_rate': self.sampling_rate,
            'measurement_time': self.measurement_time,
            'f_acoustic': self.f_acoustic,
            'Q_factor': self.Q_factor,
            'boundary_factor': self.boundary_factor
        }
    
    def update_parameter(self, parameter_name: str, value: Any) -> None:
        """更新指定參數的值"""
        if hasattr(self, parameter_name):
            setattr(self, parameter_name, value)
            self.validate_parameters()
        else:
            raise AttributeError(f"找不到參數: {parameter_name}")