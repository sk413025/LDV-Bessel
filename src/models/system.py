from dataclasses import dataclass
from typing import Optional, Dict, Any
from ..utils.verification import PhysicsVerification
from .material import MaterialProperties
import numpy as np

@dataclass
class SystemParameters:
    """雷射都卜勒振動儀系統參數類"""
    material: MaterialProperties
    f_acoustic: float = 1000.0      # 聲波頻率 [Hz]
    sampling_rate: float = 1e4      # 採樣率 [Hz]
    measurement_time: float = 0.1   # 測量時間 [s]

    def __post_init__(self):
        """初始化後的額外設置"""
        if not isinstance(self.material, MaterialProperties):
            raise TypeError("material must be an instance of MaterialProperties")
        
        # 光學系統參數（單位註釋）
        self.wavelength = 632.8e-9      # HeNe雷射波長 [m]
        self.beam_diameter = 1e-3       # 光束直徑 [m]
        self.optical_power = 1e-3       # 光功率 [W]
        self.focal_length = 50e-3       # 焦距 [m]
        self.working_distance = 100e-3  # 工作距離 [m]
        self.scan_angle = np.deg2rad(10)  # 掃描角度 [rad]
        
        # 聲學參數
        self.acoustic_pressure = 0.2    # 聲壓 [Pa]
        self.theta_incidence = np.deg2rad(30)  # 入射角度 [rad]

        self.validate_parameters()
    
    def validate_parameters(self) -> Dict[str, PhysicsVerification]:
        """驗證系統參數"""
        return {
            'wavelength': PhysicsVerification(
                is_valid=0.1e-6 <= self.wavelength <= 10e-6,
                actual_value=self.wavelength,
                valid_range=(0.1e-6, 10e-6),
                description="雷射波長"
            ),
            'optical_power': PhysicsVerification(
                is_valid=0 < self.optical_power <= 100,
                actual_value=self.optical_power,
                valid_range=(0, 100),
                description="雷射功率"
            ),
            'beam_diameter': PhysicsVerification(
                is_valid=0 < self.beam_diameter <= 0.1,
                actual_value=self.beam_diameter,
                valid_range=(0, 0.1),
                description="光束直徑"
            ),
            'sampling_rate': PhysicsVerification(
                is_valid=1 <= self.sampling_rate <= 1e6,
                actual_value=self.sampling_rate,
                valid_range=(1, 1e6),
                description="採樣率"
            ),
            'focal_length': PhysicsVerification(
                is_valid=0 < self.focal_length <= 1,
                actual_value=self.focal_length,
                valid_range=(0, 1),
                description="焦距"
            ),
            'working_distance': PhysicsVerification(
                is_valid=0 < self.working_distance <= 1,
                actual_value=self.working_distance,
                valid_range=(0, 1),
                description="工作距離"
            ),
            'scan_angle': PhysicsVerification(
                is_valid=0 <= self.scan_angle <= np.pi/2,
                actual_value=self.scan_angle,
                valid_range=(0, np.pi/2),
                description="掃描角度"
            ),
            'theta_incidence': PhysicsVerification(
                is_valid=0 <= self.theta_incidence <= np.pi/2,
                actual_value=self.theta_incidence,
                valid_range=(0, np.pi/2),
                description="入射角度 [rad]"
            ),
            'f_acoustic': PhysicsVerification(
                is_valid=0 < self.f_acoustic <= 1e5,
                actual_value=self.f_acoustic,
                valid_range=(0, 1e5),
                description="聲波頻率 [Hz]"
            ),
            'acoustic_pressure': PhysicsVerification(
                is_valid=0 <= self.acoustic_pressure <= 1e3,
                actual_value=self.acoustic_pressure,
                valid_range=(0, 1e3),
                description="聲壓 [Pa]"
            )
        }

    def get_parameters_dict(self) -> Dict[str, Any]:
        """返回系統參數的字典形式"""
        return {
            'wavelength': self.wavelength,
            'beam_diameter': self.beam_diameter,
            'optical_power': self.optical_power,
            'focal_length': self.focal_length,
            'working_distance': self.working_distance,
            'scan_angle': self.scan_angle,
            'sampling_rate': self.sampling_rate,
            'measurement_time': self.measurement_time,
            'f_acoustic': self.f_acoustic,
            'acoustic_pressure': self.acoustic_pressure
        }
    
    def update_parameter(self, parameter_name: str, value: Any) -> None:
        """更新指定參數的值"""
        if hasattr(self, parameter_name):
            setattr(self, parameter_name, value)
            self.validate_parameters()
        else:
            raise AttributeError(f"找不到參數: {parameter_name}")