from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from ..utils.verification import PhysicsVerification
from .material import MaterialProperties
import numpy as np

@dataclass
class SystemParameters:
    material: MaterialProperties
    f_acoustic: float = 1000.0      # 聲波頻率 [Hz]
    sampling_rate: float = 1e4      # 採樣率 [Hz]
    measurement_time: float = 1.0   # 測量時間 [s]

    def __post_init__(self):
        """初始化後的額外設置"""
        if not isinstance(self.material, MaterialProperties):
            raise TypeError("material must be an instance of MaterialProperties")
        
        # 先定義屬性
        # ��學參數（單位註釋）
        self.wavelength = 632.8e-9      # HeNe雷射波長 [m]
        self.beam_diameter = 1e-3       # 光束直徑 [m]
        self.optical_power = 1e-3       # 光功率 [W]
        self.w_0 = self.beam_diameter / 2  # 光束腰半徑 [m]
        self.z_R = (np.pi * self.w_0**2) / self.wavelength  # 瑞利長度 [m]
        self.beam_radius = self.beam_diameter / 2  # 光束半徑 [m]
        # 添加缺少的屬性（單位註釋）
        self.focal_length = 50e-3       # 焦距 [m]
        self.working_distance = 100e-3  # 工作距離 [m]
        self.scan_angle = np.deg2rad(10)  # 掃描角度 [rad]

        # 振動參數
        self.force_amplification = 1.5
        self.mass_correction = 0.3
        self.boundary_factor = 0.8
        self.acoustic_attenuation = 1.0
        self.acoustic_pressure = 0.2
        
        # 添加入射角度參數（單位註釋）
        self.theta_incidence = np.deg2rad(30)  # 入射角度 [rad]
        
        # 其他物理參數
        self.sound_pressure = 0.2
        self.Q_factor = 10.0

        # 添加阻尼比（無單位）
        self.material_damping_ratio = self.material.damping_ratio

        # 在所有屬性定義之後再進行驗證
        self.validate_parameters()
    
    def validate_parameters(self) -> Dict[str, PhysicsVerification]:
        """驗證所有系統參數"""
        validations = {
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