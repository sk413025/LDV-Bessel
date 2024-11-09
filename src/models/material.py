from dataclasses import dataclass
from typing import Dict
from ..utils.verification import PhysicsVerification
import numpy as np

@dataclass
class MaterialProperties:
    """材料物理特性"""
    density: float          # [kg/m³]
    youngs_modulus: float  # [Pa]
    sound_speed: float     # [m/s]
    damping_ratio: float   # 阻尼比
    surface_roughness: float  # [m]
    reflectivity: float    # 反射率
    acoustic_impedance: float  # [kg/m²s]
    poisson_ratio: float = 0.3  # 泊松比
    yield_strain: float = 0.001  # 降伏應變
    
    @classmethod
    def create_cardboard(cls):
        """創建紙箱材料特性"""
        density = 300  # kg/m³
        sound_speed = 300  # m/s
        return cls(
            density=density,
            youngs_modulus=2e9,    # 紙板楊氏模數
            sound_speed=sound_speed,
            damping_ratio=0.5,     # 較高的阻尼比
            surface_roughness=5e-6, # 修改為5µm
            reflectivity=0.3,      # 漫反射率
            acoustic_impedance=density * sound_speed,  # 聲阻抗
            poisson_ratio=0.3,     # 紙板的泊松比
            yield_strain=0.001     # 紙板的降伏應變
        )
    
    def verify_material_stability(self) -> Dict[str, PhysicsVerification]:
        """驗證材料參數的物理穩定性"""
        verifications = {}
        
        # 聲速與材料特性關係驗證
        theoretical_sound_speed = np.sqrt(self.youngs_modulus / self.density)
        sound_speed_valid = 0.1 * theoretical_sound_speed < self.sound_speed < 10 * theoretical_sound_speed
        verifications['sound_speed'] = PhysicsVerification(
            sound_speed_valid,
            self.sound_speed,
            (0.1 * theoretical_sound_speed, 10 * theoretical_sound_speed),
            "聲速與材料特性相符性"
        )
        
        # 泊松比物理限制
        poisson_valid = -1.0 < self.poisson_ratio < 0.5
        verifications['poisson_ratio'] = PhysicsVerification(
            poisson_valid,
            self.poisson_ratio,
            (-1.0, 0.5),
            "泊松比物理限制"
        )
        
        # 阻尼比驗證
        damping_valid = 0 < self.damping_ratio < 1.0
        verifications['damping'] = PhysicsVerification(
            damping_valid,
            self.damping_ratio,
            (0, 1.0),
            "阻尼比合理性"
        )
        
        # 反射率驗證
        reflectivity_valid = 0 < self.reflectivity <= 1.0
        verifications['reflectivity'] = PhysicsVerification(
            reflectivity_valid,
            self.reflectivity,
            (0, 1.0),
            "反射率物理限制"
        )
        
        return verifications