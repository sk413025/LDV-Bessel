from dataclasses import dataclass
from typing import Dict
from ..utils.verification import PhysicsVerification
from ..utils.logger import logger
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
    
    def __post_init__(self):
        """初始化後驗證材料參數"""
        # 驗證材料穩定性
        verifications = self.verify_material_stability()
        
        # 使用 logger 打印所有驗證結果
        logger.info("\n材料參數驗證結果：")
        for name, result in verifications.items():
            status = "✓ 通過" if result.is_valid else "✗ 失敗"
            logger.info(f"{name}: {status}")
            logger.debug(f"  當前值: {result.actual_value:.2e}")
            logger.debug(f"  預期範圍: ({result.valid_range[0]:.2e}, {result.valid_range[1]:.2e})")
            logger.debug(f"  說明: {result.description}")
        
        # 檢查是否有任何驗證失敗
        failed_verifications = {
            name: result for name, result in verifications.items() 
            if not result.is_valid
        }
        
        if failed_verifications:
            warnings = []
            for name, result in failed_verifications.items():
                warnings.append(
                    f"{name}: actual_value={result.actual_value}, "
                    f"valid_range={result.valid_range}, "
                    f"description='{result.description}'"  # Changed from 'message' to 'description'
                )
            raise ValueError(
                "Material properties validation failed:\n" + 
                "\n".join(warnings)
            )
    
    @classmethod
    def create_cardboard(cls):
        """創建紙箱材料特性"""
        # 更新紙板材料參數
        density = 300  # kg/m³ (保持不變)
        E = 2e9       # Pa (保持不變)
        
        # 使用理論公式計算聲速
        theoretical_sound_speed = np.sqrt(E/density)  # 約2582 m/s
        
        # 考慮到實際紙板的多層結構和非均質性，使用較低的修正係數
        sound_speed = theoretical_sound_speed * 0.6  # 約1550 m/s
        
        return cls(
            density=density,
            youngs_modulus=E,
            sound_speed=sound_speed,  # 更新的聲速值
            damping_ratio=0.5,
            surface_roughness=5e-6,
            reflectivity=0.3,
            acoustic_impedance=density * sound_speed,
            poisson_ratio=0.3,
            yield_strain=0.001
        )
    
    def verify_material_stability(self) -> Dict[str, PhysicsVerification]:
        """驗證材料參數的物理穩定性"""
        verifications = {}
        
        # 聲速驗證 - 使用更寬鬆的範圍
        theoretical_sound_speed = np.sqrt(self.youngs_modulus / self.density)
        sound_speed_valid = 0.3 * theoretical_sound_speed < self.sound_speed < 2 * theoretical_sound_speed
        verifications['sound_speed'] = PhysicsVerification(
            is_valid=sound_speed_valid,
            actual_value=self.sound_speed,
            valid_range=(0.3 * theoretical_sound_speed, 2 * theoretical_sound_speed),
            description="聲速與材料特性相符性"
        )
        
        # 泊松比物理限制
        poisson_valid = -1.0 < self.poisson_ratio < 0.5
        verifications['poisson_ratio'] = PhysicsVerification(
            is_valid=poisson_valid,
            actual_value=self.poisson_ratio,
            valid_range=(-1.0, 0.5),
            description="��松比物理限制"  # Changed from 'message' to 'description'
        )
        
        # 阻尼比驗證
        damping_valid = 0 < self.damping_ratio < 1.0
        verifications['damping'] = PhysicsVerification(
            is_valid=damping_valid,
            actual_value=self.damping_ratio,
            valid_range=(0, 1.0),
            description="阻尼比合理性"  # Changed from 'message' to 'description'
        )
        
        # 反射率驗證
        reflectivity_valid = 0 < self.reflectivity <= 1.0
        verifications['reflectivity'] = PhysicsVerification(
            is_valid=reflectivity_valid,
            actual_value=self.reflectivity,
            valid_range=(0, 1.0),
            description="反射率物理限制"  # Changed from 'message' to 'description'
        )
        
        return verifications