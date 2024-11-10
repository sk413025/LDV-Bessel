from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Type
from ..utils.verification import PhysicsVerification
from ..utils.logger import logger
import numpy as np

@dataclass
class MaterialProperties(ABC):
    """材料物理特性基礎類別"""
    density: float          
    youngs_modulus: float  
    damping_ratio: float   
    poisson_ratio: float
    surface_roughness: float
    reflectivity: float
    
    @abstractmethod
    def get_vibration_parameters(self) -> Dict[str, float]:
        """獲取振動相關參數"""
        pass
    
    @abstractmethod
    def get_optical_parameters(self) -> Dict[str, float]:
        """獲取光學相關參數"""
        pass

@dataclass
class CardboardMaterial(MaterialProperties):
    """紙箱材料特性"""
    def __init__(self, **kwargs):
        super().__init__(
            density=300,          # [kg/m³]
            youngs_modulus=2e9,   # [Pa]
            damping_ratio=0.5,    # 紙箱結構阻尼
            poisson_ratio=0.3,    # 紙材典型值
            surface_roughness=5e-6,  # [m]
            reflectivity=0.3      # 紙箱表面反射率
        )
        # self._verify_parameters()
        
    def get_vibration_parameters(self) -> Dict[str, float]:
        return {
            'density': self.density,
            'youngs_modulus': self.youngs_modulus,
            'damping_ratio': self.damping_ratio,
            'poisson_ratio': self.poisson_ratio
        }
        
    def get_optical_parameters(self) -> Dict[str, float]:
        return {
            'surface_roughness': self.surface_roughness,
            'reflectivity': self.reflectivity
        }

@dataclass
class MetalMaterial(MaterialProperties):
    """金屬材料特性"""
    def __init__(self, metal_type: str = 'aluminum'):
        properties = METAL_PROPERTIES.get(metal_type, METAL_PROPERTIES['aluminum'])
        super().__init__(**properties)
        self.metal_type = metal_type
        # self._verify_parameters()
        
    def get_vibration_parameters(self) -> Dict[str, float]:
        return {
            'density': self.density,
            'youngs_modulus': self.youngs_modulus,
            'damping_ratio': self.damping_ratio,
            'poisson_ratio': self.poisson_ratio
        }
        
    def get_optical_parameters(self) -> Dict[str, float]:
        return {
            'surface_roughness': self.surface_roughness,
            'reflectivity': self.reflectivity
        }

# 材料屬性數據庫
METAL_PROPERTIES = {
    'aluminum': {
        'density': 2700,        # [kg/m³]
        'youngs_modulus': 69e9, # [Pa]
        'damping_ratio': 0.002, # 典型值
        'poisson_ratio': 0.33,
        'surface_roughness': 0.8e-6,  # [m]
        'reflectivity': 0.91
    },
    'steel': {
        'density': 7850,
        'youngs_modulus': 200e9,
        'damping_ratio': 0.001,
        'poisson_ratio': 0.29,
        'surface_roughness': 0.4e-6,
        'reflectivity': 0.95
    }
}

class MaterialFactory:
    """材料工廠類"""
    _materials: Dict[str, Type[MaterialProperties]] = {
        'cardboard': CardboardMaterial,
        'metal': MetalMaterial
    }
    
    @classmethod
    def create(cls, material_type: str, **kwargs) -> MaterialProperties:
        """創建材料實例
        
        Args:
            material_type: 材料類型 ('cardboard', 'metal')
            **kwargs: 材料特定參數
        """
        if material_type not in cls._materials:
            raise ValueError(f"未支援的材料類型: {material_type}")
            
        return cls._materials[material_type](**kwargs)