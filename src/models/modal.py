from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import numpy as np
from scipy.special import jv
from .system import SystemParameters
from .material import MaterialProperties

class ModalAnalysisBase(ABC):
    """模態分析基礎類別"""
    def __init__(self, params: SystemParameters):
        self.params = params
        
    @abstractmethod
    def calculate_modal_frequencies(self) -> List[float]:
        """計算結構自然頻率"""
        pass
        
    @abstractmethod
    def calculate_modal_shapes(self) -> List[callable]:
        """計算模態形狀函數"""
        pass
        
    @abstractmethod
    def calculate_modal_response(self, x: float, y: float, t: float) -> float:
        """計算模態響應"""
        pass

class ClassicalModalAnalysis(ModalAnalysisBase):
    """傳統模態分析實現"""
    def __init__(self, params: SystemParameters, box_dimensions: Dict):
        super().__init__(params)
        self.box_dimensions = box_dimensions
        self._initialize_parameters()
        self.modal_frequencies = []
        self.modal_shapes = []
        
    def _initialize_parameters(self):
        h = self.box_dimensions['thickness']
        E = self.params.material.youngs_modulus
        v = self.params.material.poisson_ratio
        self.bending_stiffness = (E * h**3) / (12 * (1 - v**2))
    
    def calculate_modal_frequencies(self) -> List[float]:
        frequencies = []
        L = self.box_dimensions['length']
        W = self.box_dimensions['width']
        
        for i in range(1, 4):
            for j in range(1, 4):
                # 改進的頻率方程，考慮二維振動
                f_ij = (np.pi/2) * np.sqrt(self.bending_stiffness/
                       self.params.material.density) * ((i/L)**2 + (j/W)**2)
                
                # 應用邊界條件修正
                f_ij *= self.params.boundary_factor
                frequencies.append(f_ij)
        
        self.modal_frequencies = sorted(frequencies)
        return self.modal_frequencies
    
    def calculate_modal_shapes(self) -> List[callable]:
        L = self.box_dimensions['length']
        W = self.box_dimensions['width']
        shapes = []
        
        # 改進的模態形狀函數，考慮二維振動
        for i in range(1, 4):
            for j in range(1, 4):
                def shape_func(x, y, i=i, j=j, L=L, W=W):
                    # 應用改進的邊界條件
                    x_factor = np.sin(i*np.pi*x/L) if x <= L else 0
                    y_factor = np.sin(j*np.pi*y/W) if y <= W else 0
                    return x_factor * y_factor
                shapes.append(shape_func)
        
        self.modal_shapes = shapes
        return shapes
        
    def calculate_modal_response(self, x: float, y: float, t: float) -> float:
        modal_response = 0
        for freq, shape_func in zip(self.modal_frequencies, self.modal_shapes):
            omega_modal = 2 * np.pi * freq
            
            # 計算模態參與因子（改進的計算）
            participation_factor = self.params.Q_factor/(1 + 
                                                    abs(freq - self.params.f_acoustic))
            
            # 計算模態響應（考慮阻尼和相位）
            zeta = self.params.material.damping_ratio
            omega = 2 * np.pi * self.params.f_acoustic
            modal_phase = np.arctan2(2*zeta*omega*omega_modal, 
                                   omega_modal**2 - omega**2)
            
            modal_response += (participation_factor * shape_func(x, y) * 
                             np.exp(-zeta * omega_modal * t) * 
                             np.sin(omega_modal * t + modal_phase))
        
        return modal_response

class BesselModalAnalysis(ModalAnalysisBase):
    """Bessel模態分析實現"""
    def __init__(self, params: SystemParameters, box_dimensions: Dict):
        super().__init__(params)
        self.box_dimensions = box_dimensions
        self.max_modes = (3, 3)
        self._setup_bessel_parameters()
        
    def _setup_bessel_parameters(self):
        self.modal_frequencies = []
        self.bessel_zeros = []
        self.radius = min(self.box_dimensions['length'], 
                         self.box_dimensions['width'])/2
        
        for m in range(self.max_modes[0]):
            zeros = []
            for n in range(1, self.max_modes[1] + 1):
                x = n * np.pi
                while abs(jv(m, x)) > 1e-10:
                    x = x - jv(m, x)/jv(m-1, x)
                zeros.append(x)
            self.bessel_zeros.append(zeros)

    def calculate_modal_frequencies(self) -> List[float]:
        frequencies = []
        h = self.box_dimensions['thickness']
        rho = self.params.material.density
        E = self.params.material.youngs_modulus
        nu = self.params.material.poisson_ratio
        D = (E * h**3)/(12 * (1 - nu**2))
        
        for m, zeros in enumerate(self.bessel_zeros):
            for alpha in zeros:
                omega = (alpha/self.radius)**2 * np.sqrt(D/(rho * h))
                freq = omega/(2*np.pi)
                frequencies.append(freq)
        
        self.modal_frequencies = sorted(frequencies)
        return self.modal_frequencies

    def calculate_modal_shapes(self) -> List[callable]:
        shapes = []
        for m, zeros in enumerate(self.bessel_zeros):
            for alpha in zeros:
                def shape_func(x, y, m=m, alpha=alpha):
                    r = np.sqrt(x**2 + y**2)
                    theta = np.arctan2(y, x)
                    if r <= self.radius:
                        return jv(m, alpha*r/self.radius) * np.cos(m*theta)
                    return 0
                shapes.append(shape_func)
        
        self.modal_shapes = shapes
        return shapes

    def calculate_modal_response(self, x: float, y: float, t: float) -> float:
        modal_response = 0
        for freq, shape_func in zip(self.modal_frequencies, self.modal_shapes):
            modal_response += shape_func(x, y) * np.sin(2 * np.pi * freq * t)
        return modal_response