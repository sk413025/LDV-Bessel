from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import numpy as np
from scipy.special import jv
from .system import SystemParameters
from .material import MaterialProperties
from ..utils.logger import logger

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
        
        logger.debug("\n頻率計算參數驗證：")
        logger.debug(f"彎曲剛度: {self.bending_stiffness:.2e} N⋅m²")
        logger.debug(f"質量項: {self.params.material.density:.2e} kg/m³")
        logger.debug(f"尺寸: L={L:.3f}m, W={W:.3f}m")
        
        # 計算前9個模態頻率
        for i in range(1, 4):
            for j in range(1, 4):
                # 改進的頻率方程，考慮二維振動
                f_ij = (np.pi/2) * np.sqrt(self.bending_stiffness/
                                         (self.params.material.density * self.box_dimensions['thickness'])) * \
                      ((i/L)**2 + (j/W)**2)
                
                # 應用邊界條件修正
                f_ij *= self.params.boundary_factor
                frequencies.append(f_ij)
        
        return sorted(frequencies)
    
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
        """計算模態響應
        參考方程: w(x,y,t) = Σ Aᵢⱼφᵢⱼ(x,y)sin(ωᵢⱼt + θᵢⱼ)e⁻ᶻᵉᵗᵃ*ωᵢⱼ*ᵗ
        """
        if not self.modal_frequencies or not self.modal_shapes:
            self.modal_frequencies = self.calculate_modal_frequencies()
            self.modal_shapes = self.calculate_modal_shapes()
            
        total_response = 0
        zeta = self.params.material.damping_ratio
        omega_drive = 2 * np.pi * self.params.f_acoustic
        
        logger.debug(f"\n響應計算驗證:")
        logger.debug(f"位置: x={x:.3f}m, y={y:.3f}m")
        logger.debug(f"時間: t={t:.3f}s")
        logger.debug(f"阻尼比: {zeta:.3f}")
        logger.debug(f"驅動頻率: {self.params.f_acoustic} Hz")
        
        for idx, (freq, shape_func) in enumerate(zip(self.modal_frequencies, self.modal_shapes)):
            omega = 2 * np.pi * freq
            
            # 改進阻尼計算
            damping = np.exp(-zeta * omega * (t % (1/freq)))  # 使用模態週期
            
            # 計算各項因子
            shape_value = shape_func(x, y)
            participation = self.params.Q_factor / (1 + abs(freq - self.params.f_acoustic))
            phase = np.arctan2(2*zeta*omega*omega_drive, omega**2 - omega_drive**2)
            
            # 計算單個模態響應
            mode_response = (participation * shape_value * damping * 
                            np.sin(omega * t + phase))
            
            # 添加詳細的模態響應驗證
            logger.debug(f"\n模態({idx+1})響應計算:")
            logger.debug(f"  頻率: {freq:.1f} Hz")
            logger.debug(f"  形狀值: {shape_value:.2e}")
            logger.debug(f"  參與因子: {participation:.2e}")
            logger.debug(f"  阻尼項: {damping:.2e}")
            logger.debug(f"  相位: {phase:.2e} rad")
            logger.debug(f"  響應: {mode_response:.2e} m")
            
            total_response += mode_response
        
        # 限制最大響應
        max_amp = self.box_dimensions['thickness'] * 0.1
        total_response = np.clip(total_response, -max_amp, max_amp)
        
        logger.debug(f"\n總響應: {total_response:.2e} m")
        return total_response

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
        if not self.modal_frequencies or not self.modal_shapes:
            self.modal_frequencies = self.calculate_modal_frequencies()
            self.modal_shapes = self.calculate_modal_shapes()
        
        print("\n模態響應計算驗證：")
        print(f"計算點: x={x:.3f}m, y={y:.3f}m, t={t:.3f}s")
        
        modal_response = 0
        for idx, (freq, shape_func) in enumerate(zip(self.modal_frequencies, self.modal_shapes)):
            omega_modal = 2 * np.pi * freq
            
            # 計算並驗證各項因子
            shape_value = shape_func(x, y)
            participation_factor = self.params.Q_factor/(1 + abs(freq - self.params.f_acoustic))
            
            zeta = self.params.material.damping_ratio
            omega = 2 * np.pi * self.params.f_acoustic
            modal_phase = np.arctan2(2*zeta*omega*omega_modal, 
                                   omega_modal**2 - omega**2)
            
            # 計算時域響應項
            damping_term = np.exp(-zeta * omega_modal * t)
            oscillation_term = np.sin(omega_modal * t + modal_phase)
            
        # 單個模態的貢獻
        max_amplitude = 1e-6  # 1 μm
        mode_contribution = (participation_factor * shape_value * 
                           damping_term * oscillation_term)
        
        # 打印詳細資訊
        print(f"\n模態 {idx+1}:")
        print(f"  頻率: {freq:.2f} Hz")
        print(f"  模態形狀值: {shape_value:.2e}")
        print(f"  參與因子: {participation_factor:.2e}")
        print(f"  衰減項: {damping_term:.2e}")
        print(f"  振盪項: {oscillation_term:.2e}")
        print(f"  模態貢獻: {mode_contribution:.2e}")
        
        modal_response += mode_contribution
        
        # 添加位移限制
        max_displacement = self.box_dimensions['thickness'] * 0.1
        modal_response = np.clip(modal_response, -max_displacement, max_displacement)
        
        print(f"\n總響應: {modal_response:.2e} m")
        return modal_response
        
        for i, (freq, shape) in enumerate(zip(self.modal_frequencies, self.modal_shapes)):
            # 考慮阻尼效應
            omega = 2 * np.pi * freq
            zeta = self.params.material.damping_ratio
            damping = np.exp(-zeta * omega * t)
            
            # 改進的參與因子計算
            participation = 1.0/((i + 1) * np.sqrt(1 + (freq/self.params.f_acoustic - 1)**2))
            participation = min(participation, 1.0)
            
            # 計算響應
            response = participation * shape(x, y) * damping * np.sin(omega * t)
            response = np.clip(response, -max_amplitude, max_amplitude)
            
            modal_response += response
            
        return modal_response