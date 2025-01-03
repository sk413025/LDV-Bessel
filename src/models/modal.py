from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import numpy as np
from scipy.special import jv
from .system import SystemParameters
from .material import MaterialProperties
from ..utils.logger import logger  # 改為兩層相對導入

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

    @abstractmethod
    def calculate_single_mode_response(self, x: float, y: float, t: float, mode_idx: int) -> float:
        """計算單一模態響應"""
        pass

class ClassicalModalAnalysis(ModalAnalysisBase):
    def __init__(self, params: SystemParameters, box_dimensions: Dict):
        super().__init__(params)
        self.box_dimensions = box_dimensions
        self._initialize_parameters()
        self.log_interval = 0.1
        self.last_log_time = 0.0

    def _initialize_parameters(self):
        """初始化關鍵物理參數"""
        # 材料參數
        h = self.box_dimensions['thickness']  # [m]
        E = self.params.material.youngs_modulus  # [Pa]
        rho = self.params.material.density  # [kg/m³]
        
        # 計算彎曲剛度 D = Eh³/[12(1-ν²)] [N·m]
        self.bending_stiffness = (E * h**3) / (12 * (1 - self.params.material.poisson_ratio**2))
        
        # 計算單位面積質量 [kg/m²]
        self.mass_per_area = rho * h
        
        # 聲波參數
        self.acoustic_pressure = getattr(self.params, 'acoustic_pressure', 0.2)  # [Pa]
        self.theta_incidence = getattr(self.params, 'theta_incidence', np.deg2rad(30))  # [rad]
        
        # 記錄關鍵參數
        logger.info(f"彎曲剛度: {self.bending_stiffness:.2e} N·m")
        logger.info(f"單位��積質量: {self.mass_per_area:.2e} kg/m²")

    def calculate_modal_frequencies(self) -> List[float]:
        """計算與1000Hz附近的模態頻率"""
        frequencies = []
        L = self.box_dimensions['length']
        W = self.box_dimensions['width']
        
        # 只計算1000Hz附近的頻率（±250Hz）
        target_freq = self.params.f_acoustic  # 1000Hz
        freq_range = 250  # Hz
        
        for m in range(1, 4):
            for n in range(1, 4):
                # 計算頻率 [Hz]
                f_mn = (1 / (2 * np.pi)) * ((np.pi**2) * np.sqrt(
                    self.bending_stiffness / (self.mass_per_area)) *
                    ((m / L)**2 + (n / W)**2))
                
                if abs(f_mn - target_freq) <= freq_range:
                    frequencies.append(f_mn)
        
        return sorted(frequencies)

    def calculate_modal_shapes(self) -> List[callable]:
        """計算簡化的模態形狀函數"""
        L = self.box_dimensions['length']
        W = self.box_dimensions['width']
        shapes = []
        
        for i in range(1, 4):
            for j in range(1, 4):
                def shape_func(x, y, i=i, j=j, L=L, W=W):
                    # 簡化的正弦模態形狀
                    return np.sin(i*np.pi*x/L) * np.sin(j*np.pi*y/W)
                shapes.append(shape_func)
        
        return shapes

    def calculate_single_mode_response(self, x: float, y: float, t: float, mode_idx: int) -> float:
        """計算單一模態響應"""
        if not hasattr(self, 'modal_frequencies'):
            self.modal_frequencies = self.calculate_modal_frequencies()
            self.modal_shapes = self.calculate_modal_shapes()
            
        if mode_idx >= len(self.modal_frequencies):
            return 0.0
            
        freq = self.modal_frequencies[mode_idx]
        shape_func = self.modal_shapes[mode_idx]
        
        # 計算有效聲壓 [Pa]
        p_eff = self.acoustic_pressure * np.cos(self.theta_incidence)
        
        # 激勵頻率 [rad/s]
        omega = 2 * np.pi * self.params.f_acoustic
        
        # 計算響應
        zeta = self.params.material.damping_ratio
        shape_value = shape_func(x, y)
        freq_ratio = freq / self.params.f_acoustic
        
        # 動態放大因子
        beta = 1 / ((1 - freq_ratio**2)**2 + (2*zeta*freq_ratio)**2)**0.5
        
        # 相位角 [rad]
        phase = np.arctan2(2*zeta*freq_ratio, 1 - freq_ratio**2)
        
        # 計算響應 [m]
        mode_response = (p_eff * shape_value * beta / 
                        (self.mass_per_area * omega**2) * 
                        np.sin(omega * t - phase))
        
        return mode_response

    def calculate_modal_response(self, x: float, y: float, t: float) -> float:
        """計算總模態響應"""
        if not hasattr(self, 'modal_frequencies'):
            self.modal_frequencies = self.calculate_modal_frequencies()
            self.modal_shapes = self.calculate_modal_shapes()

        total_response = 0.0
        for idx in range(len(self.modal_frequencies)):
            if abs(self.modal_frequencies[idx] - self.params.f_acoustic) < 200:  # [Hz]
                total_response += self.calculate_single_mode_response(x, y, t, idx)
        
        # 記錄響應
        if (t - self.last_log_time) >= self.log_interval:
            logger.info(f"時間 {t:.2f}s 的響應: {total_response*1e6:.2f} μm")
            self.last_log_time = t

        return total_response

    def verify_energy_conservation(self, mode_response, omega, modal_mass):
        """驗證能量守恆"""
        kinetic = 0.5 * modal_mass * (omega * mode_response)**2
        potential = 0.5 * modal_mass * omega**2 * mode_response**2
        return abs(kinetic - potential) < 1e-6 * max(kinetic, potential)

class BesselModalAnalysis(ModalAnalysisBase):
    """Bessel模態分析實現"""
    def __init__(self, params: SystemParameters, box_dimensions: Dict):
        super().__init__(params)
        self.box_dimensions = box_dimensions
        self.max_modes = (3, 3)
        self._setup_bessel_parameters()
        self.log_interval = 1.0
        self.last_log_time = 0.0
        
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

    def calculate_single_mode_response(self, x: float, y: float, t: float, mode_idx: int) -> float:
        """計算單一模態響應"""
        if not self.modal_frequencies or not self.modal_shapes:
            self.modal_frequencies = self.calculate_modal_frequencies()
            self.modal_shapes = self.calculate_modal_shapes()
            
        if mode_idx >= len(self.modal_frequencies):
            return 0.0
            
        freq = self.modal_frequencies[mode_idx]
        shape_func = self.modal_shapes[mode_idx]
        
        omega_modal = 2 * np.pi * freq
        shape_value = shape_func(x, y)
        participation_factor = self.params.Q_factor/(1 + abs(freq - self.params.f_acoustic))
        
        zeta = self.params.material.damping_ratio
        omega = 2 * np.pi * self.params.f_acoustic
        modal_phase = np.arctan2(2*zeta*omega*omega_modal, 
                               omega_modal**2 - omega**2)
        
        # 計算時域響應項
        damping_term = np.exp(-zeta * omega_modal * t)
        oscillation_term = np.sin(omega_modal * t + modal_phase)
        
        # 計算響應
        mode_response = participation_factor * shape_value * damping_term * oscillation_term
        
        # 添加位移限制
        max_displacement = self.box_dimensions['thickness'] * 0.1
        mode_response = np.clip(mode_response, -max_displacement, max_displacement)
        
        return mode_response

    def calculate_modal_response(self, x: float, y: float, t: float) -> float:
        """計算總模態響應"""
        if not self.modal_frequencies or not self.modal_shapes:
            self.modal_frequencies = self.calculate_modal_frequencies()
            self.modal_shapes = self.calculate_modal_shapes()
            
        total_response = 0.0
        log_details = (t - self.last_log_time) >= self.log_interval
            
        for idx in range(len(self.modal_frequencies)):
            mode_response = self.calculate_single_mode_response(x, y, t, idx)
            total_response += mode_response
            
            if log_details:
                freq = self.modal_frequencies[idx]
                logger.debug(f"模態 {idx+1} 響應: {mode_response*1e6:.2f} μm (頻率: {freq:.1f} Hz)")
                
        if log_details:
            logger.info(f"總響應: {total_response*1e6:.2f} μm")
            self.last_log_time = t
            
        return total_response