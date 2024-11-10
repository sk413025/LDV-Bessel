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

class ClassicalModalAnalysis(ModalAnalysisBase):
    """傳統模態分析實現"""
    def __init__(self, params: SystemParameters, box_dimensions: Dict):
        super().__init__(params)
        self.box_dimensions = box_dimensions
        self._initialize_parameters()
        self.modal_frequencies = []
        self.modal_shapes = []
        self.log_interval = 1.0  # 每秒記錄一次
        self.last_log_time = 0.0
        
    def _initialize_parameters(self):
        h = self.box_dimensions['thickness']  # 厚度 [m]
        E = self.params.material.youngs_modulus  # 楊氏模量 [Pa]
        v = self.params.material.poisson_ratio  # 柏松比 [無單位]
        # 計算彎曲剛度 [N·m]
        self.bending_stiffness = (E * h**3) / (12 * (1 - v**2))
        
        # 計算模態質量
        area = self.box_dimensions['length'] * self.box_dimensions['width']
        self.modal_mass = self.params.material.density * area * h
        
        # 激勵力計算
        self.acoustic_pressure = self.params.acoustic_pressure if hasattr(self.params, 'acoustic_pressure') else 1.0
        self.forcing_amplitude = self.acoustic_pressure * area
        
        # 添加模態正交性相關參數
        self.mode_scaling_factors = []
        self.normalized_shapes = []
        
        # 計算板的彎曲剛度
        h = self.box_dimensions['thickness']
        E = self.params.material.youngs_modulus
        v = self.params.material.poisson_ratio
        self.bending_stiffness = (E * h**3) / (12 * (1 - v**2))
        
        # 計算單位面積質量（考慮有效質量修正） [kg/m^2]
        mass_correction = getattr(self.params, 'mass_correction', 0.3)  # 默認值0.3
        self.mass_per_area = (self.params.material.density * 
                             self.box_dimensions['thickness'] * 
                             mass_correction)
        
        # 計算聲波衰減
        self.acoustic_attenuation = getattr(self.params, 'acoustic_attenuation', 1.0)
        
        # 獲取聲壓
        self.acoustic_pressure = getattr(self.params, 'acoustic_pressure', 0.2)
        
        # 獲取邊界條件因子
        self.boundary_factor = getattr(self.params, 'boundary_factor', 0.8)
        
        # 獲取力的放大係數
        self.force_amplification = getattr(self.params, 'force_amplification', 1.5)
        
        # 使用安全的參數獲取方法
        self.theta_incidence = getattr(self.params, 'theta_incidence', np.deg2rad(30))
        self.mass_correction = getattr(self.params, 'mass_correction', 0.3)
        self.acoustic_attenuation = getattr(self.params, 'acoustic_attenuation', 1.0)
        self.acoustic_pressure = getattr(self.params, 'acoustic_pressure', 0.2)
        self.boundary_factor = getattr(self.params, 'boundary_factor', 0.8)
        self.force_amplification = getattr(self.params, 'force_amplification', 1.5)
        
        # 計算修正係數
        self.cos_theta = np.cos(self.theta_incidence)
        
        # 計算特徵頻率（考慮邊界條件）
        L = self.box_dimensions['length']
        self.characteristic_freq = ((1/(2*np.pi)) * 
                                  np.sqrt(self.bending_stiffness/
                                        (self.mass_per_area * L**4)) * 
                                  self.params.boundary_factor)
        
        # 計算聲波入射角的影響因子
        self.cos_theta = np.cos(self.params.theta_incidence)
        
        # 計算Q因子
        self.Q_factor = 1/(2 * self.params.material.damping_ratio)
        
        # 記錄阻尼比和聲壓
        logger.info(f"阻尼比（damping ratio）: {self.params.material.damping_ratio}")
        logger.info(f"聲壓（acoustic pressure）: {self.acoustic_pressure} Pa")

    def calculate_modal_mass(self, shape_func) -> float:
        """計算歸一化模態質量"""
        L = self.box_dimensions['length']
        W = self.box_dimensions['width']
        rho = self.params.material.density
        h = self.box_dimensions['thickness']
        
        # 使用數值積分計算模態質量
        dx = L/20
        dy = W/20
        mass = 0
        for x in np.arange(0, L, dx):
            for y in np.arange(0, W, dy):
                mass += shape_func(x, y)**2 * rho * h * dx * dy
        return mass

    def calculate_modal_frequencies(self) -> List[float]:
        frequencies = []
        L = self.box_dimensions['length']
        W = self.box_dimensions['width']
        h = self.box_dimensions['thickness']
        rho = self.params.material.density
        E = self.params.material.youngs_modulus
        nu = self.params.material.poisson_ratio

        # 計算板的彎曲剛度
        D = (E * h**3) / (12 * (1 - nu**2))

        # 修正的頻率計算公式
        for m in range(1, 4):
            for n in range(1, 4):
                f_mn = (1 / (2 * np.pi)) * ((np.pi**2) * np.sqrt(D / (rho * h)) *
                        ((m / L)**2 + (n / W)**2))
                frequencies.append(f_mn)

        return sorted(frequencies)

    def calculate_boundary_condition_factor(self, i: int, j: int) -> float:
        """計算邊界條件影響因子"""
        # 簡單支撐邊界條件
        return 1.0
        
    def calculate_generalized_force(self, shape_func) -> float:
        """計算廣義力"""
        L = self.box_dimensions['length']
        W = self.box_dimensions['width']
        
        # 計算激勵力的空間分布
        dx = L/20
        dy = W/20
        force = 0
        for x in np.arange(0, L, dx):
            for y in np.arange(0, W, dy):
                force += self.acoustic_pressure * shape_func(x, y) * dx * dy
        return force

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
        
        # 只在特定時間間隔記錄
        should_log = (t - self.last_log_time) >= self.log_interval
        
        if should_log:
            logger.info("=== 時間 %.2f 秒的響應計算 ===", t)
            logger.info("位置: (%.3f, %.3f) m", x, y)
            self.last_log_time = t
        
        # 記錄頻譜分析用的數據
        self.response_spectrum = []
        
        for idx, (freq, shape_func) in enumerate(zip(self.modal_frequencies, self.modal_shapes)):
            omega = 2 * np.pi * freq
            omega_drive = 2 * np.pi * self.params.f_acoustic
            
            # 計算歸一化模態質量
            modal_mass = self.calculate_modal_mass(shape_func)
            
            # 計算廣義力
            generalized_force = self.calculate_generalized_force(shape_func)
            
            # 改進的阻尼計算
            modal_damping = zeta * (1 + 0.1 * (freq/self.params.f_acoustic))
            damping = np.exp(-modal_damping * omega * (t % (1/freq)))
            
            # 修正的FRF計算，加入剛度項
            stiffness = modal_mass * omega**2
            H_omega = 1.0 / np.sqrt((stiffness - modal_mass*omega_drive**2)**2 + 
                                   (modal_damping*omega_drive/omega)**2)
            
            # 計算相位
            phase = np.arctan2(modal_damping*omega_drive/omega, 
                             stiffness - modal_mass*omega_drive**2)
            
            # 分離計算自然振動和強制振動
            shape_value = shape_func(x, y)
            # 輸出模態形狀函數值
            logger.debug(f"模態 {idx+1} 的形狀函數值: {shape_value}")
            natural_response = shape_value * damping * np.sin(omega * t)
            forced_response = shape_value * generalized_force * H_omega * \
                            np.sin(omega_drive * t - phase)
            
            # 模態響應，不重複使用Q因子
            mode_response = (natural_response + forced_response) / modal_mass
            
            # 驗證能量守恆
            if self.verify_energy_conservation(mode_response, omega, modal_mass):
                total_response += mode_response
            
            if should_log and idx < 3:
                logger.info(f"模態 {idx+1}:")
                logger.info(f"  頻率: {freq:.1f} Hz")
                logger.info(f"  FRF: {H_omega:.2e}")
                logger.info(f"  自然振動: {natural_response:.2e}")
                logger.info(f"  強制振動: {forced_response:.2e}")
                logger.info(f"  總響應: {mode_response:.2e} m")
            
            total_response += mode_response
        
        # 限制最大響應
        max_amp = self.box_dimensions['thickness'] * 0.1
        total_response = np.clip(total_response, -max_amp, max_amp)
        
        if should_log:
            # 輸出頻譜分析結果
            logger.info("\n頻譜分析:")
            sorted_modes = sorted(self.response_spectrum, 
                                key=lambda x: x['amplitude'], 
                                reverse=True)
            for mode in sorted_modes[:3]:
                logger.info(f"頻率: {mode['frequency']:.1f} Hz, "
                          f"振幅: {mode['amplitude']:.2e} m")
            logger.info(f"總響應: {total_response:.2e} m\n")
            
        return total_response

    def calculate_modal_response(self, x: float, y: float, t: float) -> float:
        if not self.modal_frequencies or not self.modal_shapes:
            self.modal_frequencies = self.calculate_modal_frequencies()
            self.modal_shapes = self.calculate_modal_shapes()

        # 1. 計算有效聲壓
        distance = np.sqrt(x**2 + y**2)
        p_eff = (self.params.sound_pressure * 
                self.cos_theta * 
                np.exp(-self.params.acoustic_attenuation * distance) *
                self.params.force_amplification)
        
        # 2. 計算有效作用力
        effective_area = self.box_dimensions['length'] * self.box_dimensions['width']
        force = p_eff * effective_area
        
        # 3. 計算強制振動參數
        omega = 2 * np.pi * self.params.f_acoustic
        omega_n = 2 * np.pi * self.characteristic_freq
        zeta = self.params.material.damping_ratio
        
        # 4. 計算複數傳遞函數
        s = 1j * omega
        denominator = omega_n**2 + 2*zeta*omega_n*s + s**2
        H = 1/denominator
        
        # 5. 計算強制振動響應
        force_amplitude = force/self.mass_per_area
        forced_response = (force_amplitude * np.abs(H) * 
                         np.sin(omega*t + np.angle(H)))

        # 6. 計算模態響應（考慮阻尼和相位）
        modal_response = 0
        for freq, shape_func in zip(self.modal_frequencies, self.modal_shapes):
            omega_modal = 2 * np.pi * freq
            
            # 計算模態參與因子（改進的計算）
            participation_factor = self.Q_factor/(1 + 
                                               abs(freq - self.params.f_acoustic))
            
            # 計算模態響應
            modal_phase = np.arctan2(2*zeta*omega*omega_modal, 
                                   omega_modal**2 - omega**2)
            
            modal_response += (participation_factor * shape_func(x, y) * 
                             np.exp(-zeta * omega_modal * t) * 
                             np.sin(omega_modal * t + modal_phase))
        
        # 7. 組合總位移（調整模態影響）
        total_response = forced_response + 0.05 * modal_response  # 降低模態影響
        
        # 8. 位移限制
        max_amp = self.box_dimensions['thickness'] * 0.1
        total_response = np.clip(total_response, -max_amp, max_amp)
        
        # 記錄日誌（如果需要）
        if (t - self.last_log_time) >= self.log_interval:
            logger.info("=== 時間 %.2f 秒的響應計算 ===", t)
            logger.info("位置: (%.3f, %.3f) m", x, y)
            logger.info("有效聲壓: %.2e Pa", p_eff)
            logger.info("強制響應: %.2e m", forced_response)
            logger.info("模態響應: %.2e m", modal_response)
            logger.info("總響應: %.2e m", total_response)
            self.last_log_time = t
            
        return total_response

    def calculate_modal_response(self, x: float, y: float, t: float) -> float:
        if not self.modal_frequencies or not self.modal_shapes:
            self.modal_frequencies = self.calculate_modal_frequencies()
            self.modal_shapes = self.calculate_modal_shapes()

        total_response = 0
        for freq, shape_func in zip(self.modal_frequencies, self.modal_shapes):
            omega = 2 * np.pi * freq
            shape_value = shape_func(x, y)
            # 簡化響應計算，移除多餘的阻尼和相位計算
            mode_response = shape_value * np.sin(omega * t)
            total_response += mode_response
        # 移除頻繁的日誌紀錄，只保留必要資訊
        if (t - self.last_log_time) >= self.log_interval:
            logger.info("時間 %.2f 秒的總響應: %.2e m", t, total_response)
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

    def calculate_modal_response(self, x: float, y: float, t: float) -> float:
        if not self.modal_frequencies or not self.modal_shapes:
            self.modal_frequencies = self.calculate_modal_frequencies()
            self.modal_shapes = self.calculate_modal_shapes()
        
        print("\n模態響應計算驗證：")
        print(f"計算點: x={x:.3f}m, y={y:.3f}m, t={t:.3f}s")
        
        modal_response = 0
        
        # 只在特定時間間隔記錄
        if (t - self.last_log_time) >= self.log_interval:
            logger.info("Bessel響應計算 t=%.2f: (%.3f, %.3f) m", t, x, y)
            self.last_log_time = t
            log_details = True
        else:
            log_details = False
        
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
        
        if log_details:
            logger.info("Bessel最終響應: %.2e m\n", modal_response)
        
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