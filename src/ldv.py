from typing import Dict, Tuple, Optional, List, Any
import numpy as np
import matplotlib.pyplot as plt
from .models.system import SystemParameters
from .models.optical import OpticalSystem
from .models.material import MaterialProperties
from .analysis.vibration import SurfaceVibrationModel

class LaserDopplerVibrometer:
    """雷射都卜勒振動儀主類"""

    def __init__(self,
                 material: MaterialProperties,
                 modal_analyzer: type = None,
                 analysis_type: str = "classical"):
        """
        初始化LDV系統
        
        Args:
            material: 待測物材料特性
            modal_analyzer: 模態分析器類型 (ClassicalModalAnalysis 或 BesselModalAnalysis)
            analysis_type: 分析類型 ("classical" 或 "bessel")
        """
        self.material = material
        self.system_params = SystemParameters(material)
        self.optical_system = OpticalSystem(
            wavelength=self.system_params.wavelength,
            beam_diameter=2*self.system_params.beam_radius,
            focal_length=self.system_params.focal_length,
            working_distance=self.system_params.working_distance,
            beam_divergence=self.system_params.scan_angle,
            optical_power=self.system_params.power
        )
        self.vibration_model = SurfaceVibrationModel(
            system_params=self.system_params,
            analysis_type=analysis_type,
            modal_analyzer=modal_analyzer
        )
        
    def setup_measurement(self, box_dimensions: Dict[str, float]) -> None:
        """
        設置測量參數
        
        Args:
            box_dimensions: 結構尺寸參數 {"length": float, "width": float, "thickness": float}
        """
        if not isinstance(box_dimensions, dict):
            raise TypeError("box_dimensions must be a dictionary")
            
        # 驗證必要的尺寸參數
        required_dims = ['length', 'width', 'thickness']
        for dim in required_dims:
            if dim not in box_dimensions:
                raise ValueError(f"Missing required dimension: {dim}")
                
        self.box_dimensions = box_dimensions
        self.vibration_model.setup_modal_analysis(box_dimensions)
        
    def measure_point(self, 
                     x: float, 
                     y: float, 
                     duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        測量單點振動
        
        Args:
            x: x座標位置
            y: y座標位置
            duration: 測量時間
            
        Returns:
            time_points: 時間陣列
            velocities: 速度陣列
        """
        time_points = np.linspace(0, duration, int(self.system_params.sampling_rate * duration))
        _, velocities = self.vibration_model.calculate_surface_response(x, y, time_points)
        
        return time_points, velocities
    
    def get_frequency_spectrum(self, 
                             velocities: np.ndarray, 
                             duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        計算頻譜
        
        Args:
            velocities: 速度時間序列
            duration: 測量時間
            
        Returns:
            frequencies: 頻率陣列
            amplitudes: 振幅陣列
        """
        return self.vibration_model.get_frequency_response(
            self.system_params.sampling_rate,
            duration
        )
    
    def get_measurement_quality(self, distance: float) -> Dict[str, float]:
        """
        評估測量品質
        
        Args:
            distance: 實際測量距離
            
        Returns:
            Dict: 包含SNR等測量品質指標
        """
        snr = self.optical_system.estimate_snr(self.material, distance)
        spot_size = self.optical_system.calculate_spot_size()
        
        return {
            "snr": snr,
            "spot_size": spot_size,
            "max_velocity": self.system_params.wavelength * self.system_params.sampling_rate / 4,
            "spatial_resolution": spot_size
        }
    
    def get_system_status(self) -> Dict[str, Dict]:
        """
        獲取系統狀態摘要
        
        Returns:
            Dict: 系統狀態資訊
        """
        return {
            "optical_parameters": self.optical_system.get_system_parameters(),
            "system_parameters": self.system_params.get_parameters_dict(),
            "material_properties": self.material.__dict__
        }
    
    def analyze_vibration(self, x: float, y: float) -> Dict[str, Any]:
        """分析指定點的振動"""
        duration = self.system_params.measurement_time
        time_points = np.linspace(0, duration, int(self.system_params.sampling_rate * duration))
        
        # 計算振動響應
        displacement, velocity = self.vibration_model.calculate_surface_response(x, y, time_points)
        
        # 計算頻譜
        frequencies, amplitudes = self.vibration_model.get_frequency_response(
            self.system_params.sampling_rate,
            duration
        )
        
        # 計算統計量
        diagnostics = {
            'displacement_stats': {
                'max': np.max(np.abs(displacement)),
                'rms': np.sqrt(np.mean(displacement**2))
            },
            'velocity_stats': {
                'max': np.max(np.abs(velocity)),
                'rms': np.sqrt(np.mean(velocity**2))
            }
        }
        
        return {
            'time': time_points,
            'displacement': displacement,
            'velocity': velocity,
            'frequencies': frequencies,
            'amplitudes': amplitudes,
            'diagnostics': diagnostics
        }
    
    def plot_comprehensive_analysis(self) -> None:
        """繪製綜合分析圖"""
        plt.rcParams['font.family'] = ['Microsoft JhengHei']

        # 測量中心點
        x, y = 0, 0
        results = self.analyze_vibration(x, y)
        
        # 創建子圖
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # 位移時域圖
        ax1.plot(results['time'], results['displacement']*1e9)
        ax1.set_xlabel('時間 (s)')
        ax1.set_ylabel('位移 (nm)')
        ax1.set_title('位移時域響應')
        ax1.grid(True)
        
        # 速度時域圖
        ax2.plot(results['time'], results['velocity']*1e3)
        ax2.set_xlabel('時間 (s)')
        ax2.set_ylabel('速度 (mm/s)')
        ax2.set_title('速度時域響應')
        ax2.grid(True)
        
        # 頻譜圖
        ax3.plot(results['frequencies'], results['amplitudes'])
        ax3.set_xlabel('頻率 (Hz)')
        ax3.set_ylabel('振幅')
        ax3.set_title('頻率響應')
        ax3.grid(True)
        
        # 測量品質指標
        quality = self.get_measurement_quality(self.system_params.working_distance)
        ax4.axis('off')
        quality_text = '\n'.join([
            f"SNR: {quality['snr']:.1f} dB",
            f"空間解析度: {quality['spatial_resolution']*1e6:.1f} μm",
            f"最大可測速度: {quality['max_velocity']*1e3:.1f} mm/s"
        ])
        ax4.text(0.1, 0.5, quality_text, fontsize=10)
        ax4.set_title('測量品質指標')
        
        plt.tight_layout()
        plt.show()