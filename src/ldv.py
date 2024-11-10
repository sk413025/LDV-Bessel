from typing import Dict, Tuple, Optional, List, Any
import numpy as np
import matplotlib.pyplot as plt
from .models.system import SystemParameters
from .models.optical import OpticalSystem
from .models.material import MaterialProperties
from .analysis.vibration import SurfaceVibrationModel
from .utils.logger import logger

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
            beam_diameter=self.system_params.beam_diameter,
            focal_length=self.system_params.focal_length,
            working_distance=self.system_params.working_distance,
            beam_divergence=self.system_params.scan_angle,
            optical_power=self.system_params.optical_power
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
        duration = self.system_params.measurement_time  # 測量時間 [s]
        sampling_rate = self.system_params.sampling_rate  # 採樣率 [Hz]
        time_step = 1 / sampling_rate  # 時間步長 [s]
        logger.info(f"採樣率（sampling rate）: {sampling_rate} Hz")
        logger.info(f"時間步長（time step）: {time_step:.2e} s")
        time_points = np.linspace(0, duration, int(sampling_rate * duration))
        
        # 計算振動響應
        displacement, velocity = self.vibration_model.calculate_surface_response(x, y, time_points)
        
        # 計算干涉信號
        interference_intensity = []
        for t_idx, t in enumerate(time_points):
            # 計算參考光和測量光
            E_ref = self.optical_system.calculate_reference_beam(t)
            E_meas = self.optical_system.calculate_measurement_beam(
                x, y, displacement[t_idx], t, self.material)
            
            # 計算干涉強度
            intensity = self.optical_system.calculate_interference_intensity(E_ref, E_meas)
            interference_intensity.append(intensity)
        
        interference_intensity = np.array(interference_intensity)
        
        # 計算頻譜
        frequencies, amplitudes = self.vibration_model.get_frequency_response(
            self.system_params.sampling_rate,
            duration
        )
        
        # 干涉信號的頻譜
        int_frequencies = np.fft.fftfreq(len(time_points), 1/self.system_params.sampling_rate)
        int_fft = np.fft.fft(interference_intensity)
        int_amplitudes = np.abs(int_fft)
        
        # 計算統計量
        diagnostics = {
            'displacement_stats': {
                'max': np.max(np.abs(displacement)),
                'min': np.min(displacement),
                'mean': np.mean(displacement),
                'rms': np.sqrt(np.mean(displacement**2))
            },
            'velocity_stats': {
                'max': np.max(np.abs(velocity)),
                'min': np.min(velocity),
                'mean': np.mean(velocity),
                'rms': np.sqrt(np.mean(velocity**2))
            },
            'interference_stats': {
                'max': np.max(interference_intensity),
                'min': np.min(interference_intensity),
                'mean': np.mean(interference_intensity),
                'modulation_depth': (np.max(interference_intensity) - np.min(interference_intensity)) / 
                                  (np.max(interference_intensity) + np.min(interference_intensity))
            }
        }
        
        # 記錄詳細統計信息
        logger.debug("\n振動分析詳細統計:")
        logger.debug("位移統計:")
        for key, value in diagnostics['displacement_stats'].items():
            logger.debug(f"  {key}: {value*1e9:.2f} nm")
            
        logger.debug("\n速度統計:")
        for key, value in diagnostics['velocity_stats'].items():
            logger.debug(f"  {key}: {value*1e3:.2f} mm/s")
            
        logger.debug("\n干涉信號統計:")
        for key, value in diagnostics['interference_stats'].items():
            logger.debug(f"  {key}: {value:.3e}")
            
        # 記錄頻率分析結果
        peak_freqs = frequencies[np.argsort(amplitudes)[-3:]]
        peak_amps = sorted(amplitudes)[-3:]
        logger.debug("\n主要振動頻率成分:")
        for freq, amp in zip(peak_freqs, peak_amps):
            logger.debug(f"  頻率: {freq:.1f} Hz, 振幅: {amp:.2e}")
            
        return {
            'time': time_points,
            'displacement': displacement,
            'velocity': velocity,
            'interference': interference_intensity,
            'frequencies': frequencies,
            'amplitudes': amplitudes,
            'interference_frequencies': int_frequencies[int_frequencies > 0],
            'interference_amplitudes': int_amplitudes[int_frequencies > 0],
            'diagnostics': diagnostics
        }
        
    def plot_comprehensive_analysis(self) -> None:
        """繪製綜合分析圖"""
        plt.rcParams['font.family'] = ['Microsoft JhengHei']
        
        # 測量中心點
        x, y = 0, 0
        results = self.analyze_vibration(x, y)
        
        # 創建 2x3 的子圖布局
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 3, figure=fig)
        
        # 1. 表面振動分析 (左上)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(results['time'], results['displacement']*1e9, 'b-', label='位移')
        ax1.set_xlabel('時間 (s)')
        ax1.set_ylabel('位移 (nm)')
        ax1.set_title('表面振動位移')
        ax1.grid(True)
        ax1.legend()
        
        # 2. 速度分析 (中上)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(results['time'], results['velocity']*1e3, 'g-', label='速度')
        ax2.set_xlabel('時間 (s)')
        ax2.set_ylabel('速�� (mm/s)')
        ax2.set_title('表面振動速度')
        ax2.grid(True)
        ax2.legend()
        
        # 3. 振動頻譜分析 (右上)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(results['frequencies'], results['amplitudes'], 'r-', label='頻譜')
        ax3.set_xlabel('頻率 (Hz)')
        ax3.set_ylabel('振幅')
        ax3.set_title('振動頻率響應')
        ax3.grid(True)
        ax3.legend()
        
        # 4. LDV 干涉信號 (左下)
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(results['time'], results['interference'], 'k-', label='干涉強度')
        ax4.set_xlabel('時間 (s)')
        ax4.set_ylabel('光強度 (a.u.)')
        ax4.set_title('LDV 干涉信號')
        ax4.grid(True)
        ax4.legend()
        
        # 5. 干涉信號頻譜 (中下)
        ax5 = fig.add_subplot(gs[1, 1])
        valid_idx = results['interference_frequencies'] > 0
        ax5.plot(results['interference_frequencies'][valid_idx],
                results['interference_amplitudes'][valid_idx],
                'b-', label='干涉頻譜')
        ax5.set_xlabel('頻率 (Hz)')
        ax5.set_ylabel('振幅')
        ax5.set_title('干涉信號頻譜')
        ax5.grid(True)
        ax5.legend()
        
        # 6. 測量品質指標 (右下)
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        quality = self.get_measurement_quality(self.system_params.working_distance)
        stats = results['diagnostics']
        
        # 組合顯示文本
        quality_text = '\n'.join([
            "測量品質指標:",
            f"SNR: {quality['snr']:.1f} dB",
            f"空間解析度: {quality['spatial_resolution']*1e6:.1f} μm",
            f"最大可測速度: {quality['max_velocity']*1e3:.1f} mm/s",
            "\n振動統計:",
            f"RMS位移: {stats['displacement_stats']['rms']*1e9:.2f} nm",
            f"RMS速度: {stats['velocity_stats']['rms']*1e3:.2f} mm/s",
            "\n干涉信號統計:",
            f"調制深度: {stats['interference_stats']['modulation_depth']:.2f}",
            f"平均強度: {stats['interference_stats']['mean']:.2e}"
        ])
        ax6.text(0.05, 0.95, quality_text, 
                transform=ax6.transAxes,
                verticalalignment='top',
                fontsize=9)
        ax6.set_title('測量參數統計')
        
        plt.tight_layout()
        plt.show()