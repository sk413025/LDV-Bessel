"""
圓形振膜尺寸效應分析程式
======================

研究背景：
--------
在麥克風設計中，振膜的尺寸對其振動特性有重要影響。本程式旨在研究不同尺寸
振膜在相同外力作用下的振動響應差異，這對於優化麥克風設計和了解尺寸縮放效應
具有重要意義。

研究目的：
--------
1. 分析振膜半徑對自然頻率的影響
2. 比較不同尺寸振膜在相同外力下的振幅響應
3. 研究振膜尺寸與其動態特性的關係

理論基礎：
--------
- 使用 Bessel 函數描述圓形振膜的模態形狀
- 考慮張力效應和模態質量的影響。注意：如果 use_total_tension=True，
  則 material['T'] 視為「總張力 (N)」，並會動態轉換成每單位長度的線張力 (N/m)；
  若 use_total_tension=False，則直接把 material['T'] 當作「線張力」(N/m)。
- 基於線性振動理論的模態疊加

主要功能：
--------
1. 計算並視覺化不同半徑振膜的振動模態
2. 動態展示強迫振動響應
3. 比較分析尺寸效應對振動特性的影響

使用方法：
--------
1. 設定振膜材料參數（張力、面密度）
2. 定義外力條件（幅值、頻率、作用位置）
3. 指定要比較的振膜半徑
4. 運行程式獲得動態視覺化結果

作者：[作者名稱]
日期：[日期]
版本：1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.special import jn, jn_zeros

def calculate_membrane_parameters(material, R, use_total_tension=False):
    """
    若 use_total_tension=True，表示 material['T'] 為「整體總張力」(單位 N)，
    並依照振膜圓周長 (2πR) 動態轉為每單位長度的線張力 (N/m)。
    若 use_total_tension=False，則 material['T'] 本身即為線張力 (N/m)，不再換算。
    """
    T_line = material['T']
    rho_s = material['rho_s']
    
    if use_total_tension:
        T_line = T_line / (2 * np.pi * R)
    
    return T_line, rho_s

def bessel_mode_shape(r_norm, theta, m, n):
    """
    計算圓形振膜的模態形狀
    
    Parameters:
    -----------
    r_norm : array-like
        歸一化半徑 (0到1)
    theta : array-like
        角度 (0到2π)
    m : int
        圓周模態數
    n : int
        徑向模態數
    """
    lambda_mn = jn_zeros(m, n)[n-1]  # Bessel函數的第n個零點
    return jn(m, lambda_mn * r_norm) * np.cos(m * theta)

def calculate_natural_frequency(m, n, R, material, use_total_tension=False):
    """計算自然頻率時，考慮是否使用總張力模式"""
    T_line, rho_s = calculate_membrane_parameters(material, R, use_total_tension)
    lambda_mn = jn_zeros(m, n)[n-1]
    return (lambda_mn / (2 * np.pi * R)) * np.sqrt(T_line / rho_s)

def forced_modal_response(r, theta, t, material, m, n, force_params, R):
    """
    計算圓形振膜的強迫振動響應
    
    Parameters:
    -----------
    r, theta : array-like
        極座標位置
    material : dict
        材料參數，包含張力T和面密度rho_s
    m, n : int
        模態數
    R : float
        振膜半徑
    """
    # 歸一化半徑
    r_norm = r / R
    
    # 計算模態形狀
    phi_mn = bessel_mode_shape(r_norm, theta, m, n)
    
    # 獲取材料參數
    T = material['T']
    modal_mass = np.pi * R**2 * material['rho_s']
    
    # 計算 Bessel 函數零點
    lambda_mn = jn_zeros(m, n)[n-1]
    
    # 考慮完整的張力效應
    tension_term = T * (lambda_mn/R)**2
    
    # 力的參數
    F0 = force_params['amplitude']
    omega = force_params['frequency'] * 2 * np.pi
    
    # 計算自然頻率
    omega_mn = calculate_natural_frequency(m, n, R, material) * 2 * np.pi
    
    # 模態阻尼
    zeta = 0.02
    
    # 獲取力的作用位置
    r0_norm = force_params['position'][0]
    theta0 = force_params['position'][1]
    
    # 計算力的參與因子（不除以面積）
    force_term = bessel_mode_shape(r0_norm, theta0, m, n)
    
    # 更新頻率響應函數
    H = 1 / (tension_term/modal_mass - omega**2 + 2j*zeta*omega_mn*omega)
    
    # 計算模態響應
    modal_response = (F0 * force_term * H * phi_mn) / modal_mass
    
    # 返回實部，並轉換為微米單位
    return np.real(modal_response * np.exp(1j * omega * t)) * 1e6

class MembraneVibrationAnimation:
    def __init__(self, material, modes, force_params, radii=[0.01, 0.02], 
                 resolution=20, duration=1.0, fps=10, use_total_tension=False):
        """
        初始化動畫物件
        Parameters:
        -----------
        material: dict，包含 'T' (可能為線張力或總張力) 与 'rho_s' (面密度)
        use_total_tension: bool，若 True 則將 material['T'] 視為「總張力 (N)」並動態換算
                           若 False (預設) 則直接視為線張力(N/m)
        """
        self.material = material
        self.modes = modes
        self.force_params = force_params
        self.radii = radii
        self.use_total_tension = use_total_tension
        
        # 創建較小的網格
        r = np.linspace(0, 1, resolution)
        theta = np.linspace(0, 2*np.pi, resolution)
        self.R, self.THETA = np.meshgrid(r, theta)
        self.X = self.R * np.cos(self.THETA)
        self.Y = self.R * np.sin(self.THETA)
        
        # 減少時間步數
        self.t = np.linspace(0, duration, int(duration * fps))
        
        # 預計算 Bessel 函數值
        self._precalculate_bessel()
        
        # 設置圖形
        self._setup_figure()
        
        # 修改動畫初始化
        self.anim = FuncAnimation(
            self.fig, 
            self._update,
            init_func=self._init_animation,  # 添加初始化函數
            frames=len(self.t),
            interval=1000/fps,
            blit=False,  # 關閉 blit
            repeat=True
        )
    
    def _precalculate_bessel(self):
        """預計算 Bessel 函數值"""
        self.bessel_values = {}
        for m, n in self.modes:
            lambda_mn = jn_zeros(m, n)[n-1]
            self.bessel_values[(m,n)] = jn(m, lambda_mn * self.R)
    
    def _calculate_response(self, R, t):
        """優化的響應計算，將 'use_total_tension' 帶入自然頻率計算"""
        Z_sum = np.zeros_like(self.X)
        omega = self.force_params['frequency'] * 2 * np.pi
        
        for m, n in self.modes:
            # 使用改良後的自然頻率計算
            omega_mn = calculate_natural_frequency(
                m, n, R, self.material, use_total_tension=self.use_total_tension
            ) * 2.0 * np.pi
            
            # 計算頻率響應
            H = 1 / (omega_mn**2 - omega**2 + 0.04j * omega_mn * omega)
            
            # 計算響應
            response = np.real(H * self.bessel_values[(m,n)] 
                               * np.cos(m * self.THETA) 
                               * np.exp(1j * omega * t))
            Z_sum += response
        
        return Z_sum * 1e6  # 轉換為微米
    
    def _update(self, frame):
        """更新動畫幀"""
        t = self.t[frame]
        artists = []  # 收集並返回本幀要更新的所有 Artist
        
        for idx, R in enumerate(self.radii):
            
            # 1) 若 surface 存在 → 嘗試 remove
            if self.surfs[idx] is not None:
                try:
                    self.surfs[idx].remove()
                except ValueError:
                    pass  # 若不在 collections 裏，就直接跳過
            
            # 2) 計算新的位移 Z，重新畫 surface
            Z = self._calculate_response(R, t)
            self.surfs[idx] = self.axes[idx].plot_surface(
                self.X * R, self.Y * R, Z,
                cmap=cm.coolwarm,
                rcount=20, ccount=20
            )
            artists.append(self.surfs[idx])
            
            # 3) 若 arrow 存在 → 嘗試 remove
            if self.arrows[idx] is not None:
                try:
                    self.arrows[idx].remove()
                except ValueError:
                    pass
            
            # 4) 更新箭頭
            self._update_force_arrow(idx, t)  
            artists.append(self.arrows[idx])
        
        # 進度條同理
        if self.progress_patch is not None:
            try:
                self.progress_patch.remove()
            except ValueError:
                pass
        
        self.ax_progress.collections.clear()
        self.progress_patch = self.ax_progress.fill_between(
            [0, frame], [0, 0], [1, 1],
            color='blue', alpha=0.3
        )
        artists.append(self.progress_patch)
        
        return artists  # 返回所有更新的藝術家
    
    def _setup_figure(self):
        """設置圖形和軸"""
        self.fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(3, len(self.radii), height_ratios=[1, 1, 0.1])
        
        self.axes = []
        self.surfs = []
        self.arrows = []
        
        # 為每個半徑創建子圖
        for i, R in enumerate(self.radii):
            ax = self.fig.add_subplot(gs[0:2, i], projection='3d')
            freq_11 = calculate_natural_frequency(1, 1, R, self.material)
            
            ax.set_title(f'R = {R*1000:.1f}mm\nNatural Freq (1,1): {freq_11:.1f} Hz')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_zlabel('Displacement (μm)')  # 改為微米單位
            
            # 設置軸範圍
            ax.set_xlim(-R, R)
            ax.set_ylim(-R, R)
            
            # 初始化表面
            surf = ax.plot_surface(
                self.X * R, 
                self.Y * R, 
                np.zeros_like(self.X),
                cmap=cm.coolwarm
            )
            
            self.axes.append(ax)
            self.surfs.append(surf)
            
            # 添加力箭頭
            arrow = self._add_force_arrow(ax, R)
            self.arrows.append(arrow)
        
        # 添加進度條
        self.ax_progress = self.fig.add_subplot(gs[2, :])
        self.ax_progress.set_xlim(0, len(self.t))
        self.ax_progress.set_ylim(0, 1)
        self.ax_progress.set_title('Progress')
        
        plt.tight_layout()
        
        # 設置 z 軸範圍
        self._set_z_limits()
    
    def _add_force_arrow(self, ax, R):
        """添加表示外力的箭頭"""
        r0_norm = self.force_params['position'][0]
        theta0 = self.force_params['position'][1]
        
        # 轉換極座標到笛卡爾座標
        x0 = r0_norm * R * np.cos(theta0)
        y0 = r0_norm * R * np.sin(theta0)
        
        # 設置箭頭基準長度（不隨半徑變化）
        self.arrow_base_length = 0.2 * R
        
        # 創建箭頭線段
        segments = [[
            (x0, y0, 0),
            (x0, y0, self.arrow_base_length)
        ]]
        arrow = Line3DCollection(segments, colors='red', alpha=0.6)
        ax.add_collection3d(arrow)
        return arrow
    
    def _update_force_arrow(self, idx, t):
        """更新力箭頭的長度"""
        R = self.radii[idx]
        r0_norm = self.force_params['position'][0]
        theta0 = self.force_params['position'][1]
        
        # 轉換極座標到笛卡爾座標
        x0 = r0_norm * R * np.cos(theta0)
        y0 = r0_norm * R * np.sin(theta0)
        
        # 箭頭長度隨時間做簡諧變化
        omega = self.force_params['frequency'] * 2 * np.pi
        scale = np.cos(omega * t)
        arrow_length = self.arrow_base_length * scale
        
        # 更新箭頭位置
        self.arrows[idx].set_segments([[
            (x0, y0, 0),
            (x0, y0, arrow_length)
        ]])
    
    def _update_progress_bar(self, frame):
        """更新進度條"""
        self.ax_progress.collections.clear()
        progress = frame / len(self.t)
        self.ax_progress.fill_between(
            [0, progress * len(self.t)],
            [0, 0],
            [1, 1],
            color='blue',
            alpha=0.3
        )
    
    def _set_z_limits(self):
        """優化的 z 軸範圍計算"""
        max_disps = []
        
        # 只計算一個時間點
        t_sample = self.t[0]
        for R in self.radii:
            Z = self._calculate_response(R, t_sample)
            max_disps.append(np.max(np.abs(Z)))
        
        # 設置統一的 z 軸範圍
        z_limit = max(max_disps) * 1.2
        for ax in self.axes:
            ax.set_zlim(-z_limit, z_limit)
    
    def _init_animation(self):
        """初始化動畫 (在 FuncAnimation(..., init_func=...) 裏被呼叫)"""
        # 初始化空容器
        self.surfs = [None] * len(self.radii)
        self.arrows = [None] * len(self.radii)
        self.progress_patch = None
        
        # 先為每個 subplot 放一個空的 surface、arrow
        for idx, R in enumerate(self.radii):
            self.surfs[idx] = self.axes[idx].plot_surface(
                self.X * R, 
                self.Y * R, 
                np.zeros_like(self.X),
                cmap=cm.coolwarm,
                rcount=20, 
                ccount=20
            )
            # 在此直接呼叫已寫好的函式 _add_force_arrow(...)
            self.arrows[idx] = self._add_force_arrow(self.axes[idx], R)
        
        # 也可以在這邊清空進度條
        self.ax_progress.collections.clear()
        return self.surfs + self.arrows

if __name__ == "__main__":
    # 定義振膜材料參數
    material = {
        'name': 'Membrane',
        'T': 2000.0,     # 張力 (N/m)
        'rho_s': 0.1,    # 面密度 (kg/m²)
    }
    
    # 減少模態數量
    modes = [(0,1), (1,1)]  # 只使用兩個主要模態
    
    force_params = {
        'amplitude': 0.001,    # 1 mN
        'frequency': 1000,     # 1000 Hz
        'position': (0.5, 0)   
    }
    
    radii = [0.01, 0.02]
    
    print("開始模擬...")
    anim = MembraneVibrationAnimation(
        material,
        modes,
        force_params,
        radii=radii,
        resolution=20,    # 降低分辨率
        duration=1.0,     # 縮短動畫時間
        fps=10           # 降低幀率
    )
    
    plt.show() 