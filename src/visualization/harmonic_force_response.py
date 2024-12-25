import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Arrow
from tqdm import tqdm
import matplotlib.gridspec as gridspec

"""
研究背景與動機:
振動問題在許多工程領域都至關重要，例如建築結構、機械設計與航太工程等，因為外部動力激振將會導致系統受到不斷累積的疲勞損傷，從而降低其使用壽命和可靠性。在這過程中，理解板的振動模態與響應不僅有助於預測結構在各種固定或動態載荷下的行為，也能在設計階段做適當之加固或抑振對策。

前後因果關係邏輯:
1. 由於外部施加的簡諧力會不斷激發板結構內部的自然頻率與模態，因此必須先求出板的彎曲剛度和自然頻率。
2. 不同模態對總位移響應的貢獻程度可藉由模態疊加理論加以量化，並評估其造成的振幅大小與波形分布。
3. 若結構阻尼偏低或外力頻率與某些模態自然頻率接近，將產生顯著共振效應並使振幅大幅上升。
4. 綜合上述結果，可藉此進一步更新材料或幾何參數設計，避免在實務運用中出現結構破壞或性能衰減。

目的:
本程式以簡支板為模型，結合彈性力學與振動理論，透過模態疊加與強迫振動響應的計算，模擬在外部簡諧力作用下板的動態行為，以提供對振動特性的可視化工具。

用途:
1. 分析不同模態對整體振動的貢獻程度。
2. 評估材料及幾何參數改變時對振幅與自然頻率的影響。
3. 視覺化觀察板在受力下的形變，輔助工程設計與研究分析。

預期結果:
藉由此程式，我們期望能夠在不同的外力頻率與阻尼比條件下，模擬出的板振幅分布能準確反映出各模態的疊加效應，並在動態過程中更直觀地觀察出可能產生的共振情形。

簡易的分析推理過程:
1. 根據材料及幾何參數計算板的彎曲剛度，進一步導出自然頻率。  
2. 逐一計算各指定模態的受迫響應，並將它們疊加得到整體位移。  
3. 檢查外力頻率與自然頻率的關係，若吻合則預期出現較顯著的振動。  
4. 最後透過動畫或圖表方式顯示結果，協助進一步優化設計或改良材料選擇。

主要使用的物理數學公式:
1. 板的彎曲剛度 (Bending Stiffness): D = (E * h^3) / [12 * (1 - ν^2)]
2. 自然頻率 ω_mn (只保留與 m、n 相關係數): 
   ω_mn = (π^2 / 2) * √(D / (ρ * h)) * (m² + n²)
3. 模態疊加：將各模態在頻率或時域的響應進行加總，反映板整體的位移分布。
4. 頻率響應函數 (FRF): H(ω) = 1 / (ω_mn² - ω² + 2jζω_mnω)
   其中 ζ 為模態阻尼比 (此程式採用 0.02 作為簡化設定)。
"""

def calculate_bending_stiffness(E, h, nu):
    """計算板的彎曲剛度"""
    return (E * h**3) / (12 * (1 - nu**2))

def mode_shape(x, y, m, n):
    """計算模態形狀"""
    return np.sin(m * np.pi * x) * np.sin(n * np.pi * y)

def calculate_modal_mass(rho, h, L, W):
    """計算模態質量"""
    return rho * h * L * W / 4  # 對於簡支板，模態質量為總質量的1/4

def forced_response(x, y, t, material, modes, force_params):
    """
    計算在簡諧力作用下板的受迫振動響應
    
    參數:
    x, y: 空間坐標點
    t: 時間點
    material: 材料參數字典
    modes: 要考慮的模態列表 [(m1,n1), (m2,n2),...]
    force_params: 力的參數字典 {'amplitude': F0, 'frequency': omega, 'position': (x0,y0)}
    """
    D = calculate_bending_stiffness(material['E'], material['h'], material['nu'])
    modal_mass = calculate_modal_mass(material['rho'], material['h'], 1.0, 1.0)
    
    # 力的參數
    F0 = force_params['amplitude']
    omega = force_params['frequency']
    x0, y0 = force_params['position']
    
    response = 0
    for m, n in modes:
        # 自然頻率
        omega_mn = (np.pi**2 / 2) * np.sqrt(D / (material['rho'] * material['h'])) * (m**2 + n**2)
        
        # 模態阻尼（假設模態阻尼比為0.02）
        zeta = 0.02
        
        # 力的模態參與因子
        force_term = mode_shape(x0, y0, m, n)
        
        # 頻率響應函數
        H = 1 / (omega_mn**2 - omega**2 + 2j*zeta*omega_mn*omega)
        
        # 該模態的響應
        modal_response = (F0 * force_term * H * mode_shape(x, y, m, n) / modal_mass)
        
        # 總響應是所有模態響應的疊加
        response += np.real(modal_response * np.exp(1j * omega * t))
    
    return response

def animate_forced_response(material, modes, force_params, duration=2.0, fps=30):
    """創建受迫振動的動畫"""
    # 創建網格
    resolution = 50
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    # 創建時間序列
    t = np.linspace(0, duration, int(duration * fps))
    
    # 設置圖形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 初始化表面
    surf = [ax.plot_surface(X, Y, np.zeros_like(X), cmap=cm.coolwarm)]
    
    # 設置視角和標籤
    ax.view_init(elev=30, azim=45)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Displacement')
    ax.set_title(f'Forced Response of {material["name"]}\nForce Frequency: {force_params["frequency"]:.1f} Hz')
    
    # 設置固定的z軸範圍
    max_disp = np.max([np.abs(forced_response(X, Y, t_i, material, modes, force_params)) for t_i in t[:5]])
    ax.set_zlim(-max_disp*1.2, max_disp*1.2)
    
    def update(frame):
        # 清除上一幀的表面
        surf[0].remove()
        # 計算新的位移
        Z = forced_response(X, Y, t[frame], material, modes, force_params)
        # 繪製新的表面
        surf[0] = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
        return surf
    
    anim = FuncAnimation(fig, update, frames=len(t), interval=1000/fps)
    return anim

def forced_modal_response(x, y, t, material, m, n, force_params):
    """計算單一模態的受迫振動響應"""
    D = calculate_bending_stiffness(material['E'], material['h'], material['nu'])
    modal_mass = calculate_modal_mass(material['rho'], material['h'], 1.0, 1.0)
    
    # 力的參數
    F0 = force_params['amplitude']
    omega = force_params['frequency']
    x0, y0 = force_params['position']
    
    # 自然頻率
    omega_mn = (np.pi**2 / 2) * np.sqrt(D / (material['rho'] * material['h'])) * (m**2 + n**2)
    
    # 模態阻尼
    zeta = 0.02
    
    # 力的模態參與因子
    force_term = mode_shape(x0, y0, m, n)
    
    # 頻率響應函數
    H = 1 / (omega_mn**2 - omega**2 + 2j*zeta*omega_mn*omega)
    
    # 該模態的響應
    modal_response = (F0 * force_term * H * mode_shape(x, y, m, n) / modal_mass)
    
    return np.real(modal_response * np.exp(1j * omega * t))

class AnimatedForcedModes:
    def __init__(self, material, modes, force_params, resolution=50, duration=2.0, fps=30):
        self.material = material
        self.modes = modes
        self.force_params = force_params
        self.fps = fps
        
        # 創建網格
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        self.X, self.Y = np.meshgrid(x, y)
        
        # 創建時間序列
        self.t = np.linspace(0, duration, int(duration * fps))
        
        # 創建圖形
        self.fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(3, len(modes), height_ratios=[1, 1, 0.1])
        
        # 初始化所有子圖
        self.ax_modes = []
        self.surfs = []
        self.arrows = []
        self.arrow_artists = []  # 存儲箭頭的藝術家對象
        
        # 為每個模態創建子圖
        for i in range(len(modes)):
            ax = self.fig.add_subplot(gs[0, i], projection='3d')
            self.ax_modes.append(ax)
            m, n = modes[i]
            ax.set_title(f'Mode ({m},{n})\nωn = {self._calculate_natural_freq(m,n):.1f} Hz')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('Displacement')
            
            # 初始化表面
            surf = ax.plot_surface(self.X, self.Y, np.zeros_like(self.X), cmap=cm.coolwarm)
            self.surfs.append(surf)
            
            # 修改箭頭的添加方式
            arrow = self._add_force_arrow(ax)
            self.arrows.append(arrow)
            self.arrow_artists.append(ax.add_collection3d(arrow))
        
        # 創建疊加結果的子圖
        self.ax_sum = self.fig.add_subplot(gs[1, :], projection='3d')
        self.ax_sum.set_title(f'Superposition (Force: {force_params["frequency"]:.1f} Hz)')
        self.ax_sum.set_xlabel('x')
        self.ax_sum.set_ylabel('y')
        self.ax_sum.set_zlabel('Displacement')
        
        # 初始化疊加表面
        self.surf_sum = self.ax_sum.plot_surface(self.X, self.Y, np.zeros_like(self.X), cmap=cm.coolwarm)
        
        # 修改疊加圖的箭頭添加方式
        self.arrow_sum = self._add_force_arrow(self.ax_sum)
        self.arrow_sum_artist = self.ax_sum.add_collection3d(self.arrow_sum)
        
        # 添加進度條軸
        self.ax_progress = self.fig.add_subplot(gs[2, :])
        self.ax_progress.set_xlim(0, len(self.t))
        self.ax_progress.set_ylim(0, 1)
        self.progress_line = self.ax_progress.fill_between([0], [0], [1], color='blue', alpha=0.3)
        self.ax_progress.set_title('Progress')
        self.ax_progress.set_xticks([])
        self.ax_progress.set_yticks([])
        
        # 設置所有圖的視角
        for ax in self.ax_modes + [self.ax_sum]:
            ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        
        # 計算最大位移用於設置z軸範圍
        max_disp = self._calculate_max_displacement()
        for ax in self.ax_modes + [self.ax_sum]:
            ax.set_zlim(-max_disp, max_disp)
        
        # 創建動畫
        self.anim = FuncAnimation(
            self.fig, self._update, frames=len(self.t),
            interval=1000/fps, blit=False)
    
    def _calculate_natural_freq(self, m, n):
        """計算自然頻率"""
        D = calculate_bending_stiffness(self.material['E'], self.material['h'], self.material['nu'])
        return (np.pi**2 / 2) * np.sqrt(D / (self.material['rho'] * self.material['h'])) * (m**2 + n**2) / (2*np.pi)
    
    def _calculate_max_displacement(self):
        """計算最大位移用於設置z軸範圍"""
        max_disp = 0
        for t_i in self.t[:5]:  # 只用前幾個時間點來估計
            for m, n in self.modes:
                Z = forced_modal_response(self.X, self.Y, t_i, self.material, m, n, self.force_params)
                max_disp = max(max_disp, np.max(np.abs(Z)))
        return max_disp * 1.2
    
    def _add_force_arrow(self, ax):
        """添加表示外力的箭頭"""
        x0, y0 = self.force_params['position']
        arrow_length = 0.2
        
        # 使用 Line3DCollection 代替 quiver
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        segments = [[(x0, y0, 0), (x0, y0, arrow_length)]]
        arrow = Line3DCollection(segments, colors='red', alpha=0.6)
        return arrow
    
    def _update_force_arrows(self, t):
        """更新外力箭頭的長度"""
        omega = self.force_params['frequency'] * 2 * np.pi
        scale = np.cos(omega * t)  # 簡諧運動
        arrow_length = 0.2 * scale  # 基礎長度 * 縮放
        x0, y0 = self.force_params['position']
        
        # 更新所有箭頭
        segments = [[(x0, y0, 0), (x0, y0, arrow_length)]]
        for arrow in self.arrows:
            arrow.set_segments(segments)
        self.arrow_sum.set_segments(segments)
    
    def _update(self, frame):
        t = self.t[frame]
        
        # 更新外力箭頭
        self._update_force_arrows(t)
        
        # 更新每個模態
        Z_sum = np.zeros_like(self.X)
        for idx, (m, n) in enumerate(self.modes):
            Z = forced_modal_response(self.X, self.Y, t, self.material, m, n, self.force_params)
            Z_sum += Z
            
            # 更新個別模態的表面
            self.surfs[idx].remove()
            self.surfs[idx] = self.ax_modes[idx].plot_surface(
                self.X, self.Y, Z, cmap=cm.coolwarm)
        
        # 更新疊加結果
        self.surf_sum.remove()
        self.surf_sum = self.ax_sum.plot_surface(
            self.X, self.Y, Z_sum, cmap=cm.coolwarm)
        
        # 更新進度條
        self.ax_progress.collections.clear()
        progress = frame / len(self.t)
        self.ax_progress.fill_between([0, frame], [0, 0], [1, 1], color='blue', alpha=0.3)

if __name__ == "__main__":
    # 定義材料參數
    material = {
        'name': 'Steel',
        'E': 200e9,    # 200 GPa
        'rho': 7800,   # 7800 kg/m³
        'nu': 0.3,
        'h': 0.01      # 10mm厚度
    }
    
    # 定義要考慮的模態
    modes = [(1,1), (2,1), (1,2)]
    
    # 定義力的參數
    force_params = {
        'amplitude': 1000,  # 1000 N
        'frequency': 100,   # 100 Hz
        'position': (0.5, 0.5)  # 在板的中心施加力
    }
    
    # 修改保存動畫的部分
    print("正在生成動畫...")
    anim_modes = AnimatedForcedModes(material, modes, force_params, duration=5.0)
    
    print("正在保存動畫...")
    try:
        anim_modes.anim.save('forced_response_modes.gif', 
                           writer='pillow', 
                           fps=15,
                           progress_callback=lambda i, n: print(f'保存進度: {i+1}/{n}'))
    except Exception as e:
        print(f"保存動畫失敗: {str(e)}")
    
    plt.show()