import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def calculate_bending_stiffness(E, h, nu):
    """計算板的彎曲剛度"""
    return (E * h**3) / (12 * (1 - nu**2))

def calculate_natural_frequency(m, n, L, D, rho, h):
    """計算矩形板的自然頻率"""
    return (np.pi**2 / (2 * L**2)) * np.sqrt(D / (rho * h)) * (m**2 + n**2)

def mode_shape(x, y, m, n):
    """計算模態形狀"""
    return np.sin(m * np.pi * x) * np.sin(n * np.pi * y)

class MaterialResponseAnimation:
    def __init__(self, materials, modes=[(1,1)], duration=5.0, fps=30):
        self.materials = materials
        self.modes = modes
        self.fps = fps
        self.duration = duration
        
        # 創建網格
        resolution = 50
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        self.X, self.Y = np.meshgrid(x, y)
        
        # 計算每個材料的頻率
        self.frequencies = {}
        for material in materials:
            D = calculate_bending_stiffness(material['E'], 0.01, material['nu'])
            material_freqs = []
            for m, n in modes:
                freq = calculate_natural_frequency(m, n, 1.0, D, material['rho'], 0.01)
                material_freqs.append(freq)
            self.frequencies[material['name']] = material_freqs
        
        # 創建圖形
        self.fig = plt.figure(figsize=(15, 5))
        
        # 為每個材料創建子圖
        self.axes = []
        self.surfs = []
        for i in range(len(materials)):
            ax = self.fig.add_subplot(1, len(materials), i+1, projection='3d')
            self.axes.append(ax)
            ax.set_title(f"{materials[i]['name']}\nf = {self.frequencies[materials[i]['name']][0]:.1f} Hz")
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('Amplitude')
            ax.set_zlim(-2, 2)
            ax.view_init(elev=30, azim=45)
            
            # 初始化表面
            surf = ax.plot_surface(self.X, self.Y, np.zeros_like(self.X), cmap=cm.coolwarm)
            self.surfs.append(surf)
        
        plt.tight_layout()
    
    def _update_frame(self, frame):
        """更新單個幀"""
        t = frame / self.fps
        
        # 更新每個材料的響應
        for ax_idx, (ax, material, surf) in enumerate(zip(self.axes, self.materials, self.surfs)):
            # 計算總響應（所有模態的疊加）
            Z = np.zeros_like(self.X)
            for (m, n), freq in zip(self.modes, self.frequencies[material['name']]):
                damping = np.exp(-t * 0.5)  # 衰減係數
                Z += mode_shape(self.X, self.Y, m, n) * np.cos(2*np.pi*freq*t) * damping
            
            # 更新表面數據
            surf.remove()
            self.surfs[ax_idx] = ax.plot_surface(self.X, self.Y, Z, cmap=cm.coolwarm)
            
            # 更新視角
            ax.view_init(elev=30, azim=45 + t*30)
            
            # 更新標題（可選）
            ax.set_title(f"{material['name']}\nTime: {t:.1f}s")
    
    def save_animation(self, filename, fps=15):
        """保存動畫為文件"""
        # 創建新的動畫對象用於保存
        save_anim = animation.FuncAnimation(
            self.fig, self._update_frame,
            frames=int(self.duration * fps),
            interval=1000/fps)
        
        # 保存為GIF
        save_anim.save(filename, writer='pillow', fps=fps)
        plt.close()  # 關閉保存用的圖形
        
        # 重新創建用於顯示的動畫
        self.anim = animation.FuncAnimation(
            self.fig, self._update_frame,
            frames=int(self.duration * self.fps),
            interval=1000/self.fps)

def plot_frequency_response(materials, modes=[(1,1), (2,1), (1,2), (2,2)], f_range=None):
    """繪製頻率響應函數"""
    if f_range is None:
        f_range = np.linspace(0, 500, 1000)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for material in materials:
        # 計算該材料的自然頻率
        D = calculate_bending_stiffness(material['E'], 0.01, material['nu'])
        natural_freqs = []
        for m, n in modes:
            freq = calculate_natural_frequency(m, n, 1.0, D, material['rho'], 0.01)
            natural_freqs.append(freq)
        
        # 計算頻率響應函數
        response = np.zeros_like(f_range)
        damping = 0.02  # 假設阻尼比
        for nat_freq in natural_freqs:
            # 使用單自由度系統的頻率響應函數
            r = f_range / nat_freq
            response += 1 / np.sqrt((1 - r**2)**2 + (2*damping*r)**2)
        
        # 繪製響應曲線
        ax.plot(f_range, 20*np.log10(response), label=material['name'])
        
        # 標記自然頻率
        for freq in natural_freqs:
            ax.axvline(x=freq, color='gray', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Frequency Response Function')
    ax.legend()
    ax.grid(True)
    
    return fig

if __name__ == "__main__":
    # 定義材料參數
    materials = [
        {
            'name': 'Steel',
            'E': 200e9,    # 200 GPa
            'rho': 7800,   # 7800 kg/m³
            'nu': 0.3
        },
        {
            'name': 'Aluminum',
            'E': 70e9,     # 70 GPa
            'rho': 2700,   # 2700 kg/m³
            'nu': 0.33
        },
        {
            'name': 'Titanium',
            'E': 110e9,    # 110 GPa
            'rho': 4500,   # 4500 kg/m³
            'nu': 0.34
        }
    ]
    
    # 創建時域響應動畫
    anim = MaterialResponseAnimation(materials, modes=[(1,1), (2,1)], duration=10.0)
    
    # 保存動畫為GIF
    print("正在保存動畫為GIF...")
    anim.save_animation('material_response.gif', fps=15)
    print("GIF保存完成")
    
    # 創建頻率響應圖
    fig_frf = plot_frequency_response(materials)
    plt.savefig('frequency_response.png')
    plt.close()
    
    # 顯示互動式動畫
    print("顯示互動式動畫...")
    plt.show()