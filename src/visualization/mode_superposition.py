import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def mode_shape(x, y, m, n, Am=1.0):
    """計算單個模態形狀"""
    return Am * np.sin(m * np.pi * x) * np.sin(n * np.pi * y)

def create_superposition_visualization(resolution=50, modes=None, amplitudes=None):
    """
    創建模態疊加的視覺化
    
    參數:
    resolution: 網格解析度
    modes: 模態序數列表，每個元素為(m,n)
    amplitudes: 各模態的振幅
    """
    if modes is None:
        modes = [(1,1), (2,2), (3,1)]
    if amplitudes is None:
        amplitudes = [1.0, 0.5, 0.3]
        
    # 創建網格
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    # 創建圖形
    fig = plt.figure(figsize=(15, 8))
    
    # 計算總行數（基於模態數量）
    n_modes = len(modes)
    
    # 繪製各個模態
    for idx, ((m, n), amp) in enumerate(zip(modes, amplitudes)):
        ax = fig.add_subplot(2, n_modes + 1, idx + 1, projection='3d')
        Z = mode_shape(X, Y, m, n, amp)
        
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
        ax.set_title(f'Mode {idx+1}: (m={m},n={n})\nA={amp:.1f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Φ')
        
    # 計算並繪製疊加結果
    ax_sum = fig.add_subplot(2, 1, 2, projection='3d')
    Z_sum = np.zeros_like(X)
    for (m, n), amp in zip(modes, amplitudes):
        Z_sum += mode_shape(X, Y, m, n, amp)
    
    surf = ax_sum.plot_surface(X, Y, Z_sum, cmap=cm.coolwarm)
    ax_sum.set_title('Superposition of All Modes')
    ax_sum.set_xlabel('x')
    ax_sum.set_ylabel('y')
    ax_sum.set_zlabel('Φ')
    
    plt.tight_layout()
    return fig

class AnimatedModes:
    def __init__(self, resolution=50, modes=None, amplitudes=None, frequencies=None, duration=2.0, fps=30):
        if modes is None:
            modes = [(1,1), (2,2), (3,1)]
        if amplitudes is None:
            amplitudes = [1.0, 0.5, 0.3]
        if frequencies is None:
            frequencies = [1.0, 2.0, 3.0]
            
        self.modes = modes
        self.amplitudes = amplitudes
        self.frequencies = frequencies
        self.fps = fps
        
        # 創建網格
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        self.X, self.Y = np.meshgrid(x, y)
        
        # 創建圖形
        self.fig = plt.figure(figsize=(15, 5))
        
        # 設置子圖
        self.ax_modes = []
        self.surfs_modes = []
        
        # 初始化所有子圖
        for i in range(len(modes)):
            ax = self.fig.add_subplot(1, len(modes) + 1, i + 1, projection='3d')
            self.ax_modes.append(ax)
            ax.set_title(f'Mode {i+1}: (m={modes[i][0]},n={modes[i][1]})')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('Φ')
            ax.set_zlim(-2, 2)
            ax.view_init(elev=30, azim=45)
            
            # 初始化表面為零
            Z = np.zeros_like(self.X)
            surf = ax.plot_surface(self.X, self.Y, Z, cmap=cm.coolwarm)
            self.surfs_modes.append(surf)
            
        # 初始化疊加圖
        self.ax_sum = self.fig.add_subplot(1, len(modes) + 1, len(modes) + 1, projection='3d')
        self.ax_sum.set_title('Superposition')
        self.ax_sum.set_xlabel('x')
        self.ax_sum.set_ylabel('y')
        self.ax_sum.set_zlabel('Φ')
        self.ax_sum.set_zlim(-2, 2)
        self.ax_sum.view_init(elev=30, azim=45)
        
        Z_sum = np.zeros_like(self.X)
        self.surf_sum = self.ax_sum.plot_surface(self.X, self.Y, Z_sum, cmap=cm.coolwarm)
        
        plt.tight_layout()
        
        # 創建動畫
        frames = int(duration * fps)
        self.anim = animation.FuncAnimation(
            self.fig, self._update, frames=frames,
            interval=1000/fps, blit=False)
    
    def _update(self, frame):
        t = frame / self.fps
        
        # 更新每個模態
        for idx, ((m, n), amp, freq) in enumerate(zip(self.modes, self.amplitudes, self.frequencies)):
            Z = mode_shape(self.X, self.Y, m, n, amp) * np.cos(2 * np.pi * freq * t)
            
            # 更新表面數據
            self.surfs_modes[idx].remove()
            self.surfs_modes[idx] = self.ax_modes[idx].plot_surface(
                self.X, self.Y, Z, cmap=cm.coolwarm)
            
        # 更新疊加結果
        Z_sum = np.zeros_like(self.X)
        for (m, n), amp, freq in zip(self.modes, self.amplitudes, self.frequencies):
            Z_sum += mode_shape(self.X, self.Y, m, n, amp) * np.cos(2 * np.pi * freq * t)
        
        self.surf_sum.remove()
        self.surf_sum = self.ax_sum.plot_surface(
            self.X, self.Y, Z_sum, cmap=cm.coolwarm)

if __name__ == "__main__":
    # 創建靜態疊加視覺化
    fig_static = create_superposition_visualization()
    plt.savefig('mode_superposition_static.png')
    plt.close()
    
    # 創建動態時間演化動畫
    anim_modes = AnimatedModes(duration=5.0)
    
    # 保存動畫為 GIF 格式
    try:
        anim_modes.anim.save('mode_superposition_dynamic.gif', writer='pillow', fps=15)
    except Exception as e:
        print(f"保存動畫失敗: {str(e)}")
        print("顯示互動式視窗...")
    
    plt.show() 