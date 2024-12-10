import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def plot_mode_shape(m, n, Lx=1.0, Ly=1.0, resolution=50):
    """
    繪製特定模態序數(m,n)下的模態形狀
    
    參數:
    m, n: 模態序數
    Lx, Ly: 板的長度和寬度
    resolution: 網格解析度
    """
    # 創建網格點
    x = np.linspace(0, Lx, resolution)
    y = np.linspace(0, Ly, resolution)
    X, Y = np.meshgrid(x, y)
    
    # 計算模態形狀
    Z = np.sin(m * np.pi * X / Lx) * np.sin(n * np.pi * Y / Ly)
    
    # 創建3D圖
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 繪製表面
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0)
    
    # 添加顏色條
    fig.colorbar(surf)
    
    # 設置標籤
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Φ(x,y)')
    ax.set_title(f'Mode Shape (m={m}, n={n})')
    
    return fig

def plot_multiple_modes():
    """
    繪製多個不同模態序數的模態形狀
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 定義要展示的模態序數組合
    modes = [(1,1), (1,2), (2,1), (2,2)]
    
    for idx, (m, n) in enumerate(modes, 1):
        ax = fig.add_subplot(2, 2, idx, projection='3d')
        
        # 創建網格點
        x = np.linspace(0, 1, 50)
        y = np.linspace(0, 1, 50)
        X, Y = np.meshgrid(x, y)
        
        # 計算模態形狀
        Z = np.sin(m * np.pi * X) * np.sin(n * np.pi * Y)
        
        # 繪製表面
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0)
        
        # 設置標籤
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Φ(x,y)')
        ax.set_title(f'Mode Shape (m={m}, n={n})')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # 展示單個模態形狀
    fig1 = plot_mode_shape(m=2, n=3)
    plt.savefig('single_mode_shape.png')
    
    # 展示多個模態形狀
    fig2 = plot_multiple_modes()
    plt.savefig('multiple_mode_shapes.png')
    
    plt.show() 