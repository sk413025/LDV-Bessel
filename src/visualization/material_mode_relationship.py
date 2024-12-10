import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def calculate_bending_stiffness(E, h, nu):
    """計算板的彎曲剛度"""
    return (E * h**3) / (12 * (1 - nu**2))

def calculate_natural_frequency(m, n, L, D, rho, h):
    """計算矩形板的自然頻率"""
    return (np.pi**2 / (2 * L**2)) * np.sqrt(D / (rho * h)) * (m**2 + n**2)

def mode_shape(x, y, m, n):
    """計算模態形狀"""
    return np.sin(m * np.pi * x) * np.sin(n * np.pi * y)

def plot_mode_shapes_for_different_materials(materials, modes=[(1,1), (2,1), (1,2), (2,2)]):
    """
    為不同材料繪製模態形狀和頻率比較
    
    參數:
    materials: 字典列表，每個字典包含材料的E（彈性模量）、rho（密度）、nu（泊松比）
    modes: 要顯示的模態序數列表
    """
    # 創建網格
    resolution = 50
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    # 設置圖形
    n_materials = len(materials)
    n_modes = len(modes)
    fig = plt.figure(figsize=(15, 4*n_materials))
    
    # 計算基準頻率（用於歸一化）
    base_material = materials[0]
    base_D = calculate_bending_stiffness(base_material['E'], 0.01, base_material['nu'])
    base_frequencies = []
    for m, n in modes:
        base_freq = calculate_natural_frequency(m, n, 1.0, base_D, base_material['rho'], 0.01)
        base_frequencies.append(base_freq)
    
    # 為每個材料繪製模態
    for mat_idx, material in enumerate(materials):
        # 計算該材料的彎曲剛度
        D = calculate_bending_stiffness(material['E'], 0.01, material['nu'])
        
        # 為每個模態創建子圖
        for mode_idx, (m, n) in enumerate(modes):
            ax = fig.add_subplot(n_materials, n_modes, mat_idx*n_modes + mode_idx + 1, projection='3d')
            
            # 計算模態形狀
            Z = mode_shape(X, Y, m, n)
            
            # 計算該模態的自然頻率
            freq = calculate_natural_frequency(m, n, 1.0, D, material['rho'], 0.01)
            freq_ratio = freq / base_frequencies[mode_idx]
            
            # 繪製表面
            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
            
            # 設置視角和標籤
            ax.view_init(elev=30, azim=45)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('Φ')
            
            # 添加標題，包含頻率比較
            title = f'Material: {material["name"]}\nMode ({m},{n})\nf/f_base = {freq_ratio:.2f}'
            ax.set_title(title)
    
    plt.tight_layout()
    return fig

def plot_frequency_comparison(materials, modes=[(1,1), (2,1), (1,2), (2,2)]):
    """繪製不同材料的頻率比較圖"""
    # 計算每個材料的頻率
    frequencies = []
    for material in materials:
        D = calculate_bending_stiffness(material['E'], 0.01, material['nu'])
        material_freqs = []
        for m, n in modes:
            freq = calculate_natural_frequency(m, n, 1.0, D, material['rho'], 0.01)
            material_freqs.append(freq)
        frequencies.append(material_freqs)
    
    # 創建柱狀圖
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(modes))
    width = 0.8 / len(materials)
    
    for i, (material, freqs) in enumerate(zip(materials, frequencies)):
        offset = (i - len(materials)/2 + 0.5) * width
        ax.bar(x + offset, freqs, width, label=material['name'])
    
    # 設置標籤
    ax.set_xlabel('Mode (m,n)')
    ax.set_ylabel('Natural Frequency (Hz)')
    ax.set_title('Natural Frequencies Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f'({m},{n})' for m, n in modes])
    ax.legend()
    
    return fig

if __name__ == "__main__":
    # 定義一些常見材料的參數
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
    
    # 繪製模態形狀比較
    fig_modes = plot_mode_shapes_for_different_materials(materials)
    plt.savefig('material_mode_shapes.png')
    plt.close()
    
    # 繪製頻率比較
    fig_freq = plot_frequency_comparison(materials)
    plt.savefig('material_frequencies.png')
    plt.close()
    
    # 顯示圖形
    plt.show() 