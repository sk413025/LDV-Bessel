from src.models.material import MaterialProperties
from src.models.modal import ClassicalModalAnalysis
from src.models.modal import BesselModalAnalysis
from src.analysis.vibration import SurfaceVibrationModel
from src.ldv import LaserDopplerVibrometer

def main():
    # 創建材料實例
    material = MaterialProperties.create_cardboard()
    
    # 使用傳統模態分析
    ldv_classical = LaserDopplerVibrometer(
        material=material,
        modal_analyzer=ClassicalModalAnalysis
    )
    print("\n使用傳統模態分析...")
    ldv_classical.plot_comprehensive_analysis()
    
    # 使用Bessel模態分析
    ldv_bessel = LaserDopplerVibrometer(
        material=material,
        modal_analyzer=BesselModalAnalysis
    )
    print("\n使用Bessel模態分析...")
    ldv_bessel.plot_comprehensive_analysis()
    
    # 比較兩種方法結果
    compare_results(ldv_classical, ldv_bessel)

def compare_results(classical, bessel):
    """比較兩種模態分析方法的結果"""
    x, y = 0, 0  # 測量點
    classical_results = classical.analyze_vibration(x, y)
    bessel_results = bessel.analyze_vibration(x, y)
    
    print("\n比較結果:")
    print("傳統模態分析:")
    print(f"最大位移: {classical_results['diagnostics']['displacement_stats']['max']*1e9:.2f} nm")
    print("\nBessel模態分析:")
    print(f"最大位移: {bessel_results['diagnostics']['displacement_stats']['max']*1e9:.2f} nm")

if __name__ == "__main__":
    main()