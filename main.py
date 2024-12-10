from src.models.material import MaterialProperties, MaterialFactory
from src.models.modal import ClassicalModalAnalysis
from src.models.modal import BesselModalAnalysis
from src.analysis.vibration import SurfaceVibrationModel
from src.ldv import LaserDopplerVibrometer

def main():
    # 使用材料工廠創建不同材料
    cardboard = MaterialFactory.create('cardboard')
    aluminum = MaterialFactory.create('metal', metal_type='aluminum')
    steel = MaterialFactory.create('metal', metal_type='steel')
    
    # 測試不同材料
    test_materials = [
        ('紙箱', cardboard),
        # ('鋁板', aluminum),
        # ('鋼板', steel)
    ]
    
    for material_name, material in test_materials:
        print(f"\n測試 {material_name} 振動響應...")
        ldv = LaserDopplerVibrometer(
            material=material,
            modal_analyzer=ClassicalModalAnalysis,
            analysis_type="classical"
        )
        
        ldv.setup_measurement({
            'length': 0.1,
            'width': 0.1,
            'thickness': 0.001
        })
        
        ldv.plot_comprehensive_analysis()

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