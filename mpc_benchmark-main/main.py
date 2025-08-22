"""
Main entry point for Comprehensive MPC Analysis System
"""

import warnings
import sys
import os
warnings.filterwarnings('ignore')

def check_data_files():
    """Check if required CSV files exist"""
    required_files = [
        './data/building_1.csv',
        './data/building_2.csv', 
        './data/building_3.csv',
        './data/weather_abha.csv',
        './data/weather_jeddah.csv',
        './data/weather_riyadh.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ MISSING DATA FILES:")
        for file_path in missing_files:
            print(f"   {file_path}")
        print("\nPlease ensure all CSV files are in the ./data/ directory")
        return False
    
    print("✅ All required CSV files found")
    return True

def check_dependencies():
    """Check if required Python packages are available"""
    required_packages = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('matplotlib.pyplot', 'plt'),
        ('seaborn', 'sns'),
        ('scipy.optimize', None)
    ]
    
    missing_packages = []
    for package, alias in required_packages:
        try:
            if alias:
                exec(f"import {package} as {alias}")
            else:
                exec(f"import {package}")
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ MISSING PYTHON PACKAGES:")
        for package in missing_packages:
            print(f"   {package}")
        print("\nPlease install missing packages with:")
        print("pip install pandas numpy matplotlib seaborn scipy")
        return False
    
    print("✅ All required packages available")
    return True


def main():
    """Main execution function"""
    print("COMPREHENSIVE MPC ANALYSIS SYSTEM - SAUDI ARABIA")
    print("="*80)
    print("This will analyze MPC performance across:")
    print("• 3 Building Types × 3 Saudi Weather Locations = 9 Combinations")
    print("• 4 Seasons per combination = 36 total simulations")
    print("• Saudi Electricity Regulatory Authority pricing structure")
    print("• Results in Saudi Riyals (SAR) with USD equivalents")
    print("• Complete performance analysis and benchmarking")
    print("• RL benchmark recommendations")
    print("="*80)
    
    # Pre-flight checks
    print("\n🔧 PERFORMING PRE-FLIGHT CHECKS...")
    print("-" * 50)
    
    # Check 1: Dependencies
    if not check_dependencies():
        return
    
    # Check 2: Data files
    if not check_data_files():
        return
    
    # Proceed with analysis
    print("\n🚀 STARTING COMPREHENSIVE ANALYSIS...")
    print("="*80)
    
    try:
        from mpc_analysis_core import ComprehensiveMPCAnalysis
        
        # Initialize analysis system with Saudi-specific output directory
        analyzer = ComprehensiveMPCAnalysis(output_dir='saudi_mpc_results')
        
        # Run complete analysis
        print("Running comprehensive analysis...")
        analyzer.run_comprehensive_analysis()
        
        # Generate executive summary
        print("Generating executive summary...")
        analyzer.generate_executive_summary()
        
        print("\n" + "="*80)
        print("🎉 COMPREHENSIVE ANALYSIS COMPLETE! 🎉")
        print("="*80)
        print("📊 Generated Analysis:")
        print("  ✓ Individual simulations completed")
        print("  ✓ Building-weather combination analyses")
        print("  ✓ Performance matrices and correlations")
        print("  ✓ Climate impact analysis")
        print("  ✓ Building performance characteristics")
        print("  ✓ Seasonal trends analysis")
        print("  ✓ RL benchmark recommendations")
        print("  ✓ Executive summary with key findings")
        print("\n📁 Results Structure:")
        print("  saudi_mpc_results/")
        print("    📁 charts/          - Individual combination charts")
        print("      📁 Building_1_Abha/")
        print("        📄 seasonal_comparison.png")
        print("        📄 annual_summary.png")
        print("        📄 winter_detailed.png")
        print("        📄 spring_detailed.png")
        print("        📄 summer_detailed.png")
        print("        📄 autumn_detailed.png")
        print("      📁 Building_1_Jeddah/")
        print("      📁 Building_1_Riyadh/")
        print("      📁 ... (9 total)")
        print("    📁 data/            - CSV and JSON data files")
        print("      📄 annual_summary.csv")
        print("      📄 seasonal_breakdown.csv")
        print("      📄 detailed_results.json")
        print("    📁 analysis/        - Comparative analysis charts")
        print("      📄 performance_matrix.png")
        print("      📄 comparative_analysis.png")
        print("      📄 climate_impact_analysis.png")
        print("      📄 building_performance_analysis.png")
        print("      📄 seasonal_trends_analysis.png")
        print("      📄 rl_benchmark_recommendations.png")
        print("      📄 statistical_summary.txt")
        print("    📄 executive_summary.txt")
        print("    📄 rl_benchmark_guide.txt")
        print("="*80)
        print("🚀 Ready for RL comparison!")
        print("📈 Total charts generated: 60+")
        print("📊 Total data files: 10+")
        print("📋 Complete Saudi Arabia benchmark established!")
        print("💰 All costs in Saudi Riyals (SAR) with USD equivalents")
        
    except ImportError as e:
        print(f"❌ Import Error: {str(e)}")
        print("Please check that all Python files are in the same directory:")
        print("- building_mpc.py")
        print("- mpc_analysis_core.py") 
        print("- data_manager.py")
        print("- chart_generator.py")
        print("- statistical_analyzer.py")
        print("- report_generator.py")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Verify CSV files are in ./data/ directory with correct names")
        print("2. Check that all Python files are present and error-free")
        print("3. Ensure all required packages are installed")
        print("4. Try running test_data_loading.py first to debug")
        
        print("\nDetailed error information:")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()