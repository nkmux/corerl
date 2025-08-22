"""
Statistical Analyzer for MPC Analysis System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class StatisticalAnalyzer:
    """
    Handles statistical analysis and specialized chart generation
    """
    
    def __init__(self):
        pass
    
    def _parse_combo_key(self, combo_key):
        """Safely parse combo_key to extract building and weather"""
        parts = combo_key.split('_')
        if len(parts) >= 2:
            building = '_'.join(parts[:-1])  # Everything except last part
            weather = parts[-1]              # Last part
        else:
            building = parts[0] if parts else 'Unknown'
            weather = 'Unknown'
        return building, weather
    
    def create_comparative_analysis(self, results_summary, output_dir):
        """Create comparative analysis charts"""
        print("Creating comparative analysis...")
        
        # Extract data for analysis
        analysis_data = []
        for combo_key, data in results_summary.items():
            building, weather = self._parse_combo_key(combo_key)
            annual = data['annual']
            
            analysis_data.append({
                'Building': building.replace('_', ' '),
                'Weather': weather,
                'Total_Cost': annual.get('total_annual_cost', 0),
                'Violation_Rate': annual.get('avg_violation_rate', 0),
                'Energy_Consumption': annual.get('avg_energy_consumption', 0),
                'Savings_Percent': annual.get('annual_savings_vs_baseline', 0),
                'Success_Rate': annual.get('avg_success_rate', 0)
            })
        
        df = pd.DataFrame(analysis_data)
        
        if df.empty:
            print("No data available for comparative analysis")
            return
        
        # Create comprehensive comparison charts
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Comprehensive MPC Performance Analysis', fontsize=18, fontweight='bold')
        
        # 1. Cost by building type
        ax1 = axes[0, 0]
        building_costs = df.groupby('Building')['Total_Cost'].mean()
        if not building_costs.empty:
            building_costs.plot(kind='bar', ax=ax1, color='steelblue')
            ax1.set_title('Average Annual Cost by Building Type')
            ax1.set_ylabel('Cost ($)')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
        
        # 2. Cost by weather location
        ax2 = axes[0, 1]
        weather_costs = df.groupby('Weather')['Total_Cost'].mean()
        if not weather_costs.empty:
            weather_costs.plot(kind='bar', ax=ax2, color='darkorange')
            ax2.set_title('Average Annual Cost by Weather Location')
            ax2.set_ylabel('Cost ($)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
        
        # 3. Savings distribution
        ax3 = axes[0, 2]
        savings_data = df['Savings_Percent'].dropna()
        if not savings_data.empty and len(savings_data) > 0:
            ax3.hist(savings_data, bins=min(10, len(savings_data)), color='green', alpha=0.7, edgecolor='black')
            ax3.set_title('Distribution of Savings vs Baseline')
            ax3.set_xlabel('Savings (%)')
            ax3.set_ylabel('Frequency')
            mean_savings = savings_data.mean()
            ax3.axvline(mean_savings, color='red', linestyle='--', 
                       label=f'Mean: {mean_savings:.1f}%')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Comfort performance
        ax4 = axes[1, 0]
        building_violations = df.groupby('Building')['Violation_Rate'].mean()
        if not building_violations.empty:
            bars = ax4.bar(building_violations.index, building_violations.values, color='coral')
            ax4.set_title('Average Comfort Violations by Building')
            ax4.set_ylabel('Violation Rate (%)')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, building_violations.values):
                if not np.isnan(value):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                            f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 5. Energy consumption patterns
        ax5 = axes[1, 1]
        weather_energy = df.groupby('Weather')['Energy_Consumption'].mean()
        if not weather_energy.empty and weather_energy.sum() > 0:
            # Ensure no negative values for pie chart
            weather_energy_positive = weather_energy[weather_energy > 0]
            if not weather_energy_positive.empty:
                ax5.pie(weather_energy_positive.values, labels=weather_energy_positive.index, autopct='%1.1f%%',
                       colors=['lightblue', 'lightgreen', 'lightcoral'])
                ax5.set_title('Energy Consumption Distribution by Weather')
        
        # 6. Correlation analysis
        ax6 = axes[1, 2]
        correlation_data = df[['Total_Cost', 'Violation_Rate', 'Energy_Consumption', 'Savings_Percent']].dropna()
        if not correlation_data.empty and len(correlation_data) > 1:
            correlation_matrix = correlation_data.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                       square=True, ax=ax6, cbar_kws={'shrink': 0.8})
            ax6.set_title('Performance Metrics Correlation')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/analysis/comparative_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed statistical analysis
        self.create_statistical_summary(df, output_dir)

    def create_statistical_summary(self, df, output_dir):
        """Create detailed statistical analysis"""
        print("Generating statistical analysis...")
        
        if df.empty:
            print("No data available for statistical summary")
            return
        
        # Calculate statistics safely
        stats_summary = {}
        
        # Overall statistics
        stats_summary['Overall Statistics'] = {}
        for col, label in [('Total_Cost', 'Mean Annual Cost'), 
                          ('Violation_Rate', 'Mean Violation Rate'),
                          ('Energy_Consumption', 'Mean Energy Consumption'),
                          ('Savings_Percent', 'Mean Savings'),
                          ('Success_Rate', 'Mean Success Rate')]:
            if col in df.columns and not df[col].empty:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if col == 'Total_Cost':
                    stats_summary['Overall Statistics'][label] = f"${mean_val:.0f} ± ${std_val:.0f}"
                elif 'Rate' in col or 'Savings' in col:
                    stats_summary['Overall Statistics'][label] = f"{mean_val:.2f}% ± {std_val:.2f}%"
                else:
                    stats_summary['Overall Statistics'][label] = f"{mean_val:.1f} ± {std_val:.1f} kW"
        
        # Best performers (only if data exists)
        if not df.empty:
            stats_summary['Best Performers'] = {}
            
            # Lowest cost
            if 'Total_Cost' in df.columns and not df['Total_Cost'].empty:
                min_cost_idx = df['Total_Cost'].idxmin()
                if not pd.isna(min_cost_idx):
                    stats_summary['Best Performers']['Lowest Cost'] = df.loc[min_cost_idx, ['Building', 'Weather', 'Total_Cost']].to_dict()
            
            # Highest savings
            if 'Savings_Percent' in df.columns and not df['Savings_Percent'].empty:
                max_savings_idx = df['Savings_Percent'].idxmax()
                if not pd.isna(max_savings_idx):
                    stats_summary['Best Performers']['Highest Savings'] = df.loc[max_savings_idx, ['Building', 'Weather', 'Savings_Percent']].to_dict()
            
            # Lowest violations
            if 'Violation_Rate' in df.columns and not df['Violation_Rate'].empty:
                min_viol_idx = df['Violation_Rate'].idxmin()
                if not pd.isna(min_viol_idx):
                    stats_summary['Best Performers']['Lowest Violations'] = df.loc[min_viol_idx, ['Building', 'Weather', 'Violation_Rate']].to_dict()
        
        # Building and weather rankings
        for group_col, group_name in [('Building', 'Building Ranking'), ('Weather', 'Weather Ranking')]:
            if group_col in df.columns:
                grouped = df.groupby(group_col).agg({
                    'Total_Cost': 'mean',
                    'Violation_Rate': 'mean', 
                    'Savings_Percent': 'mean'
                }).round(2)
                if not grouped.empty:
                    stats_summary[group_name] = grouped.to_dict()
        
        # Save statistics to file
        try:
            with open(f'{output_dir}/analysis/statistical_summary.txt', 'w') as f:
                f.write("COMPREHENSIVE MPC PERFORMANCE ANALYSIS\n")
                f.write("="*60 + "\n\n")
                
                for category, data in stats_summary.items():
                    f.write(f"{category}:\n")
                    f.write("-" * len(category) + "\n")
                    
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, dict):
                                f.write(f"  {key}:\n")
                                for subkey, subvalue in value.items():
                                    f.write(f"    {subkey}: {subvalue}\n")
                            else:
                                f.write(f"  {key}: {value}\n")
                    f.write("\n")
            print("Statistical summary saved successfully")
        except Exception as e:
            print(f"Error saving statistical summary: {e}")

    def create_climate_impact_analysis(self, results_summary, output_dir):
        """Analyze climate-specific impacts on building performance"""
        print("Analyzing climate impacts...")
        
        climate_data = {}
        for combo_key, data in results_summary.items():
            building, weather = self._parse_combo_key(combo_key)
            if weather not in climate_data:
                climate_data[weather] = {
                    'costs': [],
                    'violations': [],
                    'energy': [],
                    'savings': []
                }
            
            annual = data['annual']
            climate_data[weather]['costs'].append(annual.get('total_annual_cost', 0))
            climate_data[weather]['violations'].append(annual.get('avg_violation_rate', 0))
            climate_data[weather]['energy'].append(annual.get('avg_energy_consumption', 0))
            climate_data[weather]['savings'].append(annual.get('annual_savings_vs_baseline', 0))
        
        # Create climate comparison chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Climate Impact Analysis on MPC Performance', fontsize=16, fontweight='bold')
        
        climates = list(climate_data.keys())
        
        # Average costs by climate
        ax1 = axes[0, 0]
        avg_costs = []
        std_costs = []
        valid_climates = []
        
        for climate in climates:
            costs = [c for c in climate_data[climate]['costs'] if c > 0]  # Filter valid costs
            if costs:
                avg_costs.append(np.mean(costs))
                std_costs.append(np.std(costs))
                valid_climates.append(climate)
        
        if avg_costs:
            bars = ax1.bar(valid_climates, avg_costs, yerr=std_costs, capsize=5, 
                          color=['skyblue', 'lightgreen', 'lightcoral'][:len(valid_climates)])
            ax1.set_title('Average Annual Cost by Climate')
            ax1.set_ylabel('Annual Cost ($)')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, cost in zip(bars, avg_costs):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(std_costs)*0.1, 
                        f'${cost:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Comfort violations by climate
        ax2 = axes[0, 1]
        avg_violations = []
        valid_climates_viol = []
        
        for climate in climates:
            violations = [v for v in climate_data[climate]['violations'] if not np.isnan(v)]
            if violations:
                avg_violations.append(np.mean(violations))
                valid_climates_viol.append(climate)
        
        if avg_violations:
            ax2.bar(valid_climates_viol, avg_violations, 
                   color=['orange', 'green', 'red'][:len(valid_climates_viol)], alpha=0.7)
            ax2.set_title('Average Comfort Violations by Climate')
            ax2.set_ylabel('Violation Rate (%)')
            ax2.grid(True, alpha=0.3)
        
        # Energy consumption patterns
        ax3 = axes[1, 0]
        positions = np.arange(len(climates))
        for i, climate in enumerate(climates):
            energy_data = [e for e in climate_data[climate]['energy'] if e > 0]
            if energy_data:  # Check if data exists
                ax3.boxplot(energy_data, positions=[i], widths=0.6)
        ax3.set_xticks(positions)
        ax3.set_xticklabels(climates)
        ax3.set_title('Energy Consumption Distribution by Climate')
        ax3.set_ylabel('Average Power (kW)')
        ax3.grid(True, alpha=0.3)
        
        # Climate efficiency comparison
        ax4 = axes[1, 1]
        avg_savings = []
        valid_climates_sav = []
        
        for climate in climates:
            savings = [s for s in climate_data[climate]['savings'] if not np.isnan(s)]
            if savings:
                avg_savings.append(np.mean(savings))
                valid_climates_sav.append(climate)
        
        if avg_savings:
            colors = ['red' if x < 10 else 'orange' if x < 20 else 'green' for x in avg_savings]
            bars = ax4.bar(valid_climates_sav, avg_savings, color=colors, alpha=0.7)
            ax4.set_title('MPC Efficiency by Climate')
            ax4.set_ylabel('Average Savings vs Baseline (%)')
            ax4.axhline(y=15, color='black', linestyle='--', alpha=0.5, label='Target: 15%')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, savings in zip(bars, avg_savings):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{savings:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/analysis/climate_impact_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_building_performance_analysis(self, results_summary, output_dir):
        """Analyze building-specific performance characteristics"""
        print("Analyzing building performance characteristics...")
        
        building_data = {}
        for combo_key, data in results_summary.items():
            building, weather = self._parse_combo_key(combo_key)
            if building not in building_data:
                building_data[building] = {
                    'costs': [],
                    'violations': [],
                    'energy': [],
                    'savings': [],
                    'climates': []
                }
            
            annual = data['annual']
            building_data[building]['costs'].append(annual.get('total_annual_cost', 0))
            building_data[building]['violations'].append(annual.get('avg_violation_rate', 0))
            building_data[building]['energy'].append(annual.get('avg_energy_consumption', 0))
            building_data[building]['savings'].append(annual.get('annual_savings_vs_baseline', 0))
            building_data[building]['climates'].append(weather)
        
        # Create building comparison chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Building Type Performance Analysis', fontsize=16, fontweight='bold')
        
        buildings = list(building_data.keys())
        
        # Cost efficiency ranking
        ax1 = axes[0, 0]
        valid_buildings = []
        avg_costs = []
        
        for building in buildings:
            costs = [c for c in building_data[building]['costs'] if c > 0]
            if costs:
                valid_buildings.append(building)
                avg_costs.append(np.mean(costs))
        
        if avg_costs:
            sorted_data = sorted(zip(valid_buildings, avg_costs), key=lambda x: x[1])
            building_names = [x[0].replace('_', ' ') for x, _ in sorted_data]
            costs = [cost for _, cost in sorted_data]
            
            colors = ['gold', 'silver', '#CD7F32'][:len(building_names)]
            bars = ax1.barh(building_names, costs, color=colors)
            ax1.set_title('Building Cost Efficiency Ranking')
            ax1.set_xlabel('Average Annual Cost ($)')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, cost in zip(bars, costs):
                ax1.text(bar.get_width() + max(costs)*0.01, bar.get_y() + bar.get_height()/2, 
                        f'${cost:.0f}', va='center', fontweight='bold')
        
        # Comfort performance
        ax2 = axes[0, 1]
        valid_buildings_viol = []
        avg_violations = []
        
        for building in buildings:
            violations = [v for v in building_data[building]['violations'] if not np.isnan(v)]
            if violations:
                valid_buildings_viol.append(building)
                avg_violations.append(np.mean(violations))
        
        if avg_violations:
            colors = ['green' if x < 2 else 'orange' if x < 5 else 'red' for x in avg_violations]
            bars = ax2.bar([b.replace('_', ' ') for b in valid_buildings_viol], avg_violations, color=colors, alpha=0.7)
            ax2.set_title('Comfort Performance by Building')
            ax2.set_ylabel('Average Violation Rate (%)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
        
        # Performance consistency (coefficient of variation)
        ax3 = axes[1, 0]
        cv_costs = []
        valid_buildings_cv = []
        
        for building in buildings:
            costs = [c for c in building_data[building]['costs'] if c > 0]
            if costs and len(costs) > 1:
                cv = np.std(costs) / np.mean(costs)
                cv_costs.append(cv)
                valid_buildings_cv.append(building)
        
        if cv_costs:
            bars = ax3.bar([b.replace('_', ' ') for b in valid_buildings_cv], cv_costs, color='purple', alpha=0.7)
            ax3.set_title('Performance Consistency (Lower = More Consistent)')
            ax3.set_ylabel('Coefficient of Variation')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # Building adaptability (savings variation across climates)
        ax4 = axes[1, 1]
        savings_ranges = []
        valid_buildings_adapt = []
        
        for building in buildings:
            savings = [s for s in building_data[building]['savings'] if not np.isnan(s)]
            if savings and len(savings) > 1:
                savings_range = max(savings) - min(savings)
                savings_ranges.append(savings_range)
                valid_buildings_adapt.append(building)
        
        if savings_ranges:
            bars = ax4.bar([b.replace('_', ' ') for b in valid_buildings_adapt], savings_ranges, color='teal', alpha=0.7)
            ax4.set_title('Climate Adaptability (Higher = More Adaptive)')
            ax4.set_ylabel('Savings Range Across Climates (%)')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/analysis/building_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_seasonal_trends_analysis(self, results_summary, output_dir):
        """Analyze seasonal performance trends"""
        print("Analyzing seasonal trends...")
        
        seasonal_aggregated = {
            'Winter': {'costs': [], 'violations': [], 'energy': []},
            'Spring': {'costs': [], 'violations': [], 'energy': []},
            'Summer': {'costs': [], 'violations': [], 'energy': []},
            'Autumn': {'costs': [], 'violations': [], 'energy': []}
        }
        
        # Aggregate seasonal data across all combinations
        for combo_key, data in results_summary.items():
            for season, metrics in data['seasons'].items():
                if season in seasonal_aggregated:
                    cost = metrics.get('total_energy_cost', 0)
                    violation = metrics.get('violation_rate', 0)
                    energy = metrics.get('avg_energy_consumption', 0)
                    
                    if cost > 0:  # Only add valid data
                        seasonal_aggregated[season]['costs'].append(cost)
                    if not np.isnan(violation):
                        seasonal_aggregated[season]['violations'].append(violation)
                    if energy > 0:
                        seasonal_aggregated[season]['energy'].append(energy)
        
        # Create seasonal trends chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Seasonal Performance Trends Analysis', fontsize=16, fontweight='bold')
        
        seasons = list(seasonal_aggregated.keys())
        
        # Seasonal cost trends
        ax1 = axes[0, 0]
        avg_costs = []
        std_costs = []
        valid_seasons = []
        
        for season in seasons:
            costs = seasonal_aggregated[season]['costs']
            if costs:
                avg_costs.append(np.mean(costs))
                std_costs.append(np.std(costs))
                valid_seasons.append(season)
        
        if avg_costs:
            ax1.errorbar(valid_seasons, avg_costs, yerr=std_costs, marker='o', linewidth=2, 
                        markersize=8, capsize=5, color='blue')
            ax1.set_title('Seasonal Energy Cost Trends')
            ax1.set_ylabel('Average Cost ($)')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
        
        # Seasonal violation patterns
        ax2 = axes[0, 1]
        avg_violations = []
        valid_seasons_viol = []
        
        for season in seasons:
            violations = seasonal_aggregated[season]['violations']
            if violations:
                avg_violations.append(np.mean(violations))
                valid_seasons_viol.append(season)
        
        if avg_violations:
            bars = ax2.bar(valid_seasons_viol, avg_violations, 
                          color=['lightblue', 'lightgreen', 'orange', 'brown'][:len(valid_seasons_viol)], alpha=0.7)
            ax2.set_title('Seasonal Comfort Challenge')
            ax2.set_ylabel('Average Violation Rate (%)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Identify most challenging season
            if avg_violations:
                max_idx = np.argmax(avg_violations)
                max_violations_season = valid_seasons_viol[max_idx]
                ax2.text(0.5, 0.95, f'Most Challenging: {max_violations_season}', 
                        transform=ax2.transAxes, ha='center', va='top', 
                        bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
        
        # Energy consumption patterns
        ax3 = axes[1, 0]
        valid_positions = []
        for i, season in enumerate(seasons):
            energy_data = seasonal_aggregated[season]['energy']
            if energy_data:  # Check if data exists
                ax3.violinplot([energy_data], positions=[i], widths=0.7)
                valid_positions.append(i)
        
        if valid_positions:
            ax3.set_xticks(valid_positions)
            ax3.set_xticklabels([seasons[i] for i in valid_positions], rotation=45)
            ax3.set_title('Seasonal Energy Consumption Distribution')
            ax3.set_ylabel('Energy Consumption (kW)')
            ax3.grid(True, alpha=0.3)
        
        # Seasonal efficiency index
        ax4 = axes[1, 1]
        efficiency_index = []
        valid_seasons_eff = []
        
        for season in seasons:
            costs = seasonal_aggregated[season]['costs']
            energy = seasonal_aggregated[season]['energy']
            if costs and energy:
                avg_cost_per_kwh = np.mean(costs) / (np.mean(energy) * 24 * 14) if np.mean(energy) > 0 else 0
                if avg_cost_per_kwh > 0:
                    efficiency_index.append(1 / avg_cost_per_kwh)
                    valid_seasons_eff.append(season)
        
        if efficiency_index:
            bars = ax4.bar(valid_seasons_eff, efficiency_index, color='green', alpha=0.7)
            ax4.set_title('Seasonal Efficiency Index (Higher = Better)')
            ax4.set_ylabel('Efficiency Index')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/analysis/seasonal_trends_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_rl_benchmark_recommendations(self, results_summary, output_dir):
        """Create specific recommendations for RL benchmark"""
        print("Creating RL benchmark recommendations...")
        
        # Analyze performance distribution to set targets
        all_costs = []
        all_violations = []
        all_savings = []
        scenario_difficulty = {}
        
        for combo_key, data in results_summary.items():
            annual = data['annual']
            cost = annual.get('total_annual_cost', 0)
            violations = annual.get('avg_violation_rate', 0)
            savings = annual.get('annual_savings_vs_baseline', 0)
            
            if cost > 0:  # Only include valid data
                all_costs.append(cost)
            if not np.isnan(violations):
                all_violations.append(violations)
            if not np.isnan(savings):
                all_savings.append(savings)
            
            # Calculate difficulty score (higher cost + higher violations = more difficult)
            if cost > 0 and not np.isnan(violations):
                difficulty_score = (cost / 2000) + (violations * 10)  # Normalized difficulty
                scenario_difficulty[combo_key] = difficulty_score
        
        if not all_costs or not scenario_difficulty:
            print("Insufficient data for RL benchmark recommendations")
            return
        
        # Sort scenarios by difficulty
        sorted_scenarios = sorted(scenario_difficulty.items(), key=lambda x: x[1])
        
        # Create benchmark recommendations
        benchmark_data = {
            'performance_targets': {
                'excellent': {
                    'cost_threshold': np.percentile(all_costs, 25),  # Top 25%
                    'violation_threshold': np.percentile(all_violations, 25) if all_violations else 0,
                    'savings_threshold': np.percentile(all_savings, 75) if all_savings else 0
                },
                'good': {
                    'cost_threshold': np.percentile(all_costs, 50),  # Median
                    'violation_threshold': np.percentile(all_violations, 50) if all_violations else 0,
                    'savings_threshold': np.percentile(all_savings, 50) if all_savings else 0
                },
                'minimum': {
                    'cost_threshold': np.percentile(all_costs, 75),  # Bottom 25%
                    'violation_threshold': np.percentile(all_violations, 75) if all_violations else 0,
                    'savings_threshold': np.percentile(all_savings, 25) if all_savings else 0
                }
            },
            'training_recommendations': {
                'easy_scenarios': sorted_scenarios[:3] if len(sorted_scenarios) >= 3 else sorted_scenarios,
                'medium_scenarios': sorted_scenarios[3:6] if len(sorted_scenarios) >= 6 else [],
                'hard_scenarios': sorted_scenarios[6:] if len(sorted_scenarios) > 6 else [],
            }
        }
        
        # Create visual benchmark recommendations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('RL Benchmark Recommendations', fontsize=16, fontweight='bold')
        
        # Performance targets visualization
        ax1 = axes[0, 0]
        targets = benchmark_data['performance_targets']
        categories = ['Excellent', 'Good', 'Minimum']
        cost_targets = [targets[cat.lower()]['cost_threshold'] for cat in categories]
        
        bars = ax1.bar(categories, cost_targets, color=['gold', 'silver', '#CD7F32'], alpha=0.7)
        ax1.set_title('RL Cost Performance Targets')
        ax1.set_ylabel('Annual Cost Threshold ($)')
        ax1.grid(True, alpha=0.3)
        
        for bar, cost in zip(bars, cost_targets):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cost_targets)*0.02, 
                    f'${cost:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Scenario difficulty distribution
        ax2 = axes[0, 1]
        if scenario_difficulty:
            difficulties = list(scenario_difficulty.values())
            scenarios = [key.replace('_', '\n') for key in scenario_difficulty.keys()]
            
            colors = ['green' if d < np.percentile(difficulties, 33) else 
                     'orange' if d < np.percentile(difficulties, 67) else 'red' for d in difficulties]
            
            bars = ax2.bar(range(len(scenarios)), difficulties, color=colors, alpha=0.7)
            ax2.set_title('Scenario Difficulty Ranking')
            ax2.set_ylabel('Difficulty Score')
            ax2.set_xticks(range(len(scenarios)))
            ax2.set_xticklabels(scenarios, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
        
        # Training data split recommendation
        ax3 = axes[1, 0]
        split_data = {
            'Training (70%)': 0.7,
            'Validation (15%)': 0.15,
            'Testing (15%)': 0.15
        }
        
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        wedges, texts, autotexts = ax3.pie(split_data.values(), labels=split_data.keys(), 
                                          autopct='%1.0f%%', colors=colors, startangle=90)
        ax3.set_title('Recommended Data Split for RL Training')
        
        # Performance comparison framework
        ax4 = axes[1, 1]
        metrics = ['Cost\nReduction', 'Comfort\nMaintenance', 'Energy\nEfficiency', 
                  'Convergence\nSpeed', 'Stability\nIndex']
        mpc_scores = [100, 95, 90, 85, 100]  # MPC baseline scores
        rl_targets = [105, 98, 95, 70, 85]   # RL target scores
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax4.bar(x - width/2, mpc_scores, width, label='MPC Baseline', color='blue', alpha=0.7)
        ax4.bar(x + width/2, rl_targets, width, label='RL Target', color='red', alpha=0.7)
        
        ax4.set_title('Performance Comparison Framework')
        ax4.set_ylabel('Performance Score')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/analysis/rl_benchmark_recommendations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed recommendations to file
        try:
            with open(f'{output_dir}/rl_benchmark_guide.txt', 'w') as f:
                f.write("REINFORCEMENT LEARNING BENCHMARK GUIDE\n")
                f.write("="*50 + "\n\n")
                
                f.write("PERFORMANCE TARGETS:\n")
                f.write("-"*20 + "\n")
                for level, targets in benchmark_data['performance_targets'].items():
                    f.write(f"{level.upper()} Performance:\n")
                    f.write(f"  - Annual Cost: < ${targets['cost_threshold']:.0f}\n")
                    f.write(f"  - Violation Rate: < {targets['violation_threshold']:.1f}%\n")
                    f.write(f"  - Savings vs Baseline: > {targets['savings_threshold']:.1f}%\n\n")
                
                f.write("TRAINING STRATEGY:\n")
                f.write("-"*18 + "\n")
                
                # Easy scenarios
                easy_scenarios = benchmark_data['training_recommendations']['easy_scenarios']
                if easy_scenarios:
                    f.write("Easy Scenarios (Start Here):\n")
                    for scenario, difficulty in easy_scenarios:
                        building, weather = self._parse_combo_key(scenario)
                        f.write(f"  - {building} + {weather} Weather (Difficulty: {difficulty:.2f})\n")
                
                # Medium scenarios
                medium_scenarios = benchmark_data['training_recommendations']['medium_scenarios']
                if medium_scenarios:
                    f.write("\nMedium Scenarios (Intermediate Training):\n")
                    for scenario, difficulty in medium_scenarios:
                        building, weather = self._parse_combo_key(scenario)
                        f.write(f"  - {building} + {weather} Weather (Difficulty: {difficulty:.2f})\n")
                
                # Hard scenarios
                hard_scenarios = benchmark_data['training_recommendations']['hard_scenarios']
                if hard_scenarios:
                    f.write("\nHard Scenarios (Advanced Testing):\n")
                    for scenario, difficulty in hard_scenarios:
                        building, weather = self._parse_combo_key(scenario)
                        f.write(f"  - {building} + {weather} Weather (Difficulty: {difficulty:.2f})\n")
                
                f.write(f"\nRECOMMENDED APPROACH:\n")
                f.write("-"*20 + "\n")
                f.write("1. Start training on easy scenarios to establish basic competency\n")
                f.write("2. Gradually introduce medium and hard scenarios\n")
                f.write("3. Use curriculum learning with increasing difficulty\n")
                f.write("4. Validate on all scenarios to ensure generalization\n")
                f.write("5. Compare against MPC performance on identical test periods\n")
                
                print("RL benchmark guide saved successfully")
        except Exception as e:
            print(f"Error saving RL benchmark guide: {e}")