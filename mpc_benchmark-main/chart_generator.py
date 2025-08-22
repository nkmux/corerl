"""
Chart Generator for MPC Analysis System
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

class ChartGenerator:
    """
    Handles all chart generation for the MPC analysis
    """
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
    
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
    
    def _safe_pie_chart(self, ax, values, labels, title):
        """Safely create pie chart with non-negative data"""
        # Filter out negative and NaN values
        valid_data = []
        valid_labels = []
        
        for value, label in zip(values, labels):
            if value > 0 and not np.isnan(value):
                valid_data.append(value)
                valid_labels.append(label)
        
        if valid_data and sum(valid_data) > 0:
            ax.pie(valid_data, labels=valid_labels, autopct='%1.1f%%', startangle=90)
            ax.set_title(title)
        else:
            ax.text(0.5, 0.5, 'No Valid Data\nAvailable', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{title}: N/A')
            ax.axis('off')
    
    def create_comprehensive_charts(self, combo_key, season_results, annual_summary):
        """Create and save comprehensive charts for each combination"""
        chart_dir = f'{self.output_dir}/charts/{combo_key}'
        os.makedirs(chart_dir, exist_ok=True)
        
        # 1. Seasonal Performance Comparison
        self.create_seasonal_comparison_chart(combo_key, season_results, chart_dir)
        
        # 2. Annual Performance Summary
        self.create_annual_summary_chart(combo_key, annual_summary, chart_dir)
        
        # 3. Detailed control charts for each season
        for season_name, results in season_results.items():
            self.create_detailed_season_chart(combo_key, season_name, results, chart_dir)
        
        print(f"Charts saved in: {chart_dir}")

    def create_seasonal_comparison_chart(self, combo_key, season_results, chart_dir):
        """Create seasonal performance comparison chart"""
        if not season_results:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Seasonal Performance Analysis - {combo_key.replace("_", " ")}', fontsize=16, fontweight='bold')
        
        seasons = list(season_results.keys())
        
        # Energy cost comparison
        ax1 = axes[0, 0]
        mpc_costs = []
        baseline_costs = []
        valid_seasons = []
        
        for season in seasons:
            if 'comparison' in season_results[season]:
                comparison = season_results[season]['comparison']
                mpc_cost = comparison.get('mpc_total_cost', 0)
                baseline_cost = comparison.get('baseline_total_cost', 0)
                
                if mpc_cost > 0 and baseline_cost > 0:
                    mpc_costs.append(mpc_cost)
                    baseline_costs.append(baseline_cost)
                    valid_seasons.append(season)
        
        if mpc_costs and baseline_costs:
            x = np.arange(len(valid_seasons))
            width = 0.35
            ax1.bar(x - width/2, mpc_costs, width, label='MPC', color='blue', alpha=0.7)
            ax1.bar(x + width/2, baseline_costs, width, label='Baseline', color='red', alpha=0.7)
            ax1.set_ylabel('Energy Cost ($)')
            ax1.set_title('Seasonal Energy Costs')
            ax1.set_xticks(x)
            ax1.set_xticklabels(valid_seasons)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No Cost Data\nAvailable', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Seasonal Energy Costs: N/A')
        
        # Savings percentage
        ax2 = axes[0, 1]
        savings_pct = []
        valid_seasons_savings = []
        
        for season in seasons:
            if 'comparison' in season_results[season]:
                savings = season_results[season]['comparison'].get('cost_reduction_pct', 0)
                if not np.isnan(savings):
                    savings_pct.append(savings)
                    valid_seasons_savings.append(season)
        
        if savings_pct:
            bars = ax2.bar(valid_seasons_savings, savings_pct, color='green', alpha=0.7)
            ax2.set_ylabel('Cost Reduction (%)')
            ax2.set_title('MPC Savings vs Baseline')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, savings_pct):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No Savings Data\nAvailable', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('MPC Savings vs Baseline: N/A')
        
        # Comfort violations
        ax3 = axes[1, 0]
        mpc_violations = []
        baseline_violations = []
        valid_seasons_viol = []
        
        for season in seasons:
            if 'comparison' in season_results[season]:
                comparison = season_results[season]['comparison']
                mpc_viol = comparison.get('mpc_violation_rate', 0)
                baseline_viol = comparison.get('baseline_violation_rate', 0)
                
                if not np.isnan(mpc_viol) and not np.isnan(baseline_viol):
                    mpc_violations.append(mpc_viol)
                    baseline_violations.append(baseline_viol)
                    valid_seasons_viol.append(season)
        
        if mpc_violations and baseline_violations:
            x = np.arange(len(valid_seasons_viol))
            width = 0.35
            ax3.bar(x - width/2, mpc_violations, width, label='MPC', color='blue', alpha=0.7)
            ax3.bar(x + width/2, baseline_violations, width, label='Baseline', color='red', alpha=0.7)
            ax3.set_ylabel('Violation Rate (%)')
            ax3.set_title('Comfort Violation Rates')
            ax3.set_xticks(x)
            ax3.set_xticklabels(valid_seasons_viol)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No Violation Data\nAvailable', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Comfort Violation Rates: N/A')
        
        # Energy consumption
        ax4 = axes[1, 1]
        mpc_energy = []
        baseline_energy = []
        valid_seasons_energy = []
        
        for season in seasons:
            if 'mpc' in season_results[season] and 'baseline' in season_results[season]:
                mpc_data = season_results[season]['mpc']
                baseline_data = season_results[season]['baseline']
                
                if 'energy_consumption' in mpc_data and 'energy_consumption' in baseline_data:
                    mpc_avg = np.mean(mpc_data['energy_consumption']) if mpc_data['energy_consumption'] else 0
                    baseline_avg = np.mean(baseline_data['energy_consumption']) if baseline_data['energy_consumption'] else 0
                    
                    if mpc_avg > 0 and baseline_avg > 0:
                        mpc_energy.append(mpc_avg)
                        baseline_energy.append(baseline_avg)
                        valid_seasons_energy.append(season)
        
        if mpc_energy and baseline_energy:
            x = np.arange(len(valid_seasons_energy))
            width = 0.35
            ax4.bar(x - width/2, mpc_energy, width, label='MPC', color='blue', alpha=0.7)
            ax4.bar(x + width/2, baseline_energy, width, label='Baseline', color='red', alpha=0.7)
            ax4.set_ylabel('Average Power (kW)')
            ax4.set_title('Average Energy Consumption')
            ax4.set_xticks(x)
            ax4.set_xticklabels(valid_seasons_energy)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No Energy Data\nAvailable', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Average Energy Consumption: N/A')
        
        plt.tight_layout()
        plt.savefig(f'{chart_dir}/seasonal_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_annual_summary_chart(self, combo_key, annual_summary, chart_dir):
        """Create annual performance summary chart"""
        if not annual_summary:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Annual Performance Summary - {combo_key.replace("_", " ")}', fontsize=16, fontweight='bold')
        
        # Key metrics
        metrics = {
            'Total Cost': f"${annual_summary.get('total_annual_cost', 0):.0f}",
            'Avg Violations': f"{annual_summary.get('avg_violation_rate', 0):.1f}%",
            'Avg Energy': f"{annual_summary.get('avg_energy_consumption', 0):.1f} kW",
            'Success Rate': f"{annual_summary.get('avg_success_rate', 0):.1f}%"
        }
        
        # Summary metrics display
        ax1 = axes[0, 0]
        ax1.axis('off')
        y_pos = 0.8
        for metric, value in metrics.items():
            ax1.text(0.1, y_pos, f"{metric}:", fontsize=14, fontweight='bold')
            ax1.text(0.6, y_pos, value, fontsize=14, color='blue')
            y_pos -= 0.15
        ax1.set_title('Key Performance Metrics', fontsize=14, fontweight='bold')
        
        # Annual savings gauge
        ax2 = axes[0, 1]
        savings = annual_summary.get('annual_savings_vs_baseline', 0)
        # Ensure savings is valid and positive for pie chart
        if savings > 0 and not np.isnan(savings):
            remaining = max(0, 100 - savings)  # Ensure non-negative
            self._safe_pie_chart(ax2, [savings, remaining], ['Savings', 'Baseline'], 
                                f'Annual Savings: {savings:.1f}%')
        else:
            # If no valid savings data, show text instead
            ax2.text(0.5, 0.5, 'No Savings Data\nAvailable', 
                    ha='center', va='center', fontsize=14, 
                    transform=ax2.transAxes)
            ax2.set_title('Annual Savings: N/A', fontsize=14, fontweight='bold')
            ax2.axis('off')
        
        # Performance grade
        ax3 = axes[1, 0]
        grade = self.calculate_performance_grade(annual_summary)
        colors = {'A': 'green', 'B': 'orange', 'C': 'red'}
        ax3.text(0.5, 0.5, grade['letter'], fontsize=72, fontweight='bold', 
                ha='center', va='center', color=colors.get(grade['letter'], 'black'))
        ax3.text(0.5, 0.2, f"Score: {grade['score']:.0f}/100", fontsize=16, 
                ha='center', va='center')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        ax3.set_title('Overall Performance Grade', fontsize=14, fontweight='bold')
        
        # Benchmark comparison (radar chart)
        ax4 = axes[1, 1]
        benchmarks = {
            'Energy Cost': min(1.0, annual_summary.get('total_annual_cost', 2000) / 2000),
            'Comfort': max(0, 1 - annual_summary.get('avg_violation_rate', 0) / 100),
            'Reliability': annual_summary.get('avg_success_rate', 0) / 100
        }
        
        categories = list(benchmarks.keys())
        values = list(benchmarks.values())
        
        # Ensure we have valid values for radar chart
        if values and all(not np.isnan(v) for v in values):
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
            values += values[:1]  # Complete the circle
            angles = np.concatenate((angles, [angles[0]]))
            
            ax4.plot(angles, values, 'bo-', linewidth=2, markersize=8)
            ax4.fill(angles, values, alpha=0.25, color='blue')
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(categories)
            ax4.set_ylim(0, 1.2)
            ax4.set_title('Performance Radar', fontsize=14, fontweight='bold')
            ax4.grid(True)
        else:
            ax4.text(0.5, 0.5, 'Insufficient Data\nfor Radar Chart', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Performance Radar: N/A', fontsize=14, fontweight='bold')
            ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{chart_dir}/annual_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

    def calculate_performance_grade(self, annual_summary):
        """Calculate overall performance grade"""
        # Scoring criteria (0-100 scale)
        cost = annual_summary.get('total_annual_cost', 2000)
        violations = annual_summary.get('avg_violation_rate', 100)
        success_rate = annual_summary.get('avg_success_rate', 0)
        
        cost_score = max(0, 100 - (cost / 2000) * 50)  # Lower cost = higher score
        comfort_score = max(0, 100 - violations * 10)  # Lower violations = higher score
        reliability_score = success_rate  # Direct mapping
        
        overall_score = (cost_score + comfort_score + reliability_score) / 3
        
        if overall_score >= 85:
            letter = 'A'
        elif overall_score >= 70:
            letter = 'B'
        else:
            letter = 'C'
            
        return {'score': overall_score, 'letter': letter}

    def create_detailed_season_chart(self, combo_key, season_name, results, chart_dir):
        """Create detailed chart for specific season"""
        if 'mpc' not in results or not results['mpc'].get('time'):
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'{season_name} Detailed Analysis - {combo_key.replace("_", " ")}', fontsize=16, fontweight='bold')
        
        mpc_data = results['mpc']
        baseline_data = results.get('baseline', {})
        time_hours = np.array(mpc_data['time'])
        
        # Temperature control
        ax1 = axes[0, 0]
        if 'indoor_temperature' in mpc_data:
            ax1.plot(time_hours, mpc_data['indoor_temperature'], 'b-', label='MPC', linewidth=2)
            
            if 'indoor_temperature' in baseline_data:
                ax1.plot(time_hours, baseline_data['indoor_temperature'], 'r--', label='Baseline', alpha=0.8)
            
            ax1.axhline(y=21.0, color='k', linestyle=':', alpha=0.5, label='Comfort Bounds')
            ax1.axhline(y=25.0, color='k', linestyle=':', alpha=0.5)
            ax1.set_ylabel('Temperature (°C)')
            ax1.set_title('Temperature Control')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Energy consumption
        ax2 = axes[0, 1]
        if 'energy_consumption' in mpc_data:
            ax2.plot(time_hours, mpc_data['energy_consumption'], 'b-', label='MPC', linewidth=2)
            
            if 'energy_consumption' in baseline_data:
                ax2.plot(time_hours, baseline_data['energy_consumption'], 'r--', label='Baseline', alpha=0.8)
            
            ax2.set_ylabel('Power (kW)')
            ax2.set_title('Energy Consumption')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Control actions (MPC only)
        ax3 = axes[1, 0]
        if 'temperature_setpoint' in mpc_data and 'humidity_setpoint' in mpc_data:
            ax3.plot(time_hours, mpc_data['temperature_setpoint'], 'g-', linewidth=2, label='Temp Setpoint')
            ax3_twin = ax3.twinx()
            ax3_twin.plot(time_hours, mpc_data['humidity_setpoint'], 'orange', linewidth=2, label='Humidity Setpoint')
            ax3.set_ylabel('Temperature (°C)', color='g')
            ax3_twin.set_ylabel('Humidity (%)', color='orange')
            ax3.set_xlabel('Time (hours)')
            ax3.set_title('MPC Control Actions')
            ax3.grid(True, alpha=0.3)
        
        # Cumulative cost comparison
        ax4 = axes[1, 1]

        # Try different possible cost data keys
        mpc_cost_data = None
        baseline_cost_data = None

        # Check for various cost data formats
        for cost_key in ['energy_cost_sar', 'energy_cost', 'cost']:
            if cost_key in mpc_data and mpc_data[cost_key]:
                mpc_cost_data = mpc_data[cost_key]
                break

        for cost_key in ['energy_cost_sar', 'energy_cost', 'cost']:
            if cost_key in baseline_data and baseline_data[cost_key]:
                baseline_cost_data = baseline_data[cost_key]
                break

        if mpc_cost_data:
            mpc_cum_cost = np.cumsum(mpc_cost_data)
            ax4.plot(time_hours, mpc_cum_cost, 'b-', label='MPC', linewidth=2)
            
            if baseline_cost_data:
                baseline_cum_cost = np.cumsum(baseline_cost_data)
                ax4.plot(time_hours, baseline_cum_cost, 'r--', label='Baseline', alpha=0.8)
            
            ax4.set_ylabel('Cumulative Cost')
            ax4.set_xlabel('Time (hours)')
            ax4.set_title('Cumulative Energy Cost')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            # Debug: Show what keys are actually available
            available_keys = list(mpc_data.keys()) if mpc_data else []
            ax4.text(0.5, 0.5, f'Available keys:\n{available_keys[:5]}...', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=8)
            ax4.set_title('Cumulative Energy Cost: Debug Info')
            ax4.set_xlabel('Time (hours)')
            ax4.set_ylabel('Cost')
        
        plt.tight_layout()
        plt.savefig(f'{chart_dir}/{season_name.lower()}_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_performance_matrix(self, results_summary):
        """Create performance matrix across all combinations"""
        buildings = ['Building_1', 'Building_2', 'Building_3']
        weather_locations = ['Abha', 'Jeddah', 'Riyadh']
        
        # Initialize matrices for different metrics
        cost_matrix = np.zeros((len(buildings), len(weather_locations)))
        violation_matrix = np.zeros((len(buildings), len(weather_locations)))
        savings_matrix = np.zeros((len(buildings), len(weather_locations)))
        
        for combo_key, data in results_summary.items():
            building, weather = self._parse_combo_key(combo_key)
            
            # Find indices
            try:
                i = buildings.index(building) if building in buildings else -1
                j = weather_locations.index(weather) if weather in weather_locations else -1
                
                if i >= 0 and j >= 0:
                    annual = data.get('annual', {})
                    cost_matrix[i, j] = annual.get('total_annual_cost', 0)
                    violation_matrix[i, j] = annual.get('avg_violation_rate', 0)
                    savings_matrix[i, j] = annual.get('annual_savings_vs_baseline', 0)
            except (ValueError, IndexError):
                print(f"Warning: Could not map {building}-{weather} to matrix")
                continue
        
        # Create heatmaps
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('MPC Performance Matrix - All Combinations', fontsize=16, fontweight='bold')
        
        # Cost matrix
        sns.heatmap(cost_matrix, annot=True, fmt='.0f', cmap='Reds_r', 
                   xticklabels=weather_locations, yticklabels=buildings, ax=axes[0])
        axes[0].set_title('Annual Energy Cost ($)')
        
        # Violation matrix
        sns.heatmap(violation_matrix, annot=True, fmt='.1f', cmap='Reds', 
                   xticklabels=weather_locations, yticklabels=buildings, ax=axes[1])
        axes[1].set_title('Comfort Violation Rate (%)')
        
        # Savings matrix
        sns.heatmap(savings_matrix, annot=True, fmt='.1f', cmap='Greens', 
                   xticklabels=weather_locations, yticklabels=buildings, ax=axes[2])
        axes[2].set_title('Savings vs Baseline (%)')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/analysis/performance_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()