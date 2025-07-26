"""
Mixed Effects Model Statistical Testing for EchoMAE Results
混合効果モデルを用いた統計解析
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from pathlib import Path

results_dir = # path to the dir for the results output


class MixedEffectsEchoMAEAnalyzer:
    """Mixed Effects Echo MAE Statistical Analyzer"""
    
    def __init__(self, csv_file: str):
        self.df = pd.read_csv(csv_file)
        self.all_tasks = set(self.df['task'].unique())
        self.clinical_tasks = self._identify_clinical_tasks()
        self.demographic_tasks = self._identify_demographic_tasks()
        self._prepare_data()
    
    def _identify_clinical_tasks(self):
        """臨床タスクを自動識別"""
        demographic_keywords = ['sex_encoded', 'race_encoded', 'bias_sex', 'bias_race']
        clinical_tasks = []
        
        for task in self.all_tasks:
            if not any(keyword in task for keyword in demographic_keywords):
                clinical_tasks.append(task)
        
        return sorted(clinical_tasks)
    
    def _identify_demographic_tasks(self):
        """人口統計学的タスクを自動識別"""
        demographic_tasks = []
        possible_names = ['sex_encoded', 'race_encoded', 'bias_sex', 'bias_race']
        
        for task_name in possible_names:
            if task_name in self.all_tasks:
                demographic_tasks.append(task_name)
        
        return demographic_tasks
    
    def _prepare_data(self):
        """データの前処理 - 独立repeat-fold組み合わせを作成"""
        self.df['auc'] = pd.to_numeric(self.df['auc'], errors='coerce')
        self.processed_df = self.df.dropna(subset=['auc']).copy()
        
        if 'fold' not in self.processed_df.columns:
            self.processed_df = self.processed_df.sort_values(['task', 'adversarial_weight', 'subgroup', 'repeat_idx'])
            self.processed_df['fold'] = self.processed_df.groupby(['task', 'adversarial_weight', 'subgroup', 'repeat_idx']).cumcount() % 5

        self.processed_df = self.processed_df.sort_values(['repeat_idx', 'fold'])
        unique_combinations = self.processed_df[['repeat_idx', 'fold']].drop_duplicates().sort_values(['repeat_idx', 'fold'])
        combination_mapping = {
            (row['repeat_idx'], row['fold']): idx 
            for idx, (_, row) in enumerate(unique_combinations.iterrows())
        }
        
        self.processed_df['repeat_fold_id'] = self.processed_df.apply(
            lambda row: combination_mapping[(row['repeat_idx'], row['fold'])], axis=1
        )
        
        self.processed_df['task'] = self.processed_df['task'].astype('category')
        self.processed_df['repeat_fold_id'] = self.processed_df['repeat_fold_id'].astype('category')
        self.processed_df['adversarial_weight'] = self.processed_df['adversarial_weight'].astype(float)
        
        self.processed_df['model_type'] = self.processed_df['adversarial_weight'].apply(
            lambda x: 'Foundation_VE' if x == -1.0 else 'MVE_VE'
        )
        self.processed_df['model_type'] = self.processed_df['model_type'].astype('category')
    
    def perform_performance_analysis(self, alpha: float = 0.05):
        """Performance analysis using mixed effects model"""
        clinical_data = self.processed_df[
            (self.processed_df['subgroup'] == 'overall') &
            (self.processed_df['task'].isin(self.clinical_tasks))
        ].copy()
        
        if len(clinical_data) == 0:
            return {}

        mae_weights = sorted([w for w in clinical_data['adversarial_weight'].unique() if w >= 0])
        foundation_weight = -1.0
        
        performance_results = {}
        p_values = []
        comparison_names = []
        
        for mae_weight in mae_weights:
            comparison_data = clinical_data[
                clinical_data['adversarial_weight'].isin([foundation_weight, mae_weight])
            ].copy()
            
            if len(comparison_data) == 0:
                continue
            
            try:
                # mixed effect model: AUC ~ model_type + (1|repeat_fold_id) + vc(task)
                formula = "auc ~ model_type"
                
                model = smf.mixedlm(
                    formula,
                    comparison_data,
                    groups=comparison_data['repeat_fold_id'],
                    vc_formula={"task": "0 + C(task)"}
                )
                
                result = model.fit(reml=True)

                coef = result.params.get('model_type[T.MVE_VE]', 0)
                p_value = result.pvalues.get('model_type[T.MVE_VE]', 1.0)
                conf_int = result.conf_int().loc['model_type[T.MVE_VE]'] if 'model_type[T.MVE_VE]' in result.conf_int().index else [0, 0]

                foundation_mean = comparison_data[comparison_data['model_type'] == 'Foundation_VE']['auc'].mean()
                mae_mean = comparison_data[comparison_data['model_type'] == 'MVE_VE']['auc'].mean()
                
                performance_results[f'Weight_{mae_weight}'] = {
                    'coefficient': coef,
                    'p_value': p_value,
                    'conf_int_lower': conf_int[0],
                    'conf_int_upper': conf_int[1],
                    'foundation_mean': foundation_mean,
                    'mae_mean': mae_mean,
                    'improvement_pct': (coef / foundation_mean) * 100 if foundation_mean > 0 else 0
                }
                
                p_values.append(p_value)
                comparison_names.append(f'Weight_{mae_weight}')
                
            except Exception:
                continue
        
        # Multiple comparison correction (Holm method)
        if p_values:
            rejected, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method='holm')
            
            for i, comp_name in enumerate(comparison_names):
                performance_results[comp_name]['p_value_corrected'] = p_corrected[i]
                performance_results[comp_name]['significant'] = rejected[i]
        
        return performance_results
    
    def perform_fairness_analysis(self, alpha: float = 0.05):
        """Fairness analysis using mixed effects model"""
        demographic_tasks = self.get_demographic_tasks()
        if not demographic_tasks:
            return {}
        
        fairness_results = {}
        
        for task in demographic_tasks:
            task_data = self.processed_df[
                (self.processed_df['subgroup'] == 'overall') &
                (self.processed_df['task'] == task)
            ].copy()
            
            if len(task_data) == 0:
                continue

            if 'sex' in task.lower():
                attribute = 'sex'
            elif 'race' in task.lower():
                attribute = 'race'
            else:
                attribute = task
            
            mae_weights = sorted([w for w in task_data['adversarial_weight'].unique() if w >= 0])
            foundation_weight = -1.0
            
            attr_results = {}
            p_values = []
            comparison_names = []
            
            for mae_weight in mae_weights:
                comparison_data = task_data[
                    task_data['adversarial_weight'].isin([foundation_weight, mae_weight])
                ].copy()
                
                if len(comparison_data) == 0:
                    continue
                
                try:
                    # mixed effect model: AUC ~ model_type + (1|repeat_fold_id)
                    formula = "auc ~ model_type"
                    
                    model = smf.mixedlm(
                        formula,
                        comparison_data,
                        groups=comparison_data['repeat_fold_id']
                    )
                    
                    result = model.fit(reml=True)

                    coef = result.params.get('model_type[T.MVE_VE]', 0)
                    p_value = result.pvalues.get('model_type[T.MVE_VE]', 1.0)
                    conf_int = result.conf_int().loc['model_type[T.MVE_VE]'] if 'model_type[T.MVE_VE]' in result.conf_int().index else [0, 0]

                    foundation_mean = comparison_data[comparison_data['model_type'] == 'Foundation_VE']['auc'].mean()
                    mae_mean = comparison_data[comparison_data['model_type'] == 'MVE_VE']['auc'].mean()

                    fairness_improvement = -(coef / foundation_mean) * 100 if foundation_mean > 0 else 0
                    
                    attr_results[f'Weight_{mae_weight}'] = {
                        'coefficient': coef,
                        'p_value': p_value,
                        'conf_int_lower': conf_int[0],
                        'conf_int_upper': conf_int[1],
                        'foundation_mean': foundation_mean,
                        'mae_mean': mae_mean,
                        'fairness_improvement': fairness_improvement
                    }
                    
                    p_values.append(p_value)
                    comparison_names.append(f'Weight_{mae_weight}')
                    
                except Exception:
                    continue

            if p_values:
                rejected, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method='holm')
                
                for i, comp_name in enumerate(comparison_names):
                    attr_results[comp_name]['p_value_corrected'] = p_corrected[i]
                    attr_results[comp_name]['significant'] = rejected[i]
            
            if attr_results:
                fairness_results[attribute] = attr_results
        
        return fairness_results
    
    def get_clinical_tasks(self):
        return self.clinical_tasks
    
    def get_demographic_tasks(self):
        return self.demographic_tasks
    
    def create_figure(self, performance_results: dict, fairness_results: dict, save_path: str = None):
        """Figure 2"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        
        # Performance plot
        clinical_data = self.processed_df[
            (self.processed_df['subgroup'] == 'overall') &
            (self.processed_df['task'].isin(self.clinical_tasks))
        ].copy()
        
        if len(clinical_data) > 0:
            weight_stats = clinical_data.groupby('adversarial_weight')['auc'].agg(['mean', 'sem']).reset_index()
            weight_stats = weight_stats.sort_values('adversarial_weight')
            
            foundation_data = weight_stats[weight_stats['adversarial_weight'] == -1.0]
            mae_data = weight_stats[weight_stats['adversarial_weight'] >= 0]
            
            color = '#2E86C1'
            marker = 'o'
            
            # Foundation VE
            if len(foundation_data) > 0:
                foundation_mean = foundation_data['mean'].iloc[0]
                foundation_sem = foundation_data['sem'].iloc[0]
                ax1.errorbar([-0.15], [foundation_mean], 
                            yerr=[foundation_sem], marker=marker, markersize=8, capsize=5,
                            color=color, linewidth=2)
            
            # MAE VE
            if len(mae_data) > 0:
                mae_weights = mae_data['adversarial_weight'].values
                mae_means = mae_data['mean'].values
                mae_sems = mae_data['sem'].values
                
                ax1.errorbar(mae_weights, mae_means, yerr=mae_sems,
                            marker=marker, markersize=8, capsize=5,
                            color=color, linewidth=2)
                
                if performance_results:
                    for i, weight in enumerate(mae_weights):
                        comp_name = f'Weight_{weight}'
                        if (comp_name in performance_results and 
                            performance_results[comp_name].get('significant', False)):
                            y_pos = mae_means[i] + mae_sems[i] + 0.002
                            ax1.text(weight, y_pos, "*", ha='center', va='bottom', 
                                    fontsize=16, fontweight='bold')
                
                # X-axis
                x_ticks = [-0.15] + list(mae_weights)
                x_labels = ['Foundation\nVE'] + [str(w) for w in mae_weights]
                ax1.set_xticks(x_ticks)
                ax1.set_xticklabels(x_labels, fontsize=12)
                ax1.tick_params(axis='y', labelsize=12)
                
                ax1.axvline(x=-0.075, color='gray', linestyle='--', alpha=0.5)
        else:
            ax1.text(0.5, 0.5, 'No performance data available', 
                    ha='center', va='center', transform=ax1.transAxes,
                    fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        ax1.set_ylabel('Mean AUC for clinical tasks', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Fairness plot
        if fairness_results:
            colors = ['#E74C3C', '#8E44AD']
            markers = ['s', '^']
            
            for i, (attribute, attr_results) in enumerate(fairness_results.items()):
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]

                demographic_task = None
                for task in self.demographic_tasks:
                    if attribute.lower() in task.lower():
                        demographic_task = task
                        break
                
                if demographic_task:
                    task_data = self.processed_df[
                        (self.processed_df['subgroup'] == 'overall') &
                        (self.processed_df['task'] == demographic_task)
                    ].copy()
                    
                    if len(task_data) > 0:
                        weight_stats = task_data.groupby('adversarial_weight')['auc'].agg(['mean', 'sem']).reset_index()
                        weight_stats = weight_stats.sort_values('adversarial_weight')
                        
                        foundation_data = weight_stats[weight_stats['adversarial_weight'] == -1.0]
                        mae_data = weight_stats[weight_stats['adversarial_weight'] >= 0]

                        if len(foundation_data) > 0:
                            foundation_mean = foundation_data['mean'].iloc[0]
                            foundation_sem = foundation_data['sem'].iloc[0]
                            ax2.errorbar([-0.15], [foundation_mean], 
                                        yerr=[foundation_sem], marker=marker, markersize=8, capsize=5,
                                        color=color, alpha=0.7, linewidth=2)
                        
                        # MAE VE
                        if len(mae_data) > 0:
                            mae_weights = mae_data['adversarial_weight'].values
                            mae_means = mae_data['mean'].values
                            mae_sems = mae_data['sem'].values
                            
                            ax2.errorbar(mae_weights, mae_means, yerr=mae_sems,
                                        marker=marker, markersize=8, capsize=5,
                                        color=color, label=attribute.capitalize(), linewidth=2)

                            for j, weight in enumerate(mae_weights):
                                comp_name = f'Weight_{weight}'
                                if (comp_name in attr_results and 
                                    attr_results[comp_name].get('significant', False)):
                                    if attribute.lower() == 'sex':
                                        y_pos = mae_means[j] + mae_sems[j] + 0.005
                                        va = 'bottom'
                                    else:
                                        y_pos = mae_means[j] - mae_sems[j] - 0.005
                                        va = 'top'
                                    
                                    ax2.text(weight, y_pos, "*", ha='center', va=va, 
                                            fontsize=16, fontweight='bold', color=color)
            
            ax2.axvline(x=-0.075, color='gray', linestyle='--', alpha=0.5)
            ax2.legend(fontsize=12)
            
            if len(mae_data) > 0:
                x_ticks = [-0.15] + list(mae_weights)
                x_labels = ['Foundation\nVE'] + [str(w) for w in mae_weights]
                ax2.set_xticks(x_ticks)
                ax2.set_xticklabels(x_labels, fontsize=12)
                ax2.tick_params(axis='y', labelsize=12)
        else:
            ax2.text(0.5, 0.5, 'No fairness data available', 
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        ax2.set_xlabel('MVE VE with Adversarial Weight', fontsize=14, fontweight='bold')
        ax2.set_ylabel('AUC for demographic prediction', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def print_summary(self, performance_results: dict, fairness_results: dict, alpha: float = 0.05):
        print("\n" + "="*60)
        print("ECHO MAE MIXED EFFECTS MODEL ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\n📊 PERFORMANCE ANALYSIS (Clinical Tasks):")
        if performance_results:
            significant_count = sum(1 for r in performance_results.values() if r.get('significant', False))
            print(f"• Total comparisons: {len(performance_results)}")
            print(f"• Significant improvements: {significant_count}")
            print(f"• Alpha level: {alpha}")
            
            print(f"\n📈 DETAILED PERFORMANCE RESULTS:")
            for comp_name, result in performance_results.items():
                sig_marker = "✅" if result.get('significant', False) else "❌"
                print(f"\n{comp_name} vs Foundation VE:")
                print(f"  • Coefficient (improvement): {result['coefficient']:.4f}")
                print(f"  • 95% CI: [{result['conf_int_lower']:.4f}, {result['conf_int_upper']:.4f}]")
                print(f"  • Improvement: {result['improvement_pct']:.1f}%")
                print(f"  • Corrected p-value: {result['p_value_corrected']:.4f}")
                print(f"  • Significant: {sig_marker}")
        else:
            print("❌ No performance results available")
        
        print(f"\n🎯 FAIRNESS ANALYSIS (Demographic Tasks):")
        if fairness_results:
            for attribute, attr_results in fairness_results.items():
                print(f"\n{attribute.upper()} Prediction Analysis:")
                significant_count = sum(1 for r in attr_results.values() if r.get('significant', False))
                print(f"  • Total comparisons: {len(attr_results)}")
                print(f"  • Significant improvements: {significant_count}")
                
                for comp_name, result in attr_results.items():
                    sig_marker = "✅" if result.get('significant', False) else "❌"
                    print(f"\n  {comp_name} vs Foundation VE:")
                    print(f"    • Coefficient: {result['coefficient']:.4f}")
                    print(f"    • 95% CI: [{result['conf_int_lower']:.4f}, {result['conf_int_upper']:.4f}]")
                    print(f"    • Fairness improvement: {result['fairness_improvement']:.1f}%")
                    print(f"    • Corrected p-value: {result['p_value_corrected']:.4f}")
                    print(f"    • Significant: {sig_marker}")
        else:
            print("❌ No fairness results available")
        
        print("\n📋 MODEL SPECIFICATIONS:")
        print("Performance: AUC ~ model_type + (1|repeat_fold_id) + vc(task)")
        print("Fairness: AUC ~ model_type + (1|repeat_fold_id)")
        print("Multiple comparison correction: Holm method")
        
        print("\n" + "="*60)


def main():
    csv_file = results_dir + "/mve_evaluation_results.csv"
    output_dir = Path(results_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        analyzer = MixedEffectsEchoMAEAnalyzer(csv_file)
        
        # Performance analysis with mixed effects
        performance_results = analyzer.perform_performance_analysis()
        
        # Fairness analysis with mixed effects
        fairness_results = analyzer.perform_fairness_analysis()
        
        fig = analyzer.create_figure(
            performance_results, 
            fairness_results, 
            str(output_dir / "figure2_mixed_effects.svg")
        )

        analyzer.print_summary(performance_results, fairness_results)
        
        import json
        
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return str(obj)
        
        results = {
            'performance': performance_results,
            'fairness': fairness_results,
            'methodology': {
                'performance_model': 'AUC ~ model_type + (1|repeat_fold_id) + vc(task)',
                'fairness_model': 'AUC ~ model_type + (1|repeat_fold_id)',
                'multiple_comparison': 'Holm method',
                'alpha': 0.05
            }
        }
        
        with open(output_dir / "mixed_effects_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=convert_numpy)
        
        print(f"\n✅ Mixed effects analysis complete! Results saved to: {output_dir}")
        
        plt.show()
        return analyzer, performance_results, fairness_results, fig
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


if __name__ == "__main__":
    analyzer, performance_results, fairness_results, fig = main()