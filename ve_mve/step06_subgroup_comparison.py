import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.formula.api import mixedlm

results_dir = # path to the dir for the results output

class RepeatFoldRandomEffectsAnalyzer:
    """mixed effect model"""
    def __init__(self, csv_file: str):
        self.df = pd.read_csv(csv_file)
        self.baseline_models = ['Foundation_VE', 'MVE_VE_w0', 'MVE_VE_w0.1', 'MVE_VE_w0.2']
        self.subgroup_mapping = {
            'male': 'sex', 'female': 'sex', 
            'white': 'race', 'black': 'race', 'others': 'race',
            'view3_low': 'view_count', 'view3_medium': 'view_count', 'view3_high': 'view_count',
            'view4_lt20': 'view_count', 'view4_20to39': 'view_count',
            'view4_40to59': 'view_count', 'view4_ge60': 'view_count', 
        }
        self.results = {}
        
    def prepare_data(self) -> pd.DataFrame:
        """data"""
        self.df['auc'] = pd.to_numeric(self.df['auc'], errors='coerce')

        data = self.df[
            (self.df['model_type'].isin(self.baseline_models)) & 
            (self.df['subgroup'] != 'overall') &
            (~self.df['task'].isin(['sex_encoded', 'race_encoded']))
        ].dropna(subset=['auc']).copy()

        if 'fold' not in data.columns:
            data = data.sort_values(['model_type', 'repeat_idx', 'task', 'subgroup'])
            data['fold'] = data.groupby(['model_type', 'repeat_idx', 'task', 'subgroup']).cumcount() % 5

        data = data.sort_values(['repeat_idx', 'fold'])
        unique_combinations = data[['repeat_idx', 'fold']].drop_duplicates().sort_values(['repeat_idx', 'fold'])
        combination_mapping = {
            (row['repeat_idx'], row['fold']): idx 
            for idx, (_, row) in enumerate(unique_combinations.iterrows())
        }
        
        data['repeat_fold_id'] = data.apply(
            lambda row: combination_mapping[(row['repeat_idx'], row['fold'])], axis=1
        )

        data['subgroup_category'] = data['subgroup'].map(self.subgroup_mapping)
        
        print(f"data preparation: {len(data)} observations, {len(data['repeat_fold_id'].unique())} repeat-fold comb")
        return data
    
    def run_mixed_effects_analysis(self, data: pd.DataFrame, category: str) -> dict:
        """mixed effect model"""
        cat_data = data[data['subgroup_category'] == category].copy()

        cat_data['model_type_cat'] = pd.Categorical(cat_data['model_type'])
        cat_data['subgroup_cat'] = pd.Categorical(cat_data['subgroup'])
        cat_data['task_cat'] = pd.Categorical(cat_data['task'])
        
        # Mixed effects model
        formula = "auc ~ C(model_type_cat) + C(subgroup_cat) + C(model_type_cat):C(subgroup_cat)"

        model = mixedlm(
            formula, 
            cat_data, 
            groups=cat_data['repeat_fold_id'],
            re_formula="1",
            vc_formula={"task": "0 + C(task_cat)"}
        ).fit()

        pvalues = model.pvalues
        model_p = subgroup_p = interaction_p = np.nan
        
        for param_name, p_val in pvalues.items():
            param_lower = param_name.lower()
            if 'model_type' in param_lower and 'subgroup' not in param_lower and ':' not in param_lower:
                if np.isnan(model_p) or p_val < model_p:
                    model_p = p_val
            elif 'subgroup' in param_lower and 'model_type' not in param_lower and ':' not in param_lower:
                if np.isnan(subgroup_p) or p_val < subgroup_p:
                    subgroup_p = p_val
            elif ':' in param_lower:
                if np.isnan(interaction_p) or p_val < interaction_p:
                    interaction_p = p_val

        repeat_fold_var = model.cov_re.iloc[0,0] if hasattr(model, 'cov_re') and len(model.cov_re) > 0 else 0.0
        task_var = model.vcomp[0] if hasattr(model, 'vcomp') and len(model.vcomp) > 0 else 0.0
        residual_var = model.scale

        total_var = repeat_fold_var + task_var + residual_var
        icc_repeat_fold = repeat_fold_var / total_var if total_var > 0 else 0
        icc_task = task_var / total_var if total_var > 0 else 0

        marginal_means = cat_data.groupby(['model_type', 'subgroup']).agg({
            'auc': ['mean', 'sem', 'count']
        }).round(4)
        marginal_means.columns = ['mean', 'sem', 'n']
        marginal_means = marginal_means.reset_index()
        
        return {
            'model_pvalue': model_p,
            'subgroup_pvalue': subgroup_p,
            'interaction_pvalue': interaction_p,
            'repeat_fold_variance': repeat_fold_var,
            'task_variance': task_var,
            'residual_variance': residual_var,
            'icc_repeat_fold': icc_repeat_fold,
            'icc_task': icc_task,
            'aic': model.aic,
            'bic': model.bic,
            'marginal_means': marginal_means,
            'n_obs': len(cat_data),
            'category': category
        }
    
    def perform_analysis(self) -> dict:
        data = self.prepare_data()
        categories = ['sex', 'race', 'view_count']
        
        for category in categories:
            result = self.run_mixed_effects_analysis(data, category)
            if result:
                self.results[category] = result
                print(f"{category}: 完了")

        print(f"\n" + "="*50)
        print("Summary")
        print("="*50)
        
        for category, result in self.results.items():
            print(f"\n{category.upper()}:")
            print(f"  観測数: {result['n_obs']}")
            
            effects = [
                ('Model', result['model_pvalue']),
                ('Subgroup', result['subgroup_pvalue']),
                ('Interaction', result['interaction_pvalue'])
            ]
            
            for name, p_val in effects:
                if not np.isnan(p_val):
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    print(f"  {name}: p = {p_val:.4f} {sig}")
            
            # ICC
            if not np.isnan(result['icc_repeat_fold']):
                print(f"  ICC(repeat-fold): {result['icc_repeat_fold']:.3f}")
            if not np.isnan(result['icc_task']):
                print(f"  ICC(task): {result['icc_task']:.3f}")
        
        return self.results

    def visualize_results(self, save_path: str = None) -> plt.Figure:
        categories = ['sex', 'race', 'view_count']
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        subgroup_order = {
            'sex': ['male', 'female'],
            'race': ['white', 'black', 'others'], 
            'view_count': ['view4_lt20', 'view4_20to39', 'view4_40to59', 'view4_ge60']
        }
        
        subgroup_labels = {
            'male': 'Male', 'female': 'Female',
            'white': 'White', 'black': 'Black', 'others': 'Others',
            'view4_lt20': '<20', 'view4_20to39': '20-39',
            'view4_40to59': '40-59', 'view4_ge60': '≥60', 
        }
        
        colors = sns.color_palette("viridis", n_colors=4)
        model_labels = {
            'Foundation_VE': 'Foundation VE',
            'MVE_VE_w0': 'MVE VE w=0', 
            'MVE_VE_w0.1': 'MVE VE w=0.1',
            'MVE_VE_w0.2': 'MVE VE w=0.2'
        }
        
        for idx, category in enumerate(categories):
            ax = axes[idx]
            
            if category not in self.results:
                ax.text(0.5, 0.5, f'No {category} data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=10)
                ax.set_title(category.capitalize())
                continue
            
            marginal_means = self.results[category]['marginal_means']
            ordered_subgroups = [sg for sg in subgroup_order[category] 
                               if sg in marginal_means['subgroup'].unique()]
            
            x_pos = np.arange(len(ordered_subgroups))
            width = 0.2
            
            for model_idx, model_type in enumerate(self.baseline_models):
                model_data = marginal_means[marginal_means['model_type'] == model_type]
                
                means, sems = [], []
                for subgroup in ordered_subgroups:
                    subgroup_data = model_data[model_data['subgroup'] == subgroup]
                    if len(subgroup_data) > 0:
                        means.append(subgroup_data['mean'].iloc[0])
                        sems.append(subgroup_data['sem'].iloc[0])
                    else:
                        means.append(np.nan)
                        sems.append(np.nan)
                
                valid_idx = ~np.isnan(means)
                if np.any(valid_idx):
                    x_offset = x_pos[valid_idx] + (model_idx - 1.5) * width
                    ax.bar(x_offset, np.array(means)[valid_idx], 
                          width=width, yerr=np.array(sems)[valid_idx],
                          capsize=5, alpha=0.5, color=colors[model_idx],
                          label=model_labels[model_type])
            
            ax.set_ylabel('AUC (SEM)' if idx == 0 else '', fontsize=10)
            ax.set_title(category.capitalize(), fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([subgroup_labels.get(sg, sg) for sg in ordered_subgroups])
            ax.set_ylim(0.65, 0.95)
            ax.grid(True, alpha=0.3)
            
            if idx == 0:
                ax.legend(fontsize=10, loc='upper left')
            
            result = self.results[category]
            legend_labels = []
            
            for effect_name, p_val in [('Model', result.get('model_pvalue')), 
                                     ('Subgroup', result.get('subgroup_pvalue')), 
                                     ('Interaction', result.get('interaction_pvalue'))]:
                if not np.isnan(p_val):
                    p_text = "p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"
                    legend_labels.append(f'{effect_name}: {p_text}')
            
           
            if legend_labels:
                legend_text = '\n'.join(legend_labels)
                ax.text(0.98, 0.98, legend_text, 
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

def main():
    csv_file = results_dir + "/mve_evaluation_results.csv"
    analyzer = RepeatFoldRandomEffectsAnalyzer(csv_file)
    results = analyzer.perform_analysis()
    fig = analyzer.visualize_results(results_dir + '/figure3_subgroup.svg')

    return analyzer

if __name__ == "__main__":
    analyzer = main()