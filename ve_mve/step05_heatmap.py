import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path

results_dir = # path to the dir for the results output

class CSVHeatmapVisualizer:
    def __init__(self, csv_path: str):
        """
        Args:
            csv_path: mve_evaluation_results.csvのパス
        """
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

        self.subgroup_tasks = ['mildAS', 'rEF', 'RV_func', 'ab_E_e']
        self.target_subgroups = ['male', 'female', 'white', 'black', 'others']
        
        # データの前処理
        self._preprocess_data()
    
    def _preprocess_data(self):
        print("📊 Preprocessing CSV data...")

        self.task_mapping = {
            'mild_AS': 'mildAS',
            'rEF': 'rEF', 
            'RV_func': 'RV_func',
            'ab_E_e': 'ab_E_e'
        }

        self.reverse_task_mapping = {v: k for k, v in self.task_mapping.items()}

        self.subgroup_mapping = {
            'male': 'sex_M',
            'female': 'sex_F', 
            'white': 'race_White',
            'black': 'race_Black',
            'others': 'race_Others'
        }

        self.reverse_subgroup_mapping = {v: k for k, v in self.subgroup_mapping.items()}
        print(f"  • Loaded {len(self.df)} records")
        print(f"  • Tasks: {sorted(self.df['task'].unique())}")
        print(f"  • Subgroups: {sorted(self.df['subgroup'].unique())}")
        print(f"  • Adversarial weights: {sorted(self.df['adversarial_weight'].unique())}")
    
    def _get_task_data(self, task_name: str, exclude_weight_1: bool = False):
        # Adversarial weight
        if exclude_weight_1:
            weights_to_use = [w for w in sorted(self.df['adversarial_weight'].unique()) if w != 1.0]
        else:
            weights_to_use = sorted(self.df['adversarial_weight'].unique())

        task_data = self.df[
            (self.df['task'] == task_name) & 
            (self.df['subgroup'].isin(self.target_subgroups)) &
            (self.df['adversarial_weight'].isin(weights_to_use)) &
            (self.df['model_type'].str.startswith('MVE_VE')) 
        ].copy()

        grouped = task_data.groupby(['subgroup', 'adversarial_weight']).agg({
            'auc': 'mean',
            'accuracy': 'mean',
            'f1': 'mean'
        }).reset_index()
        
        # Pivot table（subgroup x adversarial_weight）
        pivot_table = grouped.pivot(index='subgroup', columns='adversarial_weight', values='auc')

        subgroup_order = []
        subgroup_labels = []
        
        subgroup_label_mapping = {
            'male': 'Male',
            'female': 'Female', 
            'white': 'White',
            'black': 'Black',
            'others': 'Others'
        }
        
        for subgroup in self.target_subgroups:
            if subgroup in pivot_table.index:
                subgroup_order.append(subgroup)
                subgroup_labels.append(subgroup_label_mapping.get(subgroup, subgroup))

        pivot_table = pivot_table.reindex(subgroup_order)
        pivot_table.index = subgroup_labels
        
        return pivot_table
    
    def create_figure3_subgroup_heatmap(self, save_path: str = None, exclude_weight_1: bool = False, 
                                       colormap: str = 'Blues') -> plt.Figure:
        """Figure 3: Subgroup Analysis Heatmap
        
        Args:
            save_path: Path to save the figure
            exclude_weight_1: If True, exclude weight=1.0 from the analysis
            colormap: Color scheme for heatmap
        """
        print("📊 Creating Figure 3: Subgroup Analysis Heatmap...")
        
        # Task title mapping
        task_title_mapping = {
            'mildAS': 'Mild AS',
            'rEF': 'LVEF < 50%',
            'RV_func': 'Impaired RV function',
            'ab_E_e': "E/e' > 15"
        }

        available_tasks = []
        csv_tasks = self.df['task'].unique()

        task_name_variants = {
            'mildAS': ['mild_AS', 'mildAS', 'Mild AS'],
            'rEF': ['rEF', 'LVEF_lt50', 'reduced_EF'],
            'RV_func': ['RV_func', 'RV_dysfunction', 'impaired_RV'],
            'ab_E_e': ['ab_E_e', 'elevated_E_e', 'E_e_ratio']
        }
        
        for target_task in self.subgroup_tasks:
            if target_task in csv_tasks:
                available_tasks.append(target_task)
            else:
                for variant in task_name_variants.get(target_task, []):
                    if variant in csv_tasks:
                        available_tasks.append(variant)
                        break
        
        if len(available_tasks) == 0:
            print("❌ No matching tasks found in CSV")
            print(f"Available tasks in CSV: {list(csv_tasks)}")
            return None
        
        print(f"  • Found {len(available_tasks)} matching tasks: {available_tasks}")

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.flatten()
        
        for i, task in enumerate(available_tasks[:4]): 
            ax = axes[i]

            task_df = self._get_task_data(task, exclude_weight_1)
            
            if task_df is not None and not task_df.empty:
                cax = sns.heatmap(
                    task_df,
                    annot=True,
                    fmt='.3f',
                    cmap=colormap,
                    center=0.9,
                    vmin=0.80,
                    vmax=1.0,
                    ax=ax,
                    cbar_kws={
                        'label': 'AUC',
                        'ticks': np.linspace(0.8, 1.0, 5),
                        'format': '%.2f'
                    },
                    mask=task_df.isna(),
                    linewidths=1,
                    square=False
                )

                cbar = cax.collections[0].colorbar
                cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

                title_key = task
                for mapped_task, title in task_title_mapping.items():
                    if task in task_name_variants.get(mapped_task, [mapped_task]):
                        title_key = mapped_task
                        break
                
                task_title = task_title_mapping.get(title_key, task)
                ax.set_title(f'{task_title}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Adversarial Weight', fontsize=12)
                ax.set_ylabel('Subgroup', fontsize=12)
                ax.tick_params(axis='x', rotation=0)
                ax.tick_params(axis='y', rotation=0)
                
            else:
                task_title = task_title_mapping.get(task, task)
                ax.text(0.5, 0.5, f'No valid data\nfor {task_title}', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=14, color='red', weight='bold')
                ax.set_title(f'{task_title} (No Data)', fontsize=14, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])

        for i in range(len(available_tasks), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  • Figure 3 saved: {save_path}")
        
        print("✅ Figure 3 created")
        return fig
    
    def create_summary_table(self):
        print("\n📋 Data Summary:")
        print("=" * 50)

        print(f"Total records: {len(self.df)}")
        print(f"Tasks: {self.df['task'].nunique()}")
        print(f"Subgroups: {self.df['subgroup'].nunique()}")
        print(f"Adversarial weights: {len(self.df['adversarial_weight'].unique())}")
        print(f"Repeats: {self.df['repeat_idx'].nunique()}")

        print("\nTask counts:")
        task_counts = self.df['task'].value_counts()
        for task, count in task_counts.items():
            print(f"  {task}: {count}")

        print("\nSubgroup counts:")
        subgroup_counts = self.df['subgroup'].value_counts()
        for subgroup, count in subgroup_counts.items():
            print(f"  {subgroup}: {count}")


def main():
    csv_path = results_dir + "/mve_evaluation_results.csv"  
    
    if not Path(csv_path).exists():
        print(f"❌ CSV file not found: {csv_path}")
        print("Please specify the correct path to mve_evaluation_results.csv")
        return

    visualizer = CSVHeatmapVisualizer(csv_path)
    visualizer.create_summary_table()
    fig = visualizer.create_figure3_subgroup_heatmap(
        save_path= results_dir + "/figure4_subgroup_heatmap.svg",
        exclude_weight_1=False,
        colormap='Blues'
    )
    
    if fig:
        plt.show()


if __name__ == "__main__":
    main()