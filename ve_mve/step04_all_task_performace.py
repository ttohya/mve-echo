import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
import seaborn as sns
import textwrap

plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

results_dir = # path to the dir for the results output
analysis_table = pd.read_csv(results_dir + "/mve_evaluation_results.csv")

def performace_fig(result_table, metric = 'AUC', save_path = results_dir + '/performace_auc.png'):
    metrics= result_table[(result_table['subgroup']=="overall")].copy()
    metrics_mean = metrics.groupby(['model_type','task']).mean().reset_index()
    ser_w0 = metrics_mean.copy()[metrics_mean['model_type']=='MVE_VE_w0.1'].set_index('task')[metric]
    ser_found =metrics_mean.copy()[metrics_mean['model_type']=='Foundation_VE'].set_index('task')[metric]
    common = [
        'Hypertrophic_cardiomyopathy',
        'Dilated_cardiompyopthy',
        'Pulmonary_embolism',
        'Myocardial_infarction',
        'IVC',
        'modTR',
        'modMR',
        'modAR',
        'modAS',
        'mildTR',
        'mildMR',
        'mildAR',
        'mildAS',
        'RV_func',
        'PHT',
        'ab_E_e',
        'rEF',
        'ab_LVD',
        'ab_LVMI',
        'ab_LAVI',
        'ab_Ao'
    ]

    v0 = [ser_w0[t] for t in common]
    v1 = [ser_found[t] for t in common]

    label_map = {
        'Hypertrophic_cardiomyopathy': 'Hypertrophic cardiomyopathy',
        'Dilated_cardiompyopthy': 'Dilated cardiomyopathy',
        'Pulmonary_embolism': 'Pulmonary embolism',
        'Myocardial_infarction': 'Myocardial infarction',
        'IVC': 'Dilated IVC',
        'mildTR': 'Mild TR',
        'mildMR': 'Mild MR',
        'mildAR': 'Mild AR',
        'mildAS': 'Mild AS',
        'modTR': 'Moderate TR',
        'modMR': 'Moderate MR',
        'modAR': 'Moderate AR',
        'modAS': 'Moderate AS',
        'RV_func' : 'Impaired RV Function',
        'PHT': 'TRPG > 31mmHg',
        'ab_E_e': "E/e' > 15",
        'rEF': 'LVEF < 50%',
        'ab_LVD': "LVDd > 5.8cm (M) or 5.2cm (F)",
        'ab_LVMI': "LVMI > 115g/m²  (M) or 95g/m² (F)",
        'ab_LAVI': "LAVI > 34mL/m²",
        'ab_Ao': 'Aortic Root/BSA: >2.1cm/m²',
    }

    raw_labels = [label_map.get(t, t) for t in common]
    labels = raw_labels

    df = pd.DataFrame({
        'MVE VE': v0,
        'Foundation VE': v1,

    }, index=labels)
    sns.set(style="whitegrid")
    palette = sns.color_palette("viridis_r", n_colors=2) 

    fig, ax = plt.subplots(figsize=(6, max(6, len(labels)*0.35)))

    df.plot.barh(ax=ax, color=palette, alpha=0.7)
    if metric == 'auc':
        ax.set_xlabel('AUC')
        ax.set_xlim(left=0.5, right=1)
    elif metric == 'accuracy': 
        ax.set_xlabel('Accuracy')
        ax.set_xlim(left=0.5, right=1)
    elif metric == 'f1':
        ax.set_xlabel('F1')
        ax.set_xlim(left=0.0, right=1)
   
    ymin, ymax = ax.get_xlim()


    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,  
        frameon=True, reverse=True)

    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    
    return df



auc_df = performace_fig(analysis_table, metric = 'auc', save_path = results_dir + '/performace_auc.svg')
acc_df = performace_fig(analysis_table, metric = 'accuracy', save_path = results_dir + '/performace_accuracy.svg')
f1_df = performace_fig(analysis_table, metric = 'f1', save_path = results_dir + '/performace_f1.svg')
