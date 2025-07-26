import numpy as np
import pandas as pd
from tableone import TableOne

mimic_iv_dir = # path to the mimic_iv
mimic_iv_echo_dir =  # path to the "mimic-iv-echo/0.1" directory"
ds_dir = # dataset dir which includes "ve_echoprime.csv" and "echo_ds.csv")
results_dir = # path to the dir for the results output

echo_ds = pd.read_csv(ds_dir + "/echo_ds.csv", encoding='utf_8')
patients = pd.read_csv(mimic_iv_dir + "/hosp/patients.csv.gz")[["subject_id","anchor_age","anchor_year"]]
echo_list = pd.read_csv(mimic_iv_echo_dir + "/echo-study-list.csv")[["study_id","study_datetime"]]

echo_ds_uniq = echo_ds.copy().drop(["dicom_path",'view'],axis=1).drop_duplicates(subset='study_id')
echo_list['study_id'] = 's' + echo_list['study_id'].astype(str)
patients['subject_id'] = 'p' + patients['subject_id'].astype(str)
echo_ds_uniq = echo_ds_uniq.merge(echo_list,how='left',on='study_id')
echo_ds_uniq = echo_ds_uniq.merge(patients,how='left',on="subject_id")
echo_ds_uniq['age'] = echo_ds_uniq['anchor_age'] + pd.to_datetime(echo_ds_uniq['study_datetime']).dt.year - echo_ds_uniq['anchor_year']

label_map = {
    'Hypertrophic_cardiomyopathy': 'Hypertrophic cardiomyopathy',
    'Dilated_cardiompyopthy': 'Dilated cardiomyopathy',
    'Pulmonary_embolism':'Pulmonary embolism', 
    'Myocardial_infarction':'Myocardial infarction',
    
    'IVC':"Dilated IVC",
    'mildTR': 'Mild TR or greater',
    'mildMR': 'Mild MR or greater',
    'mildAR': 'Mild AR or greater',
    'mildAS': 'Mild AS or greater',
    'modTR': 'Moderate TR or greater',
    'modMR': 'Moderate MR or greater',
    'modAR': 'Moderate AR or greater',
    'modAS': 'Moderate AS or greater',
    'RV_func' : 'Impaired RV Function',
    'PHT': 'TRPG > 31mmHg',
    'ab_E_e': "E/e' > 15",
    'rEF': 'LVEF < 50%',
    'ab_LVD': "LVDd > 5.8cm(M) or 5.2cm(F)",
    'ab_LVMI': "LVMI > 115g/m²(M) or 95g/m²(F)",
    'ab_LAVI': "LAVI > 34mL/m²",
    'ab_Ao': 'Aortic Root/BSA: >2.1cm/m²',
    
    'age':'Age',
    'view_cnt':'Video count'
}


def tableone_create(df, result_path):
    df['Age'] = df['age']

    categorical_columns = ['ab_Ao', 'ab_LVD', 'ab_LAVI', 'ab_LVMI',
       'rEF', 'RV_func', 'mildAR', 'mildAS', 'mildMR', 'mildTR', 'modAR',
       'modAS', 'modMR', 'modTR', 'PHT', 'ab_E_e', 'IVC', 'Echo_quality',
       'Dilated_cardiompyopthy', 'Hypertrophic_cardiomyopathy', 
       'Pulmonary_embolism', 'Myocardial_infarction',
       'Sex', 'Race'
                          ]

    continuous_columns = ['Age','view_cnt',]

    decimals_list = {'Age':0,
                    'view_cnt':0,
                    }

    limit_list = {
    'Hypertrophic_cardiomyopathy': 1,
    'Dilated_cardiompyopthy': 1,
    'Myocardial_infarction': 1,
    'Pulmonary_embolism': 1,
    'Echo_quality':1,
    'mildTR': 1,
    'mildMR': 1,
    'mildAR': 1,
    'mildAS': 1,
    'modTR': 1,
    'modMR': 1,
    'modAR': 1,
    'modAS': 1,
    'RV_func' : 1,
    'PHT': 1,
    'IVC':1,
    'ab_E_e': 1,
    'rEF': 1,
    'ab_LVD': 1,
    'ab_LVMI': 1,
    'ab_LAVI': 1,
    'ab_Ao': 1,
    'Sex':2,
    'Race':5,
                  }


    order_list = {
                'Sex':["F","M"],
                'Race':["White","Black","Hispanic","Asian","Other"],
                'ab_Ao':["1.0", "0.0"],
                'ab_LVD':["1.0", "0.0"],
                'ab_LAVI':["1.0", "0.0"],
                'ab_LVMI':["1.0", "0.0"],
                'rEF':["1.0", "0.0"],
                'RV_func':["1.0", "0.0"],
                'mildAR':["1.0", "0.0"],
                'mildAS':["1.0", "0.0"],
                'mildMR':["1.0", "0.0"],
                'mildTR':["1.0", "0.0"],
                'modAR':["1.0", "0.0"],
                'modAS':["1.0", "0.0"],
                'modMR':["1.0", "0.0"],
                'modTR':["1.0", "0.0"],
                'PHT':["1.0", "0.0"],
                'ab_E_e':["1.0", "0.0"],
                'IVC':["1.0", "0.0"],
                'Echo_quality':["1.0", "0.0"],
                'Dilated_cardiompyopthy':["1.0", "0.0"],
                'Hypertrophic_cardiomyopathy':["1.0", "0.0"],  
                'Myocardial_infarction':["1.0", "0.0"],
                'Pulmonary_embolism':["1.0", "0.0"],                
                }

    columns = [ 'view_cnt','Sex', 'Race', 
               'ab_Ao', 'ab_LVD', 'ab_LAVI', 'ab_LVMI',
               'rEF', 'RV_func', 
               'mildAR', 'mildAS', 'mildMR', 'mildTR',
               'modAR', 'modAS', 'modMR', 'modTR', 
               'PHT', 'ab_E_e', 'IVC', 'Echo_quality',
               'Dilated_cardiompyopthy', 'Hypertrophic_cardiomyopathy', 
               'Myocardial_infarction','Pulmonary_embolism', 
                ]

    mytable = TableOne(df, columns=columns,
                        categorical=categorical_columns,             
                        nonnormal=['Age', 'view_cnt'], 
                        decimals = decimals_list,
                        limit = limit_list,
                        order = order_list,
                        rename = label_map,
                        pval=False,
                        include_null =False,)

    print(mytable.tabulate(tablefmt="git_hub")) 
    mytable.tableone.to_csv(result_path)

tableone_create(
  echo_ds_uniq, results_dir + "/table_total_unique.csv")
