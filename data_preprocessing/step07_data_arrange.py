import pandas as pd
import numpy as np
import datetime as dt

mimic_iv_dir = # path to the MIMIC-IV data directory
mimic_iv_echo_dir = # path to the MIMIC-IV-ECHO data directory
csv_dir = # path to the data directory
dataset_dir = # path to the dataset directory

# demographics
mimic_demog = pd.read_csv(csv_dir + "/mimic_demog.csv")
# demographics
admissions = pd.read_csv(mimic_iv_dir + "/hosp/admissions.csv.gz")[['subject_id','hadm_id','admittime','dischtime']]

# ICD codes
mimic_icd_dx_allrange = pd.read_csv(csv_dir + "/mimic_icd_dx_allrange.csv")
mimic_icd_dx_inhosp = pd.read_csv(csv_dir + "/mimic_icd_dx_inhosp.csv")
# Echo measurements
echo_measurement = pd.read_csv(csv_dir + "/echo_measurement.csv")
# Echo view classifciation
view_predictions = pd.read_csv(csv_dir + "/view_predictions.csv")
view_predictions.rename(columns = {"path":"dicom_path"}, inplace=True)
# Echo dicom list
echo_list_df = pd.read_csv(mimic_iv_echo_dir + "/echo-record-list.csv")
# Echo dicom list
echo_study_list_df = pd.read_csv(mimic_iv_echo_dir + "/echo-study-list.csv")
# Echo dicom metadata
meta_df = pd.read_csv(csv_dir + "/dicom_meta_long_format.csv")


# MI, PE
"""
1. Get all records using outer join # 710,709
2. Valid if within 1 month before admission to 3 months after discharge
3. For each study, select the acquisition_datetime closest to the admission date (one admission per study)
4. MI: Positive if within valid period, diagnosed, and EF is reduced. Negative if within valid period, not diagnosed, and EF is not reduced. Otherwise, missing.
5. PE: Positive if within valid period, diagnosed, and PHT is present. Negative if within valid period, not diagnosed, and PHT is not present. Otherwise, missing.
"""
mimic_icd_dx_inhosp = mimic_icd_dx_inhosp.merge(admissions,how='inner',on=['subject_id','hadm_id'])
mimic_icd_dx_inhosp_echo = pd.merge(echo_study_list_df,mimic_icd_dx_inhosp,how='outer',on='subject_id')
# 570,195  -> 6,979 
mimic_icd_dx_inhosp_echo_valid = mimic_icd_dx_inhosp_echo.copy()[
    (pd.to_datetime(mimic_icd_dx_inhosp_echo["admittime"]) - dt.timedelta(
    days=30) < pd.to_datetime(mimic_icd_dx_inhosp_echo["study_datetime"])) & 
    (pd.to_datetime(mimic_icd_dx_inhosp_echo["dischtime"]) + dt.timedelta(
    days=90) > pd.to_datetime(mimic_icd_dx_inhosp_echo["study_datetime"]))]
# Unique study
mimic_icd_dx_inhosp_echo_valid['diff_date'] = abs(
    pd.to_datetime(mimic_icd_dx_inhosp_echo["admittime"]) - pd.to_datetime(mimic_icd_dx_inhosp_echo["study_datetime"]))

mimic_icd_dx_inhosp_echo_unique = mimic_icd_dx_inhosp_echo_valid.sort_values(['study_id','diff_date']).drop_duplicates(subset='study_id',keep='first')
mi_pe_ds_echo_measure = mimic_icd_dx_inhosp_echo_unique.merge(echo_measurement[['study_id','rEF','PHT']],how='left',on='study_id')

mi_pe_ds_echo_measure.loc[(mi_pe_ds_echo_measure['MI_candidate']==1) & 
                                    (mi_pe_ds_echo_measure['rEF']==1) ,'Myocardial_infarction'] = 1
mi_pe_ds_echo_measure.loc[(mi_pe_ds_echo_measure['MI_candidate']==0) & 
                                    (mi_pe_ds_echo_measure['rEF']==0) ,'Myocardial_infarction'] = 0
mi_pe_ds_echo_measure.loc[(mi_pe_ds_echo_measure['PE_candidate']==1) & 
                                    (mi_pe_ds_echo_measure['PHT']==1) ,'Pulmonary_embolism'] = 1
mi_pe_ds_echo_measure.loc[(mi_pe_ds_echo_measure['PE_candidate']==0) & 
                                    (mi_pe_ds_echo_measure['PHT']==0) ,'Pulmonary_embolism'] = 0


video_list = meta_df[meta_df.name=='Number of Frames'].dcm_path.apply(lambda x:os.path.basename(x)).to_list()
echo_list_df["dicom"] = echo_list_df.dicom_filepath.apply(lambda x:os.path.basename(x))
echo_list_df['video'] = echo_list_df.dicom.isin(video_list)

mimic_iv_df_ = pd.merge(mimic_demog,mimic_icd_dx_allrange,how='inner',on='subject_id')
mimic_iv_df_ = pd.merge(echo_measurement,mimic_iv_df_,how='left',on='subject_id') #leftに変更
mimic_iv_df_ = pd.merge(mimic_iv_df_,mi_pe_ds_echo_measure[['subject_id','study_id','Myocardial_infarction','Pulmonary_embolism']],how='left',on=['subject_id','study_id']) #追加


echo_list_df['dicom_path'] = echo_list_df['dicom_filepath'].apply(lambda x: mimic_iv_echo_dir + x[5:])


echo_df_ = pd.merge(echo_list_df,mimic_iv_df_,how='left',on=['subject_id','study_id'])
echo_df = echo_df_.merge(view_predictions,how='inner',on="dicom_path")


# view cnt
echo_df_uniq = echo_df.copy().drop(["dicom_path",'view'],axis=1).drop_duplicates(subset='study_id')
echo_view_cnt = echo_df.copy().groupby(['study_id']).agg({'dicom_path':'count'}).rename(columns={'dicom_path':'view_cnt'})
echo_df = echo_df.merge(echo_view_cnt,how='left',on='study_id')

# change unknown to np.nan in race categ
echo_df['Race'] = echo_df['Race'].replace({'Unknown':np.nan})
echo_df['Sex'] = echo_df['Sex'].replace({'Unknown':np.nan})

echo_df['Sex_inUKN'] =  echo_df['Sex'].replace({
    'M':'Male',
    'F':'Female',
    np.nan:'Unknown'})
echo_df['Race_inUKN'] =  echo_df['Race'].replace({
    np.nan:'Unknown',
    'Hispanic':'Others',
    'Asian':'Others',
    'Other':'Others'})
echo_df['Sex_exUKN'] =  echo_df['Sex_inUKN'].replace({
    'Unknown':np.nan})
echo_df['Race_exUKN'] =  echo_df['Race_inUKN'].replace({
    'Unknown':np.nan})

echo_df['ViewCount_category'] = 'Medium'
echo_df.loc[echo_df['view_cnt'] < 30, 'ViewCount_category'] = 'Low'
echo_df.loc[echo_df['view_cnt'] > 50, 'ViewCount_category'] = 'High'

echo_df.loc[echo_df['view_cnt'] < 20, 'ViewCount_4cat'] = '< 20'
echo_df.loc[(echo_df['view_cnt'] >= 20) & (echo_df['view_cnt'] < 40), 'ViewCount_4cat'] = '20-39'
echo_df.loc[(echo_df['view_cnt'] >= 40) & (echo_df['view_cnt'] < 60), '40-59'] = 'Medium'
echo_df.loc[echo_df['view_cnt'] >= 60, 'ViewCount_4cat'] = '>= 60'

echo_df.loc[echo_df['view_cnt'] < 20, 'ViewCount_6cat'] = '< 20'
echo_df.loc[(echo_df['view_cnt'] >= 20) & (echo_df['view_cnt'] < 30), 'ViewCount_6cat'] = '20-29'
echo_df.loc[(echo_df['view_cnt'] >= 30) & (echo_df['view_cnt'] <= 40), 'ViewCount_6cat'] = '30-39'
echo_df.loc[(echo_df['view_cnt'] >= 40) & (echo_df['view_cnt'] <= 50), 'ViewCount_6cat'] = '40-49'
echo_df.loc[(echo_df['view_cnt'] > 50) & (echo_df['view_cnt'] <= 60), 'ViewCount_6cat'] = '50-59'
echo_df.loc[echo_df['view_cnt'] > 60, 'ViewCount_6cat'] = '>= 60'

id_vars = ['subject_id', 'study_id', 'dicom_path', 'view']
echo_vars = ['ab_Ao', 'ab_LVD', 'ab_LAVI',
       'ab_LVMI', 'rEF', 'RV_func', 'mildAR', 'mildAS', 'mildMR', 'mildTR',
       'modAR', 'modAS', 'modMR', 'modTR', 'PHT', 'ab_E_e', 'IVC',
       'Echo_quality','view_cnt','ViewCount_category','ViewCount_4cat','ViewCount_6cat']
demog_vars = ['Sex', 'Race', 'Sex_inUKN', 'Race_inUKN',
              'Sex_exUKN', 'Race_exUKN',
              'Medicaid', 'Private', 'Medicare','Language']
dx_vars = ['Dilated_cardiompyopthy', 'Hypertrophic_cardiomyopathy', 'Pulmonary_embolism', 'Myocardial_infarction']

echo_ds = echo_df.copy()[echo_df.video][id_vars + echo_vars + dx_vars + demog_vars]
echo_ds["subject_id"] = "p" + echo_ds["subject_id"].astype(str).str.zfill(8)
echo_ds["study_id"] = "s" +echo_ds["study_id"].astype(str).str.zfill(8)
         
echo_ds.to_csv(dataset_dir + "/echo_ds.csv",index=False)


