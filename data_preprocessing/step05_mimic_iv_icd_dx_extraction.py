import pandas as pd

mimic_iv_dir = # path to the MIMIC-IV data directory
v

mimic_patients_df = pd.read_csv(mimic_iv_dir + "/hosp/patients.csv.gz")
mimic_admissions_df = pd.read_csv(mimic_iv_dir +"/hosp/admissions.csv.gz")
mimic_icd_df = pd.read_csv(mimic_iv_dir +"/hosp/diagnoses_icd.csv.gz")
icd_master = pd.read_csv(csv_dir + "/d_icd_diagnoses_master.csv")

# output path for the processed data
mimic_icd_df_path = csv_dir + "/mimic_icd_dx_allrange.csv"
mimic_icd_inhosp_path = csv_dir + "/mimic_icd_dx_inhosp.csv"

allrange_dx_list = ['Dilated_cardiompyopthy','Hypertrophic_cardiomyopathy']
inhosp_dx_list = ['MI_candidate', 'PE_candidate']

# diagnosis at anytimie during records
def Category_cnt_allrange(df):
    firstLoop = True
    for i in allrange_dx_list:
        df2=df.copy()
        recept_list = icd_master[
            icd_master['dx_codes']== i ]['icd_code'].to_list()
        df2[i]=df2['icd_code'].isin(recept_list).astype(int)
        df2=df2[['subject_id',i]].groupby(
            ['subject_id']).agg({i:'max'})
        if firstLoop:
            df3=df2.copy()
            firstLoop = False
        else:
            df3 = pd.merge(df3,df2,how='left',on=['subject_id'])
    return df3.reset_index(drop=False)

dx_cnt_allrange = Category_cnt_allrange(mimic_icd_df) 


# diagnosis during hospitalization
def Category_cnt_inhosp(df):
    firstLoop = True
    for i in inhosp_dx_list:
        df2=df.copy()
        recept_list = icd_master[
            icd_master['dx_codes']== i ]['icd_code'].to_list()
        df2[i]=df2['icd_code'].isin(recept_list).astype(int)
        df2=df2[['subject_id','hadm_id',i]].groupby(
            ['subject_id','hadm_id']).agg({i:'max'})
        if firstLoop:
            df3=df2.copy()
            firstLoop = False
        else:
            df3 = pd.merge(df3,df2,how='left',on=['subject_id','hadm_id'])
    return df3.reset_index(drop=False)

dx_cnt_inhosp = Category_cnt_inhosp(mimic_icd_df) 

dx_cnt_allrange.to_csv(mimic_icd_df_path,index=False)
dx_cnt_inhosp.to_csv(mimic_icd_inhosp_path,index=False)

