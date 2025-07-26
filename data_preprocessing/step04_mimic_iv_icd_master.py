import pandas as pd

mimic_iv_dir = # path to the MIMIC-IV data directory
csv_dir = # path to the csv directory
d_icd_diagnoses = pd.read_csv(mimic_iv_dir + "/hosp/d_icd_diagnoses.csv.gz")

#  hypertrophic cardiomyopathy
Hypertrophic_cardiomyopathy = d_icd_diagnoses[d_icd_diagnoses["icd_code"].str.startswith(("4251","I421","I422"))]["icd_code"].to_list()
d_icd_diagnoses[d_icd_diagnoses["icd_code"].isin([
"4251","I421","I422"])]

# Dilated_cardiompyopthy
Dilated_cardiompyopthy = d_icd_diagnoses[d_icd_diagnoses["icd_code"].str.startswith(
("4254","I420"))]["icd_code"].to_list()

PE_candidate = d_icd_diagnoses[d_icd_diagnoses["icd_code"].str.startswith(("415","I26"))]["icd_code"].to_list()
AMI_candidate = d_icd_diagnoses[d_icd_diagnoses["icd_code"].str.startswith(("410","I21"))]["icd_code"].to_list()
reMI_candidate = d_icd_diagnoses[d_icd_diagnoses["icd_code"].str.startswith(("410","I22"))]["icd_code"].to_list()
OMI_candidate = d_icd_diagnoses[d_icd_diagnoses["icd_code"].str.startswith(("412","I252"))]["icd_code"].to_list()

d_icd_diagnoses.loc[d_icd_diagnoses["icd_code"].isin(Hypertrophic_cardiomyopathy),'dx_codes'] = "Hypertrophic_cardiomyopathy"
d_icd_diagnoses.loc[d_icd_diagnoses["icd_code"].isin(Dilated_cardiompyopthy),'dx_codes'] = "Dilated_cardiompyopthy"
d_icd_diagnoses.loc[d_icd_diagnoses["icd_code"].isin(PE_candidate),'dx_codes'] = "PE_candidate"
d_icd_diagnoses.loc[d_icd_diagnoses["icd_code"].isin(AMI_candidate + reMI_candidate + OMI_candidate),'dx_codes'] = "MI_candidate"

d_icd_diagnoses_master = d_icd_diagnoses.copy()[~d_icd_diagnoses.dx_codes.isna()]
d_icd_diagnoses_master.to_csv(csv_dir + "/d_icd_diagnoses_master.csv",index=False)
