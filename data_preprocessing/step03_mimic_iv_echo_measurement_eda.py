import pandas as pd
import numpy as np

csv_dir = # path to the data directory
echo_mm_path = csv_dir + "/echo_mm_matchingfile.csv"
mimic_demog_path = csv_dir + "/mimic_demog.csv"
dicom_meta_long_format_path = csv_dir + "/dicom_meta_long_format.csv"
echo_measurement_path = csv_dir + "/echo_measurement.csv" #output path for echo measurement data


echo_mm = pd.read_csv(echo_mm_path)
echo_id = echo_mm[["subject_id", "study_id", "study_datetime"]].drop_duplicates()
mimic_demog = pd.read_csv(mimic_demog_path)[["subject_id",'Sex']]

dicom_meta_long_format = pd.read_csv(dicom_meta_long_format_path)
dicom_color_doppler = dicom_meta_long_format.copy()[(dicom_meta_long_format.name=="Region Data Type")&(dicom_meta_long_format.value=="2")]
dicom_color_doppler_list = dicom_color_doppler['dcm_path'].apply(lambda x: os.path.basename(os.path.dirname(x))).str[1:].astype(int).drop_duplicates().to_list()

# Body Surface Area
la_mapping = {
    "Normal LAVI (<34ml/m2)": 0,
    "Mild LAVI (34-41ml/m2) ": 1,
    "Mod LAVI (41-48ml/m2)": 1,
    "Severe LAVI(>48ml/m2)": 1,
    "Nl (PLA <4.1cm)": 0,
    "Mild LAE (PLA 4.1-4.9cm)": 1,
    "Mod LAE (PLA 5.0-5.9cm)": 1,
    "Sev LAE (PLA >=6.0cm)": 1,
    "LA Elongated (4Ch>5.2cm)": 1,
    "Dilated - no qualifier": np.nan, # not qualified -> nan 
    "Not well seen": np.nan,
}
tr_mapping = {
    "Mild [1+]": 1,
    "Physiologic": 0,
    "Trivial": 0,
    "Mild-mod [1-2+]": 2,
    "Mod [2+]": 3,
    "Mod-severe [3+]": 4,
    "Severe [4+]": 5,
    "Present, can't qualify": np.nan, # not qualified -> nan 
}
mr_mapping = {
    "Mild (1+)": 1,
    "Physiologic": 0,
    "Trivial": 0,
    "Mild-moderate (1-2+)": 2,
    "Mod [2+]": 3,
    "Moderate-severe (3+)": 4,
    "Severe (4+)": 5,
    "Present, can't qualify": np.nan, # not qualified -> nan 
}
ar_mapping = {
    "Mild [1+]": 1,
    "Trace": 0,
    "Trace (NL for prosthesis)": 0,
    "Mild-mod [1-2+]": 2,
    "Mod [2+]": 3,
    "Mod-severe [3+]": 4,
    "Severe [4+]": 5,
    "Present": np.nan,
}
as_mapping = {
    "Minimal": 0,
    "None, incr vel high output": 0,
    "None, incr grad from AR/stroke vol": 0,
    "Mild (>1.5cm2)": 1,
    "Mod (1.0-1.5cm2)": 2,
    "Severe (<1.0cm2)": 3,
    "Severe (<0.6cm2/m2)": 3,
    "Very severe (>5m/s;>60mmHg)": 3,
    "Low flow/low gradient severe": 3,
    "Present - not quantified": np.nan,
    "Cannot exclude": np.nan,
}
rv_mapping = {
    "Nl RV function": 0,
    "Low normal function": 0,
    "Hyperdynamic": 0,
    "Mild global RV hypo": 1,
    "Moderate global RV hypo": 1,
    "RV function depressed": 1,
    "Severe global hypo": 1,
    "Apical free wall hypo": 1,
    "Basal RV hypo (McConnell's sign)": 1,
    "Cannot assess RV function": np.nan,
    "RV not well seen": np.nan,
}
ivc_mapping = {
    "<2.2cm, >50% (0-5mmHg)": 0,
    "IVC<=2.1cm, <50% collapse (5-10mmHg)": 0,
    "NL IVC <2.2 cmd": 0,
    "Intubated, IVC<1.5cm, RA <10mmHg": 0,
    ">2.1cm, <50% (>15mmHg)": 1,
    ">2.1cm, >50% (10-15mmHg)": 1,
    "IVC dilated (>2.5cm)": 1,
    "Intubated - cannot assess RA pressure": np.nan,
    "IVC not visualized": np.nan,
}
quality_mapping= {
    "Adequate": 0,
    "Suboptimal": 0,
    "Good": 0,
    "Poor": 1,
}

# Aorta
echo_mm['Ao_Sinus'] = echo_mm[echo_mm.measurement_description == "Aorta - Sinus Diameter"]["result"]

# Aritum
echo_mm['LAD'] = echo_mm[echo_mm.measurement_description == "Left Atrium - Diastolic Dimension"]["result"]
echo_mm['LAV'] = echo_mm[echo_mm.measurement_description == "Left Atrium - Volume"]["result"]
echo_mm['LA_4C'] = echo_mm[echo_mm.measurement_description == "Left Atrium - Four Chamber Length"]["result"]
echo_mm['RA_4C'] = echo_mm[echo_mm.measurement_description == "Right Atrium - Four Chamber Length"]["result"]

# LV
echo_mm['LVDD'] = echo_mm[echo_mm.measurement_description == "Left Ventricle - End Diastolic Dimension"]["result"]
echo_mm['LVDS'] = echo_mm[echo_mm.measurement_description == "Left Ventricle - End Systolic Dimension"]["result"]
echo_mm['LVPW'] = echo_mm[echo_mm.measurement_description == "Left Ventricle - Inferolateral Wall Thickness"]["result"]
echo_mm['IVS'] = echo_mm[echo_mm.measurement_description == "Left Ventricle - Septal Wall Thickness"]["result"]
echo_mm['LVOTd'] = echo_mm[echo_mm.measurement_description == "Left Ventricle - LVOT Diameter"]["result"]

# IVC
echo_mm['IVCd'] = echo_mm[echo_mm.measurement_description == "Inferior Vena Cava - Diameter"]["result"]

# Aortic valve
echo_mm['AV_Vmax'] = echo_mm[echo_mm.measurement_description == "Aortic Valve - Peak Velocity"]["result"]

# Mitral valve
echo_mm['MV_Dec'] = echo_mm[echo_mm.measurement_description == "Mitral Valve - E Wave Deceleration Time"]["result"].astype(float)
echo_mm['MV_A'] = echo_mm[echo_mm.measurement_description == "Mitral Valve - Peak A Wave"]["result"].astype(float)
echo_mm['MV_E'] = echo_mm[echo_mm.measurement_description == "Mitral Valve - Peak E Wave"]["result"].astype(float)
echo_mm['MV_EA'] = echo_mm[echo_mm.measurement_description == "Mitral Valve - E/A Ratio"]["result"].astype(float)
echo_mm['MV_Ee'] = echo_mm[echo_mm.measurement_description == "E To E Prime Ratio"]["result"].astype(float)

# Tricuspid valve
echo_mm['TR_PG'] = echo_mm[echo_mm.measurement_description == "Tricuspid Valve - TR Pressure Gradient"]["result"]

# Body Surface Area
echo_mm['BSA'] = echo_mm[echo_mm.measurement_description == "Body Surface Area"]["result"]


# 2. Valvular measurements
# AR
echo_mm['AR'] = np.where(
    echo_mm.measurement_description == "Aortic Valve - Regurgitation",
    echo_mm['result'].map(ar_mapping),
    np.nan
)
# AS
echo_mm['AS'] = np.where(
    echo_mm.measurement_description == "Aortic Valve - Stenosis",
    echo_mm['result'].map(as_mapping),
    np.nan
)
# MR
echo_mm['MR'] = np.where(
    echo_mm.measurement_description == "Mitral Valve - Regurgitation",
    echo_mm['result'].map(mr_mapping),
    np.nan
)
# TR
echo_mm['TR'] = np.where(
    echo_mm.measurement_description == "Tricuspid Valve - Regurgitation",
    echo_mm['result'].map(tr_mapping),
    np.nan
)

# RV fanction
echo_mm['RV_func'] = np.where(
    echo_mm.measurement_description == "Right Ventricle - Function",
    echo_mm['result'].map(rv_mapping),
    np.nan
)
# IVC
echo_mm['IVC'] = np.where(
    echo_mm.measurement_description == "Inferior Vena Cava - Size Description",
    echo_mm['result'].map(ivc_mapping),
    np.nan
).astype(float)
# Technical Quality
echo_mm['Echo_quality'] = np.where(
    echo_mm.measurement_description == "Technical Quality",
    echo_mm['result'].map(quality_mapping),
    np.nan
)



valve_frag = echo_mm.study_id.isin(dicom_color_doppler_list)
echo_mm["valve_eva"] = valve_frag.astype(float).replace(0, np.nan)

echo_variables = ['subject_id', 'study_id', 'study_datetime',
    'Ao_Sinus', 'LAD', 'LAV', 'LA_4C', 'RA_4C', 'LVDD', 'LVDS',
    'LVPW', 'IVS', 'LVOTd', 'IVCd', 'AV_Vmax', 'MV_Dec', 'MV_A',
    'MV_E', 'MV_EA', 'MV_Ee', 'TR_PG', 'AR', 'AS', 'MR', 'TR',
    'RV_func', 'IVC', 'BSA', 'Echo_quality', 'valve_eva']

# 3. Stacking longformat echo measurements
# If there are multiple measurements within the same study, use the last measurement value
echo_mm_sorted = echo_mm.sort_values(by='study_id')
# Perform forward fill (LOCF) within each 'study_id' 
echo_mm_filled = echo_mm_sorted.groupby('study_id')[echo_variables].ffill()
echo_mm_sorted[echo_variables] = echo_mm_filled
# Retrieve the last entry for each 'study_id' group
echo_df = echo_mm_sorted.groupby('study_id').last().reset_index()[echo_variables]
echo_df = echo_df.merge(mimic_demog,how='left',on='subject_id')


# 4. Echo parameter calcuration
# ASE 2015 guideline based cutoff
conti_vars = [
    'Ao_Sinus', 'LAD', 'LAV', 'LA_4C', 'RA_4C', 'LVDD', 'LVDS',
    'LVPW', 'IVS', 'LVOTd', 'IVCd', 'AV_Vmax', 'MV_Dec', 'MV_A',
    'MV_E', 'MV_EA', 'MV_Ee', 'TR_PG', 'BSA']
echo_df[conti_vars] = echo_df[conti_vars].astype(float)
# normal valuve replaced to 0
echo_df.loc[(echo_df["valve_eva"]==1)&(echo_df["AR"].isna()), "AR"] = 0
echo_df.loc[(echo_df["valve_eva"]==1)&(echo_df["AS"].isna()), "AS"] = 0
echo_df.loc[(echo_df["valve_eva"]==1)&(echo_df["MR"].isna()), "MR"] = 0
echo_df.loc[(echo_df["valve_eva"]==1)&(echo_df["TR"].isna()), "TR"] = 0
# valve mild or greater (1: mild or more)
echo_df.loc[echo_df["AR"]>=0, "mildAR"] = 0
echo_df.loc[echo_df["AS"]>=0, "mildAS"] = 0
echo_df.loc[echo_df["MR"]>=0, "mildMR"] = 0
echo_df.loc[echo_df["TR"]>=0, "mildTR"] = 0
echo_df.loc[echo_df["AR"]>=1, "mildAR"] = 1
echo_df.loc[echo_df["AS"]>=1, "mildAS"] = 1
echo_df.loc[echo_df["MR"]>=1, "mildMR"] = 1
echo_df.loc[echo_df["TR"]>=1, "mildTR"] = 1
# valve moderate or severe (1: moderate or severe)
echo_df.loc[echo_df["AR"]>=0, "modAR"] = 0
echo_df.loc[echo_df["AS"]>=0, "modAS"] = 0
echo_df.loc[echo_df["MR"]>=0, "modMR"] = 0
echo_df.loc[echo_df["TR"]>=0, "modTR"] = 0
echo_df.loc[echo_df["AR"]>=3, "modAR"] = 1
echo_df.loc[echo_df["AS"]>=2, "modAS"] = 1
echo_df.loc[echo_df["MR"]>=3, "modMR"] = 1
echo_df.loc[echo_df["TR"]>=3, "modTR"] = 1
# LVEF
echo_df['LVEDV'] = (7.0 / (2.4 + echo_df['LVDD'])) * echo_df['LVDD']**3
echo_df['LVESV'] = (7.0 / (2.4 + echo_df['LVDS'])) * echo_df['LVDS']**3
echo_df['LVEF'] = ((echo_df['LVEDV'] - echo_df['LVESV']) / echo_df['LVEDV']) * 100
echo_df.loc[echo_df['LVEF']<0,'LVEF']=np.nan # convert minus values to np.nan
# rEF
echo_df.loc[echo_df['LVEF']>=50, 'rEF'] = 0
echo_df.loc[echo_df['LVEF']<50, 'rEF'] = 1
# LVM Penn Convention (より正確とされる)
echo_df['LVM_Penn'] = 1.04 * ((
    echo_df['LVDD'] + echo_df['IVS'] + echo_df['LVPW'])**3 - echo_df['LVDD']**3) - 13.6
echo_df['LVMI'] = echo_df['LVM_Penn'] / echo_df['BSA']

# PHT　(TRG of >31 mmHg)
# https://www.sciencedirect.com/science/article/pii/S2589537021001024?via%3Dihub
echo_df.loc[echo_df['TR_PG']>31, 'PHT'] = 1
echo_df.loc[echo_df['TR_PG']<=31, 'PHT'] = 0

# Aortic Root (indexed to BSA): >2.1 cm/m²
echo_df.loc[echo_df['Ao_Sinus']/echo_df['BSA']>2.1, 'ab_Ao'] = 1
echo_df.loc[echo_df['Ao_Sinus']/echo_df['BSA']<=2.1, 'ab_Ao'] = 0

# Left Atrial Volume Index: >34 mL/m²
echo_df.loc[echo_df['LAV']/echo_df['BSA']>34, 'ab_LAVI'] = 1
echo_df.loc[echo_df['LAV']/echo_df['BSA']<=34, 'ab_LAVI'] = 0

# LV End-Diastolic Dimension: M >5.8, F >5.2 cm
echo_df.loc[(echo_df['LVDD']>5.8) & (echo_df['Sex']=='M'), 'ab_LVD'] = 1
echo_df.loc[(echo_df['LVDD']>5.2) & (echo_df['Sex']=='F'), 'ab_LVD'] = 1
echo_df.loc[(echo_df['LVDD']<=5.8) & (echo_df['Sex']=='M'), 'ab_LVD'] = 0
echo_df.loc[(echo_df['LVDD']<=5.2) & (echo_df['Sex']=='F'), 'ab_LVD'] = 0

# LV Mass Index: M >115, F >95 g/m²
echo_df.loc[(echo_df['LVMI']>115) & (echo_df['Sex']=='M'), 'ab_LVMI'] = 1
echo_df.loc[(echo_df['LVMI']>95) & (echo_df['Sex']=='F'), 'ab_LVMI'] = 1
echo_df.loc[(echo_df['LVMI']<=115) & (echo_df['Sex']=='M'), 'ab_LVMI'] = 0
echo_df.loc[(echo_df['LVMI']<=95) & (echo_df['Sex']=='F'), 'ab_LVMI'] = 0

# High_e_e
echo_df.loc[echo_df['MV_Ee']>15, 'ab_E_e'] = 1
echo_df.loc[echo_df['MV_Ee']<=15, 'ab_E_e'] = 0

# 5. Save echo measurements
echo_df[['subject_id', 'study_id', 'study_datetime', 
         'ab_Ao', 'ab_LVD', 'ab_LAVI', 'ab_LVMI', 'rEF', 
         'RV_func', 'mildAR', 'mildAS', 'mildMR', 'mildTR', 
         'modAR', 'modAS', 'modMR', 'modTR', 'PHT', 
         'ab_E_e', 'IVC', 'Echo_quality']].to_csv(
echo_measurement_path,index=False)

