import pandas as pd

matching_file_path = # path to the matching file for mimic-iv echo measurements with dicom files
echo_measurements_path = # path to the echo measurements file
csv_dir = # path to the output directory
echo_mm_matchingfile_path = csv_dir + "/echo_mm_matchingfile.csv"

echo_dicom_matched_studies = pd.read_csv(matching_file_path)
mm_study_list = echo_dicom_matched_studies[~echo_dicom_matched_studies["mm_study_id"].isna()]["mm_study_id"].astype('int').to_list()

echo_dicom_matched_studies_sel = echo_dicom_matched_studies[["dcm_study_id",'mm_study_id']]

echo_mm_long = pd.read_csv(echo_measurements_path)

echo_mm_long = echo_mm_long.merge(echo_dicom_matched_studies_sel,how='inner',left_on='study_id',right_on='mm_study_id')

echo_mm_long = echo_mm_long.drop(['mm_study_id','study_id'],axis=1).rename(columns={'dcm_study_id':'study_id'})

echo_mm_long.to_csv(echo_mm_matchingfile_path,index=False)