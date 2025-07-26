from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import os, glob
import pydicom
from tqdm import tqdm

mimic_iv_echo_dir = # path to the "mimic-iv-echo/0.1" directory"
csv_dir = # path to the output directory
output_path = csv_dir + "/dicom_meta_long_format.csv"
error_log_path = csv_dir + "/dicom_processing_errors.csv"

dcm_paths = glob.glob(mimic_iv_echo_dir + "/*/*/*/*.dcm")
dcm_df = pd.DataFrame(dcm_paths,columns=['dcm_paths'])

error_records = []

# Initialize the file
with open(output_path, 'w') as f:
    f.write("dcm_path,index,tag,name,value\n")  # header

# Pre-calculate file sizes and filter
dcm_df['file_size'] = dcm_df['dcm_paths'].apply(os.path.getsize)
dcm_df = dcm_df[dcm_df['file_size'] <= 1000 * 1024 * 1024]

batch_size = 100
batch_results = []

def process_dicom(index):
    rows = []
    try:
        file_path = dcm_df['dcm_paths'][index]
        dataset = pydicom.dcmread(file_path, force=True, stop_before_pixels=True)
        for elem in dataset.iterall():
            try:
                tag = elem.tag
                name = elem.name
                value = getattr(elem, 'value', None)
                if isinstance(value, str) and len(value) > 1024:
                    value = value[:1024]
                rows.append({
                    'dcm_path': file_path,
                    'index': index,
                    'tag': str(tag),
                    'name': name,
                    'value': value
                })
            except Exception as e:
                error_records.append({
                    'file': file_path,
                    'tag': str(elem.tag),
                    'error': str(e)
                })
    except Exception as e:
        error_records.append({
            'file': file_path,
            'tag': None,
            'error': str(e)
        })
    return rows


# Run parallel processing
with ThreadPoolExecutor(max_workers=6) as executor:
    for result in tqdm(executor.map(process_dicom, dcm_df.index),
                       total=len(dcm_df.index), miniters=100):
        if result:
            batch_results.extend(result)
        if len(batch_results) >= batch_size:
            pd.DataFrame(batch_results).to_csv(output_path, mode='a', 
                                               header=False, index=False)
            batch_results = []

# Write remaining results in bulk
if batch_results:
    pd.DataFrame(batch_results).to_csv(output_path, mode='a', 
                                       header=False, index=False)

# Save errors
error_df = pd.DataFrame(error_records)
error_df.to_csv(error_log_path, index=False)

print(f"Saved long-format DICOM metadata to {output_path}")