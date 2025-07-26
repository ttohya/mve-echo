import numpy as np
import pandas as pd
import os


def load_mimic_data(data_path):
    """Load MIMIC-IV data"""
    patients = pd.read_csv(f'{data_path}/hosp/patients.csv.gz')
    admissions = pd.read_csv(f'{data_path}/hosp/admissions.csv.gz')
    return patients, admissions

def process_patients_data(patients):
    """Process patient data"""
    patients_processed = patients.copy()
    patients_processed['sex'] = patients_processed['gender'].map({"M": 0, "F": 1})
    return patients_processed

def create_race_mapping():
    return {
        # Hispanic
        'HISPANIC': 'Hispanic',
        'HISPANIC OR LATINO': 'Hispanic', 
        'HISPANIC/LATINO - PUERTO RICAN': 'Hispanic', 
        'HISPANIC/LATINO - DOMINICAN': 'Hispanic', 
        'HISPANIC/LATINO - GUATEMALAN': 'Hispanic', 
        'HISPANIC/LATINO - SALVADORAN': 'Hispanic', 
        'HISPANIC/LATINO - MEXICAN': 'Hispanic', 
        'HISPANIC/LATINO - COLUMBIAN': 'Hispanic', 
        'HISPANIC/LATINO - HONDURAN': 'Hispanic', 
        'HISPANIC/LATINO - CUBAN': 'Hispanic',
        'HISPANIC/LATINO - CENTRAL AMERICAN': 'Hispanic',
        'SOUTH AMERICAN': 'Hispanic',
        
        # Asian
        'ASIAN': 'Asian',
        'ASIAN - CHINESE': 'Asian',
        'ASIAN - SOUTH EAST ASIAN': 'Asian',
        'ASIAN - ASIAN INDIAN': 'Asian',
        'ASIAN - KOREAN': 'Asian',

        # White
        'WHITE': 'White',
        'WHITE - OTHER EUROPEAN': 'White',
        'WHITE - RUSSIAN': 'White',
        'WHITE - EASTERN EUROPEAN': 'White',
        'WHITE - BRAZILIAN': 'White',
        'PORTUGUESE': 'White',
        
        # Black
        'BLACK/AFRICAN AMERICAN': 'Black',
        'BLACK/CAPE VERDEAN': 'Black',
        'BLACK/CARIBBEAN ISLAND': 'Black',
        'BLACK/AFRICAN': 'Black',
        
        # Other/Unknown
        'AMERICAN INDIAN/ALASKA NATIVE': 'Other',
        'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 'Other',
        'MULTIPLE RACE/ETHNICITY': 'Other',
        'UNKNOWN': 'Unknown',
        'OTHER': 'Other',
        'PATIENT DECLINED TO ANSWER': 'Unknown',
        'UNABLE TO OBTAIN': 'Unknown',

    }

def process_admissions_data(admissions):
    """Process admissions data"""
    # Keep only the latest admission information
    admission_sel = (admissions[['subject_id', 'hadm_id', 'language', 'race', 'insurance']]
                    .drop_duplicates(subset='subject_id', keep='last')
                    [['subject_id', 'language', 'race', 'insurance']])
    
    # Consolidate race categories
    race_mapping = create_race_mapping()
    admission_sel['race2'] = admission_sel['race'].map(race_mapping)
    
    # Process language categories
    admission_sel['language2'] = np.where(
        admission_sel['language'] == 'English', 
        'English', 
        'Non-english'
    )
    
    # Process insurance categories
    insurance_types = ['Medicaid', 'Medicare', 'Private']
    for insurance_type in insurance_types:
        admission_sel[insurance_type] = np.where(
            admission_sel['insurance'] == insurance_type,
            insurance_type,
            f'Non-{insurance_type.lower()}'
        )
    
    return admission_sel


def create_final_demographics(patients, admissions, output_path):
    """Create final demographics dataset"""
    # Merge datasets
    mimic_demog = pd.merge(patients, admissions, how='outer', on='subject_id')
    
    # Select required columns only
    demographic_cols = ['subject_id', 'gender', 'language2', 'race2', 'Medicaid', 'Private', 'Medicare']
    mimic_demog = mimic_demog[demographic_cols]
    
    # Rename columns
    column_mapping = {
        'language2': 'Language',
        'race2': 'Race',
        'gender': 'Sex'
    }
    mimic_demog = mimic_demog.rename(columns=column_mapping)
    
    # Save results
    output_cols = ['subject_id', 'Language', 'Sex', 'Race', 'Medicaid', 'Private', 'Medicare']
    mimic_demog[output_cols].to_csv(f"{output_path}/mimic_demog.csv", index=False)
    
    return mimic_demog

def print_summary_statistics(admissions_processed):
    """Display summary statistics"""
    print("=== Summary Statistics ===")
    categorical_vars = ['race2', 'language2', 'Medicaid', 'Private', 'Medicare']
    
    for var in categorical_vars:
        if var in admissions_processed.columns:
            print(f"\n{var} distribution:")
            print(admissions_processed[var].value_counts())

# Main processing
def main():
    # Path configuration
    mimic_path = # path to the mimic-iv-3.1 data
    csv_dir = # path to the csv directory
       
    # Load data
    print("Loading MIMIC-IV data...")
    patients, admissions = load_mimic_data(mimic_path)
    
    # Process data
    print("Processing patient data...")
    patients_processed = process_patients_data(patients)
    
    print("Processing admissions data...")
    admissions_processed = process_admissions_data(admissions)
        
    # Create final dataset
    print("Creating final demographics dataset...")
    final_demog = create_final_demographics(
        patients_processed, 
        admissions_processed, 
        csv_dir
    )
    
    # Display summary statistics
    print_summary_statistics(admissions_processed)
    
    print(f"\nFinal dataset shape: {final_demog.shape}")
    print(f"Unique subjects with video data: {final_demog['subject_id'].nunique()}")
    print(f"Output saved to: {csv_dir}/mimic_demog.csv")

if __name__ == "__main__":
    main()


