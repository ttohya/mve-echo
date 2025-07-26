import pandas as pd
import torch
import os
import numpy as np
from tqdm import tqdm

device=torch.device("cuda")

csv_dir = # path to the csv data directory
dataset_dir = # path to the dataset directory
MIL_weights_file_path = # model file path to the MIL_weights.csv
all_ve_echoprime_512 = pd.read_csv(dataset_dir + "/ve_echoprime.csv")

view_predictions = pd.read_csv(csv_dir + "/view_predictions.csv")
all_ve = pd.merge(view_predictions,all_ve_echoprime_512,how='inner',on='path')
all_ve['study_id'] = all_ve['path'].apply(lambda x: os.path.basename(os.path.dirname(x)))
study_list = all_ve['study_id'].unique().tolist()

# load MIL weights
MIL_weights = pd.read_csv(MIL_weights_file_path)
non_empty_sections=MIL_weights['Section']
section_weights=MIL_weights.iloc[:,1:].to_numpy()

COARSE_VIEWS=['A2C',
 'A3C',
 'A4C',
 'A5C',
 'Apical_Doppler',
 'Doppler_Parasternal_Long',
 'Doppler_Parasternal_Short',
 'Parasternal_Long',
 'Parasternal_Short',
 'SSN',
 'Subcostal']

view_to_index = {view: idx for idx, view in enumerate(COARSE_VIEWS)}



def predict_metrics2(sel_ve):
    """
    study_embedding is a set of embeddings of all videos from the study e.g (52,512)
    Takes a study embedding as input and
    outputs a dictionary for a set of 26 features
    """
    view_indices = sel_ve["view"].map(view_to_index).values 
    out_views = torch.tensor(view_indices, dtype=torch.long)
    
    # print( torch.stack([torch.nn.functional.one_hot(out_views,11)]).shape)
    
    stack_of_view_encodings = torch.stack([torch.nn.functional.one_hot(out_views,11)]).squeeze(dim=0)
    
    ve_columns = [f"ve{str(i+1).zfill(3)}" for i in range(512)]
    stack_of_features = torch.tensor(sel_ve[ve_columns].values, dtype=torch.long)
    
    # print(stack_of_features.shape, stack_of_view_encodings.shape)
    study_embedding =  torch.cat( (stack_of_features ,stack_of_view_encodings),dim=1)

    #per_section_study_embedding has shape (15,512)
    per_section_study_embedding=torch.zeros(len(non_empty_sections),512)
    study_embedding=study_embedding.cpu()
    # make per section study embedding
    for s_dx, sec in enumerate(non_empty_sections):
        # get section weights
        this_section_weights=[section_weights[s_dx][torch.where(view_encoding==1)[0]]
                        for view_encoding in study_embedding[:,512:]]
        this_section_study_embedding = (study_embedding[:,:512] * \
                                        torch.tensor(this_section_weights,
                                                        dtype=torch.float).unsqueeze(1))
        
        #weighted average
        this_section_study_embedding=torch.sum(this_section_study_embedding,dim=0)
        per_section_study_embedding[s_dx]=this_section_study_embedding
        
    per_section_study_embedding=torch.nn.functional.normalize(per_section_study_embedding)
    
    return per_section_study_embedding


study_ve_raw = np.zeros((len(study_list), 15, 512))
valid_indices = []
valid_ids = []
failed_ids = []

for i, study in tqdm(enumerate(study_list)):
    sel_ve = all_ve.copy()[all_ve['study_id'] == study]
    try:
        study_ve_raw[i, :, :] = predict_metrics2(sel_ve)
        valid_indices.append(i)
        valid_ids.append(study) 
    except Exception as e:
        print(f"Error with study_id: {study}, skipped. Reason: {str(e)}")
        failed_ids.append(study)



study_ve_valid = study_ve_raw[valid_indices].reshape(len(valid_indices), -1)

# columns：ve0001 〜 ve7680
ve_columns = [f"ve{str(i+1).zfill(4)}" for i in range(15 * 512)]

study_ve_df = pd.DataFrame(study_ve_valid, columns=ve_columns)
study_ve_df.insert(0, "study_id", valid_ids)

study_ve_df.to_csv(dataset_dir + "/all_ve_echoprime_7680.csv",index=False)