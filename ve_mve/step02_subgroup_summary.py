import numpy as np
import pandas as pd
from tableone import TableOne

ds_dir = # dataset dir which includes "ve_echoprime.csv" and "echo_ds.csv")
results_dir = # path to the dir for the results output

echo_ds = pd.read_csv(ds_dir + "/echo_ds.csv", encoding='utf_8')
echo_ds_uniq = echo_ds.copy().drop(["dicom_path",'view'],axis=1).drop_duplicates(subset='study_id')

echo_ds_uniq['Race'] = echo_ds_uniq['Race'].replace({'Hispanic':'Other','Asian':'Other','Unknown':np.nan})
echo_ds_uniq['Sex'] = echo_ds_uniq['Sex'].replace({'F':'Female','M':'Male'})
label_map = {
    'mildAS': 'Mild AS or greater',
    'RV_func' : 'Impaired RV Function',
    'ab_E_e': "E/e' > 15",
    'rEF': 'LVEF < 50%',
}

# function for table one
def tableone_create(df, groupby_column, result_path, ):
    categorical_columns = ['mildAS','RV_func','ab_E_e','rEF']
    continuous_columns = []
    decimals_list = {}

    limit_list = {
    'mildAS':1,
    'RV_func':1,
    'ab_E_e':1,
    'rEF':1,
    }


    order_list = {
                'mildAS':["1.0","0.0"],
                'RV_func':["1.0","0.0"],
                'ab_E_e':["1.0","0.0"],
                'rEF':["1.0","0.0"],
                }

    columns = [ 'mildAS','rEF','RV_func','ab_E_e']

    mytable = TableOne(df, columns=columns,
                        categorical=categorical_columns,             
                        groupby=groupby_column, 
                        nonnormal=[], 
                        decimals = decimals_list,
                        limit = limit_list,
                        order = order_list,
                        rename = label_map,
                        pval=False,
                        include_null =False,)

    print(mytable.tabulate(tablefmt="git_hub")) 
    mytable.tableone.to_csv(result_path)

tableone_create(
  echo_ds_uniq, 
  'Sex',
  results_dir + "/subgroup_sex.csv")

tableone_create(
  echo_ds_uniq, 
  'Race',
  results_dir + "/subgroup_Race.csv")
