import glob
import os
import numpy as np
import pandas as pd

orig_file_base = '/Volumes/SeaJade 2 Backup/NZ/GM_IM_4/'
pga_file_base = '/Volumes/SeaJade 2 Backup/NZ/GM_IM_4_pga/'
merged_out_base = '/Volumes/SeaJade 2 Backup/NZ/GM_IM_4_merged/'

orig_file_list = glob.glob(orig_file_base+'**/gm_all.csv',recursive=True)
pga_file_list = glob.glob(pga_file_base+'**/gm_all.csv',recursive=True)

orig_files = np.array([('/').join(file.split('/')[-4::]) for file in orig_file_list])
pga_files = np.array([('/').join(file.split('/')[-4::]) for file in pga_file_list])

for file in orig_files:
    try:
        pga_file = pga_file_base+pga_files[pga_files == file][0]
    except:
        print('No matching PGA file for '+file)
        continue
    orig_file = orig_file_base+file
    
    orig_df = pd.read_csv(orig_file)
    pga_df = pd.read_csv(pga_file)
    
    orig_df.drop(columns=['PGA','PGV'],inplace=True)
    merged_df = pga_df.set_index(pga_df['station']+pga_df['component']).join(orig_df.set_index(orig_df['station']+orig_df['component']).drop(columns=['station','component'])).reset_index(drop=True)
    
    out_dir = merged_out_base+file
    
    if not os.path.exists(os.path.dirname(out_dir)):
        os.makedirs(os.path.dirname(out_dir))
    merged_df.to_csv(out_dir,index=False)
    
    if len(orig_df) == len(pga_df):
        pass
    else:
        print(file,len(orig_df),len(pga_df))