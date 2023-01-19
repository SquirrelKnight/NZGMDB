import pandas as pd
import os
import glob

directory = '/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/IM_catalogue'

files = glob.glob(directory+'/ground_motion*final.csv') # Check to make sure this doesn't merge in an already merged file

gm_im_dfs = pd.concat([pd.read_csv(file, low_memory=False) for file in files])
gm_im_dfs.sort_values('datetime',inplace=True)

# Excise scores that made it through the process...
gm_im_dfs = gm_im_dfs[(gm_im_dfs.score_mean_X >= 0.5) & (gm_im_dfs.score_mean_Y >= 0.5) & (gm_im_dfs.score_mean_Z >= 0.5)]

gm_im_dfs.reset_index(drop=True)

gm_im_dfs.to_csv(directory+'/complete_ground_motion_im_catalogue_final.csv',index=False)