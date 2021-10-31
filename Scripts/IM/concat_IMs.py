import pandas as pd
import os

directory = '/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/testaroo/'

gm_im_df_6 = pd.read_csv(directory+'IM_catalogue/ground_motion_im_catalogue_6_final.csv',low_memory=False)
gm_im_df_5 = pd.read_csv(directory+'IM_catalogue/ground_motion_im_catalogue_5_final.csv',low_memory=False)
gm_im_df_45 = pd.read_csv(directory+'IM_catalogue/ground_motion_im_catalogue_45_final.csv',low_memory=False)
gm_im_df_4 = pd.read_csv(directory+'IM_catalogue/ground_motion_im_catalogue_4_final.csv',low_memory=False)
gm_im_dfs = pd.concat((gm_im_df_6,gm_im_df_5,gm_im_df_45,gm_im_df_4))
# gm_im_df = pd.read_csv(directory+'IM_catalogue/ground_motion_im_catalogue_final.csv',low_memory=False)

# remove_ids = gm_im_dfs.evid.unique()
# 
# gm_im_df = gm_im_df[gm_im_df.evid.isin(remove_ids) == False]
# merged_df = pd.concat([gm_im_df,gm_im_dfs])

gm_im_dfs.to_csv(directory+'IM_catalogue/ground_motion_im_catalogue_merged_final.csv',index=False)