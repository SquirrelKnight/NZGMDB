import pandas as pd

df = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/IM_catalogue/Tables/ground_motion_im_catalogue_final_expanded.csv',low_memory=False)

df[['gmid', 'evid', 'datetime', 'sta', 'loc', 'chan', 'component',
    	'ev_lat', 'ev_lon', 'ev_depth', 'mag', 'tect_type', 'reloc',
        'sta_lat', 'sta_lon','rrup', 'rjb','Ztor', 'Vs30', 'Tsite', 'Z1.0', 'Z2.5',
        'PGA', 'PGV', 'CAV', 'AI', 'Ds575', 'Ds595', 'MMI', 'pSA_0.02', 'pSA_0.05',
        'pSA_0.1', 'pSA_0.2', 'pSA_0.3', 'pSA_0.4', 'pSA_0.5', 'pSA_0.75',
        'pSA_1.0', 'pSA_2.0', 'pSA_3.0', 'pSA_4.0', 'pSA_5.0', 'pSA_7.5',
        'pSA_10.0', 'score_X', 'f_min_X', 'score_Y', 'f_min_Y', 'score_Z', 'f_min_Z']]

df_000 = df[df.component == '000']
df_090 = df[df.component == '090']
df_ver = df[df.component == 'ver']

df_000 = df_000.drop(['score_X','f_min_X','score_Z','f_min_Z'],axis=1)
df_090 = df_090.drop(['score_Y','f_min_Y','score_Z','f_min_Z'],axis=1)
df_ver = df_ver.drop(['score_X','f_min_X','score_Y','f_min_Y'],axis=1)

df_000.to_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/IM_catalogue/Tables/ground_motion_im_catalogue_final_000.csv',index=False)
df_090.to_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/IM_catalogue/Tables/ground_motion_im_catalogue_final_090.csv',index=False)
df_ver.to_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/IM_catalogue/Tables/ground_motion_im_catalogue_final_ver.csv',index=False)