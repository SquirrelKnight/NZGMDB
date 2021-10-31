import pandas as pd
import os
import numpy as np

# Searches for unique events based on calculated GM IMs
directory = '/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/'

event_cat = pd.read_csv(directory+'earthquake_source_table_complete.csv',low_memory=False)
event_cat['datetime'] = pd.to_datetime(event_cat.datetime).astype('datetime64[ns]')

mag_df = pd.read_csv(directory+'station_magnitude_table_relocated.csv',low_memory=False)

arr_df = pd.read_csv(directory+'phase_arrival_table.csv',low_memory=False)
arr_df['datetime'] = pd.to_datetime(arr_df.datetime).astype('datetime64[ns]')

prop_df = pd.read_csv(directory+'propagation_path_table_complete.csv',low_memory=False)
# prop_df['rrup'] = prop_df.r_hyp
# prop_df['rjb'] = prop_df.r_epi

sta_df = pd.read_csv(directory+'site_table_basin.csv',low_memory=False)
sta_df = sta_df.drop_duplicates() # Just in case any data is duplicated

gm_im_df = pd.read_csv(directory+'IM_catalogue/ground_motion_im_catalogue_final.csv',low_memory=False)

unique_events = gm_im_df.evid.unique()

# Subsets unique events
event_cat_sub = event_cat[event_cat['evid'].isin(unique_events)].reset_index(drop=True)
event_cat_sub.drop_duplicates(subset='evid',inplace=True)
mag_df_sub = mag_df[mag_df['evid'].isin(unique_events)]
arr_df_sub = arr_df[arr_df['evid'].isin(unique_events)]
prop_df_sub = prop_df[prop_df['evid'].isin(unique_events)]
prop_df_sub = prop_df_sub[prop_df_sub[['evid','net','sta']].duplicated() == False]
# The below searches for AU ARPS values and removes them. There are two stations with the same site name.
indexNames = prop_df_sub[ (prop_df_sub['net'] == 'AU') & (prop_df_sub['sta'] == 'ARPS') ].index
prop_df_sub.drop(indexNames , inplace=True)
# prop_df_sub = prop_df_sub[prop_df_sub[['evid','sta']].duplicated() == True] # In case there is a duplicated station, this happens with ARPS [AU and NZ]
unique_stas = np.unique(np.append(gm_im_df['sta'].unique(),arr_df_sub['sta'].unique()))
station_sub = sta_df[sta_df['sta'].isin(unique_stas)]
directory = '/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/IM_catalogue/Tables/'


### Work on adding new parameters to the gm_im table
gm_im_df_flat = gm_im_df.copy()
gm_im_df_flat['ev_lat'] = gm_im_df_flat['evid'].map(event_cat_sub.set_index('evid')['lat'])
gm_im_df_flat['ev_lon'] = gm_im_df_flat['evid'].map(event_cat_sub.set_index('evid')['lon'])
gm_im_df_flat['ev_depth'] = gm_im_df_flat['evid'].map(event_cat_sub.set_index('evid')['depth'])
gm_im_df_flat['mag'] = gm_im_df_flat['evid'].map(event_cat_sub.set_index('evid')['mag'])
gm_im_df_flat['mag_type'] = gm_im_df_flat['evid'].map(event_cat_sub.set_index('evid')['mag_type'])
gm_im_df_flat['tect_class'] = gm_im_df_flat['evid'].map(event_cat_sub.set_index('evid')['tect_class'])
gm_im_df_flat['reloc'] = gm_im_df_flat['evid'].map(event_cat_sub.set_index('evid')['reloc'])
gm_im_df_flat['domain_no'] = gm_im_df_flat['evid'].map(event_cat_sub.set_index('evid')['domain_no'])
gm_im_df_flat['domain_type'] = gm_im_df_flat['evid'].map(event_cat_sub.set_index('evid')['domain_type'])
gm_im_df_flat['strike'] =  gm_im_df_flat['evid'].map(event_cat_sub.set_index('evid')['strike'])
gm_im_df_flat['dip'] =  gm_im_df_flat['evid'].map(event_cat_sub.set_index('evid')['dip'])
gm_im_df_flat['rake'] =  gm_im_df_flat['evid'].map(event_cat_sub.set_index('evid')['rake'])
gm_im_df_flat['f_length'] =  gm_im_df_flat['evid'].map(event_cat_sub.set_index('evid')['f_length'])
gm_im_df_flat['f_width'] =  gm_im_df_flat['evid'].map(event_cat_sub.set_index('evid')['f_width'])
gm_im_df_flat['f_type'] =  gm_im_df_flat['evid'].map(event_cat_sub.set_index('evid')['f_type'])
gm_im_df_flat['z_tor'] =  gm_im_df_flat['evid'].map(event_cat_sub.set_index('evid')['z_tor'])
gm_im_df_flat['z_bor'] =  gm_im_df_flat['evid'].map(event_cat_sub.set_index('evid')['z_bor'])
gm_im_df_flat['sta_lat'] = gm_im_df_flat['sta'].map(sta_df.set_index('sta')['lat'])
gm_im_df_flat['sta_lon'] = gm_im_df_flat['sta'].map(sta_df.set_index('sta')['lon'])
gm_im_df_flat['Vs30'] = gm_im_df_flat['sta'].map(sta_df.set_index('sta')['Vs30'])
gm_im_df_flat['Vs30_std'] = gm_im_df_flat['sta'].map(sta_df.set_index('sta')['Vs30_std'])
gm_im_df_flat['Q_Vs30'] = gm_im_df_flat['sta'].map(sta_df.set_index('sta')['Q_Vs30'])
# gm_im_df_flat['Vs30_ref'] = gm_im_df_flat['sta'].map(sta_df.set_index('sta')['Vs30_ref'])
gm_im_df_flat['Tsite'] = gm_im_df_flat['sta'].map(sta_df.set_index('sta')['Tsite'])
gm_im_df_flat['Tsite_std'] = gm_im_df_flat['sta'].map(sta_df.set_index('sta')['Tsite_std'])
gm_im_df_flat['Q_Tsite'] = gm_im_df_flat['sta'].map(sta_df.set_index('sta')['Q_Tsite'])
gm_im_df_flat['Z1.0'] = gm_im_df_flat['sta'].map(sta_df.set_index('sta')['Z1.0'])
gm_im_df_flat['Z1.0_std'] = gm_im_df_flat['sta'].map(sta_df.set_index('sta')['Z1.0_std'])
gm_im_df_flat['Q_Z1.0'] = gm_im_df_flat['sta'].map(sta_df.set_index('sta')['Q_Z1.0'])
gm_im_df_flat['Z2.5'] = gm_im_df_flat['sta'].map(sta_df.set_index('sta')['Z2.5'])
gm_im_df_flat['Z2.5_std'] = gm_im_df_flat['sta'].map(sta_df.set_index('sta')['Z2.5_std'])
gm_im_df_flat['Q_Z2.5'] = gm_im_df_flat['sta'].map(sta_df.set_index('sta')['Q_Z2.5'])
# gm_im_df_flat['Z1.0_NZVM'] = gm_im_df_flat['sta'].map(sta_df.set_index('sta')['Z1.0_NZVM'])
# gm_im_df_flat['Z2.5_NZVM'] = gm_im_df_flat['sta'].map(sta_df.set_index('sta')['Z2.5_NZVM'])
gm_im_df_flat['site_domain_no'] = gm_im_df_flat['sta'].map(sta_df.set_index('sta')['site_domain_no'])
gm_im_df_flat['r_epi'] = (gm_im_df_flat['evid']+gm_im_df['sta']).map(prop_df_sub.set_index(prop_df_sub['evid']+prop_df_sub['sta'])['r_epi'])
gm_im_df_flat['r_hyp'] = (gm_im_df_flat['evid']+gm_im_df['sta']).map(prop_df_sub.set_index(prop_df_sub['evid']+prop_df_sub['sta'])['r_hyp'])
gm_im_df_flat['r_jb'] = (gm_im_df_flat['evid']+gm_im_df['sta']).map(prop_df_sub.set_index(prop_df_sub['evid']+prop_df_sub['sta'])['r_jb'])
gm_im_df_flat['r_rup'] = (gm_im_df_flat['evid']+gm_im_df['sta']).map(prop_df_sub.set_index(prop_df_sub['evid']+prop_df_sub['sta'])['r_rup'])
gm_im_df_flat['r_x'] = (gm_im_df_flat['evid']+gm_im_df['sta']).map(prop_df_sub.set_index(prop_df_sub['evid']+prop_df_sub['sta'])['r_x'])
gm_im_df_flat['r_y'] = (gm_im_df_flat['evid']+gm_im_df['sta']).map(prop_df_sub.set_index(prop_df_sub['evid']+prop_df_sub['sta'])['r_y'])
gm_im_df_flat['r_tvz'] = (gm_im_df_flat['evid']+gm_im_df['sta']).map(prop_df_sub.set_index(prop_df_sub['evid']+prop_df_sub['sta'])['r_tvz'])

# Remove events in the flatfile from far outside of NZ
gm_im_df_flat_sub = gm_im_df_flat.copy()
gm_im_df_flat_sub.loc[gm_im_df_flat_sub.ev_lon < 0, 'ev_lon'] = 360 + gm_im_df_flat_sub.ev_lon[gm_im_df_flat_sub.ev_lon < 0]
gm_im_df_flat_sub = gm_im_df_flat_sub[(gm_im_df_flat_sub.ev_lon < 190) & (gm_im_df_flat_sub.ev_lon >155)]
gm_im_df_flat_sub = gm_im_df_flat_sub[(gm_im_df_flat_sub.ev_lat < -15)]
events_in = gm_im_df_flat_sub.evid.astype('str').values
gm_im_df_flat = gm_im_df_flat[gm_im_df_flat.evid.isin(events_in)]


# gm_im_df_flat = gm_im_df_flat[['gmid', 'evid', 'datetime', 'sta', 'loc', 'chan', 'component',
#     	'ev_lat', 'ev_lon', 'ev_depth', 'mag', 'mag_type', 'tect_class', 'domain_no', 
#     	'domain_type', 'strike', 'dip', 'rake', 'f_length', 'f_width', 'f_type', 'z_tor', 
#     	'z_bor','reloc','sta_lat', 'sta_lon', 'r_epi', 'r_hyp', 'r_jb', 'r_rup', 'r_x', 
#     	'r_y', 'r_tvz', 'Vs30_preferred', 'Vs30_preferred_model', 'Tsite', 'Z1.0', 'Z2.5', 
#     	'Z_preferred_model', 'Q_Vs30', 'Q_Z1.0', 'Q_Z2.5', 'Q_Tsite', 'site_domain_no', 
#     	'PGA', 'PGV', 'CAV', 'AI', 'Ds575', 'Ds595', 'MMI', 
#     	'pSA_0.01', 'pSA_0.02', 'pSA_0.03', 'pSA_0.04', 'pSA_0.05', 'pSA_0.075', 'pSA_0.1', 
#     	'pSA_0.12', 'pSA_0.15', 'pSA_0.17', 'pSA_0.2', 'pSA_0.25', 'pSA_0.3', 'pSA_0.4', 
#     	'pSA_0.5', 'pSA_0.6', 'pSA_0.7', 'pSA_0.75', 'pSA_0.8', 'pSA_0.9', 'pSA_1.0', 
#     	'pSA_1.25', 'pSA_1.5', 'pSA_2.0', 'pSA_2.5', 'pSA_3.0', 'pSA_4.0', 'pSA_5.0', 
#     	'pSA_6.0', 'pSA_7.5', 'pSA_10.0', 'score_X', 'f_min_X', 'score_Y', 'f_min_Y', 
#     	'score_Z', 'f_min_Z']]
gm_im_df_flat = gm_im_df_flat[['gmid', 'datetime', 'evid', 'sta', 'loc', 'chan', 'component', 'ev_lat',
       'ev_lon', 'ev_depth', 'mag', 'mag_type', 'tect_class', 'reloc',
       'domain_no', 'domain_type', 'strike', 'dip', 'rake', 'f_length',
       'f_width', 'f_type', 'z_tor', 'z_bor', 'sta_lat', 'sta_lon',
       'r_epi', 'r_hyp', 'r_jb', 'r_rup', 'r_x', 'r_y', 'r_tvz', 'Vs30',
       'Vs30_std', 'Q_Vs30', 'Tsite', 'Tsite_std', 'Q_Tsite', 'Z1.0',
       'Z1.0_std', 'Q_Z1.0', 'Z2.5', 'Z2.5_std', 'Q_Z2.5', 'site_domain_no','PGA',
       'PGV', 'CAV', 'AI', 'Ds575', 'Ds595', 'MMI', 'pSA_0.01', 'pSA_0.02',
       'pSA_0.03', 'pSA_0.04', 'pSA_0.05', 'pSA_0.075', 'pSA_0.1', 'pSA_0.12',
       'pSA_0.15', 'pSA_0.17', 'pSA_0.2', 'pSA_0.25', 'pSA_0.3', 'pSA_0.4',
       'pSA_0.5', 'pSA_0.6', 'pSA_0.7', 'pSA_0.75', 'pSA_0.8', 'pSA_0.9',
       'pSA_1.0', 'pSA_1.25', 'pSA_1.5', 'pSA_2.0', 'pSA_2.5', 'pSA_3.0',
       'pSA_4.0', 'pSA_5.0', 'pSA_6.0', 'pSA_7.5', 'pSA_10.0', 'score_X',
       'f_min_X', 'score_Y', 'f_min_Y', 'score_Z', 'f_min_Z']]
# Find null Vs30 values and infill with Foster hybrid data    
# gm_im_df_flat.loc[gm_im_df_flat.Vs30.isnull(),'Vs30'] = gm_im_df_flat[gm_im_df_flat.Vs30.isnull()].join(station_sub[['sta','Vs30_foster_hybrid']].set_index('sta'),on='sta',how='left').Vs30_foster_hybrid.values

# Separate GM IM catalogues into separate tables        
df_000 = gm_im_df[gm_im_df.component == '000']
df_090 = gm_im_df[gm_im_df.component == '090']
df_ver = gm_im_df[gm_im_df.component == 'ver']
df_rotd50 = gm_im_df[gm_im_df.component == 'rotd50']
df_rotd100 = gm_im_df[gm_im_df.component == 'rotd100']

df_000 = df_000.drop(['score_X','f_min_X','score_Z','f_min_Z'],axis=1)
df_090 = df_090.drop(['score_Y','f_min_Y','score_Z','f_min_Z'],axis=1)
df_ver = df_ver.drop(['score_X','f_min_X','score_Y','f_min_Y'],axis=1)
df_rotd50 = df_rotd50.drop(['score_Z','f_min_Z'],axis=1)
df_rotd100 = df_rotd100.drop(['score_Z','f_min_Z'],axis=1)

df_000_flat = gm_im_df_flat[gm_im_df_flat.component == '000']
df_090_flat = gm_im_df_flat[gm_im_df_flat.component == '090']
df_ver_flat = gm_im_df_flat[gm_im_df_flat.component == 'ver']
df_rotd50_flat = gm_im_df_flat[gm_im_df_flat.component == 'rotd50']
df_rotd100_flat = gm_im_df_flat[gm_im_df_flat.component == 'rotd100']

df_000_flat = df_000_flat.drop(['score_X','f_min_X','score_Z','f_min_Z'],axis=1)
df_090_flat = df_090_flat.drop(['score_Y','f_min_Y','score_Z','f_min_Z'],axis=1)
df_ver_flat = df_ver_flat.drop(['score_X','f_min_X','score_Y','f_min_Y'],axis=1)
df_rotd50_flat = df_rotd50_flat.drop(['score_Z','f_min_Z'],axis=1)
df_rotd100_flat = df_rotd100_flat.drop(['score_Z','f_min_Z'],axis=1)

###

if not os.path.exists(directory):
	os.makedirs(directory)

# Writes subset data to new csv files
event_cat_sub.to_csv(directory+'earthquake_source_table.csv',index=False)
mag_df_sub.to_csv(directory+'station_magnitude_table.csv',index=False)
arr_df_sub.to_csv(directory+'phase_arrival_table.csv',index=False)
prop_df_sub.to_csv(directory+'propagation_path_table.csv',index=False)
station_sub.to_csv(directory+'site_table.csv',index=False)
# gm_im_df.to_csv(directory+'ground_motion_im_catalogue_final_expanded.csv',index=False)
df_000.to_csv(directory+'ground_motion_im_table_000.csv',index=False)
df_090.to_csv(directory+'ground_motion_im_table_090.csv',index=False)
df_ver.to_csv(directory+'ground_motion_im_table_ver.csv',index=False)
df_rotd50.to_csv(directory+'ground_motion_im_table_rotd50.csv',index=False)
df_rotd100.to_csv(directory+'ground_motion_im_table_rotd100.csv',index=False)
df_000_flat.to_csv(directory+'ground_motion_im_table_000_flat.csv',index=False)
df_090_flat.to_csv(directory+'ground_motion_im_table_090_flat.csv',index=False)
df_ver_flat.to_csv(directory+'ground_motion_im_table_ver_flat.csv',index=False)
df_rotd50_flat.to_csv(directory+'ground_motion_im_table_rotd50_flat.csv',index=False)
df_rotd100_flat.to_csv(directory+'ground_motion_im_table_rotd100_flat.csv',index=False)


def merge_reloc():
    # Load original catalogues and relocated catologues to merge the data and then rewrite
    # to new files.
    directory = '/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/testaroo/'

    event_cat = pd.read_csv(directory+'earthquake_source_table.csv',low_memory=False)
    event_cat['datetime'] = pd.to_datetime(event_cat.datetime).astype('datetime64[ns]')
#     event_cat['reloc'] = 'no'

    mag_df = pd.read_csv(directory+'station_magnitude_table.csv',low_memory=False)
    mag_df['reloc'] = 'no'

#     sta_df = pd.read_csv(directory+'site_table.csv',low_memory=False)
# 
#     arr_df = pd.read_csv(directory+'phase_arrival_table.csv',low_memory=False)
# 
#     prop_df = pd.read_csv(directory+'propagation_path_table.csv',low_memory=False)
#     prop_df['reloc'] = 'no'

    # gm_im_df = pd.read_csv('ground_motion_im_catalogue_final.csv',low_memory=False)

    relocated_event_cat = pd.read_csv(directory+'relocated_earthquake_source_table.csv',low_memory=False)
    relocated_event_cat['datetime'] = pd.to_datetime(relocated_event_cat.datetime).astype('datetime64[ns]')
    relocated_event_cat['reloc'] = 'reyners'
    relocated_event_cat['evid'] = relocated_event_cat.evid.astype(str)
    relocated_event_cat.drop_duplicates(subset=['evid'],keep='first',inplace=True) 

    relocated_mag_df = pd.read_csv(directory+'relocated_station_magnitude_table.csv',low_memory=False)
    relocated_mag_df['reloc'] = 'reyners'
    relocated_mag_df['evid'] = relocated_mag_df.evid.astype(str)
    relocated_mag_df.drop_duplicates(subset=['magid'],keep='first',inplace=True)

    ISC_relocated_event_cat = pd.read_csv(directory+'ISC_relocated_earthquake_source_table.csv',low_memory=False)
    ISC_relocated_event_cat['datetime'] = pd.to_datetime(ISC_relocated_event_cat.datetime).astype('datetime64[ns]')
    ISC_relocated_event_cat['evid'] = ISC_relocated_event_cat.evid.astype(str)
    ISC_relocated_event_cat.drop_duplicates(subset=['evid'],keep='first',inplace=True) 

    ISC_relocated_mag_df = pd.read_csv(directory+'ISC_relocated_station_magnitude_table.csv',low_memory=False)
    ISC_relocated_mag_df['evid'] = ISC_relocated_mag_df.evid.astype(str)
    ISC_relocated_mag_df.drop_duplicates(subset=['magid'],keep='first',inplace=True)

#     relocated_prop_df = pd.read_csv(directory+'relocated_propagation_path_table.csv',low_memory=False)
#     relocated_prop_df['reloc'] = 'yes'
#     relocated_prop_df['evid'] = relocated_prop_df.evid.astype(str)
#     relocated_prop_df.drop_duplicates(subset=['evid','net','sta'],keep='first',inplace=True)

    event_cat.set_index(['evid'],inplace=True)
    relocated_event_cat.set_index(['evid'],inplace=True)
    event_cat.update(relocated_event_cat,overwrite=True)
    event_cat.reset_index(inplace=True)

    mag_df.set_index(['magid'], inplace=True)
    relocated_mag_df.set_index(['magid'], inplace=True)
    mag_df.update(relocated_mag_df)
    mag_df.reset_index(inplace=True)

    event_cat.set_index(['evid'],inplace=True)
    ISC_relocated_event_cat.set_index(['evid'],inplace=True)
    event_cat.update(ISC_relocated_event_cat,overwrite=True)
    event_cat.reset_index(inplace=True)

    mag_df.set_index(['magid'], inplace=True)
    ISC_relocated_mag_df.set_index(['magid'], inplace=True)
    mag_df.update(ISC_relocated_mag_df)
    mag_df.reset_index(inplace=True)

#     prop_df.set_index(['evid','net','sta'], inplace=True)
#     relocated_prop_df.set_index(['evid','net','sta'], inplace=True)
#     prop_df.update(relocated_prop_df)
#     prop_df.reset_index(inplace=True)

    # Outputs data combined with relocations to file
#     directory = '/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/'

    event_cat.to_csv(directory+'earthquake_source_table_relocated.csv',index=False)
    mag_df.to_csv(directory+'station_magnitude_table_relocated.csv',index=False)
#     arr_df.to_csv(directory+'phase_arrival_table_relocated.csv',index=False)
#     prop_df.to_csv(directory+'propagation_path_table_relocated.csv',index=False)


# Find all duplicated events in the relocated catalogue and search for the nearest datetime
# in the event catalogue. Write the ID to a list. Relabel duplicated IDs next!

def relocation_dups(event_df,event_cat):

	event_df['evid'] = event_df.evid.astype(str)
	event_df['datetime'] = pd.to_datetime(event_df.datetime)

	event_cat['datetime'] = pd.to_datetime(event_cat.datetime).astype('datetime64[ns]')

	all_dups = event_df[event_df.evid.duplicated(keep=False)]
	all_dups_datetimes = all_dups.datetime.values

	non_dup_ids = []
	non_dup_indices = []
	for index,row in all_dups.iterrows():
		date = row.datetime
		i = np.argmin(np.abs(event_cat.datetime - date))
		non_dup_id = event_cat.iloc[i].evid
		non_dup_mag = event_cat.iloc[i].mag
	# 	non_dup_id = event_cat.iloc[np.where(event_cat_datetimes == min(event_cat_datetimes, key=lambda d: abs(d - date)))[0][0]].evid
		print(index,row.evid,non_dup_id,row.mag,non_dup_mag)
		non_dup_ids.append(non_dup_id)
		non_dup_indices.append(index)

	event_df.loc[non_dup_indices,['evid']] = non_dup_ids

	event_df_dups = event_df[event_df.evid.duplicated()]

	drop_dups = []
	for index,row in event_df_dups.iterrows():
		evid = row.evid
		event_df_dup_pair = event_df[event_df.evid == evid]
		cat_time = event_cat[event_cat.evid == evid].datetime.iloc[0]
		i = np.argmax(np.abs(event_df_dup_pair.datetime - cat_time))
		dup_drop = event_df_dup_pair.iloc[i].name
		print(dup_drop)
		drop_dups.append(dup_drop)

	event_df = event_df.drop(index=drop_dups)
	
	return event_df

# 	event_df.to_csv('martins_new.csv',index=False)

# event_df = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/Scripts/EQ/martins_test.csv',low_memory=False)
# event_cat = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/earthquake_source_table.csv',low_memory=False)
# 
# relocation_dups(event_df, event_cat)