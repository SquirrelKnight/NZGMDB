import pandas as pd
import os
import glob
import numpy as np
from datetime import datetime as datetime
import obspy as op
from multiprocessing import Pool,cpu_count
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
import matplotlib.dates as mdates

def get_gm_data(file,preferred_dir):
# 	print(file)
	# Used to find the event information
	basedir = '/Volumes/SeaJade 2 Backup/NZ/mseed_5_revised/'
	# Used to find the matching file
# 	preferred_dir = '/Volumes/SeaJade 2 Backup/NZ/mseed_5_HH/'

	file_path = os.path.dirname(file).split('/')
	date, time = file_path[-1].split('_')
	dt = datetime.strptime(date+time,'%Y-%m-%d%H%M%S')
	
	folderA = str(dt.year)
	folderB = dt.strftime('%m_%b')
	folderC = dt.strftime('%Y-%m-%d_%H%M%S')
	
	# Sometimes the folder names do not match, this should account for that issue.
	directory = basedir+folderA+'/'+folderB+'/'+folderC
	if not os.path.exists(directory):
		folderC = (dt - datetime.timedelta(seconds=1)).strftime('%Y-%m-%d_%H%M%S')
		directory = basedir+folderA+'/'+folderB+'/'+folderC
	if not os.path.exists(directory):
		folderC = (dt + datetime.timedelta(seconds=1)).strftime('%Y-%m-%d_%H%M%S')
		directory = basedir+folderA+'/'+folderB+'/'+folderC			
	
	# Old xml name format
# 	xml_name = dt.strftime('%Y%m%d_%H%M%S.xml')
# 	xml_file = base_dir+folderA+'/'+folderB+'/'+folderC+'/'+xml_name
# 	event = op.read_events(xml_file)[0]
# 	evid = str(event.resource_id).split('/')[-1]

	# New xml name format
	xml_path = basedir+folderA+'/'+folderB+'/'+folderC+'/*.xml'
	xml_name = os.path.basename(glob.glob(xml_path)[0])
	evid = xml_name.split('.')[0]
	
	df = pd.read_csv(file).reset_index(drop=True)
	df['evid'] = evid
	df['datetime'] = pd.to_datetime(dt).to_datetime64()
	
	folderC = dt.strftime('%Y-%m-%d_%H%M%S')
	
	station_list = df.station.unique()
	
# 	chans = '[HH][HN]'
	for index, row in df.iterrows():
		sta = row.station
		file_name = glob.glob(preferred_dir+folderA+'/'+folderB+'/'+folderC+'/mseed/data/'+dt.strftime('%Y%m%d_%H%M%S_')+sta+'_*'+'_*.mseed')[0]
		print(file_name)
		loc = file_name.split('_')[-2]
		chan = file_name.split('_')[-1].split('.')[0]
		gmid = evid+'gm'+str(index+1)
		df.loc[index,['loc','chan','gmid']] = [loc,chan,gmid]
		
	return(df)


def build_df(search_dir):
    file_list = glob.glob(search_dir+'gm_all.csv')
    df = pd.DataFrame()
    for i,file in enumerate(file_list):
        date_time = file.split('/')[-2]
        df_sub = pd.read_csv(file)
        df_sub = df_sub[df_sub['component'] == 'rotd50']
        df_sub['evid'] = date_time
        df = pd.concat([df,df_sub])
    df.reset_index(drop=True,inplace=True)
    return df

ev_df = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/earthquake_source_table_complete.csv',low_memory=False)
filename = '/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/meta_earthquakes.csv'
geonet = pd.read_csv(filename,low_memory=False)
geonet = geonet.rename(columns={'publicid':'evid','magnitude':'mag'})
prop_df = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/propagation_path_table_complete.csv',low_memory=False)

search_dir = '/Volumes/SeaJade 2 Backup/NZ/GM_IM_5_HH_merged/*/*/*/'
file_list = glob.glob(search_dir+'gm_all.csv')
preferred_dir = '/Volumes/SeaJade 2 Backup/NZ/mseed_5_HH/'
argzip=zip(file_list, itertools.repeat(preferred_dir))

cores = int(cpu_count()-1)
pool = Pool(cores)
df_HH = pd.concat(pool.starmap(get_gm_data, argzip),ignore_index=True)
pool.close()
pool.join()

search_dir = '/Volumes/SeaJade 2 Backup/NZ/GM_IM_5_HN_merged/*/*/*/'
file_list = glob.glob(search_dir+'gm_all.csv')
preferred_dir = '/Volumes/SeaJade 2 Backup/NZ/mseed_5_HN/'
argzip=zip(file_list, itertools.repeat(preferred_dir))

cores = int(cpu_count()-1)
pool = Pool(cores)
df_HN = pd.concat(pool.starmap(get_gm_data, argzip),ignore_index=True)
pool.close()
pool.join()

df_HH = df_HH[df_HH['component'] == 'rotd50'].reset_index(drop=True)
df_HH = df_HH.rename(columns = {'station':'sta'})
df_HH['loc'] = df_HH['loc'].astype('int')

df_HN = df_HN[df_HN['component'] == 'rotd50'].reset_index(drop=True)
df_HN = df_HN.rename(columns = {'station':'sta'})
df_HN['loc'] = df_HN['loc'].astype('int')

gmc_results = pd.read_csv(
	'/Volumes/SeaJade 2 Backup/NZ/gmc_record/mseed_5_results_revised/results.csv',low_memory=False)
gmc_results['chan'] = gmc_results['record_id'].str.split('_').str[-1]
gmc_results['loc'] = gmc_results['record_id'].str.split('_').str[-2].astype('int')
gmc_results = gmc_results.rename(columns={'event_id':'evid','station':'sta'})

df_HH = pd.merge(df_HH,gmc_results[['score_X','f_min_X','score_Y','f_min_Y','score_Z',
	'f_min_Z','evid','sta','chan','loc']],on=['evid','sta','chan','loc'],how='left')
df_HN = pd.merge(df_HN,gmc_results[['score_X','f_min_X','score_Y','f_min_Y','score_Z',
	'f_min_Z','evid','sta','chan','loc']],on=['evid','sta','chan','loc'],how='left')


# search_dir = '/Volumes/SeaJade 2 Backup/NZ/GM_IM_5_HH/*/*/*/'
# df_HH = build_df(search_dir)
# 
# search_dir = '/Volumes/SeaJade 2 Backup/NZ/GM_IM_5_HN/*/*/*/'
# df_HN = build_df(search_dir)

df_HH = df_HH.set_index(df_HH.evid+df_HH.sta)
df_HN = df_HN.set_index(df_HN.evid+df_HN.sta)

df_HH = df_HH[df_HH.index.isin(df_HN.index) == True]
df_HN = df_HN[df_HN.index.isin(df_HH.index) == True]

df_HH = df_HH.sort_index()
df_HN = df_HN.sort_index()

score_X_diff = df_HH.score_X - df_HN.score_X
score_Y_diff = df_HH.score_Y - df_HN.score_Y
score_Z_diff = df_HH.score_Z - df_HN.score_Z

# df_HH.iloc[np.where(f_min_X_diff == f_min_X_diff.max())]

df_merged = df_HH.join(df_HN,rsuffix='_HN')
df_merged = df_merged.drop(df_merged[df_merged.PGA_HN.isnull()].index)
df_merged['score_X_diff'] = score_X_diff
df_merged['score_Y_diff'] = score_Y_diff
df_merged['score_Z_diff'] = score_Z_diff

# Merge event data with IM data
df_merged = df_merged.set_index('evid').join(geonet[['evid','mag']].set_index('evid'),how='inner').reset_index()
prop_df_sub = prop_df[prop_df['evid'].isin(df_merged.evid.unique())]
prop_df_sub = prop_df_sub[prop_df_sub[['evid','net','sta']].duplicated() == False]
df_merged['r_hyp'] = (df_merged['evid']+df_merged['sta']).map(prop_df_sub.set_index(prop_df_sub['evid']+prop_df_sub['sta'])['r_hyp'])

# psa_diff = df_merged[['pSA_0.01', 'pSA_0.02', 'pSA_0.03', 'pSA_0.04', 'pSA_0.05',
#        'pSA_0.075', 'pSA_0.1', 'pSA_0.12', 'pSA_0.15', 'pSA_0.17', 'pSA_0.2',
#        'pSA_0.25', 'pSA_0.3', 'pSA_0.4', 'pSA_0.5', 'pSA_0.6', 'pSA_0.7',
#        'pSA_0.75', 'pSA_0.8', 'pSA_0.9', 'pSA_1.0', 'pSA_1.25', 'pSA_1.5',
#        'pSA_2.0', 'pSA_2.5', 'pSA_3.0', 'pSA_4.0', 'pSA_5.0', 'pSA_6.0',
#        'pSA_7.5', 'pSA_10.0']].values / df_merged[['pSA_0.01_HN', 'pSA_0.02_HN', 'pSA_0.03_HN', 'pSA_0.04_HN', 'pSA_0.05_HN',
#        'pSA_0.075_HN', 'pSA_0.1_HN', 'pSA_0.12_HN', 'pSA_0.15_HN', 'pSA_0.17_HN', 'pSA_0.2_HN',
#        'pSA_0.25_HN', 'pSA_0.3_HN', 'pSA_0.4_HN', 'pSA_0.5_HN', 'pSA_0.6_HN', 'pSA_0.7_HN',
#        'pSA_0.75_HN', 'pSA_0.8_HN', 'pSA_0.9_HN', 'pSA_1.0_HN', 'pSA_1.25_HN', 'pSA_1.5_HN',
#        'pSA_2.0_HN', 'pSA_2.5_HN', 'pSA_3.0_HN', 'pSA_4.0_HN', 'pSA_5.0_HN', 'pSA_6.0_HN',
#        'pSA_7.5_HN', 'pSA_10.0_HN']].values

# df_merged = df_HH[['pSA_0.01', 'pSA_0.02', 'pSA_0.03', 'pSA_0.04', 'pSA_0.05',
#        'pSA_0.075', 'pSA_0.1', 'pSA_0.12', 'pSA_0.15', 'pSA_0.17', 'pSA_0.2',
#        'pSA_0.25', 'pSA_0.3', 'pSA_0.4', 'pSA_0.5', 'pSA_0.6', 'pSA_0.7',
#        'pSA_0.75', 'pSA_0.8', 'pSA_0.9', 'pSA_1.0', 'pSA_1.25', 'pSA_1.5',
#        'pSA_2.0', 'pSA_2.5', 'pSA_3.0', 'pSA_4.0', 'pSA_5.0', 'pSA_6.0',
#        'pSA_7.5', 'pSA_10.0']] - df_HN[['pSA_0.01', 'pSA_0.02', 'pSA_0.03', 'pSA_0.04', 'pSA_0.05',
#        'pSA_0.075', 'pSA_0.1', 'pSA_0.12', 'pSA_0.15', 'pSA_0.17', 'pSA_0.2',
#        'pSA_0.25', 'pSA_0.3', 'pSA_0.4', 'pSA_0.5', 'pSA_0.6', 'pSA_0.7',
#        'pSA_0.75', 'pSA_0.8', 'pSA_0.9', 'pSA_1.0', 'pSA_1.25', 'pSA_1.5',
#        'pSA_2.0', 'pSA_2.5', 'pSA_3.0', 'pSA_4.0', 'pSA_5.0', 'pSA_6.0',
#        'pSA_7.5', 'pSA_10.0']]
    
       
# y = np.log(abs(psa_diff))
# x = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.12, 0.15, 0.17, 0.2, 0.25, 0.3, 0.4, 0.5,
#      0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.5, 10.0])

# ii = 0
# for i,ye in enumerate(y):
#     print(ye)    
# #     print(ye.std())
#     if abs(ye.std()) <= 0.05:
#         if abs(ye.mean()) <= 0.5:
#             print(i, ye.mean(), ye.std(), f_min_X_diff[i], f_min_Y_diff[i], f_min_Z_diff[i], df_HH.iloc[i].sta, df_HH.iloc[i]['datetime'])
#             plt.plot(x, ye)
#             ii += 1
# plt.xlabel('pSA')
# plt.ylabel('ln(HH / HN)')
# plt.xscale('log')
# plt.show()
# 
# mean = np.mean(y)
# std = np.std(y)

stas = df_merged.sta.unique()
# col = plt.cm.jet([0,1])
z = mdates.date2num(df_merged.datetime.sort_values().unique())
z_norm = (z - z.min()) / (z.max() - z.min())
n = len(z)
# colors = plt.cm.viridis(np.linspace(0,1,n))
colors = plt.cm.viridis(z_norm)
# n_linspace = np.linspace(0,1,n)
n_linspace = z_norm
# tick_spacing = int(np.round(len(z)/6))
# tick_ints = 0,tick_spacing,tick_spacing * 2, tick_spacing * 3, tick_spacing * 4, tick_spacing * 5, -1
z_ints = (z[-1] - z[0]) / 6
# tick_values = mdates.num2date([z[tick_ints[0]],z[tick_ints[1]],z[tick_ints[2]],z[tick_ints[3]],z[tick_ints[4]],z[tick_ints[5]],z[tick_ints[6]]])
tick_values = mdates.num2date([z[0],z[0]+z_ints,z[0]+z_ints * 2,z[0]+z_ints * 3,z[0]+z_ints * 4,z[0]+z_ints * 5,z[-1]])
color_ticks = 0,1/6, 1/6 * 2, 1/6 * 3, 1/6 * 4,1/6 * 5, 1
for sta in stas:
    df_merged_sub = df_merged[df_merged.sta == sta]
    psa_diff = np.log(df_merged_sub[['pSA_0.01', 'pSA_0.02', 'pSA_0.03', 'pSA_0.04', 'pSA_0.05',
       'pSA_0.075', 'pSA_0.1', 'pSA_0.12', 'pSA_0.15', 'pSA_0.17', 'pSA_0.2',
       'pSA_0.25', 'pSA_0.3', 'pSA_0.4', 'pSA_0.5', 'pSA_0.6', 'pSA_0.7',
       'pSA_0.75', 'pSA_0.8', 'pSA_0.9', 'pSA_1.0', 'pSA_1.25', 'pSA_1.5',
       'pSA_2.0', 'pSA_2.5', 'pSA_3.0', 'pSA_4.0', 'pSA_5.0', 'pSA_6.0',
       'pSA_7.5', 'pSA_10.0']].values) - np.log(df_merged_sub[['pSA_0.01_HN', 'pSA_0.02_HN', 'pSA_0.03_HN', 'pSA_0.04_HN', 'pSA_0.05_HN',
       'pSA_0.075_HN', 'pSA_0.1_HN', 'pSA_0.12_HN', 'pSA_0.15_HN', 'pSA_0.17_HN', 'pSA_0.2_HN',
       'pSA_0.25_HN', 'pSA_0.3_HN', 'pSA_0.4_HN', 'pSA_0.5_HN', 'pSA_0.6_HN', 'pSA_0.7_HN',
       'pSA_0.75_HN', 'pSA_0.8_HN', 'pSA_0.9_HN', 'pSA_1.0_HN', 'pSA_1.25_HN', 'pSA_1.5_HN',
       'pSA_2.0_HN', 'pSA_2.5_HN', 'pSA_3.0_HN', 'pSA_4.0_HN', 'pSA_5.0_HN', 'pSA_6.0_HN',
       'pSA_7.5_HN', 'pSA_10.0_HN']].values)
    x = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.12, 0.15, 0.17, 0.2, 0.25, 0.3, 0.4, 0.5,
         0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.5, 10.0])
    y = psa_diff
    segs = np.zeros((len(y),len(x),2))
    segs[:, :, 1] = y
    segs[:, :, 0] = x
    c = colors[np.isin(z,mdates.date2num(df_merged_sub.datetime))]
    
    
    fig, ax = plt.subplots()
    ax.set_ylim(y.min() - 0.5, y.max() + 0.5)
    ax.set_xlim(x.min(), x.max())
    line_segments = LineCollection(segs, 
                                   colors=c, linestyle='solid')
    ax.add_collection(line_segments)
    axcb = fig.colorbar(line_segments)
    axcb.set_ticks(color_ticks)
    axcb.set_ticklabels(tick_values)
    axcb.set_label('Date')    
    axcb.ax.tick_params(labelsize=6)
    plt.xlabel('pSA')
    plt.ylabel('ln(HH) - ln(HN)')
    plt.xscale('log')
    plt.title('Station '+sta)
    plt.tight_layout()
    fig.savefig('datetime_figs/pSA_diff_'+str(sta)+'_datetime.jpg')
#     plt.show()
    plt.close()
#     plt.show()

z = df_merged.mag.sort_values().unique()
z_norm = (z - z.min()) / (z.max() - z.min())
n = len(z)
colors = plt.cm.viridis(z_norm)
n_linspace = z_norm
z_ints = (z[-1] - z[0]) / 6
tick_values = [z[0],z[0]+z_ints,z[0]+z_ints * 2,z[0]+z_ints * 3,z[0]+z_ints * 4,z[0]+z_ints * 5,z[-1]]
color_ticks = 0,1/6, 1/6 * 2, 1/6 * 3, 1/6 * 4,1/6 * 5, 1
for sta in stas:
    df_merged_sub = df_merged[df_merged.sta == sta]
    psa_diff = np.log(df_merged_sub[['pSA_0.01', 'pSA_0.02', 'pSA_0.03', 'pSA_0.04', 'pSA_0.05',
       'pSA_0.075', 'pSA_0.1', 'pSA_0.12', 'pSA_0.15', 'pSA_0.17', 'pSA_0.2',
       'pSA_0.25', 'pSA_0.3', 'pSA_0.4', 'pSA_0.5', 'pSA_0.6', 'pSA_0.7',
       'pSA_0.75', 'pSA_0.8', 'pSA_0.9', 'pSA_1.0', 'pSA_1.25', 'pSA_1.5',
       'pSA_2.0', 'pSA_2.5', 'pSA_3.0', 'pSA_4.0', 'pSA_5.0', 'pSA_6.0',
       'pSA_7.5', 'pSA_10.0']].values) - np.log(df_merged_sub[['pSA_0.01_HN', 'pSA_0.02_HN', 'pSA_0.03_HN', 'pSA_0.04_HN', 'pSA_0.05_HN',
       'pSA_0.075_HN', 'pSA_0.1_HN', 'pSA_0.12_HN', 'pSA_0.15_HN', 'pSA_0.17_HN', 'pSA_0.2_HN',
       'pSA_0.25_HN', 'pSA_0.3_HN', 'pSA_0.4_HN', 'pSA_0.5_HN', 'pSA_0.6_HN', 'pSA_0.7_HN',
       'pSA_0.75_HN', 'pSA_0.8_HN', 'pSA_0.9_HN', 'pSA_1.0_HN', 'pSA_1.25_HN', 'pSA_1.5_HN',
       'pSA_2.0_HN', 'pSA_2.5_HN', 'pSA_3.0_HN', 'pSA_4.0_HN', 'pSA_5.0_HN', 'pSA_6.0_HN',
       'pSA_7.5_HN', 'pSA_10.0_HN']].values)
    x = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.12, 0.15, 0.17, 0.2, 0.25, 0.3, 0.4, 0.5,
         0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.5, 10.0])
    y = psa_diff
    segs = np.zeros((len(y),len(x),2))
    segs[:, :, 1] = y
    segs[:, :, 0] = x
    c = colors[np.isin(z,df_merged_sub.mag.values)]
    
    
    fig, ax = plt.subplots()
    ax.set_ylim(y.min() - 0.5, y.max() + 0.5)
    ax.set_xlim(x.min(), x.max())
    line_segments = LineCollection(segs, 
                                   colors=c, linestyle='solid')
    ax.add_collection(line_segments)
    axcb = fig.colorbar(line_segments)
    axcb.set_ticks(color_ticks)
    axcb.set_ticklabels(tick_values)
    axcb.set_label('Magnitude')    
    axcb.ax.tick_params(labelsize=6)
    plt.xlabel('pSA')
    plt.ylabel('ln(HH) - ln(HN)')
    plt.xscale('log')
    plt.title('Station '+sta)
    plt.tight_layout()
    fig.savefig('mag_figs/pSA_diff_'+str(sta)+'_mag.jpg')
    plt.close()

z = df_merged[~df_merged.r_hyp.isnull()].r_hyp.sort_values().unique()
z_norm = (z - z.min()) / (z.max() - z.min())
n = len(z)
colors = plt.cm.viridis(z_norm)
n_linspace = z_norm
z_ints = (z[-1] - z[0]) / 6
tick_values = [z[0],z[0]+z_ints,z[0]+z_ints * 2,z[0]+z_ints * 3,z[0]+z_ints * 4,z[0]+z_ints * 5,z[-1]]
color_ticks = 0,1/6, 1/6 * 2, 1/6 * 3, 1/6 * 4,1/6 * 5, 1
for sta in stas:
    df_merged_sub = df_merged[df_merged.sta == sta]
    psa_diff = np.log(df_merged_sub[['pSA_0.01', 'pSA_0.02', 'pSA_0.03', 'pSA_0.04', 'pSA_0.05',
       'pSA_0.075', 'pSA_0.1', 'pSA_0.12', 'pSA_0.15', 'pSA_0.17', 'pSA_0.2',
       'pSA_0.25', 'pSA_0.3', 'pSA_0.4', 'pSA_0.5', 'pSA_0.6', 'pSA_0.7',
       'pSA_0.75', 'pSA_0.8', 'pSA_0.9', 'pSA_1.0', 'pSA_1.25', 'pSA_1.5',
       'pSA_2.0', 'pSA_2.5', 'pSA_3.0', 'pSA_4.0', 'pSA_5.0', 'pSA_6.0',
       'pSA_7.5', 'pSA_10.0']].values) - np.log(df_merged_sub[['pSA_0.01_HN', 'pSA_0.02_HN', 'pSA_0.03_HN', 'pSA_0.04_HN', 'pSA_0.05_HN',
       'pSA_0.075_HN', 'pSA_0.1_HN', 'pSA_0.12_HN', 'pSA_0.15_HN', 'pSA_0.17_HN', 'pSA_0.2_HN',
       'pSA_0.25_HN', 'pSA_0.3_HN', 'pSA_0.4_HN', 'pSA_0.5_HN', 'pSA_0.6_HN', 'pSA_0.7_HN',
       'pSA_0.75_HN', 'pSA_0.8_HN', 'pSA_0.9_HN', 'pSA_1.0_HN', 'pSA_1.25_HN', 'pSA_1.5_HN',
       'pSA_2.0_HN', 'pSA_2.5_HN', 'pSA_3.0_HN', 'pSA_4.0_HN', 'pSA_5.0_HN', 'pSA_6.0_HN',
       'pSA_7.5_HN', 'pSA_10.0_HN']].values)
    x = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.12, 0.15, 0.17, 0.2, 0.25, 0.3, 0.4, 0.5,
         0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.5, 10.0])
    y = psa_diff
    segs = np.zeros((len(y),len(x),2))
    segs[:, :, 1] = y
    segs[:, :, 0] = x
    if np.any(df_merged_sub.r_hyp.isnull().values):
        continue
    c = colors[np.isin(z,df_merged_sub.r_hyp.values)]
    
    
    fig, ax = plt.subplots()
    ax.set_ylim(y.min() - 0.5, y.max() + 0.5)
    ax.set_xlim(x.min(), x.max())
    line_segments = LineCollection(segs, 
                                   colors=c, linestyle='solid')
    ax.add_collection(line_segments)
    axcb = fig.colorbar(line_segments)
    axcb.set_ticks(color_ticks)
    axcb.set_ticklabels(tick_values)
    axcb.set_label('R_hyp (km)')    
    axcb.ax.tick_params(labelsize=6)
    plt.xlabel('pSA')
    plt.ylabel('ln(HH) - ln(HN)')
    plt.xscale('log')
    plt.title('Station '+sta)
    plt.tight_layout()
    fig.savefig('distance_figs/pSA_diff_'+str(sta)+'_distance.jpg')
    plt.close()

# Write ln(psa_diff) info to file
from obspy.clients.fdsn import Client as FDSN_Client

client_NZ = FDSN_Client("GEONET")
client_IU = FDSN_Client('IRIS')
inventory_NZ = client_NZ.get_stations()
inventory_IU = client_IU.get_stations(network='IU',station='SNZO,AFI,CTAO,RAO,FUNA,HNR,PMG')
inventory_AU = client_IU.get_stations(network='AU')
inventory = inventory_NZ+inventory_IU+inventory_AU

station_info = []
for network in inventory:
	for station in network:
		station_info.append([network.code, station.code, station.latitude, station.longitude, station.elevation])
station_df = pd.DataFrame(station_info,columns=['net','sta','lat','lon','elev'])
station_df = station_df.drop_duplicates().reset_index(drop=True)

stas = df_merged.sta.unique()
for i,sta in enumerate(stas):
    df_merged_sub = df_merged[df_merged.sta == sta]
    sta_sub = station_df[station_df.sta == sta].iloc[0]
    no_events = len(df_merged_sub)
    psa_diff = np.log(df_merged_sub[['pSA_0.01', 'pSA_0.02', 'pSA_0.03', 'pSA_0.04', 'pSA_0.05',
       'pSA_0.075', 'pSA_0.1', 'pSA_0.12', 'pSA_0.15', 'pSA_0.17', 'pSA_0.2',
       'pSA_0.25', 'pSA_0.3', 'pSA_0.4', 'pSA_0.5', 'pSA_0.6', 'pSA_0.7',
       'pSA_0.75', 'pSA_0.8', 'pSA_0.9', 'pSA_1.0', 'pSA_1.25', 'pSA_1.5',
       'pSA_2.0', 'pSA_2.5', 'pSA_3.0', 'pSA_4.0', 'pSA_5.0', 'pSA_6.0',
       'pSA_7.5', 'pSA_10.0']].values) - np.log(df_merged_sub[['pSA_0.01_HN', 'pSA_0.02_HN', 'pSA_0.03_HN', 'pSA_0.04_HN', 'pSA_0.05_HN',
       'pSA_0.075_HN', 'pSA_0.1_HN', 'pSA_0.12_HN', 'pSA_0.15_HN', 'pSA_0.17_HN', 'pSA_0.2_HN',
       'pSA_0.25_HN', 'pSA_0.3_HN', 'pSA_0.4_HN', 'pSA_0.5_HN', 'pSA_0.6_HN', 'pSA_0.7_HN',
       'pSA_0.75_HN', 'pSA_0.8_HN', 'pSA_0.9_HN', 'pSA_1.0_HN', 'pSA_1.25_HN', 'pSA_1.5_HN',
       'pSA_2.0_HN', 'pSA_2.5_HN', 'pSA_3.0_HN', 'pSA_4.0_HN', 'pSA_5.0_HN', 'pSA_6.0_HN',
       'pSA_7.5_HN', 'pSA_10.0_HN']].values)
    mean = np.mean(psa_diff,axis=0)
    std = np.std(psa_diff,axis=0)
    sub_df = pd.DataFrame([[sta,sta_sub.lat,sta_sub.lon,'HH','HN',no_events]],columns=['sta','lat',
        'lon','broadband','strong_motion','num_events'])
    sub_df[['mean_ln_pSA_0.01', 'mean_ln_pSA_0.02', 'mean_ln_pSA_0.03', 'mean_ln_pSA_0.04', 'mean_ln_pSA_0.05',
       'mean_ln_pSA_0.075', 'mean_ln_pSA_0.1', 'mean_ln_pSA_0.12', 'mean_ln_pSA_0.15', 'mean_ln_pSA_0.17', 'mean_ln_pSA_0.2',
       'mean_ln_pSA_0.25', 'mean_ln_pSA_0.3', 'mean_ln_pSA_0.4', 'mean_ln_pSA_0.5', 'mean_ln_pSA_0.6', 'mean_ln_pSA_0.7',
       'mean_ln_pSA_0.75', 'mean_ln_pSA_0.8', 'mean_ln_pSA_0.9', 'mean_ln_pSA_1.0', 'mean_ln_pSA_1.25', 'mean_ln_pSA_1.5',
       'mean_ln_pSA_2.0', 'mean_ln_pSA_2.5', 'mean_ln_pSA_3.0', 'mean_ln_pSA_4.0', 'mean_ln_pSA_5.0', 'mean_ln_pSA_6.0',
       'mean_ln_pSA_7.5', 'mean_ln_pSA_10.0']] = mean
    sub_df[['std_ln_pSA_0.01', 'std_ln_pSA_0.02', 'std_ln_pSA_0.03', 'std_ln_pSA_0.04', 'std_ln_pSA_0.05',
       'std_ln_pSA_0.075', 'std_ln_pSA_0.1', 'std_ln_pSA_0.12', 'std_ln_pSA_0.15', 'std_ln_pSA_0.17', 'std_ln_pSA_0.2',
       'std_ln_pSA_0.25', 'std_ln_pSA_0.3', 'std_ln_pSA_0.4', 'std_ln_pSA_0.5', 'std_ln_pSA_0.6', 'std_ln_pSA_0.7',
       'std_ln_pSA_0.75', 'std_ln_pSA_0.8', 'std_ln_pSA_0.9', 'std_ln_pSA_1.0', 'std_ln_pSA_1.25', 'std_ln_pSA_1.5',
       'std_ln_pSA_2.0', 'std_ln_pSA_2.5', 'std_ln_pSA_3.0', 'std_ln_pSA_4.0', 'std_ln_pSA_5.0', 'std_ln_pSA_6.0',
       'std_ln_pSA_7.5', 'std_ln_pSA_10.0']] = std
    if i == 0:
        df_out = sub_df
    else:
        df_out = df_out.append(sub_df)
df_out.reset_index(drop=True)
df_out.to_csv('psa_diff.csv',index=False)

stas = df_merged.sta.unique()
for sta in stas:
    df_merged_sub = df_merged[df_merged.sta == sta]
    y = np.log(df_merged_sub.PGA) - np.log(df_merged_sub.PGA_HN)
    x = df_merged_sub.PGA_HN
    

    fig, ax = plt.subplots()
    ax.scatter(x,y)
    if abs(y).max() < 0.2:
        plt.ylim(-0.2,0.2)
    plt.axhline(y=-0.1,c='darkgrey',linestyle='--')
    plt.axhline(y=0.1,c='darkgrey',linestyle='--')
    plt.title('Station '+sta)
    plt.xlabel('PGA HN')
    plt.ylabel('ln(PGA_HH) - ln(PGA_HN)')
#     plt.show()
    fig.savefig('spectral_diff_figs/PGA_diff_'+str(sta)+'.jpg')
    plt.close()















mag_low = 4.5
mag_high = 5

df = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/IM_catalogue/Tables/ground_motion_im_table_rotd50_flat.csv')
df_sub = df[(df.mag >= mag_low) & (df.mag < mag_high)].reset_index(drop=True)

df_HH = df_sub[df_sub.chan == 'HH'].reset_index(drop=True)
df_HN = df_sub[df_sub.chan == 'HN'].reset_index(drop=True)

fig, ax = plt.subplots()
ax.scatter(df_HN.r_rup,df_HN['pSA_1.0'])
ax.scatter(df_HH.r_rup,df_HH['pSA_1.0'])
plt.show()