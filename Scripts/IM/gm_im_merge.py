# This program finds all ground motion IM data and compiles it to a single catalogue.
# It also searches for additional data not provided in the additional catalogues by
# reading QuakeML files in the mseed directories. This program is very dependent on
# directory structure. It also takes advantage of multiple processes because otherwise
# it would be very slow.

import glob
import pandas as pd
import os
from datetime import datetime as datetime
import obspy as op
from multiprocessing import Pool,cpu_count
import itertools

def get_gm_data(file):
# for file in file_list:
# 	print(file)
	# Used to find the event information
    basedir = '/Volumes/SeaJade 2 Backup/NZ/mseed_'+lower_mag+'-'+upper_mag+'/'
    # Used to find the matching file
    preferred_dir = '/Volumes/SeaJade 2 Backup/NZ/mseed_'+lower_mag+'-'+upper_mag+'_preferred/'

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
    
    return df

lower_mag = '4'
upper_mag = '4.5'
# 	df_all = pd.concat([df_all,df],ignore_index=True)
search_dir = '/Volumes/SeaJade 2 Backup/NZ/GM_IM_'+lower_mag+'-'+upper_mag+'/*/*/*/'
file_list = glob.glob(search_dir+'gm_all.csv')

cores = int(cpu_count()-1)
pool = Pool(cores)
df_all = pd.concat(pool.map(get_gm_data, file_list),ignore_index=True)
pool.close()
pool.join()

# Prepare the GM IM results for writing to .CSV file

df_all = df_all[['gmid','datetime','evid','station','loc','chan','component','PGA', 'PGV', 
	'CAV', 'AI', 'Ds575', 'Ds595','MMI', 'pSA_0.01', 'pSA_0.02', 'pSA_0.03', 'pSA_0.04', 'pSA_0.05',
    'pSA_0.075', 'pSA_0.1', 'pSA_0.12', 'pSA_0.15', 'pSA_0.17', 'pSA_0.2',
    'pSA_0.25', 'pSA_0.3', 'pSA_0.4', 'pSA_0.5', 'pSA_0.6', 'pSA_0.7',
    'pSA_0.75', 'pSA_0.8', 'pSA_0.9', 'pSA_1.0', 'pSA_1.25', 'pSA_1.5',
    'pSA_2.0', 'pSA_2.5', 'pSA_3.0', 'pSA_4.0', 'pSA_5.0', 'pSA_6.0',
    'pSA_7.5', 'pSA_10.0']]
df_all = df_all.sort_values('datetime')
df_all = df_all.rename(columns = {'station':'sta'})
df_all['loc'] = df_all['loc'].astype('int')

df_all.to_csv('ground_motion_im_catalogue_'+lower_mag+'-'+upper_mag+'.csv',index=False)

# Add ground motion quality results and write to a separate 'final' file
# gmc_results = pd.read_csv(
# 	'/Volumes/SeaJade2/NZ/Reports/For Robin/GMC_cat/meta_gmc_results.csv',low_memory=False)
gmc_results = pd.read_csv(
	'/Volumes/SeaJade 2 Backup/NZ/gmc_record/mseed_'+lower_mag+'-'+upper_mag+'_results/results.csv',low_memory=False)
gmc_results['chan'] = gmc_results['record_id'].str.split('_').str[-2]
gmc_results['loc'] = gmc_results['record_id'].str.split('_').str[-3].astype('int')
gmc_results = gmc_results.rename(columns={'event_id':'evid','station':'sta'})

new_df = gmc_results.drop_duplicates(subset='record').reset_index(drop=True)

new_df[['score_mean_X', 'score_std_X','fmin_mean_X', 'fmin_std_X', 'multi_mean_X', 'multi_std_X']] = gmc_results[gmc_results.component == 'X'][['score_mean','score_std','fmin_mean', 'fmin_std','multi_mean', 'multi_std']].values
new_df[['score_mean_Y', 'score_std_Y','fmin_mean_Y', 'fmin_std_Y', 'multi_mean_Y', 'multi_std_Y']] = gmc_results[gmc_results.component == 'Y'][['score_mean','score_std','fmin_mean', 'fmin_std','multi_mean', 'multi_std']].values
new_df[['score_mean_Z', 'score_std_Z','fmin_mean_Z', 'fmin_std_Z', 'multi_mean_Z', 'multi_std_Z']] = gmc_results[gmc_results.component == 'Z'][['score_mean','score_std','fmin_mean', 'fmin_std','multi_mean', 'multi_std']].values

# new_df = pd.concat([pd.DataFrame({'record':(gmc_results.record+'_X').values,'score_mean':gmc_results.score_X.values}),
#     pd.DataFrame({'record':(gmc_results.record+'_Y').values,'score_mean':gmc_results.score_Y.values}),
#     pd.DataFrame({'record':(gmc_results.record+'_Z').values,'score_mean':gmc_results.score_Z.values})]).reset_index(drop=True)
# 

gm_final = pd.merge(df_all,new_df[['score_mean_X', 'score_std_X','fmin_mean_X', 'fmin_std_X', 'multi_mean_X', 'multi_std_X',
    'score_mean_Y', 'score_std_Y','fmin_mean_Y', 'fmin_std_Y', 'multi_mean_Y', 'multi_std_Y',
    'score_mean_Z', 'score_std_Z','fmin_mean_Z', 'fmin_std_Z', 'multi_mean_Z', 'multi_std_Z',
    'evid','sta','chan','loc']],on=['evid','sta','chan','loc'],how='left')

directory = '/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/gmc_output/'
if not os.path.exists(directory+'/IM_catalogue'):
    os.makedirs(directory+'/IM_catalogue')
gm_final.to_csv(directory+'IM_catalogue/ground_motion_im_catalogue_'+lower_mag+'-'+upper_mag+'_final.csv',index=False)