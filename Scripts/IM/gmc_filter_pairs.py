'''
Find high scoring gmc calculations across two separate channels and write them to files.
This can be used to evaluate IM calculations between instruments at the same station.
'''

import pandas as pd
import datetime
import os
from shutil import copyfile
import obspy as op
from obspy.clients.fdsn import Client as FDSN_Client
from multiprocessing import Pool,cpu_count
import numpy as np
import time as timeit
import glob

file_list = glob.glob('/Volumes/SeaJade 2 Backup/NZ/gmc_record/mseed_5_results_revised/results.csv')
high_mag = 6
low_mag = 5


df = pd.concat(pd.read_csv(file,low_memory=False) for file in file_list)
df['chan'] = df['record_id'].str.split('_').str[-1]
df['loc'] = df['record_id'].str.split('_').str[-2].astype('int')
df['record'] = df.record_id.str[0:-6]

filename = '/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/meta_earthquakes.csv'
geonet = pd.read_csv(filename,low_memory=False)
geonet = geonet.sort_values('origintime')
geonet['origintime'] = pd.to_datetime(geonet['origintime'],format='%Y-%m-%dT%H:%M:%S.%fZ')
geonet = geonet.reset_index(drop=True)
# Filters data within a range of magnitudes so that we don't end up redoing lots of good
# work already completed at higher magnitudes!
# Make sure data is from an earthquake
geonet_sub = geonet[(geonet.magnitude >= low_mag) & (geonet.magnitude < high_mag)].reset_index(drop=True)
geonet_sub.loc[geonet_sub.longitude < 0, 'longitude'] = 360 + geonet_sub.longitude[geonet_sub.longitude < 0]
geonet_sub = geonet_sub[(geonet_sub.longitude < 190) & (geonet_sub.longitude >155)]
geonet_sub = geonet_sub[(geonet_sub.latitude < -15)]
event_ids = geonet_sub.publicid.astype('str').values

df = df[(df.score_X >= 0.5) & (df.score_Y >= 0.5) & (df.score_Z >= 0.5)].reset_index(drop=True)
sorter = ['HN','BN','HH','BH','EH','SH']
# Create the dictionary that defines the order for sorting
sorterIndex = dict(zip(sorter, range(len(sorter))))

# Generate a rank column that will be used to sort
# the dataframe numerically
df['chan_rank'] = df['chan'].map(sorterIndex)

# Sorting is performed based on channel rank. Highest ranked channels are kept while
# the rest are dropped.
df.sort_values(['chan_rank'],
        ascending = [True], inplace = True)
# df.drop_duplicates('record',inplace=True)
# df.drop(columns=['record','chan_rank'],inplace=True)

df_HN = df[df.chan == 'HN']
# df_HN = df_HN.set_index(df_HN['event_id']+df_HN['station'])
df_HH = df[df.chan == 'HH']
# df_HH = df_HH.set_index(df_HH['event_id']+df_HH['station'])
df_BN = df[df.chan == 'BN']
df_BH = df[df.chan == 'BH']
# df_BH = df_BH.set_index(df_BH['event_id']+df_BH['station'])
df_EH = df[df.chan == 'EH']

df_HN_HH = df_HN.set_index(df_HN['event_id']+df_HN['station']).join(df_HH.set_index(df_HH['event_id']+df_HH['station'])[['score_X','score_Y','score_Z','chan_rank']],rsuffix='_HH')
df_HN_HH = df_HN_HH[~df_HN_HH.score_X_HH.isnull()]
df_HN_HH.chan_rank.unique()

df_HH_HN = df_HH.set_index(df_HH['event_id']+df_HH['station']).join(df_HN.set_index(df_HN['event_id']+df_HN['station'])[['score_X','score_Y','score_Z','chan_rank']],rsuffix='_HN')
df_HH_HN = df_HH_HN[~df_HH_HN.score_X_HN.isnull()]
df_HH_HN.chan_rank.unique()


basedir = '/Volumes/SeaJade 2 Backup/NZ/mseed_5_revised/'
newdir = '/Volumes/SeaJade 2 Backup/NZ/mseed_5_HN/'


def rename_xmls(basedir):
	# Small script to rename xml files so that they have the event name. This will reduce
	# read times for copy_gm_data if xml files are not already in the correct format.
	import glob
	import shutil
	import os

	xml_list = glob.glob(basedir+'/**/*.xml',recursive=True)
	for xml_file in xml_list:
		xml_dir = os.path.dirname(xml_file)
		event = op.read_events(xml_file)[0]
		evid = str(event.resource_id).split('/')[-1]
		out_file = xml_dir+'/'+evid+'.xml'
		shutil.move(xml_file,out_file)

def copy_gm_data(results):
    import glob
    import os

    for index, row in results.iterrows():
        try:
            date, time, sta, loc, chan = row.record_id.split('_')
            file_name = row.record_id+'.mseed'
            dt = datetime.datetime.strptime(date+time,'%Y%m%d%H%M%S')
            folderA = str(dt.year)
            folderB = dt.strftime('%m_%b')
            folderC = dt.strftime('%Y-%m-%d_%H%M%S')
            directory = basedir+folderA+'/'+folderB+'/'+folderC
            # Sometimes there can be rounding issues with the file path name, this should
            # check for any nearby files
            if not os.path.exists(directory):
                folderC = (dt - datetime.timedelta(seconds=1)).strftime('%Y-%m-%d_%H%M%S')
                directory = basedir+folderA+'/'+folderB+'/'+folderC
                file_name = (dt - datetime.timedelta(seconds=1)).strftime('%Y%m%d_%H%M%S'
                    )+'_'+sta+'_'+loc+'_'+chan+'.mseed'
            if not os.path.exists(directory):
                folderC = (dt + datetime.timedelta(seconds=1)).strftime('%Y-%m-%d_%H%M%S')
                directory = basedir+folderA+'/'+folderB+'/'+folderC			
                file_name = (dt + datetime.timedelta(seconds=1)).strftime('%Y%m%d_%H%M%S'
                    )+'_'+sta+'_'+loc+'_'+chan+'.mseed'
            file_path = basedir+folderA+'/'+folderB+'/'+folderC+'/mseed/data/'+file_name
            xml_path = basedir+folderA+'/'+folderB+'/'+folderC+'/*.xml'
            xml_name = os.path.basename(glob.glob(xml_path)[0])
            evid = xml_name.split('.')[0]
            if np.isin(row.event_id,event_ids):
                print(evid)
                new_file_path = newdir+folderA+'/'+folderB+'/'+folderC+'/mseed/data/'+file_name
                new_xml_path = newdir+folderA+'/'+folderB+'/'+folderC+'/'+xml_name
                if not os.path.exists(os.path.dirname(new_file_path)):
                    os.makedirs(os.path.dirname(new_file_path))
                copyfile(file_path,new_file_path)
                if not os.path.exists(new_xml_path):
                    copyfile(os.path.dirname(xml_path)+'/'+xml_name,new_xml_path)
        except FileNotFoundError:
            print(file_name+' not found!')
        except ValueError:
            print(file_name+' no instrument response')
        except KeyboardInterrupt:
            break
        except Exception as e:
            print('Exception',e.__class__,'occured for record',row.record_id)
			
			
cores = int(cpu_count()-1)
df_split = np.array_split(df_HH_HN, cores)
start_time = timeit.time()
with Pool(cores) as pool:
	pool.map(copy_gm_data,df_split)
pool.close()
pool.join()
end_time = timeit.time()-start_time

print('Took '+str(end_time)+' seconds to run')
print('All done!!!')




# df_HN_EH = df_HN.set_index(df_HN['event_id']+df_HN['station']).join(df_EH.set_index(df_EH['event_id']+df_EH['station'])[['score_X','score_Y','score_Z','chan_rank']],rsuffix='_EH')
# df_HN_EH = df_HN_EH[~df_HN_EH.score_X_EH.isnull()]
# df_HN_EH.chan_rank.unique()
# 
# df_EH_HN = df_EH.set_index(df_EH['event_id']+df_EH['station']).join(df_HN.set_index(df_HN['event_id']+df_HN['station'])[['score_X','score_Y','score_Z','chan_rank']],rsuffix='_HN')
# df_EH_HN = df_EH_HN[~df_EH_HN.score_X_HN.isnull()]
# df_EH_HN.chan_rank.unique()
# 
# 
# df_BN_HH = df_BN.set_index(df_BN['event_id']+df_BN['station']).join(df_HH.set_index(df_HH['event_id']+df_HH['station'])[['score_X','score_Y','score_Z','chan_rank']],rsuffix='_HH')
# df_BN_HH = df_BN_HH[~df_BN_HH.score_X_HH.isnull()]
# df_BN_HH.chan_rank.unique()
# 
# df_HH_BN = df_HH.set_index(df_HH['event_id']+df_HH['station']).join(df_BN.set_index(df_BN['event_id']+df_BN['station'])[['score_X','score_Y','score_Z','chan_rank']],rsuffix='_BN')
# df_HH_BN = df_HH_BN[~df_HH_BN.score_X_BN.isnull()]
# 
# 
# 
# df_BN_BH = df_BN.set_index(df_BN['event_id']+df_BN['station']).join(df_BH.set_index(df_BH['event_id']+df_BH['station'])[['score_X','score_Y','score_Z','chan_rank']],rsuffix='_BH')
# df_BN_BH = df_BN_BH[~df_BN_BH.score_X_BH.isnull()]
# 
# df_BH_BN = df_BH.set_index(df_BH['event_id']+df_BH['station']).join(df_BN.set_index(df_BN['event_id']+df_BN['station'])[['score_X','score_Y','score_Z','chan_rank']],rsuffix='_BN')
# df_BH_BN = df_BH_BN[~df_BH_BN.score_X_BN.isnull()]
# 
# 
# 
