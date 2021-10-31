# Filters and moves classified ground motion data based on specified parameters to a
# preferred file for pre-processing. The pre-processed data will then be used for GM IMs.

import pandas as pd
import datetime
import os
from shutil import copyfile
import obspy as op
from obspy.clients.fdsn import Client as FDSN_Client
from multiprocessing import Pool,cpu_count
import numpy as np
import time

basedir = '/Volumes/SeaJade 2 Backup/NZ/mseed_4_revised/'
newdir = '/Volumes/SeaJade 2 Backup/NZ/mseed_4_preferred/'
high_mag = 10
low_mag = 4

sorter = ['HN','BN','HH','BH','EH','SH']

filename = '/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/earthquakes-5.csv'
geonet = pd.read_csv(filename,low_memory=False)
geonet = geonet.sort_values('origintime')
geonet['origintime'] = pd.to_datetime(geonet['origintime'],format='%Y-%m-%dT%H:%M:%S.%fZ')
geonet = geonet.reset_index(drop=True)

# Filters data within a range of magnitudes so that we don't end up redoing lots of good
# work already completed at higher magnitudes!
geonet_sub = geonet[(geonet.magnitude >= low_mag) & (geonet.magnitude < high_mag)].reset_index(drop=True)
# Make sure data is from an earthquake
geonet_sub.loc[geonet_sub.longitude < 0, 'longitude'] = 360 + geonet_sub.longitude[geonet_sub.longitude < 0]
geonet_sub = geonet_sub[(geonet_sub.longitude < 190) & (geonet_sub.longitude >155)]
geonet_sub = geonet_sub[(geonet_sub.latitude < -15)]


# geonet_sub = geonet_sub[geonet_sub.eventtype == 'earthquake'].reset_index(drop=True)
event_ids = geonet_sub.publicid.astype('str').values
# client_NZ = FDSN_Client("GEONET")
# client_IU = FDSN_Client('IRIS')
# inventory_NZ = client_NZ.get_stations(level='response')
# inventory_IU = client_IU.get_stations(network='IU',station='SNZO',level='response')
# inventory = inventory_NZ+inventory_IU


gmc_results = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/gmc_record/mseed_4_results_revised/results.csv',low_memory=False)
# gmc_results = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/gmc_record/mseed_6_results/results.csv',low_memory=False)
gmc_results = gmc_results[(gmc_results.score_X >= 0.5) & (gmc_results.score_Y >= 0.5) & (gmc_results.score_Z >= 0.5)]
gmc_results['chan'] = gmc_results.record_id.str[-2::]
gmc_results['record'] = gmc_results.record_id.str[0:-6]

# Create the dictionary that defines the order for sorting
sorterIndex = dict(zip(sorter, range(len(sorter))))

# Generate a rank column that will be used to sort
# the dataframe numerically
gmc_results['chan_rank'] = gmc_results['chan'].map(sorterIndex)

# Sorting is performed based on channel rank. Highest ranked channels are kept while
# the rest are dropped.
gmc_results.sort_values(['chan_rank'],
        ascending = [True], inplace = True)
gmc_results.drop_duplicates('record',inplace=True)
gmc_results.drop(columns=['record','chan_rank'],inplace=True)


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
df_split = np.array_split(gmc_results, cores)

start_time = time.time()
with Pool(cores) as pool:
	pool.map(copy_gm_data,df_split)
pool.close()
pool.join()
end_time = time.time()-start_time

# print('Took '+str(end_time)+' seconds to run year '+str(year))
print('All done!!!')