import pandas as pd
import numpy as np
# from geopy.distance import geodesic
from numba import jit
import pickle
import datetime as dt
import os
from scipy.ndimage import zoom
import sys
import ray
import scipy

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def handshakes(x): 
     x = ((x-1)*x)/2 
     return int(x)

def max_tt_diff(ptraveltimes,straveltimes):
    max_diffs = np.zeros(handshakes(ptraveltimes.shape[0]))
    ii = 0
    for i in range(ptraveltimes.shape[0]-1):
        for j in range(i+1,ptraveltimes.shape[0]):
#             max_diffs[ii] = np.max((straveltimes[j]-ptraveltimes[j]) - (straveltimes[i]-ptraveltimes[i]))
            max_diffs[ii] = np.max(straveltimes[j]-straveltimes[i])
            progress(ii, handshakes(ptraveltimes.shape[0]), status='')
            ii += 1
    return max_diffs.max()

# base_phase_type = row.phase
# in_sta = row.sta_index
# in_phase_time = row['time']
# p_phases = p_phases.values
# s_phases = s_phases.values 
# time_allow_window = time_window
# threshold_phase_count = phase_threshold

def find_phase_matches(base_phase_type, in_sta, in_phase_time, p_phases, s_phases, ptraveltimes, straveltimes,
                       time_allow_window, threshold_phase_count):
	
    phase_count_per_cell = np.zeros(list(ptraveltimes[in_sta].shape))

    if base_phase_type.upper() == 'P':
        origin_times = in_phase_time - ptraveltimes[in_sta]
        for j in range(1,len(p_phases)):
            times = origin_times + ptraveltimes[int(p_phases[j,0])]
            obdiff = times - p_phases[j,1]
            phase_count_per_cell[abs(obdiff) < time_allow_window * (1 / 3)] += 5
            phase_count_per_cell[abs(obdiff) < time_allow_window * 0.5] += 4
            phase_count_per_cell[abs(obdiff) < time_allow_window] += 3
            phase_count_per_cell[abs(obdiff) < time_allow_window * 2] += 2
            phase_count_per_cell[abs(obdiff) < time_allow_window * 3] += 1

        for j in range(len(s_phases)):
            times = origin_times + straveltimes[int(s_phases[j,0])]
            obdiff = times - s_phases[j,1]
            phase_count_per_cell[abs(obdiff) < time_allow_window * (1 / 3)] += 5
            phase_count_per_cell[abs(obdiff) < time_allow_window * 0.5] += 4
            phase_count_per_cell[abs(obdiff) < time_allow_window] += 3
            phase_count_per_cell[abs(obdiff) < time_allow_window * 2] += 2
            phase_count_per_cell[abs(obdiff) < time_allow_window * 3] += 1

    if base_phase_type.upper() == 'S':
        origin_times = in_phase_time - straveltimes[in_sta]
        for j in range(len(p_phases)):
            times = origin_times + ptraveltimes[int(p_phases[j,0])]
            obdiff = times - p_phases[j,1]
            phase_count_per_cell[abs(obdiff) < time_allow_window * (1 / 3)] += 5
            phase_count_per_cell[abs(obdiff) < time_allow_window * 0.5] += 4
            phase_count_per_cell[abs(obdiff) < time_allow_window] += 3
            phase_count_per_cell[abs(obdiff) < time_allow_window * 2] += 2
            phase_count_per_cell[abs(obdiff) < time_allow_window * 3] += 1

        for j in range(1,len(s_phases)):
            times = origin_times + straveltimes[int(s_phases[j,0])]
            obdiff = times - s_phases[j,1]
            phase_count_per_cell[abs(obdiff) < time_allow_window * (1 / 3)] += 5
            phase_count_per_cell[abs(obdiff) < time_allow_window * 0.5] += 4
            phase_count_per_cell[abs(obdiff) < time_allow_window] += 3
            phase_count_per_cell[abs(obdiff) < time_allow_window * 2] += 2
            phase_count_per_cell[abs(obdiff) < time_allow_window * 3] += 1

#     phase_count_per_cell = p_phases_array + s_phases_array
    maxi = np.where(phase_count_per_cell == phase_count_per_cell.max())
    
    event_times = origin_times[maxi]
    p_phase_list = np.zeros((len(p_phases)))
    s_phase_list = np.zeros((len(s_phases)))

    maxi_stack = np.zeros(len(maxi[0]))

    for i in range(len(p_phases)):
        ots = p_phases[i,1]-event_times
        tht = ptraveltimes[int(p_phases[i,0])][maxi]
        if np.any(abs(ots - tht) <= time_allow_window):
            p_phase_list[i] = p_phases[i,0]
            maxi_stack += abs(ots - tht) <= time_allow_window
        else:
            p_phase_list[i] = -1
    #
    for i in range(len(s_phases)):
        ots = s_phases[i,1]-event_times
        tht = straveltimes[int(s_phases[i,0])][maxi]
        if np.any(abs(ots - tht) <= time_allow_window):
            s_phase_list[i] = s_phases[i,0]
            maxi_stack += abs(ots - tht) <= time_allow_window
        else:
            s_phase_list[i] = -1
    
#     p_count = np.sum(p_phase_list > -1)
#     s_count = np.sum(s_phase_list > -1)
# 
#     if p_count >= s_count:
#     	count = p_count
#     else:
#     	count = s_count
    uniques,unique_idx = np.unique(p_phase_list,return_index=True)
    p_indices_final = unique_idx[uniques != -1]
#     p_indices_final = unique_idx[np.where(uniques != -1)[0]]

    uniques,unique_idx = np.unique(s_phase_list,return_index=True)
    s_indices_final = unique_idx[uniques != -1]
#     s_indices_final = unique_idx[np.where(uniques != -1)[0]]

#     x,y,z = np.median(maxi_refined[0]),np.median(maxi_refined[1]),np.median(maxi_refined[2])
#     ex,ey,ez = np.std(maxi_refined[0]),np.std(maxi_refined[1]),np.std(maxi_refined[2])
    x,y,z = np.median(maxi[0]),np.median(maxi[1]),np.median(maxi[2])
    ex,ey,ez = np.std(maxi[0]),np.std(maxi[1]),np.std(maxi[2])
    return p_indices_final, s_indices_final, x, y, z, ex, ey, ez

@ray.remote
def assoc_multi(phases, ptraveltimes, straveltimes, time_window, phase_threshold, max_traveltime):
    phase_sub = phases.copy()
    total = len(phase_sub)
    print_progress = 0
    if phase_sub.iloc[0].name == 0: # Only run progress bar on first core
        ii = 0
        print_progress = 1
    for index, row in phase_sub.iterrows():
#         print(index)
#         print(pd.to_datetime(row.time, unit='s'))
        p_phases = phase_sub[(phase_sub.phase.str.upper() == 'P') &
                          (phase_sub['time'] < row['time'] + max_traveltime) &
                          (phase_sub['time'] >= row['time'])][['sta_index', 'time']]

        s_phases = phase_sub[(phase_sub.phase.str.upper() == 'S') &
                          (phase_sub['time'] < row['time'] + max_traveltime) &
                          (phase_sub['time'] >= row['time'])][['sta_index', 'time']]

        phase_count = p_phases.sta_index.nunique() + s_phases.sta_index.nunique()
        if p_phases.sta_index.nunique() >= s_phases.sta_index.nunique():
        	sta_count = p_phases.sta_index.nunique()
        else:
        	sta_count = s_phases.sta_index.nunique()
#         print(phase_count)

        if phase_count >= phase_threshold:
#             print(row.phase, row.sta, row.time, p_phases.values.shape, s_phases.values.shape)
            p_phasers, s_phasers, x, y, z, ex, ey, ez = find_phase_matches(row.phase, row.sta_index, 
            	row['time'], p_phases.values, s_phases.values, ptraveltimes, straveltimes, 
            	time_window * 2, phase_threshold)
#             break
#             p_phasers, s_phasers, x, y, z, ex, ey, ez = find_phase_matches(row.phase, row.sta, 
#             	row.time, p_phases.values, s_phases.values, p_zoom, s_zoom, 
#             	time_window, phase_threshold)

#             p_final_phases = p_phases[p_phasers]
#             s_final_phases = s_phases[s_phasers]
            p_final_phases = p_phases.iloc[p_phasers]
            s_final_phases = s_phases.iloc[s_phasers]
            ids = p_final_phases.index.tolist() + s_final_phases.index.tolist()
            matched_phases = phase_sub.loc[ids]

            if len(p_final_phases) >= len(s_final_phases):
                sta_count = len(p_final_phases)
            else:
                sta_count = len(s_final_phases)
            phase_count = len(matched_phases)
#             print('...boop',index,phase_count)

            if phase_count >= phase_threshold:
                print(matched_phases)
#                 print(index,str(phase_count)+' phase_sub')
#                 print(np.array((x,y,z,ex,ey,ez))/zoom_factor)
                phase_sub.loc[matched_phases[matched_phases.phase_count < phase_count].index, 'phase_count'] = phase_count
                phase_sub.loc[matched_phases[matched_phases.phase_count < phase_count].index, 'base_id'] = row['time']
                phase_sub.loc[matched_phases[matched_phases.phase_count < phase_count].index, 'x'] = x
                phase_sub.loc[matched_phases[matched_phases.phase_count < phase_count].index, 'y'] = y
                phase_sub.loc[matched_phases[matched_phases.phase_count < phase_count].index, 'z'] = z
                phase_sub.loc[matched_phases[matched_phases.phase_count < phase_count].index, 'ex'] = ex
                phase_sub.loc[matched_phases[matched_phases.phase_count < phase_count].index, 'ey'] = ey
                phase_sub.loc[matched_phases[matched_phases.phase_count < phase_count].index, 'ez'] = ez
        if print_progress:
            progress(ii, total, status='')
            ii += 1
    return phase_sub

@ray.remote
def multiprocess_events(i, sta_cat, phaseser, unique_events, ptraveltimes, straveltimes, lowq,
                        highq, Q_threshold, terr, outlier, x_spacing, y_spacing, z_spacing, 
                        coarsest_x_spacing, coarsest_y_spacing, coarsest_z_spacing, x_min, y_min, z_min, phase_threshold,
                        tt_fine_dir, zoom_factor):
    from relocator_6 import MAXI_locate_3D, uncertainty_calc, maxiscan
#     for i in range(len(unique_events)):
    std_factor = 2

    p_out = []
    event_id = unique_events[i]   
    print('...Initiating Event No. '+str(event_id))

    pick_array = np.full((1, len(sta_cat), 2), -1.00)

    picks = phaseser[phaseser.base_id == unique_events[i]].reset_index(drop=True)

    picks = picks.drop_duplicates(subset=['sta', 'phase'], keep='first') # ensures no duplicated phases are input into location algorithm

    p_picks = picks[picks.phase.str.upper() == 'P'][['sta', 'time']].copy(deep=True).drop_duplicates().reset_index(drop=True)
    p_picks.columns = ['sta', 'time']

    s_picks = picks[picks.phase.str.upper() == 'S'][['sta', 'time']].copy(deep=True).drop_duplicates().reset_index(drop=True)
    s_picks.columns = ['sta', 'time']


    pick_array[0, :, 0] = pd.merge(sta_cat.reset_index()[['index', 'sta']], p_picks, how='left',
                                   on='sta')['time'].fillna(-1)
    pick_array[0, :, 1] = pd.merge(sta_cat.reset_index()[['index', 'sta']], s_picks, how='left',
                                   on='sta')['time'].fillna(-1)


    ponsets = pick_array[:, :, 0]
    sonsets = pick_array[:, :, 1]

    origin = MAXI_locate_3D(event_id, ponsets, sonsets, ptraveltimes, straveltimes, lowq, highq, Q_threshold,
                            terr, outlier, x_spacing, y_spacing, z_spacing, 
                            coarsest_x_spacing, coarsest_y_spacing, coarsest_z_spacing, x_min, y_min, z_min, phase_threshold, tt_fine_dir,
                            sta_cat, zoom_factor, std_factor)

    ### Make association df
    if origin[2]:
        p_out = picks.copy()
        ### Mask based on sta_id and arr_time; sometimes the arr_time can be the same at 2
        ### stations
        p_mask = (p_out.time.isin(origin[2][0][2])) & (p_out.sta_index.isin(origin[2][0][0]) & (p_out.phase == 'P'))
        s_mask = (p_out.time.isin(origin[2][0][3])) & (p_out.sta_index.isin(origin[2][0][1]) & (p_out.phase == 's'))
        p_out.loc[p_mask, 'reloc'] = 'MAXI'
        p_out.loc[s_mask, 'reloc'] = 'MAXI'
        p_out.loc[p_mask, 't_res'] = origin[2][0][4]
        p_out.loc[s_mask, 't_res'] = origin[2][0][5]
        p_out.drop(columns=['time', 'sta_index', 'phase_count', 'base_id','x', 'y', 'z', 
            'ex', 'ey', 'ez'], inplace=True)
        ### Drop rows with blank entries
        p_out = p_out.dropna(axis=0,subset=['reloc']).reset_index(drop=True)

    if origin[3]:
        unc_out = origin[3][0]
    else:
        unc_out = []

    return origin[0:2], p_out, event_id, unc_out

def calculate_location(catalog_df,depth_calibrate):
    import utm
    xs = catalog_df['x'].values - 1200
    ys = catalog_df['y'].values - 1200
    depths = catalog_df['z'].values - 15
    angle = -140
    
    depths = catalog_df.z + depth_calibrate
    return lats,lons,depths

def get_phases(df,date):
    import glob
    import pandas as pd
    import datetime as dt
    import os

    df['sta'] = df['sta'].str.strip()
    ### Subset the pick files based on the chosen date
    df = df[(pd.to_datetime(df.datetime) > date-pd.Timedelta(5, unit='m')) & (pd.to_datetime(df.datetime) <= date+pd.Timedelta(24, unit='h'))]

    df_p = df[df.phase.astype(str).str.lower().str[0] == 'p'].reset_index(drop=True)
    df_p['orig_phase'] = df_p.phase
    df_p['phase'] = 'P'
    df_p['time'] = (
            pd.to_datetime(df_p['datetime']).astype('datetime64[ns]') - dt.datetime(1970, 1, 1)).dt.total_seconds()

    df_s = df[df.phase.astype(str).str.lower().str[0] == 's'].reset_index(drop=True)
    df_s['orig_phase'] = df_s.phase
    df_s['phase'] = 's'
    df_s['time'] = (
            pd.to_datetime(df_s['datetime']).astype('datetime64[ns]') - dt.datetime(1970, 1, 1)).dt.total_seconds()

    white_list = [os.path.basename(x)[0:-6] for x in glob.glob('inputs/tt/*_P.hdf')]

    phases = pd.concat([df_p, df_s]).sort_values('time').reset_index(drop=True)
    phases = phases[phases.sta.str.contains("|".join(white_list))]

    sta_cat = phases[['net', 'sta']].drop_duplicates().reset_index(drop=True)
    # 	sta_cat = sta_cat.merge(site_df[['sta','lat','lon','elev']],on='sta')
    # 	sta_cat['elev'] = sta_cat.elev/1000
    # 	sta_cat = phases[['net','sta','chan','lat','lon','elev']].drop_duplicates().reset_index(drop=True)
    sta_cat['sta'] = sta_cat.sta.str.strip()  # Remove unnecessary spaces in station name
    sta_cat_index = sta_cat['sta'].reset_index()

    phases = pd.merge(phases, sta_cat_index, on='sta', how='inner')
    phases.rename(columns={'index': 'sta_index'}, inplace=True)
    phases['base_id'] = 0
    phases['phase_count'] = 0
    phases = phases.sort_values('time')
    phases = phases.reset_index(drop=True)
    phases['x'] = 0
    phases['y'] = 0
    phases['z'] = 0
    phases['ex'] = 0
    phases['ey'] = 0
    phases['ez'] = 0

    return phases,sta_cat

def get_vmodels(sta_cat,zoom_factor):
	import numpy as np
	import h5py
	
	print('Loading p-traveltime models')
	p_files=[]
	for sta in sta_cat['sta']:
		sta = sta.strip()
		file = sta+'_P.hdf'			
		p_files.append(file)
	ptraveltimes = np.array([h5py.File('inputs/tt/'+fname,'r')['values'] for fname in p_files])
	x_min, y_min, z_min = np.array(h5py.File('inputs/tt/'+p_files[0],'r')['min_coords'])
	x_spacing, y_spacing, z_spacing = np.array(h5py.File('inputs/tt/'+p_files[0],'r')['node_intervals'])

	print('Loading s-traveltime models')    
	s_files=[]
	for sta in sta_cat['sta']:
		sta = sta.strip()
		file = sta+'_S.hdf'
		s_files.append(file)
	straveltimes = np.array([h5py.File('inputs/tt/'+fname,'r')['values'] for fname in s_files])

	p_zoom_files = []
	s_zoom_files = []
	
	print('Creating new zoomed tt models')
	for i,sta in enumerate(sta_cat['sta']):
		if not os.path.exists('inputs/tt_zoomed'):
			os.makedirs('inputs/tt_zoomed')
		if not os.path.exists('inputs/tt_zoomed/'+sta+'_P.npy'):
			p_zoom = zoom(ptraveltimes[i],(zoom_factor,zoom_factor,zoom_factor))
			np.save('inputs/tt_zoomed/'+sta+'_P.npy',p_zoom)
			p_zoom_files.append(sta+'_P.npy')
		else:
			p_zoom_files.append(sta+'_P.npy')
		if not os.path.exists('inputs/tt_zoomed/'+sta+'_S.npy'):
			s_zoom = zoom(straveltimes[i],(zoom_factor,zoom_factor,zoom_factor))
			np.save('inputs/tt_zoomed/'+sta+'_S.npy',s_zoom)
			s_zoom_files.append(sta+'_S.npy')
		else:
			s_zoom_files.append(sta+'_S.npy')

	print('Zooming p-traveltime models')
	p_zoom = np.array([np.load('inputs/tt_zoomed/'+fname) for fname in p_zoom_files])

	print('Zooming s-traveltime models')
	s_zoom = np.array([np.load('inputs/tt_zoomed/'+fname) for fname in s_zoom_files])
	return ptraveltimes, straveltimes, p_zoom, s_zoom, x_spacing, y_spacing, z_spacing
	
def get_vmodels_zoom(sta_cat,zoom_factor):
    import numpy as np
    import h5py

    p_zoom_files = []
    s_zoom_files = []

    print('Zooming p-traveltime models')
    p_files = []
    for sta in sta_cat['sta']:
        sta = sta.strip()
        file = sta + '_P.hdf'
        p_files.append(file)
        if not os.path.exists('inputs/tt_zoomed'):
            os.makedirs('inputs/tt_zoomed')
        if not os.path.exists('inputs/tt_zoomed/'+sta+'_P.npy'):
            with h5py.File('inputs/tt/' + file, 'r') as data:
                p_vel = data['values']
                p_zoom = zoom(p_vel,(zoom_factor,zoom_factor,zoom_factor))
                np.save('inputs/tt_zoomed/'+sta+'_P.npy',p_zoom)
                p_zoom_files.append(sta+'_P.npy')
        else:
            p_zoom_files.append(sta+'_P.npy')

    print('Zooming s-traveltime models')
    s_files = []
    for sta in sta_cat['sta']:
        sta = sta.strip()
        file = sta + '_S.hdf'
        s_files.append(file)
        if not os.path.exists('inputs/tt_zoomed'):
            os.makedirs('inputs/tt_zoomed')
        if not os.path.exists('inputs/tt_zoomed/'+sta+'_S.npy'):
            with h5py.File('inputs/tt/' + file, 'r') as data:
                s_vel = np.array(data['values'])
                s_zoom = zoom(s_vel,(zoom_factor,zoom_factor,zoom_factor))
                np.save('inputs/tt_zoomed/'+sta+'_S.npy',s_zoom)
                s_zoom_files.append(sta+'_S.npy')
        else:
            s_zoom_files.append(sta+'_S.npy')

    x_min, y_min, z_min = np.array(h5py.File('inputs/tt/' + p_files[0], 'r')['min_coords'])
    x_spacing, y_spacing, z_spacing = np.array(h5py.File('inputs/tt/' + p_files[0], 'r')['node_intervals'])

    print('Loading zoomed p-traveltime models')
    # 	p_zoom = zoom(ptraveltimes,(1,zoom_factor,zoom_factor,zoom_factor))
    p_zoom = np.array([np.load('inputs/tt_zoomed/'+fname) for fname in p_zoom_files])

    print('Loading zoomed s-traveltime models')
    # 	s_zoom = zoom(straveltimes,(1,zoom_factor,zoom_factor,zoom_factor))
    s_zoom = np.array([np.load('inputs/tt_zoomed/'+fname) for fname in s_zoom_files])

    return p_zoom, s_zoom, x_spacing, y_spacing, z_spacing

def rotate_back(orilat, orilon, xs, ys, angle):
	"""
	Rotate a point counterclockwise by a given angle around a given origin.

	The angle should be given in radians.
	"""
	from pyproj import CRS
	from pyproj import Transformer
	import math
	import numpy as np

	angle = np.radians(angle)
# 	ox = orilon
# 	oy = orilat
	transformer_from_latlon = Transformer.from_crs(4326, 2193) # WSG84 to New Zealand NZDG2000 coordinate transform
	transformer_to_latlon = Transformer.from_crs(2193, 4326)
	
	ox, oy = transformer_from_latlon.transform(orilat,orilon)
	px = ox+xs*1000
	py = oy-ys*1000
# 	px, py = transformer_from_latlon.transform(lats,lons)

	qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
	qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
	
	lats, lons = transformer_to_latlon.transform(qx,qy)
# 	stations['z'] = (stations.elevation-dx)/dx
	return lats, lons

def main(df,date,event_output_dir, arrival_output_dir,assoc_output_dir, uncertainty_output_dir):
    from multiprocessing import Pool
    import multiprocessing
    from itertools import repeat
    import time as timeit
    import json

#     date = pd.to_datetime('2019-07-01')
    zoom_factor = 0.5 # Start with a coarser velocity model, grid spacing will be increased
                      # by 1/zoom_factor, so that 1/0.5 will double grid spacing.
    terr = 0.7 # location terr
    lowq = 1
    highq = 1
    Q_threshold = 0.3
    outlier = 0.7 # the time error allowable for an associated phase during location.
    phase_threshold = 6 # the minimum number of stations required for the event association phase
    time_window = 0.7 # the time +/- to search for matching phases for the event association phase. Greater time will allow more phases but may lead to issues in the location phase.
    x_min, y_min, z_min = -1200, -1200, -15 # First X, Y, Z coordinate in the v-model coordinate system
    orilat = -41.7638 # Origin latitude
    orilon = 172.9037 # Origin longitude
    angle = -140 # Counter-clockwise rotation for restoration of true coordinates
    tt_fine_dir = '/Volumes/SeaJade2/Pykonal/tt_fine'

    ### Create phase list
    phases,sta_cat = get_phases(df,date)
    ### Get vmodels
    ptraveltimes, straveltimes, x_spacing, y_spacing, z_spacing = get_vmodels_zoom(sta_cat, zoom_factor)
    fine_x_spacing, fine_y_spacing, fine_z_spacing = 3, 3, 2
    coarsest_x_spacing, coarsest_y_spacing, coarsest_z_spacing = np.array((x_spacing, y_spacing, z_spacing)) / zoom_factor

    # 	print('Calculating max traveltime from given models:')
    # 	max_traveltime = max_tt_diff(ptraveltimes,straveltimes) # Use S-velocity because it is slower!

    ### Check for existing preliminarily associated arrivals
    if os.path.exists('./output/' + arrival_output_dir + '/' + date.strftime('%Y%m%d') + '_arrivals.csv'):
        print('Loading associated phases')
        phase_output = pd.read_csv('./output/' + arrival_output_dir + '/' + date.strftime('%Y%m%d') + '_arrivals.csv',low_memory=False)
    else:    
    ### Split and calculate phase data
        max_traveltime = 300 # Determined from stations that are near each other...
        print('Maximum traveltime is '+str(max_traveltime))
        ray.init(object_store_memory=30000 * 1024 * 1024)
        ptraveltimes_id = ray.put(ptraveltimes)
        straveltimes_id = ray.put(straveltimes)    
        print('Associating phases:')
        start_time = timeit.time()
        cores = int(multiprocessing.cpu_count()-1)
        split_count = int((phases['time'].values[-1] - phases['time'].values[0]) // max_traveltime) + 1
        if split_count > cores:
            split_count == cores
        df_split = np.array_split(phases, split_count)
        for i in range(0,len(df_split)):
#             df_split[i] = phases[(phases['time'] >= df_split[i]['time'].values[0]-max_traveltime) & 
#                 (phases['time'] <= df_split[i]['time'].values[-1])]
            df_split[i] = phases[(phases['time'] >= df_split[i]['time'].values[0]) & 
                (phases['time'] <= df_split[i]['time'].values[-1]+max_traveltime)]
        result_ids = []
        for i in range(len(df_split)):
            result_ids.append(assoc_multi.remote(df_split[i],ptraveltimes_id,straveltimes_id,
                time_window,phase_threshold,max_traveltime))
        phase_output = pd.concat(ray.get(result_ids))
        print("--- %s seconds ---" % (timeit.time() - start_time))
        ray.shutdown()

        phase_counts = phase_output.groupby('base_id').time.count().reset_index()
        phase_counts = phase_counts[phase_counts.base_id > 0]
        phase_counts.rename(columns = {'time':'phase_count'}, inplace = True)
        phase_counts = phase_counts[phase_counts.phase_count >= phase_threshold]

        phase_output = phase_output[phase_output.base_id.isin(phase_counts.base_id)]
        phase_output = phase_output.sort_values('phase_count')
        phase_output = phase_output[~phase_output.index.duplicated(keep='last')]
        phase_output = pd.merge(phase_output, phase_counts, on='base_id', how='right')	
        phase_output['phase_count'] = phase_output.phase_count_y
        phase_output = phase_output.drop(['phase_count_x','phase_count_y'],axis=1)
        phase_output = phase_output.sort_values('base_id')

        ### Write phase associations to a file for backup/further testing without reprocessing
        if not os.path.exists('./output/' + arrival_output_dir):
            os.makedirs('./output/' + arrival_output_dir)
        phase_output.to_csv('./output/' + arrival_output_dir + '/' + date.strftime('%Y%m%d') + '_arrivals.csv', index=None)
    
    phaseser = phase_output.copy()
    
    unique_events = phaseser.base_id.unique()
    print('...Total unique events is ' + str(len(unique_events)))

    ### Calculate event data    
    result_ids = []
    start_time = timeit.time()
    ray.init(object_store_memory=30000 * 1024 * 1024)
    ptraveltimes_id = ray.put(ptraveltimes)
    straveltimes_id = ray.put(straveltimes)    
    print('--- Calculating event locations ---')
    for i in range(len(unique_events)):
#     for i in range(0,1):
        result_ids.append(multiprocess_events.remote(i, sta_cat, phaseser, unique_events,
                                                     ptraveltimes_id, straveltimes_id, lowq, highq, Q_threshold, terr,
                                                     outlier, x_spacing, y_spacing, z_spacing, coarsest_x_spacing, coarsest_y_spacing, coarsest_z_spacing, 
                                                     x_min, y_min, z_min, phase_threshold, tt_fine_dir, zoom_factor))
    origins = ray.get(result_ids)
    print("--- %s seconds ---" % (timeit.time() - start_time))

    catalog_df = pd.DataFrame(columns=['x', 'y', 'z', 'datetime', 'minimum', 'finalgrid', 'ndef', 'evid', 'x_c', 'y_c',
                                       'z_c', 'x_err', 'y_err', 'z_err', 'theta', 'Q'])
    assoc_df = pd.DataFrame(columns=['arid', 'net', 'sta', 'loc', 'chan', 'phase', 'datetime', 't_res', 'evid',
                                       'orig_phase'])

    uncertainty_data = {}

    for origin in origins:
        event_id = str(pd.to_datetime(date).strftime('%Y%m%d')+'p'+str(len(catalog_df)+1)) 
        if len(origin[0][1]) != 0:

            if origin[0][1][0][5] != -1:  # bypass any low quality events. Should be very rare events.

                # return origin

                if origin[0][1][0][10] >= origin[0][1][0][11]:
                    major = origin[0][1][0][10]
                    minor = origin[0][1][0][11]
                else:
                    major = origin[0][1][0][11]
                    minor = origin[0][1][0][10]
                                        
                origin_data = {'x': origin[0][1][0][1], 'y': origin[0][1][0][2], 'z': origin[0][1][0][3],
                               'datetime': pd.to_datetime(origin[0][1][0][0], unit='s'), 'minimum': origin[0][1][0][4],
                               'finalgrid': origin[0][1][0][5], 'ndef': origin[0][1][0][6], 'evid': event_id,
                               'x_c': origin[0][1][0][7], 'y_c': origin[0][1][0][8], 'z_c': origin[0][1][0][9],
                               'major': major, 'minor': minor, 'z_err': origin[0][1][0][12],
                               'theta': origin[0][1][0][13], 'Q': origin[0][1][0][14]}

                uncertainty_data[event_id] = origin[3].tolist()
                origin[1]['evid'] = event_id
                origin[1]['arid'] = origin[1]['evid'] + 'a' + (origin[1].index.values + 1).astype('str')
                catalog_df = catalog_df.append(origin_data, True)
                assoc_df = assoc_df.append(origin[1], True)

        elif origin[0][0][0][0] == -1:

            print('Nothing here, pass')
            pass

        else:

            print('Skipping low quality event! ' + str(origin[2]))

    catalog_df['x'] = (catalog_df.x * fine_x_spacing) + x_min
    catalog_df['y'] = (catalog_df.y * fine_y_spacing) + y_min
    catalog_df['z'] = (catalog_df.z * fine_z_spacing) + z_min
    catalog_df['depth'] = catalog_df['z']
    catalog_df['lat'], catalog_df['lon'] = rotate_back(orilat, orilon, catalog_df.x.values,
                                                       catalog_df.y.values, angle)
#     catalog_df['x_c'] = (catalog_df.x_c * fine_x_spacing) + x_min
#     catalog_df['y_c'] = (catalog_df.y_c * fine_y_spacing) + y_min
#     catalog_df['z_c'] = (catalog_df.z_c * fine_z_spacing) + z_min
    catalog_df['y_c'], catalog_df['x_c'] = rotate_back(orilat, orilon, catalog_df.x_c.values,
                                                       catalog_df.y_c.values, angle)
    theta = list(catalog_df.loc[catalog_df['theta'] > 0, 'theta'].values)
    catalog_df.loc[catalog_df['theta'] > 0, 'theta'] = np.rad2deg(theta)
    catalog_df['theta'] = catalog_df['theta'] + angle
    catalog_df.loc[catalog_df['theta'] < 0, 'theta'] = catalog_df.loc[catalog_df['theta'] < 0, 'theta'] + 360

    nsta = assoc_df[assoc_df.reloc == 'MAXI'].groupby(['evid'])['sta'].nunique()
    for ider, evid in catalog_df.iterrows():
        catalog_df.loc[ider, 'nsta'] = nsta.loc[evid.evid]
    catalog_df['nsta'] = catalog_df['nsta'].astype('int')
    catalog_df['reloc'] = 'MAXI'

    if not os.path.exists('./output/' + event_output_dir):
        os.makedirs('./output/' + event_output_dir)
    if not os.path.exists('./output/' + assoc_output_dir):
        os.makedirs('./output/' + assoc_output_dir)
    if not os.path.exists('./output/' + uncertainty_output_dir):
        os.makedirs('./output/' + uncertainty_output_dir)

    if len(catalog_df) != 0:
        #
        catalog_df = catalog_df.sort_values('datetime').reset_index(drop=True)
        catalog_df = catalog_df[['evid', 'datetime', 'lat', 'lon', 'depth', 'ndef', 'nsta', 'reloc', 'minimum',
                                 'finalgrid', 'x', 'y', 'z', 'x_c', 'y_c', 'z_c', 'major', 'minor', 'z_err', 'theta',
                                 'Q']]
        assoc_df = assoc_df.sort_values('datetime').reset_index(drop=True)
        assoc_df = assoc_df[['arid', 'datetime', 'net', 'sta', 'loc', 'chan',
                                 'orig_phase', 'evid', 't_res', 'reloc']]
        assoc_df.rename(columns={'orig_phase': 'phase'}, inplace=True)

        catalog_df.to_csv('./output/' + event_output_dir + '/' + date.strftime('%Y%m%d') + '_origins.csv', index=None)
        assoc_df.to_csv('./output/' + assoc_output_dir + '/' + date.strftime('%Y%m%d') + '_assocs.csv',
                          index=None)
        json = json.dumps(uncertainty_data)
        with open('./output/' + uncertainty_output_dir + '/' + date.strftime('%Y%m%d') + '_uncertainties.json','w') as f:
            f.write(json)
    ray.shutdown()

if __name__ == "__main__":
    import glob
    
    phase_file = '/Volumes/SeaJade 2 Backup/NZ/EQTransformer/inputs/phases/phase_test.csv'
    event_output_dir = 'catalog_east_cape_test'
    arrival_output_dir = 'arrivals_east_cape_test'
    assoc_output_dir = 'associations_east_cape_test'
    uncertainty_output_dir = 'uncertainties_east_cape_test'
    phase_df = pd.read_csv(phase_file,low_memory=False)
    phase_df = pd.read_csv(phase_file,
        header=None,sep='\s+',low_memory=False,names=['net','sta','chan','phase','datetime'],index_col=False)

    # Subset the pick files based on the chosen date
    start_date = '2021-03-04'
    end_date = '2021-03-04'
    date_range = pd.date_range(start_date, end_date)
    
    for date in date_range:
#         print('Relocating events from ' + str(date))
        print('Locating events from ' + str(date))
        phase_df_check = phase_df[(pd.to_datetime(phase_df.datetime) > date-pd.Timedelta(5, unit='m')) & (pd.to_datetime(phase_df.datetime) <= date+pd.Timedelta(24, unit='h'))]
        if phase_df_check.size == 0:
            print('There are no phases for ' + str(date))
            continue
        else:
            ### Associate data and compute events
            main(phase_df,date,event_output_dir, arrival_output_dir, assoc_output_dir, uncertainty_output_dir)
            
            ### Calculate magnitude data
            import mag_calc
            event_file = './output/' + event_output_dir + '/' + date.strftime('%Y%m%d') + '_origins.csv'
            assoc_file = './output/' + assoc_output_dir + '/' + date.strftime('%Y%m%d') + '_assocs.csv'
            sta_corr = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/sta_corr_new.csv')
            out_directory = '/Volumes/SeaJade 2 Backup/NZ/EQTransformer/output/mag_out'
            mag_calc.calc_mags(event_file,assoc_file,sta_corr,out_directory)

# 	print('...Writing to QuakeML files...')
# 	from obspy import UTCDateTime
# 	from obspy.core.event import Arrival, Catalog, Event, Origin, Magnitude, StationMagnitude, Arrival, Pick, WaveformStreamID, Amplitude, CreationInfo
# 	
# # 	cat = Catalog()
# 	info = CreationInfo(author='jhutchinson',creation_time=UTCDateTime())
# # 	cat.creation_info = info
# 	
# 	for i,row in catalog_df.iterrows():
# 		if not os.path.exists('./output/qml/'+pd.to_datetime(row.datetime).strftime('%Y%m%d')):
# 			os.makedirs('./output/qml/'+pd.to_datetime(row.datetime).strftime('%Y%m%d'))
# 		break
# 		event = Event(resource_id = row.evid)
# 		origin = Origin(time=UTCDateTime(row.datetime),
# 						resource_id=row.evid,
# 						longitude=row.lon,
# 						latitude=row.lat,
# 						depth=row.depth,
# 						method="MAXI")
# 		mag_sub = magnitude[magnitude.evid == row.evid]
# 		mag = Magnitude(magnitude_type='cML',
# 							  mag_errors.uncertainty=row.cml_unc,
# 							  mag=row.cml,
# 							  station_count=len(mag_sub))
# 		event.magnitudes.append(mag)
# 		mag = Magnitude(magnitude_type='ML',
# 							  mag_errors.uncertainty=row.ml_unc,
# 							  mag=row.ml,
# 							  station_count=len(mag_sub))
# 		event.magnitudes.append(mag)
# 		picks = []
# 		arrs = []
# 		arrival_sub = arrival_df[arrival_df.evid == row.evid]
# 		for j,pick_row in arrival_sub.iterrows():
# 			p = Pick(time=UTCDateTime(pick_row.datetime), 
# 					waveform_id=WaveformStreamID(network_code=pick_row.net, 
# 						station_code=pick_row.sta, 
# 						location_code=pick_row['loc'], 
# 						channel_code=pick_row.chan),
# 					phase_hint=pick_row.phase, 
# 					method_id="EqTransformer")
# 			arr = Arrival(pick_id=p.resource_id,
# 						 phase=p.phase_hint,
# 						 azimuth=mag_sub[mag_sub.sta == pick_row.sta].az,
# 						 distance=mag_sub[mag_sub.sta == pick_row.sta].r_epi,
# 						 time_residual=pick_row.t_res)
# 			picks.append(p)
# 			origin.arrivals.append(arr)
# # 		origin.arrivals.append(arrs)
# 		event.origins.append(origin)
# 		mag_sub = mag_sub.dropna()
# 		mags = []
# 		amps = []
# 		for j,mag_row in mag_sub.iterrows():
# 			pick_info = next(sub for sub in [pick if pick.waveform_id.station_code in mag_row.sta else "" for pick in picks] if sub)
# 			amp = Amplitude(generic_amplitude=mag_row.amp,
# 							magnitude_hint='ML',
# 							pick_id=pick_info.resource_id,
# 							waveform_id=WaveformStreamID(network_code=pick_info.waveform_id.network_code, 
# 								station_code=pick_info.waveform_id.station_code,
# 								location_code=pick_info.waveform_id.location_code, 
# 								channel_code=pick_info.waveform_id.channel_code),
# 							unit='m/s')
# 			amps.append(amp)
# 			mag = StationMagnitude(mag=mag_row.ml,
# 							station_magnitude_type='ML',
# 							amplitude_id=amp.resource_id,
# 							origin_id=event.resource_id,
# 							waveform_id=WaveformStreamID(network_code=pick_info.waveform_id.network_code, 
# 								station_code=pick_info.waveform_id.station_code,
# 								location_code=pick_info.waveform_id.location_code,
# 								channel_code=pick_info.waveform_id.channel_code))
# 			mags.append(mag)
# 			mag = StationMagnitude(mag=mag_row.cml,
# 							station_magnitude_type='cML',
# 							amplitude_id=amp.resource_id,
# 							origin_id=event.resource_id,
# 							waveform_id=WaveformStreamID(network_code=pick_info.waveform_id.network_code, 
# 								station_code=pick_info.waveform_id.station_code,
# 								location_code=pick_info.waveform_id.location_code,
# 								channel_code=pick_info.waveform_id.channel_code))
# 			mags.append(mag)
# 			
# 		event.picks = picks
# 		event.station_magnitudes = mags
# 		event.amplitudes = amps
# 		event.preferred_origin_id = event.origins[0].resource_id
# 		event.preferred_magnitude_id = event.magnitudes[0].resource_id
# 		event.creation_info = info
# 		event.write('./output/qml/'+pd.to_datetime(row.datetime).strftime('%Y%m%d')+'/'+str(row.evid)+'.xml', format='QUAKEML')
# # 		break
# # 		cat.append(event)
# 
# 	ray.shutdown()
