import pandas as pd
import obspy as op
from obspy.taup import TauPyModel, taup_create
import numpy as np
# from shapely.geometry import Point
# from shapely.geometry.polygon import Polygon
from math import ceil
from multiprocessing import Pool
import multiprocessing
import glob
from itertools import repeat
from obspy.clients.fdsn import Client as FDSN_Client
import math
import ray
import time as timeit

# There are four NZ vmodels under NZ1DR: 0) Taupo, 1) Wellington, 2) Clyde, 3) NZ-wide
# mod_file = 'vmod/clyde'
# 
# start_model = pd.read_csv(mod_file+'_orig.tvel',delim_whitespace=True,header=1)
# df = start_model
# df['depth_ind'] = df.depth.apply(ceil) # This helps with decimal numbers... still could have conditional issues
# df = df.set_index('depth_ind').reindex(range(df.depth_ind.min(), df.depth_ind.max()+1)).interpolate().reset_index().drop(columns='depth_ind')
# df.to_csv(mod_file+'.tvel',sep='\t',index=False)
# 
# taup_create.build_taup_model('vmod/taupo.tvel',output_folder='vmod')
# taup_create.build_taup_model('vmod/wel.tvel',output_folder='vmod')
# taup_create.build_taup_model('vmod/clyde.tvel',output_folder='vmod')
# taup_create.build_taup_model('vmod/nz.tvel',output_folder='vmod')
# 
# taupo_model = TauPyModel(model='vmod/taupo.npz')
# wellington_model = TauPyModel(model='vmod/wel.npz')
# clyde_model = TauPyModel(model='vmod/clyde.npz')
# nz_model = TauPyModel(model='vmod/nz.npz')
# vmodels = (taupo_model,wellington_model,clyde_model,nz_model)
# 
# vmod_taupo = np.array([[-35.6,180],[-38,177.5],[-39.7,175.7],[-39,175],[-37,176],[-34.6,178.5]])
# vmod_wellington = np.array([[-41,178],[-43.5,175],[-42,173],[-39.7,175.7]])
# vmod_clyde = np.array([[-45.5,172],[-49,167],[-44.5,168],[-44,169]])
# taupo_polygon = Polygon(vmod_taupo)
# wellington_polygon = Polygon(vmod_wellington)
# clyde_polygon = Polygon(vmod_clyde)
# polygons = (taupo_polygon,wellington_polygon,clyde_polygon)

# def convert_eq(event_sub,event_cat,mag_df,sta_df,prop_df):
def merge_reloc(directory):
    # Load original catalogues and relocated catologues to merge the data and then rewrite
    # to new files.

    event_cat = pd.read_csv(directory+'earthquake_source_table.csv',low_memory=False)
    event_cat['datetime'] = pd.to_datetime(event_cat.datetime).astype('datetime64[ns]')

    mag_df = pd.read_csv(directory+'station_magnitude_table.csv',low_memory=False)
    mag_df['reloc'] = 'no'

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

    event_cat.to_csv(directory+'earthquake_source_table_relocated.csv',index=False)
    mag_df.to_csv(directory+'station_magnitude_table_relocated.csv',index=False)

@ray.remote
def convert_eq(row,client_NZ,client_IU,event_cat,sta_df,mag_df,sta_corr,event_df_file,mag_df_file):
# 	directory = '/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/testaroo/'

# 	unique_events = event_sub.evid.unique().astype(str)
# 	# 	arr_df_sub = arr_df.copy()[arr_df['evid'].isin(unique_events)]
# 	mag_df_sub = mag_df.copy()[mag_df['evid'].isin(unique_events)]
# 	# 	prop_df_sub = prop_df.copy()[prop_df['evid'].isin(unique_events)]


	nz20_res = 0.278
# 	for index,row in event_sub.iterrows():
	event_line = []
	mag_line = []
	phase_line = []
	prop_line = []

	ev_id = row.evid
#         ev_datetime = row.datetime
	ev_lat = row.lat
	ev_lon = row.lon
	ev_depth = row.depth
	ev_datetime = row.datetime
	if 'depth_source' in row.index:
		reloc = row.depth_source
	else:
		reloc = 'reyners'
#         ev_mag = row.mag
	print('Processing event',ev_id)
	try:

		if len(event_cat[event_cat['evid'] == ev_id]) == 0:
			print('No data for event '+str(ev_id))
			return
			
		ev = event_cat[event_cat['evid'] == ev_id].iloc[0]
# 		ev = row
		ev_mag = ev.mag

		sta_mags = mag_df[mag_df['evid'] == ev_id].copy().reset_index(drop=True)

		sta_mag_line = []
		for sta_index, sta_mag in sta_mags.iterrows():
			net = sta_mag.net
			sta = sta_mag.sta
			loc = sta_mag['loc']
			chan = sta_mag.chan
			A = sta_mag.amp
			peak = sta_mag.amp_peak
			trough = sta_mag.amp_trough
			max_amplitude = sta_mag.amp_max
			magid = sta_mag.magid
			sta_info = sta_df[sta_df.sta == sta]

			if (sta_mag.mag_type.lower() == 'ml' or sta_mag.mag_type.lower() == 'mlv') and sta_mag.isnull().mag_corr == False:
				if len(sta_corr[sta_corr.sta == sta]) == 1:
					corr = sta_corr[sta_corr.sta == sta]['corr'].values[0]
				else:
					corr = 0

				if len(sta_info) > 0:
					dist, az, b_az = op.geodetics.gps2dist_azimuth(ev_lat, ev_lon, sta_info['lat'].values[0], sta_info['lon'].values[0])
					r_epi = dist/1000
					r_hyp = (r_epi ** 2 + (ev_depth + sta_info['elev'].values[0]/1000) ** 2) ** 0.5

					R = r_hyp
					h = ev_depth
				# 					A = amp

# 					logA0R = 1.110 * np.log10(R / 100) + 0.00189 * (R - 100) + 3.0  # L. K. Hutton and David M. Boore, BSSA, 1987: The Ml for southern California
# 					ML_i = np.log10(A) + logA0R
# 						ML_i = sta_mag.mag
			# 		A = 10 ** (sta_mag_mag - logA0R)

					if (h <= 40):
						H40 = 0
					else:
						H40 = h - 40

					NZlogA0R = 0.51 + ((-0.79E-3) * R) + (-1.67 * np.log10(R)) + ((2.38E-3) * H40) + corr # Corrected Ml for NZ according to Rhoades et al., 2020

					CML_i = np.log10(A) - NZlogA0R

# 					mag_line.append([magid, net, sta, loc, chan, ev_id, ML_i, 'ML', CML_i, 
# 						'NZ20', A, reloc])
# 					sta_mag_line.append([magid, net, sta, loc, chan, ev_id,  ML_i, 'ML', 
# 						CML_i, 'NZ20', A, reloc])
				
					mag_line.append([magid, net, sta, loc, chan, ev_id, sta_mag.mag, sta_mag.mag_type, CML_i, 
						'NZ20', A, peak, trough, max_amplitude, reloc])
					sta_mag_line.append([magid, net, sta, loc, chan, ev_id, sta_mag.mag, sta_mag.mag_type, CML_i, 
						'NZ20', A, peak, trough, max_amplitude, reloc])
				else:
					mag_line.append([magid, net, sta, loc, chan, ev_id, sta_mag.mag, sta_mag.mag_type, 
						None, 'uncorrected', A, peak, trough, max_amplitude, 'no'])
					sta_mag_line.append([magid, net, sta, loc, chan, ev_id, sta_mag.mag, sta_mag.mag_type, 
						None, 'uncorrected', A, peak, trough, max_amplitude, 'no'])
			else:
				mag_line.append([magid, net, sta, loc, chan, ev_id, sta_mag.mag, sta_mag.mag_type, 
					None, 'uncorrected', A, peak, trough, max_amplitude, 'no'])
				sta_mag_line.append([magid, net, sta, loc, chan, ev_id, sta_mag.mag, sta_mag.mag_type, 
					None, 'uncorrected', A, peak, trough, max_amplitude, 'no'])

		sta_mag_df = pd.DataFrame(sta_mag_line, columns=['magid', 'net', 'sta', 'loc', 'chan', 
			'evid', 'mag', 'mag_type', 'mag_corr', 'mag_corr_method', 'amp', 'amp_peak', 
			'amp_trough', 'amp_max', 'reloc'])

		if ev.mag_type.lower() == 'ml' or ev.mag_type.lower() == 'mlv' or ev.mag_type.lower() == 'cml':
			# Require at least two stations for a preferred cml solution
			if len(sta_mag_df[sta_mag_df.mag_corr.isnull() == False].sta.unique()) > 1:
				#SeisComp3 takes a trimmed mean, rejecting the lowest and highest 12.5% ML values
				mag_data = sta_mag_df[sta_mag_df.chan.str.endswith('Z',na=False)]
				if (mag_data.mag_corr.isnull() == False).sum() < 2:
					mag_data = sta_mag_df[sta_mag_df.chan.str.endswith(('E','N','1','2'),na=False)]
					if len(mag_data[mag_data.mag_corr.isnull() == False].sta.unique()) < 2:
						mag_data = sta_mag_df
						mag_type = pref_mag_type
						mag_method = pref_mag_method
			#             print(evid,event.mag,event.mag_type,event.mag_method,event.mag_unc,event.nmag)
					else:
						mag_type = 'cMl_H'
						mag_method = 'NZ20'
				else:
					mag_type = 'cMl'
					mag_method = 'NZ20'
				if mag_type[0:3] == 'cMl':
					Q1 = mag_data.mag_corr.quantile(0.25)
					Q3 = mag_data.mag_corr.quantile(0.75)
					IQR=Q3-Q1
					lowqe_bound=Q1 - 1.5 * IQR
					upper_bound=Q3 + 1.5 * IQR
				#     print(lowqe_bound,upper_bound)

					IQR_mags = mag_data[~((mag_data.mag_corr < lowqe_bound) | (mag_data.mag_corr > upper_bound))]
					# Run a check for anomalous station magnitudes (+ or - 2 from original value)
					IQR_mags = IQR_mags[(IQR_mags.mag_corr <= ev.mag_orig + 2) & (IQR_mags.mag_corr >= ev.mag_orig - 2)]
	
					CML = IQR_mags.mag_corr.mean()
					new_std = IQR_mags.mag_corr.std()
					CML_unc = np.sqrt(nz20_res ** 2 + np.var(IQR_mags.mag_corr))

					nmag = len(IQR_mags[~IQR_mags.mag_corr.isnull()].sta.unique())

# 						event_line.append([eventid, ev_datetime, ev_lat, ev_lon, ev_depth, ev_loc_type, ev_loc_grid,
# 							CML, mag_type, mag_method, CML_unc, event.preferred_magnitude().mag, event.preferred_magnitude().magnitude_type, 
# 							event.preferred_magnitude().mag_errors.uncertainty, ev_ndef, ev_nsta, nmag, std, reloc])								
					event_line.append([ev_id, ev_datetime, ev_lat, ev_lon, ev_depth, 'SIMUL', 'nz3drx',
						CML, mag_type, mag_method, CML_unc, ev.mag_orig, ev.mag_orig_type, 
						ev.mag_orig_unc, ev.ndef, ev.nsta, ev.nmag, ev.t_res, reloc])
			else:
				event_line.append([ev_id, ev_datetime, ev_lat, ev_lon, ev_depth, ev.loc_type, ev.loc_grid,
					ev.mag, ev.mag_type, ev.mag_method, ev.mag_unc, ev.mag_orig, ev.mag_orig_type, 
					ev.mag_orig_unc, ev.ndef, ev.nsta, ev.nmag, ev.t_res, reloc])
		else:
			event_line.append([ev_id, ev_datetime, ev_lat, ev_lon, ev_depth, ev.loc_type, ev.loc_grid,
				ev.mag, ev.mag_type, ev.mag_method, ev.mag_unc, ev.mag_orig, ev.mag_orig_type, 
				ev.mag_orig_unc, ev.ndef, ev.nsta, ev.nmag, ev.t_res, reloc])
	   
	except Exception as e:
		print('There may be something wrong with event '+str(ev_id), e)
		

	events_df = pd.DataFrame(event_line, columns=['evid', 'datetime', 'lat', 'lon', 'depth', 'loc_type', 
		'loc_grid', 'mag', 'mag_type', 'mag_method', 'mag_unc', 'mag_orig', 'mag_orig_type', 'mag_orig_unc', 'ndef', 'nsta', 'nmag', 't_res', 'reloc'])	
	mags_df = pd.DataFrame(mag_line, columns=['magid', 'net', 'sta', 'loc', 'chan', 'evid', 'mag', 'mag_type', 'mag_corr', 'mag_corr_method', 'amp', 'amp_peak', 'amp_trough', 'amp_max', 'reloc'])

	events_df.to_csv(event_df_file,index=False,header=False,mode='a')
	mags_df.to_csv(mag_df_file,index=False,header=False,mode='a')
	
	return

directory = '/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/testaroo/'

event_df = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/seismicity_relocated_june11.dat',low_memory=False)
# event_df = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/martins_new.csv',low_memory=False)
event_df['evid'] = event_df.evid.astype(str)
event_cat = pd.read_csv(directory+'earthquake_source_table.csv',low_memory=False)
event_cat['evid'] = event_cat.evid.astype(str)
mag_df = pd.read_csv(directory+'station_magnitude_table.csv',low_memory=False)
mag_df['evid'] = mag_df['evid'].astype('str')

# Load station information from FDSN in case station csv is not complete
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
# 		station_df = station_df.append({'net':network.code,'sta':station.code,'lat':station.latitude,
# 			'lon':station.longitude,'elev':station.elevation},True)

sta_df = pd.DataFrame(station_info,columns=['net','sta','lat','lon','elev'])
sta_df = sta_df.drop_duplicates().reset_index(drop=True)

# sta_df = pd.read_csv(directory+'site_table.csv',low_memory=False)
# arr_df = pd.read_csv(directory+'phase_arrival_table.csv',low_memory=False)
# prop_df = pd.read_csv(directory+'propagation_path_table.csv',low_memory=False)
sta_corr = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/sta_corr_new.csv')

### Calculate for Reyners relocated data
years = np.arange(2000,2021)
for year in years:
	process_year = year
	
	event_df_sub_mask = np.isin(event_df.datetime.values.astype('datetime64[Y]').astype(int)+1970,process_year)
	event_df_sub = event_df.copy()[event_df_sub_mask].reset_index(drop=True)
	
	if len(event_df_sub) > 0 and np.any(np.isin(event_df_sub.evid.values,event_cat.evid.values)):
		mag_df_sub = mag_df[mag_df['evid'].isin(event_df_sub['evid'].unique())]
		event_cat_sub = event_cat[event_cat['evid'].isin(event_df_sub['evid'].unique())]

		event_df_file = directory+'ISC_relocated_events_'+str(year)+'.csv'
		mag_df_file = directory+'ISC_relocated_mags_'+str(year)+'.csv'

		events_df = pd.DataFrame(columns=['evid', 'datetime', 'lat', 'lon', 'depth', 'loc_type', 
			'loc_grid', 'mag', 'mag_type', 'mag_method', 'mag_unc', 'mag_orig', 'mag_orig_type', 
			'mag_orig_unc', 'ndef', 'nsta', 'nmag', 't_res', 'reloc'])	
		mags_df = pd.DataFrame(columns=['magid', 'net', 'sta', 'loc', 'chan', 'evid', 
			'mag', 'mag_type', 'mag_corr', 'mag_corr_method', 'amp', 'amp_peak', 'amp_trough', 
			'amp_max', 'reloc'])

		events_df.to_csv(event_df_file,index=False)
		mags_df.to_csv(mag_df_file,index=False)


		start_time = timeit.time()
		result_ids = []
		ray.init(object_store_memory=30000 * 1024 * 1024)
		sta_df_id = ray.put(sta_df)
		sta_corr_id = ray.put(sta_corr)
		event_cat_id = ray.put(event_cat_sub)
		mag_df_id = ray.put(mag_df_sub)
		result_ids = [convert_eq.remote(x,client_NZ,client_IU,event_cat_id,sta_df_id,mag_df_id,sta_corr_id,event_df_file,mag_df_file) for idx,x in event_df_sub.iterrows()]
# 		result_ids = convert_eq.remote(event_df_sub.iloc[0],client_NZ,client_IU,event_cat_id,sta_df_id,sta_corr_id,directory)
		results = ray.get(result_ids)
		ray.shutdown()
		print(timeit.time() - start_time)
	else:
		print('No data for '+str(year))

events_df = pd.concat([pd.read_csv(f,low_memory=False) for f in glob.glob(directory+'relocated_events_*.csv')])
mags_df = pd.concat([pd.read_csv(f,low_memory=False) for f in glob.glob(directory+'relocated_mags_*.csv')])	
# props_df = pd.concat([pd.read_csv(f,low_memory=False) for f in glob.glob(directory+'relocated_props_*.csv')])

events_df.to_csv(directory+'relocated_earthquake_source_table.csv',index=False)
mags_df.to_csv(directory+'relocated_station_magnitude_table.csv',index=False)

### Calculate for ISC data
years = np.arange(2000,2021)
for year in years:
	process_year = year
	
	event_df_sub_mask = event_df.year == process_year
	event_df_sub = event_df.copy()[event_df_sub_mask].reset_index(drop=True)
	event_df_sub = event_df_sub[(event_df_sub.depth_source == 'ISC-EHB') | (event_df_sub.depth_source == 'ISC-GEM')]
	event_df_sub['datetime'] = pd.to_datetime(event_df_sub[['year','month','day','hour','minute','second']])
	
	if len(event_df_sub) > 0 and np.any(np.isin(event_df_sub.evid.values,event_cat.evid.values)):
		mag_df_sub = mag_df[mag_df['evid'].isin(event_df_sub['evid'].unique())]
		event_cat_sub = event_cat[event_cat['evid'].isin(event_df_sub['evid'].unique())]

		event_df_file = directory+'ISC_relocated_events_'+str(year)+'.csv'
		mag_df_file = directory+'ISC_relocated_mags_'+str(year)+'.csv'

		events_df = pd.DataFrame(columns=['evid', 'datetime', 'lat', 'lon', 'depth', 'loc_type', 
			'loc_grid', 'mag', 'mag_type', 'mag_method', 'mag_unc', 'mag_orig', 'mag_orig_type', 
			'mag_orig_unc', 'ndef', 'nsta', 'nmag', 't_res', 'reloc'])	
		mags_df = pd.DataFrame(columns=['magid', 'net', 'sta', 'loc', 'chan', 'evid', 
			'mag', 'mag_type', 'mag_corr', 'mag_corr_method', 'amp', 'amp_peak', 'amp_trough', 
			'amp_max', 'reloc'])

		events_df.to_csv(event_df_file,index=False)
		mags_df.to_csv(mag_df_file,index=False)


		start_time = timeit.time()
		result_ids = []
		ray.init(object_store_memory=30000 * 1024 * 1024)
		sta_df_id = ray.put(sta_df)
		sta_corr_id = ray.put(sta_corr)
		event_cat_id = ray.put(event_cat_sub)
		mag_df_id = ray.put(mag_df_sub)
		result_ids = [convert_eq.remote(x,client_NZ,client_IU,event_cat_id,sta_df_id,mag_df_id,sta_corr_id,event_df_file,mag_df_file) for idx,x in event_df_sub.iterrows()]
# 		result_ids = convert_eq.remote(event_df_sub.iloc[0],client_NZ,client_IU,event_cat_id,sta_df_id,sta_corr_id,directory)
		results = ray.get(result_ids)
		ray.shutdown()
		print(timeit.time() - start_time)
	else:
		print('No data for '+str(year))

events_df = pd.concat([pd.read_csv(f,low_memory=False) for f in glob.glob(directory+'ISC_relocated_events_*.csv')])
mags_df = pd.concat([pd.read_csv(f,low_memory=False) for f in glob.glob(directory+'ISC_relocated_mags_*.csv')])	
# props_df = pd.concat([pd.read_csv(f,low_memory=False) for f in glob.glob(directory+'relocated_props_*.csv')])

events_df.to_csv(directory+'ISC_relocated_earthquake_source_table.csv',index=False)
mags_df.to_csv(directory+'ISC_relocated_station_magnitude_table.csv',index=False)
# props_df.to_csv(directory+'relocated_propagation_path_table.csv',index=False)
# station_df.to_csv(directory+'site_table.csv',index=False)

merge_reloc(directory)