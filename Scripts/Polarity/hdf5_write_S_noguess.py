# Reads a list of catalogue of GeoNet events and downloads the mseed data compatible
# with instrument response to an mseed directory, shown as 'root' in the mseed_write
# function. This data is used as input for the ground motion classifier, which is in
# turn used to determine which events to comput intensity measures for.

# This version contains optimizations using list comprehension and other factors that allow
# it to run 3 times faster than ver. 1.

# This version incorporates magnitude-distance scaling when searching for stations to
# download waveform data from. It is also uses a ds595 calculation in order to determine
# the full waveform time window to download.

import glob
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.core.event import read_events
from obspy import read_inventory
from obspy.signal import freqattributes,rotate
from obspy.geodetics import locations2degrees,degrees2kilometers,kilometers2degrees
from obspy.geodetics.base import gps2dist_azimuth
from obspy.core import UTCDateTime
from obspy.taup import TauPyModel
from obspy.signal.trigger import recursive_sta_lta
from multiprocessing import Pool,cpu_count
from scipy.interpolate import interp1d
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.io.mseed import ObsPyMSEEDFilesizeTooSmallError
import numpy as np
import pandas as pd
import os
import sys
import time
import numpy as np
import glob
import h5py
from pandarallel import pandarallel		# conda install -c bjrn pandarallel
import shutil

def mseed_write(row):

	root = '/Volumes/SeaJade 2 Backup/NZ/hdf5_kaikoura'

	channel_codes = 'HNZ,BNZ,HHZ,BHZ,EHZ,SHZ'
	s_channel_codes = 'HN?,BN?,HH?,BH?,EH?,SH?'
	sorter = ['HHZ','BHZ','EHZ','HNZ','BNZ','SHZ']
	sorterIndex = dict(zip(sorter, range(len(sorter))))
	s_sorter = ['HH','BH','EH','HN','BN','SH']
	s_sorterIndex = dict(zip(s_sorter, range(len(s_sorter))))

	eventid = row.evid
	try:
		# Check to see if the event is in the FDSN database, skip if it isn't!
		cat = client_NZ.get_events(eventid=eventid)
	except:
		return

	event = cat[0]
	
# 		if row.eventtype == 'outside of network interest':
# 			print('Event '+eventid+' outside of network interest')

	otime = event.preferred_origin().time
	event_lat = event.preferred_origin().latitude
	event_lon = event.preferred_origin().longitude
	event_depth = event.preferred_origin().depth/1000 # Depth in km
	folderA = otime.strftime('%Y')
	folderB = otime.strftime('%m_%b')
	folderC = otime.strftime('%Y-%m-%d_%H%M%S')
	directory = root+'/'+folderA+'/'+folderB+'/'+folderC
	hdf5_directory = root+'/'+folderA+'/'+folderB+'/'+folderC+'/hdf5/data'
	filename = otime.strftime(str(eventid)+'.xml') # should XML file be identified by the event id?
	fw = directory+'/'+filename
	if not os.path.exists(directory):
		os.makedirs(directory)
# 		print('Event',eventid,'created')
	else: # Skip events where the directory already exists
		return
	event.write(fw,format='QUAKEML')
	preferred_magnitude = event.preferred_magnitude().mag
	p_stations = []
	s_stations = []
	for pick in event.picks: 
		if pick.phase_hint[0].lower() == 'p':
			p_stations.append([pick.waveform_id.network_code,pick.waveform_id.station_code,pick.time])
		if pick.phase_hint[0].lower() == 's':
			s_stations.append([pick.waveform_id.network_code,pick.waveform_id.station_code,pick.time])
	p_sta_df = pd.DataFrame(p_stations,columns=['net','sta','arr_time']).drop_duplicates(subset=['net','sta'])
	s_sta_df = pd.DataFrame(s_stations,columns=['net','sta','arr_time']).drop_duplicates(subset=['net','sta'])
	
	data_matrix = []
	sncls_matrix = []
	y_matrix = []
	snr_matrix = []
	mag_matrix = []
	evid_matrix = []
	p_amp_matrix = []
	s_amp_matrix = []
	sp_ratio_matrix = []
	p_noise_matrix = []
	s_noise_matrix = []

	for sta_index,sta_row in p_sta_df.iterrows():
		net = sta_row.net
		sta = sta_row.sta
		arr_time = sta_row.arr_time
		if len(s_sta_df[s_sta_df.sta == sta].arr_time) != 0:
			s_arr_time = s_sta_df[s_sta_df.sta == sta].arr_time.iloc[0]
			stime_est = []
		else:
			s_arr_time = []
# 		else:
# 			s_arr_time = []
# 			inv = inventory.select(station=sta)
# 			if len(inv) != 0:
# 				sta_lat = inv[0][0].latitude
# 				sta_lon = inv[0][0].longitude
# 				sta_el = inv[0][0].elevation/1000
# 				dist, az, b_az = gps2dist_azimuth(event_lat, event_lon, sta_lat, sta_lon)
# 				r_epi = dist/1000
# 				deg = kilometers2degrees(r_epi)
# 				r_hyp = (r_epi ** 2 + (event_depth + sta_el) ** 2) ** 0.5
# 				# Estimate arrival times S phases
# 				s_arrivals = model.get_travel_times(source_depth_in_km=event_depth,distance_in_degree=deg,phase_list=['tts'])
# 				stime_est = otime + s_arrivals[0].time
# 				s_arr_time = stime_est
		try:
			# Check network
			if net == 'IU':
				time.sleep(np.random.choice(range(1,7))) # IRIS seems to disconnect if there are too many rapid connections
				for channel_code in sorter:
					try:
						st = client_IU.get_waveforms(net, sta, '*', channel_code, arr_time-4, arr_time+3.5)
						if st:
							st.merge()
							break
					except KeyboardInterrupt:
						print(directory)
						shutil.rmtree(directory)
						raise
					except:
						continue
				if s_arr_time:
					for s_channel_code in s_sorter: # Make sure S data has three components
						try:
							s_st = client_IU.get_waveforms(net, sta, '*', s_channel_code+'*', s_arr_time-4, s_arr_time+7)
							if s_st:
								s_st.merge()
								if len(s_st) == 3:
									break
								else:
									continue
						except KeyboardInterrupt:
							print(directory)
							shutil.rmtree(directory)
							raise
						except:
							continue
			elif net == 'NZ':
				for channel_code in sorter:
					try:
						st = client_NZ.get_waveforms(net, sta, '*', channel_code, arr_time-4, arr_time+3.5)
						if st:
							st.merge()
							break
					except KeyboardInterrupt:
						print(directory)
						shutil.rmtree(directory)
						raise
					except:
						continue
				if s_arr_time:
					for s_channel_code in s_sorter:
						try:
							s_st = client_NZ.get_waveforms(net, sta, '*', s_channel_code+'*', s_arr_time-4, s_arr_time+7)
							if s_st:
								s_st.merge()
								if len(s_st) == 3: # Make sure S data has three components
									break
								else:
									continue
						except KeyboardInterrupt:
							print(directory)
							shutil.rmtree(directory)
							raise
						except:
							continue

			# Get list of locations and channels
			loc_chans = np.unique(np.array([[tr.stats.location,tr.stats.channel] for tr,tr in zip(st,st)]),axis=0)
			loc_chan_df = pd.DataFrame(loc_chans,columns=['loc','chan'])
			loc_chan_df['chan_rank'] = loc_chan_df['chan'].map(sorterIndex)
			# Get trace of preferred channel. Velocity data preferred to acceleration
			tr = st[np.where(loc_chan_df.chan_rank == loc_chan_df.chan_rank.min())[0][0]].copy()
			# Demean data
			tr.detrend('demean')
			tr.taper(max_percentage=0.05)
			starttime = tr.stats.starttime
			# Pad data
			tr.trim(tr.stats.starttime-5,tr.stats.endtime,pad=True,fill_value=0)
			tr.trim(tr.stats.starttime,tr.stats.endtime+5,pad=True,fill_value=0)
			# Resample data
			tr.resample(100)
			# Filter data
			tr.filter('bandpass',freqmin=1.0,freqmax=20.0)
			# Convert acceleration to velocity
			if tr.stats.channel[1] == 'N':
				tr.integrate()
			# Trim data to 5.5 sec time window
			tr.trim(starttime+1,starttime + 6.5)
# 			tr.plot()
# 		except:
# 			continue
			# Pad with last value if data is less than 550 samples
			if tr.stats.npts < 550:
				tr.trim(tr.stats.starttime,tr.stats.starttime + 5.5,pad=True,fill_value=tr.data[-1])
			# Find p_signal in displacement
			tr_disp = tr.copy().integrate()
			tr2 = tr_disp.slice(tr_disp.stats.starttime,arr_time - 0.2)
# 			signal = abs(tr.max())
			p_noise = tr2.std()
			p_signal = np.max(abs(np.array(tr_disp.max())))
			snr = p_signal/p_noise
			# Make sure data is only 550 samples
			tr_x = tr.data[0:550]
			# Normalize the data
			tr_x = tr_x/abs(tr_x).max()
			sncls = np.array([tr.stats.network+'.'+tr.stats.station+'.'+tr.stats.location+'.'+tr.stats.channel])

			if s_arr_time:
				# Now for the S phase!
				# Get list of locations and channels
				loc_chans = np.unique(np.array([[tr.stats.location,tr.stats.channel[0:2]] for tr,tr in zip(s_st,s_st)]),axis=0)
				loc_chan_df = pd.DataFrame(loc_chans,columns=['loc','chan'])
				loc_chan_df['chan_rank'] = loc_chan_df['chan'].map(s_sorterIndex)
				# Get trace of preferred channel. Velocity data preferred to acceleration
# 				tr_s = s_st[np.where(loc_chan_df.chan_rank == loc_chan_df.chan_rank.min())[0][0]].copy()
# 				loc_chan_df[loc_chan_df.chan_rank == loc_chan_df.chan_rank.min()].chan[0]
				s_st = s_st.select(channel=loc_chan_df[loc_chan_df.chan_rank == loc_chan_df.chan_rank.min()].chan[0]+'*')
				# Demean data
				s_st.detrend('demean')
				s_st.taper(max_percentage=0.05)
				starttime = s_st[0].stats.starttime
				# Pad data
				s_st.trim(s_st[0].stats.starttime-5,s_st[0].stats.endtime,pad=True,fill_value=0)
				s_st.trim(s_st[0].stats.starttime,s_st[0].stats.endtime+5,pad=True,fill_value=0)
				# Resample data
				s_st.resample(100)
				# Filter data
				s_st.filter('bandpass',freqmin=1.0,freqmax=20.0)
				# Convert acceleration to velocity
				if s_st[0].stats.channel[1] == 'N':
					s_st.integrate()
				# Trim data to 4 sec time window
				s_st.trim(starttime+1,starttime + 11)
				
# 				if stime_est: # Search for a better estimate of the S arrival time
# 					s_trs = s_st.select(component = '[12EN]')
# 					s_ta = 0.5
# 					l_ta = 3
# 					thrOn = 2.5
# 					thrOff = 0.7
# 					trigger = []
# 					s_arr_time = []
# 					for s_tr in s_trs:
# 						if not trigger:
# 							cft = recursive_sta_lta(s_tr.data,int(s_ta * s_tr.stats.sampling_rate),int(l_ta * s_tr.stats.sampling_rate))
# # 							plot_trigger(s_tr,cft,thrOn,thrOff)
# 							if len(cft[cft >= thrOn]) > 0:
# 								trigger = np.where(cft == cft[cft >= thrOn][0])[0]
# 								s_arr_time = s_tr.stats.starttime + s_tr.times()[trigger][0]
# 							else:
# 								s_arr_time = []
			if s_arr_time:
				# Find the highest amplitude across all three channels
				s_st_disp = s_st.copy().integrate()
				s_st2 = s_st_disp.slice(s_st[0].stats.starttime,s_arr_time - 0.2)
				s_signal = np.max(abs(np.array(s_st_disp.max())))
				s_noise = s_st2[np.where(s_signal == abs(np.array(s_st_disp.max())))[0][0]].std()
				sp_ratio = s_signal/p_signal
			else:
				s_signal = -1
				sp_ratio = -1
				s_noise = -1

			if len(data_matrix) == 0:
				data_matrix = tr_x
				sncls_matrix = sncls
				y_matrix = np.array([2])
				snr_matrix = snr
				mag_matrix = preferred_magnitude
				evid_matrix = eventid
				p_amp_matrix = p_signal
				p_noise_matrix = p_noise
				if s_arr_time:
					s_amp_matrix = s_signal
					if s_signal != -1:
						sp_ratio_matrix = s_signal/p_signal
						s_noise_matrix = s_noise
					else:
						sp_ratio_matrix = -1
						s_noise_matrix = -1
				else:
					s_amp_matrix = -1
					sp_ratio_matrix = -1
					s_noise_matrix = -1
			else:
				data_matrix = np.vstack((data_matrix,tr_x))
				sncls_matrix = np.vstack((sncls_matrix,sncls))
				y_matrix = np.vstack((y_matrix,np.array([2])))
				snr_matrix = np.vstack((snr_matrix,snr))
				mag_matrix = np.vstack((mag_matrix,preferred_magnitude))
				evid_matrix = np.vstack((evid_matrix,eventid))
				p_amp_matrix = np.vstack((p_amp_matrix,p_signal))
				s_amp_matrix = np.vstack((s_amp_matrix,s_signal))
				sp_ratio_matrix = np.vstack((sp_ratio_matrix,sp_ratio))
				p_noise_matrix = np.vstack((p_noise_matrix,p_noise))
				s_noise_matrix = np.vstack((s_noise_matrix,s_noise))
				
	
		except FDSNNoDataException:
# 				print('No data',sta)
			continue
		except ObsPyMSEEDFilesizeTooSmallError:
# 				print('File size too small',sta)
			continue
# 		except KeyboardInterrupt:
# 			shutil.rmtree(directory)
# 			return
		except KeyboardInterrupt:
			print(directory)
			shutil.rmtree(directory)
			raise
		except Exception as e:
			print()
			print(e.__class__, "occurred for event "+str(eventid)+' and station '+str(sta))
			continue
	
	if not os.path.exists(hdf5_directory):
		os.makedirs(hdf5_directory)
	data_file = h5py.File(hdf5_directory+'/'+str(eventid)+'.hdf5', 'w')
	data_file.create_dataset('X', data=data_matrix)
	data_file.create_dataset('sncls', data=sncls_matrix.astype('S'))
	data_file.create_dataset('Y', data=y_matrix)
	data_file.create_dataset('snr', data=snr_matrix)
	data_file.create_dataset('mag', data=mag_matrix)
	data_file.create_dataset('evids', data=evid_matrix.astype('S'))
	data_file.create_dataset('p_amp', data=p_amp_matrix)
	data_file.create_dataset('s_amp', data=s_amp_matrix)
	data_file.create_dataset('sp_ratio', data=sp_ratio_matrix)
	data_file.create_dataset('p_noise', data=p_noise_matrix)
	data_file.create_dataset('s_noise', data=s_noise_matrix)
	data_file.close()


# TauPy is used to estimate the arrival time for events at stations, here we use a general
# global model, iasp91
model = TauPyModel(model="iasp91")

# Get station information for NZ and the SNZO station
channel_codes = 'HN?,BN?,HH?,BH?,EH?,SH?'
client_NZ = FDSN_Client("GEONET")
client_IU = FDSN_Client('IRIS')
inventory_NZ = client_NZ.get_stations(channel=channel_codes)
inventory_IU = client_IU.get_stations(network='IU',station='SNZO',channel=channel_codes)
inventory = inventory_NZ+inventory_IU

kaikoura_events = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/Scripts/Polarity/kaikoura_polarities.csv',low_memory=False)
event_ids = pd.DataFrame(kaikoura_events.evid.unique(),columns=['evid'])

start_time = time.time()
pandarallel.initialize(nb_workers=8,progress_bar=True) # Customize the number of parallel workers
event_ids.parallel_apply(lambda x: mseed_write(x),axis=1)
end_time = time.time()-start_time
print('Took '+str(end_time)+' seconds to run')