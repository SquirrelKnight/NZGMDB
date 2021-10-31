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
from progress.bar import ChargingBar
import numpy as np
import pandas as pd
import os
import sys
import typing
import time
import numpy as np
import glob
import h5py
from pandarallel import pandarallel		# conda install -c bjrn pandarallel

class Site:  # Class of site properties. initialize all attributes to None
    def __init__(self, **kwargs):
        self.name = kwargs.get("name")  # station name
        self.Rrup = kwargs.get("rrup")  # closest distance coseismic rupture (km)
        self.Rjb = kwargs.get(
            "rjb"
        )  # closest horizontal distance coseismic rupture (km)
        self.Rx = kwargs.get(
            "rx", -1.0
        )  # distance measured perpendicular to fault strike from surface projection of
        #                       # updip edge of the fault rupture (+ve in downdip dir) (km)
        self.Ry = kwargs.get(
            "ry", -1.0
        )  # horizontal distance off the end of the rupture measured parallel
        self.Rtvz = kwargs.get(
            "rtvz"
        )  # source-to-site distance in the Taupo volcanic zone (TVZ) (km)
        self.vs30measured = kwargs.get(
            "vs30measured", False
        )  # yes =True (i.e. from Vs tests); no=False (i.e. estimated from geology)
        self.vs30 = kwargs.get("vs30")  # shear wave velocity at 30m depth (m/s)
        self.z1p0 = kwargs.get(
            "z1p0"
        )  # depth (km) to the 1.0km/s shear wave velocity horizon (optional, uses default relationship otherwise)
        self.z1p5 = kwargs.get("z1p5")  # (km)
        self.z2p5 = kwargs.get("z2p5")  # (km)
        self.siteclass = kwargs.get("siteclass")
        self.orientation = kwargs.get("orientation", "average")
        self.backarc = kwargs.get(
            "backarc", False
        )  # forearc/unknown = False, backarc = True
        self.fpeak = kwargs.get("fpeak", 0)


class Fault:  # Class of fault properties. initialize all attributes to None
    def __init__(self, **kwargs):
        self.dip = kwargs.get("dip")  # dip angle (degrees)
        self.faultstyle = kwargs.get(
            "faultstyle"
        )  # Faultstyle (options described in enum below)
        self.hdepth = kwargs.get("hdepth")  # hypocentre depth
        self.Mw = kwargs.get("Mw")  # moment tensor magnitude
        self.rake = kwargs.get("rake")  # rake angle (degrees)
        self.tect_type = kwargs.get(
            "tect_type"
        )  # tectonic type of the rupture (options described in the enum below)
        self.width = kwargs.get("width")  # down-dip width of the fault rupture plane
        self.zbot = kwargs.get("zbot")  # depth to the bottom of the seismogenic crust
        self.ztor = kwargs.get("ztor")  # depth to top of coseismic rupture (km)

def estimate_z1p0(vs30):
    return (
        np.exp(28.5 - 3.82 / 8.0 * np.log(vs30 ** 8 + 378.7 ** 8)) / 1000.0
    )  # CY08 estimate in KM

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def mseed_write(row):

	vs30_default = 500

	root = '/Volumes/SeaJade 2 Backup/NZ/hdf5_kaikoura'

	channel_codes = 'HNZ,BNZ,HHZ,BHZ,EHZ,SHZ'
	s_channel_codes = 'HN?,BN?,HH?,BH?,EH?,SH?'
	sorter = ['HHZ','BHZ','EHZ','HNZ','BNZ','SHZ']
	sorterIndex = dict(zip(sorter, range(len(sorter))))
	s_sorter = ['HH','BH','EH','HN','BN','SH']
	s_sorterIndex = dict(zip(s_sorter, range(len(s_sorter))))
	damping = 0.700
	unit_factor = 1.00

# 	p_bar = 0
# 	if len(event_file) > 1:
# 		p_bar_index = event_file.iloc[0].name
# 	else:
# 		p_bar_index = event_file.name
# 	if p_bar_index == 0: # Only run progress bar on first core
# 		print('Beginning file processing:')
# 		ii = 0
# 		p_bar = 1
# 		total = len(event_file)
# 		progress(ii, total, status='')

# 	for index,row in event_file.iterrows():
	siteprop = Site()
	faultprop = Fault()
	
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
# 			print('Event',eventid,'created')
	else: # Skip events where the directory already exists
# 			print('Event',eventid,'exists...')
# 			if p_bar:
# 				progress(ii, total, status='')
# 				ii += 1
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
# 	dist_matrix = []
	mag_matrix = []
	evid_matrix = []
	p_amp_matrix = []
	s_amp_matrix = []
	sp_ratio_matrix = []

	for sta_index,sta_row in p_sta_df.iterrows():
		net = sta_row.net
		sta = sta_row.sta
		arr_time = sta_row.arr_time
		if len(s_sta_df[s_sta_df.sta == sta].arr_time) != 0:
			s_arr_time = s_sta_df[s_sta_df.sta == sta].arr_time.iloc[0]
			stime_est = []
		else:
			s_arr_time = []
			inv = inventory.select(station=sta)
			if len(inv) != 0:
				sta_lat = inv[0][0].latitude
				sta_lon = inv[0][0].longitude
				sta_el = inv[0][0].elevation/1000
				dist, az, b_az = gps2dist_azimuth(event_lat, event_lon, sta_lat, sta_lon)
				r_epi = dist/1000
				deg = kilometers2degrees(r_epi)
				r_hyp = (r_epi ** 2 + (event_depth + sta_el) ** 2) ** 0.5
				# Estimate arrival times S phases
				s_arrivals = model.get_travel_times(source_depth_in_km=event_depth,distance_in_degree=deg,phase_list=['tts'])
				stime_est = otime + s_arrivals[0].time
				s_arr_time = stime_est
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
						except:
							continue
			elif net == 'NZ':
				for channel_code in sorter:
					try:
						st = client_NZ.get_waveforms(net, sta, '*', channel_code, arr_time-4, arr_time+3.5)
						if st:
							st.merge()
							break
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
						except:
							continue

# 			if net == 'IU':
# 				time.sleep(np.random.choice(range(1,7))) # IRIS seems to disconnect if there are too many rapid connections
# 				st = client_IU.get_waveforms(net, sta, '*', channel_codes, arr_time-4, arr_time+3.5)
# 				if s_arr_time:
# 					s_st = client_IU.get_waveforms(net, sta, '*', channel_codes, s_arr_time-5, s_arr_time+8)
# 			elif net == 'NZ':
# 				st = client_NZ.get_waveforms(net, sta, '*', channel_codes, arr_time-4, arr_time+3.5)
# 				if s_arr_time:
# 					s_st = client_NZ.get_waveforms(net, sta, '*', s_channel_codes, s_arr_time-5, s_arr_time+8)
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
			tr2 = tr.slice(tr.stats.starttime,tr.stats.starttime+1)
			signal = abs(tr.max())
			noise = tr2.std()
			snr = signal/noise
			# Find p_signal in displacement
			tr_disp = tr.copy().integrate()
			p_signal = np.max(abs(np.array(tr_disp.max())))
			# Make sure data is only 400 samples
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
				
				if stime_est: # Search for a better estimate of the S arrival time
					s_trs = s_st.select(component = '[12EN]')
					s_ta = 0.5
					l_ta = 3
					thrOn = 2.5
					thrOff = 0.7
					trigger = []
					s_arr_time = []
					for s_tr in s_trs:
						if not trigger:
							cft = recursive_sta_lta(s_tr.data,int(s_ta * s_tr.stats.sampling_rate),int(l_ta * s_tr.stats.sampling_rate))
# 							plot_trigger(s_tr,cft,thrOn,thrOff)
							if len(cft[cft >= thrOn]) > 0:
								trigger = np.where(cft == cft[cft >= thrOn][0])[0]
								s_arr_time = s_tr.stats.starttime + s_tr.times()[trigger][0]
							else:
								s_arr_time = []
			if s_arr_time:
				# Find the highest amplitude across all three channels
				s_st_disp = s_st.copy().integrate()
				s_signal = np.max(abs(np.array(s_st_disp.max())))
				sp_ratio = s_signal/p_signal
			else:
				s_signal = -1
				sp_ratio = -1

			if len(data_matrix) == 0:
				data_matrix = tr_x
				sncls_matrix = sncls
				y_matrix = np.array([2])
				snr_matrix = snr
# 				dist_matrix = r_hyp
				mag_matrix = preferred_magnitude
				evid_matrix = eventid
				p_amp_matrix = p_signal
				if s_arr_time:
					s_amp_matrix = s_signal
					if s_signal != -1:
						sp_ratio_matrix = s_signal/p_signal
					else:
						sp_ratio_matrix = -1
				else:
					s_amp_matrix = -1
					sp_ratio_matrix = -1
			else:
				data_matrix = np.vstack((data_matrix,tr_x))
				sncls_matrix = np.vstack((sncls_matrix,sncls))
				y_matrix = np.vstack((y_matrix,np.array([2])))
				snr_matrix = np.vstack((snr_matrix,snr))
# 				dist_matrix = np.vstack((dist_matrix,r_hyp))
				mag_matrix = np.vstack((mag_matrix,preferred_magnitude))
				evid_matrix = np.vstack((evid_matrix,eventid))
				p_amp_matrix = np.vstack((p_amp_matrix,p_signal))
				s_amp_matrix = np.vstack((s_amp_matrix,s_signal))
				sp_ratio_matrix = np.vstack((sp_ratio_matrix,sp_ratio))
	
		except FDSNNoDataException:
# 				print('No data',sta)
			continue
		except ObsPyMSEEDFilesizeTooSmallError:
# 				print('File size too small',sta)
			continue
		except Exception as e:
# 				print()
# 				print(e.__class__, "occurred for event "+str(eventid)+' and station '+str(sta))
			continue
	if not os.path.exists(hdf5_directory):
		os.makedirs(hdf5_directory)
	data_file = h5py.File(hdf5_directory+'/'+str(eventid)+'.hdf5', 'w')
	data_file.create_dataset('X', data=data_matrix)
	data_file.create_dataset('sncls', data=sncls_matrix.astype('S'))
	data_file.create_dataset('Y', data=y_matrix)
	data_file.create_dataset('snr', data=snr_matrix)
# 	data_file.create_dataset('dist', data=dist_matrix)
	data_file.create_dataset('mag', data=mag_matrix)
	data_file.create_dataset('evids', data=evid_matrix.astype('S'))
	data_file.create_dataset('p_amp', data=p_amp_matrix)
	data_file.create_dataset('s_amp', data=s_amp_matrix)
	data_file.create_dataset('sp_ratio', data=sp_ratio_matrix)
	data_file.close()

# 		if p_bar:
# 			progress(ii, total, status='')
# 			ii += 1

# TauPy is used to estimate the arrival time for events at stations, here we use a general
# global model, iasp91
model = TauPyModel(model="iasp91")

channel_codes = 'HN?,BN?,HH?,BH?,EH?,SH?'

# Generate cubic interpolation for magnitude distance relationship
mw_rrup = np.loadtxt('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/Mw_rrup.txt')
mws = mw_rrup[:,0]
rrups = mw_rrup[:,1]
f_rrup = interp1d(mws,rrups,kind='cubic')

# file_list = np.load('eq_paths.npy')
# inventory = read_inventory('/Volumes/SeaJade2/NZ/NZ_EQ_Catalog/nz_stations.xml')
client_NZ = FDSN_Client("GEONET")
client_IU = FDSN_Client('IRIS')
inventory_NZ = client_NZ.get_stations(channel=channel_codes)
inventory_IU = client_IU.get_stations(network='IU',station='SNZO',channel=channel_codes)
inventory = inventory_NZ+inventory_IU



# filename = '/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/meta_earthquakes.csv'
# geonet = pd.read_csv(filename,low_memory=False)
# geonet = geonet.sort_values('origintime')
# geonet['origintime'] = pd.to_datetime(geonet['origintime'],format='%Y-%m-%dT%H:%M:%S.%fZ')
# geonet = geonet.reset_index(drop=True)
# 
# # process_years = np.arange(2017,2019) ### Assign a list of years to download waveforms for
# # for year in process_years:
# # 	print('Processing '+str(year))
# # 	print()
# # 	geonet_sub_mask = np.isin(geonet.origintime.values.astype('datetime64[Y]').astype(int)+1970,year)
# 
# min_date = np.datetime64('2010-01-01').astype('datetime64[D]').astype(int)+1970
# max_date = np.datetime64('2010-03-01').astype('datetime64[D]').astype(int)+1970
# # date = np.datetime64('2004').astype('datetime64[Y]').astype(int)+1970
# geonet_sub_mask = (geonet.origintime.values.astype('datetime64[D]').astype(int)+1970 >= min_date) & (geonet.origintime.values.astype('datetime64[D]').astype(int)+1970 < max_date)
# geonet_sub = geonet[geonet_sub_mask].reset_index(drop=True)
# geonet_sub = geonet_sub[(geonet_sub.magnitude >= 5) & (geonet_sub.magnitude < 6)].reset_index(drop=True) ### Start with M >= 4. Go back for smaller events later



kaikoura_events = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/Scripts/Polarity/kaikoura_polarities.csv',low_memory=False)
event_ids = pd.DataFrame(kaikoura_events.evid.unique(),columns=['evid'])

start_time = time.time()
pandarallel.initialize(nb_workers=8,progress_bar=True) # Customize the number of parallel workers
event_ids.parallel_apply(lambda x: mseed_write(x),axis=1)
end_time = time.time()-start_time
print('Took '+str(end_time)+' seconds to run')


# geonet_sub = geonet_sub[geonet_sub.eventtype != 'outside of network interest'].reset_index(drop=True) # Remove non-NZ events
# geonet_sub = geonet_sub[geonet_sub.magnitude >= 6].reset_index(drop=True)
# geonet_sub = geonet_sub[(geonet_sub.magnitude < 5) & (geonet_sub.magnitude >= 4.5)].reset_index(drop=True) ### Start with M >= 4. Go back for smaller events later

# kaikoura_id = '2016p858000'
# event = geonet[geonet.publicid == '3366146'] # Darfield
# event = geonet[geonet.publicid == '3468575'] # Christchurch
# event = geonet[geonet.publicid == '3124785'] # Dusky Sound
# mseed_write(event)

# cores = int(cpu_count()-7)
# df_split = np.array_split(event_ids, cores)
# 
# start_time = time.time()
# with Pool(cores) as pool:
# 	pool.map(mseed_write,df_split)
# pool.close()
# pool.join()
# end_time = time.time()-start_time
# 
# print()
# # print('Took '+str(end_time)+' seconds to run year '+str(year))
# print('All done!!!')
