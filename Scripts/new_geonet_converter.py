import pandas as pd
import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client as FDSN_Client
from obspy import read_inventory
import obspy as op
from multiprocessing import Pool
import multiprocessing
import glob
from obspy.signal.invsim import paz_2_amplitude_value_of_freq_resp
from obspy.signal.invsim import simulate_seismometer as seis_sim
from obspy import Trace
import copy
from scipy.signal import iirfilter, sosfreqz
import os
import warnings
import math

highcut = 20
lowcut = 1
corners = 4
velocity = False

filename = '/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/meta_earthquakes.csv'
geonet = pd.read_csv(filename,low_memory=False)
geonet = geonet.sort_values('origintime')
# geonet['origintime'] = pd.to_datetime(geonet['origintime']).astype('datetime64[ns]')
geonet['origintime'] = geonet.origintime.apply(lambda x: UTCDateTime(x).datetime)
geonet = geonet.reset_index(drop=True)
client = FDSN_Client("GEONET")

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

station_df = pd.DataFrame(station_info,columns=['net','sta','lat','lon','elev'])
station_df = station_df.drop_duplicates().reset_index(drop=True)

sta_corr = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/sta_corr.csv')

def _rms(array):
    """
    Calculate RMS of array.

    :type array: numpy.ndarray
    :param array: Array to calculate the RMS for.

    :returns: RMS of array
    :rtype: float
    """
    return np.sqrt(np.mean(np.square(array)))

def _snr(tr, noise_window, signal_window):
    """
    Compute ratio of maximum signal amplitude to rms noise amplitude.

    :param tr: Trace to compute signal-to-noise ratio for
    :param noise_window: (start, end) of window to use for noise
    :param signal_window: (start, end) of window to use for signal

    :return: Signal-to-noise ratio, noise amplitude
    """
    noise_amp = _rms(
        tr.slice(starttime=noise_window[0], endtime=noise_window[1]).data)
    if np.isnan(noise_amp):
#         Logger.warning("Could not calculate noise with this data, setting "
#                        "to 1")
        noise_amp = 1.0
    try:
        signal_amp = tr.slice(
            starttime=signal_window[0], endtime=signal_window[1]).data.max()
    except ValueError as e:
#         Logger.error(e)
        return np.nan
    return signal_amp / noise_amp

def _max_p2t(data, delta, return_peak_trough=False):
    """
    Finds the maximum peak-to-trough amplitude and period.

    Originally designed to be used to calculate magnitudes (by
    taking half of the peak-to-trough amplitude as the peak amplitude).

    :type data: numpy.ndarray
    :param data: waveform trace to find the peak-to-trough in.
    :type delta: float
    :param delta: Sampling interval in seconds
    :type return_peak_trough: bool
    :param return_peak_trough:
        Optionally return the peak and trough

    :returns:
        tuple of (amplitude, period, time) with amplitude in the same
        scale as given in the input data, and period in seconds, and time in
        seconds from the start of the data window.
    :rtype: tuple
    """
    turning_points = []  # A list of tuples of (amplitude, sample)
    for i in range(1, len(data) - 1):
        if (data[i] < data[i - 1] and data[i] < data[i + 1]) or\
           (data[i] > data[i - 1] and data[i] > data[i + 1]):
            turning_points.append((data[i], i))
    if len(turning_points) >= 1:
        amplitudes = np.empty([len(turning_points) - 1],)
        half_periods = np.empty([len(turning_points) - 1],)
    else:
        print(
            'Turning points has length: ' + str(len(turning_points)) +
            ' data have length: ' + str(len(data)))
        return 0.0, 0.0, 0.0
    for i in range(1, len(turning_points)):
        half_periods[i - 1] = (delta * (turning_points[i][1] -
                                        turning_points[i - 1][1]))
        amplitudes[i - 1] = np.abs(turning_points[i][0] -
                                   turning_points[i - 1][0])
    amplitude = np.max(amplitudes)
    period = 2 * half_periods[np.argmax(amplitudes)]
    delay = delta * turning_points[np.argmax(amplitudes)][1]
    if not return_peak_trough:
        return amplitude, period, delay
    max_position = np.argmax(amplitudes)
    peak = max(
        t[0] for t in turning_points[max_position: max_position + 2])
    trough = min(
        t[0] for t in turning_points[max_position: max_position + 2])
    return amplitude, period, delay, peak, trough

def sim_wa(inventory, trace):

    trace.remove_response(inventory=inventory, output='VEL', water_level=1)
    
    PAZ_WA = {'poles': [-6.2832 + 4.7124j, -6.2832 - 4.7124j],
              'zeros': [0 + 0j], 'gain': 1, 'sensitivity': 2800}
    
    trace.data = seis_sim(trace.data, trace.stats.sampling_rate,
                               paz_remove=None, paz_simulate=PAZ_WA,
                               water_level=1)
    
    trace.data = trace.data * 1000
    
   
    return trace

def _sim_WA(trace, inventory, water_level, velocity=False):
    """
    Remove the instrument response from a trace and simulate a Wood-Anderson.

    Returns a de-meaned, de-trended, Wood Anderson simulated trace in
    its place.

    Works in-place on data and will destroy your original data, copy the
    trace before giving it to this function!

    :type trace: obspy.core.trace.Trace
    :param trace:
        A standard obspy trace, generally should be given without
        pre-filtering, if given with pre-filtering for use with
        amplitude determination for magnitudes you will need to
        worry about how you cope with the response of this filter
        yourself.
    :type inventory: obspy.core.inventory.Inventory
    :param inventory:
        Inventory containing response information for the stations in st.
    :type water_level: float
    :param water_level: Water level for the simulation.
    :type velocity: bool
    :param velocity:
        Whether to return a velocity trace or not - velocity is non-standard
        for Wood-Anderson instruments, but institutes that use seiscomp3 or
        Antelope require picks in velocity.

    :returns: Trace of Wood-Anderson simulated data
    :rtype: :class:`obspy.core.trace.Trace`
    """
	# Helpers for local magnitude estimation
	# Note Wood anderson sensitivity is 2080 as per Uhrhammer & Collins 1990
    PAZ_WA = {'poles': [-6.28318 + 4.71239j, -6.28318 - 4.71239j],
        'zeros': [0 + 0j], 'gain': 1.0, 'sensitivity': 2800}
#     PAZ_WA = {'poles': [-5.49779 + 5.60886j, -5.49779 - 5.60886j],
#         'zeros': [0 + 0j], 'gain': 1.0, 'sensitivity': 2080}
    assert isinstance(trace, Trace)
    paz_wa = copy.deepcopy(PAZ_WA)
    # Need to make a copy because we might edit it.
    if velocity:
        paz_wa['zeros'] = [0 + 0j, 0 + 0j]
    # De-trend data
#     trace.detrend('simple')
#     # Remove response to Velocity
    try:
        trace.remove_response(
            inventory=inventory, output="VEL", water_level=water_level)
# #         trace.remove_sensitivity(inventory=inventory)
    except Exception:
        print(f"No response for {trace.id} at {trace.stats.starttime}")
        return None
    # Simulate Wood Anderson
    trace.data = seis_sim(trace.data, trace.stats.sampling_rate,
                          paz_remove=None, paz_simulate=paz_wa,
                          water_level=water_level)
    return trace

def convert_eq(eventid):
# for index,row in geonet.iterrows():
	#     for i in range(0,1):
	#         row = geonet.iloc[i]
# 	eventid = row.publicid
	nz20_res = 0.278
	eventid = str(eventid)
	try:
		print('Processing event',eventid)
		cat = client_NZ.get_events(eventid=eventid)

		event_line = []
		mag_line = []
		phase_line = []
		prop_line = []

		for event in cat:
			ev_datetime = event.preferred_origin().time
			ev_lat = event.preferred_origin().latitude
			ev_lon = event.preferred_origin().longitude
			ev_depth = event.preferred_origin().depth/1000
			try:
				ev_loc_type = str(event.preferred_origin().method_id).split('/')[1]
			except:
				ev_loc_type = None
			try:
				ev_loc_grid = str(event.preferred_origin().earth_model_id).split('/')[1]
			except:
				ev_loc_grid = None
			try:
				ev_ndef = event.preferred_origin().quality.used_phase_count
			except:
				ev_ndef = None
			try:
				ev_nsta = event.preferred_origin().quality.used_station_count
			except:
				ev_nsta = None
			try:
				std = event.preferred_origin().quality.standard_error
			except:
				std = None
			
			# Find magnitude info, deprioritize 'M' measurements
			pref_mag_type = event.preferred_magnitude().magnitude_type
			if pref_mag_type.lower() == 'm':
				pref_mag_type = 'ML'
				mb_mag = [mag for mag in event.magnitudes if mag.magnitude_type.lower() == 'mb']
				if mb_mag:
					loc_mag = mb_mag[0]
					pref_mag_type = 'Mb'
				else:
					ml_loc_mag = [mag for mag in event.magnitudes if mag.magnitude_type.lower() == 'ml']
					mlv_loc_mag = [mag for mag in event.magnitudes if mag.magnitude_type.lower() == 'mlv']
					if ml_loc_mag and mlv_loc_mag:
						ml_loc_mag = ml_loc_mag[0]
						mlv_loc_mag = mlv_loc_mag[0]
						if ml_loc_mag.station_count >= mlv_loc_mag.station_count:
							loc_mag = ml_loc_mag
						else:
							loc_mag = mlv_loc_mag
					elif ml_loc_mag:
						loc_mag = ml_loc_mag[0]
					elif mlv_loc_mag:
						loc_mag = mlv_loc_mag[0]
	# 					else:
	# 						print('...No ML for',eventid,'trying MLv')
	# 						loc_mag = [mag for mag in event.magnitudes if mag.magnitude_type.lower() == 'mlv']
	# 						if loc_mag:
	# 							loc_mag = loc_mag[0]
					else:
						print('...No ML or MLv for',eventid,'trying M')
						loc_mag = [mag for mag in event.magnitudes if mag.magnitude_type.lower() == 'm'][0]
				pref_mag = loc_mag.mag
				pref_mag_method = 'uncorrected'
				pref_mag_unc = loc_mag.mag_errors.uncertainty
				pref_mag_sta_ids = loc_mag.station_magnitude_contributions
				pref_mag_nmag = len(loc_mag.station_magnitude_contributions)
			else:
				pref_mag = event.preferred_magnitude().mag
				pref_mag_method = 'uncorrected'
				pref_mag_unc = event.preferred_magnitude().mag_errors.uncertainty
				pref_mag_nmag = len(event.preferred_magnitude().station_magnitude_contributions)

	# 			event_line.append([eventid, ev_datetime, ev_lat, ev_lon, ev_depth, ev_loc_type, ev_loc_grid,
	# 				pref_mag, pref_mag_type, pref_mag_method, pref_mag_unc, ev_ndef, ev_nsta, std])
				pref_mag_sta_ids = event.preferred_magnitude().station_magnitude_contributions


			arrivals = event.preferred_origin().arrivals

			# Gather station magnitudes
			i = 0
			sta_mag_line = []
			for pref_mag_sta in pref_mag_sta_ids:
				sta_mag = [sta_mag for sta_mag in event.station_magnitudes if sta_mag.resource_id == pref_mag_sta.station_magnitude_id][0]
				sta_mag_mag = sta_mag.mag
				sta_mag_type = sta_mag.station_magnitude_type
				net = sta_mag.waveform_id.network_code
				sta = sta_mag.waveform_id.station_code
				loc = sta_mag.waveform_id.location_code
				chan = sta_mag.waveform_id.channel_code
			
				if net == 'NZ':
					client = client_NZ
				else:
					client = client_IU
						
				amp = [amp for amp in event.amplitudes if amp.resource_id == sta_mag.amplitude_id]
			
				if amp:
					amp = amp[0]
					amp_amp = amp.generic_amplitude
					amp_time = amp.time_window.reference
				else:
					continue
	
				### Load inventory for data!
				try:
					with warnings.catch_warnings():
						warnings.filterwarnings("ignore", category=UserWarning)
						inventory_st = client.get_stations(network=net, station=sta, channel=chan[0:2]+'?', level='response')
				except:
					magid = eventid+str('m')+str(i)
					i=i+1
					mag_line.append([magid, net, sta, loc, chan, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp])
					sta_mag_line.append([magid, net, sta, loc, chan, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp])
					continue

				if sta_mag_type.lower() == 'ml' or sta_mag_type.lower() == 'mlv':

					sta_info = station_df[station_df.sta == sta]

					if len(sta_corr[sta_corr.sta == sta]) == 1:
						corr = sta_corr[sta_corr.sta == sta]['corr'].values[0]
					else:
						corr = 0
						
					if len(sta_info) > 0:
						dist, az, b_az = op.geodetics.gps2dist_azimuth(ev_lat, ev_lon, sta_info['lat'].values[0], sta_info['lon'].values[0])
						r_epi = dist/1000
						r_hyp = (r_epi ** 2 + (ev_depth + sta_info['elev'].values[0]/1000) ** 2) ** 0.5
					else: # If the station is not in the current inventory
						pick = [pick for pick in event.picks if pick.waveform_id.station_code == sta]
						if pick:
							pick = pick[0]
							arrival = [arrival for arrival in arrivals if arrival.pick_id == pick.resource_id][0]
							r_epi = op.geodetics.degrees2kilometers(arrival.distance)
							r_hyp = ((r_epi) ** 2 + (ev_depth) ** 2) ** 0.5
						else: # If there is no corresponding pick to sta_mag
							r_epi = None
							r_hyp = None

					pick = [pick for pick in event.picks if pick.waveform_id.station_code == sta]
					if pick:
						pick = pick[0]
					else: # No pick time!
						continue
					
					slow_vel = 3
					endtime = ev_datetime + r_hyp/slow_vel + 30
					
					if pick.phase_hint.lower()[0] == 'p':
						windowStart = pick.time - 35
						windowEnd = endtime
						noise_window = [windowStart,windowStart+30]
						signal_window = [pick.time-2,pick.time+30]				
					else:
						windowStart = pick.time - 45
						windowEnd = endtime
						noise_window = [windowStart,windowStart+30]
						signal_window = [pick.time-12,pick.time+20]				

					if len(chan) < 3:
						try:
							st = client.get_waveforms(net,sta,loc,chan+'?',windowStart,windowEnd)
							st = st.merge()
						except:
							continue
					else:
						try:
							st = client.get_waveforms(net,sta,loc,chan[0:2]+'?',windowStart,windowEnd)
							st = st.merge()
						except:
							continue
				
					for tr in st:
						nslc = [tr.stats.network,tr.stats.station,tr.stats.location,tr.stats.channel]
						row_exists = [row[1:5] for row in sta_mag_line if row[1:5] == nslc]
						if row_exists:
							continue
						tr = tr.copy()
	# 						response = inventory_st.select(network=tr.stats.network,
	# 							station=tr.stats.station,channel=tr.stats.channel,time=ev_datetime)

						tr = tr.split().detrend('demean').merge(fill_value=0)[0]
	# 					
						
						tr.filter('highpass', freq=lowcut,
							corners = corners)
				
	# 						tr.remove_response(inventory=inventory_st,output="VEL",water_level=1)
	# 					tr.attach_response(inventory)
	# 						tr = sim_wa(inventory, tr)
# 							snr = abs(tr.max() / np.sqrt(np.mean(np.square(tr.data))))
						snr = _snr(tr,noise_window,signal_window)
					
						if snr < 3:
							# Not a high enough SNR
							print(f'SNR for {tr.id} not high enough')
							magid = eventid+str('m')+str(i)
							i = i+1
							mag_line.append([magid, net, sta, loc, nslc[-1], eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp])
							sta_mag_line.append([magid, net, sta, loc, nslc[-1], eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp])
							continue
					
						tr = _sim_WA(tr, inventory_st, 0, velocity=False)
					
						if tr == None:
							magid = eventid+str('m')+str(i)
							i = i+1
							mag_line.append([magid, net, sta, loc, nslc[-1], eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp])
							sta_mag_line.append([magid, net, sta, loc, nslc[-1], eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp])
							continue
							

						# Calculate the normalized noise amplitude
						amplitude, period, delay, peak, trough = _max_p2t(
							tr.data, tr.stats.delta, return_peak_trough=True)
# 						print(amplitude,peak,trough)
												
						if peak >= abs(trough):
							amplitude = peak
						else:
							amplitude = abs(trough)
						
						# Calculate the absolute amplitude
						amplitude = abs(tr.max())
						
					
						# Generate poles and zeros for the filter we used earlier.
						# We need to get the gain for the digital SOS filter used by
						# obspy.
						sos = iirfilter(
							corners, [lowcut / (0.5 * tr.stats.sampling_rate)],
							btype='highpass', ftype='butter', output='sos')
						_, gain = sosfreqz(sos, worN=[1 / period],
										   fs=tr.stats.sampling_rate)
						gain = np.abs(gain[0])  # Convert from complex to real.
						if gain < 1e-2:
							print(
								f"Pick made outside stable pass-band of filter "
								f"on {tr.id}, rejecting")
							magid = eventid+str('m')+str(i)
							i = i+1
							mag_line.append([magid, net, sta, loc, nslc[-1], eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp])
							sta_mag_line.append([magid, net, sta, loc, nslc[-1], eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp])
							continue
						amplitude /= gain
# 
# 						amplitude *= 0.5
						amplitude *= 1000
# 						print(amplitude,amp_amp,amp_time,windowStart+delay,r_hyp,sta,chan)
									
						if r_epi:
							R = r_hyp
							h = ev_depth

							logA0R = 1.110 * np.log10(R / 100) + 0.00189 * (R - 100) + 3.0  # L. K. Hutton and David M. Boore, BSSA, 1987: The Ml for southern California
							logA0R = 0.2869 - 1.272*1e-3*(R) -(1.493 * np.log10(R))	+ corr	# Ristau 2016
	# 						A = 10 ** (sta_mag_mag - logA0R)

							A = amplitude
							ML_i = np.log10(A) + np.log10(R) + 0.0029 * R # Robinson, 1987, add K for station correction
	
							if (h <= 40):
								H40 = 0
							else:
								H40 = h - 40
	
							NZlogA0R = 0.51 + ((-0.79E-3) * R) + (-1.67 * np.log10(R)) + ((2.38E-3) * H40) + corr # Corrected Ml for NZ according to Rhoades et al., 2020
							CML_i = np.log10(A) - NZlogA0R
# 							ML_i = np.log10(A) - logA0R

							magid = eventid+str('m')+str(i)
							i = i+1
							mag_line.append([magid, net, sta, loc, tr.stats.channel, eventid, sta_mag_mag, sta_mag_type, CML_i, 'NZ20', A])
							sta_mag_line.append([magid, net, sta, loc, tr.stats.channel, eventid,  sta_mag_mag, sta_mag_type, CML_i, 'NZ20', A])
						else: # If there is no epicentral distance measured
							magid = eventid+str('m')+str(i)
							i = i+1
							mag_line.append([magid, net, sta, loc, tr.stats.channel, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp])
							sta_mag_line.append([magid, net, sta, loc, tr.stats.channel, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp])
				else: # If not local magnitude
					magid = eventid+str('m')+str(i)
					i=i+1
					mag_line.append([magid, net, sta, loc, chan, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp])
					sta_mag_line.append([magid, net, sta, loc, chan, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp])

			sta_mag_df = pd.DataFrame(sta_mag_line, columns=['magid', 'net', 'sta', 'loc', 'chan', 'evid', 'mag', 'mag_type', 'mag_corr', 'mag_corr_method', 'amp'])
		
			if pref_mag_type.lower() == 'ml' or pref_mag_type.lower() == 'mlv':
				# Require at least two stations for a preferred cml solution
				if len(sta_mag_df[sta_mag_df.mag_corr.isnull() == False].sta.unique()) > 1:
					#SeisComp3 takes a trimmed mean, rejecting the lowest and highest 12.5% ML values
					if len(sta_mag_df[sta_mag_df.chan.str.endswith('Z')]) > 0:
						CML_Z = np.mean(sta_mag_df[sta_mag_df.chan.str.endswith('Z')].mag_corr)
						CML_Z_unc = np.sqrt(nz20_res ** 2 + np.var(sta_mag_df[sta_mag_df.chan.str.endswith('Z')].mag_corr))
					else:
						CML_Z = np.nan()
						CML_Z_unc = np.nan()
					if len(sta_mag_df[sta_mag_df.chan.str.endswith(('E','N','1','2'))]) > 0:
						CML_H = np.mean(sta_mag_df[sta_mag_df.chan.str.endswith(('E','N','1','2'))].mag_corr)
						CML_H_unc = np.sqrt(nz20_res ** 2 + np.var(sta_mag_df[sta_mag_df.chan.str.endswith(('E','N','1','2'))].mag_corr))
					else:
						CML_H = np.nan()
						CML_H_unc = np.nan()
					# If there are no vertical solutions, use horizontal solutions
					if math.isnan(CML_Z):
						nmag = np.count_nonzero(sta_mag_df.loc[sta_mag_df.chan.str.endswith(('E','N','1','2'))].mag_corr.isnull().values == False)
						event_line.append([eventid, ev_datetime, ev_lat, ev_lon, ev_depth, ev_loc_type, ev_loc_grid,
							CML_H, 'cMl_H', 'NZ20', CML_H_unc, ev_ndef, ev_nsta, nmag, std])				
					else:
# 					if CML_Z_unc > CML_H_unc:
# 						event_line.append([eventid, ev_datetime, ev_lat, ev_lon, ev_depth, ev_loc_type, ev_loc_grid,
# 							CML_Z, 'cMl_H', 'NZ20', CML_H_unc, ev_ndef, ev_nsta, std])
# 					else:
						nmag = np.count_nonzero(sta_mag_df.loc[sta_mag_df.chan.str.endswith('Z')].mag_corr.isnull().values == False)
						event_line.append([eventid, ev_datetime, ev_lat, ev_lon, ev_depth, ev_loc_type, ev_loc_grid,
							CML_Z, 'cMl', 'NZ20', CML_Z_unc, ev_ndef, ev_nsta, nmag, std])				
# 				elif len(sta_mag_df[sta_mag_df.mag_corr.isnull() == False]) == 1:
# 					print(sta_mag_df.mag_corr)
# 					CML = sta_mag_df.mag_corr.values[0]
# 					CML_unc = nz20_res
# 					event_line.append([eventid, ev_datetime, ev_lat, ev_lon, ev_depth, ev_loc_type, ev_loc_grid,
# 						CML, 'cMl', 'NZ20', CML_unc, ev_ndef, ev_nsta, std])		
				else:
					event_line.append([eventid, ev_datetime, ev_lat, ev_lon, ev_depth, ev_loc_type, ev_loc_grid,
						pref_mag, pref_mag_type, pref_mag_method, pref_mag_unc, ev_ndef, ev_nsta, pref_mag_nmag, std])			
			else:
				event_line.append([eventid, ev_datetime, ev_lat, ev_lon, ev_depth, ev_loc_type, ev_loc_grid,
					pref_mag, pref_mag_type, pref_mag_method, pref_mag_unc, ev_ndef, ev_nsta, pref_mag_nmag, std])			
		
			# Calculate propagation paths
			for i, arrival in enumerate(arrivals):
				arid = eventid+str('a')+str(i+1)
				phase = arrival.phase
				toa = arrival.takeoff_angle
				arr_t_res = arrival.time_residual
				pick = [pick for pick in event.picks if pick.resource_id == arrival.pick_id][0]
				arr_datetime = pick.time
				net = pick.waveform_id.network_code
				sta = pick.waveform_id.station_code
				loc = pick.waveform_id.location_code
				chan = pick.waveform_id.channel_code
				phase_line.append([arid, arr_datetime, net, sta, loc, chan, phase, arr_t_res, eventid])
		
				sta_info = station_df[station_df.sta == sta]
				if len(sta_info) > 0:
					dist, az, b_az = op.geodetics.gps2dist_azimuth(ev_lat, ev_lon, sta_info['lat'].values[0], sta_info['lon'].values[0])
					r_epi = dist/1000
					r_hyp = (r_epi ** 2 + (ev_depth + sta_info['elev'].values[0]/1000) ** 2) ** 0.5
				else:
					r_epi = op.geodetics.degrees2kilometers(arrival.distance)
					r_hyp = (r_epi ** 2 + (ev_depth) ** 2) ** 0.5
					az = arrival.azimuth
					if az <= 180:
						b_az = az+180
					else:
						b_az = az-180
				prop_line.append([eventid, net, sta, r_epi, r_hyp, az, b_az, toa])
		event_df = pd.DataFrame(event_line, columns=['evid', 'datetime', 'lat', 'lon', 'depth', 'loc_type', 
			'loc_grid', 'mag', 'mag_type', 'mag_method', 'mag_unc', 'ndef', 'nsta', 'nmag', 't_res'])	
		arrival_df = pd.DataFrame(phase_line, columns=['arid', 'datetime', 'net', 'sta', 'loc', 'chan', 'phase', 't_res', 'evid'])
		mag_df = pd.DataFrame(mag_line, columns=['magid', 'net', 'sta', 'loc', 'chan', 'evid', 'mag', 'mag_type', 'mag_corr', 'mag_corr_method', 'amp'])
		prop_df = pd.DataFrame(prop_line, columns=['evid','net','sta','r_epi','r_hyp','az','b_az','toa'])

		event_df.to_csv(directory+'events_'+str(year)+'_'+str(month)+'.csv',index=False,header=False,mode='a')
		arrival_df.to_csv(directory+'arrivals_'+str(year)+'_'+str(month)+'.csv',index=False,header=False,mode='a')
		mag_df.to_csv(directory+'mags_'+str(year)+'_'+str(month)+'.csv',index=False,header=False,mode='a')
		prop_df = prop_df.drop_duplicates().reset_index(drop=True)
		prop_df.to_csv(directory+'props_'+str(year)+'_'+str(month)+'.csv',index=False,header=False,mode='a')
	# 		return event_line, phase_line, mag_line, prop_line
	# 		return event_df, arrival_df, mag_df, prop_df
	except Exception as e:
		print('There may be something wrong with event '+eventid, e)

directory = '/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output_test/'

if not os.path.exists(directory):
	os.makedirs(directory)

years = np.arange(2012,2013)
months = np.arange(2,13)
# years = np.unique(geonet.origintime.values.astype('datetime64[Y]').astype(int)+1970)
for year in years:
	for month in months:
		process_year = year
		process_month = month
	# 	process_year = [2003] ### Assign a list of years to download waveforms for
		geonet_sub_mask = (geonet.origintime.dt.year == process_year) & (geonet.origintime.dt.month == process_month)
# 		geonet_sub_mask = np.isin(geonet.origintime.values.astype('datetime64[Y]').astype(int)+1970,process_year)
		geonet_sub = geonet[geonet_sub_mask].reset_index(drop=True)

		event_list = [row.publicid for index, row in geonet_sub.iterrows()] 

		event_df = pd.DataFrame(columns=['evid', 'datetime', 'lat', 'lon', 'depth', 'loc_type', 
			'loc_grid', 'mag', 'mag_type', 'mag_method', 'mag_unc', 'ndef', 'nsta', 'nmag', 't_res'])	
		arrival_df = pd.DataFrame(columns=['arid', 'datetime', 'net', 'sta', 'loc', 'chan', 'phase', 't_res', 'evid'])
		mag_df = pd.DataFrame(columns=['magid', 'net', 'sta', 'loc', 'chan', 'evid', 'mag', 'mag_type', 'mag_corr', 'mag_corr_method', 'amp'])
		prop_df = pd.DataFrame(columns=['evid','net','sta','r_epi','r_hyp','az','b_az','toa'])

		event_df.to_csv(directory+'events_'+str(year)+'_'+str(month)+'.csv',index=False)
		arrival_df.to_csv(directory+'arrivals_'+str(year)+'_'+str(month)+'.csv',index=False)
		mag_df.to_csv(directory+'mags_'+str(year)+'_'+str(month)+'.csv',index=False)
		prop_df.to_csv(directory+'props_'+str(year)+'_'+str(month)+'.csv',index=False)

		cores = int(multiprocessing.cpu_count()-1)
		pool = Pool(cores)
	# 	event_data = pool.map(convert_eq, event_list[0:10])
	# 	event_df, arrival_df, mag_df, prop_df = zip(*pool.map(convert_eq, event_list))
		pool.map(convert_eq, event_list)
		pool.close()
		pool.join()
	
# 	event_df = pd.concat(event_df).reset_index(drop=True)
# 	arrival_df = pd.concat(arrival_df).reset_index(drop=True)
# 	mag_df = pd.concat(mag_df).reset_index(drop=True)
# 	prop_df = pd.concat(prop_df).drop_duplicates().reset_index(drop=True)

# 	events, arrivals, mags, props = [],[],[],[]
# 	
# 	for row in event_data:
# 		if row:
# 			for event in row[0]:
# 				events.append(event)
# 			for arrival in row[1]:
# 				arrivals.append(arrival)
# 			for mag in row[2]:
# 				mags.append(mag)
# 			for prop in row[3]:
# 				props.append(prop)
# 
# 	events = [event for row in event_data if row for event in row[0]]
# 	arrivals = [arrival for row in event_data if row for arrival in row[1]]
# 	mags = [mag for row in event_data if row for mag in row[2]]
# 	props = [prop for row in event_data if row for prop in row[3]]
# 
# 	event_df = pd.DataFrame(events, columns=['evid', 'datetime', 'lat', 'lon', 'depth', 'loc_type', 
# 		'loc_grid', 'mag', 'mag_type', 'mag_method', 'mag_unc', 'ndef', 'nsta', 't_res'])
# 	
# 	arrival_df = pd.DataFrame(arrivals, columns=['arid', 'datetime', 'net', 'sta', 'loc', 'chan', 'phase', 't_res', 'evid'])
# 
# 	mag_df = pd.DataFrame(mags, columns=['magid', 'net', 'sta', 'loc', 'chan', 'evid', 'mag', 'mag_type', 'mag_corr', 'mag_corr_method', 'amp'])
# 
# 	prop_df = pd.DataFrame(props, columns=['evid','net','sta','r_epi','r_hyp','az','b_az','toa'])
# 	prop_df = prop_df.drop_duplicates().reset_index(drop=True)
	
# 	if len(event_df):
# 		event_df.to_csv(directory+'events_'+str(year)+'.csv',index=False)
# 	if len(arrival_df):
# 		arrival_df.to_csv(directory+'arrivals_'+str(year)+'.csv',index=False)
# 	if len(mag_df):
# 		mag_df.to_csv(directory+'mags_'+str(year)+'.csv',index=False)
# 	if len(prop_df):
# 		prop_df.to_csv(directory+'props_'+str(year)+'.csv',index=False)
	
# 	event_list_df = pd.DataFrame(event_list,columns=['evid'])
# 	try_again = event_list_df[event_list_df.evid.isin(event_df.evid) == False].evid.tolist()
# 	try_again_data = convert_eq(try_again[0])
	

events_df = pd.concat([pd.read_csv(f,low_memory=False) for f in glob.glob(directory+'events_*.csv')])
mags_df = pd.concat([pd.read_csv(f,low_memory=False) for f in glob.glob(directory+'mags_*.csv')])	
arrivals_df = pd.concat([pd.read_csv(f,low_memory=False) for f in glob.glob(directory+'arrivals_*.csv')])	
props_df = pd.concat([pd.read_csv(f,low_memory=False) for f in glob.glob(directory+'props_*.csv')])	

station_sub = station_df[station_df['sta'].isin(arrivals_df['sta'].unique())]

events_df.to_csv(directory+'earthquake_source_table.csv',index=False)
mags_df.to_csv(directory+'station_magnitude_table.csv',index=False)
arrivals_df.to_csv(directory+'phase_arrival_table.csv',index=False)
props_df.to_csv(directory+'propagation_path_table.csv',index=False)
station_sub.to_csv(directory+'site_table.csv',index=False)
