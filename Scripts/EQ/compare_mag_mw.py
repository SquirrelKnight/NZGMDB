import pandas as pd
from scipy import stats
import numpy as np
from pandarallel import pandarallel		# conda install -c bjrn pandarallel
import glob
from obspy.clients.fdsn import Client as FDSN_Client
import obspy as op
from obspy.signal.invsim import paz_2_amplitude_value_of_freq_resp
from obspy.signal.invsim import simulate_seismometer as seis_sim
from obspy import Trace
import copy
from scipy.signal import iirfilter, sosfreqz
import os
import warnings
import math
import ray
import time as timeit


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
#         trace.remove_response(
#             inventory=inventory, output="VEL", water_level=water_level)
		# Try doing this without the whole water level thing
        trace.remove_sensitivity(inventory=inventory)
        if trace.stats.channel[1] == 'N':
        	trace.integrate()
    except Exception:
        print(f"No response for {trace.id} at {trace.stats.starttime}")
        return None
    # Simulate Wood Anderson
    trace.data = seis_sim(trace.data, trace.stats.sampling_rate,
                          paz_remove=None, paz_simulate=paz_wa,
                          water_level=water_level)
    return trace
    
### Check for mb values
def check_mb(df,client_NZ):
    row = df.copy()
    if row.mag_type == 'M':
        evid = row.evid
#         print(evid)
        cat = client_NZ.get_events(eventid=evid)
        event = cat[0]
        mag_types = [magnitude.magnitude_type.lower() for magnitude in event.magnitudes]
        if 'mw(mb)' in mag_types:
            magnitude = [magnitude for magnitude in event.magnitudes if magnitude.magnitude_type.lower() == 'mw(mb)'][0]
            rewrite = True
        elif 'mb' in mag_types:
            magnitude = [magnitude for magnitude in event.magnitudes if magnitude.magnitude_type.lower() == 'mb'][0]
            rewrite = True
        elif 'mlv' in mag_types:
            magnitude = [magnitude for magnitude in event.magnitudes if magnitude.magnitude_type.lower() == 'mlv'][0]
            rewrite = False
        elif 'ml' in mag_types:
            magnitude = [magnitude for magnitude in event.magnitudes if magnitude.magnitude_type.lower() == 'ml'][0]
            rewrite = False
        else:
            magnitude = [magnitude for magnitude in event.magnitudes if magnitude.magnitude_type.lower() == 'm'][0]
            rewrite = False       
        if rewrite == True:
            row.mag = magnitude.mag
            row.mag_type = magnitude.magnitude_type
            if 'mag_erros' in magnitude:
                row.mag_unc = magnitude.mag_errors.uncertainty
    row_out = row.copy()
    return row_out

### Recalculate magnitude
def revise_magnitudes(df,sta_mag_df_sub,sta_df,sta_corr):
    row = df.copy()
#     if row.mag_type[0:2].lower() != 'mw':
    evid = row.evid
    lat = row.lat
    lon = row.lon
    depth = row.depth
#     print(evid)
#         cat = client_NZ.get_events(eventid=evid)
#         event = cat[0]
    df_sub = sta_mag_df_sub[sta_mag_df_sub.evid.isin([evid])].reset_index(drop=True)
    for idx,sta_mag in df_sub.iterrows():
        net = sta_mag.net
        sta = sta_mag.sta
        loc = sta_mag['loc']
        chan = sta_mag.chan
        amp = sta_mag.amp
        peak = sta_mag.amp_peak
        trough = sta_mag.amp_trough
        max_amplitude = sta_mag.amp_max
        magid = sta_mag.magid
        sta_info = sta_df[(sta_df.net == net) & (sta_df.sta == sta)]
    
        if (sta_mag.mag_type.lower() == 'ml' or sta_mag.mag_type.lower() == 'mlv') and sta_mag.isnull().mag_corr == False:
            if len(sta_corr[sta_corr.sta == sta]) == 1:
                corr = sta_corr[sta_corr.sta == sta]['corr'].values[0]
            else:
                corr = 0
            if len(sta_info) > 0:
                dist, az, b_az = op.geodetics.gps2dist_azimuth(lat, lon, sta_info['lat'].values[0], sta_info['lon'].values[0])
                r_epi = dist/1000
                r_hyp = (r_epi ** 2 + (depth + sta_info['elev'].values[0]/1000) ** 2) ** 0.5
                A = max_amplitude # Amplitude used for magnitude calculation

                R = r_hyp
                h = depth

                if (h <= 40):
                    H40 = 0
                else:
                    H40 = h - 40

                NZlogA0R = 0.51 + ((-0.79E-3) * R) + (-1.67 * np.log10(R)) + ((2.38E-3) * H40) + corr # Corrected Ml for NZ according to Rhoades et al., 2020

                CML_i = np.log10(A) - NZlogA0R
                df_sub.loc[sta_mag.name,'mag_corr'] = CML_i
    return df_sub



@ray.remote
def compare_mag(df):
    # for ev_idx,row in event_df.iterrows():
    row = df.copy()
    print(row.evid,row.name)
    highcut = 20
    lowcut = 1
    corners = 4
    velocity = False

    sorter = ['HH','BH','EH','HN','BN','SH']
    channel_filter = 'HH?,BH?,EH?,HN?,BN?,SH?'
    nz20_res = 0.278 # Uncertainty of the Rhoades et al. (2020) method.
    if row.mag_type[0:2].lower() == 'mw':
        eventid = row.evid
        cat = client_NZ.get_events(eventid=eventid)
        event = cat[0]
        ev_datetime = event.preferred_origin().time
        ev_lat = event.preferred_origin().latitude
        ev_lon = event.preferred_origin().longitude
        ev_depth = event.preferred_origin().depth / 1000
    
        arrivals = event.preferred_origin().arrivals
        picks = event.picks
        pick_data = [[pick.waveform_id.network_code,
            pick.waveform_id.station_code,
            pick.waveform_id.location_code,
            pick.waveform_id.channel_code,
            pick.time,
            pick.phase_hint] for pick in picks]
        amplitudes = event.amplitudes
        pick_data.extend([[amp.waveform_id.network_code,
            amp.waveform_id.station_code,
            amp.waveform_id.location_code,
            amp.waveform_id.channel_code,
            amp.time_window.reference,
            'none'] for amp in amplitudes if not any(pick for pick in  pick_data if amp.waveform_id.station_code in pick)])
        pick_data = pd.DataFrame(pick_data,columns=['net','sta','loc','chan','time','phase_hint'])
    
        i = 0
        sta_mag_line = []
        mag_line = []
        for p_idx, pick in pick_data.iterrows():
            net = pick.net
            sta = pick.sta
            loc = pick['loc']
            chan = pick.chan
                        
            ns = [net,sta]
            row_exists = [row[1:3] for row in sta_mag_line if row[1:3] == ns]
            if row_exists:
                continue
            # Search for existing sta mag values
            sta_mag = [sta_mag for sta_mag in event.station_magnitudes if 
                ((sta_mag.waveform_id.network_code == net) &
                (sta_mag.waveform_id.station_code == sta) &
                (sta_mag.waveform_id.location_code == loc) &
                (sta_mag.waveform_id.channel_code == chan))]
# 				sta_mag = [sta_mag for sta_mag in event.station_magnitudes if sta_mag.waveform_id.station_code == sta]
            if sta_mag:
                sta_mag = sta_mag[0]
                sta_mag_mag = sta_mag.mag
                sta_mag_type = sta_mag.station_magnitude_type
                amp = [amp for amp in event.amplitudes if amp.resource_id == sta_mag.amplitude_id]
            else:
                sta_mag_mag = None
                sta_mag_type = pref_mag_type # Set to event preferred magnitude type
                amp = None
            if amp:
                amp = amp[0]
                amp_amp = amp.generic_amplitude
                amp_unit = amp.unit
                if 'unit' in amp:
                    amp_unit = amp.unit
                else:
                    amp_unit = None
            else:
                amp_amp = None
                amp_unit = None
    
            if net == 'NZ':
                client = client_NZ
            else:
                client = client_IU
                
            ### Load inventory for data!
# 				try:
            inventory_st = []
            search_channel = True
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                try:
                    inventory_st = client.get_stations(network=net, station=sta, channel=channel_filter, level='response',starttime=pick.time,endtime=pick.time)
                    for channel_code in sorter:
                        for c in inventory_st[0][0]:
                            if c.code[0:2] == channel_code:
                                loc = c.location_code
                                chan = channel_code + chan[-1]
                                search_channel = False
                                break
                        if search_channel == False:
                            break
                except:
                    pass
            if len(inventory_st) == 0:
                magid = eventid+str('m')+str(i)
                i=i+1
                mag_line.append([magid, net, sta, loc, chan, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp, None, None, None,amp_unit])
                sta_mag_line.append([magid, net, sta, loc, chan, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp, None, None, None,amp_unit])
                continue

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
                arrival = [arrival for arrival in arrivals if arrival.pick_id == pick.resource_id][0]
                r_epi = op.geodetics.degrees2kilometers(arrival.distance)
                r_hyp = ((r_epi) ** 2 + (ev_depth) ** 2) ** 0.5

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
                sta_mag = [sta_mag for sta_mag in event.station_magnitudes if ((sta_mag.waveform_id.station_code == sta) & (sta_mag.waveform_id.channel_code[2] == tr.stats.channel[2]))]
                if sta_mag:
                    sta_mag = sta_mag[0]
                    sta_mag_mag = sta_mag.mag
                    sta_mag_type = sta_mag.station_magnitude_type
                    amp = [amp for amp in event.amplitudes if amp.resource_id == sta_mag.amplitude_id]
                else:
                    sta_mag_mag = None
                    sta_mag_type = pref_mag_type # Set to event preferred magnitude type
                    amp = None
                if amp:
                    amp = amp[0]
                    amp_amp = amp.generic_amplitude
                    if 'unit' in amp:
                        amp_unit = amp.unit
                    else:
                        amp_unit = None
                else:
                    amp_amp = None
                    amp_unit = None
                tr = tr.copy()
# 						response = inventory_st.select(network=tr.stats.network,
# 							station=tr.stats.station,channel=tr.stats.channel,time=ev_datetime)

                tr = tr.split().detrend('demean').merge(fill_value=0)[0]

                tr_starttime = tr.stats.starttime
                tr_endtime = tr.stats.endtime
                tr.taper(0.05)
                tr.trim(tr.stats.starttime-5,tr.stats.endtime,pad=True,fill_value=tr.data[0])
                tr.trim(tr.stats.starttime,tr.stats.endtime+5,pad=True,fill_value=tr.data[-1])
# 					
                # Test if waveform has no measure data
                if tr.max() == 0:
                    print(f'Waveform for {tr.id} is empty')
                    magid = eventid+str('m')+str(i)
                    i = i+1
                    mag_line.append([magid, net, sta, loc, tr.stats.channel, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp, None, None, None,amp_unit])
                    sta_mag_line.append([magid, net, sta, loc, tr.stats.channel, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp, None, None, None,amp_unit])
                    continue
                
                # Test if Nyquist frequency is lower than highpass filter
# 						if tr.stats.sampling_rate/2 < lowcut:
# 							print(f'Sampling rate for {tr.id} too low')
# 							magid = eventid+str('m')+str(i)
# 							i = i+1
# 							mag_line.append([magid, net, sta, loc, tr.stats.channel, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp, None, None, None])
# 							sta_mag_line.append([magid, net, sta, loc, tr.stats.channel, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp, None, None, None])
# 							continue
                
# 						tr.filter('highpass', freq=lowcut,
# 							corners = corners)

                tr = _sim_WA(tr, inventory_st, 0, velocity=False)
        
                # Check if there is no data, or if the trace is all 0s
                if tr == None or tr.max() == 0:
                    magid = eventid+str('m')+str(i)
                    i = i+1
                    mag_line.append([magid, net, sta, loc, tr.stats.channel, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp, None, None, None,amp_unit])
                    sta_mag_line.append([magid, net, sta, loc, tr.stats.channel, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp, None, None, None,amp_unit])
                    continue

                tr.trim(tr_starttime,tr_endtime)

# 						tr.remove_response(inventory=inventory_st,output="VEL",water_level=1)
# 					tr.attach_response(inventory)
# 						tr = sim_wa(inventory, tr)
# 							snr = abs(tr.max() / np.sqrt(np.mean(np.square(tr.data))))
                # Test SNR in noise window, if available, else use alternative SNR calc
                if len(tr.slice(starttime=noise_window[0], endtime=noise_window[1]).data) > 0:
                    snr = _snr(tr,noise_window,signal_window)
                else:
                    snr = abs(tr.max() / np.sqrt(np.mean(np.square(tr.data))))
                if snr < 3:
                    # Not a high enough SNR
                    print(f'SNR for {tr.id} not high enough')
                    magid = eventid+str('m')+str(i)
                    i = i+1
                    mag_line.append([magid, net, sta, loc, tr.stats.channel, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp, None, None, None,amp_unit])
                    sta_mag_line.append([magid, net, sta, loc, tr.stats.channel, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp, None, None, None,amp_unit])
                    continue							

                # Calculate the normalized noise amplitude
                amplitude, period, delay, peak, trough = _max_p2t(
                    tr.data, tr.stats.delta, return_peak_trough=True)
# 						print(amplitude,peak,trough)
                                    
# 						if peak >= abs(trough):
# 							amplitude = peak
# 						else:
# 							amplitude = abs(trough)
# 						
# 						# Calculate the absolute amplitude
                max_amplitude = abs(tr.max())
            
        
                # Generate poles and zeros for the filter we used earlier.
                # We need to get the gain for the digital SOS filter used by
                # obspy.
# 						sos = iirfilter(
# 							corners, [lowcut / (0.5 * tr.stats.sampling_rate)],
# 							btype='highpass', ftype='butter', output='sos')
# 						_, gain = sosfreqz(sos, worN=[1 / period],
# 										   fs=tr.stats.sampling_rate)
# 						gain = np.abs(gain[0])  # Convert from complex to real.
# 						if gain < 1e-2:
# 							print(
# 								f"Pick made outside stable pass-band of filter "
# 								f"on {tr.id}, rejecting")
# 							magid = eventid+str('m')+str(i)
# 							i = i+1
# 							mag_line.append([magid, net, sta, loc, nslc[-1], eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp, None, None, None])
# 							sta_mag_line.append([magid, net, sta, loc, nslc[-1], eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp, None, None, None])
# 							continue
# 						amplitude /= gain
                amplitude *= 0.5
                amplitude *= 1000
# 						peak /= gain
                peak *= 1000
# 						trough /= gain
                trough *= 1000
# 						max_amplitude /= gain
                max_amplitude *= 1000
# 						print(amplitude,amp_amp,amp_time,windowStart+delay,r_hyp,sta,chan)
                        
                if r_epi:
                    amp_unit = 'mm'
                    
                    R = r_hyp
                    h = ev_depth

# 							logA0R = 1.110 * np.log10(R / 100) + 0.00189 * (R - 100) + 3.0  # L. K. Hutton and David M. Boore, BSSA, 1987: The Ml for southern California
# 							logA0R = 0.2869 - 1.272*1e-3*(R) -(1.493 * np.log10(R))	+ corr	# Ristau 2016
# 						A = 10 ** (sta_mag_mag - logA0R)

                    A = max_amplitude
                    ML_i = np.log10(A) + np.log10(R) + 0.0029 * R # Robinson, 1987, add K for station correction

                    if (h <= 40):
                        H40 = 0
                    else:
                        H40 = h - 40

                    NZlogA0R = 0.51 + ((-0.79E-3) * R) + (-1.67 * np.log10(R)) + ((2.38E-3) * H40) + corr # Corrected Ml for NZ according to Rhoades et al., 2021
                    CML_i = np.log10(A) - NZlogA0R
# 							ML_i = np.log10(A) - logA0R

                    magid = eventid+str('m')+str(i)
                    i = i+1
                    mag_line.append([magid, net, sta, loc, tr.stats.channel, eventid, sta_mag_mag, sta_mag_type, CML_i, 'NZ20', A, peak, trough, max_amplitude,amp_unit])
                    sta_mag_line.append([magid, net, sta, loc, tr.stats.channel, eventid,  sta_mag_mag, sta_mag_type, CML_i, 'NZ20', A, peak, trough, max_amplitude,amp_unit])
                else: # If there is no epicentral distance measured
                    magid = eventid+str('m')+str(i)
                    i = i+1
                    mag_line.append([magid, net, sta, loc, tr.stats.channel, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp, None, None, None,amp_unit])
                    sta_mag_line.append([magid, net, sta, loc, tr.stats.channel, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp, None, None, None,amp_unit])

        sta_mag_df = pd.DataFrame(sta_mag_line, columns=['magid', 'net', 'sta', 'loc', 'chan', 'evid', 'mag', 'mag_type', 'mag_corr', 'mag_corr_method', 'amp', 'amp_peak', 'amp_trough', 'amp_max','amp_unit'])

    #     print(evid)
    #         cat = client_NZ.get_events(eventid=evid)
    #         event = cat[0]
        df_sub = sta_mag_df
    #         df_sub = sta_mag_out.query("evid == "+"'"+str(evid)+"'")
        mag_data = df_sub[df_sub.chan.str.endswith('Z',na=False)]
        if (mag_data.mag_corr.isnull() == False).sum() < 2:
            mag_data = df_sub[df_sub.chan.str.endswith(('E','N','1','2'),na=False)]
            mag_type = 'cMl_H'
            if len(mag_data[mag_data.mag_corr.isnull() == False].sta.unique()) < 2:
                mag_data = df_sub
    #             print(evid,event.mag,event.mag_type,event.mag_method,event.mag_unc,event.nmag)
                mag_type = pref_mag_type
                row_out = row.copy()
            else:
                mag_type = 'cMl_H'
        else:
            mag_type = 'cMl'
        if mag_type == 'cMl' or mag_type == 'cMl_H':
    #     mags = mag_data[mag_data.mag_corr.isnull() == False].mag_corr
            std = mag_data.mag_corr.std()
            mean = mag_data.mag_corr.mean()

            # Find the inner quartile range of magnitude data and remove outliers
            Q1 = mag_data.mag_corr.quantile(0.25)
            Q3 = mag_data.mag_corr.quantile(0.75)
            IQR=Q3-Q1
            lowqe_bound=Q1 - 1.5 * IQR
            upper_bound=Q3 + 1.5 * IQR
        #     print(lowqe_bound,upper_bound)

            IQR_mags = mag_data[~((mag_data.mag_corr < lowqe_bound) |(mag_data.mag_corr > upper_bound))]
            # Filter out stations with an extremely high or low magnitude
            if len(IQR_mags.sta.unique()) <= 3:
                IQR_mags = IQR_mags[(IQR_mags.mag_corr <= row.mag_orig + 2) & (IQR_mags.mag_corr >= row.mag_orig - 2)]

        #     new_mag_data = mag_data[(mag_data.mag_corr >= mean - (2 * std)) & (mag_data.mag_corr <= mean + (2 * std))]
            new_mean = IQR_mags.mag_corr.mean()
            new_std = IQR_mags.mag_corr.std()
            CML_unc = np.sqrt(nz20_res ** 2 + np.var(IQR_mags.mag_corr))

            nmag = len(IQR_mags[~IQR_mags.mag_corr.isnull()].sta.unique())

            row[['mag','mag_unc','mag_type','mag_method','nmag','mag_orig','mag_orig_type',
                'mag_orig_unc']] = new_mean, CML_unc, mag_type, 'NZ20', nmag, pref_mag, pref_mag_type, pref_mag_unc
            row_out = row.copy()
            #     row_out = row_out.to_frame().T
            return [row_out,sta_mag_df]

# @ray.remote
def fix_mag(df,sta_mag_df_sub):
# event_sub_mask = events_final.datetime.values.astype('datetime64[Y]').astype(int)+1970 == process_year
# event_sub = events_final[event_sub_mask].reset_index(drop=True)
# for ev_idx,row in event_sub.iterrows():
    row = df.copy()
#     print(row.evid,row.name)
    if row.mag_type.lower() == 'mw':
        evid = row.evid
        cat = client_NZ.get_events(eventid=evid)
        event = cat[0]
        ev_lat = event.preferred_origin().latitude
        ev_lon = event.preferred_origin().longitude
        ev_depth = event.preferred_origin().depth / 1000
    #         ev_lat = row.lat
    #         ev_lon = row.lon
    #         ev_depth = row.depth
        df_sub = sta_mag_df_sub[sta_mag_df_sub.evid.isin([evid])].reset_index(drop=True)
    #         df_sub = df_sub[df_sub.mag_type == 'MLv'].reset_index(drop=True)
        for idx, sta_mag in df_sub.iterrows():
            r_epi = []
            net = sta_mag.net
            sta = sta_mag.sta
            sta_info = station_df[station_df.sta == sta]
            if len(sta_corr[sta_corr.sta == sta]) == 1:
                corr = sta_corr[sta_corr.sta == sta]['corr'].values[0]
            else:
                corr = 0
        
            if len(sta_info) > 0:
                dist, az, b_az = op.geodetics.gps2dist_azimuth(ev_lat, ev_lon, sta_info['lat'].values[0], sta_info['lon'].values[0])
                r_epi = dist/1000
                r_hyp = (r_epi ** 2 + (ev_depth + sta_info['elev'].values[0]/1000) ** 2) ** 0.5
            
            if r_epi:
                R = r_hyp
                h = ev_depth

                A = sta_mag.amp_max
                ML_i = np.log10(A) + np.log10(R) + 0.0029 * R # Robinson, 1987, add K for station correction

                if (h <= 40):
                    H40 = 0
                else:
                    H40 = h - 40

                NZlogA0R = 0.51 + ((-0.79E-3) * R) + (-1.67 * np.log10(R)) + ((2.38E-3) * H40) + corr # Corrected Ml for NZ according to Rhoades et al., 2020
                CML_i = np.log10(A) - NZlogA0R
                df_sub.loc[idx, 'mag_corr'] = CML_i
        
        mag_data = df_sub[df_sub.chan.str.endswith('Z',na=False)]
        if (mag_data.mag_corr.isnull() == False).sum() < 2:
            mag_data = df_sub[df_sub.chan.str.endswith(('E','N','1','2'),na=False)]
            mag_type = 'cMl_H'
            if len(mag_data[mag_data.mag_corr.isnull() == False].sta.unique()) < 2:
                mag_data = df_sub
    #             print(evid,event.mag,event.mag_type,event.mag_method,event.mag_unc,event.nmag)
                mag_type = row.mag_orig_type
                row_out = row.copy()
            else:
                mag_type = 'cMl_H'
        else:
            mag_type = 'cMl'
        if mag_type == 'cMl' or mag_type == 'cMl_H':
    #     mags = mag_data[mag_data.mag_corr.isnull() == False].mag_corr
            std = mag_data.mag_corr.std()
            mean = mag_data.mag_corr.mean()

            # Find the inner quartile range of magnitude data and remove outliers
            Q1 = mag_data.mag_corr.quantile(0.25)
            Q3 = mag_data.mag_corr.quantile(0.75)
            IQR=Q3-Q1
            lowqe_bound=Q1 - 1.5 * IQR
            upper_bound=Q3 + 1.5 * IQR
        #     print(lowqe_bound,upper_bound)

            IQR_mags = mag_data[~((mag_data.mag_corr < lowqe_bound) |(mag_data.mag_corr > upper_bound))]
            # Filter out stations with an extremely high or low magnitude
            if len(IQR_mags.sta.unique()) <= 3:
                IQR_mags = IQR_mags[(IQR_mags.mag_corr <= row.mag_orig + 2) & (IQR_mags.mag_corr >= row.mag_orig - 2)]

        #     new_mag_data = mag_data[(mag_data.mag_corr >= mean - (2 * std)) & (mag_data.mag_corr <= mean + (2 * std))]
            new_mean = IQR_mags.mag_corr.mean()
            new_std = IQR_mags.mag_corr.std()
            CML_unc = np.sqrt(nz20_res ** 2 + np.var(IQR_mags.mag_corr))
            print(evid, row.mag_orig, new_mean, np.mean(df_sub.mag_corr))

            nmag = len(IQR_mags[~IQR_mags.mag_corr.isnull()].sta.unique())

            row[['mag','mag_unc','mag_type','mag_method','nmag','mag_orig','mag_orig_type',
                'mag_orig_unc']] = new_mean, CML_unc, mag_type, 'NZ20', nmag, pref_mag, pref_mag_type, pref_mag_unc
            row_out = row.copy()
    #     row_out = row_out.to_frame().T
    return row_out


event_df = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/earthquake_source_table_complete.csv',low_memory=False)
# sta_mag_df = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/station_magnitude_table_fixed_mags.csv',low_memory=False)

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
sta_corr = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/sta_corr_new.csv')

nz20_res = 0.278

years = np.arange(2003,2021)
# months = np.arange(1,13)
# years = np.unique(geonet.origintime.values.astype('datetime64[Y]').astype(int)+1970)
for year in years:
# 	for month in months:
    process_year = year
    print('Processing for year '+str(process_year))
    
    event_sub_mask = event_df.datetime.values.astype('datetime64[Y]').astype(int)+1970 == process_year
    event_sub = event_df[event_sub_mask].reset_index(drop=True)
#     sta_mag_df_sub = sta_mag_df[sta_mag_df.evid.isin(event_sub.evid.unique())].reset_index(drop=True)

    pandarallel.initialize(nb_workers=8,progress_bar=False) # Customize the number of parallel workers
    
    result_ids = []
    ray.init()
    result_ids = [compare_mag.remote(x) for idx,x in event_sub.iterrows()]
    results = ray.get(result_ids)
    ray.shutdown()
    
    if not all(result is None for result in results):
        event_out = pd.concat([result[0].to_frame().T for result in results if result is not None])
        sta_mag_out = pd.concat([result[1] for result in results if result is not None])
    #     if None not in event_out:
        event_out.to_csv('event_MW_compare_'+str(process_year)+'.csv',index=False)
        sta_mag_out.to_csv('sta_MW_compare_'+str(process_year)+'.csv',index=False)
    else:
        print('No results for '+str(process_year))
    
events_final = pd.concat([pd.read_csv(f,low_memory=False) for f in glob.glob('event_MW_compare_*.csv')])
sta_mag_final = pd.concat([pd.read_csv(f,low_memory=False) for f in glob.glob('sta_MW_compare_*.csv')])
events_final.to_csv('earthquake_all_MW_compare.csv',index=False)
sta_mag_final.to_csv('station_all_MW_compare.csv',index=False)


sta_mag_df_sub = sta_mag_final[sta_mag_final.evid.isin(event_sub.evid.unique())].reset_index(drop=True)
