import ray
import pandas as pd
import os
import glob
from obspy.clients.fdsn import Client as FDSN_Client

@ray.remote
def local_magnitude(event, client_NZ, client_IU, station_df, sta_corr, arr_df):

    import obspy as op
    from obspy import UTCDateTime
    import numpy as np
    from magnitude.woodanderson import _sim_WA,_rms,_snr,_max_p2t
    from geopy.distance import geodesic
    import pandas as pd
    from obspy import read_inventory
    from obspy.clients.fdsn import Client as FDSN_Client
    import warnings
    from scipy.signal import iirfilter, sosfreqz

    highcut = 20
    lowcut = 1
    corners = 4
    velocity = False
    sorter = ['HH','BH','EH','HN','BN','SH']
    channel_filter = 'HH?,BH?,EH?,HN?,BN?,SH?'
    nz20_res = 0.278 # Uncertainty of the Rhoades et al. (2020) method.

    #     client_NZ = FDSN_Client("GEONET")
    #     client_IU = FDSN_Client('IRIS')
    mag_line = []
    event_line = []

    #     for ii, event in event_df[0:10].iterrows():
    evid = event.evid
    print('Calculating magnitude for event '+str(evid))
    lat = event.lat
    lon = event.lon
    depth = event.depth
    ev_datetime = UTCDateTime(event.datetime)
    arr_sub = arr_df[arr_df.evid == evid]
    
    i = 1
    sta_mag_line = []
    for jj,arrival in arr_sub.iterrows():
        net = arrival.net
        sta = arrival.sta
        if net == 'NZ':
            client = client_NZ
        else:
            client = client_IU
            
        ns = [net,sta]
        row_exists = [row[1:3] for row in sta_mag_line if row[1:3] == ns]
        if row_exists:
            continue
            
        chan = arrival.chan
        loc = '*'
        datetime = UTCDateTime(arrival.datetime)
        phase = arrival.phase

        inventory_st = []
        search_channel = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            try:
                inventory_st = client.get_stations(network=net, station=sta, channel=channel_filter, level='response',starttime=datetime,endtime=datetime)
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
            continue

        if len(sta_corr[sta_corr.sta == sta]) == 1:
            corr = sta_corr[sta_corr.sta == sta]['corr'].values[0]
        else:
            corr = 0

        sta_info = station_df[station_df.sta == sta]
        if len(sta_info) > 0:
            sta_info = sta_info.iloc[0]
            dist, az, b_az = op.geodetics.gps2dist_azimuth(lat, lon, sta_info.lat, sta_info.lon)
            r_epi = dist/1000
            r_hyp = (r_epi ** 2 + (depth + sta_info.elev/1000) ** 2) ** 0.5
        else:
            continue

        slow_vel = 3
        endtime = ev_datetime + r_hyp/slow_vel + 30

        if phase.lower()[0] == 'p':
            windowStart = datetime - 35
            windowEnd = endtime
            noise_window = [windowStart,windowStart+30]
            signal_window = [datetime-2,datetime+30]				
        else:
            windowStart = datetime - 45
            windowEnd = endtime
            noise_window = [windowStart,windowStart+30]
            signal_window = [datetime-12,datetime+20]				
        try:
            st = client.get_waveforms(net,sta,loc,chan[0:2]+'?',windowStart,windowEnd)
            st = st.merge()
        except:
            continue

        for tr in st:
            filt_data = False

            tr = tr.copy()

            tr = tr.split().detrend('demean').merge(fill_value=0)[0]

            tr_starttime = tr.stats.starttime
            tr_endtime = tr.stats.endtime
    #             tr.taper(0.05)
            tr.trim(tr.stats.starttime-5,tr.stats.endtime,pad=True,fill_value=tr.data[0])
            tr.trim(tr.stats.starttime,tr.stats.endtime+5,pad=True,fill_value=tr.data[-1])
    # 					
            # Test if waveform has no measure data
            if tr.max() == 0:
                print(f'Waveform for {tr.id} is empty')
                continue
    
            # Test if Nyquist frequency is lower than highpass filter
            if tr.stats.sampling_rate/2 < lowcut:
                print(f'Sampling rate for {tr.id} too low')
                continue
    

            tr = _sim_WA(tr, inventory_st, 0, velocity=False)
            trim_tr = tr.copy().trim(tr_starttime,tr_endtime)


            # Test SNR in noise window, if available, else use alternative SNR calc
            if len(trim_tr.slice(starttime=noise_window[0], endtime=noise_window[1]).data) > 0:
                snr = _snr(trim_tr,noise_window,signal_window)
            else:
                snr = abs(trim_tr.max() / np.sqrt(np.mean(np.square(trim_tr.data))))
            if snr < 3:
                # Not a high enough SNR
                print(f'SNR for {tr.id} not high enough, attempting to filter data')
      
                tr.filter('highpass', freq=lowcut,
                    corners = corners, zerophase=True)
                trim_tr = tr.copy().trim(tr_starttime,tr_endtime)
                if len(trim_tr.slice(starttime=noise_window[0], endtime=noise_window[1]).data) > 0:
                    snr = _snr(trim_tr,noise_window,signal_window)
                else:
                    snr = abs(trim_tr.max() / np.sqrt(np.mean(np.square(trim_tr.data))))
                if snr >= 3:
                    filt_data = True
                else:
                    print(f'SNR for {tr.id} is still not high enough')
                    continue
                    
            tr.trim(tr_starttime,tr_endtime)

            # Check if there is no data, or if the trace is all 0s
            if tr == None or tr.max() == 0:
                continue
    
            # Calculate the normalized noise amplitude
            amplitude, period, delay, peak, trough = _max_p2t(
                tr.data, tr.stats.delta, return_peak_trough=True)
    # 						print(amplitude,peak,trough)
                        
            if peak >= abs(trough):
                amplitude = peak
            else:
                amplitude = abs(trough)
    # 						
    # 						# Calculate the absolute amplitude
            max_amplitude = abs(tr.max())


            # Generate poles and zeros for the filter we used earlier.
            # We need to get the gain for the digital SOS filter used by
            # obspy.
            if filt_data:
                sos = iirfilter(
                    corners * 2, [lowcut / (0.5 * tr.stats.sampling_rate)],
                    btype='highpass', ftype='butter', output='sos')
                _, gain = sosfreqz(sos, worN=[1 / period],
                                   fs=tr.stats.sampling_rate)
                gain = np.abs(gain[0])  # Convert from complex to real.
                if gain < 1e-2:
                    print(
                        f"Pick made outside stable pass-band of filter "
                        f"on {tr.id}, rejecting")
                    continue
                amplitude /= gain
                peak /= gain
                trough /= gain
                max_amplitude /= gain
    #             amplitude *= 0.5
            amplitude *= 1000
            peak *= 1000
            trough *= 1000
            max_amplitude *= 1000
    # 						print(amplitude,amp_amp,amp_time,windowStart+delay,r_hyp,sta,chan)
            
            if r_epi:
                amp_unit = 'mm'
                R = r_hyp
                h = depth

    #                     logA0R = 1.110 * np.log10(R / 100) + 0.00189 * (R - 100) + 3.0  # L. K. Hutton and David M. Boore, BSSA, 1987: The Ml for southern California
    #                     logA0R = 0.2869 - 1.272*1e-3*(R) -(1.493 * np.log10(R))	+ corr	# Ristau 2016
    # 						A = 10 ** (sta_mag_mag - logA0R)

                A = max_amplitude
    #                 ML_i = np.log10(A) + np.log10(R) + 0.0029 * R # Robinson, 1987, add K for station correction

                if (h <= 40):
                    H40 = 0
                else:
                    H40 = h - 40

                NZlogA0R = 0.51 + ((-0.79E-3) * R) + (-1.67 * np.log10(R)) + ((2.38E-3) * H40) + corr # Corrected Ml for NZ according to Rhoades et al., 2020
                CML_i = np.log10(A) - NZlogA0R
    # 							ML_i = np.log10(A) - logA0R

                magid = str(evid)+str('m')+str(i)
                i = i+1
                mag_line.append([magid, net, sta, loc, tr.stats.channel, evid, None, None, CML_i, 'NZ20', amplitude, peak, trough, max_amplitude, amp_unit])
                sta_mag_line.append([magid, net, sta, loc, tr.stats.channel, evid,  None, None, CML_i, 'NZ20', amplitude, peak, trough, max_amplitude, amp_unit])

    sta_mag_df = pd.DataFrame(sta_mag_line, columns=['magid', 'net', 'sta', 'loc', 'chan', 'evid', 'mag', 'mag_type', 'mag_corr', 'mag_corr_method', 'amp', 'amp_peak', 'amp_trough', 'amp_max', 'amp_unit'])

    if len(sta_mag_df[sta_mag_df.mag_corr.isnull() == False].sta.unique()) > 1:
        mag_data = sta_mag_df[sta_mag_df.chan.str.endswith('Z',na=False)]
        if (mag_data.mag_corr.isnull() == False).sum() < 2:
            mag_data = sta_mag_df[sta_mag_df.chan.str.endswith(('E','N','1','2'),na=False)]
            if len(mag_data[mag_data.mag_corr.isnull() == False].sta.unique()) < 2:
                mag_data = sta_mag_df
                mag_type = None
                mag_method = None
    #             print(evid,event.mag,event.mag_type,event.mag_method,event.mag_unc,event.nmag)
            else:
                mag_type = 'cMl_H'
                mag_method = 'NZ20'
        else:
            mag_type = 'cMl'
            mag_method = 'NZ20'
        if mag_type != None:
            Q1 = mag_data.mag_corr.quantile(0.25)
            Q3 = mag_data.mag_corr.quantile(0.75)
            IQR=Q3-Q1
            lowqe_bound=Q1 - 1.5 * IQR
            upper_bound=Q3 + 1.5 * IQR
        #     print(lowqe_bound,upper_bound)

            IQR_mags = mag_data[~((mag_data.mag_corr < lowqe_bound) | (mag_data.mag_corr > upper_bound))]
            # Run a check for anomalous station magnitudes (+ or - 2 from original value)
    #                 IQR_mags = IQR_mags[(IQR_mags.mag_corr <= event.preferred_magnitude().mag + 2) & (IQR_mags.mag_corr >= event.preferred_magnitude().mag - 2)]

            CML = IQR_mags.mag_corr.mean()
            new_std = IQR_mags.mag_corr.std()
            CML_unc = np.sqrt(nz20_res ** 2 + np.var(IQR_mags.mag_corr))

            nmag = len(IQR_mags[~IQR_mags.mag_corr.isnull()].sta.unique())

            event['mag'] = CML
            event['mag_type'] = mag_type
            event['mag_method'] = mag_method
            event['mag_unc'] = CML_unc
            event['nmag'] = nmag
            event_line.append(event.tolist())
    else:
        event['mag'] = None
        event['mag_type'] = None
        event['mag_method'] = None
        event['mag_unc'] = None
        event['nmag'] = None
        event_line.append(event.tolist())

    return [event_line,mag_line]

def calc_mags(event_file,arr_file,sta_corr,out_directory,station_df):
    import numpy as np
    import pandas as pd
    
    client_NZ = FDSN_Client("GEONET")
    client_IU = FDSN_Client('IRIS')
    if len(station_df) == 0:
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

    datestr = os.path.basename(event_file).split('_')[0]
    event_df = pd.read_csv(event_file,low_memory=False)
    arr_df = pd.read_csv(arr_file,low_memory=False)

    if not os.path.exists(out_directory+'/associations'):
        os.makedirs(out_directory+'/associations')
    if not os.path.exists(out_directory+'/events'):
        os.makedirs(out_directory+'/events')
    if not os.path.exists(out_directory+'/magnitudes'):
        os.makedirs(out_directory+'/magnitudes')
    
#     for idx,event in event_df.iterrows():
#         results = local_magnitude(event,client_NZ, client_IU, station_df, sta_corr, arr_df)

    result_ids = []
    ray.init()
    result_ids = [local_magnitude.remote(event,client_NZ, client_IU, station_df, sta_corr, arr_df) for idx, event in event_df.iterrows()]
    results = ray.get(result_ids)
    ray.shutdown()
    results = np.array(results)

    event_results = [event for result in results[:,0] for event in result]
    mag_results = [mag for result in results[:,1] for mag in result]

    event_out_df = pd.DataFrame(event_results,columns=['evid', 'datetime', 'lat', 'lon', 
           'depth', 'ndef', 'nsta', 'reloc', 'minimum', 'finalgrid', 'x', 'y', 'z', 'x_c', 
           'y_c', 'z_c', 'major','minor', 'z_err', 'theta', 'Q', 'mag', 'mag_type', 
           'mag_method', 'mag_unc', 'nmag'])
    event_out_df = event_out_df[['evid', 'datetime', 'lat', 'lon', 'depth', 'mag', 'mag_type', 
           'mag_method', 'mag_unc', 'ndef', 'nsta', 'nmag', 'reloc', 'minimum', 'finalgrid', 
           'x', 'y', 'z', 'x_c', 'y_c', 'z_c', 'major','minor', 'z_err', 'theta', 'Q']]

    mag_df = pd.DataFrame(mag_results,columns=['magid', 'net', 'sta', 'loc', 'chan', 'evid', 
           'mag', 'mag_type', 'mag_corr', 'mag_corr_method', 'amp', 'amp_peak', 'amp_trough',
           'amp_max','amp_unit'])

    event_out_df.to_csv(out_directory+'/events/'+datestr+'_events.csv',index=False)
    arr_df.to_csv(out_directory+'/associations/'+datestr+'_assocs.csv',index=False)
    mag_df.to_csv(out_directory+'/magnitudes/'+datestr+'_magnitudes.csv',index=False)

if __name__ == "__main__":
    event_dir = '/Volumes/SeaJade 2 Backup/NZ/EQTransformer/output/catalog_test'
    arrival_dir = '/Volumes/SeaJade 2 Backup/NZ/EQTransformer/output/arrivals_test'
    sta_corr = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/sta_corr_new.csv')
    out_directory = '/Volumes/SeaJade 2 Backup/NZ/EQTransformer/output/mag_out_geonet'

    event_files = glob.glob(event_dir+'/*.csv')
    arrival_files = glob.glob(arrival_dir+'/*.csv')

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

    for event_file in event_files[103:365]:
        datestr = os.path.basename(event_file).split('_')[0]
        arr_file = arrival_dir+'/'+datestr+'_arrivals.csv'
        calc_mags(event_file,arr_file,sta_corr,out_directory,station_df)
#     datestr = os.path.basename(event_file).split('_')[0]
#     event_df = pd.read_csv(event_file,low_memory=False)
#     arr_file = arrival_dir+'/'+datestr+'_assoc.csv'
#     arr_df = pd.read_csv(arr_file,low_memory=False)
# 
# #     arr_df = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/EQTransformer/output/finn_arrivals_test/20190726_arrivals.csv',low_memory=False)
# #     event_df = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/EQTransformer/output/finn_catalog_test/20190726_origins.csv',low_memory=False)
# 
#     if not os.path.exists(out_directory):
#         os.makedirs(out_directory+'/arrivals')
#         os.makedirs(out_directory+'/events')
#         os.makedirs(out_directory+'/magnitudes')
# 
# 
#     result_ids = []
#     ray.init()
#     result_ids = [local_magnitude.remote(event,client_NZ, client_IU, station_df, sta_corr, arr_df) for idx, event in event_df.iterrows()]
#     results = ray.get(result_ids)
#     ray.shutdown()
#     results = np.array(results)
# 
#     event_results = [event for result in results[:,0] for event in result]
#     mag_results = [mag for result in results[:,1] for mag in result]
# 
#     event_out_df = pd.DataFrame(event_results,columns=['evid', 'datetime', 'lat', 'lon', 
#            'depth', 'ndef', 'nsta', 'reloc', 'minimum', 'finalgrid', 'x', 'y', 'z', 'x_c', 
#            'y_c', 'z_c', 'x_err','y_err', 'z_err', 'theta', 'Q', 'mag', 'mag_type', 
#            'mag_method', 'mag_unc', 'nmag'])
#     event_out_df = event_out_df[['evid', 'datetime', 'lat', 'lon', 'depth', 'mag', 'mag_type', 
#            'mag_method', 'mag_unc', 'ndef', 'nsta', 'nmag', 'reloc', 'minimum', 'finalgrid', 
#            'x', 'y', 'z', 'x_c', 'y_c', 'z_c', 'x_err','y_err', 'z_err', 'theta', 'Q']]
# 
#     mag_df = pd.DataFrame(mag_results,columns=['magid', 'net', 'sta', 'loc', 'chan', 'evid', 
#            'mag', 'mag_type', 'mag_corr', 'mag_corr_method', 'amp', 'amp_peak', 'amp_trough',
#            'amp_max'])
# 
#     event_out_df.to_csv(directory+'/events/'+datestr+'_events.csv')
#     arr_df.to_csv(directory+'/associations/'+datestr+'_assocs.csv')
#     mag_df.to_csv(directory+'/magnitudes/'+datestr+'_magnitudes.csv')