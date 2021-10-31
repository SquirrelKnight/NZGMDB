import ray
import pandas as pd
import os
import glob
from obspy.clients.fdsn import Client as FDSN_Client

@ray.remote
def local_magnitude(event, station_df, sta_corr, sta_mag_df_sub):

    from obspy import UTCDateTime
    import numpy as np
    from geopy.distance import geodesic
    import pandas as pd

#     highcut = 20
#     lowcut = 1
#     corners = 4
#     velocity = False
#     sorter = ['HH','BH','EH','HN','BN','SH']
#     channel_filter = 'HH?,BH?,EH?,HN?,BN?,SH?'
    nz20_res = 0.278 # Uncertainty of the Rhoades et al. (2020) method.

    #     client_NZ = FDSN_Client("GEONET")
    #     client_IU = FDSN_Client('IRIS')
    mag_line = []
    event_line = []

    #     for ii, event in event_df[0:10].iterrows():
    evid = str(event.evid)
    print('Calculating magnitude for event '+str(evid))
    lat = event.lat
    lon = event.lon
    depth = event.depth
    ev_datetime = UTCDateTime(event.datetime)
    mag_sub = sta_mag_df_sub[sta_mag_df_sub.evid == evid].reset_index(drop=True)
    
    sta_mag_line = []
    if len(mag_sub) > 0:
        i = 1
        for jj,magnitude in mag_sub.iterrows():
            magid = magnitude.magid
            net = magnitude.net
            sta = magnitude.sta
            loc = magnitude['loc']
            chan = magnitude.chan
            mag = magnitude.mag
            mag_type = magnitude.mag_type
            mag_corr = magnitude.mag_corr
            mag_corr_method = magnitude.mag_corr_method
            amp = magnitude.amp
            amp_peak = magnitude.amp_peak
            amp_trough = magnitude.amp_trough
            amp_max = magnitude.amp_max
            amp_unit = magnitude.amp_unit
            if np.isnan(mag_corr):
                sta_mag_line.append([magid,net,sta,loc,chan,evid,mag,mag_type,mag_corr,
                    mag_corr_method,amp,amp_peak,amp_trough,amp_max,amp_unit])
            else:
                if len(sta_corr[sta_corr.sta == sta]) == 1:
                    corr = sta_corr[sta_corr.sta == sta]['corr'].values[0]
                else:
                    corr = 0

                sta_info = station_df[station_df.sta == sta]
                r_epi = []
                if len(sta_info) > 0:
                    sta_info = sta_info.iloc[0]
                    dist, az, b_az = op.geodetics.gps2dist_azimuth(lat, lon, sta_info.lat, sta_info.lon)
                    r_epi = dist/1000
                    r_hyp = (r_epi ** 2 + (depth + sta_info.elev/1000) ** 2) ** 0.5
                if r_epi:
                    amp_unit = 'mm'
                    R = r_hyp
                    h = depth

        #                     logA0R = 1.110 * np.log10(R / 100) + 0.00189 * (R - 100) + 3.0  # L. K. Hutton and David M. Boore, BSSA, 1987: The Ml for southern California
        #                     logA0R = 0.2869 - 1.272*1e-3*(R) -(1.493 * np.log10(R))	+ corr	# Ristau 2016
        # 						A = 10 ** (sta_mag_mag - logA0R)

                    A = amp
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
                    sta_mag_line.append([magid, net, sta, loc, chan, evid,  mag, mag_type, 
                        CML_i, 'NZ20', amp, amp_peak, amp_trough, amp_max, amp_unit])
                else:
                    sta_mag_line.append([magid,net,sta,loc,chan,evid,mag,mag_type,mag_corr,
                        mag_corr_method,amp,amp_peak,amp_trough,amp_max,amp_unit])
        
        sta_mag_df = pd.DataFrame(sta_mag_line, columns=['magid', 'net', 'sta', 'loc', 
            'chan', 'evid', 'mag', 'mag_type', 'mag_corr', 'mag_corr_method', 'amp', 
            'amp_peak', 'amp_trough', 'amp_max', 'amp_unit'])
    
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
    
    return [event_line,sta_mag_line]

def calc_mags(out_directory,event_file,arr_file,mag_file,sta_corr,station_df,event_df_sub,arr_df_sub,sta_mag_df_sub):
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

    if not os.path.exists(out_directory+'/associations'):
        os.makedirs(out_directory+'/associations')
    if not os.path.exists(out_directory+'/events'):
        os.makedirs(out_directory+'/events')
    if not os.path.exists(out_directory+'/magnitudes'):
        os.makedirs(out_directory+'/magnitudes')
    
#     for idx,event in event_df.iterrows():
#         results = local_magnitude(event,station_df, sta_corr, sta_mag_df_sub)

    result_ids = []
    ray.init()
    result_ids = [local_magnitude.remote(event, station_df, sta_corr, sta_mag_df_sub) for idx, event in event_df_sub.iterrows()]
    results = ray.get(result_ids)
    ray.shutdown()
    results = np.array(results)

    event_results = [event for result in results[:,0] for event in result if result is not None]
    mag_results = [mag for result in results[:,1] for mag in result if result is not None]

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

    event_out_df.to_csv(out_directory+'/events/'+event_file,index=False)
    arr_df_sub.to_csv(out_directory+'/associations/'+arr_file,index=False)
    mag_df.to_csv(out_directory+'/magnitudes/'+mag_file,index=False)

if __name__ == "__main__":
    sta_mag_df_all = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/station_magnitude_table.csv',
        low_memory=False)
    event_dir = '/Volumes/SeaJade 2 Backup/NZ/EQTransformer/output/catalog_test'
    arrival_dir = '/Volumes/SeaJade 2 Backup/NZ/EQTransformer/output/arrivals_test'
    sta_corr = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/sta_corr_new.csv')
    out_directory = '/Volumes/SeaJade 2 Backup/NZ/EQTransformer/output/mag_out_geonet'

    event_files = glob.glob(event_dir+'/*.csv')
    arrival_files = glob.glob(arrival_dir+'/*.csv')
    event_df = pd.concat([pd.read_csv(f,low_memory=False,dtype={'evid':object}) for f in event_files])
    arr_df = pd.concat([pd.read_csv(f,low_memory=False,dtype={'evid':object}) for f in arrival_files])

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

    years = np.arange(2000,2007)
    months = np.arange(1,13)
    for year in years:
        for month in months:
            process_year = year
            process_month = month
        # 	process_year = [2003] ### Assign a list of years to download waveforms for
    # 		geonet_sub_mask = (geonet.origintime.dt.year == process_year) & (geonet.origintime.dt.month == process_month)
            event_sub_mask = np.isin(event_df.datetime.values.astype('datetime64[Y]').
                astype(int)+1970,process_year) & np.isin(event_df.datetime.values.
                astype('datetime64[M]').astype(int) % 12 + 1,process_month)
            event_df_sub = event_df[event_sub_mask].reset_index(drop=True)

            event_list = event_df_sub.evid.values 
            
            arr_df_sub = arr_df[arr_df.evid.isin(event_list)].reset_index(drop=True)
            sta_mag_df_sub = sta_mag_df_all[sta_mag_df_all.evid.isin(event_list)].reset_index(drop=True)
            
            event_file = str(process_year)+'_'+str(process_month)+'_events.csv'
            arr_file = str(process_year)+'_'+str(process_month)+'_assocs.csv'
            mag_file = str(process_year)+'_'+str(process_month)+'_magnitudes.csv'
            
            calc_mags(out_directory,event_file,arr_file,mag_file,sta_corr,station_df,event_df_sub,arr_df_sub,sta_mag_df_sub)