import pandas as pd
from scipy import stats
import numpy as np
from pandarallel import pandarallel		# conda install -c bjrn pandarallel
import glob
from obspy.clients.fdsn import Client as FDSN_Client
import obspy as op
import ray

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



# @ray.remote
def fix_mag(df,sta_mag_out):
#     for ev_idx,row in event_df.iterrows():
    row = df.copy()
#     print(row.evid,row.name)
    if row.mag_type[0:2].lower() != 'mw':
        evid = row.evid
    #     print(evid)
#         cat = client_NZ.get_events(eventid=evid)
#         event = cat[0]
        df_sub = sta_mag_out[sta_mag_out.evid.isin([evid])].reset_index(drop=True)
#         df_sub = sta_mag_out.query("evid == "+"'"+str(evid)+"'")
        mag_data = df_sub[df_sub.chan.str.endswith('Z',na=False)]
        if (mag_data.mag_corr.isnull() == False).sum() < 2:
            mag_data = df_sub[df_sub.chan.str.endswith(('E','N','1','2'),na=False)]
            mag_type = 'cMl_H'
            if (mag_data.mag_corr.isnull() == False).sum() < 2:
                mag_data = df_sub
    #             print(evid,event.mag,event.mag_type,event.mag_method,event.mag_unc,event.nmag)
                row_out = row.copy()
#                 row_out = row_out.to_frame().T
                return row_out
    #                 continue
            else:
                mag_type = 'cMl_H'
        else:
            mag_type = 'cMl'
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

        row[['mag','mag_unc','mag_type','mag_method','nmag']] = new_mean, CML_unc, mag_type, 'NZ20', nmag
    row_out = row.copy()
#     row_out = row_out.to_frame().T
    return row_out
#     print(evid,new_mean,CML_unc,mag_type,'NZ20',nmag)
#     
#     if new_mean != mean:
#         print('Revised!')
event_df = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/earthquake_source_table_complete.csv',low_memory=False)
sta_mag_df = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/station_magnitude_table_relocated.csv',low_memory=False)

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
sta_df = pd.DataFrame(station_info,columns=['net','sta','lat','lon','elev'])
sta_df = sta_df.drop_duplicates().reset_index(drop=True)
sta_corr = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/sta_corr_new.csv')

nz20_res = 0.278

years = np.arange(2000,2008)
# months = np.arange(1,13)
# years = np.unique(geonet.origintime.values.astype('datetime64[Y]').astype(int)+1970)
for year in years:
# 	for month in months:
    process_year = year
    print('Processing for year '+str(process_year))
    
    event_sub_mask = event_df.datetime.values.astype('datetime64[Y]').astype(int)+1970 == process_year
    event_sub = event_df[event_sub_mask].reset_index(drop=True)
    sta_mag_df_sub = sta_mag_df[sta_mag_df.evid.isin(event_sub.evid.unique())].reset_index(drop=True)

    pandarallel.initialize(nb_workers=8,progress_bar=True) # Customize the number of parallel workers
    
    results = event_sub.parallel_apply(lambda x: revise_magnitudes(x,sta_mag_df_sub,sta_df,sta_corr),axis=1)
    sta_mag_out = pd.concat([result for result in results]).reset_index(drop=True)
    sta_mag_out.to_csv('sta_mag_fix_'+str(process_year)+'.csv',index=False)
    
    event_out = event_sub.parallel_apply(lambda x: fix_mag(x,sta_mag_out),axis=1)  
    event_out.to_csv('event_mag_fix_'+str(process_year)+'.csv',index=False)
    
events_final = pd.concat([pd.read_csv(f,low_memory=False) for f in glob.glob('event_mag_fix_*.csv')])
sta_mag_final = pd.concat([pd.read_csv(f,low_memory=False) for f in glob.glob('sta_mag_fix_*.csv')])
events_final.to_csv('earthquake_source_table_fixed_mags.csv',index=False)
sta_mag_final.to_csv('station_magnitude_table_fixed_mags.csv',index=False)








years = np.arange(2000,2012)
# months = np.arange(1,13)
# years = np.unique(geonet.origintime.values.astype('datetime64[Y]').astype(int)+1970)
for year in years:
# 	for month in months:
    process_year = year
    print('Processing for year '+str(process_year))
    
    event_sub_mask = event_df.datetime.values.astype('datetime64[Y]').astype(int)+1970 == process_year
    event_sub = event_df[event_sub_mask].reset_index(drop=True)

    pandarallel.initialize(nb_workers=8,progress_bar=True) # Customize the number of parallel workers
    results = event_sub.parallel_apply(lambda x: check_mb(x,client_NZ),axis=1)
    results.to_csv('event_M_fix_'+str(process_year)+'.csv',index=False)
    
events_final = pd.concat([pd.read_csv(f,low_memory=False) for f in glob.glob('event_M_fix_*.csv')])
events_final.to_csv('earthquake_source_table_fixed_M.csv',index=False)
