# Scans through waveforms linked to the rotd50 flat file and determines if they are likely
# to have clipping.

import pandas as pd
import obspy as op
from gmprocess.waveform_processing.clipping import clipping_check
import numpy as np
import glob
import os
from pandarallel import pandarallel

from obspy.geodetics.base import gps2dist_azimuth

from gmprocess.waveform_processing.clipping.clipping_ann import clipNet
from gmprocess.waveform_processing.clipping.max_amp import Max_Amp
from gmprocess.waveform_processing.clipping.histogram import Histogram
from gmprocess.waveform_processing.clipping.ping import Ping
from gmprocess.waveform_processing.processing_step import ProcessingStep

M_TO_KM = 1.0 / 1000

def return_clipped(row, threshold):
    gm = row.copy()
    sta_lat = gm.sta_lat
    sta_lon = gm.sta_lon
    event_lat = gm.ev_lat
    event_lon = gm.ev_lon
    event_mag = gm.mag
    dist = gm.r_hyp
    search = search_df[search_df.evid == gm.evid].iloc[0]
    mseed_file = glob.glob(search.mseed_dir+'/*'+gm.sta+'*.mseed')[0]
    st = op.read(mseed_file)

#     dist = (
#         gps2dist_azimuth(
#             lat1=event_lat,
#             lon1=event_lon,
#             lat2=sta_lat,
#             lon2=sta_lon,
#         )[0]
#         * M_TO_KM
#     )

    event_mag = np.clip(event_mag, 3.0, 8.8)
    dist = np.clip(dist, 0.0, 645.0)

    clip_nnet = clipNet()

    max_amp_method = Max_Amp(st, max_amp_thresh=6e6)
    hist_method = Histogram(st)
    ping_method = Ping(st)
    
    inputs = [
        event_mag,
        dist,
        max_amp_method.is_clipped,
        hist_method.is_clipped,
        ping_method.is_clipped,
    ]
    prob_clip = clip_nnet.evaluate(inputs)[0][0]
    gm['clip_prob'] = prob_clip
    if prob_clip >= threshold:
        gm['clipped'] = True
    else:
        gm['clipped'] = False
    return gm


threshold = 0.2
root_dir = '/Volumes/SeaJade 2 Backup/NZ'
search_dirs = ['mseed_4-4.5_preferred','mseed_4.5-5_preferred','mseed_5-6_preferred','mseed_6-10_preferred']
xml_files = []
evids = []
mseed_dirs = []

for search_dir in search_dirs:
    search_dir = root_dir+'/'+search_dir
    xml_files = xml_files + (glob.glob(search_dir+'/**/*.xml',recursive=True))
    
evids = [os.path.basename(xml_file).split('.')[0] for xml_file in xml_files]
mseed_dirs = [os.path.dirname(xml_file)+'/mseed/data' for xml_file in xml_files]

zipped = list(zip(evids,xml_files,mseed_dirs))
search_df = pd.DataFrame(zipped,columns=['evid','xml_file','mseed_dir'])

df = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/IM_catalogue/Tables/ground_motion_im_table_rotd50_flat.csv',low_memory=False)
pandarallel.initialize(nb_workers=8,progress_bar=True)

df_out = df.parallel_apply(lambda x: return_clipped(x, threshold),axis=1)
df_out = df_out[['evid','sta','clip_prob','clipped']]
df_out.to_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/IM_catalogue/Tables/clip_table.csv',index=False)
# df_out.to_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/IM_catalogue/Tables/ground_motion_im_table_rotd50_flat_clipflag.csv',index=False)

# for idx, gm in df.iterrows():
#     print(idx,end='\r')
#     sta_lat = gm.sta_lat
#     sta_lon = gm.sta_lon
#     event_lat = gm.ev_lat
#     event_lon = gm.ev_lon
#     event_mag = gm.mag
#     dist = gm.r_hyp
#     search = search_df[search_df.evid == gm.evid].iloc[0]
#     mseed_file = glob.glob(search.mseed_dir+'/*'+gm.sta+'*.mseed')[0]
#     st = op.read(mseed_file)
# 
# #     dist = (
# #         gps2dist_azimuth(
# #             lat1=event_lat,
# #             lon1=event_lon,
# #             lat2=sta_lat,
# #             lon2=sta_lon,
# #         )[0]
# #         * M_TO_KM
# #     )
# 
#     event_mag = np.clip(event_mag, 3.0, 8.8)
#     dist = np.clip(dist, 0.0, 645.0)
# 
#     clip_nnet = clipNet()
# 
#     max_amp_method = Max_Amp(st, max_amp_thresh=6e6)
#     hist_method = Histogram(st)
#     ping_method = Ping(st)
#     df.loc[idx,'hist_clipped'] = hist_method.is_clipped
#     df.loc[idx,'ping_clipped'] = ping_method.is_clipped
