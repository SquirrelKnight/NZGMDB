import pandas as pd
import pykonal
from pyproj import Transformer
import json
import numpy as np
import math
from pandarallel import pandarallel		# conda install -c bjrn pandarallel
import glob
from math import acos,sqrt,pi
from obspy.geodetics import gps2dist_azimuth


def rotate(orilat, orilon, lats, lons, angle):
	"""
	Rotate a point counterclockwise by a given angle around a given origin.

	The angle should be given in radians.
	"""
	angle = np.radians(angle)
# 	ox = orilon
# 	oy = orilat
	transformer_from_latlon = Transformer.from_crs(4326, 2193) # WSG84 to New Zealand NZDG2000 coordinate transform
	transformer_to_latlon = Transformer.from_crs(2193, 4326)
	
	ox, oy = transformer_from_latlon.transform(orilat,orilon)
	px, py = transformer_from_latlon.transform(lats,lons)

	qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
	qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
	
	x = (qx-ox)/1000
	y = -(qy-oy)/1000
	
# 	stations['z'] = (stations.elevation-dx)/dx
	return x, y
	
def rotate_back(orilat, orilon, xs, ys, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    from pyproj import Transformer
    import math
    import numpy as np

    angle = np.radians(angle)
    # 	ox = orilon
    # 	oy = orilat
    transformer_from_latlon = Transformer.from_crs(4326, 2193)  # WSG84 to New Zealand NZDG2000 coordinate transform
    transformer_to_latlon = Transformer.from_crs(2193, 4326)

    ox, oy = transformer_from_latlon.transform(orilat, orilon)
    px = ox + xs * 1000
    py = oy - ys * 1000
    # 	px, py = transformer_from_latlon.transform(lats,lons)

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    lats, lons = transformer_to_latlon.transform(qx, qy)
    # 	stations['z'] = (stations.elevation-dx)/dx
    return lats, lons

	
def get_path(sta,arr_file):
    # Calculates azimuth and takeoff angle information using 3d traveltime grids from
    # Pykonal
    arr_file_sub = arr_file.copy()[arr_file.sta == sta.sta].reset_index(drop=True)
    lat, lon, el = stations[sta.sta]['coords']
    tts = []
    tts.append(pykonal.fields.read_hdf('/Volumes/SeaJade 2 Backup/NZ/EQTransformer/inputs/tt/'+sta.sta+'_P.hdf'))
    tts.append(pykonal.fields.read_hdf('/Volumes/SeaJade 2 Backup/NZ/EQTransformer/inputs/tt/'+sta.sta+'_S.hdf'))
    min_coords = tts[0].min_coords
    x_spacing, y_spacing, z_spacing = tts[0].node_intervals
#     tts.append(pykonal.fields.read_hdf('/Volumes/SeaJade2/Pykonal/tt_fine/'+sta.sta+'_P.hdf'))
#     tts.append(pykonal.fields.read_hdf('/Volumes/SeaJade2/Pykonal/tt_fine/'+sta.sta+'_S.hdf'))
    sta_x, sta_y = rotate(orilat, orilon, lat, lon, angle)
    sta_z = (-el/1000)
    for i,event in events.iterrows():
        x, y = rotate(orilat, orilon, event.lat, event.lon, angle)
        z = event.depth
        arrivals = arr_file_sub[(arr_file_sub.evid == event.evid) & (arr_file_sub.sta == sta.sta)]
        dist,azimuth,back_azimuth = gps2dist_azimuth(event.lat,event.lon,lat,lon)
        dist = dist / 1000
        if len(arrivals) > 0:
            for j,arrival in arrivals.iterrows():
                for k in range(0,2):
                    arr_sta = arrival.sta
                    if k == 0:
                        tt = tts[0]
                        phase = 'P'
                    else:
                        tt = tts[1]
                        phase = 'S'

                    # Get last two coordinates to calculate takeoff angle and azimuth, get first two to calculate
                    # incident angle and back azimuth
                    ray_path = tt.trace_ray(np.array([x,y,z]))
                    # Rotate ray_path coordinates back to lat,lon. If this is not done,
                    # computing azimuth is inaccurate.
                    ray_lat,ray_lon = rotate_back(orilat,orilon,ray_path[:,0],ray_path[:,1],-angle)
                    x2,x1 = ray_lon[-2::]
                    y2,y1 = ray_lat[-2::]
                    z2,z1 = ray_path[-2:,2]
                    hyp,az,b_az = gps2dist_azimuth(y1,x1,y2,x2)
                    hyp = hyp/1000

                    z_diff = z1-z2

                    rad = np.arctan(z_diff/hyp)
                    toa = np.rad2deg(rad)
                    if z_diff < 0:
                        toa = abs(toa) + 90
                    else:
                        toa = 90 - toa
                       
                    # Also add incident angle and back-azimuth
                    x1,x2 = ray_lon[1:3]
                    y1,y2 = ray_lat[1:3]
                    z1,z2 = ray_path[1:3,2]
                    x_diff = x2-x1
                    y_diff = y2-y1
                    
                    z_diff = z1-z2

                    hyp = gps2dist_azimuth(y1,x1,y2,x2)[0]
                    hyp = hyp/1000

                    ia_rad = np.arctan(z_diff/hyp)
                    ia = np.rad2deg(ia_rad)
                    if z_diff < 0:
                        ia = abs(ia) + 90
                    else:
                        ia = 90 - ia

                    if phase == 'P':
                        arr_file_sub.loc[arrival.name,['p_az','p_toa','p_baz','p_ia','dist']] = az,toa,b_az,ia,dist
                    else:
                        arr_file_sub.loc[arrival.name,['s_az','s_toa','s_baz','s_ia','dist']] = az,toa,b_az,ia,dist
                    print(azimuth,az,toa,b_az,ia,dist,sta.sta,event.evid)
    return arr_file_sub


def pd_to_HASH(arr_final,events,basedir):
    # Converts dataframe to hash output
    Blank = ''
    Hor_unc = 0
    Ver_unc = 0
    mainfile = basedir+'/kaikoura.txt'
    ampfile = basedir+'/kaikoura.amp'
    fo = open(mainfile,'w')
    fa = open(ampfile,'w')
    for i, event in events.iterrows():
        dt = pd.to_datetime(event.datetime)
        year = dt.year
        month = dt.month
        day = dt.day
        hour = dt.month
        minute = dt.minute
        second = dt.second
        if event.lat < 0:
            lat_dir = 'S'
        else:
            lat_dir = 'N'
        if event.lon < 0:
            lon_dir = 'W'
        else:
            lon_dir = 'E'
        lat_deg, lat_min = np.array(str(event.lat).split('.')).astype('int')
        lat_deg = np.abs(lat_deg)
        lat_min = float('0.'+str(lat_min)) * 60
        lon_deg, lon_min = np.array(str(event.lon).split('.')).astype('int')
        lon_deg = np.abs(lon_deg)
        lon_min = float('0.'+str(lon_min)) * 60
        mag = event.mag
#     print(year,month,day,hour,minute,second)
        # Event output (to main file)
        print('%4s%2i%2i%2i%2i%5.2f%2s%1s%5.2f%3s%1s%5.2f%5.2f%49s%5.2f%1s%5.2f%40s%4.2f%6s%16s' % 
            (year,month,day,hour,minute,second,lat_deg,lat_dir,lat_min,lon_deg,
            lon_dir,lon_min,event.depth,Blank,Hor_unc,Blank,Ver_unc,Blank,event.mag,Blank,event.evid), 
            sep="",file=fo)
        arr_sub = arr_final[arr_final.evid == event.evid]
        for j,arrival in arr_sub.iterrows():
            #Writes phase data to the main output file
            if arrival.fm == 'c':
                fm = 'u'
            elif arrival.fm == 'd':
                fm = 'd'
            else:
                fm = []
            if fm:
                print('%-4s%2s%1s%1i%50s%5.1f%3s%3i%10s%3i%1s%3i%1s%3i' % (arrival.sta,
                    Blank,fm,0,Blank,arrival.dist,Blank,arrival.p_toa,Blank,arrival.p_az,Blank,0,Blank,0), sep="",file=fo)
        print('%56s%16s'%(Blank,event.evid), sep="",file=fo)

        # Write event info to Amplitude file
        amp_sub = arr_sub[arr_sub.sp_ratio != -1]
        print('%16s%16i' % (event.evid,len(amp_sub)), sep = "", file=fa)
        if len(amp_sub) > 0:
            for j,arrival in amp_sub.iterrows():
                # Amplitude output
                print('%-4s%1s%3s%1s%2s%17s%10.3f%1s%10.3f%1s%10.3f%1s%10.3f%3s%3i%10s%3i%1s%3i%1s%3i' % 
                (arrival.sta,Blank,arrival.chan,Blank,arrival.net,Blank,arrival.p_noise,
                Blank,arrival.p_amp,Blank,arrival.s_noise,Blank,arrival.s_amp,Blank,arrival.p_toa,
                Blank,arrival.p_az,Blank,0,Blank,0), sep = "",file=fa) #Writes S/P data to amplitude output file
    fo.close()
    fa.close()

orilat = -41.7638  # Origin latitude
orilon = 172.9037  # Origin longitude
angle = 140  # Counter-clockwise rotation for restoration of true coordinates
basedir = '/Users/jesse/bin/HASH_v1.2'

Blank = ''

# arr_file = pd.read_csv('/Volumes/SeaJade 2 Backup/zross_picker/phase/2016p858725.csv',low_memory=False)
arr_dir = '/Volumes/SeaJade 2 Backup/zross_picker/phase'
arr_files = glob.glob(arr_dir+'/*.csv')[0:100]

arr_file = pd.concat([pd.read_csv(arr_file) for arr_file in arr_files])
ev_file = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/earthquake_source_table_complete.csv',low_memory=False)
events = ev_file[ev_file.evid.isin(arr_file.evid)]

sta_list = arr_file[['sta','chan']].copy()
sta_list_chan = sta_list[['sta','chan']].copy().drop_duplicates().reset_index(drop=True)
sta_list = sta_list[['sta']].drop_duplicates().reset_index(drop=True)
tts = []

with open('/Volumes/SeaJade 2 Backup/NZ/Pykonal/station_list.json') as f:
	stations = json.load(f)

pandarallel.initialize(nb_workers=8,progress_bar=False) # Customize the number of parallel workers
arr_final = sta_list.parallel_apply(lambda x: get_path(x,arr_file),axis=1)
arr_final = pd.concat((arr for arr in arr_final)).reset_index(drop=True)

arr_final.to_csv('kaikoura_phase_table.csv',index=False)

pd_to_HASH(arr_final,events,basedir)

# Generate Station file
stationfile = basedir+'/kaikoura.stations'
fs = open(stationfile,'w')
for i,sta in sta_list_chan.sort_values('sta').iterrows():
	station = sta.sta		
	chan = sta.chan
	coords = stations[station]['coords']
	lat = coords[0]
	lon = coords[1]
	el = coords[2]
	net = stations[station]['network']
	print('%-4s%1s%3s%33s%9.5f%1s%10.5f%1s%5i%23s%2s'%(station,Blank,chan,Blank,lat,Blank,
	    lon,Blank,el,Blank,net),sep='',file=fs)
fs.close()

# Generate station polarity reversal file
reversalfile = basedir+'/kaikoura.reverse'
fr = open(reversalfile,'w')
for i,sta in sta_list.sort_values('sta').iterrows():	
	Rev_beg = 0
	Rev_end = 0
	print('%-4s%1s%8i%1s%8i'%(sta.sta,Blank,Rev_beg,Blank,Rev_end),sep='',file=fr)
fr.close()

# Generate station correction file
corrfile = basedir+'/kaikoura.statcor'
fc = open(corrfile,'w')
for i,sta in sta_list_chan.sort_values('sta').iterrows():
    station = sta.sta
    channel = sta.chan
    average = 0 # Shift of mean observed log10(S/P) ratios to match average of theoretical distribution
    print('%-4s%2s%3s%1s%2s%1s%7.4f' % (station,Blank,channel,Blank,'XX',Blank,0), sep = "",file=fc)
fc.close()
