import pykonal
import pandas as pd
import numpy as np
from obspy.clients.fdsn import Client as FDSN_Client
import glob
import h5py
from obspy.geodetics import locations2degrees, degrees2kilometers
import datetime
import os
from obspy import UTCDateTime

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

def find_nearby_stations(event,station_df):
    deg_dist = locations2degrees(event.lat,event.lon,station_df.lat,station_df.lon)
    dist = degrees2kilometers(deg_dist)
    sta_sub = station_df[dist <= 150]
    
    return(sta_sub)


def compute_synthetic_arrivals(df, tt_dir, station_df):
    # Make system to randomly pick only some of the arrivals from the data set. Should have
    # a minimum of 6 arrivals per event.
    
    arr_df_all = pd.DataFrame()
    for idx, event in df.iterrows():
        sta_sub = find_nearby_stations(event,station_df).reset_index(drop=True)
        if len(sta_sub) > 0:
            print('Computing arrival times for event '+str(event.evid))
            arr_list = []
            for sta_idx, sta in sta_sub.iterrows():
                min_coords = np.array([np.floor(event.x)-1,np.floor(event.y)-1,np.floor(event.z)-1])
                max_coords = np.array([np.ceil(event.x)+2,np.ceil(event.y)+2,np.ceil(event.z)+2])
                print('... '+str(sta.sta))
                tt_p_file = tt_dir+'/'+sta.sta+'_P.hdf'
                tt_s_file = tt_dir+'/'+sta.sta+'_S.hdf'
                tt_p = pykonal.fields.read_hdf(tt_p_file,min_coords,max_coords)
                tt_s = pykonal.fields.read_hdf(tt_s_file,min_coords,max_coords)
                # Determine p and s travel times
                p_time = tt_p.value(np.array([event.x,event.y,event.z]))
                s_time = tt_s.value(np.array([event.x,event.y,event.z]))
                arr_list.append([sta.net,sta.sta,'p',event['datetime'] + datetime.timedelta(seconds=p_time),event.evid])
                arr_list.append([sta.net,sta.sta,'s',event['datetime'] + datetime.timedelta(seconds=s_time),event.evid])
            if len(arr_list) > 4:
                if len(arr_list) >= 50:
                    select = np.random.randint(4,50)
                else:
                    select = np.random.randint(4,len(arr_list))
                arr_df = pd.DataFrame(arr_list,columns=['net','sta','phase','datetime','evid'])
                arr_df = arr_df.sample(n=select).copy().reset_index(drop=True)
                arr_df_all = arr_df_all.append(arr_df)
            elif len(arr_list == 4):
                arr_df = pd.DataFrame(arr_list,columns=['net','sta','phase','datetime','evid'])
                arr_df_all = arr_df_all.append(arr_df)
        else:
            print('No stations near event '+str(event.evid))

    return arr_df_all

def add_noise(arr_df,noise_sec):
    # Add noise to P and S traveltimes
    arr_df_noise = arr_df.copy()
    arr_df_noise['datetime'] = arr_df_noise['datetime'] + pd.to_timedelta(np.random.uniform(-1*noise_sec,noise_sec,len(arr_df_noise)),'s')
    return arr_df_noise    

def compute_synthetic_events(tt_dir,station_df,event_no):
    orilat = -41.7638  # Origin latitude
    orilon = 172.9037  # Origin longitude
    angle = -140  # Counter-clockwise rotation for restoration of true coordinates
    x_nodes, y_nodes, z_nodes, x_spacing, y_spacing, z_spacing, x_min, y_min, z_min, \
        x_max, y_max, z_max = get_grid_info(tt_dir)
    loc_list = []
    i = 0
    print('Generating synthetic events, must have at least 3 nearby stations:')
    while i < event_no:
#     for i in range(0,100):
        x = round(np.random.uniform(int(x_min)+1,int(x_max)-2),2)
        y = round(np.random.uniform(int(y_min)+1,int(y_max)-2),2)
        z = round(np.random.uniform(int(0)+1,int(z_max)-2),2)
        lat, lon = rotate_back(orilat,orilon,x,y,angle)
        ev_df = pd.DataFrame(columns=['lat','lon'])
        ev_df.loc[0] = lat,lon
        sta_sub = find_nearby_stations(ev_df,station_df)
        if len(sta_sub) >= 3:
            loc_list.append([x,y,z,datetime.datetime.now()])
            print('... '+str(i))
            i += 1
    loc_list = np.array(loc_list)
    xs,ys = [loc_list[:,0],loc_list[:,1]]
    lats,lons = rotate_back(orilat,orilon,xs,ys,angle)
    depths = loc_list[:,2]
    datetimes = loc_list[:,3]
    df = pd.DataFrame(np.array([xs,ys,depths,lats,lons,datetimes]).T,columns=['x','y','z','lat','lon','datetime'])
    df['evid'] = df.index.values + 1
    
    return df

def get_grid_info(tt_dir):
    tt_files = glob.glob(tt_dir+'/*.hdf')
    pykonal.fields.read_hdf(tt_files[0])
    x_nodes, y_nodes, z_nodes = np.array(h5py.File(tt_files[0], 'r')['npts'])
    x_spacing, y_spacing, z_spacing = np.array(h5py.File(tt_files[0], 'r')['node_intervals'])
    x_min, y_min, z_min = np.array(h5py.File(tt_files[0], 'r')['min_coords'])
    x_max, y_max, z_max = [x_min + (x_nodes * x_spacing), y_min + (y_nodes * y_spacing), z_min + (z_nodes * z_spacing)]
    
    return x_nodes, y_nodes, z_nodes, x_spacing, y_spacing, z_spacing, x_min, y_min, z_min, \
        x_max, y_max, z_max

def get_sta_info(tt_dir):
    time_search = UTCDateTime(datetime.datetime.now()) # Only looks for currently active stations
    client_NZ = FDSN_Client("GEONET")
    client_IU = FDSN_Client('IRIS')
    inventory_NZ = client_NZ.get_stations(starttime=time_search,endtime=time_search)
    inventory_IU = client_IU.get_stations(network='IU',station='SNZO',starttime=time_search,endtime=time_search)
    inventory = inventory_NZ+inventory_IU
    station_info = []
    for network in inventory:
        for station in network:
            station_info.append([network.code, station.code, station.latitude, station.longitude, station.elevation])
    # 		station_df = station_df.append({'net':network.code,'sta':station.code,'lat':station.latitude,
    # 			'lon':station.longitude,'elev':station.elevation},True)

    station_df = pd.DataFrame(station_info,columns=['net','sta','lat','lon','elev'])
    station_df = station_df.drop_duplicates().reset_index(drop=True)

    white_list_files = glob.glob(tt_dir+'/*.hdf')
    white_list = np.unique([os.path.basename(file).split('_')[0] for file in white_list_files])
    station_df_good = station_df[station_df.sta.isin(white_list)]
   
    return station_df_good

noise_sec_list = [0.2,0.4,0.6,0.8,1.0]
event_no = 1000
tt_fine_dir = '/Volumes/SeaJade2/Pykonal/tt_fine'

station_df = get_sta_info(tt_fine_dir)
event_df = compute_synthetic_events(tt_fine_dir,station_df,event_no)
arr_df = compute_synthetic_arrivals(event_df, tt_fine_dir, station_df)
for noise_sec in noise_sec_list:
    arr_df_noise = add_noise(arr_df,noise_sec)
    arr_df_noise.to_csv('inputs/phases/synthetic_phases_noise_'+str(noise_sec).replace('.','')+'.csv',index=False)
arr_df.to_csv('inputs/phases/synthetic_phases_no_noise.csv',index=False)
event_df.to_csv('inputs/orig_events/synthetic_events.csv',index=False)

def plot_events(event_df):
    event_df.loc[event_df.lon < 0, 'lon'] = 360 + event_df.lon[revised_df.lon < 0]
    region = [
        event_df.lon.min() - 1,
        event_df.lon.max() + 1,
        event_df['lat'].min() - 1,
        event_df['lat'].max() + 1,
    ]

    fig = pygmt.Figure()
    fig.basemap(region=region, projection="M15c", frame=["af", "WSne"])
    fig.coast(land="white", water="skyblue")
    pygmt.makecpt(cmap="viridis",
        reverse=True,
        series=[revised_df.z.min(),revised_df.z.max()])
    fig.plot(
        x=event_df.lon,
        y=event_df.lat,
        style="c0.1c",
        color=event_df.z,
        cmap=True,
        pen="black",
        )
    fig.colorbar(frame='af+l"Depth (km)"')
    fig.show(method="external")
