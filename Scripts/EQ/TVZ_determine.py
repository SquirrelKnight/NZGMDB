import pandas as pd
from pyproj import Transformer			# conda install pyproj
import obspy as op
from pandarallel import pandarallel		# conda install -c bjrn pandarallel
from obspy.clients.fdsn import Client as FDSN_Client
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon, LineString
import numpy as np
import matplotlib.pyplot as plt
import pygmt
from IPython.display import display, Image

# Convert from WGS84 to NZGD2000
wgs2nztm = Transformer.from_crs(4326, 2193)
nztm2wgs = Transformer.from_crs(2193, 4326)

directory = '/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/'
sta_cat = pd.read_csv(directory+'site_table_response.csv',low_memory=False)

def TVZ_calc(cat,taupo_polygon):
	from shapely.geometry import Point
	from shapely.geometry.polygon import Polygon
	import numpy as np

	# Taupo VZ polygon acquired from https://www.geonet.org.nz/data/supplementary/earthquake_location_grope
	for index,row in cat.iterrows():
		sta_id = row.sta
		lat = row.lat
		lon = row.lon
		if lon < 0:
			lon = lon+360
	
		point = Point(lat,lon)

		if taupo_polygon.contains(point):
			cat.loc[index,'TVZ'] = 1
		else:
			cat.loc[index,'TVZ'] = 0
	
	return cat

# df_site = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/IM_catalogue/Tables/site_table.csv',low_memory=False)
def TVZ_path_calc(df,taupo_polygon,df_eq,tect_domain_points,wgs2nztm):
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon, LineString
    import numpy as np

    # Taupo VZ polygon acquired from https://www.geonet.org.nz/data/supplementary/earthquake_location_grope

#     for index,row in df[0:10].iterrows():
    event_id = df.evid
    sta_id = df.sta
    event = df_eq[df_eq.evid == event_id].iloc[0]
    ev_lat = event.lat
    ev_lon = event.lon
    ev_depth = event.depth
    reloc = event.reloc
    ev_transform = wgs2nztm.transform(ev_lat,ev_lon)
#         if ev_lon < 0:
#             ev_lon = ev_lon+360
    network = inventory.select(station = sta_id)[0].code
    station = inventory.select(station = sta_id)[0][0]
    sta_lat = station.latitude
    sta_lon = station.longitude
    sta_elev = station.elevation
    sta_transform = wgs2nztm.transform(sta_lat,sta_lon)
#         if sta_lon < 0:
#             sta_lon = sta_lon+360
    dist, az, b_az = op.geodetics.gps2dist_azimuth(ev_lat, ev_lon, sta_lat, sta_lon)
    r_epi = dist/1000
    r_hyp = (r_epi ** 2 + (ev_depth + sta_elev/1000) ** 2) ** 0.5

    line = LineString([[ev_transform[0],ev_transform[1]],[sta_transform[0],sta_transform[1]]])
    
    tvz_length = 0

    if line.intersection(taupo_polygon):
        line_points = line.intersection(taupo_polygon)
        tvz_length = line_points.length / 1000 / r_epi
    
    # Output evid, net, sta, r_epi, r_hyp, az, b_az, reloc, rrup, and rjb. toa has
    # been omitted. This is better left in the phase arrival table.
    return pd.Series([event_id,network,sta_id,r_epi,r_hyp,tvz_length,az,b_az,reloc,r_epi,r_hyp])


def TVZ_path_plot(df,taupo_polygon,df_eq,tect_domain_points,wgs2nztm):
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon, LineString
    import numpy as np

    # Taupo VZ polygon acquired from https://www.geonet.org.nz/data/supplementary/earthquake_location_grope

#     for index,row in df[0:10].iterrows():
    event_id = df.evid
    sta_id = df.sta
    event = df_eq[df_eq.evid == event_id].iloc[0]
    ev_lat = event.lat
    ev_lon = event.lon
    ev_depth = event.depth
    reloc = event.reloc
    ev_transform = wgs2nztm.transform(ev_lat,ev_lon)
#         if ev_lon < 0:
#             ev_lon = ev_lon+360
    network = inventory.select(station = sta_id)[0].code
    station = inventory.select(station = sta_id)[0][0]
    sta_lat = station.latitude
    sta_lon = station.longitude
    sta_elev = station.elevation
    sta_transform = wgs2nztm.transform(sta_lat,sta_lon)
#         if sta_lon < 0:
#             sta_lon = sta_lon+360
    dist, az, b_az = op.geodetics.gps2dist_azimuth(ev_lat, ev_lon, sta_lat, sta_lon)
    r_epi = dist/1000
    r_hyp = (r_epi ** 2 + (ev_depth + sta_elev/1000) ** 2) ** 0.5

    line = LineString([[ev_transform[0],ev_transform[1]],[sta_transform[0],sta_transform[1]]])
    
    tvz_length = 0

    if line.intersection(taupo_polygon):
        line_points = line.intersection(taupo_polygon)
        tvz_length = line_points.length / 1000 / r_epi
        tvz_path_lat,tvz_path_lon = nztm2wgs.transform(line_points.xy[0],line_points.xy[1])
        tvz_path_lat = np.array(tvz_path_lat)
        tvz_path_lon = np.array(tvz_path_lon)
        tvz_path_lon[tvz_path_lon < 0] = tvz_path_lon[tvz_path_lon < 0] + 360

    shape_lat,shape_lon = nztm2wgs.transform(taupo_polygon.exterior.xy[0],taupo_polygon.exterior.xy[1])
    shape_lat = np.array(shape_lat)
    shape_lon = np.array(shape_lon)
    shape_lon[shape_lon < 0] = shape_lon[shape_lon < 0] + 360
    
    path_lat,path_lon = nztm2wgs.transform(line.xy[0],line.xy[1])
    path_lat = np.array(path_lat)
    path_lon = np.array(path_lon)
    path_lon[path_lon < 0] = path_lon[path_lon < 0] + 360
       
    region = [
    path_lon.min() - 1,
    path_lon.max() + 1,
    path_lat.min() - 1,
    path_lat.max() + 1
    ]
    
    fig = pygmt.Figure()
    fig.basemap(region=region, projection="M6i", frame=True)
    fig.coast(land="white", water="skyblue")
        
    fig.plot(shape_lon,shape_lat,color='orange')
    fig.plot(x=path_lon,y=path_lat,pen='1p,blue')
    if line.intersection(taupo_polygon):
        fig.plot(x=tvz_path_lon,y=tvz_path_lat,pen='1p,red')
    fig.plot(ev_lon,ev_lat,color='white',style='c0.4c',pen='1p')
    fig.text(x=ev_lon,y=ev_lat,text=event_id,offset='0.5c')
    fig.plot(sta_lon,sta_lat,color='pink',style='t0.4c',pen='1p')
    fig.text(x=sta_lon,y=sta_lat,text=sta_id,offset='0.5c')
    fig.show(method="external")


df_im = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/IM_catalogue/Tables/ground_motion_im_table_rotd50_flat.csv',low_memory=False)
df_im_sub = df_im[df_im.duplicated(subset=['sta','evid']) == False][['evid','sta']]
df_arr = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/IM_catalogue/Tables/phase_arrival_table.csv',low_memory=False)
df_arr_sub = df_arr[df_arr.duplicated(subset=['sta','evid']) == False][['evid','sta']]
df_merged = pd.concat([df_im_sub,df_arr_sub],axis=0,ignore_index=True)
df_merged = df_merged[df_merged.duplicated() == False]

df_eq = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/IM_catalogue/Tables/earthquake_source_table.csv',low_memory=False)

client_NZ = FDSN_Client("GEONET")
client_IU = FDSN_Client('IRIS')
inventory_NZ = client_NZ.get_stations()
inventory_IU = client_IU.get_stations(network='IU',station='SNZO,AFI,CTAO,RAO,FUNA,HNR,PMG')
inventory_AU = client_IU.get_stations(network='AU')
inventory = inventory_NZ+inventory_IU+inventory_AU

tect_domain_points = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/tectonic domains/tectonic_domain_polygon_points.csv',low_memory=False)
tvz_points = tect_domain_points[tect_domain_points.domain_no == 4][['latitude','longitude']]
taupo_transform = np.dstack(np.array(wgs2nztm.transform(tvz_points.latitude,tvz_points.longitude)))[0]
taupo_polygon = Polygon(taupo_transform)

# x,y = taupo_polygon.exterior.xy
# plt.plot(x,y)
# plt.show()

# Calculate how much of the propagation path is in the TVZ
prop_df = pd.DataFrame(columns=['evid','net','sta','r_epi','r_hyp','r_tvz','az','b_az','reloc','r_epi','r_hyp'])

pandarallel.initialize(nb_workers=8) # Customize the number of parallel workers
prop_df[['evid','net','sta','r_epi','r_hyp','r_tvz','az','b_az','reloc','r_epi','r_hyp']] = df_merged.parallel_apply(lambda x: TVZ_path_calc(x,taupo_polygon,df_eq,tect_domain_points,wgs2nztm),axis=1)

# Plot desired TVZ path
TVZ_path_plot(df_im.iloc[4],taupo_polygon,df_eq,tect_domain_points,wgs2nztm)