import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from scipy.interpolate import griddata
import numpy as np
import pandas as pd
from pyproj import Transformer			# conda install pyproj
from obspy.clients.fdsn import Client as FDSN_Client
import scipy

def do_kdtree(combined_x_y_arrays,points):
    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
    dist, indexes = mytree.query(points)
    return indexes

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
    
def get_corr(row,x_range,y_range,Z_resampled):
    x_idx = find_nearest(x_range,row.x)
    y_idx = find_nearest(y_range,row.y)
    sta_corr = Z_resampled[x_idx,y_idx]
    
    return(sta_corr)

wgs2nztm = Transformer.from_crs(4326, 2193)
nztm2wgs = Transformer.from_crs(2193, 4326)
channel_codes = 'HN?,BN?,HH?,BH?,EH?,SH?'

client_NZ = FDSN_Client("GEONET")
client_IU = FDSN_Client('IRIS')
inventory_NZ = client_NZ.get_stations(channel=channel_codes)
inventory_IU = client_IU.get_stations(network='IU',station='SNZO')
inventory = inventory_NZ+inventory_IU

inv_sta = []
for network in inventory:
    for station in network:
        print(network.code,station.code,station.latitude,station.longitude)
        inv_sta.append([network.code,station.code,station.latitude,station.longitude])
sta_df = pd.DataFrame(inv_sta,columns=['net','sta','lat','lon'])
sta_df.loc[sta_df['lon'] < 0,'lon'] = 360 + sta_df.loc[sta_df['lon'] < 0,'lon']

sta_corr_df = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/site_info/loessresidualsa15buncutch.gmtdata',names=['lon','lat','corr'],header=None,low_memory=False,sep='\s+')
combined_lon_lat = np.dstack([sta_corr_df.lon.values,sta_corr_df.lat.values])[0]

points = np.dstack([sta_df.lon.values,sta_df.lat.values])[0]

corr_idx = do_kdtree(combined_lon_lat,points)
corr_val = sta_corr_df.loc[corr_idx,'corr'].values
sta_df['corr'] = corr_val
sta_df.loc[sta_df['corr'].isnull(),'corr'] = 0

sta_corr_df_orig = pd.read_csv('/Users/jesse/Downloads/bssa-2020252_supplement_tables/Table S3.csv')
sta_corr_df_orig.rename(columns={'Code':'sta','Correction':'corr'},inplace=True)
sta_df = sta_df.join(sta_corr_df_orig.set_index('sta')['corr'],on='sta',rsuffix='_orig')

compare_corr = sta_df.loc[~sta_df.corr_orig.isnull(),['sta','corr','corr_orig']]
compare_corr['corr_diff'] = compare_corr['corr'] - compare_corr['corr_orig']

compare_corr.plot.scatter('sta','corr_diff')
plt.xticks(rotation=90)
plt.show()

sta_df.loc[~sta_df.corr_orig.isnull(),'corr'] = sta_df.loc[~sta_df.corr_orig.isnull(),'corr_orig']
sta_df.drop(columns='corr_orig',inplace=True)
sta_df.to_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/sta_corr_extracted_rhoades.csv',index=False)

# Old (better?) version extrapolated for 1 km spacing

inv_sta = []
for network in inventory:
    for station in network:
        print(network.code,station.code,station.latitude,station.longitude)
        inv_sta.append([network.code,station.code,station.latitude,station.longitude])
sta_df = pd.DataFrame(inv_sta,columns=['net','sta','lat','lon'])
sta_df.loc[sta_df['lon'] < 0,'lon'] = 360 + sta_df.loc[sta_df['lon'] < 0,'lon']
sta_df['x'],sta_df['y'] = wgs2nztm.transform(sta_df.lat,sta_df.lon)
sta_df['x'] = sta_df.x/1000
sta_df['y'] = sta_df.y/1000


sta_corr_df = pd.read_csv('/Users/jesse/Downloads/bssa-2020252_supplement_tables/Table S3.csv')
sta_corr_df['x'],sta_corr_df['y'] = wgs2nztm.transform(sta_corr_df.Lat,sta_corr_df.Lon)
sta_corr_df['x'] = sta_corr_df.x/1000
sta_corr_df['y'] = sta_corr_df.y/1000
sta_corr_df.loc[sta_corr_df['Lon'] < 0,'Lon'] = 360 + sta_corr_df.loc[sta_corr_df['Lon'] < 0,'Lon']

x = sta_corr_df.x.values
y = sta_corr_df.y.values
Z = sta_corr_df.Correction.values

spacing = 1 # Change this value for lower or higher resolution spacing, currently set to 1 km

extent = (min(x), max(x), min(y), max(y))
xs,ys = np.mgrid[extent[0]:extent[1]:spacing, extent[2]:extent[3]:spacing] #2D x,y
x_range,y_range = np.arange(extent[0],extent[1],spacing),np.arange(extent[2],extent[3],spacing)
Z_resampled = griddata((x, y), Z, (xs, ys),method='linear',fill_value=0) #2D z

imshow(Z_resampled, interpolation="gaussian", extent=extent, origin="lower")
plt.show()

sta_df['corr'] = sta_df.apply(lambda x: get_corr(x,x_range,y_range,Z_resampled),axis=1)
sta_df.loc[sta_df['corr'].isnull(),'corr'] = 0

sta_df = sta_df.merge(sta_corr_df[['Code','Correction']],left_on='sta',right_on='Code',suffixes=('','_orig'),how='left')

compare_corr = sta_df.loc[~sta_df.Correction.isnull(),['sta','corr','Correction']]
compare_corr['corr_diff'] = compare_corr['corr'] - compare_corr['Correction']

compare_corr.plot.scatter('sta','corr_diff')
plt.xticks(rotation=90)
plt.show()

sta_df.loc[~sta_df.Correction.isnull(),'corr'] = sta_df.loc[~sta_df.Correction.isnull(),'Correction']
sta_df.drop(['Correction','Code'],inplace=True,axis=1)

sta_df.to_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/sta_corr_new.csv',index=False)