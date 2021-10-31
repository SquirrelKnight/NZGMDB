import pandas as pd
import os
import glob
import pygmt
import matplotlib.pyplot as plt
from matplotlib import rc
from obspy.geodetics import kilometers2degrees
rc("pdf",fonttype=42)

def sub_analysis(bad_list,cutoff,glt):
    if glt == 'greater':
        sub_bad = bad_list[bad_list[:,6] >= cutoff]
    if glt == 'lesser':
        sub_bad = bad_list[bad_list[:,6] <= cutoff]
    dist_diff = (np.sqrt(sub_bad[:,0] ** 2 + sub_bad[:,1] ** 2 + sub_bad[:,2] ** 2))
    mean_dist = np.mean(dist_diff)
    std_dist = np.std(dist_diff)
    print('Dist mean and std.',mean_dist,std_dist,'Total events:',len(sub_bad), 'Median ndef: ',np.median(sub_bad[:,6]))

directory = '/Volumes/SeaJade 2 Backup/NZ/EQTransformer/'

event_df = pd.read_csv(directory+'inputs/orig_events/synthetic_events.csv',low_memory=False)
# event_df['depth'] = event_df['z']
event_df['evid'] = event_df['evid'].astype('str')

revised_df = pd.concat([pd.read_csv(f,low_memory=False) for f in glob.glob(directory+'output/catalog_syn_02/*_origins.csv')])
revised_df['evid'] = revised_df.evid.astype('str')
revised_df = revised_df.merge(event_df[['evid','lat','lon','depth']],on='evid',suffixes = (None,'_orig'))

# Filter for removing events with nstas of 2 or less
revised_df = revised_df[revised_df.nsta > 2].reset_index(drop=True)

# Set longitudes to be all positive
revised_df.loc[revised_df.lon < 0, 'lon'] = 360 + revised_df.lon[revised_df.lon < 0]
# revised_df = revised_df[(revised_df.lon < 190) & (revised_df.lon >155)]
# revised_df = revised_df[(revised_df.lat < -15)]

revised_df.loc[revised_df.lon_orig < 0, 'lon_orig'] = 360 + revised_df.lon_orig[revised_df.lon_orig < 0]
# revised_df = revised_df[(revised_df.lon_orig < 190) & (revised_df.lon_orig >155)]
# revised_df = revised_df[(revised_df.lat_orig < -15)]
# 
# revised_df = revised_df[revised_df.depth < 100]

depth_diff = revised_df.depth - revised_df.depth_orig
depth_diff = (revised_df.depth / revised_df.depth.max()) - (revised_df.depth_orig / revised_df.depth_orig.max())
lat_diff = revised_df.lat - revised_df.lat_orig
lon_diff = revised_df.lon - revised_df.lon_orig

f, axs = plt.subplots(1, 3,figsize=(16,8),sharey=True)
axs[0].hist(depth_diff,bins='auto',edgecolor='k')
axs[0].set_xlabel('Normalized Depth difference')
axs[1].hist(lat_diff[abs(lat_diff) <= 2],bins='auto',edgecolor='k')
axs[1].set_xlabel('Latitude difference (°)')
axs[2].hist(lon_diff[abs(lon_diff) <= 2],bins='auto',edgecolor='k')
axs[2].set_xlabel('Longitude difference (°)')
for ax in axs.flat:
    ax.set(ylabel='No. of earthquakes')
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
plt.show()

error_df = revised_df[['x_c','y_c','theta','major','minor']].reset_index(drop=True)

import numpy as np
import shapely.geometry as geom
from shapely import affinity
from pyproj import Transformer			# conda install pyproj

revised_df['major'] = error_df['major']
revised_df['minor'] = error_df['minor']

wgs2nztm = Transformer.from_crs(4326, 2193, always_xy=True)

good = 0
bad = 0
bad_list = []
good_list = []
for idx,event in revised_df.iterrows():
    n = 360
    r_x,r_y = wgs2nztm.transform(event.x_c,event.y_c)
    o_x,o_y = wgs2nztm.transform(event.lon_orig,event.lat_orig)
    x_diff = (r_x - o_x) / 1000
    y_diff = (r_y - o_y) / 1000
    z_diff = (event.z_c - event.depth_orig)
    a = event.major
    b = event.minor
    angle = event.theta
    theta = np.linspace(0, np.pi*2, n)
    r = a * b  / np.sqrt((b * np.cos(theta))**2 + (a * np.sin(theta))**2)
    exy = np.stack([r * np.cos(theta), r * np.sin(theta)], 1)

    ellipse = affinity.rotate(geom.Polygon(exy), angle, 'center')
    ex, ey = ellipse.exterior.xy
    
    p = [x_diff,y_diff]
    if ellipse.contains(geom.Point(p)):
        print(event.evid + ' good')
        good_list.append([x_diff,y_diff,z_diff,event.lon_orig,event.lat_orig,event.depth,event.ndef])
        good += 1
    else:
        print(x_diff,y_diff,z_diff,'bad')
        bad_list.append([x_diff,y_diff,z_diff,event.lon_orig,event.lat_orig,event.depth,event.ndef])
        bad += 1

#         plt.plot(ex, ey, lw = 1, color='k')
#         plt.scatter(x_diff,y_diff)
#         plt.show()

        
        
bad_list = np.array(bad_list)
dist_diff = (np.sqrt(bad_list[:,0] ** 2 + bad_list[:,1] ** 2 + bad_list[:,2] ** 2))
mean_dist = np.mean(dist_diff)
std_dist = np.std(dist_diff)
print('Bad mean and std.',mean_dist,std_dist,'Total events:',len(bad_list), 'Mean ndef: ',np.mean(bad_list[:,6]))
f, axs = plt.subplots(1, 4,figsize=(16,8),sharey=True)
axs[0].hist(bad_list[:,5],bins='auto',edgecolor='k')
axs[0].set_xlabel('Depth distribution (km)')
axs[1].hist(bad_list[:,4],bins='auto',edgecolor='k')
axs[1].set_xlabel('Latitude distribution (°)')
axs[2].hist(bad_list[:,3],bins='auto',edgecolor='k')
axs[2].set_xlabel('Longitude distribution (°)')
axs[3].hist(bad_list[:,6],bins='auto',edgecolor='k')
axs[3].set_xlabel('Ndef')
for ax in axs.flat:
    ax.set(ylabel='No. of earthquakes')
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
plt.show()

good_list = np.array(good_list)
dist_diff = (np.sqrt(good_list[:,0] ** 2 + good_list[:,1] ** 2 + good_list[:,2] ** 2))
mean_dist = np.mean(dist_diff)
std_dist = np.std(dist_diff)
print('Good mean and std.',mean_dist,std_dist,'Total events:',len(good_list), 'Mean ndef: ',np.mean(good_list[:,6]))
f, axs = plt.subplots(1, 4,figsize=(16,8),sharey=True)
axs[0].hist(good_list[:,5],bins='auto',edgecolor='k')
axs[0].set_xlabel('Depth distribution (km)')
axs[1].hist(good_list[:,4],bins='auto',edgecolor='k')
axs[1].set_xlabel('Latitude distribution (°)')
axs[2].hist(good_list[:,3],bins='auto',edgecolor='k')
axs[2].set_xlabel('Longitude distribution (°)')
axs[3].hist(good_list[:,6],bins='auto',edgecolor='k')
axs[3].set_xlabel('Ndef')
for ax in axs.flat:
    ax.set(ylabel='No. of earthquakes')
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
plt.show()

######## Size is dependent on ndef #########
def plot_eqs(revised_df,x,y,z,ndef):
    # Set the region for the plot to be slightly larger than the revised_df bounds.
    region = [
        revised_df.lon.min() - 1,
        revised_df.lon.max() + 1,
        revised_df['lat'].min() - 1,
        revised_df['lat'].max() + 1,
    ]

    fig = pygmt.Figure()
    fig.basemap(region=region, projection="M15c", frame=["af", "WSne"])
    fig.coast(land="white", water="skyblue")
    pygmt.makecpt(cmap="viridis",
        reverse=True,
        series=[revised_df.depth.min(),revised_df.depth.max()])
#     fig.plot(x=x, 
#         y=y,
#         style="c0.1c",
#         color='grey',
#         pen="black")
    #         for i,line in revised_df.iterrows():
    #             fig.plot(
    #                     x = [line.lon_orig,line.lon],
    #                     y = [line.lat_orig,line.lat],
    #                     pen = '2p,yellow')
    fig.plot(
        x=x,
        y=y,
        style="cc",
        color=z,
        cmap=True,
        pen="black",
        size=np.log10(ndef) / 10
        )
#     fig.plot(data=error_df.to_numpy(), style="E", pen="0.25p,black")
    fig.colorbar(frame='af+l"Depth (km)"')
    fig.show(method="external")
    
plot_eqs(revised_df,bad_list[:,3],bad_list[:,4],bad_list[:,5],bad_list[:,6])
plot_eqs(revised_df,good_list[:,3],good_list[:,4],good_list[:,5],good_list[:,6])