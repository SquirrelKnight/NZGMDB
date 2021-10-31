import pandas as pd
import os
import glob
import pygmt
import matplotlib.pyplot as plt
from matplotlib import rc
from obspy.geodetics import kilometers2degrees
import numpy as np
rc("pdf",fonttype=42)

directory = '/Volumes/SeaJade 2 Backup/NZ/EQTransformer/'

# event_df = pd.read_csv(directory+'inputs/orig_events/earthquake_source_table_relocated.csv',low_memory=False)
finn_arr_df = pd.concat([pd.read_csv(f,low_memory=False,dtype={'evid':object}) for f in glob.glob(directory+'output/finn_arrivals_4/*_arrivals.csv')])
arr_df = pd.concat([pd.read_csv(f,low_memory=False,dtype={'evid':object}) for f in glob.glob(directory+'output/tvz_arrivals/*_arrivals.csv')])
revised_df = pd.concat([pd.read_csv(f,low_memory=False,dtype={'evid':object}) for f in glob.glob(directory+'output/mag_out/events/*_events.csv')])
revised_df['evid'] = revised_df.evid.astype('str')
# revised_df = revised_df.merge(event_df[['evid','lat','lon','depth']],on='evid',suffixes = (None,'_orig'))

# Filter for removing events with nstas of 2 or less
# revised_df = revised_df[revised_df.nsta > 2].reset_index(drop=True)

# Set longitudes to be all positive
revised_df.loc[revised_df.lon < 0, 'lon'] = 360 + revised_df.lon[revised_df.lon < 0]
# revised_df = revised_df[(revised_df.lon < 190) & (revised_df.lon >155)]
# revised_df = revised_df[(revised_df.lat < -15)]

# revised_df.loc[revised_df.lon_orig < 0, 'lon_orig'] = 360 + revised_df.lon_orig[revised_df.lon_orig < 0]
# # revised_df = revised_df[(revised_df.lon_orig < 190) & (revised_df.lon_orig >155)]
# # revised_df = revised_df[(revised_df.lat_orig < -15)]
# # 
# revised_df = revised_df[revised_df.depth < 400]
# 
# depth_diff = revised_df.depth - revised_df.depth_orig
# depth_diff = (revised_df.depth / revised_df.depth.max()) - (revised_df.depth_orig / revised_df.depth_orig.max())
# lat_diff = revised_df.lat - revised_df.lat_orig
# lon_diff = revised_df.lon - revised_df.lon_orig
# 
# f, axs = plt.subplots(1, 3,figsize=(16,8),sharey=True)
# axs[0].hist(depth_diff,bins='auto',edgecolor='k')
# axs[0].set_xlabel('Normalized Depth difference')
# axs[1].hist(lat_diff[abs(lat_diff) <= 2],bins='auto',edgecolor='k')
# axs[1].set_xlabel('Latitude difference (°)')
# axs[2].hist(lon_diff[abs(lon_diff) <= 2],bins='auto',edgecolor='k')
# axs[2].set_xlabel('Longitude difference (°)')
# for ax in axs.flat:
#     ax.set(ylabel='No. of earthquakes')
# # Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()
# plt.show()

from obspy.geodetics import degrees2kilometers,locations2degrees
for idx,event in revised_df.iterrows():
    deg_dist = locations2degrees(event.lat,event.lon,event.y_c,event.x_c)
    dist = degrees2kilometers(deg_dist)
    if dist >= 50:
        print(event.evid,dist,event.major,event.minor,event.ndef,event.mag,event.datetime,event.x,event.y)

error_df = revised_df[['x_c','y_c','theta','major','minor']].reset_index(drop=True)
error_df['major'] = error_df['major'] * 2
error_df['minor'] = error_df['minor'] * 2

# Set the region for the plot to be slightly larger than the revised_df bounds.

region = [
    revised_df.lon.min() - 1,
    revised_df.lon.max() + 1,
    revised_df['lat'].min() - 1,
    revised_df['lat'].max() + 1,
]

# region = [
#     172,
#     180,
#     -42,
#     -36,
# ]

fig = pygmt.Figure()
fig.basemap(region=region, projection="M15c", frame=True)
fig.coast(land="white", water="skyblue")
pygmt.makecpt(cmap="viridis",
    reverse=True,
    series=[revised_df.depth.min(),revised_df.depth.max()])
#         fig.plot(x=revised_df.lon_orig, 
#             y=revised_df.lat_orig,
#             style="c0.1c",
#             color='grey',
#             pen="black")
#         for i,line in revised_df.iterrows():
#             fig.plot(
#                     x = [line.lon_orig,line.lon],
#                     y = [line.lat_orig,line.lat],
#                     pen = '2p,yellow')
fig.plot(
    x=revised_df.lon,
    y=revised_df.lat,
    style="cc",
    size=0.05 * 1.5 ** revised_df.mag,
#     style="c0.1c",
    color=revised_df.depth,
    cmap=True,
    pen="black",
    )
# fig.plot(data=error_df.to_numpy(), style="E", pen="0.25p,black")
# fig.text(text=revised_df.evid.tolist(),x=revised_df.x_c.tolist(),y=revised_df.y_c.to_list())
fig.colorbar(frame='af+l"Depth (km)"')
with fig.inset(position="jTL+w3c/6c+o0.2c", margin=0, box=True):
    fig.basemap(region=[0, 3, 0, 6], projection="X3c/6c", frame=True)
    mag_circles = np.array([[0.5, 5, 0.05 * 1.5 ** 1],
                    [0.5225, 4.375, 0.05 * 1.5 ** 2],
                    [0.53375, 3.75, 0.05 * 1.5 ** 3],
                    [0.55063, 3.125, 0.05 * 1.5 ** 4],
                    [0.57594, 2.5, 0.05 * 1.5 ** 5],
                    [0.61391, 1.875, 0.05 * 1.5 ** 6],
                    [0.67086, 1.25, 0.05 * 1.5 ** 7],
                    [0.75629, .625, 0.05 * 1.5 ** 8]])   
    text_array = np.array([[2, 5, '1'],
                    [2, 4.375, '2'],
                    [2, 3.75, '3'],
                    [2, 3.125, '4'],
                    [2, 2.5, '5'],
                    [2, 1.875, '6'],
                    [2, 1.25, '7'],
                    [2, .625, '8']])
    fig.plot(data=mag_circles,style='cc',color='white',pen='black')
    fig.text(text='Magnitude',x=0.5,y=5.5,justify='LB')
    for vals in text_array:
        fig.text(text=str(vals[2]),x=float(vals[0]),y=float(vals[1]))
fig.show(method="external")


df = revised_df[['lon','lat','depth','z_err']].copy()
df.rename(columns={'lon':'x','lat':'y','z_err':'z-error'},inplace=True)
df['z-error'] = kilometers2degrees(df['z-error'])

fig = pygmt.Figure()
with fig.subplot(
    nrows=1,
    ncols=2,
    figsize=('11i','6i'),
    autolabel=True,
#     frame='WSne',
#     title='New Zealand earthquakes by depth',
    sharex = 'b',
    sharey = 'l',
):
#     for i in range(1):
#         for j in range(2):
#             index = i * 2 + j
    with fig.set_panel(panel=[0,0]):
        fig.basemap(region=region, projection="M?", frame=["af", "WSne"])
        fig.coast(land="black", water="skyblue")
        pygmt.makecpt(cmap="viridis",
            reverse=True,
            series=[revised_df.depth.min(),revised_df.depth.max()])
        fig.plot(
#             x=revised_df.lon,
#             y=revised_df.lat,
            data=df.to_numpy(),
            style="c0.15c",
#             color=revised_df.depth,
            cmap=True,
            pen="black",
            error_bar=['+p1p,white']
            )
#         fig.colorbar(frame='af+l"Depth (km)"')
    with fig.set_panel(panel=[0,1]):
        fig.basemap(region=region, projection="M?", frame=["af", "wSne"])
        fig.coast(land="black", water="skyblue")
        pygmt.makecpt(cmap="viridis",
            reverse=True,
            series=[revised_df.depth.min(),revised_df.depth.max()])
        fig.plot(x=revised_df.lon_orig, 
            y=revised_df.lat_orig,
            #          sizes=0.05 * 1.5 ** revised_df.mag,
            style="c0.15c",
            color=revised_df.depth_orig,
            cmap=True,
            pen="black")
        fig.colorbar(position="JBC+o-2.5i/0.5i",frame='af+l"Depth (km)"')
fig.show(method="external")

# Set the region for the plot to be slightly larger than the revised_df bounds.
sub_region = [
    165,
    170,
    -48,
    -44,
]

fig = pygmt.Figure()
with fig.subplot(
    nrows=1,
    ncols=2,
    figsize=('11i','7i'),
    autolabel=True,
    sharex = 'b',
    sharey = 'l',
):
    with fig.set_panel(panel=[0,0]):
        fig.basemap(region=sub_region, projection="M?", frame=["af", "WSne"])
        fig.coast(land="black", water="skyblue")
        pygmt.makecpt(cmap="viridis",
            reverse=True,
            series=[revised_df.depth.min(),revised_df.depth.max()])
        fig.plot(data=error_df.to_numpy(), style="E", pen="1p,white")
        fig.plot(
            x=revised_df.lon,
            y=revised_df.lat,
#             data=df.to_numpy(),
            #          sizes=0.05 * 1.5 ** revised_df.mag,
            style="c0.15c",
            color=revised_df.depth,
            cmap=True,
            pen="black",
            )
#         fig.colorbar(frame='af+l"Depth (km)"')
    with fig.set_panel(panel=[0,1]):
        fig.basemap(region=sub_region, projection="M?", frame=["af", "wSne"])
        fig.coast(land="black", water="skyblue")
        pygmt.makecpt(cmap="viridis",
            reverse=True,
            series=[revised_df.depth.min(),revised_df.depth.max()])
        fig.plot(x=revised_df.lon_orig, 
            y=revised_df.lat_orig,
            #          sizes=0.05 * 1.5 ** revised_df.mag,
            style="c0.15c",
            color=revised_df.depth_orig,
            cmap=True,
            pen="black")
        fig.colorbar(position="JBC+o-2.5i/0.5i",frame='af+l"Depth (km)"')
fig.show(method="external")
