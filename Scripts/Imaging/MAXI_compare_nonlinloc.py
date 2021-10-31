import pandas as pd
import os
import glob
import pygmt
import matplotlib.pyplot as plt
from matplotlib import rc
from obspy.geodetics import kilometers2degrees
rc("pdf",fonttype=42)

directory = '/Volumes/SeaJade 2 Backup/NZ/EQTransformer/'

event_df = pd.read_csv('/Users/jesse/Downloads/NLL_merged_2000-01-01T000000.000000Z-2001-01-01T000000.000000Z.csv',low_memory=False)
event_df.drop(columns='Unnamed: 0',inplace=True)
event_df['evid'] = event_df.event_id.str.split('/',expand=True)[1]
event_df.rename(columns={'latitude':'lat','longitude':'lon'},inplace=True)

revised_df = pd.concat([pd.read_csv(f,low_memory=False) for f in glob.glob(directory+'output/catalog_all/*_origins.csv')])
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


# Set the region for the plot to be slightly larger than the revised_df bounds.
region = [
    revised_df.lon.min() - 1,
    revised_df.lon.max() + 1,
    revised_df['lat'].min() - 1,
    revised_df['lat'].max() + 1,
]

error_df = revised_df[['lon','lat','theta','x_err','y_err']].reset_index(drop=True)
# error_df = revised_df[['x_c','y_c','theta','x_err','y_err']].reset_index(drop=True)
# error_df['x_err'] = kilometers2degrees(error_df['x_err'] * 10)
# error_df['y_err'] = kilometers2degrees(error_df['y_err'] * 10)
error_df.loc[error_df.x_err <= error_df.y_err,'major'] = error_df.loc[error_df.x_err <= error_df.y_err,'y_err']
error_df.loc[error_df.x_err > error_df.y_err,'major'] = error_df.loc[error_df.x_err > error_df.y_err,'x_err']
error_df.loc[error_df.x_err >= error_df.y_err,'minor'] = error_df.loc[error_df.x_err >= error_df.y_err,'y_err']
error_df.loc[error_df.x_err < error_df.y_err,'minor'] = error_df.loc[error_df.x_err < error_df.y_err,'x_err']
error_df.drop(columns=['x_err','y_err'],inplace=True)

fig = pygmt.Figure()
with fig.subplot(
    nrows=1,
    ncols=1,
    figsize=('11i','7i'),
    autolabel=True,
    sharex = 'b',
    sharey = 'l',
):
    with fig.set_panel(panel=[0,0]):
        fig.basemap(region=region, projection="M?", frame=["af", "WSne"])
        fig.coast(land="black", water="skyblue")
        pygmt.makecpt(cmap="viridis",
            reverse=True,
            series=[revised_df.depth.min(),revised_df.depth.max()])
        fig.plot(x=revised_df.lon_orig, 
            y=revised_df.lat_orig,
            style="c0.1c",
            color='grey',
            pen="black")
#         for i,line in revised_df.iterrows():
#             fig.plot(
#                     x = [line.lon_orig,line.lon],
#                     y = [line.lat_orig,line.lat],
#                     pen = '2p,yellow')
        fig.plot(
            x=revised_df.lon,
            y=revised_df.lat,
            style="c0.1c",
            color=revised_df.depth,
            cmap=True,
            pen="black",
            )
#         fig.plot(data=error_df.to_numpy(), style="E", pen="0.25p,white")
        fig.colorbar(frame='af+l"Depth (km)"')
#     with fig.set_panel(panel=[0,1]):
#         fig.basemap(region=region, projection="M?", frame=["af", "wSne"])
#         fig.coast(land="black", water="skyblue")
#         pygmt.makecpt(cmap="viridis",
#             reverse=True,
#             series=[revised_df.depth.min(),revised_df.depth.max()])
#         fig.plot(x=revised_df.lon_orig, 
#             y=revised_df.lat_orig,
#             #          sizes=0.05 * 1.5 ** revised_df.mag,
#             style="c0.15c",
#             color=revised_df.depth_orig,
#             cmap=True,
#             pen="black")
#         fig.colorbar(position="JBC+o-2.5i/0.5i",frame='af+l"Depth (km)"')
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
