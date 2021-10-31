# This script generates waterfall plots from mseed data for individual events.
#   Requirements: obspy, numpy, matplotlib, glob, and AfshariStewart_2016_Ds (modified by
#   J. Hutchinson)

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import obspy as op
import glob
from obspy.clients.fdsn import Client as FDSN_Client
import os
from obspy import read_inventory
from obspy.core import AttribDict
from obspy.geodetics import locations2degrees,degrees2kilometers,kilometers2degrees
import numpy as np
from obspy.taup import TauPyModel
import AfshariStewart_2016_Ds as asds
import pandas as pd
from obspy import UTCDateTime

class Site:  # Class of site properties. initialize all attributes to None
    def __init__(self, **kwargs):
        self.name = kwargs.get("name")  # station name
        self.Rrup = kwargs.get("rrup")  # closest distance coseismic rupture (km)
        self.Rjb = kwargs.get(
            "rjb"
        )  # closest horizontal distance coseismic rupture (km)
        self.Rx = kwargs.get(
            "rx", -1.0
        )  # distance measured perpendicular to fault strike from surface projection of
        #                       # updip edge of the fault rupture (+ve in downdip dir) (km)
        self.Ry = kwargs.get(
            "ry", -1.0
        )  # horizontal distance off the end of the rupture measured parallel
        self.Rtvz = kwargs.get(
            "rtvz"
        )  # source-to-site distance in the Taupo volcanic zone (TVZ) (km)
        self.vs30measured = kwargs.get(
            "vs30measured", False
        )  # yes =True (i.e. from Vs tests); no=False (i.e. estimated from geology)
        self.vs30 = kwargs.get("vs30")  # shear wave velocity at 30m depth (m/s)
        self.z1p0 = kwargs.get(
            "z1p0"
        )  # depth (km) to the 1.0km/s shear wave velocity horizon (optional, uses default relationship otherwise)
        self.z1p5 = kwargs.get("z1p5")  # (km)
        self.z2p5 = kwargs.get("z2p5")  # (km)
        self.siteclass = kwargs.get("siteclass")
        self.orientation = kwargs.get("orientation", "average")
        self.backarc = kwargs.get(
            "backarc", False
        )  # forearc/unknown = False, backarc = True
        self.fpeak = kwargs.get("fpeak", 0)


class Fault:  # Class of fault properties. initialize all attributes to None
    def __init__(self, **kwargs):
        self.dip = kwargs.get("dip")  # dip angle (degrees)
        self.faultstyle = kwargs.get(
            "faultstyle"
        )  # Faultstyle (options described in enum below)
        self.hdepth = kwargs.get("hdepth")  # hypocentre depth
        self.Mw = kwargs.get("Mw")  # moment tensor magnitude
        self.rake = kwargs.get("rake")  # rake angle (degrees)
        self.tect_type = kwargs.get(
            "tect_type"
        )  # tectonic type of the rupture (options described in the enum below)
        self.width = kwargs.get("width")  # down-dip width of the fault rupture plane
        self.zbot = kwargs.get("zbot")  # depth to the bottom of the seismogenic crust
        self.ztor = kwargs.get("ztor")  # depth to top of coseismic rupture (km)

def estimate_z1p0(vs30):
    return (
        np.exp(28.5 - 3.82 / 8.0 * np.log(vs30 ** 8 + 378.7 ** 8)) / 1000.0
    )  # CY08 estimate in KM


def waterfall_plotter(mseed_dir,out_dir):
    # Channel codes to check for
    channel_codes = 'HN?,BN?,HH?,BH?,EH?,SH?'
    sorter = ['HH','BH','EH','HN','BN','SH']
    
    # Channel components to check for
    components = '[1N]','[2E]','Z'
    
    # Get station inventory from the FDSN webservice
    client_NZ = FDSN_Client("GEONET")
    client_IU = FDSN_Client('IRIS')
    inventory_NZ = client_NZ.get_stations(channel=channel_codes)
    inventory_IU = client_IU.get_stations(network='IU',station='SNZO',channel=channel_codes)
    inventory = inventory_NZ+inventory_IU

    model = TauPyModel(model="iasp91")
    
    event_df = pd.read_csv(event_dir,low_memory=False)
    assoc_df = pd.read_csv(assoc_dir,low_memory=False)
    
    for ev_idx, event in event_df[0:10].iterrows():
        lat = event.lat
        lon = event.lon
        depth = event.depth
        otime = UTCDateTime(event['datetime'])
        assoc_sub = assoc_df[assoc_df.evid == event.evid].reset_index(drop=True)
    
        assoc_stas = assoc_sub.sta.unique()
    
        sts = op.core.stream.Stream()
        for sta in assoc_stas:
            assocs = assoc_sub[assoc_sub.sta == sta]
            if len(assocs) > 1:
                assoc_p = assocs[assocs.phase.astype(str).str.lower() == 'p'].iloc[0]
                assoc_s = assocs[assocs.phase.astype(str).str.lower() == 's'].iloc[0]
                ptime_est = UTCDateTime(assoc_p['datetime']) - otime
                p_res = assoc_p['t_res']
                stime_est = UTCDateTime(assoc_s['datetime']) - otime
                s_res = assoc_s['t_res']
                assoc = assocs.iloc[0]
            else:
                assoc = assocs.iloc[0]
                if assoc.phase.lower() == 'p':
                    ptime_est = UTCDateTime(assoc['datetime']) - otime
                    p_res = assoc['t_res']
                    stime_est = []
                    s_res = []
                elif assoc.phase.lower() == 's':
                    ptime_est = []
                    p_res = []
                    stime_est = UTCDateTime(assoc['datetime']) - otime
                    s_res = assoc['t_res']
    #     for assoc_idx, assoc in assoc_sub.iterrows():
            net = assoc.net
    #         sta = assoc.sta
            sta_info = station_df[(station_df.net == net) & (station_df.sta == sta)].iloc[0]
            sta_lat = sta_info.lat
            sta_lon = sta_info.lon
            sta_el = sta_info.elev / 1000
            deg_dist = locations2degrees(lat,lon,sta_lat,sta_lon)
            dist_km = degrees2kilometers(deg_dist)
            r_hyp = ((dist_km) ** 2 + (depth + sta_el) ** 2) ** 0.5
    
            # Create siteprop and faultprop objects for use with the Afshari and Stewart ds595
            # function.
            siteprop = Site()
            faultprop = Fault()
            faultprop.Mw = event.mag
            vs30_default = 500

            siteprop.Rrup = r_hyp
            siteprop.vs30 = vs30_default
            siteprop.z1p0 = estimate_z1p0(siteprop.vs30)

            ds, ds_std = asds.Afshari_Stewart_2016_Ds(siteprop, faultprop, 'Ds595')


    #         p_arrivals = model.get_travel_times(source_depth_in_km=depth,distance_in_degree=deg_dist,phase_list=['ttp'])
    #         s_arrivals = model.get_travel_times(source_depth_in_km=depth,distance_in_degree=deg_dist,phase_list=['tts'])
    #         ptime_est = p_arrivals[0].time # Estimated earliest P arrival time from taup
    #         stime_est = s_arrivals[0].time
            slow_vel = 2 # Assumed slowest velocity of earthquake (Rayleigh or Love)
            surface_est = model.get_travel_times(source_depth_in_km=depth,distance_in_degree=deg_dist,phase_list=['2kmps'])[0].time # Time window has some distance dependence

            if net == 'NZ':
                client = client_NZ
            elif net == 'IU':
                client = client_IU
            elif net == 'AU':
                client = client_AU
            for channel_code in sorter:
                try:
#                     st = client.get_waveforms(network=net, station=sta, location='*', channel=channel_code+'?',
#                     starttime=otime, endtime=otime + surface_est)
                    st = client.get_waveforms(network=net, station='TOZ', location='*', channel=channel_code+'?',
                    starttime=otime, endtime=otime + 120)
                    break
                except:
                    continue

            for tr in st:
                # Creates coordinates attribute dictionary for each trace
                tr.stats.arrivals = {}
                tr.stats.arrivals.p = ptime_est
                tr.stats.arrivals.p_res = ptime_est + p_res
                tr.stats.arrivals.s = stime_est
                tr.stats.arrivals.s_res = stime_est + s_res
    #             tr.stats.arrivals.ds595 = stime_est + ds
    #             tr.stats.arrivals.surface = surface_est
                tr.stats.coordinates = {}
                tr.stats.distance = {}
                tr.stats.distance = dist_km * 1000
                tr.stats.coordinates.latitude = sta_lat
                tr.stats.coordinates.longitude = sta_lon
            sts += st
        sts.detrend('demean')
        sts.taper(max_percentage=0.05)
        sts.filter('highpass', freq=1)
        sts.sort(['starttime'])
    
   
        # Create range of distances to plot. The defaut is 0 - 1000 km in 100 km increments.
        distances = np.arange(0,1000,100)

    #     # First cycle through channel codes
    #     for chan in channel_codes.split(','):
        # Cycle through channel components
        for component in components:
            sts_chan = sts.select(component=component)
            if len(sts_chan) == 0:
                continue
            for distance in distances:
                sub_sts = op.core.stream.Stream()
                min_distance = distance
                max_distance = distance+100
                for tr in sts_chan:
                    if tr.stats.distance/1000 > min_distance and tr.stats.distance/1000 <= max_distance:
                        sub_sts += tr
                print(sub_sts)
                if len(sub_sts) == 0:
                    continue
                starttime_diff = sub_sts[0].stats.starttime - sts[0].stats.starttime
                try:
                    fig = plt.figure(figsize=(12,8))
                    if len(sub_sts) == 1:
                        f = 1.e3
                        sub_sts += sub_sts[0].copy()
                        sub_sts[1].stats.distance = sub_sts[0].stats.distance*(1+1./f)
                    else:
                        f = 1
                    sub_sts.plot(type='section',time_down=True, linewidth=.25, grid_linewidth=.25, show=False, fig=fig,
                        orientation='horizontal',scale=f)
                except:
                    print('No data for range '+str(min_distance)+' - '+str(max_distance))
                    continue
                ax = fig.axes[0]
                transform = blended_transform_factory(ax.transAxes, ax.transData)
                transform2 = blended_transform_factory(ax.transData, ax.transData)
                y_lims = ax.get_ylim()
                y_add = (y_lims[1] - y_lims[0]) / 25
                # Add arrival information to plot
                for tr in sub_sts:
                    if tr.stats.arrivals.p:
                        ax.text(otime + tr.stats.arrivals.p-sub_sts[0].stats.starttime,tr.stats.distance / 1e3, 'P', va="center", ha="left",
                            transform=transform2)
                        ax.vlines(otime + tr.stats.arrivals.p-sub_sts[0].stats.starttime,tr.stats.distance / 1e3-y_add,tr.stats.distance / 1e3+y_add,
                            transform=transform2,linestyles='solid',colors='blue')
                        ax.vlines(otime + tr.stats.arrivals.p_res-sub_sts[0].stats.starttime,tr.stats.distance / 1e3-y_add,tr.stats.distance / 1e3+y_add,
                            transform=transform2,linestyles='dashed',colors='blue')
                    if tr.stats.arrivals.s:
                        ax.text(otime + tr.stats.arrivals.s-sub_sts[0].stats.starttime,tr.stats.distance / 1e3, 'S', va="center", ha="left",
                            transform=transform2)
                        ax.vlines(otime + tr.stats.arrivals.s-sub_sts[0].stats.starttime,tr.stats.distance / 1e3-y_add,tr.stats.distance / 1e3+y_add,
                            transform=transform2,linestyles='solid',colors='red')
                        ax.vlines(otime + tr.stats.arrivals.s_res-sub_sts[0].stats.starttime,tr.stats.distance / 1e3-y_add,tr.stats.distance / 1e3+y_add,
                            transform=transform2,linestyles='dashed',colors='red')
    #                     ax.text(otime + tr.stats.arrivals.ds595-sub_sts[0].stats.starttime,tr.stats.distance / 1e3, 'ds', va="center", ha="left",
    #                         transform=transform2)
    #                     ax.text(otime + tr.stats.arrivals.surface-sub_sts[0].stats.starttime,tr.stats.distance / 1e3, 'surf', va="center", ha="left",
    #                         transform=transform2)
                    ax.text(1.0, tr.stats.distance / 1e3, tr.stats.station,
                            va="center", ha="left", transform=transform)
                plt.title('Event'+str(event.evid)+' - '+component+' at '+str(min_distance)+'-'+str(max_distance)+' km')
    #                 if not os.path.exists(out_dir+str(event.evid)):
    #                     os.makedirs(out_dir+str(event.evid))
    #                 plt.savefig(out_dir+str(event.evid)+'/'+str(event.evid)+'_'+str(distance)+'_'+str(chan[0:2])+component+'.png')
                plt.show()
                plt.close()

# These should be the only variable that needs to be altered.
event_dir = '/Volumes/SeaJade 2 Backup/NZ/EQTransformer/output/mag_out/events/20210304_events.csv'
assoc_dir = '/Volumes/SeaJade 2 Backup/NZ/EQTransformer/output/mag_out/associations/20210304_assocs.csv'
out_dir = '/Volumes/SeaJade 2 Backup/NZ/Reports/Waterfall_Plots/'

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


waterfall_plotter(mseed_dir,out_dir)