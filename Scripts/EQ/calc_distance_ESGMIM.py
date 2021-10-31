from abc import ABC
import pandas as pd
import mag_scaling
from pyproj import Transformer			# conda install pyproj
import math
import numpy as np
from typing import List, Union, Dict
from qcore import nhm, formats, geo, srf
import os
import src_site_dist
import glob
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon, LineString
from pandarallel import pandarallel		# conda install -c bjrn pandarallel
import obspy as op
from obspy.clients.fdsn import Client as FDSN_Client
from obspy import UTCDateTime

# Convert from WGS84 to NZGD2000
wgs2nztm = Transformer.from_crs(4326, 2193)
nztm2wgs = Transformer.from_crs(2193, 4326)

class Fault(ABC):
    subfault_spacing = 0.1
    type = 0
    _mag = None
    _moment = None
    name = None
    _latitude = None
    _longitude = None
    _strike = None
    _rake = None
    _dip = None
    _depth = None
    _width = None
    _length = None
    _shypo = None
    _dhypo = None

    @property
    def pid(self):
        return self.name

    @pid.setter
    def pid(self, value):
        self.name = value

    @property
    def mom(self):
        if self._moment is None:
            self._moment = mag2mom(self._mag)
        return self._moment

    @property
    def latitude(self):
        return self._latitude

    @latitude.setter
    def latitude(self, value):
        self._latitude = value

    @property
    def longitude(self):
        return self._longitude

    @longitude.setter
    def longitude(self, value):
        self._longitude = value

    @property
    def magnitude(self):
        return self._mag

    @magnitude.setter
    def magnitude(self, mag):
        if mag > 11:
            raise ValueError(
                f"Given mag {mag} is greater than theoretically possible 11"
            )
        self._mag = mag

    @property
    def length(self):
        return self._length

    @property
    def width(self):
        return self._width

    def to_dict(self):
        return {
            "type": self.type,
            "magnitude": self._mag,
            "moment": self.mom,
            "name": self.name,
            "longitude": self.longitude,
            "latitude": self.latitude,
            "strike": self._strike,
            "rake": self._rake,
            "dip": self._dip,
            "depth": self._depth,
        }

    @property
    def rake(self):
        return self._rake

    @rake.setter
    def rake(self, value):
        value = ((value + 180) % 360) - 180
        self._set_rake(value)

    def _set_rake(self, value):
        self._rake = value

    @property
    def dip(self):
        return self._dip

    @dip.setter
    def dip(self, value):
        if 90 < value < 180:
            self.strike = self.strike + 180
            self.rake = -self.rake
            value = 180 - value
        elif value < 0 or value > 180:
            # The fault is now above ground
            raise ValueError(f"Invalid dip value: {value}")

        self._set_dip(value)

    def _set_dip(self, value):
        self._dip = value

def TVZ_path_calc(row,sta_df,taupo_polygon,tect_domain_points,wgs2nztm,r_epis):
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon, LineString
    import numpy as np

    # Taupo VZ polygon acquired from https://www.geonet.org.nz/data/supplementary/earthquake_location_grope

#     for index,row in df[0:10].iterrows():
    event_id = row.evid
    ev_lat = row.lat
    ev_lon = row.lon
    ev_depth = row.depth
    reloc = row.reloc
    ev_transform = wgs2nztm.transform(ev_lat,ev_lon)
#         if ev_lon < 0:
#             ev_lon = ev_lon+360
    network = sta_df.net
    station = sta_df.sta
    sta_lat = sta_df.lat
    sta_lon = sta_df.lon
    sta_elev = sta_df.elev
    sta_transform = wgs2nztm.transform(sta_lat,sta_lon)
#         if sta_lon < 0:
#             sta_lon = sta_lon+360
    tvz_lengths = []
    ii = 0
    for i,sta in sta_df.iterrows():
        sta_transform = wgs2nztm.transform(sta.lat,sta.lon)
#         dist, az, b_az = op.geodetics.gps2dist_azimuth(ev_lat, ev_lon, sta.lat, sta.lon)
#         dists.append(dist)
#         azs.append(az)
#         b_azs.append(b_az)
#         r_epi = dist/1000
#         r_hyp = (r_epi ** 2 + (ev_depth + sta.elev/1000) ** 2) ** 0.5
#         r_epis.append(r_epi)
#         r_hyps.append(r_hyp)
        line = LineString([[ev_transform[0],ev_transform[1]],[sta_transform[0],sta_transform[1]]])
    
        tvz_length = 0

        if line.intersection(taupo_polygon):
            line_points = line.intersection(taupo_polygon)
            tvz_length = line_points.length / 1000 / r_epis[ii]
            if tvz_length > 1:
                tvz_length = 1
        tvz_lengths.append(tvz_length)
        ii += 1
    
    # Output evid, net, sta, r_epi, r_hyp, az, b_az, reloc, rrup, and rjb. toa has
    # been omitted. This is better left in the phase arrival table.
    return tvz_lengths


def rotate_back(x, y, xs, ys, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    from pyproj import CRS
    from pyproj import Transformer
    import math
    import numpy as np

    angle = np.radians(angle)
    # 	ox = orilon
    # 	oy = orilat
    # 	transformer_from_latlon = Transformer.from_crs(4326, 2193) # WSG84 to New Zealand NZDG2000 coordinate transform
    # 	transformer_to_latlon = Transformer.from_crs(2193, 4326)

    # 	ox, oy = transformer_from_latlon.transform(orilat,orilon)
    ox, oy = x,y
#     px = ox+xs*1000
#     py = oy-ys*1000
    px = xs
    py = ys
    # 	px, py = transformer_from_latlon.transform(lats,lons)

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    lats, lons = nztm2wgs.transform(qx,qy)
    # 	stations['z'] = (stations.elevation-dx)/dx
    return lats, lons


# Get event magnitude, location
# Get event strike, dip, rake

# for i, row in event_df[0:10].iterrows():
def gregorny(df,df_merged,df_station,taupo_polygon,cmt_df,domain_focal_df,srf_files,srfs,geonet_cmt_df,event_out_file,props_out_file):
    row = df.copy()
#     row = event_df[event_df.evid == '1506176'].iloc[0].copy()
#     print(row.name)
    evid = row.evid
    lat = row.lat
    lon = row.lon
    depth = row.depth
    mag = row.mag
    reloc = row.reloc
    if row.mag != -9 and ~np.isnan(row.mag):
        try:
            # Calculate area from event tectonic type
            sta_df = df_merged[df_merged.evid.isin([evid])].merge(df_station[['net','sta','lat','lon','elev']],on=['sta'])
            indexNames = sta_df[ (sta_df['net'] == 'AU') & (sta_df['sta'] == 'ARPS') ].index #remove AU ARPS
            sta_df.drop(indexNames , inplace=True)
            sta_df['depth'] = sta_df['elev']/-1000
            stations = sta_df[['lon','lat','depth']].to_numpy()
            # If crustal
            if row.tect_class == 'Crustal': # Also use for slab
                magnitude_scaling_relation = mag_scaling.MagnitudeScalingRelations.LEONARD2014
            # If other
            else:
                magnitude_scaling_relation = mag_scaling.MagnitudeScalingRelations.SKARLATOUDIS2016
            # Allen 2017 for Slab events or just Leonard (talk to Mike)?

            srf_flag = np.isin(evid,srfs)
    
            if srf_flag:
                srf_file = srf_files[np.where(np.isin(srfs,evid) == True)[0][0]]
                srf_points = srf.read_srf_points(srf_file)
                srf_header = srf.read_header(srf_file,idx=True)
                ztor = srf_points[0][2]
                dbottom = srf_points[-1][2]
                f_type = 'srf'
                cmt = geonet_cmt_df[geonet_cmt_df.PublicID == evid].iloc[0]
        
        #         if len(srf_header) > 1:
                strike1 = cmt.strike1
                dip1 = cmt.dip1
                rake1 = cmt.rake1
                strike2 = cmt.strike2
                dip2 = cmt.dip2
                rake2 = cmt.rake2
           
                fault_strike = srf_header[0]['strike']
                strike1_diff = abs(fault_strike-strike1)
                if strike1_diff > 180:
                    strike1_diff = 360-strike1_diff
                strike2_diff = abs(fault_strike-strike2)
                if strike2_diff > 180:
                    strike2_diff = 360-strike2_diff

                if strike1_diff < strike2_diff:
                    strike = strike1
                    rake = rake1
                    dip = dip1
                else:
                    strike = strike2
                    rake = rake2
                    dip = dip2
                length = np.sum([header['length'] for header in srf_header])
                dip_dist = np.mean([header['width'] for header in srf_header])
            else:
                cmt = cmt_df[cmt_df.PublicID == evid]
                cmt_2 = geonet_cmt_df[geonet_cmt_df.PublicID == evid]
                if len(cmt) > 0:
                    cmt = cmt.iloc[0]
                    strike = cmt.strike1
                    dip = cmt.dip1
                    rake = cmt.rake1
                    f_type = 'cmt'
                elif len(cmt_2) > 0:
                    cmt_2 = cmt_2.iloc[0]
                    f_type = 'cmt_unc'
                    strike1 = cmt_2.strike1
                    dip1 = cmt_2.dip1
                    rake1 = cmt_2.rake1
                    strike2 = cmt_2.strike2
                    dip2 = cmt_2.dip2
                    rake2 = cmt_2.rake2
                    if row.domain_no != 0:
                        domain = domain_focal_df[domain_focal_df.Domain_No == row.domain_no].iloc[0]
                        do_strike = domain.strike
                        do_rake = domain.rake
                        do_dip = domain.dip
                    
                        strike1_diff = abs(do_strike-strike1)
                        if strike1_diff > 180:
                            strike1_diff = 360-strike1_diff
                        strike2_diff = abs(do_strike-strike2)
                        if strike2_diff > 180:
                            strike2_diff = 360-strike2_diff
                        if strike1_diff < strike2_diff:
                            strike = strike1
                            rake = rake1
                            dip = dip1
                        else:
                            strike = strike2
                            rake = rake2
                            dip = dip2
                    else:
                        strike = strike1
                        rake = rake1
                        dip = dip1
                else:
                    f_type = 'domain'
                    if row.domain_no == 0:
                        strike = 220
                        dip = 45
                        rake = 90
                    else:
                        domain = domain_focal_df[domain_focal_df.Domain_No == row.domain_no].iloc[0]
                        strike = domain.strike
                        rake = domain.rake
                        dip = domain.dip
                fault = Fault
                fault.name = evid
                fault.latitude = lat
                fault.longitude = lon
                fault.depth = depth
                fault.magnitude = mag
                fault.moment = float(mag_scaling.mag2mom(mag))
                fault.strike = strike
                fault.dip = dip
                fault.rake = rake
                fault.magnitude_scaling_relation = magnitude_scaling_relation

                srf_points = []
                srf_header = []
    
                POINTS_PER_KILOMETER = (
                    1 / 0.1
                )  # 1km divided by distance between points (1km/0.1km gives 100m grid)

                dip_dir = strike + 90
                if dip_dir >= 360:
                    dip_dir = dip_dir - 360

                area = mag_scaling.get_area(fault)
                dip_dist = mag_scaling.get_width(fault)
                length = mag_scaling.get_length(fault)
        
                x,y = wgs2nztm.transform(fault.latitude,fault.longitude)
                z = fault.depth
    
                x1 = x-(length*1000/2)
                x2 = x+(length*1000/2)
                y1 = y
                y2 = y
    
                lat1,lon1 = rotate_back(x,y,x1,y1,strike)
                lat2,lon2 = rotate_back(x,y,x2,y2,strike)
    
                height = math.sin(math.radians(dip)) * dip_dist
                width = abs(height / math.tan(math.radians(dip)))
                dtop = z - (height / 2) # The same as Ztor
                if dtop < 0:
                    dtop = 0
                ztor = dtop
                dbottom = dtop + height

                end_strike = geo.ll_bearing(lon1, lat1, lon2, lat2)

                nstrike = int(round(length * POINTS_PER_KILOMETER))
                if nstrike == 0:
                    rxs, rys = [],[]
        
                    r_epis = geo.get_distances(np.dstack([sta_df.lon.values,sta_df.lat.values])[0],lon,lat)
                    r_hyps = np.sqrt(r_epis ** 2 + (depth - sta_df.depth.values) ** 2)
                    azs = np.array([geo.ll_bearing(lon,lat,station[0],station[1]) for station in stations])
                    b_azs = np.array([geo.ll_bearing(station[0],station[1],lon,lat) for station in stations])
                    tvz_lengths = TVZ_path_calc(row,sta_df,taupo_polygon,tect_domain_points,wgs2nztm,r_epis)

                    prop_df = pd.DataFrame(columns=['evid','net','sta','r_epi','r_hyp','r_jb','r_rup','r_x','r_y','r_tvz','az','b_az','reloc'])
                    prop_df[['evid','net','sta']] = sta_df[['evid','net','sta']]
                    prop_df['r_epi'] = r_epis
                    prop_df['r_hyp'] = r_hyps
                    prop_df['r_jb'] = r_epis
                    prop_df['r_rup'] = r_hyps
                    if len(rxs) > 0:
                        prop_df['r_x'] = rxs
                    if len(rys) > 0:
                        prop_df['r_y'] = rys
                    prop_df['r_tvz'] = tvz_lengths
                    prop_df['az'] = azs
                    prop_df['b_az'] = b_azs
                    prop_df['f_type'] = f_type
                    prop_df['reloc'] = reloc
    
                    row['strike'] = strike
                    row['dip'] = dip
                    row['rake'] = rake
                    row['f_length'] = length
                    row['f_width'] = dip_dist
                    row['f_type'] = f_type
                    row['z_tor'] = dtop
                    row['z_bor'] = dbottom
            
                    row_out = row.copy()
                    row_out = row_out.to_frame().T
                    row_out.to_csv(event_out_file,index=False,header=False,mode='a')
                    prop_df.to_csv(props_out_file,index=False,header=False,mode='a')
                    return
        #             return row, prop_df
                strike_dist = length / nstrike  

                for j in range(nstrike+1):
                    top_lat, top_lon = geo.ll_shift(lat1, lon1, strike_dist * j, end_strike)
                    srf_points.append((top_lon, top_lat, dtop))

                ndip = int(round(dip_dist * POINTS_PER_KILOMETER))
                hdip_dist = width / ndip
                vdip_dist = height / ndip

                plane_offset = 0
                for j in range(1, ndip+1): #added plus 1 to ndip so that it has the bottom srf points
                    hdist = j * hdip_dist
                    vdist = j * vdip_dist + dtop
                    for local_lon, local_lat, local_depth in srf_points[
                        plane_offset : plane_offset + nstrike + 1
                    ]:
                        new_lat, new_lon = geo.ll_shift(
                            local_lat, local_lon, hdist, dip_dir
                        )
                        srf_points.append([new_lon, new_lat, vdist])
                plane_offset += nstrike * ndip
                srf_header.append({"nstrike": nstrike, "ndip": ndip, "strike": strike})

                srf_points = np.array(srf_points)

        #     plt.plot([lon1,lon2],[lat1,lat2],'b')
        #     plt.plot(lon,lat,'bo')
        #     plt.plot(lon1,lat1,'ro')
        #     plt.plot(lon2,lat2,'go')
        #     plt.axis('equal')
        #     plt.show()
    
            rrups, rjbs = src_site_dist.calc_rrup_rjb(srf_points, stations)
            if srf_header:
                rxs, rys = src_site_dist.calc_rx_ry(srf_points, srf_header, stations)
            else:
                rxs, rys = [],[]
    
            # rotate line to be along strike
            # determine x1,y1,x2,y2 (cartesian coordinates)
            # convert to lat1,lon1,lat2,lon2 (geographic coordinates)
            # determine dtop and dbottom
            # generate srf header and srf points
            # determine stations from arrivals and ims (lat, lon, depth) and write to stations np array
            # calculate distances rjb, rrup, rx, ry

            # old tvz function to calculate r_hyp, r_epi
    #         dists, azs, b_azs, r_epis, r_hyps, tvz_lengths = TVZ_path_calc(row,sta_df,taupo_polygon,tect_domain_points,wgs2nztm)
        
            # New equations to calculate r_hyp and r_epi
            r_epis = geo.get_distances(np.dstack([sta_df.lon.values,sta_df.lat.values])[0],lon,lat)
            r_hyps = np.sqrt(r_epis ** 2 + (depth - sta_df.depth.values) ** 2)
            azs = np.array([geo.ll_bearing(lon,lat,station[0],station[1]) for station in stations])
            b_azs = np.array([geo.ll_bearing(station[0],station[1],lon,lat) for station in stations])
            tvz_lengths = TVZ_path_calc(row,sta_df,taupo_polygon,tect_domain_points,wgs2nztm,r_epis)

            prop_df = pd.DataFrame(columns=['evid','net','sta','r_epi','r_hyp','r_jb','r_rup','r_x','r_y','r_tvz','az','b_az','reloc'])
            prop_df[['evid','net','sta']] = sta_df[['evid','net','sta']]
            prop_df['r_epi'] = r_epis
            prop_df['r_hyp'] = r_hyps
            prop_df['r_jb'] = rjbs
            prop_df['r_rup'] = rrups
            if len(rxs) > 0:
                prop_df['r_x'] = rxs
            if len(rys) > 0:
                prop_df['r_y'] = rys
            prop_df['r_tvz'] = tvz_lengths
            prop_df['az'] = azs
            prop_df['b_az'] = b_azs
            prop_df['f_type'] = f_type
            prop_df['reloc'] = reloc
    
            row['strike'] = strike
            row['dip'] = dip
            row['rake'] = rake
            row['f_length'] = length
            row['f_width'] = dip_dist
            row['f_type'] = f_type
            row['z_tor'] = ztor
            row['z_bor'] = dbottom
    
            row_out = row.copy()
            row_out = row_out.to_frame().T
            row_out.to_csv(event_out_file,index=False,header=False,mode='a')
            prop_df.to_csv(props_out_file,index=False,header=False,mode='a')
            return
        except Exception as e:
            print(e)
    else:
        sta_df = df_merged[df_merged.evid == evid].merge(df_station[['net','sta','lat','lon','elev']],on=['sta'])
        indexNames = sta_df[ (sta_df['net'] == 'AU') & (sta_df['sta'] == 'ARPS') ].index #remove AU ARPS
        sta_df.drop(indexNames , inplace=True)
        sta_df['depth'] = sta_df['elev']/-1000
        stations = sta_df[['lon','lat','depth']].to_numpy()
        
        r_epis = geo.get_distances(np.dstack([sta_df.lon.values,sta_df.lat.values])[0],lon,lat)
        r_hyps = np.sqrt(r_epis ** 2 + (depth - sta_df.depth.values) ** 2)
        azs = np.array([geo.ll_bearing(lon,lat,station[0],station[1]) for station in stations])
        b_azs = np.array([geo.ll_bearing(station[0],station[1],lon,lat) for station in stations])
        tvz_lengths = TVZ_path_calc(row,sta_df,taupo_polygon,tect_domain_points,wgs2nztm,r_epis)

        prop_df = pd.DataFrame(columns=['evid','net','sta','r_epi','r_hyp','r_jb','r_rup','r_x','r_y','r_tvz','az','b_az','reloc'])
        prop_df[['evid','net','sta']] = sta_df[['evid','net','sta']]
        prop_df['r_epi'] = r_epis
        prop_df['r_hyp'] = r_hyps
        prop_df['r_jb'] = ''
        prop_df['r_rup'] = ''
        prop_df['r_x'] = ''
        prop_df['r_y'] = ''
        prop_df['r_tvz'] = tvz_lengths
        prop_df['az'] = azs
        prop_df['b_az'] = b_azs
        prop_df['f_type'] = ''
        prop_df['reloc'] = reloc

        row['strike'] = ''
        row['dip'] = ''
        row['rake'] = ''
        row['f_length'] = ''
        row['f_width'] = ''
        row['f_type'] = ''
        row['z_tor'] = depth
        row['z_bor'] = depth

        row_out = row.copy()
        row_out = row_out.to_frame().T
        row_out.to_csv(event_out_file,index=False,header=False,mode='a')
        prop_df.to_csv(props_out_file,index=False,header=False,mode='a')
    
#     return row_out, prop_df

directory = '/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/'

geonet_cmt_df = pd.read_csv(directory+'focal/GeoNet_CMT_solutions.csv',low_memory=False)    
cmt_df = pd.read_csv(directory+'focal/GeoNet_CMT_solutions_20201129_PreferredNodalPlane_v1.csv',low_memory=False)
# event_df = pd.read_csv(directory+'converted_output/earthquake_source_table_fixed_mags.csv',low_memory=False)
event_df = pd.read_csv(directory+'converted_output/earthquake_source_table_relocated_tectdomain.csv',low_memory=False)
### Filter data to be within lat/lon limitations.
event_df_copy = event_df.copy()
event_df_copy.loc[event_df_copy.lon < 0, 'lon'] = 360 + event_df_copy.lon[event_df_copy.lon < 0]
event_df = event_df[((event_df_copy.lon < 190) & (event_df_copy.lon >155)) & (event_df_copy.lat < -15)].reset_index(drop=True)

domain_focal_df = pd.read_csv(directory+'focal/focal_mech_tectonic_domain_v1.csv',low_memory=False)
srf_files = glob.glob(directory+'focal/SrfSourceModels/*.srf')
srfs = [os.path.basename(srf_file).split('.')[0] for srf_file in srf_files]
df_im = pd.read_csv(directory+'converted_output/IM_catalogue/ground_motion_im_catalogue_final.csv',low_memory=False)
df_im_sub = df_im[df_im.duplicated(subset=['sta','evid']) == False][['evid','sta']]
df_arr = pd.read_csv(directory+'converted_output/phase_arrival_table.csv',low_memory=False)
df_arr_sub = df_arr[df_arr.duplicated(subset=['sta','evid']) == False][['evid','sta']]
df_merged = pd.concat([df_im_sub,df_arr_sub],axis=0,ignore_index=True)
df_merged = df_merged[df_merged.duplicated() == False]
# df_station = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/site_table_response.csv',low_memory=False)

tect_domain_points = pd.read_csv(directory+'tectonic domains/tectonic_domain_polygon_points.csv',low_memory=False)
tvz_points = tect_domain_points[tect_domain_points.domain_no == 4][['latitude','longitude']]
taupo_transform = np.dstack(np.array(wgs2nztm.transform(tvz_points.latitude,tvz_points.longitude)))[0]
taupo_polygon = Polygon(taupo_transform)

# Load station information from FDSN in case station csv is not complete
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

df_station = pd.DataFrame(station_info,columns=['net','sta','lat','lon','elev'])
df_station = df_station.drop_duplicates().reset_index(drop=True)

# for i,x in event_df[213916:213926].iterrows():
#     results = gregorny(x,df_merged,df_station,taupo_polygon,cmt_df,domain_focal_df,srf_files,srfs,geonet_cmt_df)
#     events_out = pd.concat([events_out,results[0].to_frame().T],ignore_index=True)
#     props_out = pd.concat([props_out,results[1]],ignore_index=True)
#     print(i)

years = np.arange(2000,2021)
# months = np.arange(1,13)
# years = np.unique(geonet.origintime.values.astype('datetime64[Y]').astype(int)+1970)
for year in years:
# 	for month in months:
    process_year = year
    print('Processing for year '+str(process_year))
#     event_out_file = directory+'converted_output/earthquake_source_table_complete_'+str(process_year)+'.csv'
#     props_out_file = directory+'converted_output/propagation_path_table_complete_'+str(process_year)+'.csv'
    event_out_file = directory+'converted_output/earthquake_source_table_complete_prop_'+str(process_year)+'.csv'
    props_out_file = directory+'converted_output/propagation_path_table_complete_prop_'+str(process_year)+'.csv'
#     process_month = month
# 	process_year = [2003] ### Assign a list of years to download waveforms for
    event_sub_mask = event_df.datetime.values.astype('datetime64[Y]').astype(int)+1970 == process_year
#     event_sub_mask = event_df.datetime.dt.year == process_year
# 		geonet_sub_mask = np.isin(geonet.origintime.values.astype('datetime64[Y]').astype(int)+1970,process_year)
    event_sub = event_df[event_sub_mask].reset_index(drop=True)
    
    if len(event_sub) > 0:
        event_list = event_sub.evid.unique()
#         geonet_cmt_df_sub = geonet_cmt_df[geonet_cmt_df['PublicID'].isin(event_list)]
#         cmt_df_sub = cmt_df[cmt_df['PublicID'].isin(event_list)]
        df_merged_sub = df_merged[df_merged['evid'].isin(event_list)] # Subsetting the data makes the program run significantly faster

        events_out = pd.DataFrame(columns=['evid', 'datetime', 'lat', 'lon', 'depth', 'loc_type', 'loc_grid',
               'mag', 'mag_type', 'mag_method', 'mag_unc', 'mag_orig', 'mag_orig_type', 'mag_orig_unc', 'ndef', 'nsta', 'nmag',
               't_res', 'reloc', 'tect_class', 'tect_method', 'domain_no',
               'domain_type', 'strike', 'dip', 'rake', 'f_length', 'f_width', 'f_type',
               'z_tor', 'z_bor'])
        props_out = pd.DataFrame(columns=['evid', 'net', 'sta', 'r_epi', 'r_hyp', 'r_jb', 'r_rup', 'r_x', 'r_y',
               'r_tvz', 'az', 'b_az', 'reloc', 'f_type'])
        events_out.to_csv(event_out_file,index=False)
        props_out.to_csv(props_out_file,index=False)

        pandarallel.initialize(nb_workers=8,progress_bar=True) # Customize the number of parallel workers
#         for idx,x in event_sub.iterrows():
#             print(idx)
#             gregorny(x,df_merged,df_station,taupo_polygon,cmt_df,domain_focal_df,srf_files,srfs,geonet_cmt_df,event_out_file,props_out_file)
        event_sub.parallel_apply(lambda x: gregorny(x,df_merged_sub,df_station,taupo_polygon,cmt_df,domain_focal_df,srf_files,srfs,geonet_cmt_df,event_out_file,props_out_file),axis=1)
        print('')
        print('Completed processing for year '+str(process_year))
# 

# events_out_final = pd.concat([pd.read_csv(f,low_memory=False) for f in glob.glob(directory+'converted_output/earthquake_source_table_complete_*.csv')])
# props_out_final = pd.concat([pd.read_csv(f,low_memory=False) for f in glob.glob(directory+'converted_output/propagation_path_table_complete_*.csv')])
events_out_final = pd.concat([pd.read_csv(f,low_memory=False) for f in glob.glob(directory+'converted_output/earthquake_source_table_complete_prop_*.csv')])
props_out_final = pd.concat([pd.read_csv(f,low_memory=False) for f in glob.glob(directory+'converted_output/propagation_path_table_complete_prop_*.csv')])

# events_out_final.to_csv(directory+'converted_output/earthquake_source_table_complete.csv',index=False)
# props_out_final.to_csv(directory+'converted_output/propagation_path_table_complete.csv',index=False)
events_out_final.to_csv(directory+'converted_output/earthquake_source_table_complete.csv',index=False)
props_out_final.to_csv(directory+'converted_output/propagation_path_table_complete.csv',index=False)
# Unpack results
# for x in results:
#     events_out = pd.concat([events_out,x[0]],ignore_index=True)
#     props_out = pd.concat([props_out,x[1]],ignore_index=True)

# Write to complete files
# df_events = df_events[['evid', 'datetime', 'lat', 'lon', 'depth', 'loc_type', 'loc_grid',
#        'mag', 'mag_type', 'mag_method', 'mag_unc', 'ndef', 'nsta', 'nmag',
#        't_res', 'reloc', 'tect_class', 'tect_method', 'domain_no',
#        'domain_type', 'strike', 'dip', 'rake', 'f_length', 'f_width', 'ztor',
#        'dbottom']]