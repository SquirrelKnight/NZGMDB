from obspy import UTCDateTime
from obspy.clients.fdsn import Client as FDSN_Client
from obspy import read_inventory
import numpy as np
import pandas as pd
import json
import requests

def get_elevation(lat = None, long = None):
    '''
        script for returning elevation in m from lat, long
    '''
    if lat is None or long is None: return None
    
    query = 'https://api.opentopodata.org/v1/nzdem8m?locations='+str(lat)+','+str(long)
    
    # Request with a timeout for slow responses
    r = requests.get(query)

    # Only get the json response in case of 200 or 201
    if r.status_code == 200 or r.status_code == 201:
        elevation = json_normalize(r.json(), 'results')['elevation'].values[0]
    else: 
        elevation = None
    return elevation

# channel_codes = ('HHN','HHE','HHZ','HH1','HH2','BHN','BHE','BHZ','BH1','BH2','EHN','EHE','EHZ','EH1','EH2')

# channel_codes = ('HHN','HHE','HHZ','HH1','HH2','BHN','BHE','BHZ','BH1','BH2','EHN','EHE','EHZ','EH1','EH2','HN1','HN2','HNZ','HNE','HNN','BNN','BNE','BNZ','BN1','BN2')

channel_codes = 'HN?,BN?,HH?,BH?,EH?,SH?'

pref_channel_order = ('HH','BH','HN','BN','EH','SH')

client = FDSN_Client("GEONET")

# station_db = 'inputs/stations.csv'
# fo = open(station_db,'w')
# print('snet','sta','location','chan','on','off','lat','lon','elev',file=fo,sep=',')

# starttime = '2020-03-01 00:00:00.000'
# endtime = '2020-05-01 00:00:00.000'
# latitude = -43.55
# longitude = 172.5
# maxradius = 1
filter_network = []
filter_station = []
channel_list=["HH[ZNE]", "BH[ZNE]", "HN[ZNE]", "BN[ZNE]", "EH[ZNE]", "SH[ZNE]"]
chan_priority=[ch[:2] for ch in channel_list]

# inventory = client.get_stations(level="response",starttime = starttime, endtime = endtime, latitude = latitude, longitude = longitude, maxradius = maxradius)
# inventory = client.get_stations(level="response")
client_NZ = FDSN_Client("GEONET")
client_IU = FDSN_Client('IRIS')
inventory_NZ = client_NZ.get_stations(level='channel',channel=channel_codes)
inventory_IU = client_IU.get_stations(network='IU',station='SNZO',level='channel',channel=channel_codes)
inventory = inventory_NZ+inventory_IU


station_list = {}
for ev in inventory:
	net = ev.code
	if net not in filter_network:
		for st in ev:
			station = st.code
			print(str(net)+"--"+str(station))
			if station not in filter_station:
				elv = st.elevation
				lat = st.latitude
				lon = st.longitude
				if elv > 9000:
					print(station,elv)
					elv = get_elevation(lat, lon)
					print(elv)
					if elv == None:
						elv = 0
					time.sleep(1) # Sometimes the server doesn't send back data, add sleep timer
			station_list[str(station)] ={"network": net,
											"coords": [lat, lon, elv]
											}

sta_df = pd.read_csv('/Volumes/SeaJade 2 Backup/github/delta/network/stations.csv',
    low_memory=False)

sta_list = ['HARZ', 'OTW', 'MGZ', 'CNZ', 'GFW', 'MOW', 'BLW', 'BBW', 'WAHZ', 'RAEZ', 'DFE', 
    'PATZ', 'PAHZ', 'TTH', 'OTAZ', 'ECNZ', 'PARZ', 'LWHI', 'LOTA', 'LKOW', 'LPOR', 'LTAT', 
    'LWAI', 'GREC', 'LRAN', 'LWTT', 'LCRO', 'TRUC', 'LGLS', 'LPIH', 'LMOT', 'LBST', 'LGDS', 
    'LPAP', 'LROS', 'LTIM', 'MIMC', 'LPEN', 'LS16', 'LMOW', 'LAHV', 'LS25', 'LOPO', 'LMAN', 
    'LDEN', 'LATK', 'LWON', 'TUIC', 'TAPC', 'LMAT', 'LMAU', 'LJAI', 'TIKO', 'MATA', 'INFR', 
    'NGA1', 'NGA3', 'KQ05', 'RHOC', 'THQ2', 'WTSZ', 'WPSZ', 'WMSZ', 'WDSZ']
    
sta_df_sub = sta_df[sta_df.Station.isin(sta_list)]
for i,row in sta_df_sub.iterrows():
    if row.Elevation > 9000:
        print(station,row.Elevation)
        row.Elevation = get_elevation(row.Latitude, row.Longitude)
        print(row.Elevation)
        if row.Elevation == None:
            row.Elevation = 0
        time.sleep(1) # Sometimes the server doesn't send back data, add sleep timer
    station_list[str(row.Station)] = {"network": 'NZ', 
        "coords": [row.Latitude, row.Longitude, row.Elevation]}

with open('station_list_no_chan.json', 'w') as fp:
	json.dump(station_list, fp)