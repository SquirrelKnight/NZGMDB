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

sta_df = pd.read_csv('/Volumes/SeaJade 2 Backup/github/delta/network/stations.csv',
    low_memory=False)

sta_list = ['HARZ', 'OTW', 'MGZ', 'CNZ', 'GFW', 'MOW', 'BLW', 'BBW', 'WAHZ', 'RAEZ', 'DFE', 
    'PATZ', 'PAHZ', 'TTH', 'OTAZ', 'ECNZ', 'PARZ', 'LWHI', 'LOTA', 'LKOW', 'LPOR', 'LTAT', 
    'LWAI', 'GREC', 'LRAN', 'LWTT', 'LCRO', 'TRUC', 'LGLS', 'LPIH', 'LMOT', 'LBST', 'LGDS', 
    'LPAP', 'LROS', 'LTIM', 'MIMC', 'LPEN', 'LS16', 'LMOW', 'LAHV', 'LS25', 'LOPO', 'LMAN', 
    'LDEN', 'LATK', 'LWON', 'TUIC', 'TAPC', 'LMAT', 'LMAU', 'LJAI', 'TIKO', 'MATA', 'INFR', 
    'NGA1', 'NGA3', 'KQ05', 'RHOC', 'THQ2', 'WTSZ', 'WPSZ', 'WMSZ', 'WDSZ']
    
sta_df_sub = sta_df[sta_df.Station.isin(sta_list)]

station_list = {}
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

with open('station_list_extra.json', 'w') as fp:
	json.dump(station_list, fp)