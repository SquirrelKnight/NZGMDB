import pandas as pd
from obspy.clients.fdsn import Client as FDSN_Client
import numpy as np
import fiona
from pyproj import Transformer

# def TVZ_calc(cat):
#     from shapely.geometry import Point
#     from shapely.geometry.polygon import Polygon
#     import numpy as np
#     from pyproj import Transformer			# conda install pyproj
# 
#     wgs2nztm = Transformer.from_crs(4326, 2193)
# 
#     # Taupo VZ polygon acquired from https://www.geonet.org.nz/data/supplementary/earthquake_location_grope
#     tect_domain_points = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/tectonic domains/tectonic_domain_polygon_points.csv',low_memory=False)
#     tvz_points = tect_domain_points[tect_domain_points.domain_no == 4][['latitude','longitude']]
#     taupo_transform = np.dstack(np.array(wgs2nztm.transform(tvz_points.latitude,tvz_points.longitude)))[0]
#     taupo_polygon = Polygon(taupo_transform)
# 
#     for index,row in cat.iterrows():
#         sta_id = row.sta
#         lat = row.lat
#         lon = row.lon
#         if lon < 0:
#             lon = lon+360
# 
#         point = Point(lat,lon)
# 
#         if taupo_polygon.contains(point):
#             cat.loc[index,'TVZ'] = 1
#         else:
#             cat.loc[index,'TVZ'] = 0
# 
#     return cat

def get_domains(df,shapes,transformer):
	from shapely.geometry import Point	# conda install shapely
	from shapely.geometry.polygon import Polygon
	import numpy as np
	
	x,y = transformer.transform(df.lon,df.lat)
	points = [x,y]

	point = Point(points)

	no, name, type = [],[],[]
	for layer in shapes:
		domain_no = layer['properties']['Domain_No']
		domain_name = layer['properties']['DomainName']
		domain_type = layer['properties']['DomainType']
		geometry_type = layer['geometry']['type']
		geometry_coords = layer['geometry']['coordinates']
		if geometry_type == 'MultiPolygon':
			for coords in geometry_coords:		
				polygon = Polygon(coords[0])
				if polygon.contains(point):
					no, name, type = domain_no, domain_name, domain_type
		else:
			polygon = Polygon(geometry_coords[0])
			if polygon.contains(point):
				no, name, type = domain_no, domain_name, domain_type
	if not no:
		no, name, type = 0, 'Oceanic', None
	return pd.Series([no, name, type])

	
shape = fiona.open("/Users/jesse/Downloads/TectonicDomains/TectonicDomains_Feb2021_8_NZTM.shp")

# Writes layers of the shape file to a shapes array; prevents read errors if accessing
# the shape file directly
shapes = []
for layer in shape:
	domain_no = layer['properties']['Domain_No']
	domain_name = layer['properties']['DomainName']
# 	print(domain_no,domain_name)
	shapes.append(layer)

wgs2nztm = Transformer.from_crs(4326, 2193, always_xy=True)

client_NZ = FDSN_Client("GEONET")
client_IU = FDSN_Client('IRIS')

client_NZ = FDSN_Client("GEONET")
client_IU = FDSN_Client('IRIS')
inventory_NZ = client_NZ.get_stations()
inventory_IU = client_IU.get_stations(network='IU',station='SNZO,AFI,CTAO,RAO,FUNA,HNR,PMG')
inventory = inventory_NZ+inventory_IU
station_info = []
for network in inventory:
	for station in network:
		station_info.append([network.code, station.code, station.latitude, station.longitude, station.elevation])
sta_df = pd.DataFrame(station_info,columns=['net','sta','lat','lon','elev'])
sta_df = sta_df.drop_duplicates().reset_index(drop=True)

sc_df = pd.read_excel('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/site_info/NSHM SMS Site Metadata v1.2.xlsx')
sc_df.rename(columns={'Name':'sta', 'Lat':'lat', 'Long':'lon', 'NZS1170SiteClass':'site_class', 'Vs30_median':'Vs30',
       'Sigmaln_Vs30':'Vs30_std', 'T_median':'Tsite', 'sigmaln_T':'Tsite_std', 'Q_T':'Q_Tsite',
       'D_T':'D_Tsite', 'T_Ref':'Tsite_ref', 'Z1.0_median':'Z1.0', 'sigmaln_Z1.0':'Z1.0_std', 'Z1.0_Ref':'Z1.0_ref',
       'Z2.5_median':'Z2.5', 'sigmaln_Z2.5':'Z2.5_std', 'Z2.5_Ref':'Z2.5_ref'},inplace=True)
merged_df = sc_df.set_index(sc_df.sta).join(sta_df.set_index(sta_df.sta)[['net','elev']],how='left').reset_index(drop=True)
merged_df[['site_domain_no', 'site_domain_name', 'site_domain_type']] = merged_df.apply(lambda x: get_domains(x,shapes,wgs2nztm),axis=1)       
 
merged_final_df = merged_df[['net', 'sta', 'lat', 'lon', 'elev', 'site_class', 'Vs30', 'Vs30_std', 
       'Q_Vs30', 'Vs30_Ref', 'Tsite', 'Tsite_std', 'Q_Tsite', 'D_Tsite', 'Tsite_ref',
       'Z1.0', 'Z1.0_std', 'Q_Z1.0', 'Z1.0_ref', 'Z2.5', 'Z2.5_std', 'Q_Z2.5',
       'Z2.5_ref', 'site_domain_no']].copy()

drop_index = merged_final_df[(merged_final_df.net == 'AU') & (merged_final_df.sta == 'ARPS')].index

merged_final_df.drop(index=drop_index,inplace=True)

merged_final_df.to_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/testaroo/site_table_response.csv',index=False)      
       
# sc_df = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/site_info/site_class_metadata_Kaiser2017.csv',low_memory=False)
# sc_df.rename(columns={'Vs30':'Vs30_kaiser_perrin'},inplace=True)
# perrin_df = pd.read_excel('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/site_info/Perrin_et_al_2015_NZSEE_Vs30_SiteClass_at_SMS_Output.xls')
# perrin_df = perrin_df.rename(columns={'Station':'sta','SITECLASS':'site_class','MIDPOINT_VS':'Vs30_kaiser_perrin'})
# perrin_df['Vs30_kaiser_perrin'] = perrin_df['Vs30_kaiser_perrin'].astype('float')
# 
# nzvm_df = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/site_info/site_table_20210518_z.csv',low_memory=False)
# nzvm_df.rename(columns={'Station_Name':'sta','Z_1.0(km)':'Z1.0_NZVM','Z_2.5(km)':'Z2.5_NZVM','sigma':'Z1.0_std_NZVM'},inplace=True)
# nzvm_df['Z1.0_NZVM'] = nzvm_df['Z1.0_NZVM'] * 1000
# nzvm_df['Z2.5_NZVM'] = nzvm_df['Z2.5_NZVM'] * 1000
# 
# foster_df = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/site_info/foster_vs30points_nominal_20210602.csv')
# foster_df.rename(columns={'geology_mvn_vs30':'Vs30_foster_geology','geology_mvn_stdv':'Vs30_foster_geology_std',
# 	'mvn_vs30':'Vs30_foster_hybrid','mvn_stdv':'Vs30_foster_hybrid_std','vs30':'Vs30_foster','stdv':'Vs30_foster_std'}, inplace=True)
# 	
# nshm_df = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/site_info/NSHMSiteClassificationData_May2021.csv', low_memory = False)
# nshm_df.rename(columns={'Name':'sta','Vs30 mean':'Vs30_W21','Vs30 Std Dev':'Vs30_std_W21','T mean':'Tsite_W21',
#     'Q_T':'Q_Tsite_W21','D_T':'D_Tsite_W21','Z1.0 mean':'Z1.0_W21','Z1.0 std dev':'Z1.0_std_W21','Z2.5 std dev':'Z2.5_std_W21',
#     'Z2.5 mean':'Z2.5_W21','Q_Vs30':'Q_Vs30_W21','Q_Z1.0':'Q_Z1.0_W21','Q_Z2.5':'Q_Z2.5_W21'}, inplace=True)
# nshm_df = nshm_df.drop(['Lat','Long','NZS1170 Site Class', 'Site Class Basis', 'T Std Dev'],axis=1)
# 
# vs30_df = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/site_info/vs30_preferred_1.1.csv',low_memory = False)
# vs30_df.rename(columns={'Unnamed: 0':'sta','Source':'Vs30_preferred_model','Vs30 (m/s)':'Vs30_preferred','QVs30':'Q_Vs30'}, inplace=True)
# 
# merged_df = sta_df.set_index('sta').join(sc_df.set_index('Station'),how='left').reset_index()
# merged_df = merged_df.drop(['References '],axis=1)
# merged_df = merged_df.rename(columns={'index':'sta','Site Class':'site_class','Zb':'Z1.0_kaiser','Q_Zb':'Q_Z1.0_kaiser'})
# 
# merged_df.loc[merged_df.site_class.isnull(),['sta','site_class','Vs30_kaiser_perrin']] = merged_df.loc[
# 	merged_df.site_class.isnull(),['sta','site_class','Vs30_kaiser_perrin']].join(perrin_df[['sta',
# 	'site_class','Vs30_kaiser_perrin']].set_index('sta'),how='left', on='sta', 
# 	lsuffix='_old').drop(['site_class_old','Vs30_kaiser_perrin_old'],axis=1)
# 	
# merged_df['Z2.5'] = None
# 
# merged_df = merged_df.drop_duplicates()
# 
# merged_df = merged_df.set_index('sta').join(nzvm_df.set_index('sta'),how='left').reset_index()
# 
# merged_df = merged_df.set_index('sta').join(foster_df[['sta','Vs30_foster',
# 	'Vs30_foster_std']].set_index('sta'),how='left').reset_index()
# 	
# merged_df = merged_df.set_index('sta').join(nshm_df.set_index('sta'),how='left').reset_index()
# 	
# # merged_df = TVZ_calc(merged_df)
# merged_df[['site_domain_no', 'site_domain_name', 'site_domain_type']] = merged_df.apply(lambda x: get_domains(x,shapes,wgs2nztm),axis=1)
# 
# merged_df.loc[merged_df.Vs30_kaiser_perrin.isnull() == False,'Vs30_preferred_model'] = 'kaiser-perrin'
# merged_df.loc[merged_df.Vs30_kaiser_perrin.isnull() == False,'Vs30_preferred'] = merged_df[merged_df.Vs30_kaiser_perrin.isnull() == False].Vs30_kaiser_perrin.values
# merged_df.loc[merged_df.Vs30_kaiser_perrin.isnull(),'Vs30_preferred_model'] = 'W21'
# merged_df.loc[merged_df.Vs30_kaiser_perrin.isnull(),['Vs30_preferred','Tsite','Q_Tsite','D_Tsite']] = merged_df.loc[
#     merged_df.Vs30_kaiser_perrin.isnull(),['Vs30_W21','Tsite_W21','Q_Tsite_W21','D_Tsite_W21']].values
# merged_df.loc[merged_df.Vs30_preferred.isnull(),'Vs30_preferred_model'] = 'foster'
# merged_df.loc[merged_df.Vs30_preferred.isnull(),'Vs30_preferred'] = merged_df[merged_df.Vs30_preferred.isnull()].Vs30_foster.values
# merged_df.loc[merged_df.Vs30_preferred.isnull(),'Vs30_preferred_model'] = np.nan
# # merged_df.loc[merged_df['Z1.0'].isnull() == False,'Z_preferred_model'] = 'kaiser'
# # merged_df.loc[merged_df['Z1.0'].isnull(),'Z_preferred_model'] = 'NSHM'
# # merged_df.loc[merged_df['Z1.0'].isnull(),['Z1.0','Z2.5','Q_Z1.0','Q_Z2.5','Z1.0_std','Z2.5_std']] = merged_df.loc[
# #     merged_df['Z1.0'].isnull(),['Z1.0_NSHM','Z2.5_NSHM','Q_Z1.0_NSHM','Q_Z2.5_NSHM','Z1.0_std_NSHM',
# #     'Z2.5_std_NSHM']].values
# merged_df['Z_preferred_model'] = 'W21'
# merged_df[['Z1.0','Z2.5','Q_Z1.0','Q_Z2.5','Z1.0_std','Z2.5_std']] = merged_df[['Z1.0_W21',
#     'Z2.5_W21','Q_Z1.0_W21','Q_Z2.5_W21','Z1.0_std_W21','Z2.5_std_W21']].values
# 
# merged_df.loc[merged_df['Z1.0'].isnull(),'Z_preferred_model'] = 'NZVM'
# merged_df.loc[merged_df['Z1.0'].isnull(),['Z1.0','Z2.5','Z1.0_std']] = merged_df.loc[
#     merged_df['Z1.0'].isnull(),['Z1.0_NZVM','Z2.5_NZVM','Z1.0_std_NZVM']].values
# merged_df.loc[merged_df['Z1.0'].isnull(),'Z_preferred_model'] = np.nan
# 
# # merged_df.update(vs30_df[['sta', 'Vs30_preferred_model', 'Vs30_preferred','Q_Vs30']])
# merged_df = merged_df.merge(vs30_df[['sta', 'Vs30_preferred_model', 'Vs30_preferred','Q_Vs30']].set_index('sta'),on='sta',how='left')
# merged_df['Vs30_preferred_model'] = merged_df['Vs30_preferred_model_y']
# merged_df['Vs30_preferred'] = merged_df['Vs30_preferred_y']
# merged_df['Q_Vs30'] = merged_df['Q_Vs30_y']
# merged_df.drop(columns=['Vs30_preferred_model_y','Vs30_preferred_y','Q_Vs30_y','Vs30_preferred_model_x','Vs30_preferred_x','Q_Vs30_x'],inplace=True)
# 
# # Search for duplicate site locations. Merge data based on which site has fewer null values
# duplicated = merged_df[merged_df.duplicated(subset=['lat','lon']) == True]
# for i,duplicate in duplicated.iterrows():
#     dup_test = merged_df[(merged_df.lat == duplicate.lat) & (merged_df.lon == duplicate.lon)]
#     null_vals = dup_test.isna().sum(axis=1)
#     merged_df.loc[dup_test[null_vals == null_vals.max()].index,merged_df.drop(columns=['sta','net']).columns] =  merged_df.loc[dup_test[null_vals == null_vals.min()].index,merged_df.drop(columns=['sta','net']).columns].values
# 
# # Put Z2.5 in km
# merged_df['Z2.5'] = (merged_df['Z2.5']/1000).values
# 
# merged_final_df = merged_df[['sta', 'net', 'lat', 'lon', 'elev', 'site_class', 'Vs30_preferred',
#     'Vs30_preferred_model','Vs30_kaiser_perrin', 'Vs30_W21', 'Vs30_std_W21',
# 	'Vs30_foster','Vs30_foster_std','Tsite', 'Z_preferred_model','Z1.0', 'Z2.5', 'Z1.0_kaiser', 'Z1.0_W21', 'Z2.5_W21',
# 	'Z1.0_std_W21', 'Z2.5_std_W21', 'Z1.0_NZVM', 'Z2.5_NZVM', 'Z1.0_std_NZVM', 'Q_Vs30', 
# 	'Q_Tsite', 'D_Tsite', 'Q_Z1.0', 'Q_Z2.5', 'site_domain_no']].copy()
# 
# drop_index = merged_final_df[(merged_final_df.net == 'AU') & (merged_final_df.sta == 'ARPS')].index
# 
# merged_final_df.drop(index=drop_index,inplace=True)
# 
# merged_final_df.to_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/site_table_response.csv',index=False)