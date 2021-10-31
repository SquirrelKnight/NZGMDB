import numpy as np
import pandas as pd
from pandarallel import pandarallel		# conda install -c bjrn pandarallel
import fiona							# conda install fiona
from pyproj import Transformer			# conda install pyproj

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

	
event_cat = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/testaroo/for_elena_tectclass.csv',low_memory=False)

shape = fiona.open("/Users/jesse/Downloads/TectonicDomains/TectonicDomains_Feb2021_8_NZTM.shp")

# Writes layers of the shape file to a shapes array; prevents read errors if accessing
# the shape file directly
shapes = []
for layer in shape:
	domain_no = layer['properties']['Domain_No']
	domain_name = layer['properties']['DomainName']
	print(domain_no,domain_name)
	shapes.append(layer)

wgs2nztm = Transformer.from_crs(4326, 2193, always_xy=True)

pandarallel.initialize(nb_workers=4) # Customize the number of parallel workers
event_cat[['domain_no', 'domain_name', 'domain_type']] = event_cat.parallel_apply(lambda x: get_domains(x,shapes,wgs2nztm),axis=1)
event_cat.drop(columns='domain_name',inplace=True)

event_cat.to_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/testaroo/for_elena_tectclass_domain.csv',index=False)