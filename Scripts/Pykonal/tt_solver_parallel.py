import pykonal
import numpy as np
import json
from pyproj import CRS
from pyproj import Transformer
import math
import os
from requests import get
from pandas import json_normalize
import time
from multiprocessing import Pool
import ray

# import pandas as pd

def wrapper(indices):
    solve_tt(*indices)


def get_elevation(lat = None, long = None):
    '''
        script for returning elevation in m from lat, long
    '''
    if lat is None or long is None: return None
    
    query = 'https://api.opentopodata.org/v1/nzdem8m?locations='+str(lat)+','+str(long)
    
    # Request with a timeout for slow responses
    r = get(query)

    # Only get the json response in case of 200 or 201
    if r.status_code == 200 or r.status_code == 201:
        elevation = json_normalize(r.json(), 'results')['elevation'].values[0]
    else: 
        elevation = None
    return elevation

def rotate(orilat, orilon, lats, lons, angle):
	"""
	Rotate a point counterclockwise by a given angle around a given origin.

	The angle should be given in radians.
	"""
	angle = np.radians(angle)
# 	ox = orilon
# 	oy = orilat
	transformer_from_latlon = Transformer.from_crs(4326, 2193) # WSG84 to New Zealand NZDG2000 coordinate transform
	transformer_to_latlon = Transformer.from_crs(2193, 4326)
	
	ox, oy = transformer_from_latlon.transform(orilat,orilon)
	px, py = transformer_from_latlon.transform(lats,lons)

	qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
	qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
	
	x = (qx-ox)/1000
	y = -(qy-oy)/1000
	
# 	stations['z'] = (stations.elevation-dx)/dx
	return x, y

# Load station information
@ray.remote
def solve_tt(station, stations, x_spacing,y_spacing, z_spacing, min_coords, max_coords, orilat, orilon, oridep, angle, input_dir):
	print('Solving traveltimes for '+str(station))

	# Load the P-velocity model
	vp_model = np.load('vmodel/vp_3.npy')
	vs_model = np.load('vmodel/vs_3.npy')
	x_int, y_int, z_int = vp_model.shape
	
	lat, lon, el = stations[station]['coords']
	
	# Some stations have filler elevations that are about 9000 meters, this searches for
	# the actual elevations.
	if el > 9000:
		print(station,el)
		el = get_elevation(lat, lon)
		print(el)
		if el == None:
			el = 0
		time.sleep(1) # Sometimes the server doesn't send back data, add sleep timer
	
	# Convert to Transverse Mercator and rotate coordinates
	sta_x_orig, sta_y_orig = rotate(orilat, orilon, lat, lon, angle)
	x = (sta_x_orig/x_spacing) - (min_coords[0]/x_spacing)
	y = (sta_y_orig/y_spacing) - (min_coords[0]/y_spacing)
	z = ((-el/1000) - min_coords[2])/z_spacing
	
	# Adjust for model spacing, keep center coordinate as origin
	sta_x = round(x)
	sta_y = round(y)
	sta_z = round(z)
	
	print(station,lat,lon,sta_x,sta_y,sta_z)
	
	# Check to see if stations are within bounds
	if sta_x < 0 or sta_x > round((max_coords[0] - min_coords[0])/x_spacing):
		print(station+" not in x-bounds")
		return
	if sta_y < 0 or sta_y > round((max_coords[1] - min_coords[1])/y_spacing):
		print(station+" not in y-bounds")
		return
	if sta_z < 0 or sta_z > round((max_coords[2] - min_coords[2])/z_spacing):
		print(station+" not in z-bounds")
		return
	# Check to see if velocity file exists
	if not os.path.exists(input_dir+'/'+station+'_P.hdf'):
		print(station)

		# Initialize the solver
		solver = pykonal.solver.PointSourceSolver(coord_sys="cartesian")
		solver.velocity.min_coords = round(min_coords[0]), round(min_coords[1]), round(min_coords[2])
		solver.velocity.node_intervals = x_spacing, y_spacing, z_spacing
		solver.velocity.npts = x_int, y_int, z_int
		solver.velocity.values = vp_model
		solver.src_loc = np.array([sta_x_orig,sta_y_orig,-el/1000])
		# Solve the system.
		solver.solve()

		print('...Solved P traveltimes for '+str(station))
		# Save traveltime solution to file
		if not os.path.exists(input_dir):
			os.mkdir(input_dir)
		solver.traveltime.to_hdf(input_dir+'/'+station+'_P.hdf')
	
		# Note: the file can be read later by entering test = pykonal.fields.read_hdf('inputs/'+station+'_P.hdf')
	else:
		print(input_dir+'/'+station+'_P.hdf already exists')

	if not os.path.exists(input_dir+'/'+station+'_S.hdf'):	
		# Initialize the solver for the S velocities
		# Initialize the solver
		solver = pykonal.solver.PointSourceSolver(coord_sys="cartesian")
		solver.velocity.min_coords = round(min_coords[0]), round(min_coords[1]), round(min_coords[2])
		solver.velocity.node_intervals = x_spacing, y_spacing, z_spacing
		solver.velocity.npts = x_int, y_int, z_int
		solver.velocity.values = vs_model
		solver.src_loc = np.array([sta_x_orig,sta_y_orig,-el/1000])

		# Solve the system.
		solver.solve()

		# Save traveltime solution to file
		if not os.path.exists(input_dir):
			os.mkdir(input_dir)
		solver.traveltime.to_hdf(input_dir+'/'+station+'_S.hdf')
	
		print('...Solved S traveltimes for '+str(station))
		# Note: the file can be read later by entering test = pykonal.fields.read_hdf('inputs/'+station+'_S.hdf')
	else:
		print(input_dir+'/'+station+'_S.hdf already exists')
	return

orilat = -41.7638 # Origin latitude
orilon = 172.9037 # Origin longitude
oridep = 0 # Origin depth
angle = 140
input_dir = '/Volumes/SeaJade2/Pykonal/tt_fine'

# stations = pd.read_csv(stfile,header=None,names=['station','latitude','longitude','elevation'],sep='\t')
# lats = stations.latitude.values
# lons = stations.longitude.values
# stations['x'],stations['y'] = rotate(orilat, orilon, lats, lons, -angle, dx)
# stations['z'] = -stations['elevation']

# Load the coordinates
coordinates = np.load('vmodel/coords_3.npy',mmap_mode='r')
min_coords = coordinates[0,0,0]
max_coords = coordinates[-1,-1,-1]
x_spacing = 3 # Spacing in the x-direction
y_spacing = 3 # Spacing in the y-direction
z_spacing = 2 # Spacing in the z-direction


# Load station file
with open('station_list.json') as f:
# with open('station_list_extra.json') as f:
	stations = json.load(f)

ray.init(num_cpus=4)

# vp_model = np.load('vmodel/vp_3.npy')
# vs_model = np.load('vmodel/vs_3.npy')
# vp_model_id = ray.put(vp_model)
# vs_model_id = ray.put(vs_model)

results_ids = []
for station in list(stations)[400::]:
	results_ids.append(solve_tt.remote(station, stations, x_spacing, y_spacing, z_spacing, min_coords,
		max_coords, orilat, orilon, oridep, angle,
		input_dir))
results = ray.get(results_ids)
print('Finished traveltime computations')
ray.shutdown()

# with Pool(1) as p:
# 	p.map(wrapper,[(station, stations, x_spacing, y_spacing, z_spacing, min_coords,
# 		max_coords, orilat, orilon, oridep, angle,
# 		input_dir) for station in stations])

		