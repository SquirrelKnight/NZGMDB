import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import time
from multiprocessing import Pool
import ray

# This program is setup to use a text file as input. It requires the header to be
# in the second row with depth, Vp, and Vs. The program will
# extrapolate between depth points and interpolate between them to create a smoothed
# velocity structure. This can be used as an input for Pykonal. The results will be
# written out to .npy files.
#
# This version has implemented parallel computations for large grids. Results are quickly
# written to the numpy array.

def rotate(orilat, orilon, lats, lons, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    from pyproj import Transformer
    import math
    import numpy as np

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
    return x, y

def rotate_back(orilat, orilon, xs, ys, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    from pyproj import Transformer
    import math
    import numpy as np

    angle = np.radians(angle)
    # 	ox = orilon
    # 	oy = orilat
    transformer_from_latlon = Transformer.from_crs(4326, 2193)  # WSG84 to New Zealand NZDG2000 coordinate transform
    transformer_to_latlon = Transformer.from_crs(2193, 4326)

    ox, oy = transformer_from_latlon.transform(orilat, orilon)
    px = ox + xs * 1000
    py = oy - ys * 1000
    # 	px, py = transformer_from_latlon.transform(lats,lons)

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    lats, lons = transformer_to_latlon.transform(qx, qy)
    # 	stations['z'] = (stations.elevation-dx)/dx
    return lats, lons

def test_wrapper(indices):
    p,s,l,ik = test(*indices)
    return p,s,l,ik

@ray.remote
def test_2(fn_p,fn_s,i,z_int,ii):
	p = []
	s = []
	l = []
	ijks = []
	for ik,k in enumerate(z_int):
		for ij,j in enumerate(y_int):
			print(i,j,k)
			p.append(fn_p([i,j,k]))
			s.append(fn_s([i,j,k]))
			l.append([i,j,k])
			ijks.append([ii,ij,ik])
	return p,s,l,ijks

def mk_3d_model(x_spacing,y_spacing,z_spacing,input_file):   #### Fix inputs for nz, nzmin, and nzmax
	start_model = pd.read_csv(input_file,delim_whitespace=True,header=1)
	df = start_model
	df.rename(columns={'Vp':'vp','Vs':'vs','x(km)':'x','y(km)':'y','Depth(km_BSL)':'z'},inplace=True)
	depth_min = df.z.min()
	depth_max = df.z.max()
	x_unique = df['x'].unique()
	y_unique = df['y'].unique()
	z_unique = df['z'].unique()
	vp = df.vp.values
	vs = df.vs.values
	Vp = np.zeros((len(x_unique),len(y_unique),len(z_unique)))
	Vs = np.zeros((len(x_unique),len(y_unique),len(z_unique)))
	ii = 0
	for k in range(len(z_unique)):
		for j in range(len(y_unique)):
			for i in range(len(x_unique)):
				Vp[i,j,k] = vp[ii]
				Vs[i,j,k] = vs[ii]
				ii += 1
	fn_p = RegularGridInterpolator((x_unique,y_unique,z_unique), Vp, bounds_error=False, fill_value=None)
	fn_s = RegularGridInterpolator((x_unique,y_unique,z_unique), Vs, bounds_error=False, fill_value=None)
	
	x_int = np.arange(int(x_unique[0]),int(x_unique[-1]),x_spacing)
	y_int = np.arange(int(y_unique[0]),int(y_unique[-1]),y_spacing)
	z_int = np.arange(int(z_unique[0]),int(z_unique[-1]),z_spacing)
	
	Vp_new = np.zeros((x_int.size,y_int.size,z_int.size))
	Vs_new = np.zeros((x_int.size,y_int.size,z_int.size))
	loc = np.zeros((x_int.size,y_int.size,z_int.size,3))

	ray.init()
	start_time = time.time()
	result_ids = []
	ii = 0
	for i in x_int:
		result_ids.append(test_2.remote(fn_p,fn_s,i,z_int,ii))
		ii += 1
	results = ray.get(result_ids)
	ray.shutdown()

	for result in results:
		for vel_p in result[0]:
			print(vel_p)
	vel_ps = [vel_p for result in results for vel_p in result[0]]
	vel_ss = [vel_s for result in results for vel_s in result[1]]
	ls = [l for result in results for l in result[2]]
	ijks = [ijk for result in results for ijk in result[3]]
	for i in range(len(vel_ps)):
		Vp_new[tuple(ijks[i])] = vel_ps[i]
		Vs_new[tuple(ijks[i])] = vel_ss[i]
		loc[tuple(ijks[i])] = np.array(ls[i])
	print("--- %s seconds ---" % (time.time() - start_time))
		
	return Vp_new, Vs_new, loc
	
### The input file should be a standard txt file from the Eberhart-Phillips model
input_file = "/Volumes/SeaJade 2 Backup/NZ/Pykonal/vmodel/nzw2p2/vlnzw2p2dnxyzltln.tbl.txt"
x_spacing = 10 # Spacing in the x-direction
y_spacing = 10 # Spacing in the y-direction
z_spacing = 4 # Spacing in the z-direction
# x_spacing = 3 # Spacing in the x-direction
# y_spacing = 3 # Spacing in the y-direction
# z_spacing = 2 # Spacing in the z-direction

Vp_new, Vs_new, loc = mk_3d_model(x_spacing,y_spacing,z_spacing,input_file)

np.save('vp_3.npy',Vp_new)
np.save('vs_3.npy',Vs_new)  
np.save('coords_3.npy',loc)


# orilat = -41.7638 # Origin latitude
# orilon = 172.9037 # Origin longitude
# oridep = 0 # Origin depth
# angle = 140
# 
# vmod_file = '/Volumes/SeaJade 2 Backup/NZ/Pykonal/vmodel/ni/tomography_model_crust.xyz'
# vmod_file_2 = '/Volumes/SeaJade 2 Backup/NZ/Pykonal/vmodel/ni/tomography_model_mantle.xyz'
# vmod_file_3 = '/Volumes/SeaJade 2 Backup/NZ/Pykonal/vmodel/ni/tomography_model_shallow.xyz'
# file_list = [vmod_file,vmod_file_2,vmod_file_3]
# # 	+ Header: First 4 lines of each file define the following quantities
# # 		x_min y_min z_min x_max y_max z_max
# # 		dx dy dz
# # 		nx ny nz
# # 		vp_min vp_max vs_min vs_max rho_min rho_max
# vmod_df = pd.concat([pd.read_csv(file,delim_whitespace=True,header=None,skiprows=4,
#     names=['x','y','z','vp','vs','rho','Qp','Qs']) for file in file_list])
# vmod_df['z'] = vmod_df['z'] * -1
# 
# # Transform coordinates to NZW2.2
# from pyproj import Transformer
# 
# transformer_to_latlon = Transformer.from_crs(27260,4326) # NZGD49 to WSG84
# lat, lon = transformer_to_latlon.transform(vmod_df.x,vmod_df.y)
# 
# x,y = rotate(orilat, orilon, lat, lon, angle)
# 
# vmod_df['x'] = x
# vmod_df['y'] = y
# 
# vmod_df.sort_values(['z','y','x'],inplace=True)
# x = (vmod_df.x).values
# y = (vmod_df.y).values
# z = (vmod_df.z / 1000).values
# vp = vmod_df.vp.values / 1000
# vs = vmod_df.vs.values / 1000
# 
# depth_min = vmod_df.z.min()
# depth_max = vmod_df.z.max()
# x_unique = vmod_df['x'].unique()
# y_unique = vmod_df['y'].unique()
# z_unique = vmod_df['z'].unique()
# Vp = np.zeros((len(x_unique),len(y_unique),len(z_unique)))
# Vs = np.zeros((len(x_unique),len(y_unique),len(z_unique)))
# ii = 0
# for k in range(len(z_unique)):
# 	for j in range(len(y_unique)):
# 		for i in range(len(x_unique)):
# 			Vp[i,j,k] = vp[ii]
# 			Vs[i,j,k] = vs[ii]
# 			ii += 1
