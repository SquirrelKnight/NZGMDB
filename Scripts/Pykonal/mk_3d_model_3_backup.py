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

def test_wrapper(indices):
    p,s,l,ik = test(*indices)
    return p,s,l,ik

def test(i,j,k,depth_min,z_spacing):
    p = fn_p([i,j,k])
    s = fn_s([i,j,k])
    l = [i,j,k]
    ik = int((k - depth_min)/z_spacing)
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
	fn_p = RegularGridInterpolator((x_unique,y_unique,z_unique), Vp)
	fn_s = RegularGridInterpolator((x_unique,y_unique,z_unique), Vs)
	
	x_int = int(np.ceil((x_unique[-1]-int(x_unique[0]))/x_spacing))
	y_int = int(np.ceil((y_unique[-1]-int(y_unique[0]))/y_spacing))
	z_int = int(np.ceil((z_unique[-1]-int(z_unique[0]))/z_spacing))
	x_int = np.arange(int(x_unique[0]),int(x_unique[-1]),x_spacing)
	y_int = np.arange(int(y_unique[0]),int(y_unique[-1]),y_spacing)
	z_int = np.arange(int(z_unique[0]),int(z_unique[-1]),z_spacing)
	
	Vp_new = np.zeros((x_int.size,y_int.size,z_int.size))
	Vs_new = np.zeros((x_int.size,y_int.size,z_int.size))
	loc = np.zeros((x_int.size,y_int.size,z_int.size,3))
	ik = 0
	start_time = time.time()
# 	for k in range((int(z_unique[0])),int(z_unique[-1]),z_spacing):
	ik = 0
	ij = 0
	for j in y_int:
		ii = 0
		print(ii,ij)
		for i in x_int:
			print(ii,ij)
	#             print(ii,ij,ik)
	#             Vp_new[ii,ij,ik],Vs_new[ii,ij,ik],loc[ii,ij,ik] = test(i,j,k)
			with Pool(8) as p:
	#         p = Pool(4)
				results = p.map(test_wrapper,[(i,j,k,depth_min,z_spacing) for k in z_int])
			Vps = [result[0] for result in results]
			Vss = [result[1] for result in results]
			locs = [result[2] for result in results]
			iks = [result[3] for result in results]
# 			for idx,ik in enumerate(iks):
			Vp_new[ii,ij,iks] = np.ndarray.flatten(np.array(Vps))
			Vs_new[ii,ij,iks] = np.ndarray.flatten(np.array(Vss))
			loc[ii,ij,iks] = np.array(locs)
			ii += 1
		ij += 1
	print("--- %s seconds ---" % (time.time() - start_time))

	ray.init()
	start_time = time.time()
	result_ids = []
# 	ik = 0
# 	for k in z_int[0:1]:
# 	ij = 0
# 	for j in y_int:
	ii = 0
	for i in x_int:
# 				p = fn_p([i,j,k])
# 				s = fn_s([i,j,k])
# 				l = [i,j,k]
		
		result_ids.append(test_2.remote(fn_p,fn_s,i,z_int,ii))
# 				p,s,l = test_2(fn_p,fn_s,i,j,k)
		
# 			for idx,ik in enumerate(iks):
# 			Vp_new[ii,ij,ik] = p
# 			Vs_new[ii,ij,ik] = s
# 			loc[ii,ij,ik] = l
		ii += 1
# 		ij += 1
# 		ik += 1
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
input_file = "/Volumes/SeaJade 2 Backup/NZ/Pykonal/vmodel/vlnzw2p2dnxyzltln.tbl.txt"
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

