# This program relocates events using a grid search method. It loads in event and phase
# information and parallelizes locations. Each location is determined by find the maximum
# intersection of arrivals on a coarse grid, which is then improved by progressively
# finer grids. Travel time grids are generated with the program Pykonal, but Pykonal
# is not required for this program to function.
#
# This version of the program implements the inclusion of better error evaluation based
# on the percentile of standard deviations in the nodes surrounding the maximum intersection.
# It also implements dynamic loading of finer-spaced travel time grids generated with Pykonal.
# This is useful in situations where the source velocity model has more finely spaced nodes
# than the coarse travel time grid.
#
# Further, this version will calculate coarser travel time grids based on a zoom factor.
# These coarser grids will be used for an initial search to further save memory and allow
# for faster computations for large events with many phases.
#
# Optimizes loading of traveltime data to cut processing times, especially with larger
# data sets. Fixed the implementation of error estimations so that the xc and yc values
# should now be correct.
#
# Redesigned maxiscan to be more efficient.

import pandas as pd
import numpy as np
# from geopy.distance import geodesic
import datetime as dt
import os
from scipy.ndimage import zoom
import sys
import ray
import scipy

from scipy import ndimage as ndi

# def uncertainty_calc(full,all_estimates,est_std,grid_x_spacing,grid_y_spacing,grid_z_spacing,fine_x_spacing,fine_y_spacing,fine_z_spacing,p_x_mod,p_y_mod,
#     p_z_mod,ps_x_mod,ps_y_mod,ps_z_mod,x_min,y_min,z_min,p_x_min,p_x_max,p_y_min,p_y_max,p_z_min,p_z_max):
#     from scipy import ndimage as ndi
#     import numpy as np
#     from skimage.measure import EllipseModel
#     
#     dfd = full - 1
#     dfn = dfd
#     fs = (all_estimates[:,1] ** 2) / (est_std ** 2)
#     p_vals = 1-scipy.stats.f.cdf(fs, dfn, dfd)
#        
#     ps_out = all_estimates[np.where(p_vals >= 0.05)][:,1:5]
#     ps_out[:,1] = (ps_out[:,1] - ps_x_mod) * grid_x_spacing + p_x_mod * fine_x_spacing + x_min
#     ps_out[:,2] = (ps_out[:,2] - ps_y_mod) * grid_y_spacing + p_y_mod * fine_y_spacing + y_min
#     ps_out[:,3] = (ps_out[:,3] - ps_z_mod) * grid_z_spacing + p_z_mod * fine_z_spacing + z_min
#  
# #     fig = plt.figure(figsize = (8,8))
# #     ax = plt.axes(projection='3d')
# #     ax.grid()
# #     ax.scatter(ps_out[:,1],ps_out[:,2],ps_out[:,3],c=ps_out[:,0],cmap = plt.cm.viridis)
# # #     ax.scatter(p_x_out,p_y_out,p_z_out,c='r',marker='*',s=200)
# #     ax.invert_zaxis()
# #     plt.show()        
# 
#     ell = EllipseModel()
#     if ps_out[:, 1].std() > 0 and ps_out[:, 2].std() > 0:
#         ell.estimate(np.unique(ps_out[:,1:3],axis=0))
#     if ell.params:
#         xc, yc, a, b, theta = ell.params
#         # Check to see if a or b axes are too large. This can happen when fitting an ellipse
#         # to too few data.
#         if a  > np.sqrt(((ps_out[:,1].max() - ps_out[:,1].min()) * grid_x_spacing) ** 2 \
#             + ((ps_out[:,2].max() - ps_out[:,2].min()) * grid_y_spacing) ** 2) or \
#             b  > np.sqrt(((ps_out[:,1].max() - ps_out[:,1].min()) * grid_x_spacing) ** 2 \
#             + ((ps_out[:,2].max() - ps_out[:,2].min()) * grid_y_spacing) ** 2):
#             x_err, y_err, z_err = ps_out[:, 1].std(), ps_out[:, 2].std(), ps_out[:,3].std()
#             xc = np.mean(ps_out[:, 1])
#             yc = np.mean(ps_out[:, 2])
#             zc = np.mean(ps_out[:, 3])
#             if x_err > y_err:
#                 theta = np.pi / 2
#             else:
#                 theta = 0
#         else:
#             zc = np.mean(ps_out[:,3])
#         
#             z_err = ps_out[:,3].std()
#             x_err = a
#             y_err = b
#         
#     else:
#         x_err, y_err, z_err = ps_out[:, 1].std(), ps_out[:, 2].std(), ps_out[:,3].std()
#         xc = np.mean(ps_out[:, 1])
#         yc = np.mean(ps_out[:, 2])
#         zc = np.mean(ps_out[:, 3])
#         if x_err > y_err:
#             theta = np.pi / 2
#         else:
#             theta = 0
#         
#     return ps_out, xc, yc, zc, x_err, y_err, z_err, theta
def uncertainty_calc(full,stds,est_std,grid_x_spacing,grid_y_spacing,grid_z_spacing,fine_x_spacing,fine_y_spacing,fine_z_spacing,p_x_mod,p_y_mod,
    p_z_mod,ps_x_mod,ps_y_mod,ps_z_mod,x_min,y_min,z_min,p_x_min,p_x_max,p_y_min,p_y_max,p_z_min,p_z_max):
    from scipy import ndimage as ndi
    import numpy as np
    from skimage.measure import EllipseModel
    
    dfd = full - 1
    dfn = dfd
    fs = stds ** 2 / est_std ** 2
    p_vals = 1-scipy.stats.f.cdf(fs, dfn, dfd)
    
    ps_in = np.where(p_vals >= 0.05)
    ps_out = np.zeros([4,ps_in[0].size])   
    ps_out[0] = stds[ps_in]
    ps_out[1] = (ps_in[0] - ps_x_mod) * grid_x_spacing + p_x_mod * fine_x_spacing + x_min
    ps_out[2] = (ps_in[1] - ps_y_mod) * grid_y_spacing + p_y_mod * fine_y_spacing + y_min
    ps_out[3] = (ps_in[2] - ps_z_mod) * grid_z_spacing + p_z_mod * fine_z_spacing + z_min
#     fig = plt.figure(figsize = (8,8))
#     ax = plt.axes(projection='3d')
#     ax.grid()
#     ax.scatter(ps_out[:,1],ps_out[:,2],ps_out[:,3],c=ps_out[:,0],cmap = plt.cm.viridis)
# #     ax.scatter(p_x_out,p_y_out,p_z_out,c='r',marker='*',s=200)
#     ax.invert_zaxis()
#     plt.show()        

    ell = EllipseModel()
    if ps_out[1].std() > 0 and ps_out[2].std() > 0:
        ell.estimate(np.array([ps_out[1],ps_out[2]]).T)
    if ell.params:
        xc, yc, a, b, theta = ell.params
        # Check to see if a or b axes are too large. This can happen when fitting an ellipse
        # to too few data.
        if a  > np.sqrt(((ps_out[1].max() - ps_out[1].min()) * grid_x_spacing) ** 2 \
            + ((ps_out[2].max() - ps_out[2].min()) * grid_y_spacing) ** 2) or \
            b  > np.sqrt(((ps_out[1].max() - ps_out[1].min()) * grid_x_spacing) ** 2 \
            + ((ps_out[2].max() - ps_out[2].min()) * grid_y_spacing) ** 2):
            x_err, y_err, z_err = ps_out[1].std(), ps_out[2].std(), ps_out[3].std()
            xc = np.mean(ps_out[1])
            yc = np.mean(ps_out[2])
            zc = np.mean(ps_out[3])
            if x_err > y_err:
                theta = np.pi / 2
            else:
                theta = 0
        else:
            zc = np.mean(ps_out[3])
        
            z_err = ps_out[3].std()
            x_err = a
            y_err = b
        
    else:
        x_err, y_err, z_err = ps_out[1].std(), ps_out[2].std(), ps_out[3].std()
        xc = np.mean(ps_out[1])
        yc = np.mean(ps_out[2])
        zc = np.mean(ps_out[3])
        if x_err > y_err:
            theta = np.pi / 2
        else:
            theta = 0
        
    return ps_out, xc, yc, zc, x_err, y_err, z_err, theta

def local_maxima_3D(data, order=1):
    """Detects local maxima in a 3D array

    Parameters
    ---------
    data : 3d ndarray
    order : int
        How many points on each side to use for the comparison

    Returns
    -------
    coords : ndarray
        coordinates of the local maxima
    values : ndarray
        values of the local maxima
    """
    size = 1 + 2 * order
    footprint = np.ones((size, size, size))
    footprint[order, order, order] = 0

    filtered = ndi.maximum_filter(data, footprint=footprint)
    mask_local_maxima = data > filtered
    coords = np.asarray(np.where(mask_local_maxima)).T
    values = data[mask_local_maxima]

    return coords, values


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


def handshakes(x):
    x = ((x - 1) * x) / 2
    return int(x)


def max_tt_diff(ptraveltimes, straveltimes):
    max_diffs = np.zeros(handshakes(ptraveltimes.shape[0]))
    ii = 0
    for i in range(ptraveltimes.shape[0] - 1):
        for j in range(i + 1, ptraveltimes.shape[0]):
            #             max_diffs[ii] = np.max((straveltimes[j]-ptraveltimes[j]) - (straveltimes[i]-ptraveltimes[i]))
            max_diffs[ii] = np.max(straveltimes[j] - straveltimes[i])
            progress(ii, handshakes(ptraveltimes.shape[0]), status='')
            ii += 1
    return max_diffs.max()


def maxiscan_old(pstation, ps_station, ps_arrival, ptraveltimes, straveltimes, terr, maxi):
    start = timeit.time()
    for j in range(0, len(ps_station) - 1):
        for k in range(j + 1, len(ps_station)):
            obdiff = ps_arrival[j] - ps_arrival[k]
            if j < len(pstation):
                t1 = ptraveltimes[int(ps_station[j])]
            else:
                t1 = straveltimes[int(ps_station[j])]
            if k < len(pstation):
                t2 = ptraveltimes[int(ps_station[k])]
            else:
                t2 = straveltimes[int(ps_station[k])]
            #                 vk = 's'
            thdiff = t1 - t2
            maxi[np.abs(thdiff - obdiff) < terr * (1 / 3)] += 5
            maxi[(np.abs(thdiff - obdiff) >= terr * (1 / 3)) & (np.abs(thdiff - obdiff) < terr * 0.5)] += 4
            maxi[(np.abs(thdiff - obdiff) >= terr * 0.5) & (np.abs(thdiff - obdiff) < terr)] += 3
            maxi[(np.abs(thdiff - obdiff) >= terr) & (np.abs(thdiff - obdiff) < terr * 2)] += 2
            maxi[(np.abs(thdiff - obdiff) >= terr * 2) & (np.abs(thdiff - obdiff) < terr * 3)] += 1
    print(timeit.time() - start)
    return maxi

def maxiscan(pstation, ps_station, ps_arrival, ptraveltimes, straveltimes, terr, maxi):
#     start = timeit.time()
    for j in range(0, len(ps_station) - 1):
        for k in range(j + 1, len(ps_station)):            
            obdiff = ps_arrival[j] - ps_arrival[k]
            if j < len(pstation):
                t1 = ptraveltimes[int(ps_station[j])]
            else:
                t1 = straveltimes[int(ps_station[j])]
            if k < len(pstation):
                t2 = ptraveltimes[int(ps_station[k])]
            else:
                t2 = straveltimes[int(ps_station[k])]
            #                 vk = 's'
            thdiff = np.subtract(t1,t2)
            abs_diff = np.abs(np.subtract(thdiff,obdiff))
            maxi[abs_diff < terr * (1 / 3)] += 5
            maxi[(abs_diff >= terr * (1 / 3)) & (abs_diff < terr * 0.5)] += 4
            maxi[(abs_diff >= terr * 0.5) & (abs_diff < terr)] += 3
            maxi[(abs_diff >= terr) & (abs_diff < terr * 2)] += 2
            maxi[(abs_diff >= terr * 2) & (abs_diff < terr * 3)] += 1
#     print(timeit.time() - start)
    return maxi

def MAXI_locate_3D(event_no, ponsets, sonsets, ptraveltimes, straveltimes, lowq, highq, Q_threshold,
                   terr, outlier, x_spacing, y_spacing, z_spacing, coarsest_x_spacing, coarsest_y_spacing, coarsest_z_spacing, 
                   x_min, y_min, z_min, phase_threshold, tt_fine_dir, sta_cat, zoom_factor, std_factor):
    """ Computes the location of event hypocenters in 3-D space

    Parameters
    ----------
    event_no : str
        Event identification number
    ponsets : ndarray
        Onset times for P phases
    sonsets : ndarray
        Onset times for S phases
    ptraveltimes : 3D numpy array
        Travel time grids for P phases
    straveltimes : 3D numpy array
        Travel time grids for S phases
    lowq : float
        Low quality filter
    highq : float
        High quality filter
    Q_threshold : float
        Q-value minimum threshold
    terr : float
        Time error (in seconds)
    outlier : float
        Value to determine outlier stations (in seconds)
    x_spacing:  int
        X spacing of the initial travel time grids.
    y_spacing : int
        Y spacing of the initial travel time grids
    z_spacing : int
        Z spacing of the initial travel time grids
    z_min : int
        Minimum depth of the travel time grid coordinate system (in km)
    phase_threshold : int
        Minimum number of phases allowed to compute hypocenters
    tt_fine_dir : str
        Directory of the finely spaced travel time grids
    sta_cat : Pandas DataFrame
        Pandas DataFrame of the stations used for an event.

    Returns
    -------
    origin_data : 2D array
        Array of event, origin, and arrival information for an individual event
    """
    import math
    import h5py
    from skimage.measure import EllipseModel

    total = len(ponsets)
    events = []
    catalog = []
    arrivals = []
    unc_grid = []

    for i in range(0, total):

        p = ponsets[i]
        s = sonsets[i]

        PandS = 0
        for j in range(0, len(p)):
            if p[j] > 0 and s[j] > 0:
                PandS = PandS + 1

        positivep = 0
        parrival = []
        pstation = []
        num = 0
        for j in p:
            if j > 0:
                parrival.append(j)
                pstation.append(num)
                positivep = positivep + 1
            num = num + 1

        positives = 0
        sarrival = []
        sstation = []
        num = 0
        for j in s:
            if j > 0:
                sarrival.append(j)
                sstation.append(num)
                positives = positives + 1
            num = num + 1

        pstation = np.array(pstation)
        sstation = np.array(sstation)
        parrival = np.array(parrival)
        sarrival = np.array(sarrival)

        phase_count = len(pstation) + len(sstation)
        PandS = len(np.unique(np.append(pstation, sstation)))

        if phase_count < phase_threshold:
            print('No.' + str(event_no) + ' not enough phases \n')
            events.append([-1, -1, -1, -1, -1, -1])
            continue

        full = positives + positivep
        MAXI_ceil = full * (full - 1) / 2
        ps_station = np.concatenate([pstation, sstation]).astype(int)
        ps_arrival = np.concatenate([parrival, sarrival])

        # Make the starting terr broad to get a general idea of where the eq is, should
        # multiply terr by two, assuming a grid spacing greater
        # than 2 km.
        terr_mod = np.sqrt(np.min([x_spacing, y_spacing, z_spacing]))
        start_terr = terr_mod * terr

        maxi = np.zeros(ptraveltimes[0].shape)
        maxi = maxiscan(pstation, ps_station, ps_arrival, ptraveltimes, straveltimes, terr * 2, maxi)
        m = maxi.max()
        Q = m / MAXI_ceil / 5

        p_x, p_y, p_z = np.where(maxi == m)
        x_std, y_std, z_std = np.std(p_x) * coarsest_x_spacing, np.std(p_y) * coarsest_y_spacing, np.std(p_z) * coarsest_z_spacing
        # Use the median initial values to filter out outlier stations
        p_x_filt, p_y_filt, p_z_filt = int(np.median(p_x)), int(np.median(p_y)), int(np.median(p_z))
           
        # If the event is initially located in a negative depth, search down for the most
        # likely location
        if int(np.median(p_z) * coarsest_z_spacing) + z_min < 0:
            for ii, j in enumerate(np.arange(z_min, 100, coarsest_z_spacing)):
                if j > 0:
                    z_min_maxi = ii
                    break
            m = maxi[:, :, z_min_maxi:].max()
            p_x, p_y, p_z = np.where(maxi[:, :, z_min_maxi:] == m)
            p_z = p_z + z_min_maxi
            x_std, y_std, z_std = np.std(p_x) * coarsest_x_spacing, np.std(p_y) * coarsest_y_spacing, np.std(p_z) * coarsest_z_spacing
            p_x_filt, p_y_filt, p_z_filt = int(np.median(p_x)), int(np.median(p_y)), int(np.median(p_z))

        # Pull out coordinates to use for error estimation
        maxi_x_min = int(p_x.min()) - 1 if int(p_x.min()) - 1 > 0 else 0
        maxi_y_min = int(p_y.min()) - 1 if int(p_y.min()) - 1 > 0 else 0
        maxi_z_min = int(p_z.min()) - 1 if int(p_z.min()) - 1 > 0 else 0
        maxi_x_max = int(p_x.max()) + 2 if int(p_x.max()) + 2 < maxi.shape[0] else maxi.shape[0]
        maxi_y_max = int(p_y.max()) + 2 if int(p_y.max()) + 2 < maxi.shape[1] else maxi.shape[1]
        maxi_z_max = int(p_z.max()) + 2 if int(p_z.max()) + 2 < maxi.shape[2] else maxi.shape[2]
        maxi_sub = maxi[maxi_x_min:maxi_x_max, maxi_y_min:maxi_y_max, maxi_z_min:maxi_z_max]
        maxi_sub_ind = np.where(maxi_sub >= m - maxi.std() * std_factor)
        maxi_sub_length = len(maxi_sub_ind[0])
        maxi_sub_start = 0

        while maxi_sub_length > maxi_sub_start:
            maxi_sub_start = maxi_sub_length
            maxi_x_min = maxi_x_min - 1 if maxi_x_min - 1 > 0 else 0
            maxi_y_min = maxi_y_min - 1 if maxi_y_min - 1 > 0 else 0
            maxi_z_min = maxi_z_min - 1 if maxi_z_min - 1 > 0 else 0
            maxi_x_max = maxi_x_max + 1 if maxi_x_max + 1 < maxi.shape[0] else maxi.shape[0]
            maxi_y_max = maxi_y_max + 1 if maxi_y_max + 1 < maxi.shape[1] else maxi.shape[1]
            maxi_z_max = maxi_z_max + 1 if maxi_z_max + 1 < maxi.shape[2] else maxi.shape[2]
            maxi_sub = maxi[maxi_x_min:maxi_x_max, maxi_y_min:maxi_y_max, maxi_z_min:maxi_z_max]
            maxi_sub_ind = np.where(maxi_sub >= m - maxi.std() * std_factor)
#             maxi_sub_ind = np.where(((maxi_sub - m) ** 2) / m <= 1)
            maxi_sub_length = len(maxi_sub_ind[0])

        p_x_min = maxi_sub_ind[0].min() + maxi_x_min - 3
        p_x_max = maxi_sub_ind[0].max() + maxi_x_min + 4
        p_y_min = maxi_sub_ind[1].min() + maxi_y_min - 3
        p_y_max = maxi_sub_ind[1].max() + maxi_y_min + 4
        p_z_min = maxi_sub_ind[2].min() + maxi_z_min - 3
        p_z_max = maxi_sub_ind[2].max() + maxi_z_min + 4

        if p_x_min <= 0:
            p_x_min = 0
        if p_y_min <= 0:
            p_y_min = 0
        if p_z_min <= 0:
            p_z_min = 0

        if p_x_max >= ptraveltimes.shape[1]:
            p_x_max = ptraveltimes.shape[1]
        if p_y_max >= ptraveltimes.shape[2]:
            p_y_max = ptraveltimes.shape[2]
        if p_z_max >= ptraveltimes.shape[3]:
            p_z_max = ptraveltimes.shape[3]
        
        # Check the min and max for p_z to see if it is poorly constrained. Events that
        # are very poorly constrained will consume too much memory and take too long to
        # compute, and are rejected. This value is set to 400 by default. Remove stations
        # and then test again. If it is still bad, end computations.
        if (p_z_max - p_z_min) * coarsest_z_spacing > 400 or \
            (p_x_max - p_x_min) * coarsest_x_spacing > 400 or \
            (p_y_max - p_y_min) * coarsest_y_spacing > 400:
            print('No.' + str(event_no) + ' event is poorly constrained, attempting to \
            excise poor picks. \n')

            estimatetimes = []
            for j in range(0, len(pstation)):
                t = parrival[j] - ptraveltimes[pstation[j],p_x_filt,p_y_filt,p_z_filt]
                estimatetimes.append(t)

            for j in range(0, len(sstation)):
                t = sarrival[j] - straveltimes[sstation[j],p_x_filt,p_y_filt,p_z_filt]
                estimatetimes.append(t)

            eventtime0 = np.median(estimatetimes)

            # Remove outlier stations
            for j in range(0, len(pstation)):
                ot = parrival[j] - eventtime0
                tht = ptraveltimes[pstation[j], p_x_filt, p_y_filt, p_z_filt]
#                 print(abs(ot - tht))
                if abs(ot - tht) > outlier * 2:
                    pstation[j] = -1

            for j in range(0, len(sstation)):
                ot = sarrival[j] - eventtime0
                tht = straveltimes[sstation[j], p_x_filt, p_y_filt, p_z_filt]
    #             print(abs(ot - tht))
                if abs(ot - tht) > outlier * 2:
                    sstation[j] = -1
            parrival = parrival[pstation >= 0]
            pstation = pstation[pstation >= 0]
            sarrival = sarrival[sstation >= 0]
            sstation = sstation[sstation >= 0]
            ps_station = np.concatenate([pstation, sstation]).astype(int)
            ps_arrival = np.concatenate([parrival, sarrival])
            p_count = len(parrival)
            s_count = len(sarrival)
            phase_count = p_count + s_count

            full = phase_count
            MAXI_ceil = full * (full - 1) / 2
            maxi = np.zeros(ptraveltimes[0].shape)
            maxi = maxiscan(pstation, ps_station, ps_arrival, ptraveltimes, straveltimes, terr * 2, maxi)
            m = maxi.max()
            Q = m / MAXI_ceil / 5

            p_x, p_y, p_z = np.where(maxi == m)
            x_std, y_std, z_std = np.std(p_x) * coarsest_x_spacing, np.std(p_y) * coarsest_y_spacing, np.std(p_z) * coarsest_z_spacing
            # Use the median initial values to filter out outlier stations
            p_x_filt, p_y_filt, p_z_filt = int(np.median(p_x)), int(np.median(p_y)), int(np.median(p_z))
           
            # If the event is initially located in a negative depth, search down for the most
            # likely location
            if int(np.median(p_z) * coarsest_z_spacing) + z_min < 0:
                for ii, j in enumerate(np.arange(z_min, 100, coarsest_z_spacing)):
                    if j > 0:
                        z_min_maxi = ii
                        break
                m = maxi[:, :, z_min_maxi:].max()
                p_x, p_y, p_z = np.where(maxi[:, :, z_min_maxi:] == m)
                p_z = p_z + z_min_maxi
                x_std, y_std, z_std = np.std(p_x) * coarsest_x_spacing, np.std(p_y) * coarsest_y_spacing, np.std(p_z) * coarsest_z_spacing
                p_x_filt, p_y_filt, p_z_filt = int(np.median(p_x)), int(np.median(p_y)), int(np.median(p_z))

            # Pull out coordinates to use for error estimation
            maxi_x_min = int(p_x.min()) - 1 if int(p_x.min()) - 1 > 0 else 0
            maxi_y_min = int(p_y.min()) - 1 if int(p_y.min()) - 1 > 0 else 0
            maxi_z_min = int(p_z.min()) - 1 if int(p_z.min()) - 1 > 0 else 0
            maxi_x_max = int(p_x.max()) + 2 if int(p_x.max()) + 2 < maxi.shape[0] else maxi.shape[0]
            maxi_y_max = int(p_y.max()) + 2 if int(p_y.max()) + 2 < maxi.shape[1] else maxi.shape[1]
            maxi_z_max = int(p_z.max()) + 2 if int(p_z.max()) + 2 < maxi.shape[2] else maxi.shape[2]
            maxi_sub = maxi[maxi_x_min:maxi_x_max, maxi_y_min:maxi_y_max, maxi_z_min:maxi_z_max]
            maxi_sub_ind = np.where(maxi_sub >= m - maxi.std() * std_factor)
    #         maxi_sub_ind = np.where(((maxi_sub - m) ** 2) / m <= 1)
            maxi_sub_length = len(maxi_sub_ind[0])
            maxi_sub_start = 0

            while maxi_sub_length > maxi_sub_start:
                maxi_sub_start = maxi_sub_length
                maxi_x_min = maxi_x_min - 1 if maxi_x_min - 1 > 0 else 0
                maxi_y_min = maxi_y_min - 1 if maxi_y_min - 1 > 0 else 0
                maxi_z_min = maxi_z_min - 1 if maxi_z_min - 1 > 0 else 0
                maxi_x_max = maxi_x_max + 1 if maxi_x_max + 1 < maxi.shape[0] else maxi.shape[0]
                maxi_y_max = maxi_y_max + 1 if maxi_y_max + 1 < maxi.shape[1] else maxi.shape[1]
                maxi_z_max = maxi_z_max + 1 if maxi_z_max + 1 < maxi.shape[2] else maxi.shape[2]
                maxi_sub = maxi[maxi_x_min:maxi_x_max, maxi_y_min:maxi_y_max, maxi_z_min:maxi_z_max]
                maxi_sub_ind = np.where(maxi_sub >= m - maxi.std() * std_factor)
    #             maxi_sub_ind = np.where(((maxi_sub - m) ** 2) / m <= 1)
                maxi_sub_length = len(maxi_sub_ind[0])

            p_x_min = maxi_sub_ind[0].min() + maxi_x_min - 3
            p_x_max = maxi_sub_ind[0].max() + maxi_x_min + 4
            p_y_min = maxi_sub_ind[1].min() + maxi_y_min - 3
            p_y_max = maxi_sub_ind[1].max() + maxi_y_min + 4
            p_z_min = maxi_sub_ind[2].min() + maxi_z_min - 3
            p_z_max = maxi_sub_ind[2].max() + maxi_z_min + 4

            if p_x_min <= 0:
                p_x_min = 0
            if p_y_min <= 0:
                p_y_min = 0
            if p_z_min <= 0:
                p_z_min = 0

            if p_x_max >= ptraveltimes.shape[1]:
                p_x_max = ptraveltimes.shape[1]
            if p_y_max >= ptraveltimes.shape[2]:
                p_y_max = ptraveltimes.shape[2]
            if p_z_max >= ptraveltimes.shape[3]:
                p_z_max = ptraveltimes.shape[3]

            if (p_z_max - p_z_min) * coarsest_z_spacing > 400 or \
                (p_x_max - p_x_min) * coarsest_x_spacing > 400 or \
                (p_y_max - p_y_min) * coarsest_y_spacing > 400:
                print('No.' + str(event_no) + ' error: event is poorly constrained. \n')
                events.append([Q, m, -1, p_x_filt, p_y_filt, p_z_filt])
                continue 
            
        # Pull data from the finer p and s traveltime grids to better locate the event
        tt_dir = '/Volumes/SeaJade 2 Backup/NZ/EQTransformer/inputs/tt'
        all_station_list = sta_cat.sta.unique()
        sub_station_list = sta_cat.loc[np.unique(ps_station),'sta'].values
        sub_station_ids = np.unique(ps_station)
        sta_file = tt_dir + '/' + all_station_list[0] + '_P.hdf'
        p_x_npts, p_y_npts, p_z_npts = np.array(h5py.File(sta_file, 'r')['npts'])

        # Find minimum an maximum coordinates for fine traveltime grids
        p_x_min = int(round(p_x_min / zoom_factor))
        p_y_min = int(round(p_y_min / zoom_factor))
        p_z_min = int(round(p_z_min / zoom_factor))
        p_x_max = int(round(p_x_max / zoom_factor))
        p_y_max = int(round(p_y_max / zoom_factor))
        p_z_max = int(round(p_z_max / zoom_factor))

        if p_x_min < 0:
            p_x_min = 0
        if p_x_max > p_x_npts:
            p_x_max = p_x_npts

        if p_y_min < 0:
            p_y_min = 0
        if p_y_max > p_y_npts:
            p_y_max = p_y_npts

        if p_z_min < 0:
            p_z_min = 0
        if p_z_max > p_z_npts:
            p_z_max = p_z_npts

        # Add to  subsequent p_x coordinates for the correct X,Y,Z positions in the intermediate
        # grid
        p_x_mod = p_x_min
        p_y_mod = p_y_min
        p_z_mod = p_z_min

######### Load intermediate traveltime grids on subset data ##############################
        tt_dir = '/Volumes/SeaJade 2 Backup/NZ/EQTransformer/inputs/tt'
#         p_files = tt_dir + '/' + all_station_list + '_P.hdf'
        p_files = tt_dir + '/' + sub_station_list + '_P.hdf'
        s_files = tt_dir + '/' + sub_station_list + '_S.hdf'
        coarseptraveltimes = np.zeros((len(all_station_list),p_x_max - p_x_min,p_y_max - 
            p_y_min,p_z_max - p_z_min))
        coarsestraveltimes = np.zeros((len(all_station_list),p_x_max - p_x_min,p_y_max - 
            p_y_min,p_z_max - p_z_min))
        for j,idx in enumerate(sub_station_ids):
            coarseptraveltimes[idx] = np.array(h5py.File(p_files[j], 'r')['values'][p_x_min:p_x_max, p_y_min:p_y_max,
                                     p_z_min:p_z_max])
            coarsestraveltimes[idx] = np.array(h5py.File(s_files[j], 'r')['values'][p_x_min:p_x_max, p_y_min:p_y_max,
                                     p_z_min:p_z_max])
#         coarseptraveltimes = np.array([h5py.File(fname, 'r')['values'][p_x_min:p_x_max, p_y_min:p_y_max,
#                                      p_z_min:p_z_max] for fname in p_files])

#         s_files = tt_dir + '/' + all_station_list + '_S.hdf'
#         coarsestraveltimes = np.array([h5py.File(fname, 'r')['values'][p_x_min:p_x_max, p_y_min:p_y_max,
#                                      p_z_min:p_z_max] for fname in s_files])

        maxi = np.zeros(coarseptraveltimes[0].shape)
        maxi = maxiscan(pstation, ps_station, ps_arrival, coarseptraveltimes, coarsestraveltimes, terr, maxi)
        m = maxi.max()
        p_x, p_y, p_z = np.where(maxi == m)
        x_std, y_std, z_std = np.std(p_x) * x_spacing, np.std(p_y) * y_spacing, np.std(p_z) * z_spacing
        
        
        # Use the median initial values to filter out outlier stations
        p_x_filt, p_y_filt, p_z_filt = int(np.median(p_x)), int(np.median(p_y)), int(np.median(p_z))

#         if p_z_filt * z_spacing + z_min > 400:
#             print('No.' + str(event_no) + ' error: event is too deep. \n')
#             events.append([Q, m, -1, p_x_filt, p_y_filt, p_z_filt])
#             continue

        # If the event is initially located in a negative depth, search down for the most
        # likely location
        if int((np.median(p_z) + p_z_mod) * z_spacing) + z_min < 0:
            for ii, j in enumerate(np.arange((p_z_mod * z_spacing) + z_min, 100, z_spacing)):
#             for ii, j in enumerate(np.arange(z_min, 100, z_spacing)):
                if j > 0:
                    z_min_maxi = ii
                    break
            m = maxi[:, :, z_min_maxi:].max()
            p_x, p_y, p_z = np.where(maxi[:, :, z_min_maxi:] == m)
            p_z = p_z + z_min_maxi
            x_std, y_std, z_std = np.std(p_x) * x_spacing, np.std(p_y) * y_spacing, np.std(p_z) * z_spacing
            p_x_filt, p_y_filt, p_z_filt = int(np.median(p_x)), int(np.median(p_y)), int(np.median(p_z))

        # Pull out coordinates to use for error estimation
        maxi_x_min = int(p_x.min()) - 1 if int(p_x.min()) - 1 > 0 else 0
        maxi_y_min = int(p_y.min()) - 1 if int(p_y.min()) - 1 > 0 else 0
        maxi_z_min = int(p_z.min()) - 1 if int(p_z.min()) - 1 > 0 else 0
        maxi_x_max = int(p_x.max()) + 2 if int(p_x.max()) + 2 < maxi.shape[0] else maxi.shape[0]
        maxi_y_max = int(p_y.max()) + 2 if int(p_y.max()) + 2 < maxi.shape[1] else maxi.shape[1]
        maxi_z_max = int(p_z.max()) + 2 if int(p_z.max()) + 2 < maxi.shape[2] else maxi.shape[2]
        maxi_sub = maxi[maxi_x_min:maxi_x_max, maxi_y_min:maxi_y_max, maxi_z_min:maxi_z_max]
        maxi_sub_ind = np.where(maxi_sub >= m - maxi.std() * std_factor)
        maxi_sub_length = len(maxi_sub_ind[0])
        maxi_sub_start = 0

        while maxi_sub_length > maxi_sub_start:
            maxi_sub_start = maxi_sub_length
            maxi_x_min = maxi_x_min - 1 if maxi_x_min - 1 > 0 else 0
            maxi_y_min = maxi_y_min - 1 if maxi_y_min - 1 > 0 else 0
            maxi_z_min = maxi_z_min - 1 if maxi_z_min - 1 > 0 else 0
            maxi_x_max = maxi_x_max + 1 if maxi_x_max + 1 < maxi.shape[0] else maxi.shape[0]
            maxi_y_max = maxi_y_max + 1 if maxi_y_max + 1 < maxi.shape[1] else maxi.shape[1]
            maxi_z_max = maxi_z_max + 1 if maxi_z_max + 1 < maxi.shape[2] else maxi.shape[2]
            maxi_sub = maxi[maxi_x_min:maxi_x_max, maxi_y_min:maxi_y_max, maxi_z_min:maxi_z_max]
            maxi_sub_ind = np.where(maxi_sub >= m - maxi.std() * std_factor)
            maxi_sub_length = len(maxi_sub_ind[0])

        p_x_min = maxi_sub_ind[0].min() + maxi_x_min - 2
        p_x_max = maxi_sub_ind[0].max() + maxi_x_min + 3
        p_y_min = maxi_sub_ind[1].min() + maxi_y_min - 2
        p_y_max = maxi_sub_ind[1].max() + maxi_y_min + 3
        p_z_min = maxi_sub_ind[2].min() + maxi_z_min - 2
        p_z_max = maxi_sub_ind[2].max() + maxi_z_min + 3

        if p_x_min <= 0:
            p_x_min = 0
        if p_y_min <= 0:
            p_y_min = 0
        if ((p_z_min + p_z_mod) * z_spacing) + z_min < 0:
            for ii, j in enumerate(np.arange(z_min, 100, z_spacing)):
                if j > 0:
                    p_z_min = ii
                    break
        if p_z_min < 0:
            p_z_min = 0

        if p_x_max >= coarseptraveltimes.shape[1]:
            p_x_max = coarseptraveltimes.shape[1]
        if p_y_max >= coarseptraveltimes.shape[2]:
            p_y_max = coarseptraveltimes.shape[2]
        if p_z_max >= coarseptraveltimes.shape[3]:
            p_z_max = coarseptraveltimes.shape[3]
        if p_z_max <= p_z_min:
            p_z_max = p_z_min + 3

#         start = timeit.time()
#         all_estimates = []
#         for p_x_test in range(p_x_min, p_x_max):
#             for p_y_test in range(p_y_min, p_y_max):
#                 for p_z_test in range(p_z_min, p_z_max):
#                     estimatetimes = []
#                     for j in range(0, len(pstation)):
#                         t = parrival[j] - coarseptraveltimes[pstation[j], p_x_test, p_y_test, p_z_test]
#                         estimatetimes.append(t)
#                     for j in range(0, len(sstation)):
#                         t = sarrival[j] - coarsestraveltimes[sstation[j], p_x_test, p_y_test, p_z_test]
#                         estimatetimes.append(t)
#                     all_estimates.append(
#                         [np.median(estimatetimes), np.std(estimatetimes), p_x_test, p_y_test, p_z_test])
#         all_estimates = np.array(all_estimates)
#         est_id = np.where(all_estimates[:, 1] == all_estimates[:, 1].min())[0]
#         final_estimate = all_estimates[est_id]
#         eventtime0, est_std, p_x_filts, p_y_filts, p_z_filts = final_estimate[0]
#         p_x_filt, p_y_filt, p_z_filt = int(p_x_filts), int(p_y_filts), int(p_z_filts)
#         print(timeit.time() - start)
        if len(pstation) > 0 and len(sstation) > 0:
            t = (parrival - coarseptraveltimes[pstation,p_x_min:p_x_max,p_y_min:p_y_max,p_z_min:p_z_max].T).T
            t2 = (sarrival - coarsestraveltimes[sstation,p_x_min:p_x_max,p_y_min:p_y_max,p_z_min:p_z_max].T).T
            estimatetimes = np.append(t,t2,axis=0)
        elif len(pstation) > 0 and len(sstation) == 0:
            t = (parrival - coarseptraveltimes[pstation,p_x_min:p_x_max,p_y_min:p_y_max,p_z_min:p_z_max].T).T
            estimatetimes = t
        elif len(sstation) > 0 and len(pstation) == 0:
            t2 = (sarrival - coarsestraveltimes[sstation,p_x_min:p_x_max,p_y_min:p_y_max,p_z_min:p_z_max].T).T
            estimatetimes = t2
        medians = np.median(estimatetimes,axis=0)
        stds = np.std(estimatetimes,axis=0)
        est_std = stds.min()
        eventtime0 = medians[np.where(stds == est_std)][0]
        p_x_filt,p_y_filt,p_z_filt = np.array(np.where(stds == est_std)).T[0]+[p_x_min,p_y_min,p_z_min]

        # Remove outlier stations
        for j in range(0, len(pstation)):
            ot = parrival[j] - eventtime0
            tht = coarseptraveltimes[pstation[j], p_x_filt, p_y_filt, p_z_filt]
#             print(abs(ot - tht))
            if abs(ot - tht) > outlier * 2:
                pstation[j] = -1

        for j in range(0, len(sstation)):
            ot = sarrival[j] - eventtime0
            tht = coarsestraveltimes[sstation[j], p_x_filt, p_y_filt, p_z_filt]
#             print(abs(ot - tht))
            if abs(ot - tht) > outlier * 2:
                sstation[j] = -1
        parrival = parrival[pstation >= 0]
        pstation = pstation[pstation >= 0]
        sarrival = sarrival[sstation >= 0]
        sstation = sstation[sstation >= 0]
        ps_station = np.concatenate([pstation, sstation]).astype(int)
        ps_arrival = np.concatenate([parrival, sarrival])
        p_count = len(parrival)
        s_count = len(sarrival)
        phase_count = p_count + s_count

        if phase_count < phase_threshold:
            print('No.' + str(event_no) + ' not enough phases \n')
            events.append([Q, m, eventtime0, p_x, p_y, p_z])
            print([eventtime0, p_x_filt, p_y_filt, p_z_filt, -1, -1])
            continue

        # Pull data from the finer p and s traveltime grids to better locate the event
        all_station_list = sta_cat.sta.unique()
        sub_station_list = sta_cat.loc[np.unique(ps_station),'sta'].values
        sub_station_ids = np.unique(ps_station)
        fine_sta_file = tt_fine_dir + '/' + all_station_list[0] + '_P.hdf'
        fine_x_spacing, fine_y_spacing, fine_z_spacing = np.array(h5py.File(fine_sta_file, 'r')['node_intervals'])
        p_x_npts, p_y_npts, p_z_npts = np.array(h5py.File(fine_sta_file, 'r')['npts'])

        # Determine coarse to fine grid spacing
        coarse_fine_x = (x_spacing / fine_x_spacing)
        coarse_fine_y = (y_spacing / fine_y_spacing)
        coarse_fine_z = (z_spacing / fine_z_spacing)
        
        # Find minimum an maximum coordinates for fine traveltime grids
        p_x_min = int(round((p_x_min + p_x_mod) * coarse_fine_x) - math.ceil(coarse_fine_x))
        p_y_min = int(round((p_y_min + p_y_mod) * coarse_fine_y) - math.ceil(coarse_fine_y))
        p_z_min = int(round((p_z_min + p_z_mod) * coarse_fine_z) - math.ceil(coarse_fine_z))
        p_x_max = int(round((p_x_max + p_x_mod) * coarse_fine_x) + math.ceil(coarse_fine_x))
        p_y_max = int(round((p_y_max + p_y_mod) * coarse_fine_y) + math.ceil(coarse_fine_y))
        p_z_max = int(round((p_z_max + p_z_mod) * coarse_fine_z) + math.ceil(coarse_fine_z))

        if p_x_min < 0:
            p_x_min = 0
        if p_x_max > p_x_npts:
            p_x_max = p_x_npts

        if p_y_min < 0:
            p_y_min = 0
        if p_y_max > p_y_npts:
            p_y_max = p_y_npts

        if (p_z_min * fine_z_spacing) + z_min < 0:
            # Fix the p_z_min value so that it does not go into negative depths
            for ii, j in enumerate(np.arange(z_min, 100, fine_z_spacing)):
                if j > 0:
                    p_z_min = ii
                    break
        if p_z_max <= p_z_min:
            p_z_max = p_z_min + 3
        if p_z_max > p_z_npts:
            p_z_max = p_z_npts

        # Add to all subsequent p_x coordinates for the correct X,Y,Z positions in the fine
        # grid
        p_x_mod = p_x_min
        p_y_mod = p_y_min
        p_z_mod = p_z_min

################# Load refined traveltime grids on subset data ###########################
        tt_dir = '/Volumes/SeaJade 2 Backup/NZ/EQTransformer/inputs/tt'
#         p_files = tt_dir + '/' + all_station_list + '_P.hdf'
        p_files = tt_fine_dir + '/' + sub_station_list + '_P.hdf'
        s_files = tt_fine_dir + '/' + sub_station_list + '_S.hdf'
        fineptraveltimes = np.zeros((len(all_station_list),p_x_max - p_x_min,p_y_max - 
            p_y_min,p_z_max - p_z_min))
        finestraveltimes = np.zeros((len(all_station_list),p_x_max - p_x_min,p_y_max - 
            p_y_min,p_z_max - p_z_min))
        for j,idx in enumerate(sub_station_ids):
            fineptraveltimes[idx] = np.array(h5py.File(p_files[j], 'r')['values'][p_x_min:p_x_max, p_y_min:p_y_max,
                                     p_z_min:p_z_max])
            finestraveltimes[idx] = np.array(h5py.File(s_files[j], 'r')['values'][p_x_min:p_x_max, p_y_min:p_y_max,
                                     p_z_min:p_z_max])

        maxi = np.zeros(fineptraveltimes[0].shape)
        maxi = maxiscan(pstation, ps_station, ps_arrival, fineptraveltimes, finestraveltimes, terr, maxi)
        m = maxi.max()
        # 		Q = m / MAXI_ceil / 5

        p_x, p_y, p_z = np.where(maxi == m)
        x_std, y_std, z_std = np.std(p_x) * fine_x_spacing, np.std(p_y) * fine_y_spacing, np.std(p_z) * fine_z_spacing

        # In case of more than one value, find the median!
        p_x, p_y, p_z = int(np.median(p_x)), int(np.median(p_y)), int(np.median(p_z))

#         all_estimates = []
#         for p_x_test in range(0, fineptraveltimes[0].shape[0]):
#             for p_y_test in range(0, fineptraveltimes[0].shape[1]):
#                 for p_z_test in range(0, fineptraveltimes[0].shape[2]):
#                     estimatetimes = []
#                     for j in range(0, len(pstation)):
#                         t = parrival[j] - fineptraveltimes[pstation[j], p_x_test, p_y_test, p_z_test]
#                         estimatetimes.append(t)
#                     # 					print(estimatetimes)
#                     for j in range(0, len(sstation)):
#                         t = sarrival[j] - finestraveltimes[sstation[j], p_x_test, p_y_test, p_z_test]
#                         estimatetimes.append(t)
#                     all_estimates.append(
#                         [np.median(estimatetimes), np.std(estimatetimes), p_x_test, p_y_test, p_z_test])
#         all_estimates = np.array(all_estimates)
#         est_id = np.where(all_estimates[:, 1] == all_estimates[:, 1].min())[0]
# 
#         final_estimate = all_estimates[est_id]
#         eventtime0, est_std, p_x, p_y, p_z = final_estimate[0]
#         p_x = int(p_x)
#         p_y = int(p_y)
#         p_z = int(p_z)

        if len(pstation) > 0 and len(sstation) > 0:
            t = (parrival - fineptraveltimes[pstation,0:fineptraveltimes[0].shape[0],0:fineptraveltimes[0].shape[1],0:fineptraveltimes[0].shape[2]].T).T
            t2 = (sarrival - finestraveltimes[sstation,0:fineptraveltimes[0].shape[0],0:fineptraveltimes[0].shape[1],0:fineptraveltimes[0].shape[2]].T).T
            estimatetimes = np.append(t,t2,axis=0)
        elif len(pstation) > 0 and len(sstation) == 0:
            t = (parrival - fineptraveltimes[pstation,0:fineptraveltimes[0].shape[0],0:fineptraveltimes[0].shape[1],0:fineptraveltimes[0].shape[2]].T).T
            estimatetimes = t
        elif len(sstation) > 0 and len(pstation) == 0:
            t2 = (sarrival - finestraveltimes[sstation,0:fineptraveltimes[0].shape[0],0:fineptraveltimes[0].shape[1],0:fineptraveltimes[0].shape[2]].T).T
            estimatetimes = t2
        medians = np.median(estimatetimes,axis=0)
        stds = np.std(estimatetimes,axis=0)
        est_std = stds.min()
        eventtime0 = medians[np.where(stds == est_std)][0]
        p_x,p_y,p_z = np.array(np.where(stds == est_std)).T[0]

        ### Use the F-test (aka Harley's test) to find statistically significant standard deviations
        full = len(pstation) + len(sstation)
        
        ps_x_mod = 0
        ps_y_mod = 0
        ps_z_mod = 0
        grid_x_spacing = fine_x_spacing
        grid_y_spacing = fine_y_spacing
        grid_z_spacing = fine_z_spacing
        
        ps_out, xc, yc, zc, x_err, y_err, z_err, theta = uncertainty_calc(full,stds,est_std,grid_x_spacing,grid_y_spacing,grid_z_spacing,fine_x_spacing,fine_y_spacing,fine_z_spacing,p_x_mod,p_y_mod,
            p_z_mod,ps_x_mod,ps_y_mod,ps_z_mod,x_min,y_min,z_min,p_x_min,p_x_max,p_y_min,p_y_max,p_z_min,p_z_max)

        MAXI_ceil = full * (full - 1) / 2
        m = maxi[p_x, p_y, p_z]
        Q = m / MAXI_ceil / 5

        events.append([Q, m, eventtime0, p_x + p_x_mod, p_y + p_y_mod, p_z + p_z_mod])

        print('No.' + str(event_no) + ' : Quality = ' + str(Q))

        if Q < Q_threshold:
            print('No.' + str(event_no) + ' low quality event \n')
            catalog.append(
                [eventtime0, p_x + p_x_mod, p_y + p_y_mod, p_z + p_z_mod, -1, -1, -1, xc, yc, zc, x_err, y_err, z_err,
                 theta, Q])
            print([eventtime0, p_x + p_x_mod, p_y + p_y_mod, p_z + p_z_mod, -1, -1, x_err, y_err, z_err])
            continue

        for j in range(0, len(pstation)):
            ot = parrival[j] - eventtime0
            tht = fineptraveltimes[pstation[j], p_x, p_y, p_z]
            if abs(ot - tht) > outlier * 2:
                pstation[j] = -1

        for j in range(0, len(sstation)):
            ot = sarrival[j] - eventtime0
            tht = finestraveltimes[sstation[j], p_x, p_y, p_z]
            if abs(ot - tht) > outlier * 2:
                sstation[j] = -1

        parrival = parrival[pstation >= 0]
        pstation = pstation[pstation >= 0]
        sarrival = sarrival[sstation >= 0]
        sstation = sstation[sstation >= 0]
        ps_station = np.concatenate([pstation, sstation]).astype(int)
        ps_arrival = np.concatenate([parrival, sarrival])
        p_count = len(parrival)
        s_count = len(sarrival)
        if p_count >= s_count:
            sta_count = p_count
        else:
            sta_count = s_count
        phase_count = p_count + s_count

        parrivals = np.array(pstation)
        parrivals = parrivals[parrivals >= 0]

        sarrivals = np.array(sstation)
        sarrivals = sarrivals[sarrivals >= 0]

        parrivals_res = []
        sarrivals_res = []

        if len(parrivals) > 0:
            for j in range(0, len(pstation)):
                t = parrival[j] - (eventtime0 + fineptraveltimes[pstation[j], p_x, p_y, p_z])
                parrivals_res.append(t)
        if len(sarrivals) > 0:
            for j in range(0, len(sstation)):
                t = sarrival[j] - (eventtime0 + finestraveltimes[sstation[j], p_x, p_y, p_z])
                sarrivals_res.append(t)

        if phase_count >= phase_threshold:
            clean = phase_count
            minimum = est_std
            print('No.' + str(event_no) + ' minimum = ' + str(minimum) + 's')
            print('No.' + str(event_no) + ' clean arrivals: ' + str(clean))

            finaltime = eventtime0

            parrivals = np.array(pstation)
            parrivals = parrivals[parrivals >= 0]

            sarrivals = np.array(sstation)
            sarrivals = sarrivals[sarrivals >= 0]

            if len(parrivals) > 0:
                parrivals_time = ponsets[0][parrivals]
            else:
                parrivals_time = []
            if len(sarrivals) > 0:
                sarrivals_time = sonsets[0][sarrivals]
            else:
                sarrivals_time = []
        else:
            print('No.' + str(event_no) + ' not enough picks or event outside of area?')
            catalog.append(
                [eventtime0, p_x + p_x_mod, p_y + p_y_mod, p_z + p_z_mod, -1, -1, -1, xc, yc, zc, x_err, y_err, z_err,
                 theta, Q])
            print([eventtime0, p_x + p_x_mod, p_y + p_y_mod, p_z + p_z_mod, -1, -1])
            continue
        ########################################
        # Begin computations for 1 km spacing
        startrefineygrid = 1 / fine_x_spacing  # km
        startrefinexgrid = 1 / fine_y_spacing  # km
        startrefinedepgrid = 1 / fine_z_spacing  # km

        x_mod = 0
        y_mod = 0
        z_mod = 0
        
        n_in_x = int(np.ceil(10 / fine_x_spacing))
        n_in_y = int(np.ceil(10 / fine_y_spacing))
        n_in_z = int(np.ceil(10 / fine_z_spacing))

        p_x_min = p_x - n_in_x
        if p_x_min < 0:
            x_mod = (abs(p_x_min) * fine_x_spacing) / 2
            p_x_min = 0

        p_y_min = p_y - n_in_y
        if p_y_min < 0:
            y_mod = (abs(p_y_min) * fine_y_spacing) / 2
            p_y_min = 0

        p_z_min = p_z - n_in_z
        if p_z_min < 0:
            z_mod = (abs(p_z_min) * fine_z_spacing) / 2
            p_z_min = 0

        p_x_max = p_x + n_in_x + 1
        if p_x_max > fineptraveltimes.shape[1]:
            x_mod = ((fineptraveltimes.shape[1] - p_x_max) * fine_x_spacing) / 2
            p_x_max = fineptraveltimes.shape[1]

        p_y_max = p_y + n_in_y + 1
        if p_y_max > fineptraveltimes.shape[2]:
            y_mod = ((fineptraveltimes.shape[2] - p_y_max) * fine_y_spacing) / 2
            p_y_max = fineptraveltimes.shape[2]

        p_z_max = p_z + n_in_z + 1
        if p_z_max > fineptraveltimes.shape[3]:
            z_mod = ((fineptraveltimes.shape[3] - p_z_max) * fine_z_spacing) / 2
            p_z_max = fineptraveltimes.shape[3]

        refine_factor_1_x = ((fine_x_spacing * (p_x_max - p_x_min - 1)) + 1) / (p_x_max - p_x_min)
        refine_factor_1_y = ((fine_y_spacing * (p_y_max - p_y_min - 1)) + 1) / (p_y_max - p_y_min)
        refine_factor_1_z = ((fine_z_spacing * (p_z_max - p_z_min - 1)) + 1) / (p_z_max - p_z_min)

        refineptraveltimes = np.zeros((len(all_station_list),int((p_x_max - p_x_min) * refine_factor_1_x),
            int((p_y_max - p_y_min) * refine_factor_1_y), int((p_z_max - p_z_min) * refine_factor_1_z)))
        refinestraveltimes = np.zeros((len(all_station_list),int((p_x_max - p_x_min) * refine_factor_1_x),
            int((p_y_max - p_y_min) * refine_factor_1_y), int((p_z_max - p_z_min) * refine_factor_1_z)))
        p_zoom_sub = zoom(fineptraveltimes[np.unique(ps_station), p_x_min:p_x_max, p_y_min:p_y_max, p_z_min:p_z_max],
                                  [1, refine_factor_1_x, refine_factor_1_y, refine_factor_1_z])
        s_zoom_sub = zoom(finestraveltimes[np.unique(ps_station), p_x_min:p_x_max, p_y_min:p_y_max, p_z_min:p_z_max],
                                  [1, refine_factor_1_x, refine_factor_1_y, refine_factor_1_z])
        for j, idx in enumerate(np.unique(ps_station)):
            refineptraveltimes[idx] = p_zoom_sub[j]
            refinestraveltimes[idx] = s_zoom_sub[j]

        x_max, y_max, z_max = refineptraveltimes.shape[1:4]
#         all_estimates = []
#         for p_x_test in range(0, x_max):
#             for p_y_test in range(0, y_max):
#                 for p_z_test in range(0, z_max):
#                     estimatetimes = []
#                     for j in range(0, len(pstation)):
#                         t = parrival[j] - refineptraveltimes[pstation[j], p_x_test, p_y_test, p_z_test]
#                         estimatetimes.append(t)
#                     for j in range(0, len(sstation)):
#                         t = sarrival[j] - refinestraveltimes[sstation[j], p_x_test, p_y_test, p_z_test]
#                         estimatetimes.append(t)
#                     all_estimates.append(
#                         [np.median(estimatetimes), np.std(estimatetimes), p_x_test, p_y_test, p_z_test])
#         all_estimates = np.array(all_estimates)
#         est_id = np.where(all_estimates[:, 1] == all_estimates[:, 1].min())[0]
#         final_estimate = all_estimates[est_id]
#         eventtime0, est_std, p_x_refined, p_y_refined, p_z_refined = final_estimate[0]
#         p_x_refined = int(p_x_refined)
#         p_y_refined = int(p_y_refined)
#         p_z_refined = int(p_z_refined)

        if len(pstation) > 0 and len(sstation) > 0:
            t = (parrival - refineptraveltimes[pstation,0:x_max,0:y_max,0:z_max].T).T
            t2 = (sarrival - refinestraveltimes[sstation,0:x_max,0:y_max,0:z_max].T).T
            estimatetimes = np.append(t,t2,axis=0)
        elif len(pstation) > 0 and len(sstation) == 0:
            t = (parrival - refineptraveltimes[pstation,0:x_max,0:y_max,0:z_max].T).T
            estimatetimes = t
        elif len(sstation) > 0 and len(pstation) == 0:
            t2 = (sarrival - refinestraveltimes[sstation,0:x_max,0:y_max,0:z_max].T).T
            estimatetimes = t2
        medians = np.median(estimatetimes,axis=0)
        stds = np.std(estimatetimes,axis=0)
        est_std = stds.min()
        eventtime0 = medians[np.where(stds == est_std)][0]
        p_x_refined,p_y_refined,p_z_refined = np.array(np.where(stds == est_std)).T[0]

        p_x_refined_coord, p_y_refined_coord, p_z_refined_coord = p_x_refined - (
                ((refineptraveltimes.shape[1]) - 1) / 2) + x_mod, p_y_refined - (((refineptraveltimes.shape[
            2]) - 1) / 2) + y_mod, p_z_refined - (((refineptraveltimes.shape[3]) - 1) / 2) + z_mod
        p_x_final, p_y_final, p_z_final = p_x_mod + p_x + p_x_refined_coord / fine_x_spacing, p_y_mod + p_y + p_y_refined_coord / fine_y_spacing, p_z_mod + p_z + p_z_refined_coord / fine_z_spacing

        full = len(pstation) + len(sstation)
        ps_x_mod = (((refineptraveltimes.shape[1]) - 1) / 2) + x_mod
        ps_y_mod = (((refineptraveltimes.shape[2]) - 1) / 2) + y_mod
        ps_z_mod = (((refineptraveltimes.shape[3]) - 1) / 2) + z_mod
        grid_x_spacing = 1
        grid_y_spacing = 1
        grid_z_spacing = 1
        
        dfd = full - 1
        dfn = dfd
        fs = stds ** 2 / est_std ** 2
        p_vals = 1-scipy.stats.f.cdf(fs, dfn, dfd)
       
        ps = np.where(p_vals >= 0.05)
        ps_x_unique = np.unique(ps[0])
        ps_y_unique = np.unique(ps[1])
        ps_z_unique = np.unique(ps[2])
        if ps_x_unique.min() != 0 and ps_x_unique.max() != x_max - 1 and \
            ps_y_unique.min() != 0 and ps_y_unique.max() != y_max - 1 and \
            ps_z_unique.min() != 0 and ps_z_unique.max() != z_max - 1:
            ps_out, xc, yc, zc, x_err, y_err, z_err, theta = uncertainty_calc(full,stds,est_std,grid_x_spacing,grid_y_spacing,grid_z_spacing,fine_x_spacing,fine_y_spacing,fine_z_spacing,p_x_final,p_y_final,
                p_z_final,ps_x_mod,ps_y_mod,ps_z_mod,x_min,y_min,z_min,p_x_min,p_x_max,p_y_min,p_y_max,p_z_min,p_z_max)            

        if phase_count >= phase_threshold:
            improve = (minimum - est_std) / minimum
            if improve >= 0:
                parrivals_res = []
                sarrivals_res = []

                minimum = est_std
                print('No.' + str(event_no) + ' minimum = ' + str(minimum) + 's')
                print('No.' + str(event_no) + ' clean arrivals: ' + str(clean))

                finaltime = eventtime0

                parrivals = np.array(pstation)
                parrivals = parrivals[parrivals >= 0]

                sarrivals = np.array(sstation)
                sarrivals = sarrivals[sarrivals >= 0]

                if len(parrivals) > 0:
                    for j in range(0, len(pstation)):
                        t = parrival[j] - (
                                eventtime0 + refineptraveltimes[pstation[j], p_x_refined, p_y_refined, p_z_refined])
                        parrivals_res.append(t)
                if len(sarrivals) > 0:
                    for j in range(0, len(sstation)):
                        t = sarrival[j] - (
                                eventtime0 + refinestraveltimes[sstation[j], p_x_refined, p_y_refined, p_z_refined])
                        sarrivals_res.append(t)

                if len(parrivals) > 0:
                    parrivals_time = ponsets[0][parrivals]
                else:
                    parrivals_time = []
                if len(sarrivals) > 0:
                    sarrivals_time = sonsets[0][sarrivals]
                else:
                    sarrivals_time = []

            else:
                print('No.' + str(event_no) + ' cannot be refined further, minimum increased')
                arrivals.append([parrivals, sarrivals, parrivals_time, sarrivals_time, np.array(parrivals_res),
                                 np.array(sarrivals_res)])
                catalog.append(
                    [finaltime, p_x + p_x_mod, p_y + p_y_mod, p_z + p_z_mod, minimum, fine_x_spacing, clean, xc, yc, zc,
                     x_err, y_err, z_err, theta, Q])
                unc_grid.append(ps_out)
                print(
                    [finaltime, p_x + p_x_mod, p_y + p_y_mod, p_z + p_z_mod, minimum, fine_x_spacing, clean, xc, yc, zc,
                     x_err, y_err, z_err, theta, Q])
                continue
        else:
            print('No.' + str(event_no) + ' cannot be refined further, phases below threshold')
            arrivals.append([parrivals, sarrivals, parrivals_time, sarrivals_time, np.array(parrivals_res),
                             np.array(sarrivals_res)])
            catalog.append(
                [finaltime, p_x + p_x_mod, p_y + p_y_mod, p_z + p_z_mod, minimum, fine_x_spacing, clean, xc, yc, zc,
                 x_err, y_err, z_err, theta, Q])
            unc_grid.append(ps_out)
            print([finaltime, p_x + p_x_mod, p_y + p_y_mod, p_z + p_z_mod, minimum, fine_x_spacing, clean, xc, yc, zc,
                   x_err, y_err, z_err, theta, Q])
            continue

        ########################################################################
        # Begin computations for 0.1 km spacing

        startrefineygrid = startrefineygrid * 0.1  # km
        startrefinexgrid = startrefinexgrid * 0.1  # km
        startrefinedepgrid = startrefinedepgrid * 0.1  # km

        refine_factor_2 = 10

        x_mod = 0
        y_mod = 0
        z_mod = 0

        p_x_refined_min = p_x_refined - 1
        if p_x_refined_min < 0:
            x_mod = (abs(p_x_refined_min) * refine_factor_2) / 2
            p_x_refined_min = 0

        p_y_refined_min = p_y_refined - 1
        if p_y_refined_min < 0:
            y_mod = (abs(p_y_refined_min) * refine_factor_2) / 2
            p_y_refined_min = 0

        p_z_refined_min = p_z_refined - 1
        if p_z_refined_min < 0:
            z_mod = (abs(p_z_refined_min) * refine_factor_2) / 2
            p_z_refined_min = 0

        p_x_refined_max = p_x_refined + 2
        if p_x_refined_max > refineptraveltimes.shape[1]:
            x_mod = ((refineptraveltimes.shape[1] - p_x_refined_max) * refine_factor_2) / 2
            p_x_refined_max = refineptraveltimes.shape[1]

        p_y_refined_max = p_y_refined + 2
        if p_y_refined_max > refineptraveltimes.shape[2]:
            y_mod = ((refineptraveltimes.shape[2] - p_y_refined_max) * refine_factor_2) / 2
            p_y_refined_max = refineptraveltimes.shape[2]

        p_z_refined_max = p_z_refined + 2
        if p_z_refined_max > refineptraveltimes.shape[3]:
            z_mod = ((refineptraveltimes.shape[3] - p_z_refined_max) * refine_factor_2) / 2
            p_z_refined_max = refineptraveltimes.shape[3]

        refine_factor_2_x = ((refine_factor_2 * (p_x_refined_max - p_x_refined_min - 1)) + 1) / (
                p_x_refined_max - p_x_refined_min)
        refine_factor_2_y = ((refine_factor_2 * (p_y_refined_max - p_y_refined_min - 1)) + 1) / (
                p_y_refined_max - p_y_refined_min)
        refine_factor_2_z = ((refine_factor_2 * (p_z_refined_max - p_z_refined_min - 1)) + 1) / (
                p_z_refined_max - p_z_refined_min)

        p_zoom_sub = zoom(refineptraveltimes[np.unique(ps_station), p_x_refined_min:p_x_refined_max, p_y_refined_min:p_y_refined_max, p_z_refined_min:p_z_refined_max],
                                  [1, refine_factor_2_x, refine_factor_2_y, refine_factor_2_z])
        s_zoom_sub = zoom(refinestraveltimes[np.unique(ps_station), p_x_refined_min:p_x_refined_max, p_y_refined_min:p_y_refined_max, p_z_refined_min:p_z_refined_max],
                                  [1, refine_factor_2_x, refine_factor_2_y, refine_factor_2_z])
        refineptraveltimes = np.zeros((len(all_station_list),int((p_x_refined_max - p_x_refined_min) * refine_factor_2_x),
            int((p_y_refined_max - p_y_refined_min) * refine_factor_2_y), int((p_z_refined_max - p_z_refined_min) * refine_factor_2_z)))
        refinestraveltimes = np.zeros((len(all_station_list),int((p_x_refined_max - p_x_refined_min) * refine_factor_2_x),
            int((p_y_refined_max - p_y_refined_min) * refine_factor_2_y), int((p_z_refined_max - p_z_refined_min) * refine_factor_2_z)))
        for j, idx in enumerate(np.unique(ps_station)):
            refineptraveltimes[idx] = p_zoom_sub[j]
            refinestraveltimes[idx] = s_zoom_sub[j]

#         x_max, y_max, z_max = refineptraveltimes.shape[1:4]
#         all_estimates = []
#         for p_x_test in range(0, x_max):
#             for p_y_test in range(0, y_max):
#                 for p_z_test in range(0, z_max):
#                     estimatetimes = []
#                     for j in range(0, len(pstation)):
#                         t = parrival[j] - refineptraveltimes[pstation[j], p_x_test, p_y_test, p_z_test]
#                         estimatetimes.append(t)
#                     for j in range(0, len(sstation)):
#                         t = sarrival[j] - refinestraveltimes[sstation[j], p_x_test, p_y_test, p_z_test]
#                         estimatetimes.append(t)
#                     all_estimates.append(
#                         [np.median(estimatetimes), np.std(estimatetimes), p_x_test, p_y_test, p_z_test])
#         all_estimates = np.array(all_estimates)
#         est_id = np.where(all_estimates[:, 1] == all_estimates[:, 1].min())[0]
#         final_estimate = all_estimates[est_id]
#         eventtime0, est_std, p_x_refined, p_y_refined, p_z_refined = final_estimate[0]
#         p_x_refined = int(p_x_refined)
#         p_y_refined = int(p_y_refined)
#         p_z_refined = int(p_z_refined)
        
        if len(pstation) > 0 and len(sstation) > 0:
            t = (parrival - refineptraveltimes[pstation,0:x_max,0:y_max,0:z_max].T).T
            t2 = (sarrival - refinestraveltimes[sstation,0:x_max,0:y_max,0:z_max].T).T
            estimatetimes = np.append(t,t2,axis=0)
        elif len(pstation) > 0 and len(sstation) == 0:
            t = (parrival - refineptraveltimes[pstation,0:x_max,0:y_max,0:z_max].T).T
            estimatetimes = t
        elif len(sstation) > 0 and len(pstation) == 0:
            t2 = (sarrival - refinestraveltimes[sstation,0:x_max,0:y_max,0:z_max].T).T
            estimatetimes = t2
        medians = np.median(estimatetimes,axis=0)
        stds = np.std(estimatetimes,axis=0)
        est_std = stds.min()
        eventtime0 = medians[np.where(stds == est_std)][0]
        p_x_refined,p_y_refined,p_z_refined = np.array(np.where(stds == est_std)).T[0]

        p_x_refined_coord, p_y_refined_coord, p_z_refined_coord = p_x_refined - (
                ((refineptraveltimes.shape[1]) - 1) / 2) + x_mod, p_y_refined - (((refineptraveltimes.shape[
            2]) - 1) / 2) + y_mod, p_z_refined - (((refineptraveltimes.shape[3]) - 1) / 2) + z_mod
        p_x_final_old, p_y_final_old, p_z_final_old = p_x_final, p_y_final, p_z_final
        p_x_final, p_y_final, p_z_final = p_x_final + p_x_refined_coord / (
                fine_x_spacing * refine_factor_2), p_y_final + p_y_refined_coord / (
                                                  fine_y_spacing * refine_factor_2), p_z_final + p_z_refined_coord / (
                                                  fine_z_spacing * refine_factor_2)

        full = len(pstation) + len(sstation)
        ps_x_mod = (((refineptraveltimes.shape[1]) - 1) / 2) + x_mod
        ps_y_mod = (((refineptraveltimes.shape[2]) - 1) / 2) + y_mod
        ps_z_mod = (((refineptraveltimes.shape[3]) - 1) / 2) + z_mod
        grid_x_spacing = 1 / refine_factor_2
        grid_y_spacing = 1 / refine_factor_2
        grid_z_spacing = 1 / refine_factor_2
        
        dfd = full - 1
        dfn = dfd
        fs = stds ** 2 / est_std ** 2
        p_vals = 1-scipy.stats.f.cdf(fs, dfn, dfd)
       
        ps = np.where(p_vals >= 0.05)
        ps_x_unique = np.unique(ps[0])
        ps_y_unique = np.unique(ps[1])
        ps_z_unique = np.unique(ps[2])
        if ps_x_unique.min() != 0 and ps_x_unique.max() != x_max - 1 and \
            ps_y_unique.min() != 0 and ps_y_unique.max() != y_max - 1 and \
            ps_z_unique.min() != 0 and ps_z_unique.max() != z_max - 1:
            ps_out, xc, yc, zc, x_err, y_err, z_err, theta = uncertainty_calc(full,stds,est_std,grid_x_spacing,grid_y_spacing,grid_z_spacing,fine_x_spacing,fine_y_spacing,fine_z_spacing,p_x_final,p_y_final,
                p_z_final,ps_x_mod,ps_y_mod,ps_z_mod,x_min,y_min,z_min,p_x_min,p_x_max,p_y_min,p_y_max,p_z_min,p_z_max)            

        if phase_count >= phase_threshold:
            improve = (minimum - est_std) / minimum
            if improve >= 0:
                parrivals_res = []
                sarrivals_res = []

                minimum = est_std
                print('No.' + str(event_no) + ' minimum = ' + str(minimum) + 's')
                print('No.' + str(event_no) + ' clean arrivals: ' + str(clean))

                finaltime = eventtime0

                parrivals = np.array(pstation)
                parrivals = parrivals[parrivals >= 0]

                sarrivals = np.array(sstation)
                sarrivals = sarrivals[sarrivals >= 0]

                if len(parrivals) > 0:
                    for j in range(0, len(pstation)):
                        t = parrival[j] - (
                                eventtime0 + refineptraveltimes[pstation[j], p_x_refined, p_y_refined, p_z_refined])
                        parrivals_res.append(t)
                if len(sarrivals) > 0:
                    for j in range(0, len(sstation)):
                        t = sarrival[j] - (
                                eventtime0 + refinestraveltimes[sstation[j], p_x_refined, p_y_refined, p_z_refined])
                        sarrivals_res.append(t)

                if len(parrivals) > 0:
                    parrivals_time = ponsets[0][parrivals]
                else:
                    parrivals_time = []
                if len(sarrivals) > 0:
                    sarrivals_time = sonsets[0][sarrivals]
                else:
                    sarrivals_time = []

            else:
                print('No.' + str(event_no) + ' cannot be refined further, minimum increased')
                arrivals.append([parrivals, sarrivals, parrivals_time, sarrivals_time, np.array(parrivals_res),
                                 np.array(sarrivals_res)])
                catalog.append(
                    [finaltime, p_x_final_old, p_y_final_old, p_z_final_old, minimum, 1, clean, xc, yc, zc, x_err,
                     y_err, z_err, theta, Q])
                unc_grid.append(ps_out)
                print([finaltime, p_x_final_old, p_y_final_old, p_z_final_old, minimum, 1, clean, xc, yc, zc, x_err,
                       y_err, z_err, theta, Q])
                continue
        else:
            print('No.' + str(event_no) + ' cannot be refined further, phases below threshold')
            arrivals.append([parrivals, sarrivals, parrivals_time, sarrivals_time, np.array(parrivals_res),
                             np.array(sarrivals_res)])
            catalog.append(
                [finaltime, p_x_final_old, p_y_final_old, p_z_final_old, minimum, 1, clean, xc, yc, zc, x_err, y_err,
                 z_err, theta, Q])
            unc_grid.append(ps_out)
            print([finaltime, p_x_final_old, p_y_final_old, p_z_final_old, minimum, 1, clean, xc, yc, zc, x_err, y_err,
                   z_err, theta, Q])
            continue
        ##########################################################################################
        # Begin computations for 0.01 km spacing

        startrefineygrid = startrefineygrid * 0.1  # km
        startrefinexgrid = startrefinexgrid * 0.1  # km
        startrefinedepgrid = startrefinedepgrid * 0.1  # km
        refine_factor_3 = 10

        x_mod = 0
        y_mod = 0
        z_mod = 0

        p_x_refined_min = p_x_refined - 1
        if p_x_refined_min < 0:
            x_mod = (abs(p_x_refined_min) * refine_factor_3) / 2
            p_x_refined_min = 0

        p_y_refined_min = p_y_refined - 1
        if p_y_refined_min < 0:
            y_mod = (abs(p_y_refined_min) * refine_factor_3) / 2
            p_y_refined_min = 0

        p_z_refined_min = p_z_refined - 1
        if p_z_refined_min < 0:
            z_mod = (abs(p_z_refined_min) * refine_factor_3) / 2
            p_z_refined_min = 0

        p_x_refined_max = p_x_refined + 2
        if p_x_refined_max > refineptraveltimes.shape[1]:
            x_mod = ((refineptraveltimes.shape[1] - p_x_refined_max) * refine_factor_3) / 2
            p_x_refined_max = refineptraveltimes.shape[1]

        p_y_refined_max = p_y_refined + 2
        if p_y_refined_max > refineptraveltimes.shape[2]:
            y_mod = ((refineptraveltimes.shape[2] - p_y_refined_max) * refine_factor_3) / 2
            p_y_refined_max = refineptraveltimes.shape[2]

        p_z_refined_max = p_z_refined + 2
        if p_z_refined_max > refineptraveltimes.shape[3]:
            z_mod = ((refineptraveltimes.shape[3] - p_z_refined_max) * refine_factor_3) / 2
            p_z_refined_max = refineptraveltimes.shape[3]

        refine_factor_3_x = ((refine_factor_3 * (p_x_refined_max - p_x_refined_min - 1)) + 1) / (
                p_x_refined_max - p_x_refined_min)
        refine_factor_3_y = ((refine_factor_3 * (p_y_refined_max - p_y_refined_min - 1)) + 1) / (
                p_y_refined_max - p_y_refined_min)
        refine_factor_3_z = ((refine_factor_3 * (p_z_refined_max - p_z_refined_min - 1)) + 1) / (
                p_z_refined_max - p_z_refined_min)

        p_zoom_sub = zoom(refineptraveltimes[np.unique(ps_station), p_x_refined_min:p_x_refined_max, p_y_refined_min:p_y_refined_max, p_z_refined_min:p_z_refined_max],
                                  [1, refine_factor_3_x, refine_factor_3_y, refine_factor_3_z])
        s_zoom_sub = zoom(refinestraveltimes[np.unique(ps_station), p_x_refined_min:p_x_refined_max, p_y_refined_min:p_y_refined_max, p_z_refined_min:p_z_refined_max],
                                  [1, refine_factor_3_x, refine_factor_3_y, refine_factor_3_z])
        refineptraveltimes = np.zeros((len(all_station_list),int((p_x_refined_max - p_x_refined_min) * refine_factor_3_x),
            int((p_y_refined_max - p_y_refined_min) * refine_factor_3_y), int((p_z_refined_max - p_z_refined_min) * refine_factor_3_z)))
        refinestraveltimes = np.zeros((len(all_station_list),int((p_x_refined_max - p_x_refined_min) * refine_factor_3_x),
            int((p_y_refined_max - p_y_refined_min) * refine_factor_3_y), int((p_z_refined_max - p_z_refined_min) * refine_factor_3_z)))
        for j, idx in enumerate(np.unique(ps_station)):
            refineptraveltimes[idx] = p_zoom_sub[j]
            refinestraveltimes[idx] = s_zoom_sub[j]

#         x_max, y_max, z_max = refineptraveltimes.shape[1:4]
#         all_estimates = []
#         for p_x_test in range(0, x_max):
#             for p_y_test in range(0, y_max):
#                 for p_z_test in range(0, z_max):
#                     estimatetimes = []
#                     for j in range(0, len(pstation)):
#                         t = parrival[j] - refineptraveltimes[pstation[j], p_x_test, p_y_test, p_z_test]
#                         estimatetimes.append(t)
#                     for j in range(0, len(sstation)):
#                         t = sarrival[j] - refinestraveltimes[sstation[j], p_x_test, p_y_test, p_z_test]
#                         estimatetimes.append(t)
#                     all_estimates.append(
#                         [np.median(estimatetimes), np.std(estimatetimes), p_x_test, p_y_test, p_z_test])
#         all_estimates = np.array(all_estimates)
#         est_id = np.where(all_estimates[:, 1] == all_estimates[:, 1].min())[0]
#         final_estimate = all_estimates[est_id]
#         eventtime0, est_std, p_x_refined, p_y_refined, p_z_refined = final_estimate[0]
#         p_x_refined = int(p_x_refined)
#         p_y_refined = int(p_y_refined)
#         p_z_refined = int(p_z_refined)
        
        if len(pstation) > 0 and len(sstation) > 0:
            t = (parrival - refineptraveltimes[pstation,0:x_max,0:y_max,0:z_max].T).T
            t2 = (sarrival - refinestraveltimes[sstation,0:x_max,0:y_max,0:z_max].T).T
            estimatetimes = np.append(t,t2,axis=0)
        elif len(pstation) > 0 and len(sstation) == 0:
            t = (parrival - refineptraveltimes[pstation,0:x_max,0:y_max,0:z_max].T).T
            estimatetimes = t
        elif len(sstation) > 0 and len(pstation) == 0:
            t2 = (sarrival - refinestraveltimes[sstation,0:x_max,0:y_max,0:z_max].T).T
            estimatetimes = t2
        medians = np.median(estimatetimes,axis=0)
        stds = np.std(estimatetimes,axis=0)
        est_std = stds.min()
        eventtime0 = medians[np.where(stds == est_std)][0]
        p_x_refined,p_y_refined,p_z_refined = np.array(np.where(stds == est_std)).T[0]
       
        p_x_refined_coord, p_y_refined_coord, p_z_refined_coord = p_x_refined - (
                ((refineptraveltimes.shape[1]) - 1) / 2) + x_mod, p_y_refined - (((refineptraveltimes.shape[
            2]) - 1) / 2) + y_mod, p_z_refined - (((refineptraveltimes.shape[3]) - 1) / 2) + z_mod
        p_x_final_old, p_y_final_old, p_z_final_old = p_x_final, p_y_final, p_z_final
        p_x_final, p_y_final, p_z_final = p_x_final + p_x_refined_coord / (
                fine_x_spacing * refine_factor_2 * refine_factor_3), p_y_final + p_y_refined_coord / (
                                                  fine_y_spacing * refine_factor_2 * refine_factor_3), p_z_final + p_z_refined_coord / (
                                                  fine_z_spacing * refine_factor_2 * refine_factor_3)

        full = len(pstation) + len(sstation)
        ps_x_mod = (((refineptraveltimes.shape[1]) - 1) / 2) + x_mod
        ps_y_mod = (((refineptraveltimes.shape[2]) - 1) / 2) + y_mod
        ps_z_mod = (((refineptraveltimes.shape[3]) - 1) / 2) + z_mod
        grid_x_spacing = 1 / (refine_factor_2 * refine_factor_3)
        grid_y_spacing = 1 / (refine_factor_2 * refine_factor_3)
        grid_z_spacing = 1 / (refine_factor_2 * refine_factor_3)
        
        dfd = full - 1
        dfn = dfd
        fs = stds ** 2 / est_std ** 2
        p_vals = 1-scipy.stats.f.cdf(fs, dfn, dfd)
       
        ps = np.where(p_vals >= 0.05)
        ps_x_unique = np.unique(ps[0])
        ps_y_unique = np.unique(ps[1])
        ps_z_unique = np.unique(ps[2])
        if ps_x_unique.min() != 0 and ps_x_unique.max() != x_max - 1 and \
            ps_y_unique.min() != 0 and ps_y_unique.max() != y_max - 1 and \
            ps_z_unique.min() != 0 and ps_z_unique.max() != z_max - 1:
            ps_out, xc, yc, zc, x_err, y_err, z_err, theta = uncertainty_calc(full,stds,est_std,grid_x_spacing,grid_y_spacing,grid_z_spacing,fine_x_spacing,fine_y_spacing,fine_z_spacing,p_x_final,p_y_final,
                p_z_final,ps_x_mod,ps_y_mod,ps_z_mod,x_min,y_min,z_min,p_x_min,p_x_max,p_y_min,p_y_max,p_z_min,p_z_max)            

        if phase_count >= phase_threshold:
            improve = (minimum - est_std) / minimum
            if improve >= 0:
                parrivals_res = []
                sarrivals_res = []

                minimum = est_std
                print('No.' + str(event_no) + ' minimum = ' + str(minimum) + 's')
                print('No.' + str(event_no) + ' clean arrivals: ' + str(clean))

                finaltime = eventtime0

                parrivals = np.array(pstation)
                parrivals = parrivals[parrivals >= 0]

                sarrivals = np.array(sstation)
                sarrivals = sarrivals[sarrivals >= 0]

                if len(parrivals) > 0:
                    for j in range(0, len(pstation)):
                        t = parrival[j] - (
                                eventtime0 + refineptraveltimes[pstation[j], p_x_refined, p_y_refined, p_z_refined])
                        parrivals_res.append(t)
                if len(sarrivals) > 0:
                    for j in range(0, len(sstation)):
                        t = sarrival[j] - (
                                eventtime0 + refinestraveltimes[sstation[j], p_x_refined, p_y_refined, p_z_refined])
                        sarrivals_res.append(t)

                if len(parrivals) > 0:
                    parrivals_time = ponsets[0][parrivals]
                else:
                    parrivals_time = []
                if len(sarrivals) > 0:
                    sarrivals_time = sonsets[0][sarrivals]
                else:
                    sarrivals_time = []
                arrivals.append([parrivals, sarrivals, parrivals_time, sarrivals_time, np.array(parrivals_res),
                                 np.array(sarrivals_res)])
                catalog.append(
                    [finaltime, p_x_final, p_y_final, p_z_final, minimum, 0.01, clean, xc, yc, zc, x_err, y_err, z_err,
                     theta, Q])
                unc_grid.append(ps_out)
                print(
                    [finaltime, p_x_final, p_y_final, p_z_final, minimum, 0.01, clean, xc, yc, zc, x_err, y_err, z_err,
                     theta, Q])
            else:
                print('No.' + str(event_no) + ' cannot be refined further, minimum increased')
                arrivals.append([parrivals, sarrivals, parrivals_time, sarrivals_time, np.array(parrivals_res),
                                 np.array(sarrivals_res)])
                catalog.append(
                    [finaltime, p_x_final_old, p_y_final_old, p_z_final_old, minimum, 0.1, clean, xc, yc, zc, x_err,
                     y_err, z_err, theta, Q])
                unc_grid.append(ps_out)
                print([finaltime, p_x_final_old, p_y_final_old, p_z_final_old, minimum, 0.1, clean, xc, yc, zc, x_err,
                       y_err, z_err, theta, Q])
                continue
        else:
            print('No.' + str(event_no) + ' cannot be refined further, phases below threshold')
            arrivals.append([parrivals, sarrivals, parrivals_time, sarrivals_time, np.array(parrivals_res),
                             np.array(sarrivals_res)])
            catalog.append(
                [finaltime, p_x_final_old, p_y_final_old, p_z_final_old, minimum, 0.1, clean, xc, yc, zc, x_err, y_err,
                 z_err, theta, Q])
            print(
                [finaltime, p_x_final_old, p_y_final_old, p_z_final_old, minimum, 0.1, clean, xc, yc, zc, x_err, y_err,
                 z_err, theta, Q])
            continue
    return [events, catalog, arrivals, unc_grid]


# %%
@ray.remote
def multiprocess_events(i, sta_cat, phaseser, unique_events, ptraveltimes, straveltimes, lowq,
                        highq, Q_threshold, terr, outlier, x_spacing, y_spacing, z_spacing, 
                        coarsest_x_spacing, coarsest_y_spacing, coarsest_z_spacing, x_min, y_min, z_min, phase_threshold,
                        tt_fine_dir, zoom_factor):
#     for i in range(0,len(unique_events)):
    std_factor = 2

    p_out = []
    event_id = unique_events[i]
    print('...Initiating Event No. ' + str(event_id))

    pick_array = np.full((1, len(sta_cat), 2), -1.00)

    picks = phaseser[phaseser.base_id == unique_events[i]].reset_index(drop=True)

    picks = picks.drop_duplicates(subset=['sta', 'phase'],
                                  keep='first')  # ensures no duplicated phases are input into location algorithm

    p_picks = picks[picks.phase == 'P'][['sta', 'time']].copy(deep=True).drop_duplicates().reset_index(drop=True)
    p_picks.columns = ['sta', 'time']

    s_picks = picks[picks.phase == 's'][['sta', 'time']].copy(deep=True).drop_duplicates().reset_index(drop=True)
    s_picks.columns = ['sta', 'time']

    print('......P picks: '+str(len(p_picks))+', S picks: '+str(len(s_picks))+', Total: '+str(len(picks)))

    pick_array[0, :, 0] = pd.merge(sta_cat.reset_index()[['index', 'sta']], p_picks, how='left',
                                   on='sta')['time'].fillna(-1)
    pick_array[0, :, 1] = pd.merge(sta_cat.reset_index()[['index', 'sta']], s_picks, how='left',
                                   on='sta')['time'].fillna(-1)

    ponsets = pick_array[:, :, 0]
    sonsets = pick_array[:, :, 1]

#     start_time = timeit.time()
    origin = MAXI_locate_3D(event_id, ponsets, sonsets, ptraveltimes, straveltimes, lowq, highq, Q_threshold,
                            terr, outlier, x_spacing, y_spacing, z_spacing, 
                            coarsest_x_spacing, coarsest_y_spacing, coarsest_z_spacing, x_min, y_min, z_min, phase_threshold, tt_fine_dir,
                            sta_cat, zoom_factor, std_factor)
#     print("--- %s seconds ---" % (timeit.time() - start_time))

    if origin[2]:
        p_out = picks.copy()
        # Mask based on sta_id and arr_time; sometimes the arr_time can be the same at 2
        # stations
        p_mask = (p_out.time.isin(origin[2][0][2])) & (p_out.sta_index.isin(origin[2][0][0]) & (p_out.phase == 'P'))
        s_mask = (p_out.time.isin(origin[2][0][3])) & (p_out.sta_index.isin(origin[2][0][1]) & (p_out.phase == 's'))
        p_out.loc[p_mask, 'reloc'] = 'MAXI'
        p_out.loc[s_mask, 'reloc'] = 'MAXI'
        p_out.loc[p_mask, 't_res'] = origin[2][0][4]
        p_out.loc[s_mask, 't_res'] = origin[2][0][5]
        p_out.drop(columns=['index', 'time', 'sta_index', 'phase_count', 'base_id'], inplace=True)

    if origin[3]:
        unc_out = origin[3][0]
    else:
        unc_out = []

    return origin[0:2], p_out, event_id, unc_out


def get_phases(date, event_file, phase_df, mag_lower, mag_upper):
    import glob
    import pandas as pd
    import datetime as dt
    import os

    # load all rnn picks in folder
#     event_df = pd.read_csv(event_file, low_memory=False)
#     df = pd.read_csv(phase_file, low_memory=False)
    df = phase_df.copy()
#     # 	site_df = pd.read_csv(site_file,low_memory=False)
# 
#     # Subset the pick files based on the chosen date
#     event_df = event_df[(event_df.mag >= mag_lower) & (event_df.mag < mag_upper)]
#     event_df = event_df[(pd.to_datetime(event_df.datetime).astype('datetime64[ns]') >= date) & (
#             pd.to_datetime(event_df.datetime).astype('datetime64[ns]') < date + pd.Timedelta(24, unit='h'))]
#     event_list = event_df.evid.unique()
    df = df[df['evid'].isin(event_list)]

    df_p = df[df.phase.astype(str).str.lower().str[0] == 'p'].reset_index(drop=True)
    df_p['orig_phase'] = df_p.phase
    df_p['phase'] = 'P'
    df_p['time'] = (
            pd.to_datetime(df_p['datetime']).astype('datetime64[ns]') - dt.datetime(1970, 1, 1)).dt.total_seconds()

    df_s = df[df.phase.astype(str).str.lower().str[0] == 's'].reset_index(drop=True)
    df_s['orig_phase'] = df_s.phase
    df_s['phase'] = 's'
    df_s['time'] = (
            pd.to_datetime(df_s['datetime']).astype('datetime64[ns]') - dt.datetime(1970, 1, 1)).dt.total_seconds()

    white_list = [os.path.basename(x)[0:-6] for x in glob.glob('inputs/tt/*_P.hdf')]

    phases = pd.concat([df_p, df_s]).sort_values('time').reset_index(drop=True)
    phases = phases[phases.isin(white_list).sta == True].reset_index(drop=True)

    sta_cat = phases[['net', 'sta']].drop_duplicates().reset_index(drop=True)
    # 	sta_cat = sta_cat.merge(site_df[['sta','lat','lon','elev']],on='sta')
    # 	sta_cat['elev'] = sta_cat.elev/1000
    # 	sta_cat = phases[['net','sta','chan','lat','lon','elev']].drop_duplicates().reset_index(drop=True)
    sta_cat['sta'] = sta_cat.sta.str.strip()  # Remove unnecessary spaces in station name
    sta_cat_index = sta_cat['sta'].reset_index()

    phases = pd.merge(phases, sta_cat_index, on='sta', how='inner')
    phases.rename(columns={'index': 'sta_index'}, inplace=True)
    phases = phases.sort_values('time')
    phases = phases.reset_index(drop=True)

    return phases, sta_cat


def get_vmodels(sta_cat, zoom_factor):
    # 	import pykonal
    import numpy as np
    import h5py

    print('Loading p-traveltime models')
    p_files = []
    for sta in sta_cat['sta']:
        sta = sta.strip()
        file = sta + '_P.hdf'
        p_files.append(file)
    ptraveltimes = np.array([h5py.File('inputs/tt/' + fname, 'r')['values'] for fname in p_files])
    # 	ptraveltimes = np.array([pykonal.fields.read_hdf('inputs/tt/'+fname).values for fname in p_files])
    # 	vmod_attr = pykonal.fields.read_hdf('inputs/tt/'+p_files[0])
    x_min, y_min, z_min = np.array(h5py.File('inputs/tt/' + p_files[0], 'r')['min_coords'])
    x_spacing, y_spacing, z_spacing = np.array(h5py.File('inputs/tt/' + p_files[0], 'r')['node_intervals'])
    # 	ptraveltimes = np.array([np.load('inputs/tt/'+fname) for fname in p_files])

    print('Loading s-traveltime models')
    s_files = []
    for sta in sta_cat['sta']:
        sta = sta.strip()
        file = sta + '_S.hdf'
        s_files.append(file)
    straveltimes = np.array([h5py.File('inputs/tt/' + fname, 'r')['values'] for fname in s_files])

    return ptraveltimes, straveltimes, x_spacing, y_spacing, z_spacing

def get_vmodels_zoom(sta_cat,zoom_factor):
    import numpy as np
    import h5py

    p_zoom_files = []
    s_zoom_files = []

    print('Zooming p-traveltime models')
    p_files = []
    for sta in sta_cat['sta']:
        sta = sta.strip()
        file = sta + '_P.hdf'
        p_files.append(file)
        if not os.path.exists('inputs/tt_zoomed'):
            os.makedirs('inputs/tt_zoomed')
        if not os.path.exists('inputs/tt_zoomed/'+sta+'_P.npy'):
            with h5py.File('inputs/tt/' + file, 'r') as data:
                p_vel = data['values']
                p_zoom = zoom(p_vel,(zoom_factor,zoom_factor,zoom_factor))
                np.save('inputs/tt_zoomed/'+sta+'_P.npy',p_zoom)
                p_zoom_files.append(sta+'_P.npy')
        else:
            p_zoom_files.append(sta+'_P.npy')

    print('Zooming s-traveltime models')
    s_files = []
    for sta in sta_cat['sta']:
        sta = sta.strip()
        file = sta + '_S.hdf'
        s_files.append(file)
        if not os.path.exists('inputs/tt_zoomed'):
            os.makedirs('inputs/tt_zoomed')
        if not os.path.exists('inputs/tt_zoomed/'+sta+'_S.npy'):
            with h5py.File('inputs/tt/' + file, 'r') as data:
                s_vel = np.array(data['values'])
                s_zoom = zoom(s_vel,(zoom_factor,zoom_factor,zoom_factor))
                np.save('inputs/tt_zoomed/'+sta+'_S.npy',s_zoom)
                s_zoom_files.append(sta+'_S.npy')
        else:
            s_zoom_files.append(sta+'_S.npy')

    x_min, y_min, z_min = np.array(h5py.File('inputs/tt/' + p_files[0], 'r')['min_coords'])
    x_spacing, y_spacing, z_spacing = np.array(h5py.File('inputs/tt/' + p_files[0], 'r')['node_intervals'])

    print('Loading zoomed p-traveltime models')
    # 	p_zoom = zoom(ptraveltimes,(1,zoom_factor,zoom_factor,zoom_factor))
    p_zoom = np.array([np.load('inputs/tt_zoomed/'+fname) for fname in p_zoom_files])

    print('Loading zoomed s-traveltime models')
    # 	s_zoom = zoom(straveltimes,(1,zoom_factor,zoom_factor,zoom_factor))
    s_zoom = np.array([np.load('inputs/tt_zoomed/'+fname) for fname in s_zoom_files])

    return p_zoom, s_zoom, x_spacing, y_spacing, z_spacing


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




def main(date, event_list, phase_df, event_output_dir, arrival_output_dir, mag_lower, mag_upper, uncertainty_output_dir):
    import time as timeit
    import json

    # 	date = pd.to_datetime('2011-01-27')
    zoom_factor = 0.5  # Start with a coarser velocity model, grid spacing will be increased
    # by 1/zoom_factor, so that 1/0.5 will double grid spacing.
    terr = 0.7  # location terr, this results in stable solutions
    lowq = 1
    highq = 1
    Q_threshold = 0.3
    outlier = 0.7  # the time error allowable for an associated phase during location.
    phase_threshold = 4  # the minimum number of stations required for the event association phase
    time_window = 2  # the time +/- to search for matching phases for the event association phase. Greater time will allow more phases but may lead to issues in the location phase.
    x_min, y_min, z_min = -1200, -1200, -15  # First X, Y, Z coordinate in the v-model coordinate system

    orilat = -41.7638  # Origin latitude
    orilon = 172.9037  # Origin longitude
    angle = -140  # Counter-clockwise rotation for restoration of true coordinates
    tt_fine_dir = '/Volumes/SeaJade2/Pykonal/tt_fine'
  
    # Create phase list
    phases, sta_cat = get_phases(date, event_list, phase_df, mag_lower, mag_upper)

    if phases.size == 0:
        print('There are no phases for ' + str(date))
#         ray.shutdown()
        return
        
    # Get vmodels
    ptraveltimes, straveltimes, x_spacing, y_spacing, z_spacing = get_vmodels_zoom(sta_cat, zoom_factor)
    fine_x_spacing, fine_y_spacing, fine_z_spacing = 3, 3, 2
    coarsest_x_spacing, coarsest_y_spacing, coarsest_z_spacing = np.array((x_spacing, y_spacing, z_spacing)) / zoom_factor

    phase_counts = phases.groupby('evid').time.count().reset_index()
    phase_counts.rename(columns={'time': 'phase_count'}, inplace=True)

    phase_output = phases[phases.evid.isin(phase_counts.evid)]
    phase_output = pd.merge(phase_output, phase_counts, on='evid', how='right')
    phase_output['base_id'] = phase_output.evid

    phaseser = pd.merge(sta_cat.reset_index()[['index', 'sta']], phase_output, how='left',
                        left_on='index', right_on='sta_index').drop(columns='sta_y')

    phaseser.rename(columns={'sta_x': 'sta'}, inplace=True)

    unique_events = phaseser.base_id.unique()
    print('...Total unique events is ' + str(len(unique_events)))
    
    ray.init(object_store_memory=25000 * 1024 * 1024)
    ptraveltimes_id = ray.put(ptraveltimes)
    straveltimes_id = ray.put(straveltimes)

    ### Calculate event data
    result_ids = []
    start_time = timeit.time()
    print('--- Calculating event locations ---')
    for i in range(len(unique_events)):
#     for i in range(38,39):
        result_ids.append(multiprocess_events.remote(i, sta_cat, phaseser, unique_events,
                                                     ptraveltimes_id, straveltimes_id, lowq, highq, Q_threshold, terr,
                                                     outlier, x_spacing, y_spacing, z_spacing, coarsest_x_spacing, coarsest_y_spacing, coarsest_z_spacing, 
                                                     x_min, y_min, z_min, phase_threshold, tt_fine_dir, zoom_factor))
    origins = ray.get(result_ids)
    print('')
    print("--- %s seconds ---" % (timeit.time() - start_time))

    catalog_df = pd.DataFrame(columns=['x', 'y', 'z', 'datetime', 'minimum', 'finalgrid', 'ndef', 'evid', 'x_c', 'y_c',
                                       'z_c', 'major', 'minor', 'z_err', 'theta', 'Q'])
    arrival_df = pd.DataFrame(columns=['arid', 'net', 'sta', 'loc', 'chan', 'phase', 'datetime', 't_res', 'evid',
                                       'orig_phase'])
    uncertainty_data = {}

    for origin in origins:
        if len(origin[0][1]) != 0:

            if origin[0][1][0][5] != -1:  # bypass any low quality events. Should be very rare events.

                # return origin
                if origin[0][1][0][10] >= origin[0][1][0][11]:
                    major = origin[0][1][0][10]
                    minor = origin[0][1][0][11]
                else:
                    major = origin[0][1][0][11]
                    minor = origin[0][1][0][10]
                                        
                origin_data = {'x': origin[0][1][0][1], 'y': origin[0][1][0][2], 'z': origin[0][1][0][3],
                               'datetime': pd.to_datetime(origin[0][1][0][0], unit='s'), 'minimum': origin[0][1][0][4],
                               'finalgrid': origin[0][1][0][5], 'ndef': origin[0][1][0][6], 'evid': origin[2],
                               'x_c': origin[0][1][0][7], 'y_c': origin[0][1][0][8], 'z_c': origin[0][1][0][9],
                               'major': major, 'minor': minor, 'z_err': origin[0][1][0][12],
                               'theta': origin[0][1][0][13], 'Q': origin[0][1][0][14]}

                uncertainty_data[origin[2]] = origin[3].tolist()
                catalog_df = catalog_df.append(origin_data, True)
                arrival_df = arrival_df.append(origin[1], True)

        elif origin[0][0][0][0] == -1:

            print('Nothing here, pass')
            pass

        else:

            print('Skipping low quality event! ' + str(origin[2]))

    catalog_df['x'] = (catalog_df.x * fine_x_spacing) + x_min
    catalog_df['y'] = (catalog_df.y * fine_y_spacing) + y_min
    catalog_df['z'] = (catalog_df.z * fine_z_spacing) + z_min
    catalog_df['depth'] = catalog_df['z']
    catalog_df['lat'], catalog_df['lon'] = rotate_back(orilat, orilon, catalog_df.x.values,
                                                       catalog_df.y.values, angle)
#     catalog_df['x_c'] = (catalog_df.x_c * fine_x_spacing) + x_min
#     catalog_df['y_c'] = (catalog_df.y_c * fine_y_spacing) + y_min
#     catalog_df['z_c'] = (catalog_df.z_c * fine_z_spacing) + z_min
    catalog_df['y_c'], catalog_df['x_c'] = rotate_back(orilat, orilon, catalog_df.x_c.values,
                                                       catalog_df.y_c.values, angle)
    theta = list(catalog_df.loc[catalog_df['theta'] > 0, 'theta'].values)
    catalog_df.loc[catalog_df['theta'] > 0, 'theta'] = np.rad2deg(theta)
    catalog_df['theta'] = catalog_df['theta'] + angle
    catalog_df.loc[catalog_df['theta'] < 0, 'theta'] = catalog_df.loc[catalog_df['theta'] < 0, 'theta'] + 360

    nsta = arrival_df[arrival_df.reloc == 'MAXI'].groupby(['evid'])['sta'].nunique()
    for ider, evid in catalog_df.iterrows():
        catalog_df.loc[ider, 'nsta'] = nsta.loc[evid.evid]
    catalog_df['nsta'] = catalog_df['nsta'].astype('int')
    catalog_df['reloc'] = 'MAXI'

    if not os.path.exists('./output/' + event_output_dir):
        os.makedirs('./output/' + event_output_dir)
    if not os.path.exists('./output/' + arrival_output_dir):
        os.makedirs('./output/' + arrival_output_dir)
    if not os.path.exists('./output/' + uncertainty_output_dir):
        os.makedirs('./output/' + uncertainty_output_dir)

    if len(catalog_df) != 0:
        #
        catalog_df = catalog_df.sort_values('datetime').reset_index(drop=True)
        catalog_df = catalog_df[['evid', 'datetime', 'lat', 'lon', 'depth', 'ndef', 'nsta', 'reloc', 'minimum',
                                 'finalgrid', 'x', 'y', 'z', 'x_c', 'y_c', 'z_c', 'major', 'minor', 'z_err', 'theta',
                                 'Q']]
        arrival_df = arrival_df.sort_values('datetime').reset_index(drop=True)
        arrival_df = arrival_df[['arid', 'datetime', 'net', 'sta', 'loc', 'chan',
                                 'orig_phase', 'evid', 't_res', 'reloc']]
        arrival_df.rename(columns={'orig_phase': 'phase'}, inplace=True)

        catalog_df.to_csv('./output/' + event_output_dir + '/' + date.strftime('%Y%m%d') + '_origins.csv', index=None)
        arrival_df.to_csv('./output/' + arrival_output_dir + '/' + date.strftime('%Y%m%d') + '_arrivals.csv',
                          index=None)
        json = json.dumps(uncertainty_data)
        with open('./output/' + uncertainty_output_dir + '/' + date.strftime('%Y%m%d') + '_uncertainties.json','w') as f:
            f.write(json)
    ray.shutdown()


if __name__ == "__main__":
#     event_file = 'inputs/orig_events/synthetic_events.csv'
#     phase_file = 'inputs/phases/synthetic_phases_noise_04.csv'
    event_file = 'inputs/orig_events/earthquake_source_table_with_cmt.csv'
    phase_file = 'inputs/phases/phase_arrival_table_relocated.csv'
#     event_file = 'output/catalog_east_cape/20210304_origins.csv'
#     phase_file = 'output/associations_east_cape/20210304_assocs.csv'

    mag_lower = -999
    mag_upper = 10
    event_output_dir = 'catalog_test'
    arrival_output_dir = 'arrivals_test'
    uncertainty_output_dir = 'uncertainties_test'
    start_date = '2012-04-10'
    end_date = '2016-01-01'
    date_range = pd.date_range(start_date, end_date)
    event_df = pd.read_csv(event_file, low_memory=False)
    phase_df = pd.read_csv(phase_file, low_memory=False)

    # Subset the pick files based on the chosen date
    event_df = event_df[(event_df.mag >= mag_lower) & (event_df.mag < mag_upper)].reset_index(drop=True)
    
    for date in date_range:
        print('Relocating events from ' + str(date))
        event_check = event_df[(pd.to_datetime(event_df.datetime).astype('datetime64[ns]') >= date) & (
                pd.to_datetime(event_df.datetime).astype('datetime64[ns]') < date + pd.Timedelta(24, unit='h'))]
        event_list = event_check.evid.unique()
#         event_list = event_check.evid.unique()[0::]
        if event_list.size == 0:
            print('There are no events for ' + str(date))
            continue
        else:
            main(date, event_list, phase_df, event_output_dir, arrival_output_dir, mag_lower, mag_upper, uncertainty_output_dir)
