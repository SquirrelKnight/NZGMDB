import pandas as pd
import os
import glob
import pygmt
import matplotlib.pyplot as plt
from matplotlib import rc
from obspy.geodetics import kilometers2degrees
rc("pdf",fonttype=42)

event_df = pd.read_csv(directory+'inputs/orig_events/earthquake_source_table_relocated.csv',low_memory=False)
# event_df['depth'] = event_df['z']
event_df['evid'] = event_df['evid'].astype('str')

revised_df = pd.concat([pd.read_csv(f,low_memory=False) for f in glob.glob(directory+'output/catalog_all/*_origins.csv')])
revised_df['evid'] = revised_df.evid.astype('str')
revised_df = revised_df.merge(event_df[['evid','lat','lon','depth']],on='evid',suffixes = (None,'_orig'))


# error_df['x_err'] = error_df['x_err'] * 2
# error_df['y_err'] = error_df['y_err'] * 2
# error_df = revised_df[['x_c','y_c','theta','x_err','y_err']].reset_index(drop=True)
# error_df['x_err'] = kilometers2degrees(error_df['x_err'] * 10)
# error_df['y_err'] = kilometers2degrees(error_df['y_err'] * 10)
revised_df.loc[revised_df.x_err <= revised_df.y_err,'major'] = revised_df.loc[revised_df.x_err <= revised_df.y_err,'y_err']
revised_df.loc[revised_df.x_err > revised_df.y_err,'major'] = revised_df.loc[revised_df.x_err > revised_df.y_err,'x_err']
revised_df.loc[revised_df.x_err >= revised_df.y_err,'minor'] = revised_df.loc[revised_df.x_err >= revised_df.y_err,'y_err']
revised_df.loc[revised_df.x_err < revised_df.y_err,'minor'] = revised_df.loc[revised_df.x_err < revised_df.y_err,'x_err']
revised_df['major'] = revised_df['x_err']
revised_df['minor'] = revised_df['y_err']

import numpy as np
import shapely.geometry as geom
from shapely import affinity
from pyproj import Transformer			# conda install pyproj

revised_df['major'] = error_df['major']
revised_df['minor'] = error_df['minor']

wgs2nztm = Transformer.from_crs(4326, 2193, always_xy=True)

for idx,event in revised_df.iterrows():
    n = 360
    r_x,r_y = wgs2nztm.transform(event.x_c,event.y_c)
    o_x,o_y = wgs2nztm.transform(event.lon,event.lat)
    x_diff = (r_x - o_x) / 1000
    y_diff = (r_y - o_y) / 1000
    a = event.copy().major
    b = event.copy().minor
    angle = event.theta
    theta = np.linspace(0, np.pi*2, n)
    r = a * b  / np.sqrt((b * np.cos(theta))**2 + (a * np.sin(theta))**2)
    exy = np.stack([r * np.cos(theta), r * np.sin(theta)], 1)

    ellipse = affinity.rotate(geom.Polygon(exy), angle, 'center')
    ex, ey = ellipse.exterior.xy
    
    p = [x_diff,y_diff]
    if ellipse.contains(geom.Point(p)):
#         print(event.evid + 'moo')
        continue
    else:
#         print(x_diff,y_diff,'meow')
        if abs(y_diff) > 100 or abs(x_diff) > 100:
            print(event.evid,event.datetime,event.major,event.minor,x_diff,y_diff,event.y_c,event.x_c,event.lat,event.lon)
    
            import matplotlib.pyplot as plt
            plt.plot(ex, ey, lw = 1, color='k')
            plt.scatter(x_diff,y_diff)
            plt.show()
