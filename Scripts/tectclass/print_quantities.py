# MRD, 20200331
# Python module to do visualization and analysis of actual and predicted labels

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import standard packages and modules
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import shutil
from datetime import timedelta, datetime
import glob
from math import radians, cos, sin, asin, sqrt
from pandas import HDFStore
from pathlib import Path
import sys, os
import csv
import scipy
from scipy.fftpack import fft
from scipy import interpolate
from pylab import *

# Import locally saved packages and modules
from utility_funcs import makes_dir


########################################################################################################################
# SCRIPT RUNNING
########################################################################################################################


if __name__ == "__main__":

    # Get root directory, output path, and set global variables
    cwd = os.getcwd()

    out_path = cwd + '/out/df_details'
    makes_dir(out_path)


    df = pd.read_csv(
        '\\'.join(
            cwd.split('\\')[:-1] + 
            ['flatfiles','geonet_df.csv']
            ),
        low_memory=False,
    )

    print('\nMiscellaneous Quantities')
    print('Records: ',len(df.A_Record.unique()))
    print('Events: ',len(df.A_Source.unique()))
    print('Stations: ',len(df.A_Site.unique()))

    print('\nRecords by tecttype')
    for tecttype in df.A_TectClass.unique():
        print('{}: {}'.format(
            tecttype,
            len(df[df.A_TectClass == tecttype])
            ))


    print('\nRecords by instrument type')
    for instrument in df.A_Instrument.unique():
        print('{}: {}'.format(
            instrument,
            len(df[df.A_Instrument == instrument])
            ))

    print('\nEvents by tecttype')
    for tecttype in df.A_TectClass.unique():
        print('{}: {}'.format(
            tecttype,
            len(df[df.A_TectClass == tecttype].A_Source.unique())
            ))




