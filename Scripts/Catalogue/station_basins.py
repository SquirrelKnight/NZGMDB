"""
A script to check what stations are contained within basin outlines.

First argument is the station file and the subsequent arguments are N basin file outlines.
"""

import argparse
from pathlib import Path
import pandas as pd

import matplotlib.path as mpltPath
import numpy as np

from qcore import formats

def get_basin_full_path(basin_list):
    vm_dir = Path('/Users/jesse/Downloads/Velocity-Model-development')
    return [vm_dir / basin for basin in basin_list]


v203_basin_list = get_basin_full_path(
    [
        "Data/Basins/Wellington/v19p6/Wellington_Polygon_Wainuiomata_WGS84.txt",
        "Data/Boundaries/NewCanterburyBasinBoundary_WGS84_1m.txt",
        "Data/Boundaries/BPVBoundary.txt",
        "Data/SI_BASINS/Cheviot_Polygon_WGS84.txt",
        "Data/SI_BASINS/Hanmer_Polygon_WGS84.txt",
        "Data/SI_BASINS/Kaikoura_Polygon_WGS84.txt",
        "Data/SI_BASINS/Marlborough_Polygon_WGS84_v0p1.txt",
        "Data/SI_BASINS/Nelson_Polygon_WGS84.txt",
        "Data/SI_BASINS/NorthCanterbury_Polygon_WGS84.txt",
        "Data/Boundaries/WaikatoHaurakiBasinEdge_WGS84.txt",
    ]
)
v204_basin_list = v203_basin_list + get_basin_full_path(
    [
        "Data/USER20_BASINS/WanakaOutlineWGS84.txt",
        "Data/USER20_BASINS/WakatipuBasinOutlineWGS84.txt",
        "Data/USER20_BASINS/alexandra_outline.txt",
        "Data/USER20_BASINS/ranfurly_outline.txt",
        "Data/USER20_BASINS/NE_otago/NE_otago_A_outline.txt",
        "Data/USER20_BASINS/NE_otago/NE_otago_B_outline.txt",
        "Data/USER20_BASINS/NE_otago/NE_otago_C_outline.txt",
        "Data/USER20_BASINS/NE_otago/NE_otago_D_outline.txt",
        "Data/USER20_BASINS/NE_otago/NE_otago_E_outline.txt",
        "Data/USER20_BASINS/bal_outline_WGS84.txt",
        "Data/USER20_BASINS/dun_outline_WGS84.txt",
        "Data/USER20_BASINS/mos_outline_WGS84.txt",
        "Data/USER20_BASINS/Murchison_Basin_Outline_v1_WGS84.txt",
        "Data/USER20_BASINS/hakataramea_outline_WGS84.txt",
        "Data/USER20_BASINS/waitaki_outline_WGS84.txt",
        "Data/USER20_BASINS/mackenzie_basin_outline_nzmg.txt",
    ]
)
v205_basin_list = v204_basin_list
v206_basin_list = v205_basin_list + get_basin_full_path(
    [
        "Data/USER20_BASINS/SpringsJ_basin_outline_v1_WGS84.txt",
        "Data/USER20_BASINS/Karamea_basin_outline_v1_WGS84.txt",
        "Data/USER20_BASINS/CollingwoodBasinOutline_1_WGS84_v1.txt",
        "Data/USER20_BASINS/CollingwoodBasinOutline_2_WGS84_v1.txt",
        "Data/USER20_BASINS/CollingwoodBasinOutline_3_WGS84_v1.txt",
    ]
)

# Wellington Basin update has the same outline so have not updated it for this
v207_basin_list = v206_basin_list + get_basin_full_path(
    [
        "Data/Basins/Greater_Wellington_and_Porirua/v21p7/GreaterWellington1_Outline_WGS84.dat",
        "Data/Basins/Greater_Wellington_and_Porirua/v21p7/GreaterWellington2_Outline_WGS84.dat",
        "Data/Basins/Greater_Wellington_and_Porirua/v21p7/GreaterWellington3_Outline_WGS84.dat",
        "Data/Basins/Greater_Wellington_and_Porirua/v21p7/GreaterWellington4_Outline_WGS84.dat",
        "Data/Basins/Greater_Wellington_and_Porirua/v21p7/GreaterWellington5_Outline_WGS84.dat",
        "Data/Basins/Greater_Wellington_and_Porirua/v21p7/GreaterWellington6_Outline_WGS84.dat",
        "Data/Basins/Greater_Wellington_and_Porirua/v21p7/Porirua1_Outline_WGS84.dat",
        "Data/Basins/Greater_Wellington_and_Porirua/v21p7/Porirua2_Outline_WGS84.dat",
        "Data/Basins/Napier_Hawkes_Bay/v21p7/HawkesBay1_Outline_WGS84_delim.dat",
        "Data/Basins/Napier_Hawkes_Bay/v21p7/HawkesBay2_Outline_WGS84_delim.dat",
        "Data/Basins/Napier_Hawkes_Bay/v21p7/HawkesBay3_Outline_WGS84_delim.dat",
        "Data/Basins/Napier_Hawkes_Bay/v21p7/HawkesBay4_Outline_WGS84_delim.dat",
        "Data/Basins/Napier_Hawkes_Bay/v21p7/Napier1_Outline_WGS84_delim.dat",
        "Data/Basins/Napier_Hawkes_Bay/v21p7/Napier2_Outline_WGS84_delim.dat",
        "Data/Basins/Napier_Hawkes_Bay/v21p7/Napier3_Outline_WGS84_delim.dat",
        "Data/Basins/Napier_Hawkes_Bay/v21p7/Napier4_Outline_WGS84_delim.dat",
        "Data/Basins/Napier_Hawkes_Bay/v21p7/Napier5_Outline_WGS84_delim.dat",
        "Data/Basins/Napier_Hawkes_Bay/v21p7/Napier6_Outline_WGS84_delim.dat",
    ]
)
basin_outlines = {
    "2.03": v203_basin_list,
    "2.04": v204_basin_list,
    "2.05": v205_basin_list,
    "2.06": v206_basin_list,
    "2.07": v207_basin_list,
}

stat_file = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/testaroo/site_table_response.csv',low_memory=False)
# stat_file['basin'] = None
paths = []
for outline_fp in v207_basin_list:
    outline = np.loadtxt(outline_fp)

    path = mpltPath.Path(outline)

    for idx,stat in stat_file.iterrows():
        if path.contains_point((stat.lon, stat.lat)):
#             print(stat.lon, stat.lat, stat.sta, True, outline_fp.name)
            basin_name = outline_fp.name.split('_')[0].split('.')[0]
            stat_file.loc[idx,'basin'] = basin_name
            print(stat.sta, basin_name)
stat_file.loc[stat_file.basin == 'NewCanterburyBasinBoundary','basin'] = 'Canterbury'
stat_file.loc[stat_file.basin == 'BPVBoundary','basin'] = 'Banks Peninsula volcanics'
stat_file.loc[stat_file.basin == 'waitaki','basin'] = 'Waitaki'
stat_file.loc[stat_file.basin == 'Napier1','basin'] = 'Napier'
stat_file.loc[stat_file.basin == 'mackenzie','basin'] = 'Mackenzie'
stat_file.loc[stat_file.basin == 'NorthCanterbury','basin'] = 'North Canterbury'
stat_file.loc[stat_file.basin == 'dun','basin'] = 'Dun'
stat_file.loc[stat_file.basin == 'WakatipuBasinOutlineWGS84','basin'] = 'Wakatipu'
stat_file.loc[stat_file.basin == 'WaikatoHaurakiBasinEdge','basin'] = 'Waikato Hauraki'
stat_file.loc[stat_file.basin == 'HawkesBay1','basin'] = 'Hawkes Bay'
stat_file.loc[stat_file.basin == 'WanakaOutlineWGS84','basin'] = 'Wanaka'
stat_file.loc[stat_file.basin == 'Porirua1','basin'] = 'Porirua'
stat_file.loc[stat_file.basin == 'SpringsJ','basin'] = 'Springs Junction'
stat_file.loc[stat_file.basin == 'CollingwoodBasinOutline','basin'] = 'Collingwood'
stat_file.loc[stat_file.basin == 'GreaterWellington4','basin'] = 'Greater Wellington'

stat_file.to_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/testaroo/site_table_basin.csv',index=False)

#         else:
#             print(stat.lon, stat.lat, stat.sta, False, None)
