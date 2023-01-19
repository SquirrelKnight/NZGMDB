# MRD, 20200331
# Python module to do visualization with transects

# Import locally saved packages and modules
from tectclass import *

cwd = os.getcwd()
out_path = cwd + '/out'
makes_dir(out_path)

df = pd.read_csv(
    '\\'.join(
        cwd.split('\\')[:-1] + 
        ['flatfiles','geonet_df.csv']
        ),
    low_memory=False,
)

R_a, R_a_syn, R_b, R_c, R_abc = xyz_fault_points(
    fault_file=sub + '/Hik_Williams_2012/new_charles_low_res.txt', # Williams, 2012
    sep='\s+',
    d_s=10, # Williams et al., 2012
    d_d=47, # Williams et al., 2012
    )

R_a, R_a_syn, R_b, R_c, R_abc = xyz_fault_points(
    fault_file=sub + '/Slab2_2018/ker/ker_slab2_dep_02.24.18.xyz', # Hayes et al., 2018
    sep=',',
    d_s=10, # Hayes et al., 2018
    d_d=47, # Hayes et al., 2018
    R_a = R_a,
    R_a_syn = R_a_syn,
    R_b = R_b,
    R_c = R_c,
    R_abc = R_abc,
    )

R_a, R_a_syn, R_b, R_c, R_abc = xyz_fault_points(
    fault_file=sub + '/Slab2_2018/puy/puy_slab2_dep_02.26.18.xyz', # Hayes et al., 2018
    sep=',',
    d_s=11, # Hayes et al., 2018
    d_d=30, # Hayes et al., 2018
    R_a = R_a,
    R_a_syn = R_a_syn,
    R_b = R_b,
    R_c = R_c,
    R_abc = R_abc,
    )

fault_dict = {
    'abc':R_abc,
    'a':R_a,
    'b': R_b,
    'c': R_c,
    # 'a_syn': R_a_syn,
}



transect_dict = {
    # "A-A'":((174,-38),(179,-41)),
    # "B-B'":((172,-39),(176,-43)),
    # "C-C'":((165,-45),(168.5,-48.5)),
    # "D-D'":((163,-47.5),(167,-49)),
    # "E-E'":((162.5,-50),(180,-35)),

    # "A-A'":((179,-33),(183,-37)),
    # "B-B'":((178,-34),(182,-38)),
    # "C-C'":((177,-35),(181,-39)),
    # "D-D'":((176,-36),(180,-40)),
    # "E-E'":((175,-37),(179,-41)),
    # "F-F'":((174,-38),(178,-42)),
    # "G-G'":((173,-39),(177,-43)),
    # "H-H'":((172,-40),(176,-44)),
    # "I-I'":((171,-41),(175,-45)),
    # "J-J'":((170,-42),(174,-46)),
    # "K-K'":((169,-43),(173,-47)),
    # "L-L'":((167,-43),(171,-47)),
    # "M-M'":((166,-44),(170,-48)),
    # "N-N'":((165,-45),(169,-49)),
    # "O-O'":((164,-46),(168,-50)),
    # "P-P'":((163,-47),(167,-51)),
    # "Q-Q'":((162,-48),(166,-52)),

    # "A-A'":((177,-33),(183,-37)),
    # "B-B'":((176,-34),(182,-38)),
    # "C-C'":((175,-35),(181,-39)),
    # "D-D'":((174,-36),(180,-40)),
    # "E-E'":((173,-37),(179,-41)),
    # "F-F'":((172,-38),(178,-42)),
    # "G-G'":((171,-39),(177,-43)),
    # "H-H'":((170,-40),(176,-44)),
    # "I-I'":((169,-41),(175,-45)),
    # "J-J'":((168,-42),(174,-46)),
    # "K-K'":((167,-43),(173,-47)),
    # "L-L'":((165,-43),(171,-47)),
    # "M-M'":((164,-44),(170,-48)),
    # "N-N'":((163,-45),(169,-49)),
    # "O-O'":((162,-46),(168,-50)),
    # "P-P'":((161,-47),(167,-51)),
    # "Q-Q'":((160,-48),(166,-52)),

    # Concise for journal letter size page
    # "A-A'":((174,-36),(180,-40)),
    # "B-B'":((172,-38),(178,-42)),
    # "C-C'":((170,-40),(176,-44)),
    # "D-D'":((164,-44),(170,-48)),
    # "E-E'":((163,-45),(169,-49)),
    # "F-F'":((162,-46),(168,-50)),

    # One per subduction surface
    "Hik-Hik'":((172,-38),(178,-42)),
    "Puy-Puy'":((163,-45),(169,-49)),
}

marker_dict = {
   'Crustal': '1',
   'Interface': '2',
   'Outer-rise': '3',
   'Slab': '4',
   'Slab-(NGA-Interface)': '+',
   'Undetermined': 'x',
}

colour_dict = {
   'Crustal':'g',
   'Interface':'r',
   'Outer-rise':'b',
   'Slab':'c',
   'Slab-(NGA-Interface)':'m',
   'Undetermined':'k',
}


###############################################################################
# SCRIPT RUNNING
###############################################################################

if __name__ == "__main__":




    # for tecttype, df_a  in df.groupby(['NGASUB_TectClass']):
    #     df_a.drop_duplicates(subset=['A_Source'],keep='first',inplace=True)
    vis = VisualizeData(data = df, path=out_path)

    # vis.transect_scatter(
    #     source_long='A_Pos_Longitude',
    #     source_lat='A_Latitude', 
    #     source_depth='A_Depth',
    #     markerby='A_TectClass',
    #     fault_dict=fault_dict,
    #     shown_faults=['puy_slab2_dep_02.26.18','new_charles_low_res'],
    #     shown_regions=['a','b','c'],
    #     transect_dict=transect_dict,
    #     transect_thickness=50,
    #     marker_opacity=1.0, 
    #     cmin=0, 
    #     cmax=300,
    #     longmin=geo_centre[1] - 12,
    #     longmax=geo_centre[1] + 8,
    #     latmin=geo_centre[0] - 10,
    #     latmax=geo_centre[0] + 10, 
    #     depthmin=500,
    #     depthmax=1,
    #     yscale='log',
    #     cmap = cmap_viridis_r,
    #     colours = colour_dict,
    #     markers = marker_dict,
    #     extra='A_TectClass',
    #     )

    # vis.transect_scatter(
    #     source_long='A_Pos_Longitude',
    #     source_lat='A_Latitude', 
    #     source_depth='A_Depth',
    #     markerby='NGASUB_TectClass_HikPuy',
    #     fault_dict=fault_dict,
    #     shown_faults=['puy_slab2_dep_02.26.18','new_charles_low_res'],
    #     shown_regions=['a','b','c'],
    #     transect_dict=transect_dict,
    #     transect_thickness=50,
    #     marker_opacity=1.0, 
    #     cmin=0, 
    #     cmax=300,
    #     longmin=geo_centre[1] - 12,
    #     longmax=geo_centre[1] + 8,
    #     latmin=geo_centre[0] - 10,
    #     latmax=geo_centre[0] + 10, 
    #     depthmin=500,
    #     depthmax=1,
    #     yscale='linear',
    #     cmap = cmap_viridis_r,
    #     colours = colour_dict,
    #     markers = marker_dict,
    #     extra='NGA-SUB using Hikurangi',
    #     )

    # vis.transect_scatter(
    #     source_long='A_Pos_Longitude',
    #     source_lat='A_Latitude', 
    #     source_depth='A_Depth',
    #     markerby='NGASUB_TectClass_HikPuy',
    #     fault_dict=fault_dict,
    #     shown_faults=['puy_slab2_dep_02.26.18','new_charles_low_res'],
    #     shown_regions=['a','b','c'],
    #     transect_dict=transect_dict,
    #     transect_thickness=50,
    #     marker_opacity=1.0, 
    #     cmin=0, 
    #     cmax=300,
    #     longmin=geo_centre[1] - 12,
    #     longmax=geo_centre[1] + 8,
    #     latmin=geo_centre[0] - 10,
    #     latmax=geo_centre[0] + 10, 
    #     depthmin=500,
    #     depthmax=1,
    #     yscale='log',
    #     cmap = cmap_viridis_r,
    #     colours = colour_dict,
    #     markers = marker_dict,
    #     extra='NGA-SUB using Hikurangi (log)',
    #     )

    # vis.transect_scatter(
    #     source_long='A_Pos_Longitude',
    #     source_lat='A_Latitude', 
    #     source_depth='A_Depth',
    #     markerby='NGASUB_TectClass_KerPuy',
    #     fault_dict=fault_dict,
    #     shown_faults=['puy_slab2_dep_02.26.18','ker_slab2_dep_02.24.18'],
    #     shown_regions=['a','b','c'],
    #     transect_dict=transect_dict,
    #     transect_thickness=50,
    #     marker_opacity=1.0, 
    #     cmin=0, 
    #     cmax=300,
    #     longmin=geo_centre[1] - 12,
    #     longmax=geo_centre[1] + 8,
    #     latmin=geo_centre[0] - 10,
    #     latmax=geo_centre[0] + 10, 
    #     depthmin=500,
    #     depthmax=1,
    #     yscale='linear',
    #     cmap = cmap_viridis_r,
    #     colours = colour_dict,
    #     markers = marker_dict,
    #     extra='NGA-SUB using Kermadec',
    #     )

    # vis.transect_scatter(
    #     source_long='A_Pos_Longitude',
    #     source_lat='A_Latitude', 
    #     source_depth='A_Depth',
    #     markerby='NGASUB_TectClass_KerPuy',
    #     fault_dict=fault_dict,
    #     shown_faults=['puy_slab2_dep_02.26.18','ker_slab2_dep_02.24.18'],
    #     shown_regions=['a','b','c'],
    #     transect_dict=transect_dict,
    #     transect_thickness=50,
    #     marker_opacity=1.0, 
    #     cmin=0, 
    #     cmax=300,
    #     longmin=geo_centre[1] - 12,
    #     longmax=geo_centre[1] + 8,
    #     latmin=geo_centre[0] - 10,
    #     latmax=geo_centre[0] + 10, 
    #     depthmin=500,
    #     depthmax=1,
    #     yscale='log',
    #     cmap = cmap_viridis_r,
    #     colours = colour_dict,
    #     markers = marker_dict,
    #     extra='NGA-SUB using Kermadec (log)',
    #     )

    # vis.transect_scatter(
    #     source_long='A_Pos_Longitude',
    #     source_lat='A_Latitude', 
    #     source_depth='A_Depth',
    #     markerby='NZSMDB_TectClass',
    #     fault_dict=fault_dict,
    #     shown_faults=['puy_slab2_dep_02.26.18','ker_slab2_dep_02.24.18','new_charles_low_res'],
    #     shown_regions=['abc'],
    #     transect_dict=transect_dict,
    #     transect_thickness=50,
    #     marker_opacity=1.0, 
    #     cmin=0, 
    #     cmax=300,
    #     longmin=geo_centre[1] - 12,
    #     longmax=geo_centre[1] + 8,
    #     latmin=geo_centre[0] - 10,
    #     latmax=geo_centre[0] + 10, 
    #     depthmin=300,
    #     depthmax=-10,
        # yscale='log',
    #     cmap = cmap_viridis_r,
    #     colours = colour_dict,
    #     markers = marker_dict,
    #     extra='NZ-SMDB with Kermadec and Hikurangi alternatives',
    #     )

    vis.transect_scatter(
        source_long='A_Pos_Longitude',
        source_lat='A_Latitude', 
        source_depth='A_Depth',
        markerby='A_TectClass',
        backgrounds = ['north_island_bw_path','south_island_bw_path'],
        fault_dict=fault_dict,
        shown_faults=['puy_slab2_dep_02.26.18','new_charles_low_res'],
        shown_regions=['a','b','c'],
        transect_dict=transect_dict,
        transect_thickness=50,
        marker_size=10,
        marker_opacity=1, 
        extra='A_TectClass',
        )







