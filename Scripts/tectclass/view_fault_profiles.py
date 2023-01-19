# MRD, 20200331
# Python script to do visualization of the NZHM subduction faults

# Import locally saved packages and modules
from tectclass import *

########################################################################################################################
# SCRIPT RUNNING
########################################################################################################################


if __name__ == "__main__":

    out_path = cwd + '/out'
    makes_dir(out_path)

    # Load in the fault geometries the  2013 NZ fault atlas
    sub_faults = nhm.load_nhm(nhm_path=sub_surface_path + 'NZSHM_2013/subfaults_NZ_2013.txt',skiprows=14)
    f_p, o_p, f_s = fault_points(sub_faults)


    #########################################################################################################
    # PRODUCE PLOTS
    #########################################################################################################

    # Checking the locations of the proj/surface points
    df_fp = pd.DataFrame()
    i_t = False
    for k, i in f_p.items():
        i_t = np.transpose(i)
        temp = pd.DataFrame()
        temp['A_Pos_Longitude'] = list(i_t[0][:])
        temp['A_Latitude'] = list(i_t[1][:])
        df_fp = df_fp.append(temp)
    df_fp['A_Depth'] = 0
    df_fp['A_Magnitude'] = 3

    vis = VisualizeData(data = df_fp, path=out_path)
    vis.scatter_histogram(['A_Pos_Longitude','A_Latitude'], 
            marker_scale='A_Magnitude',
            marker_colour_scale='A_Depth', 
            marker_opacity=0.5, 
            xbinmax=200, 
            ybinmax=200, 
            cmin=0, 
            cmax=200,
            xmax=geo_centre[1] + 10, 
            xmin=geo_centre[1] - 10, 
            ymax=geo_centre[0] + 10,
            ymin=geo_centre[0] - 10, 
            cmap = cmap_viridis_r,
            extra='fault_proj',
            )

    df_op = pd.DataFrame()
    i_t = False
    for k, i in o_p.items():
        i_t = np.transpose(i)
        temp = pd.DataFrame()
        temp['A_Pos_Longitude'] = list(i_t[0][:])
        temp['A_Latitude'] = list(i_t[1][:])
        df_op = df_op.append(temp)
    df_op['A_Depth'] = 100
    df_op['A_Magnitude'] = 3

    vis = VisualizeData(data = df_op, path=out_path)
    vis.scatter_histogram(['A_Pos_Longitude','A_Latitude'], 
            marker_scale='A_Magnitude',
            marker_colour_scale='A_Depth', 
            marker_opacity=0.5, 
            xbinmax=200, 
            ybinmax=200, 
            cmin=0, 
            cmax=200,
            xmax=geo_centre[1] + 10, 
            xmin=geo_centre[1] - 10, 
            ymax=geo_centre[0] + 10,
            ymin=geo_centre[0] - 10, 
            cmap = cmap_viridis_r,
            extra='offshore_proj',
            )

    df_fs = pd.DataFrame()
    i_t = False
    for k, i in f_s.items():
        i_t = np.transpose(i)
        temp = pd.DataFrame()
        # temp['A_Pos_Longitude'] = list(i_t[0][:])
        temp['A_Depth'] = list(i_t[2][:])
        df_fs = df_fs.append(temp)
    # df_fs['A_Depth'] = 100
    # df_fs['A_Magnitude'] = 3

    df_fs = pd.concat([df_fs,df_fp[['A_Pos_Longitude','A_Latitude','A_Magnitude']]],axis=1)

    vis = VisualizeData(data = df_fs, path=out_path)
    vis.scatter_histogram(['A_Pos_Longitude','A_Latitude'], 
            marker_scale='A_Magnitude',
            marker_colour_scale='A_Depth', 
            marker_opacity=0.5, 
            xbinmax=200, 
            ybinmax=200, 
            cmin=0, 
            cmax=50,
            xmax=geo_centre[1] + 10, 
            xmin=geo_centre[1] - 10, 
            ymax=geo_centre[0] + 10,
            ymin=geo_centre[0] - 10, 
            cmap = cmap_viridis_r,
            extra='surface',
            )
