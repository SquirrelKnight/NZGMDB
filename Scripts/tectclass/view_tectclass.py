# MRD, 20200331

# Import locally saved packages and modules
from tectclass import *

########################################################################################################################
# SCRIPT RUNNING
########################################################################################################################


if __name__ == "__main__":

    out_path = cwd + '/out'
    makes_dir(out_path)

    df = pd.read_csv(
        '\\'.join(
            cwd.split('\\')[:-1] + 
            ['flatfiles','geonet_df.csv']
            )
    )

    for tecttype, df_a  in df.groupby(['NGASUB_TectClass']):
        df_a.drop_duplicates(subset=['A_Source'],keep='first',inplace=True)
        vis = VisualizeData(data = df_a, path=out_path)

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
            extra=str(tecttype),
            )

        vis.scatter_histogram(['CMT_X_Rot', 'A_Source_Elevation'], 
            marker_scale='A_Magnitude', 
            marker_colour_scale='A_Depth', 
            marker_opacity=0.5, 
            xbinmax=200,
            ybinmax=200,
            cmin=0,
            cmax=200,
            xmax=300,
            xmin=-300,
            ymax=0,
            ymin=-600,
            cmap = cmap_viridis_r,
            extra=str(tecttype),
            )

