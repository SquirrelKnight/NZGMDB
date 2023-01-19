# MRD, 20200331
# Python module to do visualization with transects

# Import locally saved packages and modules
from tectclass import *
# plt.style.use('labelvis')
plt.style.use('presentation')

########################################################################################################################
# SCRIPT RUNNING
########################################################################################################################


if __name__ == "__main__":

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

    # for tecttype, df_a  in df.groupby(['NGASUB_TectClass']):
    #     df_a.drop_duplicates(subset=['A_Source'],keep='first',inplace=True)
    vis = VisualizeData(
        data = df, 
        path=out_path,
        )

    vis.pie_chart()

    # vis.scatter_histogram(
    #     ['A_Depth', 'A_Magnitude'],
    #     marker_size=5,
    #     marker_opacity=1,
    #     xscale='linear',
    #     xmax=600,
    #     xmin=0,
    #     xbinmax=10000,
    #     xbinwidth=10,
    #     ymin=3,
    #     ymax=8,
    #     ybinmax=10000,
    #     ybinwidth=0.1,
    #     )


    # vis.scatter_histogram(
    #     ['A_Duration', 'Record_PGA_Vertical_g'],
    #     marker_size=5,
    #     marker_opacity=1,
    #     xscale='linear',
    #     xmax=500,
    #     xmin=0,
    #     xbinmax=10000,
    #     ymin=0,
    #     ymax=100,
    #     ybinmax=10000,
    #     )

    # vis.scatter_histogram(
    #     ['A_Depth', 'A_Magnitude'],
    #     marker_scale='A_Magnitude',
    #     marker_colour_scale='CMT_Depth',
    #     marker_opacity=0.5,
    #     xbinmax=200,
    #     ybinmax=200,
    #     cmin=0,
    #     cmax=200,
    #     xmax=300,
    #     xmin=-300,
    #     ymax=0,
    #     ymin=-600,
    #     extra='',
    #     )
