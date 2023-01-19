def TVZ_path_plot(df,polygon,line,nztm2wgs,r_epi,sta):
    import pygmt
    import matplotlib.pyplot as plt
    
    if line.intersection(polygon):
        line_points = line.intersection(polygon)
        tvz_length = line_points.length / 1000 / r_epi
        tvz_path_lat,tvz_path_lon = nztm2wgs.transform(line_points.xy[0],line_points.xy[1])
        tvz_path_lat = np.array(tvz_path_lat)
        tvz_path_lon = np.array(tvz_path_lon)
        tvz_path_lon[tvz_path_lon < 0] = tvz_path_lon[tvz_path_lon < 0] + 360

    shape_lat,shape_lon = nztm2wgs.transform(polygon.exterior.xy[0],polygon.exterior.xy[1])
    shape_lat = np.array(shape_lat)
    shape_lon = np.array(shape_lon)
    shape_lon[shape_lon < 0] = shape_lon[shape_lon < 0] + 360
    
    path_lat,path_lon = nztm2wgs.transform(line.xy[0],line.xy[1])
    path_lat = np.array(path_lat)
    path_lon = np.array(path_lon)
    path_lon[path_lon < 0] = path_lon[path_lon < 0] + 360
       
    region = [
    path_lon.min() - 1,
    path_lon.max() + 1.4,
    path_lat.min() - 1.5,
    path_lat.max() + 2.5
    ]
    
    fig = pygmt.Figure()
    fig.basemap(region=region, projection="M6i", frame=True)
    fig.coast(land="white", water="skyblue")
        
    fig.plot(shape_lon,shape_lat,color='orange')
    fig.plot(x=path_lon,y=path_lat,pen='1p,blue')
    if line.intersection(polygon):
        fig.plot(x=tvz_path_lon,y=tvz_path_lat,pen='1p,red')
    fig.plot(path_lon[0],path_lat[0],color='white',style='c0.4c',pen='1p')
    fig.text(x=path_lon[0],y=path_lat[0],text=event_id,offset='0.5c')
    fig.plot(path_lon[1],path_lat[1],color='pink',style='t0.4c',pen='1p')
    fig.text(x=path_lon[1],y=path_lat[1],text=sta.sta,offset='0.5c')
    fig.show(method="external")