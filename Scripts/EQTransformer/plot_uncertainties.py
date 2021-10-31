def plot_uncertainty(data_in,event):
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize = (8,8))
    ax = plt.axes(projection='3d')
    ax.grid()
    ax.scatter(data_in[1],data_in[2],data_in[3],c=data_in[0],cmap = plt.cm.viridis)
    ax.scatter(event.x,event.y,event.z,c='r',marker='*',s=200)
    ax.invert_zaxis()
    plt.show()        

def show_uncertainty(input_uncertainties_file,input_events_file,evid):
    import pandas as pd
    import json
    import numpy as np
    
    input_events = pd.read_csv(input_events_file,low_memory=False)
    with open(input_uncertainties_file,'r') as f:
        unc_data = json.load(f)
    
    evid = str(evid)
    event = input_events[input_events.evid == evid].iloc[0]
    data_in = np.array(unc_data[evid])
    print(evid,event.ndef)
    plot_uncertainty(data_in,event)

def show_uncertainties(input_uncertainties_file,input_events_file):
    import pandas as pd
    import json
    import numpy as np
    
    input_events = pd.read_csv(input_events_file,low_memory=False)
    with open(input_uncertainties_file,'r') as f:
        unc_data = json.load(f)
    
    for idx,event in input_events.iterrows():
        evid = str(event.evid)
        data_in = np.array(unc_data[evid])
        print(evid,event.ndef)
        plot_uncertainty(data_in,event)
    
input_uncertainties_file = '/Volumes/SeaJade 2 Backup/NZ/EQTransformer/output/uncertainties_test/20010806_uncertainties.json'
input_events_file = '/Volumes/SeaJade 2 Backup/NZ/EQTransformer/output/catalog_test/20010806_origins.csv'
evid = '20210304p5'

show_uncertainty(input_uncertainties_file,input_events_file,evid)

show_uncertainties(input_uncertainties_file,input_events_file)

