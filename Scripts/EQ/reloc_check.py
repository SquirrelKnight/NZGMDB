import h5py
import pandas as pd

filename = '/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/martins_events.h5'

events = pd.read_hdf(filename,key='events')
events['otime'] = pd.to_datetime(events['otime'],format='%Y-%m-%dT%H:%M:%S.%f')
events.rename(columns={'EVID':'publicid'},inplace=True)

martins = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/martins_new.csv',low_memory=False)
not_in = martins[martins.evid.astype(str).isin(unique_ids) == False]