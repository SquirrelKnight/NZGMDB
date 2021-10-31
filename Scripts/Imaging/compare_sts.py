import obspy as op
from obspy.clients.fdsn import Client as FDSN_Client

channel_codes = 'HN?,BN?,HH?,BH?,EH?,SH?'
client_NZ = FDSN_Client("GEONET")
client_IU = FDSN_Client('IRIS')
inventory_NZ = client_NZ.get_stations(station='FOZ',channel=channel_codes,level='response')
inventory_IU = client_IU.get_stations(network='IU',station='SNZO',channel=channel_codes,level='response')
inventory = inventory_NZ+inventory_IU

st = op.read('/Volumes/SeaJade 2 Backup/NZ/mseed_6_revised/2014/10_Oct/2014-10-13_051341/mseed/data/20141013_051341_FOZ_10_HH.mseed')
st_2 = op.read('/Volumes/SeaJade 2 Backup/NZ/mseed_6_revised/2014/10_Oct/2014-10-13_051341/mseed/data/20141013_051341_FOZ_20_BN.mseed')

pre_filt = [0.001, 0.005, 45, 50]
st_acc = st.copy().remove_response(pre_filt=pre_filt,inventory=inventory,output='ACC',water_level = None)
st_2_acc = st_2.copy().remove_response(inventory=inventory,output='ACC',water_level = None)

st_combined = st_acc+st_2_acc
st_combined.plot()

st_acc = st.copy().remove_sensitivity(inventory=inventory).differentiate()
# st_acc = st.copy().remove_response(inventory=inventory, water_level=0, output='ACC')
st_2_acc = st_2.copy().remove_sensitivity(inventory=inventory).detrend('demean')

st_combined = st_acc+st_2_acc
st_combined.plot()

/Volumes/SeaJade 2 Backup/NZ/mseed_45_preferred/2019/10_Oct/2019-10-28_144701/mseed/data/20191028_144701_WKZ_20_BN.mseed