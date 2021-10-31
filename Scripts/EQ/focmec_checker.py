import glob
import obspy as op

search_dir = '/Volumes/SeaJade 2 Backup/NZ/mseed_35/2010'
event_xmls = glob.glob(search_dir+'/**/*.xml',recursive=True)

for event_xml in event_xmls:
	cat = op.read_events(event_xml)
	for event in cat:
		print(event.focal_mechanisms)
	