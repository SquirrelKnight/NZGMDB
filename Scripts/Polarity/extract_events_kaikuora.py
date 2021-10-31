import obspy as open

cat = op.read_events('/Users/jesse/Downloads/STREWN_merged_polarities.xml')

data = []
for event in cat:
    evid = str(event.resource_id).split('/')[-1]
    for pick in event.picks:
        phase = pick.phase_hint
        polarity = pick.polarity
        pick_time = pick.time
        pick_sta = pick.waveform_id.station_code
        pick_loc = pick.waveform_id.location_code
        pick_cha = pick.waveform_id.channel_code
        data.append([evid,pick_sta,pick_loc,pick_cha,pick_time,phase,polarity])
        
df = pd.DataFrame(data,columns=['evid','sta','loc','cha','datetime','phase','polarity'])
df.to_csv('kaikoura_polarities.csv',index=False)