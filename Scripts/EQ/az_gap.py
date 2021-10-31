# az_coverage script
import pandas as pd
import numpy as np

props = pd.read_csv('/Volumes/SeaJade2/NZ/NZ_EQ_Catalog/converted_output/propagation_path_table.csv',low_memory=False)
events = props.evid.unique()
for event in events:
	props_sub = props[props.evid == str(event)]
	props_sub = props_sub.sort_values('az').az.values
	diffs = np.abs(np.diff(np.append(props_sub,props_sub[0])))
	diffs[diffs > 180] = 360 - diffs[diffs > 180]
	az_gap = diffs.max()
	print(event,az_gap)