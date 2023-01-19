import subprocess
import os
# IFS=$'\n' # make newlines the only separator
cmd = 'python3 /Users/jesse/bin/gm_classifier/gm_classifier/scripts/extract_features.py --ko_matrices_dir /Volumes/SeaJade\\ 2\\ backup/NZ/konno_matrices'
in_dir = '/Volumes/SeaJade\\ 2\\ backup/NZ/gmc_record'
out_dir = '/Volumes/SeaJade\\ 2\\ backup/NZ'

in_folders=['mseed_4.5-5', 'mseed_4-4.5']


for i,folder in enumerate(in_folders):
    if i == 0:
        years=[]
    else:
        years=[2021]
    if len(years) > 0:
        for year in years:
            year = str(year)
            print(cmd+' '+in_dir+'/'+folder+'/'+year+' '+out_dir+'/'+folder+'/'+year+' mseed')
            os.system(cmd+' '+in_dir+'/'+folder+'/'+year+' '+out_dir+'/'+folder+'/'+year+' mseed')
