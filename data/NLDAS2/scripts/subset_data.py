import sys
import subprocess
import pandas as pd
import os


# ======================================================= #
# Process command line argument
# ======================================================= #
start_time = sys.argv[1]  # YYYYMMDD-HH
end_time = sys.argv[2]  # YYYYMMDD-HH
start_date = start_time[:8]
end_date = end_time[:8]
start_hour = int(start_time[-2:])
end_hour = int(end_time[-2:])


# ======================================================= #
# Other parameters
# ======================================================= #
# Paths
raw_data_dir = '../raw_data'
output_basedir = '../ArkRed/hourly_nc'  # output files will be put under <output_basedir>/YYYY/

# Subset domain info  # Arkansas Red domain
lat_min = 31.18
lat_max = 39.32
lon_min = -106.57
lon_max = -91.0

# ======================================================= #
# Process and subset
# ======================================================= #
# create subdir for each year
start_year = int(start_date[:4])
end_year = int(end_date[:4])
for year in range(start_year, end_year+1):
    subdir = os.path.join(output_basedir, '{}'.format(year))
    if not os.path.exists(subdir):
        os.makedirs(subdir)

# Loop over each hourly data
dates = pd.date_range(start_date, end_date, freq='1D')
for t, date in enumerate(dates):
    for hour in range(0, 24):
        if t == 0 and hour < start_hour:
            continue
        if t == (len(dates) - 1) and hour > end_hour:
            continue
        print("Processing", date, hour)
        output_dir = os.path.join(output_basedir, '{}'.format(date.year))
        # 1) Convert grb to netCDF
        grb_path = os.path.join(
            raw_data_dir,
            "NLDAS_FORA0125_H.A{}.{:04d}.002.grb".format(
                date.strftime('%Y%m%d'), hour*100))
        output_nc = os.path.join(
            output_dir, 'converted_nc.{}.{:02d}.nc'.format(date.strftime('%Y%m%d'), hour))
        subprocess.call("cdo -f nc copy {} {}".format(grb_path, output_nc), shell=True)
        # 2) Subset domain
        output_subset_nc = os.path.join(
            output_dir, 'subset.{}.{:02d}.nc'.format(date.strftime('%Y%m%d'), hour))
        subprocess.call(
            "ncks -O -d lat,{},{} -d lon,{},{} {} {}".format(
                lat_min, lat_max, lon_min, lon_max, output_nc, output_subset_nc),
            shell=True)
        os.remove(output_nc)
        # 3) Only keep VIC-needed variables and delete the rest to reduce size
        output_final_nc = os.path.join(
            output_dir, 'force.{}.{:02d}.nc'.format(date.strftime('%Y%m%d'), hour))
        subprocess.call(
            "ncks -O -x -v var153,var157,var228 {} {}".format(
                output_subset_nc, output_final_nc),
            shell=True)
        os.remove(output_subset_nc)





