
import sys
import subprocess
import pandas as pd


# ======================================================= #
# Process command line argument
# ======================================================= #
start_date = sys.argv[1]  # YYYYMMDD
end_date = sys.argv[2]  # YYYYMMDD

# ======================================================= #
# Other parameters
# ======================================================= #
output_dir = '../raw_data'

# ======================================================= #
# Download hourly NLDAS-2 data
# ======================================================= #
dates = pd.date_range(start_date, end_date, freq='1D')
for date in dates:
    day_of_year = date.dayofyear
    for hour in range(0, 24):
        print('Downloading', date, hour)
        # Download the grb file
        grb_filename = "NLDAS_FORA0125_H.A{}.{:04d}.002.grb".format(date.strftime('%Y%m%d'), hour*100)
        subprocess.call(("wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies "
                         "--keep-session-cookies https://hydro1.gesdisc.eosdis.nasa.gov/data/"
                         "NLDAS/NLDAS_FORA0125_H.002/{}/{:03d}/"
                         "{}").format(
                            date.year, day_of_year, grb_filename),
                        shell=True)
        # Download the grb.xml file
        subprocess.call(("wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies "
                         "--keep-session-cookies https://hydro1.gesdisc.eosdis.nasa.gov/data/"
                         "NLDAS/NLDAS_FORA0125_H.002/{}/{:03d}/"
                         "{}.xml").format(
                            date.year, day_of_year, grb_filename),
                        shell=True)
        # Move both files to output_dir
        subprocess.call("mv {} {}/{}".format(grb_filename, output_dir, grb_filename),
            shell=True)
        subprocess.call("mv {}.xml {}/{}.xml".format(grb_filename, output_dir, grb_filename),
            shell=True)


