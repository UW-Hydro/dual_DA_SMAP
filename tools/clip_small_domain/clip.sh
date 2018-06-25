#!/bin/bash
# This script simply clips a subdomain (rectangular) from a netCDF file. Will not deal with active/inactive cells.

# --- Process input arguments --- #
target_domain_nc=$1  # will use the lat lon dimensions to clip
input_nc=$2
output_nc=$3

# --- Extract lat lon info from the target domain --- #
lat_min=`cdo sinfo $target_domain_nc | grep "lat       : first" | head -n 1 | awk '{print $5}'`
lat_max=`cdo sinfo $target_domain_nc | grep "lat       : first" | head -n 1 | awk '{print $8}'`
lon_min=`cdo sinfo $target_domain_nc | grep "lon       : first" | head -n 1 | awk '{print $5}'`
lon_max=`cdo sinfo $target_domain_nc | grep "lon       : first" | head -n 1 | awk '{print $8}'`

# --- Use ncks to clip --- #
ncks -O -d lat,$lat_min,$lat_max -d lon,$lon_min,$lon_max $input_nc $output_nc

