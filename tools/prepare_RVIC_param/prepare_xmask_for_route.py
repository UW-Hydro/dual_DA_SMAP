#!/usr/local/anaconda/bin/python

''' This scripts prepares a xmask file for Lohmann routine, using consistant method as in the inverse routing method (i.e., using haversine formula) '''

import numpy as np
import argparse

from tonic.io import read_configobj
from da_utils import generate_xmask_for_route

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str,  help="config file for this script")
args = parser.parse_args()
cfg = read_configobj(args.cfg)

#========================================================#
# Calculate xmask (i.e., flow distance)
#========================================================#
flow_distance = generate_xmask_for_route(cfg['INPUT']['fdir_path'])

#========================================================#
# Write to file
#========================================================#
f = open(cfg['OUTPUT']['output_xmask_path'], 'w')
#--- Write header lines (copy from flow direction file) ---#
f_fdir = open(cfg['INPUT']['fdir_path'], 'r')
for i in range(6):
    line = f_fdir.readline().rstrip("\n")  # read from flow direction file
    f.write(line + "\n")
f_fdir.close()
#--- Write xmask values ---#
for i in range(len(flow_distance)):
    for j in range(len(flow_distance[0])):
        if flow_distance[i,j]==-1:
            f.write('{:d} '.format(int(flow_distance[i,j])))
        else:
            f.write('{:.1f} '.format(flow_distance[i,j]))
    f.write("\n")

f.close()






