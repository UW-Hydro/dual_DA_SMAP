#!/bin/bash

set -e

# ======================================================== #
# Parse in parameters from MOCOM
# ======================================================== #
# MOCOM run csh
run_csh=$1
bas=$2

# ======================================================== #
# Extract info from the run_sch
# ======================================================== #
# MOCOM output log file
run_ident=`grep "set run_ident" $run_csh | awk '{print $4}'`  # ID

log=`grep "set optim_log" $run_csh | awk '{print $4}'`
log="${log/\$bas/$bas}"
log=${log/\$run_ident/$run_ident}

num_sets=`grep "set num_sets" $run_csh | awk '{print $4}'`  # Total number of param sets in the populatin
num_tests=`grep "set num_tests" $run_csh | awk '{print $4}'`  # Number of objective function
num_param=`grep "set num_param" $run_csh | awk '{print $4}'`  # Number of parameters to calibrate

stor_dir=`grep "set stor_dir" $run_csh | awk '{print $4}'`  # MOCOM output basedir
stor_dir="${stor_dir/\$bas/$bas}"

optimize_cfg=`grep "set optimize_cfg" $run_csh | awk '{print $4}'`  # MOCOM output basedir
optimize_cfg="${optimize_cfg/\$bas/$bas}"

# ======================================================== #
# Set up output dir
# ======================================================== #
out_dir=$stor_dir/$run_ident/plot_final
mkdir -p $out_dir

# ======================================================== #
# Load log file and extract info of the final calibrated set
# ======================================================== #
# Find line number of the start of the final set result section in the log file 
line_num=`grep -rne "Current generation for generation" $log | tail -n 1 | awk -F':' {'print $1}'`
# Get the lines of the result section
tail -n +$(($line_num+4)) $log | head -n $num_sets > $out_dir/result.tmp

# ======================================================== #
# Plot for all sets
# ======================================================== #
source activate da2

while read line; do
    soln_num="${line##* }"  # parameter solution set number
    echo "Plotting $soln_num"
    # Plot
    echo $optmize_cfg $stor_dir/$run_ident $(seq -f "%05g" $soln_num $soln_num) $out_dir
    python plot.py $optimize_cfg $stor_dir/$run_ident $(seq -f "%05g" $soln_num $soln_num) $out_dir 
done < $out_dir/result.tmp

source deactivate


