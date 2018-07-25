#!/bin/bash
# This script runs MOCOM-UA_newer and contains all the necessary input
# arguements.  Modify this script to change input arguements.

# Created by:		Marketa McGuire
# Last Modified:	June 13, 2003
#                       August, 2006, AWW:  cleaned up and documented
#                                           implementing monthly only version
#                       2009/01/28  PN:  Updating comments and cleaning up a bit.  More to come, though.
# 
#############################################################################

# TODO:  use commandline arguments as arguments to set ...

# TODO this cannot be implemented until MASTER/SLAVE scripts are changed to find the right config file
#if [ "$1" != "" ] ; then
#  BASINCONFIG="${1}"
#else
  BASINCONFIG="${PWD}"/basin.conf
#fi

set -u

[[ -e "${BASINCONFIG}" ]] || { echo 'Configuration file ${BASINCONFIG} not found.' ; exit 1 ; }


source "${BASINCONFIG}" || { echo 'Failed to execute basin configuration file.  Exiting.' ; exit 1 ; } # Includes autocal configuration

# Fail early if things are missing
[[ -e "${MOCOM}" ]] || { echo 'File not found: '"${MOCOM}"'.  Exiting.' ; exit 1 ; }
[[ -e "${VICBIN}" ]] || { echo 'File not found: '"${VICBIN}"'.  Exiting.' ; exit 1 ; }
[[ -d "${TEMPLATEDIR}" ]] || { echo 'File not found: '"${TEMPLATEDIR}"'/.  Exiting.' ; exit 1 ; }
[[ -e "${GLOBALFILESRC}" ]] || { echo 'File not found: '"${GLOBALFILESRC}"'.  Exiting.' ; exit 1 ; }
[[ -e "$(dirname "${BASH_SOURCE[0]}")"/"${STATSSCRIPTNAME}" ]] || { echo 'File not found: '"${}"'.  Exiting.' ; exit 1 ; }                    #TODO 2 lines here.
[[ -e "$(dirname "${BASH_SOURCE[0]}")"/"${PLOTSCRIPTNAME}" ]] || { echo 'File not found: '"${}"'.  Exiting.' ; exit 1 ; }
[[ -e "${SOILFILESRC}" ]] || { echo 'File not found: '"${}"'.  Exiting.' ; exit 1 ; }
[[ -d "$(dirname "${FORCINGS}")" ]] || { echo 'Directory not found: '"$(dirname "${FORCINGS}")"'.  Exiting.' ; exit 1 ; }
[[ -e "${STATSOBSFLOW}" ]] || { echo 'File not found: '"${STATSOBSFLOW}"'.  Exiting.' ; exit 1 ; }
[[ -e "${PLOTOBSFLOW}" ]] || { echo 'File not found: '"${PLOTOBSFLOW}"'.  Exiting.' ; exit 1 ; }


RUNID=$(date +%F-%H%M)   # This identifies a specific run of MOCOM

LOG=${PWD}/optim_log.${BASIN}.${RUNID}  # Log of parameter set generations, evolution, and final set

$MOCOM $NSTART $NSETS $NPARAM $NTESTS $RUN_SCRIPT $RUNID $LOG $PARAM_RANGE_FILE
