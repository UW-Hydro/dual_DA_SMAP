#!/bin/bash
# This script was written to create the parameter files for each new iteration, 
# run and route VIC, and process the output needed for MOCOM-UA to specify the 
# subsequent calibration parameter set.

# AWW: the script is called by the MOCOM-UA C-program: 
# major steps below:
#  1.  modify soil file with parameters for new iteration
#  2.  run vic and route output
#  3.  calculate statistics
#  4.  store outputs

if [ "$8" == "" ]; then
  echo "Usage: $0 <mocom run> <model run> <Bi> <Ds> <Ws> <Dsmax> <d2> <PADJ>"
  exit 1
fi

  set -u

  #################################
  ##         Parameters          ##
  #################################

  MOCOMRUN=$1
  MODELRUN=$2

  BI=$3
  DS=$4
  WS=$5
  D2=$6
  DSMAX=$7
  PADJ=$8

  source basin.conf

  # (use PWD instead) BASINDIR=/home1x/cics/pcic/vic/autocal/models/$BAS   #TODO  change concurrent to runs or something
  WORKDIR=$PWD/runs/$MOCOMRUN/$MODELRUN

  echo "$(date +%F\ %H:%M) Dispatching model run ( ${MODELRUN} ): " BI=$BI DS=$DS WS=$WS DSMAX=$DSMAX D2=$D2 PADJ=$PADJ

  #################################
  ##  Preliminary run dir setup  ##
  #################################

  rm -Rf $WORKDIR >& /dev/null
  mkdir -p $WORKDIR

  ##   Config Script   ##
  echo 'source ../../../basin.conf' >> "${WORKDIR}"/slave.conf

  ##   Generate Parameter File   ##
  echo '# Optimization parameters for basin '"${BASIN}"', initiated '"${MOCOMRUN}"', run #'"${MODELRUN}"  >> "${WORKDIR}"/params.txt
  echo BI=${BI}        >> $WORKDIR/params.txt
  echo DS=${DS}        >> $WORKDIR/params.txt
  echo WS=${WS}        >> $WORKDIR/params.txt
  echo D2=${D2}        >> $WORKDIR/params.txt
  echo DSMAX=${DSMAX}  >> $WORKDIR/params.txt
  echo PADJ=${PADJ}    >> $WORKDIR/params.txt


  #################################
  ##      Dispatch PBS job       ##
  #################################

  echo -n >| $WORKDIR/tmp.sh
  echo \#PBS -q express >> $WORKDIR/tmp.sh
      ## May need to change time value in this next param
  echo \#PBS -l walltime=5:00:00 >> $WORKDIR/tmp.sh
  echo \#PBS -l mem=1000mb >> $WORKDIR/tmp.sh
  echo \#PBS -l ncpus=1 >> $WORKDIR/tmp.sh
  echo \#PBS -W Output_Path=$WORKDIR >> $WORKDIR/tmp.sh
  echo \#PBS -W Error_Path=$WORKDIR >> $WORKDIR/tmp.sh
  echo cd $WORKDIR >> $WORKDIR/tmp.sh
    ## clean this next line up later, ugh
  echo  /bin/bash "$(readlink -f "$(dirname "${BASH_SOURCE[0]}")")"/optimize_wb_SLAVE.sh >> $WORKDIR/tmp.sh

  chmod a+x $WORKDIR/tmp.sh
  qsub $WORKDIR/tmp.sh >& $WORKDIR/jobid
#  /bin/csh $WORKDIR/tmp.sh >&| $WORKDIR/jobid   # Enable this to disable use of job dispatch system

