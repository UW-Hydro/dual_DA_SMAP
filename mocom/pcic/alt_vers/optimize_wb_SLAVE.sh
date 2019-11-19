#!/bin/bash -u
touch SLAVE.started."$(date +%F-%H%M)"


###################
##  Environment  ##
###################


MOCOMRUN="$(basename "$(dirname "${PWD}")")"
MODELRUN="$(basename "${PWD}")"

source slave.conf
source params.txt

STATS="${PWD}"/flow/"${BASIN}".stats
GLOBALFILE="${PWD}"/"$(basename "${GLOBALFILESRC}")"
SOILFILEDEST="${PWD}"/"$(basename "${SOILFILESRC}")"
#R2FILE=../stats_"${MODELRUN}".txt


#################
##    Setup    ##
#################

# Cleanup
rm -Rf flux flow rout_input rout_inp.*

# Setup
mkdir -p flux flow

# TODO consider a central location for upstream simulated flows, and stick it in a variable
cp -a ../../../flow ./
# TODO set the following up in autocal.conf
#  cp $SOURCEDIR/obs/$BASIN.obs.day $BASINDIR/$BASIN.obs.day
#  cp $SOURCEDIR/obs/$BASIN.obs.day.kaf $BASINDIR/$BASIN.obs.day.kaf


cp -r "${TEMPLATEDIR}"/rout_input .
mv ./rout_input/station_RENAME ./rout_input/station_"${BASIN}"
cp "${TEMPLATEDIR}"/rout_inp.RENAME ./rout_inp."${BASIN}"
sed -r -i 's:^(ROUTED_FILE[ \t].*/)[^/]*$:\1'"${ROUTED_FILE_NAME}"':g' ./rout_inp."${BASIN}"
## $BASINDIR/rout_input/*  --  Flag basin-specific lines in station_$BASIN.frs e.g. "0  0  BAKER  95  72  -999  1" -> "1  0  BAKER  95  72  -999  1"
sed -r -i 's/^0( .* '"${BASIN}"' .*)/1\1/g' ./rout_input/station_"${BASIN}"
sed -r -i 's/RENAMEBASINSTRING/'"${BASIN}"'/g' ./rout_inp.$BASIN
  

cp "${GLOBALFILESRC}" "${GLOBALFILE}"
sed -i -r 's:^(FORCING1[ \t]+)[^ \t]+:\1'"${FORCINGS}"':g'    "${GLOBALFILE}"
sed -i -r 's:RENAMEBASINSTRING:'"${BASIN}"':g'                "${GLOBALFILE}"
sed -i -r 's:^(PADJ[ \t]+)[^ \t]+:\1'"${PADJ}"':g'            "${GLOBALFILE}"
sed -i -r 's:^(SOIL[ \t]+)[^ \t]+:\1'"${SOILFILEDEST}"':g'    "${GLOBALFILE}"
sed -i -r 's:^(RESULT_DIR[ \t]+)[^ \t]+:\1'"${PWD}"/flux':g'  "${GLOBALFILE}"

awk '{if($1==1){$5 = '"${BI}"'; $6 = '"${DS}"'; $7 = '"${DSMAX}"'; $8 = '"${WS}"'; $24 = '"${D2}"';} print $0;}' "${SOILFILESRC}" >| "${SOILFILEDEST}"
# awk '{if($1==1){$5 = '"${BI}"'; $6 = '"${DS}"'; $7 = '"${DSMAX}"'; $8 = '"${WS}"'; $24 = '"${D2}"'; $25 = '"${D3}"';} print $0;}' "${SOILFILESRC}" >| "${SOILFILEDEST}"


#########################################
##    Run VIC, routing, stats, plot    ##
#########################################

echo "$(date +%F\ %H:%M)  Running VIC Model"
#/home1x/cics/pcic/vic/autocal/sources/vicnl_407_ALBEDO3_PADJv1/vicNl_albedo3 -g $GLOBALFILE >& vic.out
"${VICBIN}" -g $GLOBALFILE >& vic.out


echo "$(date +%F\ %H:%M)  Running Routing Model"
/home1x/cics/pcic/vic/autocal/sources/rout_vc2/rout ./rout_inp."${BASIN}" >& rout.out  #TODO: fix path here; input copied above, what are we using?
[[ -e flow/"${BASIN}".day ]] || { echo "Routing failed.  See rout.out for more information." ; exit 1 ; }

echo "$(date +%F\ %H:%M)  Calculating output stats"
"$(readlink -f "$(dirname "${BASH_SOURCE[0]}")")"/"${STATSSCRIPTNAME}" flow/"${BASIN}".day "${STATSOBSFLOW}" "${STATS}" "${STAT_ST_PER}" "${STAT_END_PER}" -99.0 tempr2file.txt

echo "$(date +%F\ %H:%M)  Plotting flows"
RUN="${BASIN}"/"${MOCOMRUN}"/"${MODELRUN}"
/bin/csh "$(readlink -f "$(dirname "${BASH_SOURCE[0]}")")"/"${PLOTSCRIPTNAME}" "${BASIN}" "${PLOTOBSFLOW}" "${RUN}"

mv tempr2file.txt stats.txt

mv flow/"${BASIN}".day .

rm -Rf "${SOILFILEDEST}" flux/ flow/ >& /dev/null
rm -Rf rout_inp.* rout_input/ >& /dev/null  # TODO make this unnecessary
