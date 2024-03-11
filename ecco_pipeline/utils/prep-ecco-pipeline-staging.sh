#!/bin/bash

# bash script that stages ecco-obs-pipeline files
# so they they can be rsync'd to pfe

# Ian Fenty, 2024-03-11

# processed observations are located at /ecco_nfs_1/shared/ECCO-pipeline-ep/observations 

# each dataset lives in a subdirectory of the above
# loop through all dataset sub-directories
for i in /ecco_nfs_1/shared/ECCO-pipeline-ep/observations/* 
do 
   echo "found dataset: $i"

   # each branch of the dataset directory tree contains  harvested, transformed, and aggregated files
   # we want the aggregated files
   # loop through all subdirectories with the word "aggregated"
   for ff in `find $i -maxdepth 3 -type d |grep aggregated`
   do 

     # cut out the first part of the path, everything between /ecco_nfs_1/ to .../observations/
     export fg=`echo $ff | cut -d'/' -f6-`

     # make a local directory starting with the dataset name and ending with "aggregated"
     echo "making subdirectory $fg"
     mkdir -p $fg

     # softlink the contents of /aggregated/* to the new local directory
     echo "... softlinking contents to $fg"
     ln -s $ff/* $fg

   done
done
