#!/bin/bash

EXECDIR="./Release"
EXECNAME="Heated_Plate_2D"

MAXRUN=5
MAXSIZE=1025
MAXTHREADS=9

TIMEOUT=10

# Header
echo "Algorithm,Openmp,Size,Run,Nthreads,Time,Iterations,Err"

size=128
while [ $size -lt $MAXSIZE ]; do
# Jacobi Single Threaded
   run=0
   while [ $run -lt $MAXRUN ]; do
      stat=`${EXECDIR}/${EXECNAME} -m $size -n $size | grep "Wallclock" | awk '{print $3 $7 $10 $13}'`
      let run=run+1
      echo "Jacobi,Y,$size,$run,$stat"
      sleep $TIMEOUT
   done

# Red Black Gauss-Seidel Single Threaded
   run=0
   while [ $run -lt $MAXRUN ]; do
      stat=`${EXECDIR}/${EXECNAME} -g -r -m $size -n $size | grep "Wallclock" | awk '{print $3 $7 $10 $13}'`
      let run=run+1
      echo "Red Black Gauss-Seidel,Y,$size,$run,$stat"
      sleep $TIMEOUT
   done
   let size=size+128
done
