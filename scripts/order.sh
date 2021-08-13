#!/bin/bash

# My first script

echo "Hello World!"

for (( num=24*3600; num<=24*3600*1000; num=num+24*3600 ));
    do
	    day=$(printf "%08d\n" $num);
	    mv Temperature_$num.nc Temperature_$day.nc
	    echo $day
    done


cdo cat Temper*.nc T_merge.nc
