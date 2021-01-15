#!/bin/bash
#remove some .nc files from the Fields directory 
START=5097600 # from time
END=7862400 # to time
STEP=3600*24
echo "Delete files in loop"

for (( c=$START; c<=$END; c+=STEP ))
do
	echo -n "$c "
	rm *"_"$c".nc"
done

echo
echo "done!"
