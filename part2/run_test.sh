#!/bin/bash

algo=$1

# if [ -n $OMP_NUM_THREADS ]; then 
#     OMP_NUM_THREADS=16
# fi

# Set the Makefile path
MAKEFILE="Makefile"

make -s -f $MAKEFILE clean 
make -s -f $MAKEFILE 

# Run the program and measure execution time
echo "Running ${algo}"
start_time=$(date +%s.%N)
./heat-omp ./data_files/test_${algo}.dat > /dev/null
end_time=$(date +%s.%N)
elapsed_time=$(echo "$end_time - $start_time" | bc)

# Compare the output
if diff -q heat.ppm ../test/heat.ppm_${algo}_should > /dev/null; then
    echo -e "\e[32mOutput of ${algo} is correct\e[0m"  # Green color
else
    echo -e "\e[31mOutput of ${algo} differs\e[0m"  # Red color
fi

# Display elapsed time
echo "Elapsed time: $elapsed_time seconds"
