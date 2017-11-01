#!/bin/bash -e
#
# Usage: ./run.sh <part#>
# Example: ./run.sh 1
#

if [ $1 == 1 ] ; then
    python train.py --data_file data/hw4_data.bin --optimizer Adam -lr 1e-2 --batch_size 48 --model_file model.py
# else
#     echo "You should eat a bit more fruit."
fi

