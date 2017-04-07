#!/bin/bash
# Usage:
# ./experiments/scripts/test.sh GPU NET DATASET HYPERNET
# Example:
# ./experiments/scripts/test.sh 0 VGG16 detrac 1-3-5

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3
HYPER_NET=$4
DATA_PATH='/datacenter/1/DETRAC'

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  detrac)
    TRAIN_IMDB="detrac_trainval"
    TEST_IMDB="detrac_test"
    PT_DIR="detrac"
    ITERS=70000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac


test_order="01"
for i in $test_order
do
    time ./tools/test_net.py --gpu ${GPU_ID} \
        --def models/${PT_DIR}/${NET}/test_${HYPER_NET}.prototxt \
        --net data/vgg16_eb_1-3-5_final.caffemodel \
	--imdb ${TEST_IMDB} \
        --test $i \
  	--data ${DATA_PATH} \
        ${EXTRA_ARGS}
     echo $i"+++++++++++++++++++++++++++++++++"
done
