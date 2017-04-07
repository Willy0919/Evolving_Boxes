#!/bin/bash
# Usage:
# # ./experiments/scripts/test.sh GPU NET DATASET HYPERNET
# Example:
# ./experiments/scripts/train.sh 0 VGG16 detrac 1-3-5

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3
HYPER_NET=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}
DATA_PATH='/datacenter/1/DETRAC'

case $DATASET in
  detrac)
    TRAIN_IMDB="detrac_train"
    TEST_IMDB="detrac_test"
    PT_DIR="detrac"
    ITERS=70000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/evb_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${NET}/solver_${HYPER_NET}.prototxt \
  --weights data/imagenet_models/${NET}.v2.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --data ${DATA_PATH} \
  --net ${HYPER_NET}
  ${EXTRA_ARGS}

