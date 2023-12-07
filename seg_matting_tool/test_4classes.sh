#!/bin/sh
PARTITION=gpu
PYTHON=python

dataset=$1
exp_name=$2
exp_dir=exp/${dataset}/${exp_name}
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${result_dir}
cp seg_matting_tool/test_4classes.sh seg_matting_tool/test_4classes.py ${config} ${exp_dir}

export PYTHONPATH=./
#sbatch -p $PARTITION --gres=gpu:1 -c2 --job-name=test_4classes \
$PYTHON -u seg_matting_tool/test_4classes.py \
  --config=${config} \
  2>&1 | tee ${result_dir}/test_4classes-$now.log
