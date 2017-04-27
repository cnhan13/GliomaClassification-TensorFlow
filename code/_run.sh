#!/bin/bash
set -e
NUM_SET=5
IS_TUMOR_CROPPED=1
MODEL_ID="a"
MAX_NUM_EVALS=10
TRAIN_STEPS=200
IS_RESET=0
DROPBOX_DIR="/home/cnhan21/Dropbox/dl-fyp-result/"
TRAIN_RESULTS="/home/cnhan21/dl/BRATS2015/tumor_cropped/train*"
EVAL_RESULTS="/home/cnhan21/dl/BRATS2015/tumor_cropped/eval*"
MODEL_CODE_LOG="/home/cnhan21/dl/code/p5c1_*"
for ((i=1; i<=$NUM_SET; i++))
do
  for ((j=1; j<=$MAX_NUM_EVALS; j++))
  do
    date
    START=$(( ($j - 1) * $TRAIN_STEPS + 1))
    END=$(($j * $TRAIN_STEPS))
    echo "Training set $i, tumor_cropped $IS_TUMOR_CROPPED, set_char $MODEL_ID, training steps interval [$START; $END]"
    train_out="p5c1_train_set${i}_tumor${IS_TUMOR_CROPPED}_model${MODEL_ID}_${START}-to-${END}"
    python p5c1_train.py $i $IS_TUMOR_CROPPED $IS_RESET $MODEL_ID $j | tee $train_out
    date
    echo "Evaluating set $i, tumor_cropped $IS_TUMOR_CROPPED"
    eval_out="p5c1_eval_set${i}_tumor${IS_TUMOR_CROPPED}_model${MODEL_ID}_steps${END}"
    python p5c1_eval.py $i $IS_TUMOR_CROPPED $MODEL_ID | tee $eval_out
    rsync -vru $TRAIN_RESULTS $DROPBOX_DIR
    rsync -vru $EVAL_RESULTS $DROPBOX_DIR
    rsync -vu $MODEL_CODE_LOG ${DROPBOX_DIR}code-log/
  done
done
