#!/bin/bash
set -e


# Worker Intro
echo
echo "<<<<<-----"
echo "$(date) Enter ${0}, pid: $(cat ${0}.pid)"

# touch ${0}.done

# Worker Variables

# Variables for training. Already configured
ME_HOME="/home/cnhan21/"
DL=$ME_HOME"dl/"
CODE_DIR=$DL"/code/"
TF=$DL"bin/activate"

# Change according to the running model:
# p5c1_train - p5c1_eval
# p5c1_train_vgge - p5c1_eval_vgge
# p5c1_train_vggd - p5c1_eval_vggd
# p5c1_train_vggb - p5c1_eval_vggb
# p5c1_train_vggf - p5c1_eval_vggf
# etc.
TRAIN_NAME="p5c1_train_vggf"
EVAL_NAME="p5c1_eval_vggf"



# Variables for sync-ing using Dropbox for remote run. MUST RECONFIG
BRATS=$DL"BRATS2015/"
TUMOR_CROPPED=$BRATS"tumor_cropped/"
TRAIN_RESULTS=$TUMOR_CROPPED"train*"
EVAL_RESULTS=$TUMOR_CROPPED"eval*"
DROPBOX=$ME_HOME"Dropbox/"
DL_FYP_EXP=$DROPBOX"dl-fyp-exp/"
TRY=$DL_FYP_EXP"convnet_lr001/"
MODEL_CODE_LOG=$DL"code/p5c1_*"


# Worker Body
echo "WORKER: ${0##*/}"
echo "-----All bashes by $(whoami)-----"
ps -fu $(whoami) | grep bash

#echo "-----Bash of current control-----"
#ps -fu $(whoami) | grep -P "$(cat ${0}.pid)\s*1"
#[not-used]ps -opid= | grep -P "^\s*$(cat ${0}.pid)$"

#echo "-----Processes relating to Bash of current control-----"
#ps -fu $(whoami) | grep -P "$(cat ${0}.pid)"

#echo "-----Listing files in ${TUMOR_CROPPED}-----"

echo "-----Load Virtual Env. TensorFlow-----"
source ${TF}

#echo "-----Check Tensorflow version-----"
#python -c 'import tensorflow as tf; print(tf.__version__)' # for Python 2
#pip list --format=columns| grep tensorflow

#echo "-----List cron jobs of $(whoami)-----"
#crontab -u $(whoami) -l

echo "-----Run-----"
NUM_SET=6
IS_TUMOR_CROPPED=1
MAX_NUM_EVALS=100
TRAIN_STEPS=200 # 20,000 steps total
IS_RESET=0 # just use it 0. No handling of '1'
for ((j=1; j<=$MAX_NUM_EVALS; j++))
do
  for ((i=0; i<$NUM_SET; i++))
  do
    date
    START=$(( ($j - 1) * $TRAIN_STEPS + 1))
    END=$(($j * $TRAIN_STEPS))
    echo "Training set $i, tumor_cropped $IS_TUMOR_CROPPED, training steps interval [$START; $END]"
    train_out="${CODE_DIR}${TRAIN_NAME}_set${i}_tumor${IS_TUMOR_CROPPED}_steps${START}-to-${END}.log"
    python ${CODE_DIR}${TRAIN_NAME}.py $i $IS_TUMOR_CROPPED $IS_RESET $j | tee $train_out
    date
    echo "Evaluating set $i, tumor_cropped $IS_TUMOR_CROPPED"
    eval_out="${CODE_DIR}${EVAL_NAME}_set${i}_tumor${IS_TUMOR_CROPPED}_steps${END}.log"
    python ${CODE_DIR}${EVAL_NAME}.py $i $IS_TUMOR_CROPPED $j | tee $eval_out
    # Uncomment below code and adjust appropriate dirs to get always-synced results
    #rsync -vru $TRAIN_RESULTS $TRY
    #rsync -vru $EVAL_RESULTS $TRY
    #rsync -vu ${TUMOR_CROPPED}stats* ${TRY}/
    #rsync -vu ${CODE_DIR}p5c1_* ${TRY}run_log/
  done
done

echo "-----Close Virtual Env. TensorFlow-----"
deactivate

# Worker End
if [ -f ${0}.pid ]; then
  echo "$(date) Exit ${0}, pid: $(cat ${0}.pid)"
fi

echo "----->>>>>"
