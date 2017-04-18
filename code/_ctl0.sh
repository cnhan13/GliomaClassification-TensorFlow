#!/bin/bash
#set -e

# Worker Intro
echo
echo "<<<<<-----"
echo "$(date) Enter ${0}, pid: $(cat ${0}.pid)"

# Worker Variables
DL="/home/cnhan21/dl/"
BRATS=$DL"BRATS2015/"
TUMOR_CROPPED=$BRATS"tumor_cropped/"

TF=$DL"bin/activate"

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

echo "-----Load Virtual Env. with TensorFlow-----"
source ${TF}

#echo "-----Check Tensorflow version-----"
#python -c 'import tensorflow as tf; print(tf.__version__)' # for Python 2
#pip list --format=columns| grep tensorflow

#echo "-----List cron jobs of $(whoami)-----"
#crontab -u $(whoami) -l

echo "-----Run-----"


# Worker End
if [ -f ${0}.pid ]; then
  echo "$(date) Exit ${0}, pid: $(cat ${0}.pid)"
  rm -r ${0}.pid
fi
echo "----->>>>>"
