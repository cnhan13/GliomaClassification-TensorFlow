#!/bin/bash
#set -e

CTL="/home/cnhan21/Dropbox/cj/ctl"

# Control Intro
echo
echo "<<<<<-----"
echo "$(date) Enter ${0}, pid: $(cat ${0}.pid)"

for CTL_SCRIPT in ${CTL}/*.sh
do
  echo
  echo "$(date) Start iteration: ${CTL_SCRIPT}"

  if [ -f ${CTL_SCRIPT}.done ]; then
    echo "${CTL_SCRIPT}.done exists. Skip to the next script."
    continue
  fi

  if [ -f ${CTL_SCRIPT}.kill ]; then
    echo "Attempt to kill ${CTL_SCRIPT}"
    if $(kill $(cat ${CTL_SCRIPT}.pid)); then
      echo "Successfully killed ${CTL_SCRIPT}"
      if [ -f ${CTL_SCRIPT}.pid ]; then
        rm -v ${CTL_SCRIPT}.pid
      fi
    else
      echo "Failed to kill ${CTL_SCRIPT}"
    fi
    continue
  fi

  if [ -f ${CTL_SCRIPT}.pid ]; then
    echo "${CTL_SCRIPT} is being run"
    continue
  fi

  bash ${CTL_SCRIPT} >> ${CTL_SCRIPT}.log 2>&1 & echo $! > ${CTL_SCRIPT}.pid

  echo
  echo "$(date) End iteration: ${CTL_SCRIPT}"
done

# Control End
if [ -f ${0}.pid ]; then
  echo "$(date) Exit ${0}, pid: $(cat ${0}.pid)"
  rm -r ${0}.pid
fi
echo "----->>>>>"
# echo "# arguments called with ----> ${@}    "
# echo "# \$1 ----------------------> $1      "
# echo "# \$2 ----------------------> $2      "
# echo "# path to me ---------------> ${0}    "
# echo "# parent path --------------> ${0%/*} "
# echo "# my name ------------------> ${0##*/}"

