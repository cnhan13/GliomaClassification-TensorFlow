#!/bin/bash
#set -e

CTL="/home/cnhan21/Dropbox/cj/ctl"
SCD_NAME=${0##*/}
CSD_LOG="/home/cnhan21/Dropbox/cj/controller.log"
MX_LOG_SZ_CTL=3000 # CONTROLLER
MX_LOG_SZ_SCD=3000 # SCHEDULER

# Limit log size
LOG_SZ_SCD=$(stat -c%s $CSD_LOG)
if [ "$LOG_SZ_SCD" -gt "$MX_LOG_SZ_SCD" ]; then
  rm $CSD_LOG
fi

# Control Intro
echo
echo "<<<<<-----"
echo "$(date) Enter ${0}, pid: $(cat ${0}.pid)"

if [ -e ${0}.pid_old ]; then
  if (ps -u $(whoami) -opid= | grep -P "^\s*$(cat ${0}.pid_old)$" &> /dev/null); then
    echo "-----Old ${SCD_NAME} is running on-----"
    ps -fu $(whoami) | grep -P "$(cat ${0}.pid_old)"

    if [ -e ${0}.kill ]; then
      echo "Attempt to kill old ${SCD_NAME}"

      if $(kill $(cat ${0}.pid_old)); then
        echo "Successfully killed old ${SCD_NAME}"
      else
        echo "Failed to kill old ${SCD_NAME}"
      fi
    fi
  elif [ -e ${0}.done ]; then
    echo "${SCD_NAME} was done. Exiting."
  elif [ -e ${0}.kill ]; then
    echo "${SCD_NAME} is on kill. Exiting."
  else
    for CTL_SCRIPT in ${CTL}/*.sh; do
      CTL_NAME=${CTL_SCRIPT##*/}
      echo
      echo "$(date) Start iteration: ${CTL_NAME}"

      if (ps -u $(whoami) -opid= | grep -P "^\s*$(cat ${CTL_SCRIPT}.pid)$" &> /dev/null); then
        echo "-----${CTL_NAME} is running on-----"
        ps -fu $(whoami) | grep -P "$(cat ${CTL_SCRIPT}.pid)"

        if [ -e ${CTL_SCRIPT}.kill ]; then
          echo "Attempt to kill ${CTL_NAME}"

          if $(kill $(cat ${CTL_SCRIPT}.pid)); then
            echo "Successfully killed ${CTL_NAME}"
            if [ -e ${CTL_SCRIPT}.pid ]; then
              rm -v ${CTL_SCRIPT}.pid
            fi
          else
            echo "Failed to kill ${CTL_NAME}"
          fi
        fi
      elif [ -e ${CTL_SCRIPT}.done ]; then
        echo "${CTL_NAME} was done. Skip to the next script."
        continue
      elif [ -e ${CTL_SCRIPT}.kill ]; then
        echo "${CTL_NAME} is on kill. Skip to the next script."
        continue
      else
        LOG_SZ_CTL=$(stat -c%s ${CTL_SCRIPT}.log)
        if [ "$LOG_SZ_CTL" -gt "$MX_LOG_SZ_CTL" ]; then
          bash ${CTL_SCRIPT} > ${CTL_SCRIPT}.log 2>&1 & echo $! > ${CTL_SCRIPT}.pid
        else
          bash ${CTL_SCRIPT} >> ${CTL_SCRIPT}.log 2>&1 & echo $! > ${CTL_SCRIPT}.pid
        fi
      fi
      
      echo "$(date) End iteration: ${CTL_SCRIPT}"
    done

    if [ -e ${0}.pid ]; then
      mv -v ${0}.pid ${0}.pid_old
    fi
  fi
else
  mv -v ${0}.pid ${0}.pid_old
fi

# Control End
if [ -f ${0}.pid ]; then
  echo "$(date) Exit ${0}, pid: $(cat ${0}.pid)"
elif [ -f ${0}.pid_old ]; then
  echo "$(date) Exit ${0}, pid: $(cat ${0}.pid_old)"
fi
echo "----->>>>>"
# echo "# arguments called with ----> ${@}    "
# echo "# \$1 ----------------------> $1      "
# echo "# \$2 ----------------------> $2      "
# echo "# path to me ---------------> ${0}    "
# echo "# parent path --------------> ${0%/*} "
