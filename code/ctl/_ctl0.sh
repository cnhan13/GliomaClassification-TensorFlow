#!/bin/bash
set -e

# Worker Intro
echo
echo "<<<<<-----"
echo "$(date) Enter ${0}, pid: $(cat ${0}.pid)"

# Worker Body
echo "This is worker 0"



# Worker End
if [ -f ${0}.pid ]; then
  echo "$(date) Exit ${0}, pid: $(cat ${0}.pid)"
  rm -r ${0}.pid
fi
echo "----->>>>>"
