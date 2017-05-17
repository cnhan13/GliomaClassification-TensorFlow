#!/bin/bash

echo "$(date): ${0}"
nvidia-smi
# Check who is running which process -> who is consuming so much gpu
#ps aux | grep /usr/bin/python

#cp ~/dl/BRATS2015/tumor_cropped/test_list* ~/Dropbox/dl-fyp-exp/input_list/
#cp ~/dl/BRATS2015/tumor_cropped/train_list* ~/Dropbox/dl-fyp-exp/input_list/
# DONE

#rm -rf ~/dl/BRATS2015/tumor_cropped/eval0
#rm -rf ~/dl/BRATS2015/tumor_cropped/eval1
#rm -rf ~/dl/BRATS2015/tumor_cropped/eval2
#rm -rf ~/dl/BRATS2015/tumor_cropped/eval3
#rm -rf ~/dl/BRATS2015/tumor_cropped/eval4
#rm -rf ~/dl/BRATS2015/tumor_cropped/eval5
#rm -rf ~/dl/BRATS2015/tumor_cropped/train0
#rm -rf ~/dl/BRATS2015/tumor_cropped/train1
#rm -rf ~/dl/BRATS2015/tumor_cropped/train2
#rm -rf ~/dl/BRATS2015/tumor_cropped/train3
#rm -rf ~/dl/BRATS2015/tumor_cropped/train4
#rm -rf ~/dl/BRATS2015/tumor_cropped/train5
#rm ~/dl/BRATS2015/tumor_cropped/stats*
ls -la ~/dl/BRATS2015/tumor_cropped/

#rm ~/dl/BRATS2015/tumor_cropped/test_list*
#rm ~/dl/BRATS2015/tumor_cropped/train_list*
# DONE

#touch ${0}.done

#python ~/Dropbox/cj/dl/code/brats_gen_input_list.py > ~/Dropbox/dl-fyp-exp/input_list.log
# DONE

#ls -la ~/Dropbox/cj/ctl/
