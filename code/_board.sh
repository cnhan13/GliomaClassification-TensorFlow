TRAIN_DIR="~/dl/BRATS2015/tumor_cropped/"
tensorboard --logdir=$TRAIN_DIR --port 6007 & sleep 1000 & google-chrome 0.0.0.0:6007
