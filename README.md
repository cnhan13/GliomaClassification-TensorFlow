# Requirement
Need TensorFlow 1.0.1, either in virtualenv (must be in dl/)
to run the training code or coming with the system.
To check TensorFlow version, use:
python -c 'import tensorflow as tf; print(tf.__version__)'  # for Python 2

Also install the following packages in the environment
if need to run the input generator again using:
sudo pip install scikit-image
sudo pip install SimpleITK

Run the controller: _ctl2.sh which will alternatively train and evaluate after
certain number of steps. In this controller, it assumes the correct TensorFlow
version is in the virtualenv, which is activated by "source ${TF}",
you can find it in _ctl2.sh. Run the controller using this command:
bash ~/dl/code/_ctl2.sh

# Visualization
You can see the TensorBoard of the currently running code using this command:
tensorboard --logdir=~/dl/BRATS2015/tumor_cropped/

# Note:
The code is fuzzy. If running into any errors, please read carefully first before changing the code.
