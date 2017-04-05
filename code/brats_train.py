import tensorflow as tf

import numpy as np
import skimage.io as sio
import SimpleITK as sitk

import brats
import brats_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '~/dl/BRATS2015/brats_train',
                           """Directory where to write event logs"""
                           """and checkpoints.""")
tf.app.flags.DEFINE_integer('max_steps', 100,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('set_number', 10,
                            """Number of set to run.""")


def train():
  """Train BRATS for a number of steps"""
  # with tf.Graph().as_default():
  #   global_step = tf.contrib.framework.get_or_create_global_step()
  #   mris, labels = inputs()

  #DEBUG TOOLS
  #from tensorflow.python import debug as tf_debug
  #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
  
  # Define a scalar tensor for the set number, so that we can alter it at sess.run() time
  #set_number_tensor = tf.placeholder_with_default(1, shape=[])
  #set_number_tensor = tf.placeholder_with_default(1, shape=[FLAGS.set_number])
  #records, labels = brats.inputs(set_number_tensor)
  
  init_op = tf.global_variables_initializer()
  
  # debugging queue hangs
  config = tf.ConfigProto()
  config.operation_timeout_in_ms = 5000

  with tf.Session() as sess:
    records, labels = brats.inputs(1)
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    #records, labels = tf.run(labels, feed_dict={set_number_tensor: 1})

    print "records"
    print records.eval().shape
    print "labels"
    print labels.eval()

    print "Done"
    coord.request_stop()
    coord.join(threads)
    coord.clear_stop()
    sess.close()


def main(argv=None): # pylint: disable=unused-argument
  for i in xrange(15,16):
    _, _ = brats.get_input_list(i)

  #train()

if __name__ == '__main__':
  tf.app.run()
