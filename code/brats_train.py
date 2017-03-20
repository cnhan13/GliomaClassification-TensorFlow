import tensorflow as tf

import numpy as np
import skimage.io as sio
import SimpleITK as sitk

import brats

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '~/dl/BRATS2015/brats_train',
                           """Directory where to write event logs"""
                           """and checkpoints.""")
tf.app.flags.DEFINE_integer('max_steps', 100,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def train():
  """Train BRATS for a number of steps"""
  # with tf.Graph().as_default():
  #   global_step = tf.contrib.framework.get_or_create_global_step()
  #   mris, labels = inputs()

  #DEBUG TOOLS
  #from tensorflow.python import debug as tf_debug
  #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

  mris, labels = brats.inputs()
  
  init_op = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    record = mris[:, 1, :, :, :, :]
    print record
    a = record.eval()
    print a.shape
    print labels.eval()

    print "Done"
    coord.request_stop()
    coord.join(threads)
    sess.close()

def my_reader(directory):
  filenames = ["/t1.mha", "/t1c.mha", "/t2.mha", "/flair.mha", "/ot.mha"]
  m = sio.imread(directory + filenames[0], plugin='simpleitk')
  t = np.array([m])
  for i in xrange(1,5):
    p = sio.imread(directory + filenames[i], plugin='simpleitk')
    t = np.append(t, [p], axis=0)
  
  return t 

def train_dev():
  directories = tf.train.match_filenames_once(brats.FLAGS.data_dir + "*brats*")
  directories_queue = tf.train.string_input_producer(directories)
  directory = directories_queue.dequeue()

  t = tf.py_func(my_reader, [directory], tf.int16)

  #mris = tf.constant(t, tf.float32)

  f_name = tf.string_split([directory + "/t1.mha"], ' ')

  init_op = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print "t"
    a = t.eval()
    
    print a.shape
    for i in xrange(5):
      print np.sum(a[i])

    coord.request_stop()
    coord.join(threads)
    sess.close()

  return

def main(argv=None): # pylint: disable=unused-argument
  #train()
  train_dev()

if __name__ == '__main__':
  tf.app.run()
