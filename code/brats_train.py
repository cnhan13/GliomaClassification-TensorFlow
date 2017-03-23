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


def train():
  """Train BRATS for a number of steps"""
  # with tf.Graph().as_default():
  #   global_step = tf.contrib.framework.get_or_create_global_step()
  #   mris, labels = inputs()

  #DEBUG TOOLS
  #from tensorflow.python import debug as tf_debug
  #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

  records, labels = brats.inputs()
  
  init_op = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    mri = records[:, 1, :, :, :, :]
    print mri 
    _mri = mri.eval()
    print _mri.shape
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

def _const1(): return tf.constant([1])
def _const4(): return tf.constant([4])

def train_dev():
  directories = tf.train.match_filenames_once(brats.FLAGS.data_dir + "*brats*")
  directories_queue = tf.train.string_input_producer(directories)
  directory = directories_queue.dequeue()

  t = tf.py_func(my_reader, [directory], tf.int16)
  a = t[0]
  print t
  print a

  _H_72 = tf.constant(72, dtype=tf.uint8)

  directory_uint8 = tf.decode_raw(directory, tf.uint8)
  
  compare_op = tf.equal(directory_uint8[45], _H_72)
  
  label = tf.cond(compare_op, _const4, _const1)

  #mris = tf.constant(t, tf.float32)

  f_name = tf.string_split([directory + "/t1.mha"], ' ')

  init_op = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print "t"
    print t
    print t.eval().shape
    print "a"
    print a
    print a.eval().shape
    
    for i in xrange(5):
      print np.sum([i])
    print "label"
    print label.eval()

    coord.request_stop()
    coord.join(threads)
    sess.close()

  return

def train_dev2():
  records, label_batch = brats_input.inputs(brats.FLAGS.data_dir,
                              len(brats.FLAGS.data_dir),
                              brats.FLAGS.batch_size)
  
  init_op = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print "records"
    print records
    print records.eval()


    coord.request_stop()
    coord.join(threads)
    sess.close()

  return


def main(argv=None): # pylint: disable=unused-argument
  #train()
  #train_dev()
  train_dev2()

if __name__ == '__main__':
  tf.app.run()
