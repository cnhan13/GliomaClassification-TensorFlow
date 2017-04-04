import numpy as np
import tensorflow as tf

import os.path
import pickle
import re
import sys

from datetime import datetime
import time


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 3, """Number of images to process in a batch.""")

tf.app.flags.DEFINE_integer('max_steps', 20, """Number of batches to train.""")

tf.app.flags.DEFINE_integer('set_quantity', 10, """Number of sets to run.""")

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_integer('log_frequency', 10,
                           """How often to log results to the console.""")

tf.app.flags.DEFINE_integer('operation_timeout_in_ms', 60000,
                            """Time to wait for queue to load data.""")

### farmer ###
#tf.app.flags.DEFINE_string('common_dir',
#                           '/home/ubuntu/dl/BRATS2015/',
#                           """Path to 'input list' files.""")
#
#tf.app.flags.DEFINE_string('brain_dir',
#                           'brain_cropped/',
#                           """Directory to 'brain cropped' data files.""")
#
#tf.app.flags.DEFINE_string('tumor_dir',
#                           'tumor_cropped/',
#                           """Directory to 'tumor cropped' data files.""")
#
#tf.app.flags.DEFINE_string('in_dir',
#                           'BRATS2015_Training/',
#                           """Directory to *.in records.""")

### audi ###
tf.app.flags.DEFINE_string('common_dir',
                           '/home/cnhan21/dl/BRATS2015/',
                           """Path to 'input list' files.""")

tf.app.flags.DEFINE_string('brain_dir',
                           'brain_cropped/',
                           """Directory to 'brain cropped' data files.""")

tf.app.flags.DEFINE_string('tumor_dir',
                           'tumor_cropped/',
                           """Directory to 'tumor cropped' data files.""")

tf.app.flags.DEFINE_string('in_dir',
                           'BRATS2015_Training/',
                           """Directory to *.in records.""")

tf.app.flags.DEFINE_string('train_dir',
                           'train',
                           """Directory where to write event logs """
                           """and checkpoint.""")

""" Read BRATS """

# Global constants describing the BRATS data set
NUM_FILES_PER_ENTRY = 5
MRI_DIMS = 3
#MHA_HEIGHT = 155
#MHA_WIDTH = 240
#MHA_DEPTH = 240
MHA_HEIGHT = 149
MHA_WIDTH = 185
MHA_DEPTH = 162
MHA_CHANNEL = 1

NUM_CLASSES = 2

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 6
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10 # DON'T KNOW YET

# Contants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 20         # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.


def deb(tensor, msg):
  return tf.Print(tensor, [tensor], message=msg + ": ", summarize=4)

def _const0(): return tf.constant([0])
def _const1(): return tf.constant([1])

def read_brats(filename_queue, label_idx):
  class BRATSRecord(object):
    pass
  result = BRATSRecord()

  reader = tf.WholeFileReader()
  f_name_reader, f_raw_reader = reader.read(filename_queue)

  f_data = tf.decode_raw(f_raw_reader, tf.int16)

  f_name_uint8 = tf.decode_raw(f_name_reader, tf.uint8)

  _H_72 = tf.constant(72, dtype=tf.uint8)

  compare_label = tf.equal(f_name_uint8[label_idx], _H_72)

  result.label = tf.cond(tf.reshape(compare_label, []), _const1, _const0)

  result.mris = tf.reshape(f_data, [NUM_FILES_PER_ENTRY,
                                   MHA_HEIGHT,
                                   MHA_WIDTH,
                                   MHA_DEPTH,
                                   MHA_CHANNEL])
  return result


def generate_record_and_label_batch(mris, label, min_queue_examples,
                                 batch_size, shuffle):
  # Generate batch
  num_preprocess_threads = 4

  if shuffle:
    records, label_batch = tf.train.shuffle_batch(
        [mris, label],
        batch_size = batch_size,
        num_threads = num_preprocess_threads,
        capacity = min_queue_examples + 3 * batch_size,
        min_after_dequeue = min_queue_examples)
  else:
    records, label_batch = tf.train.batch(
        [mris, label],
        batch_size = batch_size,
        num_threads = num_preprocess_threads,
        capacity = min_queue_examples + 3 * batch_size)
  
  # Display the training mris in the visualizer. HOW?
  # tf.summary.image('images', images)

  return records, tf.reshape(label_batch, [batch_size])


def inputs(is_tumor_cropped, is_train_list, batch_size):
  ## Create a queue of filenames to read
  _list, label_idx = get_list(is_tumor_cropped, is_train_list)

  filename_queue = tf.train.string_input_producer(_list)

  read_input = read_brats(filename_queue, label_idx)

  casted_mris = tf.cast(read_input.mris, tf.float32)

  read_input.label.set_shape([1])

  # Ensure random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.2
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d BRATS records before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of 5-mri records and labels by building up a queue of records
  return generate_record_and_label_batch(casted_mris, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)


def get_list(is_tumor_cropped=False, is_train=True):

  data_dir = FLAGS.common_dir
  data_dir += FLAGS.tumor_dir if is_tumor_cropped else FLAGS.brain_dir

  list_name = data_dir

  if is_train:
    list_name += 'train_list' + str(inputs.set_number)
  else:
    list_name += 'test_list' + str(inputs.set_number)

  in_dir = data_dir + FLAGS.in_dir

  with open(list_name, 'rb') as f:
    _list = pickle.load(f)

    _list = [in_dir + record for record in _list]

    print "List name: " + list_name
    print "Set number: " + str(inputs.set_number)
    print "Number of input files: " + str(len(_list))

    return _list, len(in_dir)
  
  return [], 0


""" brats.py """

TOWER_NAME = 'tower'

def _activation_summary(x):
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activation', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  dtype = tf.float32
  var = _variable_on_cpu(name,
                         shape,
                         tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def inference(mris):
  # T1 T1c T2 Flair OT
  # conv1
  # (batch_size, 5, 149, 185, 162)
  with tf.variable_scope('conv1_t1') as scope:
    kernel_t1 = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 1, 2],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_t1 = tf.nn.conv3d(mris[:, 0, :, :, :, :],
                           kernel_t1,
                           [1, 2, 2, 2, 1],
                           padding='SAME')
    biases_t1 = _variable_on_cpu('biases', [2], tf.constant_initializer(0.0))
    pre_activation_t1 = tf.nn.bias_add(conv_t1, biases_t1)
    conv1_t1 = tf.nn.relu(pre_activation_t1, name=scope.name)
    _activation_summary(conv1_t1)

  with tf.variable_scope('conv1_t1c') as scope:
    kernel_t1c = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 1, 2],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_t1c = tf.nn.conv3d(mris[:, 1, :, :, :, :],
                           kernel_t1c,
                           [1, 2, 2, 2, 1],
                           padding='SAME')
    biases_t1c = _variable_on_cpu('biases', [2], tf.constant_initializer(0.0))
    pre_activation_t1c = tf.nn.bias_add(conv_t1c, biases_t1c)
    conv1_t1c = tf.nn.relu(pre_activation_t1c, name=scope.name)
    _activation_summary(conv1_t1c)

  with tf.variable_scope('conv1_t2') as scope:
    kernel_t2 = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 1, 2],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_t2 = tf.nn.conv3d(mris[:, 2, :, :, :, :],
                           kernel_t2,
                           [1, 2, 2, 2, 1],
                           padding='SAME')
    biases_t2 = _variable_on_cpu('biases', [2], tf.constant_initializer(0.0))
    pre_activation_t2 = tf.nn.bias_add(conv_t2, biases_t2)
    conv1_t2 = tf.nn.relu(pre_activation_t2, name=scope.name)
    _activation_summary(conv1_t2)
  
  with tf.variable_scope('conv1_fl') as scope:
    kernel_fl = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 1, 2],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_fl = tf.nn.conv3d(mris[:, 3, :, :, :, :],
                           kernel_fl,
                           [1, 2, 2, 2, 1],
                           padding='SAME')
    biases_fl = _variable_on_cpu('biases', [2], tf.constant_initializer(0.0))
    pre_activation_fl = tf.nn.bias_add(conv_fl, biases_fl)
    conv1_fl = tf.nn.relu(pre_activation_fl, name=scope.name)
    _activation_summary(conv1_fl)
  
  with tf.variable_scope('conv1_ot') as scope:
    kernel_ot = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 1, 2],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_ot = tf.nn.conv3d(mris[:, 4, :, :, :, :],
                           kernel_ot,
                           [1, 2, 2, 2, 1],
                           padding='SAME')
    biases_ot = _variable_on_cpu('biases', [2], tf.constant_initializer(0.0))
    pre_activation_ot = tf.nn.bias_add(conv_ot, biases_ot)
    conv1_ot = tf.nn.relu(pre_activation_ot, name=scope.name)
    _activation_summary(conv1_ot)

  print conv1_ot

  # pool1
  pool1_t1 = tf.nn.max_pool3d(conv1_t1,
                              ksize=[1, 3, 3, 3, 1],
                              strides=[1, 2, 2, 2, 1],
                              padding='SAME',
                              name='pool1_t1')
  
  pool1_t1c = tf.nn.max_pool3d(conv1_t1c,
                              ksize=[1, 3, 3, 3, 1],
                              strides=[1, 2, 2, 2, 1],
                              padding='SAME',
                              name='pool1_t1c')

  pool1_t2 = tf.nn.max_pool3d(conv1_t2,
                              ksize=[1, 3, 3, 3, 1],
                              strides=[1, 2, 2, 2, 1],
                              padding='SAME',
                              name='pool1_t2')

  pool1_fl = tf.nn.max_pool3d(conv1_fl,
                              ksize=[1, 3, 3, 3, 1],
                              strides=[1, 2, 2, 2, 1],
                              padding='SAME',
                              name='pool1_fl')

  pool1_ot = tf.nn.max_pool3d(conv1_ot,
                              ksize=[1, 3, 3, 3, 1],
                              strides=[1, 2, 2, 2, 1],
                              padding='SAME',
                              name='pool1_ot')
  print pool1_ot

  # conv2
  with tf.variable_scope('conv2_t1') as scope:
    kernel_t1 = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 2, 1],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_t1 = tf.nn.conv3d(pool1_t1,
                           kernel_t1,
                           [1, 2, 2, 2, 1],
                           padding='SAME')
    biases_t1 = _variable_on_cpu('biases', [1], tf.constant_initializer(0.1))
    pre_activation_t1 = tf.nn.bias_add(conv_t1, biases_t1)
    conv2_t1 = tf.nn.relu(pre_activation_t1, name=scope.name)
    _activation_summary(conv2_t1)

  with tf.variable_scope('conv2_t1c') as scope:
    kernel_t1c = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 2, 1],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_t1c = tf.nn.conv3d(pool1_t1c,
                           kernel_t1c,
                           [1, 2, 2, 2, 1],
                           padding='SAME')
    biases_t1c = _variable_on_cpu('biases', [1], tf.constant_initializer(0.1))
    pre_activation_t1c = tf.nn.bias_add(conv_t1c, biases_t1c)
    conv2_t1c = tf.nn.relu(pre_activation_t1c, name=scope.name)
    _activation_summary(conv2_t1c)

  with tf.variable_scope('conv2_t2') as scope:
    kernel_t2 = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 2, 1],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_t2 = tf.nn.conv3d(pool1_t2,
                           kernel_t2,
                           [1, 2, 2, 2, 1],
                           padding='SAME')
    biases_t2 = _variable_on_cpu('biases', [1], tf.constant_initializer(0.1))
    pre_activation_t2 = tf.nn.bias_add(conv_t2, biases_t2)
    conv2_t2 = tf.nn.relu(pre_activation_t2, name=scope.name)
    _activation_summary(conv2_t2)
  
  with tf.variable_scope('conv2_fl') as scope:
    kernel_fl = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 2, 1],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_fl = tf.nn.conv3d(pool1_fl,
                           kernel_fl,
                           [1, 2, 2, 2, 1],
                           padding='SAME')
    biases_fl = _variable_on_cpu('biases', [1], tf.constant_initializer(0.1))
    pre_activation_fl = tf.nn.bias_add(conv_fl, biases_fl)
    conv2_fl = tf.nn.relu(pre_activation_fl, name=scope.name)
    _activation_summary(conv2_fl)
  
  with tf.variable_scope('conv2_ot') as scope:
    kernel_ot = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 2, 1],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_ot = tf.nn.conv3d(pool1_ot,
                           kernel_ot,
                           [1, 2, 2, 2, 1],
                           padding='SAME')
    biases_ot = _variable_on_cpu('biases', [1], tf.constant_initializer(0.1))
    pre_activation_ot = tf.nn.bias_add(conv_ot, biases_ot)
    conv2_ot = tf.nn.relu(pre_activation_ot, name=scope.name)
    _activation_summary(conv2_ot)

  print conv2_ot

  # pool2
  pool2_t1 = tf.nn.max_pool3d(conv2_t1,
                              ksize=[1, 3, 3, 3, 1],
                              strides=[1, 2, 2, 2, 1],
                              padding='SAME',
                              name='pool2_t1')
  
  pool2_t1c = tf.nn.max_pool3d(conv2_t1c,
                              ksize=[1, 3, 3, 3, 1],
                              strides=[1, 2, 2, 2, 1],
                              padding='SAME',
                              name='pool2_t1c')

  pool2_t2 = tf.nn.max_pool3d(conv2_t2,
                              ksize=[1, 3, 3, 3, 1],
                              strides=[1, 2, 2, 2, 1],
                              padding='SAME',
                              name='pool2_t2')

  pool2_fl = tf.nn.max_pool3d(conv2_fl,
                              ksize=[1, 3, 3, 3, 1],
                              strides=[1, 2, 2, 2, 1],
                              padding='SAME',
                              name='pool2_fl')

  pool2_ot = tf.nn.max_pool3d(conv2_ot,
                              ksize=[1, 3, 3, 3, 1],
                              strides=[1, 2, 2, 2, 1],
                              padding='SAME',
                              name='pool2_ot')
  
  print pool2_ot

  # conv3
  with tf.variable_scope('conv3_t1') as scope:
    kernel_t1 = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 1, 1],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_t1 = tf.nn.conv3d(pool2_t1,
                           kernel_t1,
                           [1, 1, 1, 1, 1],
                           padding='SAME')
    biases_t1 = _variable_on_cpu('biases', [1], tf.constant_initializer(0.1))
    pre_activation_t1 = tf.nn.bias_add(conv_t1, biases_t1)
    conv3_t1 = tf.nn.relu(pre_activation_t1, name=scope.name)
    _activation_summary(conv3_t1)

  with tf.variable_scope('conv3_t1c') as scope:
    kernel_t1c = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 1, 1],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_t1c = tf.nn.conv3d(pool2_t1c,
                           kernel_t1c,
                           [1, 1, 1, 1, 1],
                           padding='SAME')
    biases_t1c = _variable_on_cpu('biases', [1], tf.constant_initializer(0.1))
    pre_activation_t1c = tf.nn.bias_add(conv_t1c, biases_t1c)
    conv3_t1c = tf.nn.relu(pre_activation_t1c, name=scope.name)
    _activation_summary(conv3_t1c)

  with tf.variable_scope('conv3_t2') as scope:
    kernel_t2 = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 1, 1],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_t2 = tf.nn.conv3d(pool2_t2,
                           kernel_t2,
                           [1, 1, 1, 1, 1],
                           padding='SAME')
    biases_t2 = _variable_on_cpu('biases', [1], tf.constant_initializer(0.1))
    pre_activation_t2 = tf.nn.bias_add(conv_t2, biases_t2)
    conv3_t2 = tf.nn.relu(pre_activation_t2, name=scope.name)
    _activation_summary(conv3_t2)
  
  with tf.variable_scope('conv3_fl') as scope:
    kernel_fl = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 1, 1],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_fl = tf.nn.conv3d(pool2_fl,
                           kernel_fl,
                           [1, 1, 1, 1, 1],
                           padding='SAME')
    biases_fl = _variable_on_cpu('biases', [1], tf.constant_initializer(0.1))
    pre_activation_fl = tf.nn.bias_add(conv_fl, biases_fl)
    conv3_fl = tf.nn.relu(pre_activation_fl, name=scope.name)
    _activation_summary(conv3_fl)
  
  with tf.variable_scope('conv3_ot') as scope:
    kernel_ot = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 1, 1],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_ot = tf.nn.conv3d(pool2_ot,
                           kernel_ot,
                           [1, 1, 1, 1, 1],
                           padding='SAME')
    biases_ot = _variable_on_cpu('biases', [1], tf.constant_initializer(0.1))
    pre_activation_ot = tf.nn.bias_add(conv_ot, biases_ot)
    conv3_ot = tf.nn.relu(pre_activation_ot, name=scope.name)
    _activation_summary(conv3_ot)

  print conv3_ot

  # conv4
  with tf.variable_scope('conv4_t1') as scope:
    kernel_t1 = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 1, 1],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_t1 = tf.nn.conv3d(pool2_t1,
                           kernel_t1,
                           [1, 1, 1, 1, 1],
                           padding='SAME')
    biases_t1 = _variable_on_cpu('biases', [1], tf.constant_initializer(0.1))
    pre_activation_t1 = tf.nn.bias_add(conv_t1, biases_t1)
    conv4_t1 = tf.nn.relu(pre_activation_t1, name=scope.name)
    _activation_summary(conv4_t1)

  with tf.variable_scope('conv4_t1c') as scope:
    kernel_t1c = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 1, 1],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_t1c = tf.nn.conv3d(pool2_t1c,
                           kernel_t1c,
                           [1, 1, 1, 1, 1],
                           padding='SAME')
    biases_t1c = _variable_on_cpu('biases', [1], tf.constant_initializer(0.1))
    pre_activation_t1c = tf.nn.bias_add(conv_t1c, biases_t1c)
    conv4_t1c = tf.nn.relu(pre_activation_t1c, name=scope.name)
    _activation_summary(conv4_t1c)

  with tf.variable_scope('conv4_t2') as scope:
    kernel_t2 = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 1, 1],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_t2 = tf.nn.conv3d(pool2_t2,
                           kernel_t2,
                           [1, 1, 1, 1, 1],
                           padding='SAME')
    biases_t2 = _variable_on_cpu('biases', [1], tf.constant_initializer(0.1))
    pre_activation_t2 = tf.nn.bias_add(conv_t2, biases_t2)
    conv4_t2 = tf.nn.relu(pre_activation_t2, name=scope.name)
    _activation_summary(conv4_t2)
  
  with tf.variable_scope('conv4_fl') as scope:
    kernel_fl = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 1, 1],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_fl = tf.nn.conv3d(pool2_fl,
                           kernel_fl,
                           [1, 1, 1, 1, 1],
                           padding='SAME')
    biases_fl = _variable_on_cpu('biases', [1], tf.constant_initializer(0.1))
    pre_activation_fl = tf.nn.bias_add(conv_fl, biases_fl)
    conv4_fl = tf.nn.relu(pre_activation_fl, name=scope.name)
    _activation_summary(conv4_fl)
  
  with tf.variable_scope('conv4_ot') as scope:
    kernel_ot = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 1, 1],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_ot = tf.nn.conv3d(pool2_ot,
                           kernel_ot,
                           [1, 1, 1, 1, 1],
                           padding='SAME')
    biases_ot = _variable_on_cpu('biases', [1], tf.constant_initializer(0.1))
    pre_activation_ot = tf.nn.bias_add(conv_ot, biases_ot)
    conv4_ot = tf.nn.relu(pre_activation_ot, name=scope.name)
    _activation_summary(conv4_ot)

  print conv4_ot

  # pool4
  pool4_t1 = tf.nn.max_pool3d(conv4_t1,
                              ksize=[1, 3, 3, 3, 1],
                              strides=[1, 2, 2, 2, 1],
                              padding='SAME',
                              name='pool4_t1')
  
  pool4_t1c = tf.nn.max_pool3d(conv4_t1c,
                              ksize=[1, 3, 3, 3, 1],
                              strides=[1, 2, 2, 2, 1],
                              padding='SAME',
                              name='pool4_t1c')

  pool4_t2 = tf.nn.max_pool3d(conv4_t2,
                              ksize=[1, 3, 3, 3, 1],
                              strides=[1, 2, 2, 2, 1],
                              padding='SAME',
                              name='pool4_t2')

  pool4_fl = tf.nn.max_pool3d(conv4_fl,
                              ksize=[1, 3, 3, 3, 1],
                              strides=[1, 2, 2, 2, 1],
                              padding='SAME',
                              name='pool4_fl')

  pool4_ot = tf.nn.max_pool3d(conv4_ot,
                              ksize=[1, 3, 3, 3, 1],
                              strides=[1, 2, 2, 2, 1],
                              padding='SAME',
                              name='pool4_ot')
  print pool4_ot

  # local5
  with tf.variable_scope('local5') as scope:
    """
    TensorFlow r12.1:
    tf.concat(concat_dim, values, name='concat')
    
    TensorFlow r1.0
    tf.concat(values, axis, name='concat')
    """
    reshape = tf.concat([tf.reshape(pool4_t1, [FLAGS.batch_size, -1]),
                        tf.reshape(pool4_t1c, [FLAGS.batch_size, -1]),
                        tf.reshape(pool4_t2, [FLAGS.batch_size, -1]),
                        tf.reshape(pool4_fl, [FLAGS.batch_size, -1]),
                        tf.reshape(pool4_ot, [FLAGS.batch_size, -1])],
                        axis=1)
    print reshape
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local5 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local5)

  print local5

  # local6
  with tf.variable_scope('local6') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local6 = tf.nn.relu(tf.matmul(local5, weights) + biases, name=scope.name)
    _activation_summary(local6)

  print local6

  with tf.variable_scope('local7') as scope:
    weights = _variable_with_weight_decay('weights', shape=[192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
    local7 = tf.nn.relu(tf.matmul(local6, weights) + biases, name=scope.name)
    _activation_summary(local7)
    
  print local7

  return local7

def loss(logits, labels):
  # Calculate the average cross entropy loss across the batch
  #logits = deb(logits, "logits")
  labels = tf.cast(labels, tf.int64)
  #labels = deb(labels, "labels")
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  #cross_entropy = deb(cross_entropy, "cross entropy")
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss + all weight decay terms (L2 loss)
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach scalar summary
  for l in losses + [total_loss]:
    # Each loss is named '(raw)'
    # The moving average loss is named with the original loss name
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  # Variables that affect the learning rate
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def proceed(is_tumor_cropped=False):
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    records, labels = inputs(is_tumor_cropped=is_tumor_cropped,
                             is_train_list=True,
                             batch_size=FLAGS.batch_size)
    
    batch_logits = inference(records)

    batch_loss = loss(batch_logits, labels)

    batch_loss = deb(batch_loss, "batch loss: ")

    train_op = train(batch_loss, global_step)
    
    ### Break hanging queue - DEBUGGING only
    # config = tf.ConfigProto()
    # config.operation_timeout_in_ms = FLAGS.operation_timeout_in_ms
    # config.log_device_placement = FLAGS.log_device_placement
    # 
    # sess = tf.Session(config=config)

    # sess.run(tf.global_variables_initializer())

    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # for step in xrange(FLAGS.max_steps):
    #   print "Step: " + str(step)
    #   print(sess.run(train_op))

    # coord.request_stop()
    # coord.join(threads)

    # sess.close()

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(batch_loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=proceed.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(batch_loss),
               _LoggerHook()],
        config=tf.ConfigProto(
          log_device_placement=FLAGS.log_device_placement)) as ma_sess: # my in French, seance
      while not ma_sess.should_stop():
        ma_sess.run(train_op)



def main(argv=None):
  """
    Terminal parameters
    sys.argv[1]: set_number to load as train_list + str(set_number)
    sys.argv[2]:
      0: is_tumor_crop = False
      1: is_tumor_crop = True
  """

  inputs.set_number = sys.argv[1]
  is_tumor_cropped = (sys.argv[2] == 0)
  proceed.train_dir = FLAGS.common_dir
  proceed.train_dir += FLAGS.tumor_dir if is_tumor_cropped else FLAGS.brain_dir
  proceed.train_dir += FLAGS.train_dir + inputs.set_number

  if tf.gfile.Exists(proceed.train_dir):
    tf.gfile.DeleteRecursively(proceed.train_dir)
  tf.gfile.MakeDirs(proceed.train_dir)

  proceed(is_tumor_cropped=is_tumor_cropped)
  

if __name__ == '__main__':
  tf.app.run()
