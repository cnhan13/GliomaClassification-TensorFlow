import numpy as np
import tensorflow as tf

import os.path
import pickle
import re
import sys

from datetime import datetime
import time


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 5,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_train_steps_per_eval', 200,
                            """Number of steps between 2 evaluations.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('per_process_gpu_memory_fraction', 0.5,
                            """Fraction of GPU memory used for training""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                           """How often to log results to the console.""")
tf.app.flags.DEFINE_integer('operation_timeout_in_ms', 60000,
                            """Time to wait for queue to load data.""")

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


# Global constants describing the BRATS data set
NUM_FILES_PER_ENTRY = 5
MRI_DIMS = 3
MHA_DEPTH = 155
MHA_HEIGHT = 240
MHA_WIDTH = 240
BRAIN_DEPTH = 149
BRAIN_HEIGHT = 185
BRAIN_WIDTH = 162
TUMOR_DEPTH = 115
TUMOR_HEIGHT = 166
TUMOR_WIDTH = 129

VOLUME_DEPTH = MHA_DEPTH
VOLUME_HEIGHT = MHA_HEIGHT
VOLUME_WIDTH = MHA_WIDTH
VOLUME_CHANNEL = 1
VARIANCE_EPSILON = 1.0 / (VOLUME_DEPTH * VOLUME_HEIGHT * VOLUME_WIDTH)
NUM_CLASSES = 2

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = -1
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = -1 # DON'T KNOW YET

# Contants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 100         # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.96  # Learning rate decay factor
INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.

def deb(tensor, msg):
  return tf.Print(tensor, [tensor], message=msg + ": ", summarize=100)

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
                                   VOLUME_DEPTH,
                                   VOLUME_HEIGHT,
                                   VOLUME_WIDTH,
                                   VOLUME_CHANNEL])
  return result


def generate_record_and_label_batch(mris, label, min_queue_examples,
                                 batch_size, shuffle):
  # Generate batch
  num_preprocess_threads = 16

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

def inputs_distorted(is_tumor_cropped, is_train_list, batch_size, set_number_str):
  ## Create a queue of filenames to read
  _list, label_idx = get_list(set_number_str, is_tumor_cropped, is_train_list)

  filename_queue = tf.train.string_input_producer(_list)

  read_input = read_brats(filename_queue, label_idx)

  casted_mris = tf.cast(read_input.mris, tf.float32)
  
  ot = casted_mris[4, :, :, :, :]

  t1 = tf.image.random_brightness(casted_mris[0, :, :, :, :], max_delta=63)
  t1 = tf.image.random_contrast(t1, lower=0.2, upper=1.8)
  t1_mean, t1_var = tf.nn.moments(t1, [0, 1, 2])
  t1 = tf.nn.batch_normalization(t1, t1_mean, t1_var, None, None, VARIANCE_EPSILON) * ot
  
  t1c = tf.image.random_brightness(casted_mris[1, :, :, :, :], max_delta=63)
  t1c = tf.image.random_contrast(t1c, lower=0.2, upper=1.8)
  t1c_mean, t1c_var = tf.nn.moments(t1c, [0, 1, 2])
  t1c = tf.nn.batch_normalization(t1c, t1c_mean, t1c_var, None, None, VARIANCE_EPSILON) * ot
  
  t2 = tf.image.random_brightness(casted_mris[2, :, :, :, :], max_delta=63)
  t2 = tf.image.random_contrast(t2, lower=0.2, upper=1.8)
  t2_mean, t2_var = tf.nn.moments(t2, [0, 1, 2])
  t2 = tf.nn.batch_normalization(t2, t2_mean, t2_var, None, None, VARIANCE_EPSILON) * ot

  fl = tf.image.random_brightness(casted_mris[3, :, :, :, :], max_delta=63)
  fl = tf.image.random_contrast(fl, lower=0.2, upper=1.8)
  fl_mean, fl_var = tf.nn.moments(fl, [0, 1, 2])
  fl = tf.nn.batch_normalization(fl, fl_mean, fl_var, None, None, VARIANCE_EPSILON) * ot
  
  normalized_mris = tf.stack([t1, t1c, t2, fl])

  read_input.label.set_shape([1])

  # Ensure random shuffling has good mixing properties.
  if is_train_list:
    min_fraction_of_examples_in_queue = 0.8
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d BRATS records before starting to train. '
           'This will take a few minutes.' % min_queue_examples)
  else:
    min_fraction_of_examples_in_queue = 1.0
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d BRATS records before starting to evaluate. '
           'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of 5-mri records and labels by building up a queue of records
  return generate_record_and_label_batch(normalized_mris, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)


def inputs(is_tumor_cropped, is_train_list, batch_size, set_number_str, num_epochs, capacity):
  ## Create a queue of filenames to read
  _list, label_idx = get_list(set_number_str, is_tumor_cropped, is_train_list)

  filename_queue = tf.train.string_input_producer(_list,
                                                  num_epochs=num_epochs,
                                                  capacity=capacity)

  read_input = read_brats(filename_queue, label_idx)

  casted_mris = tf.cast(read_input.mris, tf.float32)
  
  ot = casted_mris[4, :, :, :, :]

  t1 = casted_mris[0, :, :, :, :]
  t1_mean, t1_var = tf.nn.moments(t1, [0, 1, 2])
  t1 = tf.nn.batch_normalization(t1, t1_mean, t1_var, None, None, VARIANCE_EPSILON) * ot
  
  t1c = casted_mris[1, :, :, :, :]
  t1c_mean, t1c_var = tf.nn.moments(t1c, [0, 1, 2])
  t1c = tf.nn.batch_normalization(t1c, t1c_mean, t1c_var, None, None, VARIANCE_EPSILON) * ot
  
  t2 = casted_mris[2, :, :, :, :]
  t2_mean, t2_var = tf.nn.moments(t2, [0, 1, 2])
  t2 = tf.nn.batch_normalization(t2, t2_mean, t2_var, None, None, VARIANCE_EPSILON) * ot

  fl = casted_mris[3, :, :, :, :]
  fl_mean, fl_var = tf.nn.moments(fl, [0, 1, 2])
  fl = tf.nn.batch_normalization(fl, fl_mean, fl_var, None, None, VARIANCE_EPSILON) * ot
  
  normalized_mris = tf.stack([t1, t1c, t2, fl])

  read_input.label.set_shape([1])

  # Ensure random shuffling has good mixing properties.
  if is_train_list:
    min_fraction_of_examples_in_queue = 0.8
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d BRATS records before starting to train. '
           'This will take a few minutes.' % min_queue_examples)
  else:
    min_fraction_of_examples_in_queue = 1.0
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d BRATS records before starting to evaluate. '
           'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of 5-mri records and labels by building up a queue of records
  return generate_record_and_label_batch(normalized_mris, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)


def get_list(set_number_str, is_tumor_cropped=False, is_train=True):
  global VOLUME_DEPTH
  global VOLUME_WIDTH
  global VOLUME_HEIGHT
  global NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  global NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  data_dir = FLAGS.common_dir
  if is_tumor_cropped:
    data_dir += FLAGS.tumor_dir
    VOLUME_DEPTH = TUMOR_DEPTH
    VOLUME_WIDTH = TUMOR_WIDTH
    VOLUME_HEIGHT = TUMOR_HEIGHT
  else:
    data_dir += FLAGS.brain_dir
    VOLUME_DEPTH = BRAIN_DEPTH
    VOLUME_WIDTH = BRAIN_WIDTH
    VOLUME_HEIGHT = BRAIN_HEIGHT

  list_name = data_dir

  if is_train:
    list_name += 'train_list' + set_number_str
  else:
    list_name += 'test_list' + set_number_str

  in_dir = data_dir + FLAGS.in_dir

  with open(list_name, 'rb') as f:
    _list = pickle.load(f)

    _list = [in_dir + record for record in _list]

    if is_train:
      NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = len(_list)
    else:
      NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = len(_list)

    print "List name: " + list_name
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


def _4conv_pool(input_layer, conv1_scope, conv2_scope,
                conv3_scope, conv4_scope, pool_scope,
                conv_kernel_shape, conv_kernel_stride,
                conv_mid_kernel_shape, conv_mid_kernel_stride,
                pool_kernel_shape, pool_kernel_stride):

  with tf.variable_scope(conv1_scope) as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=conv_kernel_shape,
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv3d(input_layer,
                         kernel,
                         conv_kernel_stride,
                         padding='SAME')
    biases = _variable_on_cpu('biases',
                              conv_kernel_shape[-1],
                              tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv1)

  with tf.variable_scope(conv2_scope) as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=conv_mid_kernel_shape,
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv3d(conv1,
                         kernel,
                         conv_mid_kernel_stride,
                         padding='SAME')
    biases = _variable_on_cpu('biases',
                              conv_mid_kernel_shape[-1],
                              tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)

  with tf.variable_scope(conv3_scope) as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=conv_mid_kernel_shape,
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv3d(conv2,
                         kernel,
                         conv_mid_kernel_stride,
                         padding='SAME')
    biases = _variable_on_cpu('biases',
                              conv_mid_kernel_shape[-1],
                              tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv3)

  with tf.variable_scope(conv4_scope) as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=conv_mid_kernel_shape,
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv3d(conv3,
                         kernel,
                         conv_mid_kernel_stride,
                         padding='SAME')
    biases = _variable_on_cpu('biases',
                              conv_mid_kernel_shape[-1],
                              tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv4)

  pool = tf.nn.max_pool3d(conv4,
                          ksize=pool_kernel_shape,
                          strides=pool_kernel_stride,
                          padding='SAME',
                          name=pool_scope)
  return pool


def _2conv_pool(input_layer, conv1_scope, conv2_scope, pool_scope,
                conv_kernel_shape, conv_kernel_stride,
                conv_mid_kernel_shape, conv_mid_kernel_stride,
                pool_kernel_shape, pool_kernel_stride):

  with tf.variable_scope(conv1_scope) as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=conv_kernel_shape,
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv3d(input_layer,
                         kernel,
                         conv_kernel_stride,
                         padding='SAME')
    biases = _variable_on_cpu('biases',
                              conv_kernel_shape[-1],
                              tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv1)

  with tf.variable_scope(conv2_scope) as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=conv_mid_kernel_shape,
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv3d(conv1,
                         kernel,
                         conv_mid_kernel_stride,
                         padding='SAME')
    biases = _variable_on_cpu('biases',
                              conv_mid_kernel_shape[-1],
                              tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)

  pool = tf.nn.max_pool3d(conv2,
                          ksize=pool_kernel_shape,
                          strides=pool_kernel_stride,
                          padding='SAME',
                          name=pool_scope)
  return pool


def inference(mris, keep_prob):
  # T1 T1c T2 Flair OT
  # conv1
  # (batch_size, 5, 149, 185, 162)
  # (batch_size, 5, 115, 168, 129)

  group1_2_t1 = _2conv_pool(mris[:, 0, :, :, :, :], 'conv1_t1', 'conv2_t1', 'pool2_t1',
                        [3, 3, 3, 1, 4], [1, 1, 1, 1, 1],
                        [3, 3, 3, 4, 4], [1, 1, 1, 1, 1],
                        [1, 2, 2, 2, 1], [1, 2, 2, 2, 1])
  group1_2_t1c = _2conv_pool(mris[:, 1, :, :, :, :],
                        'conv1_t1c', 'conv2_t1c', 'pool2_t1c',
                        [3, 3, 3, 1, 4], [1, 1, 1, 1, 1],
                        [3, 3, 3, 4, 4], [1, 1, 1, 1, 1],
                        [1, 2, 2, 2, 1], [1, 2, 2, 2, 1])
  group1_2_t2 = _2conv_pool(mris[:, 2, :, :, :, :],
                        'conv1_t2', 'conv2_t2', 'pool2_t2',
                        [3, 3, 3, 1, 4], [1, 1, 1, 1, 1],
                        [3, 3, 3, 4, 4], [1, 1, 1, 1, 1],
                        [1, 2, 2, 2, 1], [1, 2, 2, 2, 1])
  group1_2_fl = _2conv_pool(mris[:, 3, :, :, :, :],
                        'conv1_fl', 'conv2_fl', 'pool2_fl',
                        [3, 3, 3, 1, 4], [1, 1, 1, 1, 1],
                        [3, 3, 3, 4, 4], [1, 1, 1, 1, 1],
                        [1, 2, 2, 2, 1], [1, 2, 2, 2, 1])
  print group1_2_t1
  
  group3_4_t1 = _2conv_pool(group1_2_t1, 'conv3_t1', 'conv4_t1', 'pool4_t1',
                        [3, 3, 3, 4, 8], [1, 1, 1, 1, 1],
                        [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                        [1, 2, 2, 2, 1], [1, 2, 2, 2, 1])
  group3_4_t1c = _2conv_pool(group1_2_t1c, 'conv3_t1c', 'conv4_t1c', 'pool4_t1c',
                        [3, 3, 3, 4, 8], [1, 1, 1, 1, 1],
                        [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                        [1, 2, 2, 2, 1], [1, 2, 2, 2, 1])
  group3_4_t2 = _2conv_pool(group1_2_t2, 'conv3_t2', 'conv4_t2', 'pool4_t2',
                        [3, 3, 3, 4, 8], [1, 1, 1, 1, 1],
                        [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                        [1, 2, 2, 2, 1], [1, 2, 2, 2, 1])
  group3_4_fl = _2conv_pool(group1_2_fl, 'conv3_fl', 'conv4_fl', 'pool4_fl',
                        [3, 3, 3, 4, 8], [1, 1, 1, 1, 1],
                        [3, 3, 3, 8, 8], [1, 1, 1, 1, 1],
                        [1, 2, 2, 2, 1], [1, 2, 2, 2, 1])
  print group3_4_t1
  
  group5_6_7_8_t1 = _4conv_pool(group3_4_t1,
      'conv5_t1', 'conv6_t1', 'conv7_t1', 'conv8_t1', 'pool8_t1',
                        [3, 3, 3, 8, 16], [1, 1, 1, 1, 1],
                        [3, 3, 3, 16, 16], [1, 1, 1, 1, 1],
                        [1, 2, 2, 2, 1], [1, 2, 2, 2, 1])
  group5_6_7_8_t1c = _4conv_pool(group3_4_t1c,
      'conv5_t1c', 'conv6_t1c', 'conv7_t1c', 'conv8_t1c', 'pool8_t1c',
                        [3, 3, 3, 8, 16], [1, 1, 1, 1, 1],
                        [3, 3, 3, 16, 16], [1, 1, 1, 1, 1],
                        [1, 2, 2, 2, 1], [1, 2, 2, 2, 1])
  group5_6_7_8_t2 = _4conv_pool(group3_4_t2,
      'conv5_t2', 'conv6_t2', 'conv7_t2', 'conv8_t2', 'pool8_t2',
                        [3, 3, 3, 8, 16], [1, 1, 1, 1, 1],
                        [3, 3, 3, 16, 16], [1, 1, 1, 1, 1],
                        [1, 2, 2, 2, 1], [1, 2, 2, 2, 1])
  group5_6_7_8_fl = _4conv_pool(group3_4_fl,
      'conv5_fl', 'conv6_fl', 'conv7_fl', 'conv8_fl', 'pool8_fl',
                        [3, 3, 3, 8, 16], [1, 1, 1, 1, 1],
                        [3, 3, 3, 16, 16], [1, 1, 1, 1, 1],
                        [1, 2, 2, 2, 1], [1, 2, 2, 2, 1])
  print group5_6_7_8_t1
  
  group9_10_11_12_t1 = _4conv_pool(group5_6_7_8_t1,
      'conv9_t1', 'conv10_t1', 'conv11_t1', 'conv12_t1', 'pool12_t1',
                        [3, 3, 3, 16, 32], [1, 1, 1, 1, 1],
                        [3, 3, 3, 32, 32], [1, 1, 1, 1, 1],
                        [1, 2, 2, 2, 1], [1, 2, 2, 2, 1])
  group9_10_11_12_t1c = _4conv_pool(group5_6_7_8_t1c,
      'conv9_t1c', 'conv10_t1c', 'conv11_t1c', 'conv12_t1c', 'pool12_t1c',
                        [3, 3, 3, 16, 32], [1, 1, 1, 1, 1],
                        [3, 3, 3, 32, 32], [1, 1, 1, 1, 1],
                        [1, 2, 2, 2, 1], [1, 2, 2, 2, 1])
  group9_10_11_12_t2 = _4conv_pool(group5_6_7_8_t2,
      'conv9_t2', 'conv10_t2', 'conv11_t2', 'conv12_t2', 'pool12_t2',
                        [3, 3, 3, 16, 32], [1, 1, 1, 1, 1],
                        [3, 3, 3, 32, 32], [1, 1, 1, 1, 1],
                        [1, 2, 2, 2, 1], [1, 2, 2, 2, 1])
  group9_10_11_12_fl = _4conv_pool(group5_6_7_8_fl,
      'conv9_fl', 'conv10_fl', 'conv11_fl', 'conv12_fl', 'pool12_fl',
                        [3, 3, 3, 16, 32], [1, 1, 1, 1, 1],
                        [3, 3, 3, 32, 32], [1, 1, 1, 1, 1],
                        [1, 2, 2, 2, 1], [1, 2, 2, 2, 1])
  print group9_10_11_12_t1

  group13_14_15_16_t1 = _4conv_pool(group9_10_11_12_t1,
      'conv13_t1', 'conv14_t1', 'conv15_t1', 'conv16_t1', 'pool16_t1',
                        [3, 3, 3, 32, 64], [1, 1, 1, 1, 1],
                        [3, 3, 3, 64, 64], [1, 1, 1, 1, 1],
                        [1, 2, 2, 2, 1], [1, 2, 2, 2, 1])
  group13_14_15_16_t1c = _4conv_pool(group9_10_11_12_t1c,
      'conv13_t1c', 'conv14_t1c', 'conv15_t1c', 'conv16_t1c', 'pool16_t1c',
                        [3, 3, 3, 32, 64], [1, 1, 1, 1, 1],
                        [3, 3, 3, 64, 64], [1, 1, 1, 1, 1],
                        [1, 2, 2, 2, 1], [1, 2, 2, 2, 1])
  group13_14_15_16_t2 = _4conv_pool(group9_10_11_12_t2,
      'conv13_t2', 'conv14_t2', 'conv15_t2', 'conv16_t2', 'pool16_t2',
                        [3, 3, 3, 32, 64], [1, 1, 1, 1, 1],
                        [3, 3, 3, 64, 64], [1, 1, 1, 1, 1],
                        [1, 2, 2, 2, 1], [1, 2, 2, 2, 1])
  group13_14_15_16_fl = _4conv_pool(group9_10_11_12_fl,
      'conv13_fl', 'conv14_fl', 'conv15_fl', 'conv16_fl', 'pool16_fl',
                        [3, 3, 3, 32, 64], [1, 1, 1, 1, 1],
                        [3, 3, 3, 64, 64], [1, 1, 1, 1, 1],
                        [1, 2, 2, 2, 1], [1, 2, 2, 2, 1])
  print group13_14_15_16_t1


  # local17
  with tf.variable_scope('local17') as scope:
    """
    TensorFlow r1.0
    tf.concat(values, axis, name='concat')
    """
    reshape = tf.concat([tf.reshape(group13_14_15_16_t1, [FLAGS.batch_size, -1]),
                        tf.reshape(group13_14_15_16_t1c, [FLAGS.batch_size, -1]),
                        tf.reshape(group13_14_15_16_t2, [FLAGS.batch_size, -1]),
                        tf.reshape(group13_14_15_16_fl, [FLAGS.batch_size, -1])],
                        axis=1)
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 256],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    local17 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    local17 = tf.nn.dropout(local17, keep_prob)
    _activation_summary(local17)
  print local17

  # local18
  with tf.variable_scope('local18') as scope:
    weights = _variable_with_weight_decay('weights', shape=[256, 128],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
    local18 = tf.nn.relu(tf.matmul(local17, weights) + biases, name=scope.name)
    local18 = tf.nn.dropout(local18, keep_prob)
    _activation_summary(local18)
  print local18

  with tf.variable_scope('local19') as scope:
    weights = _variable_with_weight_decay('weights', shape=[128, NUM_CLASSES],
                                          stddev=1/128.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
    local19 = tf.add(tf.matmul(local18, weights), biases, name=scope.name)
    local19 = tf.nn.dropout(local19, keep_prob)
    _activation_summary(local19)
  print local19

  return local19


def loss(logits, labels):
  # Calculate the average cross entropy loss across the batch
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
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
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size # 224 / 5 = 44
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY) # 44 * 100 = 880

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


def proceed(max_steps, is_tumor_cropped=False, with_reset=False):
  """
  with_reset:
    False - Proceed to restore variables for training
    True - Proceed with a new set of variables
  """

  with tf.Graph().as_default():
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    ckpt = tf.train.get_checkpoint_state(proceed.train_dir)

    global_step = tf.contrib.framework.get_or_create_global_step()

    last_global_step = -1
    if (not with_reset) and ckpt and ckpt.model_checkpoint_path:
      last_global_step = int(
          ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
      saver = tf.train.Saver()

    records, labels = inputs_distorted(is_tumor_cropped=is_tumor_cropped,
                                       is_train_list=True,
                                       batch_size=FLAGS.batch_size,
                                       set_number_str=proceed.set_number_str)
    
    batch_logits = inference(records, keep_prob)

    batch_loss = loss(batch_logits, labels)

    train_op = train(batch_loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = last_global_step
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(batch_loss,
            feed_dict={keep_prob: 0.5})  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.5f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=proceed.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=max_steps),
               tf.train.NanTensorHook(batch_loss),
               _LoggerHook()],
        config=config) as ma_sess: # my in French, seance

      if (not with_reset) and ckpt and ckpt.model_checkpoint_path:
        saver.restore(ma_sess, ckpt.model_checkpoint_path)

      while not ma_sess.should_stop():
        ma_sess.run(train_op)


def main(argv=None):
  """
    Terminal parameters
    sys.argv[1]: set_number_str to load as train_list + set_number_str
    sys.argv[2]:
      0: is_tumor_crop = False
      1: is_tumor_crop = True
  """

  proceed.set_number_str = sys.argv[1]
  is_tumor_cropped = (sys.argv[2] == '1')
  with_reset = (sys.argv[3] == '1')
  num_evals = int(sys.argv[4])

  max_steps = num_evals * FLAGS.num_train_steps_per_eval

  proceed.train_dir = FLAGS.common_dir
  proceed.train_dir += FLAGS.tumor_dir if is_tumor_cropped else FLAGS.brain_dir
  proceed.train_dir += FLAGS.train_dir + proceed.set_number_str

  if with_reset:
    if tf.gfile.Exists(proceed.train_dir):
      tf.gfile.DeleteRecursively(proceed.train_dir)
    tf.gfile.MakeDirs(proceed.train_dir)
  else:
    if not tf.gfile.Exists(proceed.train_dir):
      tf.gfile.MakeDirs(proceed.train_dir)

  proceed(max_steps, is_tumor_cropped=is_tumor_cropped, with_reset=with_reset)
  

if __name__ == '__main__':
  tf.app.run()
