import numpy as np
import tensorflow as tf

import os.path
import pickle
import re

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 3, """Number of images to process in a batch.""")

tf.app.flags.DEFINE_integer('max_steps', 100, """Number of batches to train.""")

tf.app.flags.DEFINE_integer('set_quantity', 10, """Number of sets to run.""")

### farmer ###
tf.app.flags.DEFINE_string('list_dir',
                           '/home/ubuntu/dl/BRATS2015/',
                           """Path to 'input list' files.""")

tf.app.flags.DEFINE_string('data_dir',
                           FLAGS.list_dir + 'BRATS2015_Training/',
                           """Path to the BRATS *.in files.""")

""" Read BRATS """

# Global constants describing the BRATS data set
NUM_FILES_PER_ENTRY = 5
MRI_DIMS = 3
MHA_HEIGHT = 155
MHA_WIDTH = 240
MHA_DEPTH = 240
MHA_CHANNEL = 1

NUM_CLASSES = 2

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 20
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10 # DON'T KNOW YET

def _const1(): return tf.constant([1])
def _const4(): return tf.constant([4])

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

  result.label = tf.cond(compare_label, _const4, _const1)

  result.mris = tf.reshape(f_data, [NUM_FILES_PER_ENTRY,
                                   MHA_HEIGHT,
                                   MHA_WIDTH,
                                   MHA_DEPTH,
                                   MHA_CHANNEL])
  return result


def generate_record_and_label_batch(mris, label, min_queue_examples,
                                 batch_size, shuffle):
  # Generate batch
  num_preprocess_threads = 8

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

def inputs(is_train_list, label_idx, batch_size):
  ## Create a queue of filenames to read
  _list = get_list(FLAGS.data_dir, inputs.set_number, is_train_list)

  filename_queue = tf.train.string_input_producer(_list)

  read_input = read_brats(filename_queue, label_idx)

  casted_mris = tf.cast(read_input.mris, tf.float32)

  read_input.label.set_shape([1])

  # Ensure random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d BRATS records before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of 5-mri records and labels by building up a queue of records
  return generate_record_and_label_batch(casted_mris, read_input.label,
                                      min_queue_examples, batch_size,
                                      shuffle=False)

def get_list(data_dir, set_number, is_train=True):
  list_name = ""
  if is_train:
    list_name = FLAGS.list_dir + 'train_list' + str(set_number)
  else:
    list_name = FLAGS.list_dir + 'test_list' + str(set_number)

  _list = []
  with open(list_name, 'rb') as f:
    _list = pickle.load(f)

  _list = [data_dir + record for record in _list]

  print "List name: " + list_name
  print "Set number: " + str(set_number)
  print "Number of input files: " + str(len(_list))

  return _list

""" brats.py """

TOWER_NAME = 'tower'


def _activation_summary(x):
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activation', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  dtype = tf.float32
  var = _variable_on_cpu(name,
                         shape,
                         tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def inference(mris):
  # T1 T1c T2 Flair OT
  # conv1
  with tf.variable_scope('conv1_t1') as scope:
    kernel_t1 = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 1, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_t1 = tf.nn.conv3d(mris[:, 0, :, :, :, :],
                           kernel_t1,
                           [1, 1, 1, 1, 1],
                           padding='SAME')
    biases_t1 = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation_t1 = tf.nn.bias_add(conv_t1, biases_t1)
    conv1_t1 = tf.nn.relu(pre_activation_t1, name=scope.name)
    _activation_summary(conv1_t1)

  with tf.variable_scope('conv1_t1c') as scope:
    kernel_t1c = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 1, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_t1c = tf.nn.conv3d(mris[:, 1, :, :, :, :],
                           kernel_t1c,
                           [1, 1, 1, 1, 1],
                           padding='SAME')
    biases_t1c = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation_t1c = tf.nn.bias_add(conv_t1c, biases_t1c)
    conv1_t1c = tf.nn.relu(pre_activation_t1c, name=scope.name)
    _activation_summary(conv1_t1c)

  with tf.variable_scope('conv1_t2') as scope:
    kernel_t2 = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 1, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_t2 = tf.nn.conv3d(mris[:, 2, :, :, :, :],
                           kernel_t2,
                           [1, 1, 1, 1, 1],
                           padding='SAME')
    biases_t2 = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation_t2 = tf.nn.bias_add(conv_t2, biases_t2)
    conv1_t2 = tf.nn.relu(pre_activation_t2, name=scope.name)
    _activation_summary(conv1_t2)
  
  with tf.variable_scope('conv1_fl') as scope:
    kernel_fl = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 1, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_fl = tf.nn.conv3d(mris[:, 3, :, :, :, :],
                           kernel_fl,
                           [1, 1, 1, 1, 1],
                           padding='SAME')
    biases_fl = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation_fl = tf.nn.bias_add(conv_fl, biases_fl)
    conv1_fl = tf.nn.relu(pre_activation_fl, name=scope.name)
    _activation_summary(conv1_fl)
  
  with tf.variable_scope('conv1_ot') as scope:
    kernel_ot = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 1, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_ot = tf.nn.conv3d(mris[:, 4, :, :, :, :],
                           kernel_ot,
                           [1, 1, 1, 1, 1],
                           padding='SAME')
    biases_ot = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation_ot = tf.nn.bias_add(conv_ot, biases_ot)
    conv1_ot = tf.nn.relu(pre_activation_ot, name=scope.name)
    _activation_summary(conv1_ot)

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

  # conv2
  with tf.variable_scope('conv2_t1') as scope:
    kernel_t1 = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_t1 = tf.nn.conv3d(pool1_t1,
                           kernel_t1,
                           [1, 1, 1, 1, 1],
                           padding='SAME')
    biases_t1 = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation_t1 = tf.nn.bias_add(conv_t1, biases_t1)
    conv2_t1 = tf.nn.relu(pre_activation_t1, name=scope.name)
    _activation_summary(conv2_t1)

  with tf.variable_scope('conv2_t1c') as scope:
    kernel_t1c = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_t1c = tf.nn.conv3d(pool1_t1c,
                           kernel_t1c,
                           [1, 1, 1, 1, 1],
                           padding='SAME')
    biases_t1c = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation_t1c = tf.nn.bias_add(conv_t1c, biases_t1c)
    conv2_t1c = tf.nn.relu(pre_activation_t1c, name=scope.name)
    _activation_summary(conv2_t1c)

  with tf.variable_scope('conv2_t2') as scope:
    kernel_t2 = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_t2 = tf.nn.conv3d(pool1_t2,
                           kernel_t2,
                           [1, 1, 1, 1, 1],
                           padding='SAME')
    biases_t2 = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation_t2 = tf.nn.bias_add(conv_t2, biases_t2)
    conv2_t2 = tf.nn.relu(pre_activation_t2, name=scope.name)
    _activation_summary(conv2_t2)
  
  with tf.variable_scope('conv2_fl') as scope:
    kernel_fl = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_fl = tf.nn.conv3d(pool1_fl,
                           kernel_fl,
                           [1, 1, 1, 1, 1],
                           padding='SAME')
    biases_fl = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation_fl = tf.nn.bias_add(conv_fl, biases_fl)
    conv2_fl = tf.nn.relu(pre_activation_fl, name=scope.name)
    _activation_summary(conv2_fl)
  
  with tf.variable_scope('conv2_ot') as scope:
    kernel_ot = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_ot = tf.nn.conv3d(pool1_ot,
                           kernel_ot,
                           [1, 1, 1, 1, 1],
                           padding='SAME')
    biases_ot = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation_ot = tf.nn.bias_add(conv_ot, biases_ot)
    conv2_ot = tf.nn.relu(pre_activation_ot, name=scope.name)
    _activation_summary(conv2_ot)

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

  # local3
  with tf.variable_scope('local3') as scope:
    reshape = tf.concat([tf.reshape(pool2_t1, [FLAGS.batch_size, -1]),
                        tf.reshape(pool2_t1c, [FLAGS.batch_size, -1]),
                        tf.reshape(pool2_t2, [FLAGS.batch_size, -1]),
                        tf.reshape(pool2_fl, [FLAGS.batch_size, -1]),
                        tf.reshape(pool2_ot, [FLAGS.batch_size, -1])],
                        axis=1)
    print reshape.get_shape()
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  with tf.variable_scope('local5') as scope:
    weights = _variable_with_weight_decay('weights', shape=[192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
    local5 = tf.nn.relu(tf.matmul(local4, weights) + biases, name=scope.name)
    _activation_summary(local5)

  return local5

def proceed():
  records, labels = inputs(is_train_list=True,
                            label_idx=len(FLAGS.data_dir),
                            batch_size=FLAGS.batch_size)

  batch_logits = inference(records)

  #batch_loss = loss(batch_logits, labels)

  #train_op = train(loss, global_step)
  
  # break hanging queue - DEBUGGING only
  config = tf.ConfigProto()
  config.operation_timeout_in_ms = 50000
  sess = tf.Session(config=config)

  sess.run(tf.global_variables_initializer())

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  for step in xrange(FLAGS.max_steps):
    print "Step: " + str(step)
    print(sess.run(batch_logits))

  coord.request_stop()
  coord.join(threads)

  sess.close()


def main(argv=None):
  for set_number in xrange(1, FLAGS.set_quantity):
    inputs.set_number = set_number
    proceed()

if __name__ == '__main__':
  tf.app.run()
