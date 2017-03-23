import tensorflow as tf

import brats_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 5, """Number of images to process in a batch.""")
### nac ###
# tf.app.flags.DEFINE_string('data_dir',
#                            '/home/nhan/Desktop/x2goshared/BRATS2015/BRATS2015_Training/trial_in/',
#                            """Path to the BRATS *.in files.""")

### audi ###
#tf.app.flags.DEFINE_string('data_dir',
#                           '/home/cnhan21/media/disk/_home_nhan_Desktop_x2goshared/BRATS2015/BRATS2015_Training/trial_in/',
#                           """Path to the BRATS *.in files.""")
#tf.app.flags.DEFINE_string('data_dir', '/home/cnhan21/_home_nhan_Desktop_x2goshared/BRATS2015/BRATS2015_Training/trial_in', """Path to the BRATS *.in files.""")

### farmer ###
tf.app.flags.DEFINE_string('data_dir',
                           '/home/ubuntu/dl/BRATS2015/BRATS2015_Training/',
                           """Path to the *brats* directories""")

TOWER_NAME = 'tower'


def _activation_summary(x):
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activation', x)
  tf.summar.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


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


def inputs():
  
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')

  records, labels = brats_input.inputs(data_dir=FLAGS.data_dir,
                                             label_idx=len(FLAGS.data_dir),
                                             batch_size=FLAGS.batch_size)
  return records, labels


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
                                         shape=[3, 3, 3, 1, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_t1 = tf.nn.conv3d(pool1_t1,
                           kernel_t1,
                           [1, 1, 1, 1, 1],
                           padding='SAME')
    biases_t1 = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation_t1 = tf.nn.bias_add(conv_t1, biases_t1)
    conv2_t1 = tf.nn.relu(pre_activation_t1, name=scope.name)
    _activation_summary(conv2_t1)

  with tf.variable_scope('conv2_t1c') as scope:
    kernel_t1c = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 1, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_t1c = tf.nn.conv3d(pool1_t1c,
                           kernel_t1c,
                           [1, 1, 1, 1, 1],
                           padding='SAME')
    biases_t1c = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation_t1c = tf.nn.bias_add(conv_t1c, biases_t1c)
    conv2_t1c = tf.nn.relu(pre_activation_t1c, name=scope.name)
    _activation_summary(conv2_t1c)

  with tf.variable_scope('conv2_t2') as scope:
    kernel_t2 = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 1, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_t2 = tf.nn.conv3d(pool1_t2,
                           kernel_t2,
                           [1, 1, 1, 1, 1],
                           padding='SAME')
    biases_t2 = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation_t2 = tf.nn.bias_add(conv_t2, biases_t2)
    conv2_t2 = tf.nn.relu(pre_activation_t2, name=scope.name)
    _activation_summary(conv2_t2)
  
  with tf.variable_scope('conv2_fl') as scope:
    kernel_fl = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 1, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_fl = tf.nn.conv3d(pool1_fl,
                           kernel_fl,
                           [1, 1, 1, 1, 1],
                           padding='SAME')
    biases_fl = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation_fl = tf.nn.bias_add(conv_fl, biases_fl)
    conv2_fl = tf.nn.relu(pre_activation_fl, name=scope.name)
    _activation_summary(conv2_fl)
  
  with tf.variable_scope('conv2_ot') as scope:
    kernel_ot = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 1, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv_ot = tf.nn.conv3d(pool1_ot,
                           kernel_ot,
                           [1, 1, 1, 1, 1],
                           padding='SAME')
    biases_ot = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
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
    reshape = tf.stack([tf.reshape(pool2_t1, [FLAGS.batch_size, -1]),
                        tf.reshape(pool2_t1c, [FLAGS.batch_size, -1]),
                        tf.reshape(pool2_t2, [FLAGS.batch_size, -1]),
                        tf.reshape(pool2_fl, [FLAGS.batch_size, -1]),
                        tf.reshape(pool2_ot, [FLAGS.batch_size, -1])])

                              
def weird_inference(mris):
  # Layers
  # h1 = tf.nn.conv3d(f_data1, f_labels)
  # h2 = tf.nn.conv3d(f_data2, f_labels)
  # h3 = tf.nn.conv3d(f_data3, f_labels)
  # h4 = tf.nn.conv3d(f_data4, f_labels)
  # w1 = tf.nn.pool3d(h1, f_labels)
  # w2 = tf.nn.pool3d(h2, f_labels)
  # w3 = tf.nn.pool3d(h3, f_labels)
  # w4 = tf.nn.pool3d(h4, f_labels)
  # hh1 = tf.nn.conv3d(w1, f_labels)
  # hh2 = tf.nn.conv3d(w2, f_labels)
  # hh3 = tf.nn.conv3d(w3, f_labels)
  # hh4 = tf.nn.conv3d(w4, f_labels)
  # ww1 = tf.nn.pool3d(hh1, f_labels)
  # ww2 = tf.nn.pool3d(hh2, f_labels)
  # ww3 = tf.nn.pool3d(hh3, f_labels)
  # ww4 = tf.nn.pool3d(hh4, f_labels)

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 3, 1, 64], stddev=5e-2, wd=0.0)
    conv = tf.nn.conv3d(f_data, kernel, [1, 1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv1)

    # pool1
    pool1 = tf.nn.max_pool3d(conv1, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
  
  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 3, 64, 64], stddev=5e-2, wd=0.0)
    conv = tf.nn.conv3d(norm1, kernel, [1, 1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    # pool2
    pool2 = tf.nn.max_pool3d(norm2, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # linear layer (WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', shape=[192, NUM_CLASSES], stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights) + biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear
 
  
