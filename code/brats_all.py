import numpy as np
import tensorflow as tf

import os.path
import pickle

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 2, """Number of images to process in a batch.""")

tf.app.flags.DEFINE_integer('set_quantity', 10, """Number of sets to run.""")

### farmer ###
tf.app.flags.DEFINE_string('list_dir',
                           '/home/ubuntu/dl/BRATS2015/',
                           """Path to 'input list' files.""")

tf.app.flags.DEFINE_string('data_dir',
                           FLAGS.list_dir + 'BRATS2015_Training/',
                           """Path to the BRATS *.in files.""")


# Global constants describing the BRATS data set
NUM_FILES_PER_ENTRY = 5
MRI_DIMS = 3
MHA_HEIGHT = 155
MHA_WIDTH = 240
MHA_DEPTH = 240
MHA_CHANNEL = 1

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

  #f_name = tf.string_split([f_name_reader], '')
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

def inputs(_list, is_train_list, label_idx, batch_size):
  ## Create a queue of filenames to read
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
  print _list

  return _list


def proceed():
  train_list = tf.placeholder(tf.string, shape=[None])
  test_list = tf.placeholder(tf.string, shape=[None]) # evaluate test data of a set

  records, labels = inputs(_list = train_list,
                            is_train_list=True,
                            label_idx=len(FLAGS.data_dir),
                            batch_size=FLAGS.batch_size)

  #batch_logits = inference(records)

  #batch_loss = loss(batch_logits, labels)

  #train_op = train(loss, global_step)
  
  # break hanging queue
  config = tf.ConfigProto()
  config.operation_timeout_in_ms = 5000
  
  sess = tf.Session(config=config)
  #sess.run(tf.global_variables_initializer())

  for set_number in xrange(1, FLAGS.set_quantity):
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    label_batch = sess.run(labels, 
                          feed_dict={train_list: get_list(FLAGS.data_dir,
                                                          set_number,
                                                          is_train=True)})
    print label_batch

    coord.request_stop()
    coord.join(threads)
    coord.clear_stop()

  sess.close()
    


def main(argv=None):
  proceed()

if __name__ == '__main__':
  tf.app.run()
