from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import sys

import p5c1_train as p5c1

FLAGS = tf.app.flags.FLAGS
### audi ###
tf.app.flags.DEFINE_string('eval_dir',
                           'eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_integer('num_examples', 200,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                            """Whether to run eval only once.""")


def eval_once(saver, summary_writer, top_k_op, summary_op,
              keep_prob, logits, labels):
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(evaluate.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      
      saver.restore(sess, ckpt.model_checkpoint_path)

      # /my-path/train1/model.ckpt-0,
      # extract global_step fro mit.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / p5c1.FLAGS.batch_size))
      true_count = 0
      total_sample_count = num_iter * p5c1.FLAGS.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        #predictions = sess.run([top_k_op], feed_dict={keep_prob: 1.0})
        result = sess.run([top_k_op, logits, labels], feed_dict={keep_prob: 1.0})
        true_count += np.sum(result[0]) # predictions
        print "RESULT"
        print result[0]
        print result[1]
        print result[2]
        step += 1

      # Compute precision @ 1
      precision = 1.0 * true_count / total_sample_count
      print('%s: precision @ 1 = %.5f | %.5f | %.5f' % (datetime.now(), precision, true_count, total_sample_count))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op, feed_dict={keep_prob: 1.0}))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
  return true_count, total_sample_count


def evaluate(is_tumor_cropped=False):
  """Eval BRATS for a number of steps."""
  with tf.Graph().as_default() as g:
    # get records and labels for BRATS
    eval_data = FLAGS.eval_data == 'test'
    records, labels = p5c1.inputs(is_tumor_cropped,
                                  is_train_list=False,
                                  batch_size=p5c1.FLAGS.batch_size,
                                  set_number=evaluate.set_number)
    keep_prob = tf.placeholder(tf.float32)
    logits = p5c1.inference(records, keep_prob)
    labels = p5c1.deb(labels, "labels")
    logits = p5c1.deb(logits, "logits")

    # calculate predictions
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    
    # restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        p5c1.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore) 
    # build the summary operation based on the TF collection of summaries
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(evaluate.eval_dir, g)
    
    sum_true_count = 0
    sum_total_sample_count = 0
    for i in xrange(5):
      true_count, total_sample_count = \
          eval_once(saver, summary_writer, top_k_op, summary_op,
                    keep_prob, logits, labels)
      sum_true_count += true_count
      sum_total_sample_count += total_sample_count
    
    average_precision = 1.0 * sum_true_count / sum_total_sample_count
    print('Average precision = %.5f | True count = %d | Sample count = %d' %
        (average_precision, sum_true_count, sum_total_sample_count))
    

def main(argv=None):
  
  evaluate.set_number = sys.argv[1]
  is_tumor_cropped = (sys.argv[2] == '1')
  model_id = sys.argv[3]

  evaluate.eval_dir = p5c1.FLAGS.common_dir
  evaluate.eval_dir += p5c1.FLAGS.tumor_dir if is_tumor_cropped else p5c1.FLAGS.brain_dir
  evaluate.checkpoint_dir = evaluate.eval_dir + p5c1.FLAGS.train_dir \
                          + evaluate.set_number + "_" + model_id
  evaluate.eval_dir += FLAGS.eval_dir + evaluate.set_number \
                      + "_" + model_id

  #if tf.gfile.Exists(evaluate.eval_dir):
  #  tf.gfile.DeleteRecursively(evaluate.eval_dir)
  tf.gfile.MakeDirs(evaluate.eval_dir)

  evaluate(is_tumor_cropped=is_tumor_cropped)

if __name__ == '__main__':
  tf.app.run()
