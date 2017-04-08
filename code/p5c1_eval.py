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


def eval_once(saver, summary_writer, top_k_op,
              summary_op, keep_prob, labels):
  with tf.Session() as sess:
    class CntObj(object):
      pass
    cnt = CntObj

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
      cnt.true = 0
      cnt.total_sample = num_iter * p5c1.FLAGS.batch_size
      cnt.true_high = 0
      cnt.high = 0
      cnt.true_low = 0
      cnt.low = 0
      step = 0
      while step < num_iter and not coord.should_stop():
        #predictions = sess.run([top_k_op], feed_dict={keep_prob: 1.0})
        result = sess.run([top_k_op, labels], feed_dict={keep_prob: 1.0})
        cnt.true += np.sum(result[0]) # predictions
        high_count = np.sum(result[1])
        cnt.high += high_count
        cnt.low += p5c1.FLAGS.batch_size - high_count
        cnt.true_high += np.sum(result[0] * result[1])
        cnt.true_low += np.sum((1 - result[0]) * (1 - result[1]))
        step += 1

      # Compute precision @ 1
      precision = 1.0 * cnt.true / cnt.total_sample
      print('%s: precision @ 1 = %.5f | %.5f | %.5f' %
          (datetime.now(), precision, cnt.true, cnt.total_sample))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op, feed_dict={keep_prob: 1.0}))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
  return cnt          


def evaluate(is_tumor_cropped=False):
  """Eval BRATS for a number of steps."""
  with tf.Graph().as_default() as g:
    class SumObj(object):
      pass
    sm = SumObj()
    
    class AvgObj(object):
      pass
    avg = AvgObj()

    # get records and labels for BRATS
    eval_data = FLAGS.eval_data == 'test'
    records, labels = p5c1.inputs(is_tumor_cropped,
                                  is_train_list=False,
                                  batch_size=p5c1.FLAGS.batch_size,
                                  set_number=evaluate.set_number)
    keep_prob = tf.placeholder(tf.float32)
    logits = p5c1.inference(records, keep_prob)

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
    
    # sm: sum, cnt: count
    sm.true_count = 0
    sm.total_sample_count = 0
    sm.true_high_count = 0
    sm.high_count = 0
    sm.true_low_count = 0
    sm.low_count = 0
    for i in xrange(5):
      cnt = eval_once(saver, summary_writer, top_k_op,
                    summary_op, keep_prob, labels)
      sm.true_count += cnt.true
      sm.total_sample_count += cnt.total_sample
      sm.true_high_count += cnt.true_high
      sm.high_count += cnt.high
      sm.true_low_count += cnt.true_low
      sm.low_count += cnt.low
    
    avg.precision = 1.0 * sm.true_count / sm.total_sample_count
    avg.high_precision = 1.0 * sm.true_high_count / sm.high_count
    avg.low_precision = 1.0 * sm.true_low_count / sm.low_count
    
    print('Average precision = %.5f | True count = %d | Sample count = %d' %
        (avg.precision, sm.true_count, sm.total_sample_count))
    print('\tAverage high precision = %.5f | True high count = %d | High count = %d' %
        (avg.high_precision, sm.true_high_count, sm.high_count))
    print('\tAverage low precision = %.5f | True low count = %d | Low count = %d' %
        (avg.low_precision, sm.true_low_count, sm.low_count))
    

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

  tf.gfile.MakeDirs(evaluate.eval_dir)

  evaluate(is_tumor_cropped=is_tumor_cropped)

if __name__ == '__main__':
  tf.app.run()
