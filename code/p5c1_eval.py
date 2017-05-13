from datetime import datetime
import math
import time
import pickle

import numpy as np
import tensorflow as tf

import sys

import p5c1_train as p5c1

FLAGS = tf.app.flags.FLAGS
### audi ###
tf.app.flags.DEFINE_string('eval_dir',
                           'eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_integer('num_examples', 50,
                            """Number of examples to run. Queue capacity\
                                for the string_input_producer""")
tf.app.flags.DEFINE_boolean('run_once', True,
                            """Whether to run eval only once.""")
tf.app.flags.DEFINE_integer('capacity', 50,
                            """Queue capacity for the producer""")
tf.app.flags.DEFINE_integer('num_epochs', 1,
                            """Produce # times before generating an OutOfRange error""")
tf.app.flags.DEFINE_integer('num_stats', 12,
                            """# stats in a stats list""")


def print_stat(stat, name=''):
  print('\n%s' % (datetime.now()))
  print('\t%s Accuracy @ 1 = %.5f | True count = %d | Sample count = %d' %
      (name, stat[7], stat[2], stat[1]))
  print('\t%s Recall = %.5f | True high count = %d | High count = %d' %
      (name, stat[8], stat[4], stat[3]))
  print('\t%s Specificity = %.5f | True low count = %d | Low count = %d' %
      (name, stat[9], stat[6], stat[5]))
  print('\t%s Precision = %.5f' % (name, stat[10]))
  print('\t%s F1 Score = %.5f' % (name, stat[11]))


def write_stats(stats_file_path, global_step, stat):
  print("Modifying stats file: %s" % stats_file_path)
  if int(global_step) != p5c1.FLAGS.num_train_steps_per_eval:
    with open(stats_file_path, 'rb') as stats_file:
      stats = pickle.load(stats_file)
  else:
    stats = [[], [], [], [], [], [], [], [], [], [], [], []]
  
  for i in range(FLAGS.num_stats):
    stats[i].append(stat[i])

  with open(stats_file_path, 'wb') as stats_file:
    pickle.dump(stats, stats_file)


def add_summary(summary_writer, global_step, accuracy,
    recall, specificity, precision, f1_score, name=''):
  summary = tf.Summary()
  #summary.ParseFromString(sess.run(summary_op, feed_dict={keep_prob: 1.0}))
  summary.value.add(tag='/'.join([name, 'Accuracy']), simple_value=accuracy)
  summary.value.add(tag='/'.join([name, 'Recall']), simple_value=recall)
  summary.value.add(tag='/'.join([name, 'Specificity']), simple_value=specificity)
  summary.value.add(tag='/'.join([name, 'Precision']), simple_value=precision)
  summary.value.add(tag='/'.join([name, 'F1 Score']), simple_value=f1_score)
  summary_writer.add_summary(summary, global_step)


def eval_once(saver, summary_writer, top_k_op,
              summary_op, keep_prob, labels):
  with tf.Session() as sess:
    class CntObj(object):
      pass
    cnt = CntObj

    # handle num_epochs in tf.string_input_producer
    sess.run(tf.local_variables_initializer())

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
        cnt.true_low += np.sum(result[0] * (1 - result[1]))
        step += 1

      # Compute @ 1
      accuracy = 1.0 * cnt.true / cnt.total_sample
      recall = 1.0 * cnt.true_high / cnt.high
      specificity = 1.0 * cnt.true_low / cnt.low
      precision = 1.0 * cnt.true_high / (cnt.true_high + cnt.low - cnt.true_low)
      f1_score = 2.0 * precision * recall / (precision + recall)
      add_summary(summary_writer, global_step, accuracy,
          recall, specificity, precision, f1_score, name='Set')

      """
      Description: Write to individual stats file of each set.
            There are 6 sets (0-5), hence 6 stats file in each set's directory.

      Stats file path + name: [eval_dir]/stats
      e.g. Stats file: ~/Dropbox/dl-fyp-exp/try4/eval2_a/stats

      List format:
      [
        [global_step(200), total_sample, true, high, true_high, low, true_low,
          accuracy, recall, specificity, precision, f1_score],
        [global_step(400), ...],
        ...
      ]

      List index: i = global_step / num_train_steps_per_eval

      Note: The file cannot be appended because it only contains 1 list.
            Therefore, at each global_step, the file will be overwritten
            with the appended list.
      """
      stats_file_path = evaluate.eval_stats_dir + 'stats' + evaluate.set_number_str

      stat = [global_step, cnt.total_sample, cnt.true,
        cnt.high, cnt.true_high, cnt.low, cnt.true_low,
        accuracy, recall, specificity, precision, f1_score]
      
      print_stat(stat)
      write_stats(stats_file_path, global_step, stat)

      """
      Description: Write overall stats file.

      Entries in this stats file are calculated from stats file of each set
      at the equivalent global_step
      """
      if int(evaluate.set_number_str) == 5:
        overall_stats_file_path = evaluate.eval_stats_dir + 'stats'

        current_overall_stat = [global_step, cnt.total_sample, cnt.true, \
            cnt.high, cnt.true_high, cnt.low, cnt.true_low, \
            0.0, 0.0, 0.0, 0.0, 0.0]

        for i in range(int(evaluate.set_number_str)):
          stats_file_path = evaluate.eval_stats_dir + 'stats' + str(i)
          print "\tRetrieving stats from: " + stats_file_path
          with open(stats_file_path, 'rb') as stats_file:
            stats = pickle.load(stats_file)
            for j in range(1, 7): # adding from cnt.total_sample to cnt.true_low
              current_overall_stat[j] += stats[j][-1]
        """
        Formula:
        accuracy = 1.0 * cnt.true / cnt.total_sample
        recall = 1.0 * cnt.true_high / cnt.high
        specificity = 1.0 * cnt.true_low / cnt.low
        precision = 1.0 * cnt.true_high / (cnt.true_high + cnt.low - cnt.true_low)
        f1_score = 2.0 * precision * recall / (precision + recall)
        """
        # Overall accuracy
        current_overall_stat[7] = 1.0 * current_overall_stat[2] / current_overall_stat[1]
        # Overall recall
        current_overall_stat[8] = 1.0 * current_overall_stat[4] / current_overall_stat[3]
        # Overall specificity
        current_overall_stat[9] = 1.0 * current_overall_stat[6] / current_overall_stat[5]
        # Overall precision
        current_overall_stat[10] = 1.0 * current_overall_stat[4] / \
            (current_overall_stat[4] + current_overall_stat[5] - current_overall_stat[6])
        # Overall f1 score
        current_overall_stat[11] = 2.0 * current_overall_stat[10] * \
            current_overall_stat[8] / (current_overall_stat[10] + current_overall_stat[8])
        
        print_stat(current_overall_stat, name='Overall')
        write_stats(overall_stats_file_path, global_step, current_overall_stat)
        add_summary(summary_writer, global_step,
            current_overall_stat[7], # accuracy
            current_overall_stat[8], # recall
            current_overall_stat[9], # specificity
            current_overall_stat[10], # precision
            current_overall_stat[11], # f1 score
            name='Overall')

      print "@DONE"
    except Exception as e:
      print "@EXCEPTION"
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
  return


def evaluate(is_tumor_cropped=False):
  """Eval BRATS for a number of steps."""
  with tf.Graph().as_default() as g:

    # get records and labels for BRATS
    records, labels = p5c1.inputs(is_tumor_cropped,
                                  is_train_list=False,
                                  batch_size=p5c1.FLAGS.batch_size,
                                  set_number_str=evaluate.set_number_str,
                                  num_epochs=FLAGS.num_epochs,
                                  capacity=FLAGS.num_examples)

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
    
    eval_once(saver, summary_writer, top_k_op, summary_op, keep_prob, labels)

    summary_writer.close()
    

def main(argv=None):
  
  evaluate.set_number_str = sys.argv[1]
  is_tumor_cropped = (sys.argv[2] == '1')
  evaluate.num_evals = int(sys.argv[3])

  evaluate.eval_dir = p5c1.FLAGS.common_dir
  evaluate.eval_dir += p5c1.FLAGS.tumor_dir if is_tumor_cropped else p5c1.FLAGS.brain_dir
  evaluate.eval_stats_dir = evaluate.eval_dir
  evaluate.checkpoint_dir = evaluate.eval_dir + p5c1.FLAGS.train_dir \
                          + evaluate.set_number_str
  evaluate.eval_dir += FLAGS.eval_dir + evaluate.set_number_str

  tf.gfile.MakeDirs(evaluate.eval_dir)

  evaluate(is_tumor_cropped=is_tumor_cropped)

if __name__ == '__main__':
  tf.app.run()
