import tensorflow as tf
import os.path
import random
import pickle

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('list_dir',
                           '/home/cnhan21/dl/BRATS2015/tumor_cropped/',
                           """Path to tumor cropped input lists""")
tf.app.flags.DEFINE_string('data_dir',
                           FLAGS.list_dir + 'BRATS2015_Training/',
                           """Path to the tumor cropped BRATS *.in files""")

def gen_input_list(idx, test_high_cnt, test_low_cnt,\
    test_high_overlap, test_low_overlap):

  #[LIST-CREATE-WITH-PROB VERSION]
  #def gen_input_list(idx, train_high, train_low, test_high, test_low)

  train_list_name = FLAGS.list_dir + 'train_list' + str(idx)
  test_list_name = FLAGS.list_dir + 'test_list' + str(idx)

  if tf.gfile.Exists(train_list_name) and tf.gfile.Exists(test_list_name):
    print "Fetching existed list of files:"
    
    # open created list of 'train files'
    with open(train_list_name, 'rb') as f:
      train_list = pickle.load(f)
      print "File: {}".format(train_list_name)

    with open(test_list_name, 'rb') as f:
      test_list = pickle.load(f)
      print "File: {}".format(test_list_name)

  else:
    print "Creating new list of files to train:"

    # create new list of train files
    
    """
    [START LIST-CREATE-WITH-PROB]
    # Probability of taking a record of high grade
    # Train data info:        220 H, 54 L
    # Train quantity target:  73 H, 41 L
    # Train ratio target:     73/220 H, 41/54 H
    #prob_take_high = 1./220
    #prob_take_low = 1./54
    prob_take_high = 73./220
    prob_take_low = 41./54

    train_list = []
    test_list = []
    count_high_train = 0
    count_high_test = 0

    while (count_high_train != train_high[idx-1] or \
        len(train_list) != train_high[idx-1] + train_low[idx-1] or \
        count_high_test != test_high[idx-1] or \
        len(test_list) != test_high[idx-1] + test_low[idx-1]):

      train_list = []
      test_list = []
      count_high_train = 0
      count_high_test = 0
    
      for (_, _, filenames) in os.walk(FLAGS.data_dir):
        for filename in filenames:
          prob_take = random.random()
          if (filename[0] == 'H' and prob_take < prob_take_high) or \
              (filename[0] == 'L' and prob_take < prob_take_low):
            
            train_list.append(filename)
            if filename[0] == 'H':
              count_high_train += 1

          else:
            test_list.append(filename)
            if filename[0] == 'H':
              count_high_test += 1

        break
      #print "train_high: " + str(count_high_train) + " train_low: " + str(len(train_list) - count_high_train)
      #print "test_high: " + str(count_high_test) + " test_low: " + str(len(test_list) - count_high_test)
    [END LIST-CREATE-WITH-PROB]
    """

    # [START LIST-CREATE-WITH-PORTION]
    high_fnames = []
    low_fnames = []
    for (_, _, fnames) in os.walk(FLAGS.data_dir):
      for fname in fnames:
        if (fname[0] == 'H'):
          high_fnames.append(fname)
        else:
          low_fnames.append(fname)

    test_list = []
    train_list = []
    count_high_train = 0
    count_high_test = 0

    tmp_high = (test_high_cnt - test_high_overlap) * idx
    if idx == 5:
      tmp_high -= 1 # overlap 6 between next last and last sets

    for i in range(len(high_fnames)):
      if i >= tmp_high and i < tmp_high + test_high_cnt:
        test_list.append(high_fnames[i])
        count_high_test += 1
      else:
        train_list.append(high_fnames[i])
        count_high_train += 1
    
    tmp_low = (test_low_cnt - test_low_overlap) * idx
    for i in range(len(low_fnames)):
      if i >= tmp_low and i < tmp_low + test_low_cnt:
        test_list.append(low_fnames[i])
      else:
        train_list.append(low_fnames[i])
    
    # [END LIST-CREATE-WITH-PORTION]

    with open(train_list_name, 'wb') as f:
      pickle.dump(train_list, f)

    print "Total train files: {0} High: {1} Low: {2}"\
          .format(len(train_list),
                  count_high_train,
                  len(train_list) - count_high_train)
    print "File: {}".format(train_list_name)
    
    with open(test_list_name, 'wb') as f:
      pickle.dump(test_list, f)

    print "Total test files: {0} High: {1} Low: {2}"\
          .format(len(test_list),
                  count_high_test,
                  len(test_list) - count_high_test)
    print "File: {}".format(test_list_name)
    
  return high_fnames, low_fnames

def main(argv=None):
  # TODO: For version with probability
  #train_high = [70, 67, 71, 76, 73]
  #train_low = [43, 39, 42, 39, 43]
  #test_high = [150, 153, 149, 144, 147]
  #test_low = [11, 15, 12, 15, 11]

  # TODO:
  test_high_cnt = 41
  test_high_overlap = 5
  test_low_cnt = 9
  test_low_overlap = 0

  for i in xrange(6):
    """
    [LIST-WITH-PROB VERSION]
    high_fnames, low_fnames = gen_input_list(i, train_high, train_low, test_high, test_low)
    """
    # [LIST-CREATE-WITH-PORTION FUNCTION CALL]
    high_fnames, low_fnames = gen_input_list(i, test_high_cnt, test_low_cnt,\
        test_high_overlap, test_low_overlap)
    if i == 0:
      prev_high_fnames = high_fnames
      prev_low_fnames = low_fnames
    else:
      if cmp(prev_high_fnames, high_fnames) != 0:
        print "i: {0}, contradiction".format(i)
        prev_high_fnames = high_fnames
        prev_low_fnames = low_fnames


if __name__ == '__main__':
  tf.app.run()
