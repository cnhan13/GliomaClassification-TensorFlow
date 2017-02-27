import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

IMAGE_SIZE = 24

NUM_CLASSES = 10
NUM_EX_PER_EPOCH_FOR_TRAIN = 50000
NUM_EX_PER_EPOCH_FOR_EVAL = 10000

# Constants describing the training process
MOVING_AVERAGE_DECAY = 0.9999		# The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0		# Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1	# Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1			# Initial learning rate


tf.app.flags.DEFINE_string('data_dir', '~/Documents/dl/tf-notebook/prepInput/cifar-10-batches-py',"")
tf.app.flags.DEFINE_string('train_dir', '~/Documents/dl/tf-notebook/prepInput/cifar-10-batches-py', "")
tf.app.flags.DEFINE_string('eval_dir', '~/Documents/dl/tf-notebook/prepInput/cifar-10-batches-py/',"")
tf.app.flags.DEFINE_string('checkpoint_dir', '~/Documents/dl/tf-notebook/prepInput/cifar-10-batches-py/training/', "")
tf.app.flags.DEFINE_integer('batch_size', 128, "")
tf.app.flags.DEFINE_boolean('use_fp16', False, "")
tf.app.flags.DEFINE_string('eval_data', 'test',"")
tf.app.flags.DEFINE_boolean('run_once', False, "")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60*5, "")
tf.app.flags.DEFINE_integer('num_examples', 10000, "")
tf.app.flags.DEFINE_integer('max_steps', 100000, "")
tf.app.flags.DEFINE_boolean('log_device_placement', False, "")


def loadAll():
	filenames = []
	for i in range(1,6):
		file_name = FLAGS.data_dir + "/data_batch_" + str(i)
		filenames.append(file_name)
	return filenames

def read_cifar10(filename_queue):
	class CIFAR10File(object):
		pass
	result = CIFAR10File()

	# Save the dimension in this record object
	label_bytes = 1
	result.height = 32
	result.width = 32
	result.depth = 3
	image_bytes = result.height * result.width * result.depth

	record_bytes = label_bytes + image_bytes

	# no header or footer in CIFAR10 format, so leave header_bytes and footer_bytes at their default of 0
	reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
	result.key, value = reader.read(filename_queue)

	# convert from a string to a vector of uint8 that is record_bytes long
	record_bytes = tf.decode_raw(value, tf.uint8)

	# The first bytes represent the label, which converts from uint8->int32
	result.label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

	# the remaining bytes after the label represent the image, which reshapes from [depth * height * width] to [depth, height, width]
	depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]), [result.depth, result.height, result.width])

	# conver from [depth, height, width] to [height, width, depth]
	result.uint8image = tf.transpose(depth_major, [1, 2, 0])
	return result

def inputs(eval_data, data_dir, batch_size):

	if not eval_data:
		filenames = [os.path.join(data_dir, 'data_batch%d' % i) for i in xrange(1,6)]
		num_examples_per_epoch = NUM_EX_PER_EPOCH_FOR_TRAIN
	else:
		filenames = [os.path.join(data_dir, 'test_batch')]
		num_examples_per_epoch = NUM_EX_PER_EPOCH_FOR_EVAL

	# Create a queue that produces the filenames to read
	filename_queue = tf.train.string_input_producer(loadAll())

	# Read examples from files in the filename queue
	read_input = read_cifar10(filename_queue)
	reshaped_image = tf.cast(read_input.uint8image, tf.float32)

	height = IMAGE_SIZE
	width = IMAGE_SIZE

	# Image processing for evaluation
	# Crop the central [height, width] of the image
	resize_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, width, height)

	# Subtract off the mean and divide by the variance of the pixels.
	float_image = tf.image.per_image_whitening(resized_image)

	# Ensure that the random shuffling has good mixing properties
	min_fraction_of_examples_in_queue = 0.4
	min_queue_examples = int(NUM_EX_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

	# Generate a batch of images and labels by building up a queue of examples.
	return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=False)

def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
	num_preprocess_threads = 16
	if shuffle:
		images, label_batch = tf.train.shuffle_batch(
			[image, label],
			batch_size=batch_size,
			num_threads=num_preprocess_threads,
			capacity=min_queue_examples + 3*batch_size,
			min_after_dequeue=min_queue_examples)
	else:
		images, label_batch = tf.train.batch(
			[image, label],
			batch_size=batch_size,
			num_threads=num_preprocess_threads,
			capacity=min_queue_examples + 3*batch_size)

	# Display the training images in the visualizer
	tf.image_summary('images', iamges)

	return images, tf.reshape(label_batch, [batch_size])

def inputs(eval_data):
	images, labels = inputs(eval_data=eval_data, data_dir=data_dir, batch_size=FLAGS.batch_size)
	if FLAGS.use_fp16:
		images = tf.cast(images, tf.float16)
		labels = tf.cast(labels, tf.float16)
	return images, labels

def _variable_on_cpu(name, shape, initializer):
	with tf.device('/cpu:0'):
		dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
		var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
	return var

def _variable_with_weight_decay(name, shape, stddev, wd):
	dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
	var = _variable_on_cpu(
		name,
		shape,
		tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
	if wd is not None:
		weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var

def _activation_summary(x):
	tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
	tf.histogram_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def inference(images):
	""" Build the CIFAR10 model"""
	with tf.variable_scope('conv1') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[5,5,3,64], stddev=5e-2, wd=0.0)
		conv = tf.nn.conv3d(images, kernel, [1,1,1,1], padding='SAME')
		biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
		bias = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(bias, name=scope.name)
		_activation_summary(conv1)

	# pool1
	pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1],padding='SAME',name='pool1')
	# norm1
	norm1 = tf.nn.lrn(pool1, 4, bias=1., alpha=0.001 / 9.0, beta=0.75,name='norm1')
	# conv2
	with tf.variable_scope('conv2') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[5,5,64,64],stddev=5e-2,wd=0.0)
		conv = tf.nn.conv3d(norm1, kernel, [1,1,1,1],padding='SAME')
		biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
		bias = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(bias, name=scope.name)
		_activation_summary(conv2)

	# norm2
	norm2 = tf.nn.lrn(conv2, 4, bias=1., alpha=0.001 / 9.0, beta=0.75, name='norm2')
	# pool2
	pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool2')

	# local3
	with tf.variable_scope('local3') as scope:
		# move everything into depth so we can perform a single matrix multiply.
		reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
		dim = reshape.get_shape()[1].value
		weights = _variable_with_weight_decay('weights', shape=[dim,384], stddev=0.04, wd=0.004)
		biases = _variabel_on_cpu('biases', [384], tf.constant_inializer(0.1))
		local3 = tf.nn.relu(tf.matmul(reshape,weights) + biases, name=scope.name)
		_activation_summary(local3)

	# local4
	with tf.variable_scope('local4') as scope:
		weights = _variable_with_weight_decay('weights', [384,192], stddev=0.04, wd=0.004)
		biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
		local4 = tf.relu(tf.matmul(local3,weights) + biases, name=scope.name)
		_activation_summary(local4)

	# softmax, i.e. softmax(WX + b)
	with tf.variable_scope('softmax_linear') as scope:
		weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES], stddev=1/192.0, wd=0.0)
		biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
		softmax_linear = tf.add(tf.matmul(local4, weights),biases, name=scope.name)
		_activation_summary(softmax_linear)

	return softmax_linear

def eval_once(saver, summary_writer, top_k_op, summary_op):
	"""Run Eval once.

	Args:
		saver: Saver
		summary_writer: Summary writer
		top_k_op: Top K op
		summary_op: Summary op
	"""
	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			# Restores from checkpoint
			saver.restore(sess, ckpt.model_checkpoint_path)
			# Assuming model_checkpoint_path looks something like:
			# /my-favorite-path/cifar-10-batches-py/training/model.ckpt-0, extract global step from it.
			global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
		else:
			print('No checkpoint file found')
			return

		# Start the queue runners
		coord = tf.train.Coordinator()
		try:
			threads = []
			for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
				threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

			num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
			true_count = 0 # Counts the number of correct predictions
			total_sample_count = num_iter * FLAGS.batch_size
			step = 0
			while step < num_iter and not coord.should_stop():
				predictions = sess.run([top_k_op])
				true_count += np.sum(predictions)
				step += 1

			# Compute precision @ 1.
			precision = true_count / total_sample_count
			print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

			summary = tf.Summary()
			summary.ParseFromString(sess.run(summary_op))
			summary.value.add(tag='Precision @ 1', simple_value=precision)
			summary_writer.add_summary(summary, global_step)
		except Exception as e: # pylint: disable=broad-except
			coord.request_stop(e)

		coord.request_stop()
		coord.join(threads, stop_grace_period_secs=10)

def evaluate():
	"""Evaluate CIFAR10 for a number of steps """
	with tf.Graph().as_default() as g:
		# Get images and labels for CIFAR10
		eval_data = FLAGS.eval_data == 'test'
		images, labels = inputs(eval_data=eval_data)

		# Build a Graph that computes the logits predictions from the inference model.
		logits = inference(images)

		# Calculate predictions
		top_k_op = tf.nn.in_top_k(logits, labels, 1)

		# Restore the moving average version of the learned variable for eval.
		variable_averages = tf.train.ExponentialMovingAverage(
			MOVING_AVERAGE_DECAY)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)

		# Build the summary operation based on the TF collection of Summaries
		summary_op = tf.merge_all_summaries()

		summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

		while True:
			eval_once(saver, summary_writer, top_k_op, summary_op)
			if FLAGS.run_once:
				break
			time.sleep(FLAGS.eval_interval_secs)

def distorted_inputs(data_dir, batch_size):
	"""Construct distorted input for CIFAR training using the Reader ops.

	Args:
		data_dir: Path to the CIFAR-10 data directory
		batch_size: Number of images per batch

	Returns:
		images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
		labels: Labels. 1D tensor of [batch_size] size.
	"""
	filenames = [os.path.join(data_dir, 'data_batch_%d' %i) for i in xrange(1, 6)]
	for f in filenames:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)

	# Create a queue that produces the filenames to read.
	filename_queue = tf.train.string_input_producer(filenames)

	# Read examples from files in the filename queue.
	read_input = read_cifar10(filename_queue)
	reshaped_image = tf.cast(read_input.uint8image, tf.float32)

	height = IMAGE_SIZE
	width = IMAGE_SIZE

	# Image processing for training the network. Note the many random distortions applied to the image.

	# Randomly crop a [height, width] section of the image.
	distorted_iamge = tf.random_crop(reshaped_image, [height, width, 3])

	# Randomly flip the image horizontally.
	distorted_image = tf.image.random_flip_left_right(distorted_image)

	# Because these operations are not commutative, consider randomizing
	# the order their operation.
	distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
	distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

	# Subtract off the mean and divide by the variance of the pixels.
	float_image = tf.image.per_image_whitening(distorted_image)

	# Ensure that the random shuffling has good mixing properties.
	min_fraction_of_examples_in_queue = 0.4
	min_queue_examples = int(NUM_EX_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
	print ('Filling queue with %d CIFAR images before starting to train. '
		'This will take a few minutes.' % min_queue_examples)

	# Generate a batch of images and labels by building up a queue of examples.
	return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=True)

def distorted_inputs():
	"""Construct distorted input for CIFAR training using the Reader ops.

	Returns:
		images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
		labels: Labels. 1D tensor of [batch_size] size.

	Raises:
		ValueError: If no data_dir
	"""
	if not FLAGS.data_dir:
		raise ValueError('Please supply a data_dir')
	images, labels = distorted_inputs(data_dir=data_dir, batch_size=FLAGS.batch_size)
	if FLAGS.use_fp16:
		images = tf.cast(images, tf.float16)
		labels = tf.cast(labels, tf.float16)
	return images, labels

def loss(logits, labels):
	""" Add L2Loss to all the trainable variables.

	ADd summary for "Loss" and "Loss/avg".
	Args:
		logits: Logits from inference().
		labels: Labels from distorted_inputs or inputs(). 1-D tensor of shape [batch_size]

	Returns:
		Loss tensor of type float.
	"""
	# Calculate the average cross entropy loss across the batch.
	labels = tf.cast(labels, tf.int64)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logts, labels, name='cross_entropy_per_examle')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses', cross_entropy_mean)

	# The total loss is defined as the cross entropy loss plus all of the weight
	# decay terms (L2 loss).
	return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
	"""Add summaries for losses in CIFAR-10 model.

	Generates moving average for all losses and associated summaries for visualizing the performance of the network.

	Args:
		total_loss: Total loss from loss().
	Returns:
		loss_averages_op: op for generating moving averages of losses.
	"""
	# Compute the moving average of all individual losses and the total loss.
	loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
	losses = tf.get_collection('losses')
	loss_averages_op = loss_averages.apply(losses + [total_loss])

	# Attach a scalar summary to all individual losses and the total loss; do the same for the averaged version of the losses.
	for l in losses + [total_loss]:
		# Name each loss as '(raw)' and name the moving average version of the loss as the original loss name.
		tf.scalar_summary(l.op.name +' (raw)', l)
		tf.scalar_summary(l.op.name, loss_averages.average(l))

	return loss_averages_op


def train(total_loss, global_step):
	"""Train CIFAR-10 model.

	Create an optimizer and apply to all trainable variables. Add moving average for all trainable variables.

	Args:
		total_loss: Total loss from loss().
		global_step: Integer Variable counting the number of training steps processed.
	Returns:
		train_op: op for training.
	"""
	# Variables that affect learning rate.
	num_batches_per_epoch = NUM_EX_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
	decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

	# Decay the learning rate exponentially based on the number of steps.
	lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)
	tf.scalar_summary('learning_rate', lr)

	# Generate moving averages of all losses and associated summaries.
	loss_averages_op = _add_loss_summaries(total_loss)

	# Compute the gradients.
	with tf.control_dependencies([loss_averages_op]):
		opt = tf.train.GradientDescentOptimizer(lr)
		grads = opt.compute_gradients(total_loss)

	# Apply gradients.
	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

	# Add historgrams for trainable variables.
	for var in tf.trainable_variables():
		tf.histogram_summary(var.op.name, var)

	# Add histograms for gradients
	for grad, var in grads:
		if grad is not None:
			tf.histogram_summary(var.op.name + '/gradients', grad)

	# Track the moving averages of all trainable variables.
	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	variables_average_op = variable_averages.apply(tf.trainable_variables())

	with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
		train_op = tf.no_op(name='train')

	return train_op


def train():
	"""Train CIFAR-10 for a number of steps."""
	with tf.Graph().as_default():
		global_step = tf.Variabel(0, trainable=False)

		# Get images and labels for CIFAr-10
		images, labels = distorted_inputs()

		# Build a Graph that computes the logits predictions from the inference model.
		logits = inference(images)

		# Calculate loss.
		loss = loss(logits, labels)

		# Build a Graph that trains the model with one batch of examples and updates the model parameters
		train_op = train(loss, global_step)

		# Create a saver.
		saver = tf.train.Saver(tf.all_variables())

		# Build the summary operation based on the TF collection  of Summaries.
		summary_op = tf.merge_all_summaries()

		# Build an initialization operation to run below.
		init = tf.initialize_all_variables()

		# Start running operations on the Graph.
		sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
		sess.run(init)

		# Start the queue runners.
		tf.train.start_queue_runners(sess=sess)

		summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

		for step in xrange(FLAGS.max_steps):
			start_time = time.time()
			_, loss_value = sess.run([train_op, loss])
			duration = time.time() - start_time

			assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

			if step % 10 == 0:
				num_examples_per_step = FLAGS.batch_size
				examples_per_sec = num_examples_per_step / duration
				sec_per_batch = float(duration)

				format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
					'sec/batch)')
				print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

			if step % 100 == 0:
				summary_str = sess.run(summary_op)
				summary_writer.add_summary(summary_str, step)

				# Save the model checkpoint periodically.
				if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
					checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
					saver.save(sess, checkpoint_path, global_step=step)



def main(argv=None): # pylint: disable=unused-argument
	evaluate()

if __name__ == '__main__':
	tf.app.run()
# evaluate()

# def batch2tf(batch):
# 	out = np.reshape(batch, (len(batch),-1,3), order='F')
# 	out = np.reshape(out, (len(out),32,32,3))
# 	return out

# # file_name = "data_batch_" + str(1)
# # print file_name
# # data = unpickle(file_name)
# # print data.keys()
# # # data.keys is ['data', 'labels', 'batch_label', 'filenames']
# # batch1 = batch2tf(data['data'])

# # print batch1.shape

# sess = tf.InteractiveSession()

# # Load file into dictionary
# def unpickle(file):
# 	import cPickle
# 	fo = open(file, 'rb')
# 	data_dict = cPickle.load(fo)
# 	fo.close()
# 	return data_dict

# # Parse dictionary data to images and labels
# def input(d, L):
# 	# Parse images
# 	print "Came here"
# 	x = np.reshape(d, (len(d),-1,3), order='F')
# 	x = np.reshape(x, (len(x),32,32,3))
# 	x = tf.constant(x, name="X")
# 	# Parse labels
# 	labels = []
# 	for i in range(10):
# 		labels.insert(i, tf.equal(L,[i]))
# 	labels_return = tf.transpose(tf.pack(labels))
# 	return tf.to_float(x), tf.to_float(labels_return)

# def weight_variable(shape):
# 	initial = tf.truncated_normal(shape,stddev=0.1)
# 	return tf.Variable(initial)

# def bias(shape):
# 	initial = tf.constant(0.1,shape=shape)
# 	return tf.Variable(initial)

# def conv2d(x,W):
# 	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

# def maxpool_2x2(x):
# 	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# x = tf.placeholder(tf.float32, shape=None) # to fetch tensor of any shape
# y_ = tf.placeholder(tf.float32, shape=[None,10])

# W_conv1 = weight_variable([5,5,3,32])
# b_conv1 = bias([32])

# h_conv1 = tf.nn.relu(conv2d(x,W_conv1) + b_conv1)
# h_pool1 = maxpool_2x2(h_conv1)

# W_conv2 = weight_variable([5,5,32,64])
# b_conv2 = bias([64])

# h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) +b_conv2)
# h_pool2 = maxpool_2x2(h_conv2)

# W_fc1 = weight_variable([8 * 8 * 64, 1024])
# b_fc1 = bias([1024])

# h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# W_fc2 = weight_variable([1024, 10])
# b_fc2 = bias([10])

# y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# with tf.Session() as sess:

# 	sess.run(tf.initialize_all_variables())
# 	print "Starting Sess"

# 	fs_list = loadAll()
# 	d = np.array([])
# 	L = []
# 	for i in range(len(fs_list)):
# 		datai = unpickle(fs_list[i]);

# 		if (i==0):
# 			d = datai['data']
# 			print d.shape
# 			L = datai['labels']
# 		else:
# 			d = np.vstack((d,datai['data']))
# 			L = L+datai['labels']

# 	data, labels = input(d,L)
# 	image_batch, label_batch = tf.train.shuffle_batch(
# 		[data, labels], batch_size=50, capacity=30000, min_after_dequeue=10000, num_threads=2, enqueue_many=True)

# 	coord = tf.train.Coordinator()
# 	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# 	print "Starting main loop"
# 	for i in range(200):
# 		imgs, lbls = sess.run([image_batch, label_batch])
# 		if i%10==0:
# 			train_accuracy = accuracy.eval(feed_dict={x:imgs, y_:lbls, keep_prob: 1.0})
# 			print("step %d, training accuracy %g"%(i, train_accuracy))
# 		train_step.run(feed_dict={x:imgs, y_:lbls, keep_prob: 0.5})

# 	print "Done training"

# 	test_dict = unpickle("test_batch")
# 	print "here"
# 	test_data, test_labels = input(test_dict['data'], test_dict['labels'])
# 	test_imgs, test_lbls = sess.run([test_data, test_labels])
# 	print test_imgs, test_lbls
# 	print("test accuracy %g"%accuracy.eval(feed_dict={x:test_imgs, y_:test_lbls, keep_prob: 1.0}))

# 	writer = tf.train.SummaryWriter('./my_graph', sess.graph)
# 	writer.close()

# 	coord.request_stop()
# 	coord.join(threads)
# 	sess.close()
