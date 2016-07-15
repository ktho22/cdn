import tensorflow as tf
import numpy as np
import input_data

WORK_DIRECTORY = '/mnt/hdd1/thkim/dataset/MNIST'
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.
mnist_width = 28
n_visible = mnist_width * mnist_width
n_hidden = 10
nBatch = 128

def model(data, train=True):
    """The Model definition."""
    
    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
        stddev=0.1,
        seed=SEED))
    conv1_biases = tf.Variable(tf.zeros([32]))
    conv2_weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 64],
        stddev=0.1,
        seed=SEED))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal(
        [mnist_width // 4 * mnist_width // 4 * 64, 512],
        stddev=0.1,
        seed=SEED))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
    fc2_weights = tf.Variable(
        tf.truncated_normal([512, NUM_LABELS],
        stddev=0.1,
        seed=SEED))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))
    
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
        hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases



''' Data '''
mnist = input_data.read_data_sets("/mnt/hdd1/thkim/dataset/MNIST", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
teX = np.reshape(teX,(len(teX),mnist_width,mnist_width,NUM_CHANNELS))
trX = np.reshape(trX,(len(trX),mnist_width,mnist_width,NUM_CHANNELS))

''' create node for input data '''
#X = tf.placeholder("float32", [nBatch, n_visible], name='X')
X = tf.placeholder(
      tf.float32,
      shape=(nBatch, mnist_width, mnist_width, NUM_CHANNELS))

''' create nodes for hidden variables '''
def c_init(type='samples'):
    if type=='samples':
        C_init = tf.convert_to_tensor(trX[np.random.randint(0,len(trX),n_hidden)].T)
    elif type=='random':
        C_max = 4 * np.sqrt(6. / (n_hidden + n_hidden))
        C_init = tf.random_uniform(shape=[n_hidden, n_hidden],
                           minval=-C_max,maxval=C_max)
    else:
        C_init = None
    return C_init

C = tf.Variable(c_init('random'), name='C')
X2 = model(X)

C2 = tf.expand_dims(tf.transpose(C),0)
X2 = tf.expand_dims(X2,1)

dist = tf.reduce_sum(tf.square(tf.sub(X2,C2)),2)
cost = tf.reduce_mean(tf.reduce_min(dist,1))
choice = tf.argmin(dist,1)

learning_rate = tf.train.exponential_decay(
      1e-5,                # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,          # Decay step.
      0.95,                # Decay rate.
      staircase=True)
  # Use simple momentum for the optimization.
train_op = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(cost)

    
#train_op = tf.train.GradientDescentOptimizer(1e-5).minimize(cost)  # construct an optimizer
#train_op = tf.train.AdadeltaOptimizer(1e-3).minimize(cost)
#train_op = tf.train.AdamOptimizer().minimize(cost)

''' Launch the graph in a session '''
sess = tf.Session()
C_acc = np.zeros((100,10,10))

tf.initialize_all_variables().run(session=sess)
print('-1', sess.run(cost, feed_dict={X: teX[:nBatch]}))
C_acc[0] = sess.run(C)


for i in range(100):
    for start, end in zip(range(0, len(trX), nBatch), range(nBatch, len(trX), nBatch)):
        input_ = trX[start:end]
        _,trainCost = sess.run([train_op,cost], feed_dict={X: input_})
        if cnt % 100 == 0: print "Iter", cnt, trainCost/nBatch
    if cnt<len(i): C_acc[cnt] = sess.run(C)
   
    out =  sess.run([cost,dist,choice], feed_dict={X: teX[:nBatch]})
    print 'Test iter %d %5.3f'%(i, out[0])
    #print out[2]
