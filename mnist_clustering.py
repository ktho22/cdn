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
alpha_ = 0.5

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
fc3_weights = tf.Variable(
    tf.truncated_normal([n_hidden, NUM_LABELS],
    stddev=0.1,
    seed=SEED))
fc3_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

''' Data '''
mnist = input_data.read_data_sets("/mnt/hdd1/thkim/dataset/MNIST", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
teX = np.reshape(teX,(len(teX),mnist_width,mnist_width,NUM_CHANNELS))
trX = np.reshape(trX,(len(trX),mnist_width,mnist_width,NUM_CHANNELS))
teY = np.where(teY==1)[1]
trY = np.where(trY==1)[1]

test_idx = np.zeros((128),dtype=np.int64)
test_idx[:10] = np.asarray([883, 933, 1060, 871, 1128, 1237, 956, 995, 1152, 940])

''' create node for input data '''
#X = tf.placeholder("float32", [nBatch, n_visible], name='X')
X = tf.placeholder(
      "float32",
      shape=(nBatch, mnist_width, mnist_width, NUM_CHANNELS),name='X')
Y = tf.placeholder("int64", shape=(nBatch,),name='Y')

''' create nodes for hidden variables '''

C_max = 4 * np.sqrt(6. / (n_hidden + n_hidden))
C_init = tf.random_uniform(shape=[n_hidden, n_hidden],
                   minval=-C_max,maxval=C_max)


C = tf.Variable(C_init, name='C')

conv = tf.nn.conv2d(X,
                    conv1_weights,
                    strides=[1, 1, 1, 1],
                    padding='SAME')

relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
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
pool_shape = pool.get_shape().as_list()
reshape = tf.reshape(
    pool,
    [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases,name='hidden')

FC = tf.matmul(hidden, fc2_weights) + fc2_biases
#if train:
#    hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)


C2 = tf.expand_dims(tf.transpose(C),0)
X2 = tf.expand_dims(FC,1)

dist = tf.reduce_sum(tf.square(tf.sub(X2,C2)),2)
loss1 = tf.reduce_mean(tf.reduce_min(dist,1))
choice = tf.argmin(dist,1)

batch = tf.Variable(0)
learning_rate = tf.train.exponential_decay(
      1e-5,                # Base learning rate.
      batch * nBatch,  # Current index into the dataset.
      trX.shape[0],          # Decay step.
      0.95,                # Decay rate.
      staircase=True)

loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(FC, Y))
#pred = tf.nn.softmax(dist)
pred = tf.nn.softmax(FC)

regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                  tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))

alpha = tf.placeholder('float')
cost = (0.5-alpha)*loss1 + (0.5+alpha)*loss2 + 5e-4 * regularizers


train_op = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(cost,global_step=batch)
#train_op = tf.train.GradientDescentOptimizer(1e-3).minimize(cost)  # construct an optimizer
#train_op = tf.train.AdadeltaOptimizer(1e-3).minimize(cost)
#train_op = tf.train.AdamOptimizer().minimize(cost)

''' Launch the graph in a session '''
sess = tf.Session()
C_acc = np.zeros((100,10,10))
tf.initialize_all_variables().run(session=sess)

print('-1', sess.run(cost, feed_dict={X: teX[:nBatch], Y:  teY[:nBatch],alpha:alpha_}))
C_acc[0] = sess.run(C)

for i in range(1000):
    if i== 0:
        idx = np.random.randint(0,len(trX),128)
        set_op = C.assign(sess.run(FC,feed_dict={X:trX[idx]})[:10])
        print "Before assign: ", sess.run(choice,feed_dict={X:trX[test_idx]})[:10]
        
        #sess.run(set_op)
       
        print "After assign: ", sess.run(choice,feed_dict={X:trX[test_idx]})[:10]
    if i >= 20 and alpha_ >= 0.05:
        alpha_= alpha_ - 0.05
    if alpha_ < 0:
        alpha_ = 0
    cnt = 0
    for start, end in zip(range(0, len(trX), nBatch), range(nBatch, len(trX), nBatch)):
        cnt += 1
        input_ = trX[start:end]
        gt_ = trY[start:end]
        _,trainCost,l1,l2, p = sess.run([train_op,cost,loss1,loss2, pred], feed_dict={X: input_, Y: gt_, alpha:alpha_})
        if cnt % 100 == 0: 
            print "Iter %d, L %5.2f, L1 %5.2f, L2 %5.2f, P %3.2f" % (cnt, trainCost, l1, l2, np.sum(np.argmax(p, 1) == gt_)/float(nBatch))
    
    if i<100: C_acc[i] = sess.run(C)
   
    out =  sess.run([cost,loss1,loss2,dist,choice, pred], feed_dict={X: teX[:nBatch], Y: teY[:nBatch], alpha:alpha_})
    
    print '===================================Test iter %d %5.3f %5.3f %5.3f %2.1f %3.2f'\
        %(i, out[0], out[1], out[2], alpha_, np.sum(np.argmax(out[5], 1) == teY[:nBatch])/float(nBatch))
    print '===================================', sess.run(choice,feed_dict={X:trX[test_idx]})[:10]
    
