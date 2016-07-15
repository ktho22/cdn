import tensorflow as tf
import numpy as np
import input_data

mnist_width = 28
n_visible = mnist_width * mnist_width
n_hidden = 10
nBatch = None

''' Data '''
mnist = input_data.read_data_sets("/mnt/hdd1/thkim/dataset/MNIST", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels


''' create node for input data '''
X = tf.placeholder("float32", [nBatch, n_visible], name='X')
H = tf.placeholder("float32", [nBatch, n_hidden], name='H')

''' create nodes for hidden variables '''
#C_max = 4 * np.sqrt(6. / (n_visible + n_hidden))
#C_init = tf.random_uniform(shape=[n_visible, n_hidden],
#                           minval=-C_max,
#                           maxval=C_max)
#C_init = tf.placeholder("float32",[n_visible,n_hidden])

C_init = tf.convert_to_tensor(trX[np.random.randint(0,len(trX),n_hidden)].T)
C = tf.Variable(C_init, name='C')

C2 = tf.expand_dims(tf.transpose(C),0)
X2 = tf.expand_dims(X,1)
print C2.get_shape()
print X2.get_shape()

dist = tf.reduce_sum(tf.square(tf.sub(X2,C2)),2)
cost = tf.reduce_mean(tf.reduce_min(dist,1))
choice = tf.argmin(dist,1)

#train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)  # construct an optimizer
train_op = tf.train.AdadeltaOptimizer(1e-3).minimize(cost)
#train_op = tf.train.AdamOptimizer().minimize(cost)

''' Launch the graph in a session '''
sess = tf.Session()


nInputBatch = 128
tf.initialize_all_variables().run(session=sess)
print('-1', sess.run(cost, feed_dict={X: teX[:nInputBatch]}))

for i in range(100):
    cnt = 0
    for start, end in zip(range(0, len(trX), nInputBatch), range(nInputBatch, len(trX), nInputBatch)):
        cnt += 1
        input_ = trX[start:end]
        _,trainCost = sess.run([train_op,cost], feed_dict={X: input_})
        #if cnt % 100 == 0: print "Iter", cnt, trainCost/nInputBatch

    out =  sess.run([cost,dist,choice], feed_dict={X: teX})
    print 'Test iter %d %5.3f'%(i, out[0])
    print out[2]
