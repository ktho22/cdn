import gzip
import os
import sys
import time
import numpy as np
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf
import ipdb
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = './MNIST'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 0
SEED = 66478
BATCH_SIZE = 100
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100
NUM_CLUSTER = 10

BASE_ALPHA = 0.5
eps = 1e-7

NUM_SUP = 100
#NUM_SUP = 1000

def maybe_download(filename):
    if not tf.gfile.Exists(WORK_DIRECTORY):
        tf.gfile.MakeDirs(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.Size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath

#def extract_data(filename, num_images):
#    print('Extracting', filename)
#    with gzip.open(filename) as bytestream:
#        bytestream.read(16)
#        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
#        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
#        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
#        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE , 1)
#    return data
#
#def extract_labels(filename, num_images):
#    print('Extracting', filename)
#    with gzip.open(filename) as bytestream:
#        bytestream.read(8)
#        buf = bytestream.read(1 * num_images)
#        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
#    return labels

def error_rate(predictions, labels):
    return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0])
def extract_data(filename, num_images):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
    return data

def extract_labels(filename, num_images):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels
train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

train_data = extract_data(train_data_filename, 60000)
train_labels = extract_labels(train_labels_filename, 60000)
test_data = extract_data(test_data_filename, 10000)
test_labels = extract_labels(test_labels_filename, 10000)

validation_data = train_data[:VALIDATION_SIZE, ...]
validation_labels = train_labels[:VALIDATION_SIZE, ...]
trian_data = train_data[VALIDATION_SIZE:, ...]
trian_labels = train_labels[VALIDATION_SIZE:, ...]

np.random.seed(seed=SEED)
train_data_idx = np.random.permutation(60000 - VALIDATION_SIZE)
#unsup_idx = train_data_upsup_idx[:NUM_UNSUP]
train_data_sup = train_data[train_data_idx[:NUM_SUP]]
train_labels_sup = train_labels[train_data_idx[:NUM_SUP]]
train_data_unsup = train_data[train_data_idx[NUM_SUP:], ...]
train_labels_unsup = train_labels[train_data_idx[NUM_SUP:], ...]

num_epochs = NUM_EPOCHS

train_size = train_labels.shape[0]

'''vars'''
train_data_node = tf.placeholder(tf.float32, shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
train_labels_node = tf.placeholder(tf.int64, shape = (BATCH_SIZE,))

eval_data_node = tf.placeholder(tf.float32, shape = (EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
eval_labels_node = tf.placeholder(tf.int64, shape = (EVAL_BATCH_SIZE,))

conv1_weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 32], stddev = 0.1, seed = SEED), name='conv1_weights')
conv1_biases = tf.Variable(tf.zeros([32]), name='conv1_biases')

conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev = 0.1, seed = SEED), name='conv2_weights')
conv2_biases = tf.Variable(tf.constant(0.1, shape = [64]), name='conv2_biases')

fc1_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512], stddev = 0.1, seed = SEED), name='fc1_weights')
fc1_biases = tf.Variable(tf.constant(0.1, shape = [512]), name='fc1_biases')

fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS], stddev = 0.1, seed = SEED), name='fc2_weights')
fc2_biases = tf.Variable(tf.constant(0.1, shape = [NUM_LABELS]), name='fc2_biases')

C_max = 4 * np.sqrt(6. / (NUM_CLUSTER + NUM_CLUSTER))
C_init = tf.random_uniform(shape = [NUM_CLUSTER, NUM_CLUSTER], minval = -C_max, maxval = C_max)
C = tf.Variable(C_init, name = 'C')

'''model'''
conv1 = tf.nn.conv2d(train_data_node, conv1_weights, strides = [1,1,1,1], padding = 'SAME')
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1,1,1,1], padding = 'SAME')
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
pool2 = tf.nn.max_pool(relu2, ksize = [1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

pool_shape = pool2.get_shape().as_list()
reshape = tf.reshape(pool2, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
#if trian:
#    hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
FC = tf.matmul(hidden, fc2_weights) + fc2_biases

#C2 = tf.expand_dims(tf.transpose(C), 0)
#X2 = tf.expand_dims(FC, 1)
#dist = tf.reduce_sum(tf.square(tf.sub(X2, C2)),2) / NUM_CLUSTER
#avg_dist = tf.reduce_mean(dist)
#sim = tf.exp(-dist)

dist = FC
sim = tf.sigmoid(dist)
hist = tf.reduce_sum(sim,0) / np.float(BATCH_SIZE) + eps
loghist = tf.log(hist)

'''losses'''
loss_cluster = -tf.reduce_mean(tf.reduce_max(sim, 1))
#loss_cluster = tf.reduce_mean(tf.reduce_min(sim, 1))
#loss_cly = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(dist, train_labels_node))
loss_cly = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(FC, train_labels_node))

'''reg'''
alpha = tf.placeholder('float')
weight_decay = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) + tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
ent = tf.reduce_sum(tf.mul(hist, loghist))
#loss_sup = alpha * (loss_cluster + ent)+ (1-alpha) * loss_cly + 5e-4 * weight_decay
loss_sup = (1-alpha) * loss_cly + 5e-4 * weight_decay
#loss_unsup = alpha * (loss_cluster + ent) + 5e-4 * weight_decay
loss_unsup = alpha * (loss_cluster + ent)
loss_lenet5 = loss_cly + 5e-4 * weight_decay

batch = tf.Variable(0)

#learning_rate = tf.train.exponential_decay(0.01, batch * BATCH_SIZE, train_size, 0.95, staircase = True)
learning_rate = 0.1

total_vars = tf.trainable_variables()
#total_vars.remove(total_vars[8])
sup_vars = total_vars
#optimizer_sup = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(loss_sup, global_step=batch)
optimizer_sup = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss_sup, global_step=batch)
#optimizer_sup_clu = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss_sup, global_step=batch)
#optimizer_sup = tf.group(optimizer_sup_cly, optimizer_sup_clu)
#optimizer_unsup = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(loss_unsup, global_step=batch)
optimizer_unsup = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss_unsup, global_step=batch)
#optimizer_unsup = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss_unsup, global_step=batch, var_list = [total_vars[8]])
optimizer_lenet5 = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss_lenet5, global_step=batch)

train_prediction = tf.nn.softmax(FC)

start_time = time.time()
with tf.Session() as sess:
    test_prev_best = float(0)
    tf.initialize_all_variables().run()
    num_sup = NUM_SUP
    num_unsup = 60000 - NUM_SUP - VALIDATION_SIZE
    batch_size = BATCH_SIZE
    sup_offset = 0
    unsup_offset = 0
    max_iter = 100000
    alpha_ = 0
    num_te = 10000
    init_step = 1000
    #init_step = np.inf
    inc_alpha_ = 0.05
    step_alpha_ = 1000
    # unsup set = sup + unsup
    train_data_unsup = np.concatenate([train_data_sup, train_data_unsup], axis=0)
    train_labels_unsup = np.concatenate([train_labels_sup, train_labels_unsup], axis=0)
    num_unsup = 60000 - VALIDATION_SIZE
    for step in xrange(max_iter):
        if (sup_offset + batch_size) >= num_sup:
            sup_offset = 0
        if (unsup_offset + batch_size) >= num_unsup:
            unsup_offset = 0
        batch_data_sup = train_data_sup[sup_offset:sup_offset + batch_size, ...]
        batch_labels_sup = train_labels_sup[sup_offset:sup_offset + batch_size, ...]


        batch_data_unsup = train_data_unsup[unsup_offset:unsup_offset + batch_size, ...]
        #batch_labels_unsup = train_labels_sup[unsup_offset:unsup_offset + batch_size, ...]
        if step == init_step:
            max_iter = num_sup // batch_size
            offset = 0
            feature_set = np.zeros([num_sup, NUM_LABELS])
            for tr_step in xrange(max_iter):
                train_batch_data = train_data_sup[offset:offset + batch_size, ...]
                #train_batch_labels = train_labels[offset:offset + batch_size, ...]
                feed_dict = {train_data_node: train_batch_data}#, train_labels_node: train_batch_labels}
                feature_set[offset:offset+batch_size, ...] = sess.run([FC], feed_dict = feed_dict)[0]
                offset += batch_size
            num_residual = num_sup - (max_iter * batch_size)
            residual_data = np.zeros([batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
            residual_data[:num_residual, ...] = train_data_sup[offset: , ...]
            feed_dict = {train_data_node: residual_data}
            feature_set[offset:,...] = sess.run([FC], feed_dict = feed_dict)[0][:num_residual]

            init_cluster = np.zeros([NUM_CLUSTER, NUM_LABELS])
            for cidx in xrange(NUM_LABELS):
                init_cluster[cidx] =  np.mean(feature_set[train_labels_sup == cidx])
            print 'Initializie clusters'
            set_op = C.assign(init_cluster)
            sess.run(set_op)

            alpha_ = 0.05

        if (alpha_ > 0):
            alpha_ = inc_alpha_ * (step // step_alpha_)
        if (alpha_ >= 0.5):
            alpha_ = 0.5
        alpha_ =0.5
        feed_dict_sup = {train_data_node: batch_data_sup, train_labels_node: batch_labels_sup, alpha: alpha_}
        feed_dict_unsup = {train_data_node: batch_data_unsup, alpha: alpha_}

        _, l_sup, loss_cluster_sup, loss_cly_sup, weight_decay_sup, ent_sup, predictions = sess.run([optimizer_sup, loss_sup, loss_cluster, loss_cly, weight_decay, ent, train_prediction], feed_dict = feed_dict_sup)
        if(step >= init_step):
            _, l_unsup, loss_cluster_unsup, weight_decay_unsup, ent_unsup = sess.run([optimizer_unsup, loss_unsup, loss_cluster, weight_decay, ent], feed_dict = feed_dict_unsup)
        acc = np.sum(np.argmax(predictions, 1) == batch_labels_sup) / float(batch_size)


        if (step % 100) == 0:
            print 'alpha: %0.2f' %(alpha_)
            print 'iter: %06d Supervised: Total: %05.2f, ClusterLoss: %05.2f, Entropy: %05.2f, Class_loss: %05.2f, Acc: %3.2f' %(step, l_sup, loss_cluster_sup, ent_sup, loss_cly_sup, acc*100)
            #print 'iter: %06d Supervised: Total: %05.2f, Class_loss: %05.2f, Acc: %3.2f' %(step, l_sup, loss_cly_sup, acc)
            if(step > init_step):
                print '           Unsupervised: Total: %05.2f, ClusterLoss: %05.2f, Entropy: %05.2f' % (l_unsup, loss_cluster_unsup, ent_unsup)
            #print('loss_sup: ' + str(accum_sup/100) + '\n')
            #print('loss_unsup: ' + str(accum_unsup/100) + '\n')
        if (step % 500) == 0:
            max_iter = num_te // batch_size
            offset = 0
            correct = 0
            for te_step in xrange(max_iter):
                test_batch_data = test_data[offset:offset + batch_size, ...]
                test_batch_labels = test_labels[offset:offset + batch_size, ...]
                feed_dict = {train_data_node: test_batch_data}
                predictions = sess.run([train_prediction], feed_dict = feed_dict)
                correct += np.sum(np.argmax(predictions[0], 1) == test_batch_labels)
                offset += batch_size
            #ipdb.set_trace()
            num_residual = num_te - (max_iter * batch_size)
            residual_data = np.zeros([batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
            residual_data[:num_residual, ...] = test_data[offset: , ...]
            #residual_labels[:num_residual, ...] = test_batch_labels[offset + batch_size: , ...]
            residual_labels = test_labels[offset: , ...]
            feed_dict = {train_data_node: residual_data}

            predictions = sess.run([train_prediction], feed_dict = feed_dict)
            correct += np.sum(np.argmax(predictions[0][:num_residual], 1) == residual_labels)
            acc = correct / float(10000) * 100
            print 'iter: %06d, TEST_Acc: %3.2f, Prev_Best: %3.2f' %(step, acc, test_prev_best)
            if acc > test_prev_best:
                test_prev_best = acc
        sup_offset += batch_size
        unsup_offset += batch_size

