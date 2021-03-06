import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# data
x = tf.placeholder("float", shape=[None, 784])
y_real = tf.placeholder("float", shape=[None, 10])

def variable_W(shape):
    initial = tf.truncated_normal(shape, stddev=1.0)
    return tf.Variable(initial)

def variable_b(shape):
    initial = tf.zeros(shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],padding="SAME")

def max_pooling(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# model
x_image = tf.reshape(x, [-1, 28, 28, 1])
W_conv1 = variable_W([5, 5, 1, 32])
b_conv1 = variable_b([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # 第一个卷积层
h_pool1 = max_pooling(h_conv1) # 第一个pooling层

W_conv2 = variable_W([5, 5, 32, 64])
b_conv2 = variable_b([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # 第二个卷积层
h_pool2 = max_pooling(h_conv2) # 第二个pooling层

W_fc1 = variable_W([7*7*64, 1024])
b_fc1 = variable_b([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) # 第一个全连接层

keep_prob = tf.placeholder("float")
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob) # dropout层

W_fc2 = variable_W([1024, 10])
b_fc2 = variable_b([10])
y = tf.nn.softmax(tf.matmul(h_fc1_dropout, W_fc2) + b_fc2) # 第二个全连接层 映射到分类标签上

# train
cross_entropy = -tf.reduce_sum(y_real * tf.log(y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
num_correct = tf.equal(tf.argmax(y_real, 1), tf.argmax(y, 1))
accuracy_correct = tf.reduce_mean(tf.cast(num_correct, "float"))

se = tf.InteractiveSession()
se.run(tf.initialize_all_variables()) # 所有变量初始化

mnist = input_data.read_data_sets("data/",one_hot=True)
for i in range(1):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy_correct.eval(feed_dict= {x: batch[0], y_real:batch[1], keep_prob:1.0})
        print("step %d, train accuracy is %f"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_real:batch[1], keep_prob:1.0})
print("Train Finished")

# test
print("Test accuracy is %f"%accuracy_correct.eval(feed_dict={x: mnist.test.images, y_real: mnist.test.labels, keep_prob:1.0}))