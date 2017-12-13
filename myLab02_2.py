# myLab02_2

import tensorflow as tf
import numpy as np

data = np.loadtxt('magic.csv', delimiter=',', dtype=np.float32)

# 4/5 data for training set, 1/5 data for test set
train_set = data[:int(len(data)*4/5), :]
test_set = data[int(len(data)*4/5):, :]

x_data = train_set[:, 0:-1]
y_data = train_set[:, [-1]]

X = tf.placeholder(tf.float32, shape=[None, 10])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([10, 1], seed=0), name='weight')
b = tf.Variable(tf.random_normal([1], seed=0), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
# Hypothesis : logistic function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(len(train_set)):
    sess.run(train, feed_dict={X: x_data, Y: y_data})

test_x_data = test_set[:, 0:-1]
test_y_data = test_set[:, [-1]]

a = sess.run(accuracy, feed_dict={X: test_x_data, Y: test_y_data})
print("Accuracy : {:.2%}".format(a))
