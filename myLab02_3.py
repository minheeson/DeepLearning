# myLab02_3

import tensorflow as tf
import numpy as np

data = np.loadtxt('image.csv', delimiter=',', dtype=np.float32)

training_set = data[:int(len(data)*4/5), :]
test_set = data[int(len(data)*4/5):, :]

x_data = training_set[:, 0:-1]
y_data = training_set[:, [-1]]

nb_classes = 7

X = tf.placeholder(tf.float32, shape=[None, 19])
Y = tf.placeholder(tf.int32, shape=[None, 1])

Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W = tf.Variable(tf.random_normal([19, nb_classes], seed=0), name='weight')
b = tf.Variable(tf.random_normal([nb_classes], seed=0), name='bias')

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(len(data)):
    sess.run(train, feed_dict={X: x_data, Y: y_data})

test_x_data = test_set[:, 0:-1]
test_y_data = test_set[:, [-1]]

a = sess.run(accuracy, feed_dict={X: test_x_data, Y: test_y_data})
print("Accuracy : {:.2%}".format(a))
