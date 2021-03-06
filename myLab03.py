# myLab03

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes], seed=0))
b = tf.Variable(tf.random_normal([nb_classes], seed=0))

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 50
batch_size = 100

train_accuracy = []
validation_accuracy = []
test_accuracy = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples / batch_size)
        avg_acc = 0

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            acc, _ = sess.run([accuracy, optimizer], feed_dict={X: batch_xs, Y:batch_ys})
            avg_acc += acc / total_batch

        train_accuracy.append(avg_acc)
        validation_accuracy.append(accuracy.eval(session=sess,
                                    feed_dict={X: mnist.validation.images, Y: mnist.validation.labels}))
        test_accuracy.append(accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
        
    plt.plot(train_accuracy, 'k--', label='train')
    plt.plot(validation_accuracy, 'r--', label='validation')
    plt.plot(test_accuracy, 'b--', label='test')
    plt.legend()
    plt.show()
