# myLab01_5

import tensorflow as tf
import numpy as np

data = np.loadtxt('data.csv', delimiter=',', dtype=np.float32)

x_data = data[:, 0:-1]
y_data = data[:, [-1]]

X = tf.placeholder(tf.float32, shape=[None, 5])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([5,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# data 4/5 : training set
for step in range(int(len(data)*4/5)):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})

    if step % 2000 == 0:
        print("step", step, "\nPrediction:", hy_val[step])

print("\nAverage Cost(training set): ", np.mean(cost_val),"\n")

# data 1/5 : testing set
for step in range(int(len(data)*4/5), len(data)):
    hy_val = sess.run(hypothesis, feed_dict={X: x_data})
    if step < len(data)*4/5+5:
        print("step", step,"\nPrediction:" , hy_val[step], "\nActual: ", y_data[step])
        
