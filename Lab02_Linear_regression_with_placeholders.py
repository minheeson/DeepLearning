# Lab 2 : Linear Regression with placeholders

import tensorflow as tf

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# Hypothesis xW+b
hypothesis = X * W + b

# Cost/Loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()

# Initializes global variables in th graph
sess.run(tf.global_variables_initializer())

# Fit the line with new training data
print("step\t cost\t\t W\t\t b")
for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict = {X:[1, 2, 3], Y:[1, 2, 3]})
    if step % 200 == 0:
        print(step,"\t", cost_val,"\t", W_val,"\t", b_val)

# Testing our model
print("\nTesting our model")
print(sess.run(hypothesis, feed_dict={X:[5]}))
print(sess.run(hypothesis, feed_dict={X:[2.5]}))
print(sess.run(hypothesis, feed_dict={X:[1.5, 3.5]}))

                
