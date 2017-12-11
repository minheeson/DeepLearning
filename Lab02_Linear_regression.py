# Lab 2 : Linear Regression

import tensorflow as tf

# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis xW+b
hypothesis = x_train * W + b

# Cost/Loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Minimize
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()

# Initializes global variables in th graph
sess.run(tf.global_variables_initializer())

# Fit the line
print("step\t cost\t\t W\t\t b")
for step in range(2001):
    sess.run(train)
    if step % 200 == 0:
        print(step,"\t", sess.run(cost),"\t", sess.run(W),"\t", sess.run(b))
         
                
