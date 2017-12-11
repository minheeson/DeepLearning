# Lab 1 : TensorFlow Basics

import tensorflow as tf

# Computational Graph
print("Computational Graph")
# Create a constant 
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

# Start a TF session
sess = tf.Session()

# Run the op and get result 
print("sess.run(node1, node2) : " , sess.run([node1, node2]))
print("sess.run(node3) : " , sess.run(node3))


# Placeholder
print("\nusing Placeholder")
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a+b

print(sess.run(adder_node, feed_dict = {a:3, b:4.5}))
print(sess.run(adder_node, feed_dict = {a:[1,3], b:[2,4]}))


