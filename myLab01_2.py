# myLab01_2

import tensorflow as tf

a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)

c = tf.multiply(a, b)
d = tf.add(a,b)
e = tf.add(c,d)

sess = tf.Session()

print(sess.run(e, feed_dict={a:[1,3,5,7,9], b:[2,4,6,8,10]}))
