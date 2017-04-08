from __future__ import print_function
import numpy as np
import tarfile
import os
from __future__ import division
import tensorflow as tf

def csv_to_numpy_array(filePath, delimiter):
    return np.genfromtxt(filePath, delimiter=delimiter, dtype=None)

# Import MNIST data
print("loading training data")
trainX = csv_to_numpy_array("data/trainX.csv", delimiter="\t")
trainY = csv_to_numpy_array("data/trainY.csv", delimiter="\t")
print("loading test data")
testX = csv_to_numpy_array("data/testX.csv", delimiter="\t")
testY = csv_to_numpy_array("data/testY.csv", delimiter="\t")

# Parameters
learning_rate = 0.01
training_epochs = 100000
display_step = 1

# tf Graph Input

x = tf.placeholder(tf.float32, [None, 2955]) 
y = tf.placeholder(tf.float32, [None, 2]) 

# Set model weights
W = tf.Variable(tf.zeros([2955, 2]))
b = tf.Variable(tf.zeros([2]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()
	
with tf.Session() as sess:
     sess.run(init)
     # Training cycle
     for epoch in range(training_epochs):
             avg_cost = 0.
             _, c = sess.run([optimizer, cost], feed_dict={x: trainX, y: trainY})
             if (epoch+1) % display_step == 0:
                     print("Epoch:", '%04d' % (epoch+1))
     print("Optimization Finished!")
     correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
     print("Accuracy:", accuracy.eval({x: testX, y: testY}))
     for j in range(104):
             print("instance: %d %s" %(j,str(sess.run(pred, feed_dict={x: testX[[j],:]}))))
             print(testY[[j],:])
