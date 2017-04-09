################
### PREAMBLE ###
################

from __future__ import division
import tensorflow as tf
import numpy as np
import tarfile
import os

###################
### IMPORT DATA ###
###################

def csv_to_numpy_array(filePath, delimiter):
    return np.genfromtxt(filePath, delimiter=delimiter, dtype=None)

def import_data():
    if "data" not in os.listdir(os.getcwd()):
        # Untar directory of data if we haven't already
        tarObject = tarfile.open("data.tar.gz")
        tarObject.extractall()
        tarObject.close()
        print("Extracted tar to current directory")
    else:
        # we've already extracted the files
        pass
    print("loading training data")
    trainX = csv_to_numpy_array("data/trainX.csv", delimiter="\t")
    trainY = csv_to_numpy_array("data/trainY.csv", delimiter="\t")
    print("loading test data")
    testX = csv_to_numpy_array("data/testX.csv", delimiter="\t")
    testY = csv_to_numpy_array("data/testY.csv", delimiter="\t")
    return trainX,trainY,testX,testY

trainX,trainY,testX,testY = import_data()

#########################
### GLOBAL PARAMETERS ###
#########################

numFeatures = trainX.shape[1]
numLabels = trainY.shape[1]

numEpochs = 5000
learningRate = 0.01

####################
### PLACEHOLDERS ###
####################

X = tf.placeholder(tf.float32, [None, numFeatures])
yGold = tf.placeholder(tf.float32, [None, numLabels])

#################
### VARIABLES ###
#################

weights = tf.Variable(tf.zeros([numFeatures,numLabels]))
bias = tf.Variable(tf.zeros([1,numLabels]))
######################
### PREDICTION OPS ###
######################

init_OP = tf.global_variables_initializer()

apply_weights_OP = tf.matmul(X, weights, name="apply_weights")
activation_OP = tf.nn.softmax(tf.add(apply_weights_OP, bias, name="add_bias"))

#####################
### EVALUATION OP ###
#####################

cost_OP = tf.nn.l2_loss(activation_OP-yGold, name="squared_error_cost")

#######################
### OPTIMIZATION OP ###
#######################

training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP)

#####################
### RUN THE GRAPH ###
#####################

sess = tf.Session()
sess.run(init_OP)

correct_predictions_OP = tf.equal(tf.argmax(activation_OP,1),tf.argmax(yGold,1))
accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))

cost = 0
diff = 1

# Training epochs
for i in range(numEpochs):
    if i > 1 and diff < .0001:
        print("change in cost %g; convergence."%diff)
        break
    else:
        # Run training step
        step = sess.run(training_OP, feed_dict={X: trainX, yGold: trainY})
        # Report occasional stats
        if i % 10 == 0:
            train_accuracy, newCost = sess.run(
                [accuracy_OP, cost_OP], 
                feed_dict={X: trainX, yGold: trainY}
            )
            diff = abs(newCost - cost)
            cost = newCost
            print("step %d, training accuracy %g"%(i, train_accuracy))
            print("step %d, cost %g"%(i, newCost))
            print("step %d, change in cost %g"%(i, diff))
print("final accuracy on test set: %s" %str(sess.run(accuracy_OP, feed_dict={X: testX, yGold: testY})))

for j in range(104):
    print(" Pobabilities of object %d :  %s and true label is: %s" %(j,str(sess.run(activation_OP, feed_dict={X: testX[[j],:]})),str(testY[[j],[1]])))

##############################
### SAVE TRAINED VARIABLES ###
##############################

# Create Saver
saver = tf.train.Saver()
sess.close()
