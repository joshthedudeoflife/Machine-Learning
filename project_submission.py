import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split

print("-- Read in CSV file")
data = pd.read_csv('/Users/joshua.chou/Documents/Project_output.csv')
#scanning host = 0
#scaning host c&C =1
#scannin malicious host =2
#Spamming;Malware IP =5
#Spamming Host =6
#Spamming;Malware Domain =4
#Spamming = 3
#Scanning  Malware IP = 7
# Scanning Malware Domain =8
#Scanning Malisiouv Host = 9...#this goes on for another 23 different possible outputs

# Converts the labels as strings 
classes, yi = np.unique(data["Type"], return_inverse=True)

# Build the data matrix
print("-- Build data matrix and labels")
data = np.column_stack((data["ID"], data["Reliability"], data["Risk"])).astype(np.float32)
print data
yi = yi.astype(np.float32)
print yi


#split this up into train and test
random_state = 42
Xtrain, Xtest, ytrain, ytest = train_test_split(data, yi, test_size=0.2, random_state=random_state)

#define relevant variables
n_features = data.shape[1] 
n_classes = len(classes)
print n_classes
n1 = int((n_features + n_classes) / 2)

#[7]
#need to convert the training and test outputs into binary data one hot encoding
def one_hot_encoding(ytrain, num_classes):
    N = ytrain.shape[0]
    # create an array of zeroes
    y = np.zeros((N, num_classes), dtype=np.float32) # Create an array of size (N x num_classes)    
    # provide a set of row and column indices
    y[np.arange(N, dtype=np.int), ytrain.astype(np.int)] = 1
    print y
    return y

##define one-hot encoding versions of training and test for Tensorflow
print("-- Perform one-hot encoding")
ytrain_enc = one_hot_encoding(ytrain, n_classes)
ytest_enc = one_hot_encoding(ytest, n_classes)

# define the placeholders buckets that want to receive data, will update weights to minize loss
print('-- Build the graph')
X = tf.placeholder(tf.float32, shape=(None, n_features))
y = tf.placeholder(tf.float32, shape=(None, n_classes)) 

#build the neural network
n_neurons1 = n_features
n_neurons2 = 10
n_neurons3 = 10
n_neurons4 = n_classes
W1 = tf.Variable(tf.random_normal([n_neurons1, n_neurons2]))

# Initializing the bias terms so that they're all zeroes
b1 = tf.Variable(tf.zeros([n_neurons2]))
# Computwe hidden layer
y1 = tf.matmul(X, W1)
y11 = tf.nn.tanh(tf.add(y1, b1)) 
W2 = tf.Variable(tf.random_normal([n_neurons2, n_neurons3]))
# Initializing the bias terms so that they're all zeroes
b2 = tf.Variable(tf.zeros([n_neurons3]))
# Computing the outputs of the hidden layer
y2 = tf.matmul(y11, W2)
y22 = tf.nn.tanh(tf.add(y2, b2)) # This output contains the raw scores of the prediction of each class
W3 = tf.Variable(tf.random_normal([n_neurons3, n_neurons4]))
# Initializing the bias terms so that they're all zeroes
b3 = tf.Variable(tf.zeros([n_neurons4]))
# Computing the outputs of the hidden layer
ym = tf.matmul(y22, W3)
ypred = tf.add(ym, b3)


#Build the loss function and softmax layer so we can actually do the classification
entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=ypred))

# build an optimizer so we can optimize the loss function
optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
# can also use optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
loss = optimizer.minimize(entropy)

#[7]
#create an operation that actually initializes the variables
init = tf.global_variables_initializer()

print("-- Train the network")
#begin training
with tf.Session() as sess: # Begin the session
    # initialize the variables
    sess.run(init)
    
    # define the batch size
    batch_size = 128
    
    # define the number of epochs
    num_epochs = 1
    num_training_examples = Xtrain.shape[0]
    num_testing_examples = Xtest.shape[0]
    
    #calculate the number of iterations required to complete one epoc
    num_batches_per_epoch = int(np.ceil(num_training_examples / batch_size))
    
    # Variable that displays the progress for every 50 iterations
    show_every = 50
    
    predicted_classes = [] 

    #epochs are one iteration over all of the training data
    for i in range(num_epochs):
        # For each batch.
        for j in range(num_batches_per_epoch):
            #get the jth batch
            start = j * batch_size
            # ensures dont overshoot
            end = min(num_training_examples, (j + 1) * batch_size)
            Xbatch = Xtrain[start:end] # Grabs the jth batch of training examples
            ybatch = ytrain_enc[start:end]
            
            #[7]
            #takes batck of training lables and feeds into network, see what the predicted lables are
            sess.run(loss, feed_dict={X:Xbatch, y:ybatch})
            if j % show_every == 0:
                
                # make an operation that finds the accuracy on the training set
                class_pred = tf.argmax(ypred, 1)
                class_true = tf.argmax(y, 1)
                matches = tf.equal(class_pred, class_true)
                
                # add up how many times both outputs match tf.cast converts the Boolean array into floating point and just add all the matches together
                acc = tf.reduce_sum(tf.cast(matches, tf.float32))
                
                # compute the training matches
                acc_train = []
                
                # compute train and test labels
                train_labels = []
                test_labels = []
                train_actual_labels = []
                test_actual_labels = []
                X_train = []
                X_test = []
                
                #[7]
                # for each batch of examples, calculate how many theres a match
                for b in range(0, num_training_examples, batch_size):
                    # Get the right batch
                    start = b
                    end = min(num_training_examples, b + batch_size)
                    btchx = Xtrain[start:end]
                    btchy = ytrain_enc[start:end]
                    X_train.extend(btchx)
                    
                    # Calculate how many times the inputs and outputs agree
                    pred, actual, num_times = sess.run([class_pred, class_true, acc], feed_dict={X:btchx, y:btchy})
                    train_labels.append(pred)
                    train_actual_labels.append(actual)
                    
                    # Add this count to a list
                    acc_train.append(num_times)
                #[7]
                # same for the testing
                acc_test = []
                for b in range(0, num_testing_examples, batch_size):
                    start = b
                    end = min(num_testing_examples, b + batch_size)
                    btchx = Xtest[start:end]
                    X_test.extend(btchx)
                    btchy = ytest_enc[start:end]
                    pred, actual, num_times = sess.run([class_pred, class_true, acc], feed_dict={X:btchx, y:btchy})
                    test_labels.append(pred)
                    test_actual_labels.append(actual)

                    test_labels.append(pred)
                    acc_test.append(num_times)
                
                train_labels_final = []
                train_labels_acc_final = []
                for (l,m) in zip(train_labels, train_actual_labels):
                    train_labels_final.extend(l)
                    train_labels_acc_final.extend(m)
                
                test_labels_final = []
                test_labels_acc_final = []
                for (l,m) in zip(test_labels, test_actual_labels):
                    test_labels_final.extend(l)
                    test_labels_acc_final.extend(m)
                
                # Print out the actual classes detected
                X_final = []
                X_final.extend(X_train)
                X_final.extend(X_test)
                dct = {}
                train_labels_final.extend(test_labels_final)
                train_labels_acc_final.extend(test_labels_acc_final)
                dct['predicted'] = classes[train_labels_final]
                dct['actual'] = classes[train_labels_acc_final]
                predicted_classes.append(dct)

                # calculate the final classification accuracy for each
                acc_train = np.array(acc_train).sum() / float(num_training_examples)
                acc_test = np.array(acc_test).sum() / float(num_testing_examples)
                
                # Display the progress
                print("Epoch #{:d}, Iteration #{:d} - Training Accuracy: {:.20f}, Testing Accuracy: {:.20f}\n".format(i + 1, j + 1, acc_train, acc_test))

with open('output.txt', 'w') as f:
    predicted = predicted_classes[-1]['predicted']
    actual = predicted_classes[-1]['actual']
    f.write("ID,Reliability,Risk,Predicted,Actual\n")
    for ((x, y, z), p, a) in zip(X_final, predicted, actual):
        f.write("{},{},{},{:s},{:s}\n".format(x, y, z, p, a))