# IMPORTANT! I am using TensorFlow 2.0 Alpha release, but this guide is
# for TensorFlow 1.X so I use the backwards compatible API 
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
# IMPORTANT! If you are using TensorFlow 1.X (which is probably most
# likely), you should use this import statement instead:
# import tensorflow as tf
import numpy as np
%matplotlib inline
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt

################################################################################
# Model Construction                                                           #
################################################################################

# Input Layer Placeholders
X = tf.placeholder(tf.float32, shape=(2, None), name='X')

# Linear Layer 1 Variables
W0_1 = tf.get_variable('W0_1', shape=(10, 1), initializer=tf.zeros_initializer)
W_1 = tf.get_variable('W_1', shape=(2, 10), initializer=tf.random_normal_initializer(0.0, tf.sqrt(1 / 2)))

# Linear Layer 2 Variables
W0_2 = tf.get_variable('W0_2', shape=(10, 1), initializer=tf.zeros_initializer)
W_2 = tf.get_variable('W_2', shape=(10, 10), initializer=tf.random_normal_initializer(0.0, tf.sqrt(1 / 10)))

# Linear Layer 3 Variables
W0_3 = tf.get_variable('W0_3', shape=(2, 1), initializer=tf.zeros_initializer)
W_3 = tf.get_variable('W_3', shape=(10, 2), initializer=tf.random_normal_initializer(0.0, tf.sqrt(1 / 10)))

################################################################################
# Model Construction                                                           #
################################################################################

# Input Layer Activation
A_0 = X

# Linear Layer 1 Pre-Activation
Z_1 = tf.add(tf.matmul(tf.transpose(W_1), A_0), W0_1)

# ReLU Layer 1 Activation
A_1 = tf.maximum(tf.constant(0.0), Z_1)

# Linear Layer 2 Pre-Activation
Z_2 = tf.add(tf.matmul(tf.transpose(W_2), A_1), W0_2)

# ReLU Layer 2 Activation
A_2 = tf.maximum(tf.constant(0.0), Z_2)

# Linear Layer 3 Pre-Activation
Z_3 = tf.add(tf.matmul(tf.transpose(W_3), A_2), W0_3)

# Softmax Layer 3 Activation
A_3 = tf.divide(tf.exp(Z_3), tf.reduce_sum(tf.exp(Z_3), axis=0, keepdims=True))

################################################################################
# Prediction                                                                   #
################################################################################

# Data
X_train = np.array([[-0.23390341,  1.18151883, -2.46493986,  1.55322202,  1.27621763,
                      2.39710997, -1.34403040, -0.46903436, -0.64673502, -1.44029872,
                     -1.37537243,  1.05994811, -0.93311512,  1.02735575, -0.84138778,
                     -2.22585412, -0.42591102,  1.03561105,  0.91125595, -2.26550369],
                    [-0.92254932, -1.10309630, -2.41956036, -1.15509002, -1.04805327,
                      0.08717325,  0.81847250, -0.75171045,  0.60664705,  0.80410947,
                     -0.11600488,  1.03747218, -0.67210575,  0.99944446, -0.65559838,
                     -0.40744784, -0.58367642,  1.05972780, -0.95991874, -1.41720255]])
Y_train = np.array([[0., 0., 1., 0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 1., 1., 1., 0., 0., 1.],
                    [1., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 1., 1., 0.]])

# Plotted Data
plt.scatter(X_train[0,:], X_train[1,:], c=Y_train[1,:], cmap=ListedColormap(['#1f77b4', '#ff7f0e']))

# Execution
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())    

    A_3_eval = sess.run(A_3, feed_dict={ X: X_train })

# Predicted Output
print(A_3_eval)

################################################################################
# Metrics                                                                      #
################################################################################

# Target Placeholders
Y = tf.placeholder(tf.float32, shape=(2, None), name='Y')

# Negative Log-Likelihood Multi-Class Loss
L = tf.negative(tf.reduce_sum(tf.multiply(Y, tf.log(A_3))))

# Accuracy
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, axis=0), tf.argmax(A_3, axis=0)), tf.float32))

# Execution
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())    

    L_eval, accuracy_eval = sess.run([L, accuracy], feed_dict={ X: X_train, Y: Y_train })

# Evaluated Metrics
print(L_eval, accuracy_eval)

################################################################################
# Backward Pass                                                                #
################################################################################

# Negative Log-Likelihood Multi-Class Gradient
dLdA_3 = -Y / A_3

# Softmax Layer 3 Gradient
dLdZ_3 = tf.einsum('ikj,kj->ij', tf.add(tf.einsum('jk,jk,ji->ijk', A_3, tf.subtract(tf.constant(1.0), A_3), tf.eye(2)), tf.einsum('jk,ik,ji->ijk', tf.negative(A_3), A_3, tf.subtract(tf.constant(1.0), tf.eye(2)))), dLdA_3)

# Linear Layer 3 Weight Gradients
dLdW_3 = tf.matmul(A_2, tf.transpose(dLdZ_3))
dLdW0_3 = tf.reduce_sum(dLdZ_3, axis=1, keepdims=True)

# Linear Layer 3 Gradient
dLdA_2 = tf.matmul(W_3, dLdZ_3)

# ReLU Layer 2 Gradient
dLdZ_2 = tf.multiply(tf.sign(A_2), dLdA_2)

# Linear Layer 2 Weight Gradients
dLdW_2 = tf.matmul(A_1, tf.transpose(dLdZ_2))
dLdW0_2 = tf.reduce_sum(dLdZ_2, axis=1, keepdims=True)

# Linear Layer 2 Gradient
dLdA_1 = tf.matmul(W_2, dLdZ_2)

# ReLU Layer 1 Gradient
dLdZ_1 = tf.multiply(tf.sign(A_1), dLdA_1)

# Linear Layer 1 Weight Gradients
dLdW_1 = tf.matmul(A_0, tf.transpose(dLdZ_1))
dLdW0_1 = tf.reduce_sum(dLdZ_1, axis=1, keepdims=True)

# Linear Layer 1 Gradient
dLdA_0 = tf.matmul(W_1, dLdZ_1)

################################################################################
# Parameter Update                                                             #
################################################################################

# Linear Layer 1 Weight Updates
W_1_sgd_step = W_1.assign_sub(tf.scalar_mul(0.005, dLdW_1))
W0_1_sgd_step = W0_1.assign_sub(tf.scalar_mul(0.005, dLdW0_1))

# Linear Layer 2 Weight Updates
W_2_sgd_step = W_2.assign_sub(tf.scalar_mul(0.005, dLdW_2))
W0_2_sgd_step = W0_2.assign_sub(tf.scalar_mul(0.005, dLdW0_2))

# Linear Layer 3 Weight Updates
W_3_sgd_step = W_3.assign_sub(tf.scalar_mul(0.005, dLdW_3))
W0_3_sgd_step = W0_3.assign_sub(tf.scalar_mul(0.005, dLdW0_3))

# Grouped
sgd_step = tf.group(W_3_sgd_step, W0_3_sgd_step, 
                    W_2_sgd_step, W0_2_sgd_step, 
                    W_1_sgd_step, W0_1_sgd_step)

# Execution
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
    L_eval_before = sess.run(L, feed_dict={ X: X_train, Y: Y_train })
    
    sess.run(sgd_step, feed_dict={ X: X_train, Y: Y_train })
    
    L_eval_after = sess.run(L, feed_dict={ X: X_train, Y: Y_train })

# Compare Loss
print(L_eval_before, L_eval_after)

################################################################################
# Train                                                                        #
################################################################################

# Checkpoint Saver
saver = tf.train.Saver()

# Training Loop
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
    for it in range(100000):
        
        t = np.random.randint(20)
        
        L_eval, _ = sess.run([L, sgd_step], feed_dict={ X: X_train[:, t:t+1], Y: Y_train[:, t:t+1] })
        
        if it % 250 == 1:
            accuracy_eval = sess.run(accuracy, feed_dict={ X: X_train, Y: Y_train })
            print('Iteration =', it, '\tAcc =', accuracy_eval, '\tLoss =', L_eval, flush=True)
            
    saver.save(sess, 'model.ckpt')

# View Decision Regions
with tf.Session() as sess:
    
    # Restore variables
    saver.restore(sess, 'model.ckpt')
    
    # Create a grid of points to classify
    xx1, xx2 = np.meshgrid(np.arange(-3, 3, 0.005), np.arange(-3, 3, 0.005))

    # Flatten the grid to pass into model
    grid = np.c_[xx1.ravel(), xx2.ravel()].T

    # Predict classification at every point on the grid
    Z = sess.run(A_3, feed_dict={ X: grid, Y: Y_train })[1, :].reshape(xx1.shape)

    # Plot the prediction regions.
    plt.imshow(Z, interpolation='bicubic', origin='lower', extent=[-3, 3, -3, 3], 
               cmap=ListedColormap(['#1f77b4', '#ff7f0e']), alpha=0.55, aspect='auto')

    # Plot the original points.
    plt.scatter(X_train[0,:], X_train[1,:], c=Y_train[1,:], cmap=ListedColormap(['#1f77b4', '#ff7f0e']))
