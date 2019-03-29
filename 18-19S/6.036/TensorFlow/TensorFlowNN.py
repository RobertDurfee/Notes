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
# Layers                                                                       #
################################################################################

class Layer:
    """Abstract base layer for our neural network."""

    def __init__(self):
        """Initializes layer constants necessary to construct the graph
            for training. Likely: just dimension information or nothing
            at all."""

    def build(self):
        """Adds any variables to the graph required by the layer. (Such
            as weight matrices.)"""

    def build_forward(self, X):
        """Connects the layer to the previous layer through a new 
            operation in the forward pass process.
            
        Args:
            X (Tensor): A tensor representing the inputs to the layer.
                Likely: A or Z depending on the layer.
        
        Returns:
            Tensor: A tensor representing the outputs of the layer. Likely:
                Z or A depending on the layer.
                
        """

    def build_backward(self, dL):
        """Connects the layer to the next layer through a new operation
            in the backward pass process.
            
        Args:
            dL (Tensor): A tensor representing the gradient of the loss
                of the network with respect to the outputs of the current
                layer. Likely: dLdA or dLdZ depending on the layer.
        
        Returns:
            Tensor: A tensor representign the gradient of the loss of the
                network will respect to the inputs of the current layer.
                Likely: dLdZ or dLdA depending on the layer.
        
        """

    def build_sgd_step(self, eta):
        """Updates trainable variables based off the results from the
            backward pass.
            
        Args:
            eta (float): The learning rate to use for the stochastic
                gradient descent update step.
        
        """


class Linear(Layer):
    """Simple layer fully-connecting inputs to outputs linearly."""

    def __init__(self, m, n):
        """Initializes the dimensions of the layer.
        
        Args:
            m (int): Number of input features to the layer.
            n (int): Number of output features of the layer.
        
        """
        self.m = m
        self.n = n

    def build(self):
        """Creates the trainable variables on the default graph for 
            the Linear layer."""
        with tf.variable_scope(name_or_scope=None, default_name='Linear'):
            
            self.W = tf.get_variable(name='W', shape=(self.m, self.n), initializer=tf.random_normal_initializer(0.0, tf.sqrt(1 / self.m)))
            self.W0 = tf.get_variable(name='W0', shape=(self.n, 1), initializer=tf.zeros_initializer)

    def build_forward(self, A):
        """Connects the linear layer to the previous layer by operating
            on the previous activation.
            
        Args:
            A (Tensor): An m by b tensor representing the activations from
                the previous layer with a batch of size b.
                
        Returns:
            Tensor: An n by b tensor, Z, representing the pre-activations
                as the output from this linear layer.
        
        """
        # We need this input later when computing the backward path.
        self.A = A
        
        return tf.transpose(self.W) @ self.A + self.W0

    def build_backward(self, dLdZ):
        """Connects the next layer to the current layer using backward 
            process. Also records gradients of the loss with respect to
            weights for later stochastic gradient descent updates.
        
        Args:
            dLdZ (Tensor): An n by b tensor representing the gradient of
                the loss with respect to the current layer's 
                pre-activations for a batch of size b.
                
        Returns:
            Tensor: An m by b tensor, dLdA, representing the gradient of 
                the loss with respect to the previous layer's activations.
        
        """
        # We store these gradients for use later in the sgd_step
        self.dLdW = self.A @ tf.transpose(dLdZ)
        self.dLdW0 = tf.reduce_sum(dLdZ, axis=1, keepdims=True)

        return self.W @ dLdZ

    def build_sgd_step(self, eta):
        """Constructs the training update operations for the Linear layer
            weight parameters.
            
        Args:
            eta (float): The learning rate to use for the stochastic
                gradient descent update step.
        
        Returns:
            Operation: An operation that executes the stochastic gradient
                descent step for all weights.
        
        """
        return tf.group(self.W.assign_sub(eta * self.dLdW),
                        self.W0.assign_sub(eta * self.dLdW0))


class ReLU(Layer):
    """Applies relu activation function to all inputs."""

    def build_forward(self, Z):
        """Connects the previous later to the current layer using the
            ReLU operation forward pass.
            
        Args:
            Z (Tensor): An m by b tensor representing the pre-activations
                from the previous layer for a batch of size b.
        
        Returns:
            Tensor: An n by b tensor, A, representing the activations from
                the current layer for a batch of size b. (Note: n and m 
                are equal.)
                
        """
        # We need this activation when computing the backward step later
        self.A = tf.maximum(0.0, Z)
        
        return self.A

    def build_backward(self, dLdA):
        """Connects the current layer to the next using the backward pass 
            process.
            
        Args:
            dLdA (Tensor): An n by b tensor representing the gradient of
                the loss with respect to the current layer's activations
                for a batch of size b.
        
        Returns:
            Tensor: An m by b tensor, dLdZ, representing the gradient of
                the loss with respect to the previous layer's activations
                for a batch of size b. (Note: n and m are equal.)
        
        """
        return tf.sign(self.A) * dLdA
    

class Tanh(Layer):
    """Applies hyperbolic tangent activation function to all inputs."""

    def build_forward(self, Z):
        """Connects the previous layer to the current layer using the
            backward pass process.
        
        Args:
            Z (Tensor): An m by b tensor representing the previous layer's
                pre-activation for a batch of size b.
        
        Returns:
            Tensor: An n by b tensor, A, representing the layer's 
                activation for a batch of size b. (Note: m and n are
                equal.)
                
        """
        # We need this activation when computing the backward step later
        self.A = tf.tanh(Z)
        
        return self.A
    
    def build_backward(self, dLdA):
        """Connects the next layer to the current layer using the
            backward pass process.
            
        Args:
            dLdA (Tensor): An n by b tensor representing the gradient of
                the loss with respect to this layer's activation for a 
                batch of size b.
                
        Returns:
            Tensor: An m by b tensor, dLdZ, representing the gradient of 
                the loss with respect to the previous layer's 
                pre-activation for a batch of size b. (Note: n and m are 
                equal.)
                
        """
        return (1.0 - self.A ** 2.0) * dLdA


class Softmax(Layer):
    """Applies the softmax activation function to layer inputs."""

    def build_forward(self, Z):
        """Connects the previous layer to the current layer using the
            forward pass process.
            
        Args:
            Z (Tensor): An m by b tensor representing the previous layer's
                pre-activation for a batch of size b.
        
        Returns:
            Tensor: An n by b tensor, A, representing the current layer's
                activation for a batch of size b. (Note: m and n are
                equal.)
                
        """
        # We need this activation when computing the backward step later
        self.A = tf.exp(Z) / tf.reduce_sum(tf.exp(Z), axis=0, keepdims=True)
        
        return self.A
        
    def build_backward(self, dLdA):
        """Connects the next layer to the current layer using the
            backward pass process.
            
        Args:
            dLdA (Tensor): An n by b tensor representing the gradient of
                the loss with respect to the current layer's activation.
                
        Returns:
            Tensor: An m by b tensor, dLdZ, representing the gradient of 
            the loss with respect to the previous layer's pre-activation.
            (Note: n and m are equal.)
            
        """
        n = dLdA.shape[0]
        
        # This is just a way to compute dLdZ by using the provided dLdA
        # and softmax's dAdZ tensor. Or you can assume dLdZ is passed in.
        return tf.einsum('ikj,kj->ij', tf.einsum('jk,jk,ji->ijk', self.A, 1.0 - self.A, tf.eye(n)) + tf.einsum('jk,ik,ji->ijk', -self.A, self.A, 1.0 - tf.eye(n)), dLdA)
    

class NLLM(Layer):
    """Computes the negative log-likelihood multi-class loss for neural
        network outputs and expected outputs."""

    def build_forward(self, A, Y):
        """Connects the neural network to the loss layer using the forward 
            pass process.
            
        Args:
            A (Tensor): An n by b tensor representing the neural network's
                outputs for a batch of size b.
            Y (Tensor): An n by b tensor representing the expected outputs
                from the neural network for a batch of size b.
        
        Returns:
            float: A scalar, L, which represents the loss of the neural
                network for a batch of size b.
        
        """
        # We will need both of these later to compute the backward pass.
        self.A = A
        self.Y = Y

        return -tf.reduce_sum(self.Y * tf.log(self.A))
    
    def build_backward(self):
        """Starts off the whole backward pass process.
        
        Returns:
            Tensor: An n by b tensor, dLdA, representing the gradient of
                the loss with respect to the neural network's outputs.
                
        """
        return -self.Y / self.A


class Accuracy(Layer):
    """Computes the accuracy of the current neural network outputs and
        expected outputs."""

    def build_forward(self, A, Y):
        """Connects the neural network to the accuracy layer using the forward
            pass process.
            
        Args:
            A (Tensor): An n by b tensor representing the neural network's
                outputs for a batch of size b.
            Y (Tensor): An n by b tensor representing the expected outputs
                from the neural network for a batch of size b.
        
        Returns:
            float: A scalar, acc, which represents the accuracy of the neural
                network for a batch of size b.
        
        """        
        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(A, axis=0), tf.argmax(Y, axis=0)), tf.float32))
    
################################################################################
# Model                                                                        #
################################################################################

class Sequential:
    """A standard neural network model with linearly stacked layers."""

    def __init__(self, layers):
        """Initializes the model by saving the layers provided. Note the
            model is not ready for training yet. Please call the build
            method.
            
        Args:
            layers (list of Layer): A list of layers in sequential order
                to construct the model from.
        
        """
        self.layers = layers
        
        # When we build layer, we need to know input and output dimensions.
        self.m = self.layers[0].m
        self.n = self.layers[0].n
        for layer in self.layers[1:]:
            self.n = getattr(layer, 'n', self.n)
        
        # This is the static graph representing our model
        self.graph = tf.Graph()
        
        # This is the runtime instance of our model
        self.sess = tf.Session(graph=self.graph)

    def build(self, eta, loss, metrics):
        """Constructs the model's graph from the provided layers with the
            specified loss and metrics.
            
        Args:
            eta (float): A scalar representing the learning rate for
                stochastic gradient descent.
            loss (Layer): A layer used to construct the objective for
                stochastic gradient descent.
            metrics (list of Layers): A list of layers to use when
                evaluating model performance.
                
        """
        # This ensures that the graph we add our variables to the graph
        # unique to the model.
        with self.graph.as_default():
        
            self.X = tf.placeholder(name='X', shape=(self.m, None), dtype=tf.float32)
            self.Y = tf.placeholder(name='Y', shape=(self.n, None), dtype=tf.float32)

            for layer in self.layers + [loss] + metrics:
                layer.build()

            self.forward = self.build_forward(self.X)
            self.loss_forward = loss.build_forward(self.forward, self.Y)
            self.metrics_forward = tf.tuple([metric.build_forward(self.forward, self.Y) for metric in metrics])

            loss_backward = loss.build_backward()
            self.build_backward(loss_backward)

            self.build_sgd_step(eta)
            
            initializer = tf.global_variables_initializer()
        
        # This initializes the variables in our graph using the current 
        # instance session
        self.sess.run(initializer)

    def build_forward(self, X):
        """Constructs the entire forward pass for the network by 
            connecting each layer together.
        
        Args:
            X (Tensor): An m by b tensor representing the placeholder 
                inputs to the neural network for a batch of size b.
                
        Returns:
            Tensor: An n by b tensor representing the final output of the 
                neural network for a batch of size b to be evaluated in a 
                graph session.
                
        """        
        self.forward = X
        for layer in self.layers:
            self.forward = layer.build_forward(self.forward)
        
        return self.forward

    def build_backward(self, dL):
        """Constructs the neural network's backward pass by connecting 
            each layer together.
            
        Args:
            dL (Tensor): An n by b tensor representing the gradient of the
                loss with respect to the output of the neural network.
                
        """        
        for layer in self.layers[::-1]:
            dL = layer.build_backward(dL)

    def build_sgd_step(self, eta):
        """Constructs the stochastic gradient descent training update step 
            for the entire neural network.
            
        Args:
            eta (float): A learning rate for stochastic gradient descent.
            
        Returns:
            Operation: An operation that executes each layer's stochastic
                gradient descent update step in parallel.
                
        """        
        sgd_steps = [layer.build_sgd_step(eta) for layer in self.layers]
        self.sgd_step = tf.group(*[sgd_step for sgd_step in sgd_steps if sgd_step is not None])
        
        return self.sgd_step

    def sgd(self, X_train, Y_train, epochs=100):
        """Performs stochastic gradient descent on the model.
        
        Args:
            X_train (ndarray): A d by n NumPy array representing n input
                training points each with d features.
            Y_train (ndarray): A c by n NumPy array representing n output
                training points each with c features.
            epochs (int): Number of iterations to run stochastic gradient
                descent.
                
        """
        _, n = X_train.shape
        
        for epoch in range(epochs):

            t = np.random.randint(n)
            
            Xt = X_train[:, t:t + 1]
            Yt = Y_train[:, t:t + 1]

            self.sess.run(self.sgd_step, feed_dict={ self.X: Xt, self.Y: Yt })
            
            if epoch % 250 == 1:
                
                metrics_eval = self.sess.run(self.metrics_forward, feed_dict={ self.X: X_train, self.Y: Y_train })
                print('Iteration =', epoch, '\tAcc =', metrics_eval[1], '\tLoss =', metrics_eval[0], flush=True)

    def predict(self, X):
        """Returns the output from the forward pass.
        
        Args:
            X (ndarray): A d by n NumPy array representing n points each
                with d features to predict outputs.
                
        Returns:
            ndarray: A c by n NumPy array representing the outputs from
                n points each with c features.
                
        """
        return self.sess.run(self.forward, feed_dict={ self.X: X })

################################################################################
# Training                                                                     #
################################################################################

X = np.array([[-0.23390341,  1.18151883, -2.46493986,  1.55322202,  1.27621763,
                2.39710997, -1.34403040, -0.46903436, -0.64673502, -1.44029872,
               -1.37537243,  1.05994811, -0.93311512,  1.02735575, -0.84138778,
               -2.22585412, -0.42591102,  1.03561105,  0.91125595, -2.26550369],
              [-0.92254932, -1.10309630, -2.41956036, -1.15509002, -1.04805327,
                0.08717325,  0.81847250, -0.75171045,  0.60664705,  0.80410947,
               -0.11600488,  1.03747218, -0.67210575,  0.99944446, -0.65559838,
               -0.40744784, -0.58367642,  1.05972780, -0.95991874, -1.41720255]])

Y = np.array([[0., 0., 1., 0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 1., 1., 1., 0., 0., 1.],
              [1., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 1., 1., 0.]])

plt.scatter(X[0,:], X[1,:], c=Y[1,:], cmap=ListedColormap(['#1f77b4', '#ff7f0e']))

model = Sequential([Linear(2, 10), ReLU(),
                    Linear(10, 10), ReLU(),
                    Linear(10, 2), Softmax()])

model.build(0.005, NLLM(), [NLLM(), Accuracy()])

model.sgd(X, Y, 100000)

# Create a grid of points to classify
xx1, xx2 = np.meshgrid(np.arange(-3, 3, 0.005), np.arange(-3, 3, 0.005))

# Flatten the grid to pass into model
grid = np.c_[xx1.ravel(), xx2.ravel()].T

# Predict classification at every point on the grid
Z = model.predict(grid)[1, :].reshape(xx1.shape)

# Plot the prediction regions.
plt.imshow(Z, interpolation='bicubic', origin='lower', extent=[-3, 3, -3, 3], 
           cmap=ListedColormap(['#1f77b4', '#ff7f0e']), alpha=0.55, aspect='auto')

# Plot the original points.
plt.scatter(X[0,:], X[1,:], c=Y[1,:], cmap=ListedColormap(['#1f77b4', '#ff7f0e']))
