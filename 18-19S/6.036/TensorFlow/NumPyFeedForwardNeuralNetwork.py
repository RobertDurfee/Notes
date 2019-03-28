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

    def forward(self, X):
        """Executes the forward pass through the layer
            
        Args:
            X (ndarray): A matrix representing the inputs to the layer.
                Likely: A or Z depending on the layer.
        
        Returns:
            ndarray: A tensor representing the outputs of the layer. 
                Likely: Z or A depending on the layer.
                
        """

    def backward(self, dL):
        """Computes the backward pass for the layer.
            
        Args:
            dL (ndarray): A matrix representing the gradient of the loss
                of the network with respect to the outputs of the current
                layer. Likely: dLdA or dLdZ depending on the layer.
        
        Returns:
            ndarray: A matrix representing the gradient of the loss of the
                network will respect to the inputs of the current layer.
                Likely: dLdZ or dLdA depending on the layer.
        
        """

    def sgd_step(self, eta):
        """Updates trainable variables based off the results from the
            backward pass.
            
        Args:
            eta (float): The learning rate to use for the stochastic
                gradient descent update step.
                
        """


class Linear(Layer):
    """Simple layer fully-connecting inputs to outputs linearly."""

    def __init__(self, m, n):
        """Initializes the layer based on input and output dimensions. 

        Note: Kernel is initialized using normal distribution with mean 0 
        and variance 1 / m. All biases are initialized to zero.

        Args:
            m (int): Number of input features to the layer.
            n (int): Number of output features of the layer.

        """
        self.m = m
        self.n = n

        self.W0 = np.zeros((self.n, 1))
        self.W = np.random.normal(0, np.sqrt(1 / self.m), (self.m, self.n))

    def forward(self, A):
        """Computes the forward pass of the layer.
            
        Args:
            A (ndarray): An m by b matrix representing the activations from
                the previous layer with a batch of size b.
                
        Returns:
            ndarray: An n by b matrix, Z, representing the pre-activations
                as the output from this linear layer.
        
        """
        # We need this input later when computing the backward path.
        self.A = A

        return np.transpose(self.W) @ self.A + self.W0

    def backward(self, dLdZ):
        """Computes the backward pass for the layer. Also records gradients
            of the loss with respect to weights for later stochastic 
            gradient descent updates.
        
        Args:
            dLdZ (ndarray): An n by b matrix representing the gradient of
                the loss with respect to the current layer's 
                pre-activations for a batch of size b.
                
        Returns:
            ndarray: An m by b matrix, dLdA, representing the gradient of 
                the loss with respect to the previous layer's activations.
        
        """
        # We store these gradients for use later in the sgd_step
        self.dLdW = self.A @ np.transpose(dLdZ)
        self.dLdW0 = np.sum(dLdZ, axis=1, keepdims=True)

        return self.W @ dLdZ

    def sgd_step(self, eta):
        """Updates the layer's variables using stochastic gradient
            descent.
            
        Args:
            eta (float): The learning rate to use for the stochastic
                gradient descent update step.
        
        """
        self.W -= eta * self.dLdW
        self.W0 -= eta * self.dLdW0
        

class ReLU(Layer):
    """Applies relu activation function to all inputs."""
       
    def forward(self, Z):
        """Compute the forward pass output for the layer. 
            
        Args:
            Z (ndarray): An m by b matrix representing the pre-activations
                from the previous layer for a batch of size b.
        
        Returns:
            ndarray: An n by b matrix, A, representing the activations from
                the current layer for a batch of size b. (Note: n and m 
                are equal.)
                
        """
        # We need this activation when computing the backward step later
        self.A = np.maximum(0.0, Z)
        
        return self.A

    def backward(self, dLdA):
        """Computes the backward pass for the layer.
        
        Args:
            dLdA (ndarray): An n by b matrix representing the gradient of
                the loss with respect to the current layer's activations
                for a batch of size b.
        
        Returns:
            ndarray: An m by b matrix, dLdZ, representing the gradient of
                the loss with respect to the previous layer's activations
                for a batch of size b. (Note: n and m are equal.)
        
        """
        return np.sign(self.A) * dLdA

    
class Tanh(Layer):
    """Applies hyperbolic tangent activation function to all inputs."""

    def forward(self, Z):
        """Computes the forward pass activation for the layer.
        
        Args:
            Z (ndarray): An m by b matrix representing the previous layer's
                pre-activation for a batch of size b.
        
        Returns:
            ndarray: An n by b matrix, A, representing the layer's 
                activation for a batch of size b. (Note: m and n are
                equal.)
                
        """
        # We need this activation when computing the backward step later
        self.A = np.tanh(Z)

        return self.A

    def backward(self, dLdA):
        """Computes the backward pass for the layer.
        
        Args:
            dLdA (ndarray): An n by b matrix representing the gradient of
                the loss with respect to this layer's activation for a 
                batch of size b.
                
        Returns:
            ndarray: An m by b matrix, dLdZ, representing the gradient of 
                the loss with respect to the previous layer's 
                pre-activation for a batch of size b. (Note: n and m are 
                equal.)
                
        """
        return (1.0 - self.A ** 2.0) * dLdA

class SoftMax(Layer):
    """Applies the softmax activation function to layer inputs."""

    def forward(self, Z):
        """Computes the forward pass activations for the layer.
            
        Args:
            Z (ndarray): An m by b matrix representing the previous layer's
                pre-activation for a batch of size b.
        
        Returns:
            ndarray: An n by b matrix, A, representing the current layer's
                activation for a batch of size b. (Note: m and n are
                equal.)
                
        """
        # We need this activation when computing the backward step later
        self.A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
        
        return self.A

    def backward(self, dLdA):
        """Computes the backward pass for the layer.
            
        Args:
            dLdA (ndarray): An n by b matrix representing the gradient of
                the loss with respect to the current layer's activation.
                
        Returns:
            ndarray: An m by b tensor, dLdZ, representing the gradient of 
                the loss with respect to the previous layer's 
                pre-activation. (Note: n and m are equal.)
            
        """
        n = dLdA.shape[0]
        
        # This is just a way to compute dLdZ by using the provided dLdA
        # and softmax's dAdZ tensor. Or you can assume dLdZ is passed in.
        return np.einsum('ikj,kj->ij', np.einsum('jk,jk,ji->ijk', self.A, 1.0 - self.A, np.eye(n)) + np.einsum('jk,ik,ji->ijk', -self.A, self.A, 1.0 - np.eye(n)), dLdA)


class NLLM(Layer):
    """Computes the negative log-likelihood multi-class loss for neural
        network outputs and expected outputs."""

    def forward(self, A, Y):
        """Computes the loss of predictions vs expected outputs.
            
        Args:
            A (ndarray): An n by b matrix representing the neural network's
                outputs for a batch of size b.
            Y (ndarray): An n by b matrix representing the expected outputs
                from the neural network for a batch of size b.
        
        Returns:
            float: A scalar, L, which represents the loss of the neural
                network for a batch of size b.
        
        """
        # We will need both of these later to compute the backward pass.
        self.A = A
        self.Y = Y

        return -np.sum(self.Y * np.log(self.A))

    def backward(self):
        """Computes the backward step for the loss.
        
        Returns:
            Tensor: An n by b tensor, dLdA, representing the gradient of
                the loss with respect to the neural network's outputs.
                
        """
        return -self.Y / self.A

################################################################################
# Model                                                                        #
################################################################################

class Sequential:
    """A standard neural network model with linearly stacked layers."""

    def __init__(self, layers, loss):
        """Initialize the layers and the loss for the network.
        
        Args:
            layers (list of Layer): A list of layers in sequential order
                to construct the model from.
            loss (Layer): A layer used to construct the objective for
                stochastic gradient descent.
        
        """
        self.layers = layers
        self.loss = loss

    def forward(self, X):
        """Predicts the output for a training input batch.
        
        Args:
            X (ndarray): A d by b matrix of points to predict with 
                dimension d and batch size b.
        
        Returns:
            ndarray: A c by b matrix representing the predicted outputs 
                with c features of the neural network for a batch size b.
        
        """
        for layer in self.layers:
            X = layer.forward(X)
            
        return X

    def backward(self, dL):
        """Computes the gradients of the loss with respect to each weight
        in the neural network to prepare for stochastic gradient descent.
        
        Args:
            dL (ndarray): An n by b tensor representing the gradient of the
                loss with respect to the output of the neural network.
        
        """
        for layer in self.layers[::-1]:
            dL = layer.backward(dL)

    def sgd_step(self, eta):
        """Runs a single update step on the weight matrices throughout the
        neural network using stochastic gradient descent.
        
        Args:
            eta (float): A learning rate for stochastic gradient descent.
        
        """
        for layer in self.layers:
            layer.sgd_step(eta)

    def sgd(self, X_train, Y_train, epochs, eta):
        """Trains the neural network by running stochastic gradient descent.
        
        Args:
            X_train (ndarray): A d by n NumPy array representing n input
                training points each with d features.
            Y_train (ndarray): A c by n NumPy array representing n output
                training points each with c features.
            epochs (int): Number of iterations to run stochastic gradient
                descent.
            eta (float): A learning rate for stochastic gradient descent.
        
        """
        _, n = X.shape
        
        for epoch in range(epochs):
            
            t = np.random.randint(n)
            
            Xt = X[:, t:t + 1]
            Yt = Y[:, t:t + 1]
            
            loss = self.loss.forward(self.forward(Xt), Yt)
            self.backward(self.loss.backward())            
            self.sgd_step(eta)
            
            if epoch % 250 == 1:
                
                acc = np.mean(np.argmax(self.forward(X_train), axis=0) == np.argmax(Y_train, axis=0))
                print('Iteration =', epoch, '\tAcc =', acc, '\tLoss =', loss, flush=True)

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
                    Linear(10, 2), SoftMax()], NLLM())

model.sgd(X, Y, 100000, 0.005)

# Create a grid of points to classify
xx1, xx2 = np.meshgrid(np.arange(-3, 3, 0.005), np.arange(-3, 3, 0.005))

# Flatten the grid to pass into model
grid = np.c_[xx1.ravel(), xx2.ravel()].T

# Predict classification at every point on the grid
Z = model.forward(grid)[1,:].reshape(xx1.shape)

# Plot the prediction regions.
plt.imshow(Z, interpolation='bicubic', origin='lower', extent=[-3, 3, -3, 3], 
           cmap=ListedColormap(['#1f77b4', '#ff7f0e']), alpha=0.55, aspect='auto')

# Plot the original points.
plt.scatter(X[0,:], X[1,:], c=Y[1,:], cmap=ListedColormap(['#1f77b4', '#ff7f0e']))
