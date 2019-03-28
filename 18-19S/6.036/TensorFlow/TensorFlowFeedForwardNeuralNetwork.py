import tensorflow as tf


class Layer:

    def __init__(self):
        pass

    def build(self):
        pass
    
    def build_forward(self):
        pass

    def build_backward(self):
        pass

    def build_train_step(self):
        pass


class Linear(Layer):

    def __init__(self, m, n):

        self.m = m
        self.n = n
    
    def build(self):

        with tf.variable_scope(name_or_scope=None, default_name='Linear'):
            
            self.W = tf.get_variable(name='W', shape=(self.m, self.n), initializer=tf.random_normal_initializer(0.0, tf.sqrt(1 / self.m)))
            self.W0 = tf.get_variable(name='W0', shape=(self.n, 1), initializer=tf.zeros_initializer)

    def build_forward(self, A):

        self.A = A

        self.forward = tf.add(tf.matmul(tf.transpose(self.W), self.A), self.W0)
        return self.forward
    
    def build_backward(self, dLdZ):

        self.dLdZ = dLdZ

        self.dLdW = tf.matmul(self.A, tf.transpose(self.dLdZ))
        self.dLdW0 = tf.reduce_sum(self.dLdZ, axis=1, keepdims=True)

        self.backward = tf.matmul(self.W, self.dLdZ)
        return self.backward
    
    def build_train_step(self, eta):

        self.eta = eta

        self.train_step_W = self.W.assign(tf.subtract(self.W, tf.scalar_mul(self.eta, self.dLdW)))
        self.train_step_W0 = self.W0.assign(tf.subtract(self.W0, tf.scalar_mul(self.eta, self.dLdW0)))

        self.train_step = tf.group(self.train_step_w, self.train_step_W0)
        return self.train_step


class Tanh(Layer):

    def __init__(self, m, n):

        self.m = m
        self.n = n

    def build_forward(self, Z):

        self.Z = Z

        self.forward = tf.tanh(self.Z)
        return self.forward
    
    def build_backward(self, dLdA):

        self.dLdA = dLdA

        self.backward = tf.multiply(tf.subtract(1.0, tf.square(self.forward)), self.dLdA)
        return self.backward
    

class ReLU(Layer):

    def __init__(self, m, n):

        self.m = m
        self.n = n

    def build_forward(self, Z):

        self.Z = Z

        self.forward = tf.maximum(0.0, self.Z)
        return self.forward
    
    def build_backward(self, dLdA):

        self.dLdA = dLdA

        self.backward = tf.multiply(tf.sign(self.forward), self.dLdA)
        return self.backward
    

class Softmax(Layer):

    def __init__(self, m, n):

        self.m = m
        self.n = n

    def build_forward(self, Z):

        self.Z = Z

        self.forward = tf.divide(tf.exp(self.Z), tf.reduce_sum(tf.exp(self.Z), axis=0, keepdims=True))
        return self.forward
    
    def build_backward(self, dLdA):

        self.dLdA = dLdA

        self.backward = tf.einsum('ikj,kj->ij', tf.add(tf.einsum('jk,jk,ji->ijk', self.forward, tf.subtract(1.0, self.forward), tf.eye(self.n)), tf.einsum('jk,ik,ji->ijk', tf.negative(self.forward), self.forward, tf.subtract(1.0, tf.eye(self.n)))), self.dLdA)
        return self.backard
        
    
class NLLM(Layer):

    def build_forward(self, A, Y):

        self.A = A
        self.Y = Y

        self.forward = tf.negative(tf.reduce_sum(tf.multiply(self.Y, tf.log(self.A))))
        return self.forward
    
    def build_backward(self):

        self.backward = tf.negative(tf.divide(self.Y, self.A))
        return self.backward


class Accuracy(Layer):

    def build_forward(self, A, Y):

        self.A = A
        self.Y = Y

        self.forward = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.Y, axis=0), tf.argmax(self.A, axis=0)), tf.float32))
        return self.forward

    
class Sequential:

    def __init__(self, layers):

        self.layers = layers

    def build(self, eta, loss, metrics):

        self.X = tf.placeholder(name='X', shape=(self.layers[0].m, None), dtype=tf.float32)
        self.Y = tf.placeholder(name='Y', shape=(self.layers[-1].n, None), dtype=tf.float32)

        for layer in self.layers:
            layer.build()
        loss.build()
        for metric in metrics:
            metric.build()

        self.forward = self.build_forward(self.X)
        for layer in self.layers:
            self.forward = layer.build_forward(self.forward)
        self.loss_forward = loss.build_forward(self.forward, self.Y)
        metric_forwards = []
        for metric in metrics:
            metric_forwards.append(metric.build_forward)
        self.metric_forward = tf.tuple(metric_forwards)

        self.loss_backward = loss.build_backward()
        self.backward = self.loss_backward
        for layer in self.layers[::-1]:
            self.backward = layer.build_backward(self.backward)
        metric_backwards = []

        train_steps = []
        for layer in self.layers:
            train_steps.append(layer.build_train_step(eta))
        self.train_step = tf.group(*[train_step for train_step in train_steps if train_step is not None])

        self.initializer = tf.global_variables_initializer()
    
    def train(self, X_train, Y_train, epochs=100):

        d, n = X_train.shape

        with tf.Session() as sess:

            sess.run(self.initializer)

            for epoch in range(epochs):

                t = np.random.randint(n)

                loss_eval, _ = sess.run(self.train_step, feed_dict={ self.X: X_train[:, t:t + 1], self.Y: Y_train[:, t:t + 1] })
                

model = Sequential([Linear(2, 10), ReLU(10, 10),
                    Linear(10, 10), ReLU(10, 10),
                    Linear(10, 2), Softmax(2, 2)])
model.build(0.005, NLLM(), [NLLM(), Accuracy()])

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

model.train(X, Y, 100000)
