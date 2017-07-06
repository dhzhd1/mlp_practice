import numpy as np
from sklearn.utils import shuffle


def sigmoid(x):
    """
    Sigmoid (logistic) function
    :param x: array-like shape(n_sample, n_feature)
    :return: simgoid value (array like)
    """
    return 1.0 / (1 + np.exp(-x))


def dsigmoid(x):
    """
    Derivative of sigmoid function
    :param x: array-like shape(n_sample, n_feature)
    :return: derivative value (array like)
    """
    return sigmoid(x) * (1.0 - sigmoid(x))


def tanh(x):
    """
    Tanh function
    :param x: array-like shape(n_sample, n_feature)
    :return: tanh value (array like)
    """
    return (2.0 / np.exp(-2 * x) + 1) - 1


def dtanh(x):
    """
    Derivative of tanh function
    :param x: array-like shape(n_sample, n_feature)
    :return: derivative value (array like)
    """
    return 1 - tanh(x) ** tanh(x)


def softmax(X):
    """
    softmax function
    :param X:
    :return:
    """
    return (np.exp(X).T / np.sum(np.exp(X), axis=1, dtype=float)).T


def onehot(y):
    n = len(np.unique(y))
    m = y.shape[0]
    b = np.zeros((m, n))
    for i in xrange(m):
        b[i, y[i]] = 1
    return b


class MLP:
    def __init__(self, input_size, output_size, hidden_layer_size=[100], batch_size=200, activation="sigmoid",
                 output_layer='softmax', loss='cross_entropy', lr=0.01, reg_lambda=0.0001, momentum=0.9, verbose=10):
        """
        Multilayer perceptron Class
        :param input_size: int, input size (n_feature)
        :param output_size: int,  output node size (n_class)
        :param hidden_layer_size: list of int, each int represents the size of a hidden layer
        :param batch_size: int, batch_size
        :param activation: string, activation function ['sigmoid', 'tanh']
        :param output_layer: string, output layer type ['softmax']
        :param loss: string, loss type ['cross_entropy']
        :param lr: float, learning rate
        :param reg_lambda: float, lambda of regularization
        :param verbose: int, print flag
        :param momentum: float, momentum
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.momentum = momentum
        self.batch_size = batch_size
        self.n_layers = len(hidden_layer_size)  # only hidden layers
        self.activation = activation
        self.verbose = verbose

        if activation == 'sigmoid':
            self.activation_func = sigmoid
            self.activation_dfunc = dsigmoid
        elif activation == 'tanh':
            self.activation_func = tanh
            self.activation_dfunc = dtanh
        else:
            raise ValueError("Currently only support 'sigmoid' or 'tanh' activations func!")

        self.loss = loss
        if output_layer == 'softmax':
            self.output_layer = softmax
        else:
            raise ValueError('Currently only Support softmax output_layer!')

        self.nclass = output_size

        self.weights = []  # store weights
        self.bias = []  # store bias
        self.layers = []  # store forwarding activation values
        self.deltas = []  # store errors for backprop

    def get_weight_bound(self, fan_in, fan_out):
        """
        Generate bound value for random weights initialization
        :param fan_in: layer input size
        :param fan_out: layer output size
        :return: float, bound
        """
        if self.activation == 'sigmoid':
            init_bound = np.sqrt(2. / (fan_in + fan_out))
        elif self.activation == 'tanh':
            init_bound = np.sqrt(6. / (fan_in + fan_out))
        return init_bound

    def fit(self, X, y, max_epochs, shuffle_data):
        """
        fit the model given data X and label y
        :param X: array-like, shape(n_samples, n_features)
        :param y: array-like, shape(n_samples, 1)
        :param max_epochs: int, max iterations
        :param shuffle_data: bool, if shuffle the data.
        :return: MLP model object
        """
        n_samples, n_features = X.shape
        if y.shape[0] != n_samples:
            raise ValueError("Shapes of X and y don't fit!")

        # generate weights
        # Weights and bias connecting input layer and first hidden layer
        init_bound = self.get_weight_bound(n_features, self.hidden_layer_size[0])
        self.weights.append(np.random.uniform(-init_bound, init_bound, size=(n_features, self.hidden_layer_size[0])))
        self.bias.append(np.random.uniform(-init_bound, init_bound, self.hidden_layer_size[0]))

        # Weights and bias connecting hidden layers
        for i in range(1, len(self.hidden_layer_size)):
            init_bound = self.get_weight_bound(self.hidden_layer_size[i - 1], self.hidden_layer_size[i])
            self.weights.append(np.random.uniform(-init_bound, init_bound,
                                                  size=(self.hidden_layer_size[i - 1], self.hidden_layer_size[i])))
            self.bias.append(np.random.uniform(-init_bound, init_bound, self.hidden_layer_size[i]))

        # Weights and bias connecting last hidden layer and output layer
        init_bound = self.get_weight_bound(self.hidden_layer_size[-1], self.output_size)
        self.weights.append(
            np.random.uniform(-init_bound, init_bound, size=(self.hidden_layer_size[-1], self.output_size)))
        self.bias.append(np.random.uniform(-init_bound, init_bound, self.output_size))

        # pre-allocate memory for both activations and errors
        # for input layer
        self.layers.append(np.empty((self.batch_size, self.input_size)))
        # for hidden layers
        for i in range(0, self.n_layers):
            self.layers.append(np.empty((self.batch_size, self.hidden_layer_size[i])))
            self.deltas.append(np.empty((self.batch_size, self.hidden_layer_size[i])))
        # for output layer
        self.layers.append(np.empty((self.batch_size, self.output_size)))
        self.deltas.append(np.empty((self.batch_size, self.output_size)))

        # main loop
        for i in xrange(max_epochs):

            # shuffle data
            if shuffle_data:
                X = shuffle(X, n_samples=n_samples, random_state=0)
                y = shuffle(y, random_state=0)

            # iterate every batch
            for batch in xrange(0, n_samples, self.batch_size):
                # call forward function
                self.forward(X[batch:batch + self.batch_size])
                # call backward function
                self.backward(X[batch:batch + self.batch_size], y[batch:batch + self.batch_size])

            if i % self.verbose == 0:
                # Compute Loss and Training Accuracy
                loss = self.compute_loss(X, y)
                acc = self.score(X, y)
                print('Epoch {}: loss = {}, accuracy = {}'.format(i, loss, acc))

        return self

    def compute_loss(self, X, y):
        """
        Compute loss
        :param X: data, array-like, shape(n_sample, n_feature)
        :param y: label, array-like, shape(n_sample, 1)
        :return: loss value
        """
        n_samples = X.shape[0]
        probs = self.forward(X)

        # This program uses the cross-entropy method as the loss function.
        y_mat = onehot(y)
        # data_loss = np.sum(y_mat * np.log(probs))
        data_loss = np.sum(y_mat * np.log(probs) + (1 - y_mat) * np.log(1 - probs))
        # Add Normalization
        data_loss += (1 / 2) * self.reg_lambda * np.sum(self.weights[-1] ** 2, dtype=float)

        return 1. / n_samples * data_loss

    def forward(self, X):
        # input layer
        self.layers[0] = X

        # Compute hidden layers output
        for i in xrange(1, self.n_layers + 1):
            self.layers[i] = self.activation_func(np.dot(self.layers[i - 1], self.weights[i - 1]) + self.bias[i - 1])
        # Compute output layer via 'softmax'
        self.layers[-1] = self.output_layer(np.dot(self.layers[-2], self.weights[-1]) + self.bias[-1])
        return self.layers[-1]

    def backward(self, X, y):
        if self.loss == 'cross_entropy':
            self.deltas[-1] = self.layers[-1]
            # cross_entropy loss backprop
            # self.deltas[-1][range(X.shape[0]), y] -= 1
            self.deltas[-1][range(X.shape[0])] -= 1

        # update deltas
        for i in xrange(2, len(self.deltas) + 1):
            self.deltas[-i] = np.dot(np.dot(self.deltas[-i + 1], self.weights[-i + 1].T).T,
                                     self.activation_dfunc(self.layers[-i]))

        # update weights and bias
        # hidden layer
        for j in xrange(self.n_layers, 0, -1):
            # self.weights[j] -= self.lr * (np.dot
            #                               (np.dot(self.layers[j + 1].T, self.deltas[j]),
            #                                self.weights[j]) + self.reg_lambda * self.weights[j])
            self.weights[j] -= self.lr * (np.dot(self.layers[j].T, self.deltas[j]) + self.reg_lambda * self.weights[j])
            self.bias[j] -= self.lr * np.sum(self.deltas[j])
            # for i in range(len(self.hidden_layer_size), -1, -1):
            #         d_weight = np.dot(self.layers[i].T, self.deltas[i]) + self.reg_lambda * self.weights[i]
            #         d_bias = np.sum(self.deltas[i], axis=0)
            #         self.weights[i] += -1 * self.lr * d_weight
            #         self.bias[i] += -1. * self.lr * d_bias

    def predict(self, X):
        """
        predicting probability outputs
        :param X: array-like, shape(n_samples, n_features)
        :return: array-like, predicted probabilities
        """
        return self.forward(X)

    def score(self, X, y):
        """
        compute accuracy
        :param X: array-like, shape(n_samples, n_features)
        :param y: ground truth labels array-like, shape(n_samples, 1)
        :return: float, accuracy
        """
        n_samples = X.shape[0]

        # compute compute accuracy
        pred = np.argmax(self.predict(X), axis=1)
        acc = np.mean(pred[:, None] == y)
        return acc


def my_mlp():
    # from sklearn.datasets import fetch_mldata
    # mnist = fetch_mldata("MNIST original")
    # X, y = mnist.data / 255., mnist.target
    # X_train, X_test = X[:60000], X[60000:]
    # y_train, y_test = y[:60000], y[60000:]

    import sklearn.datasets
    from sklearn.preprocessing import MinMaxScaler

    dataset = sklearn.datasets.load_digits()
    X_train = dataset.data[:1500]
    X_test = dataset.data[1500:]
    y_train = dataset.target[:1500]
    y_test = dataset.target[1500:]

    # data normalization
    # X_train = (X_train - np.mean(X_train, axis=1, keepdims=True)) / np.std(X_train, axis=1, keepdims=True)
    # X_test = (X_test - np.mean(X_test, axis=1, keepdims=True)) / np.std(X_test, axis=1, keepdims=True)
    minmax_scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    X_train = minmax_scaler.fit_transform(X_train)
    X_test = minmax_scaler.fit_transform(X_test)

    network = MLP(input_size=64, output_size=10, hidden_layer_size=[128], batch_size=200, activation="sigmoid",
                  output_layer='softmax', loss='cross_entropy', lr=0.1)

    network.fit(X_train, y_train, 100, True)

    acc = network.score(X_test, y_test)
    print('Test Accuracy: {}'.format(acc))


def sklearn_mlp():
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_mldata
    from sklearn.neural_network import MLPClassifier

    # mnist =fetch_mldata("MNIST original")
    # X, y = mnist.data / 255., mnist.target
    # X_train, X_test = X[:60000], X[60000:]
    # y_train, y_test = y[:60000], y[60000:]

    import sklearn.datasets
    dataset = sklearn.datasets.load_digits()
    X_train = dataset.data[:1500]
    X_test = dataset.data[1500:]
    y_train = dataset.target[:1500]
    y_test = dataset.target[1500:]

    mlp = MLPClassifier(hidden_layer_sizes=128, max_iter=100, alpha=1e-4,
                        solver='sgd', activation='logistic', verbose=10, tol=1e-4, random_state=1,
                        learning_rate_init=.01)
    mlp.fit(X_train, y_train)
    print("Training set score: %f" % mlp.score(X_train, y_train))
    print("Test set score: %f" % mlp.score(X_test, y_test))


def main():
    print('Class 2 Multiple Layer Perceptron (MLP) Example')
    my_mlp()

    print ('')

    print('Class 2 sklearn MLP Example')
    sklearn_mlp()


if __name__ == "__main__":
    main()
