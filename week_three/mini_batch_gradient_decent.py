import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer


def load_data(csv_file_name, col_name_list):
    data_frame = pd.read_csv(csv_file_name, names=col_name_list)
    data_array = data_frame.values
    feature = data_array[:, :-1]
    result = data_array[:, -1]
    feature = normalization_data(norm_type='l2', data_set=feature)
    return feature, result


def split_data_set(x_data, y_data, t_percent):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=t_percent, random_state=29)
    return x_train, x_test, y_train, y_test


def normalization_data(norm_type, data_set):
    """
    :param norm_type: 'l1','l2', 'min_max'
    :param data_set:
    :return:
    """
    if norm_type == 'l2':
        normalizer = Normalizer(norm=norm_type)
        normal_data = normalizer.fit_transform(data_set)
    if norm_type == 'l1':
        normalizer = Normalizer(norm=norm_type)
        normal_data = normalizer.fit_transform(data_set)
    if norm_type == 'min_max':
        normalizer = MinMaxScaler(feature_range=(0, 1))
        normal_data = normalizer.fit_transform(data_set)
    return normal_data


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def hypothesis(z, w):
    return sigmoid(np.dot(np.c_[np.ones(z.shape[0]), z], w.T)).reshape(z.shape[0], )


def logistic_gradient(x, y, w):
    y_ = hypothesis(x, w)
    loss = (1.0 / x.shape[0]) * np.sum(y * np.log(y_) + (1 - y) * np.log(1 - y_))
    gradient = np.dot((y_ - y), np.c_[np.ones(x.shape[0]), x])
    return -loss, gradient


def logistic_regression(x, y, w, max_epoch, lr, batch_size, epsilon, momentum=None, adagrad=False, ):
    # Implement optimizer: Momentum, Adagrad
    loss_change = 1.
    loss_history = [1.]
    momentum_vector = [0.]
    gradient_history = [np.ones(9)]
    i = 0
    while i < max_epoch and loss_change > epsilon:
        for j in range(0, x.shape[0], batch_size):
            x_batch = x[j: batch_size + j, :]
            y_batch = y[j: batch_size + j]
            l, g = logistic_gradient(x_batch, y_batch, w)
            if momentum is not None:
                g_ = momentum * momentum_vector[-1] + lr * g
                momentum_vector.append(g_)
            elif adagrad is True:
                g_ = lr * 1.0 * g / np.sqrt(np.sum(gradient_history, axis=0))
                gradient_history.append(g_)
            else:
                g_ = lr * g
            w -= g_
            loss_change = abs(l - loss_history[-1])
            loss_history.append(l)
            # print('Epoch [{}]: Loss={}; Loss Change: {}'.format(i, l, loss_change))
            if loss_change < epsilon:
                # print('Stop @: Epoch [{}]: Loss={}; Loss Change: {}'.format(i, l, loss_change))
                break
        i += 1
    return np.array(loss_history), w


def validating_and_result(test_set, weight, str_solver_name):
    # Validation
    value, prob = predict(test_set, weight)
    accuracy = np.sum(value == y_test) * 1.0 / test_set.shape[0]

    # Print Result: Coefficents and intercept
    print("Training by {}:".format(str_solver_name))
    print("Coefficients: {}".format(weight[0, 1:]))
    print("Intercept: {}".format(weight[0, 0]))
    print("Accuracy: {}".format(accuracy))
    print("")


def predict(x, w):
    predict_prob = sigmoid(np.dot(w, np.c_[np.ones(x.shape[0]), x].T))
    predict_value = np.where(predict_prob > 0.5, 1, 0)
    return predict_value, predict_prob


if __name__ == '__main__':
    # Loading data from csv file and split to training and testing data set
    col_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data_file = './pima-indians-diabetes.data.csv'
    x_data, y_data = load_data(data_file, col_names)
    x_train, x_test, y_train, y_test = split_data_set(x_data, y_data, 0.8)

    # Initialization training parameters
    max_epoch = 10
    update_epsilon = 1e-5
    learning_rate = 0.01
    train_loss = []
    batch_size = 32

    # Initialize weight
    np.random.seed(0)
    w = np.random.rand(1, x_train.shape[1] + 1)

    # This problem is a binary-classification problem. Use Logistic Regression
    # Training by using normal mini-batch SGD
    train_loss_one, trained_w_one = logistic_regression(x_train, y_train, w, max_epoch, learning_rate, batch_size,
                                                        update_epsilon)
    validating_and_result(x_test, trained_w_one, 'Mini Batch SGD')

    # Training by using mini-batch SGD with Momentum
    train_loss_two, trained_w_two = logistic_regression(x_train, y_train, w, max_epoch, learning_rate, batch_size,
                                                        update_epsilon, momentum=0.9)
    validating_and_result(x_test, trained_w_two, 'Mini Batch SGD with Momentum')

    # Training by using mini-batch SGD with Adagrad
    train_loss_three, trained_w_three = logistic_regression(x_train, y_train, w, max_epoch, learning_rate, batch_size,
                                                            update_epsilon, adagrad=True)
    validating_and_result(x_test, trained_w_three, 'Mini Batch SGD with Adagrad')

    # Draw Cost Function loss
    plot.plot(range(len(train_loss_one[0:])), train_loss_one[0:], color='red', label='Mini Batch SGD')
    plot.plot(range(len(train_loss_two[0:])), train_loss_two[0:], color='blue', label='Mini Batch SGD with Momentum')
    plot.plot(range(len(train_loss_three[0:])), train_loss_three[0:], color='green',
              label='Mini Batch SDG with Adagrad')
    plot.legend(bbox_to_anchor=(1., 1.), loc=0, borderaxespad=0.)
    plot.show()
